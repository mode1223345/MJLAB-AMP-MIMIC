"""ONNX export for AMP HIM policies (Gazebo / Noetix deploy I/O).

N3 (0905): ``obs [1, 480] -> actions [1, 29]`` (5 × 96 history, HIM folded in).



Default ``joint_order="deploy"`` remaps Isaac Lab / Gazebo URDF joint order ↔
mjlab MuJoCo training order so the ONNX matches existing deploy artifacts.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Literal, Sequence

import torch
import torch.nn as nn

from mjlab.asset_zoo.robots.N3.constants import (
  N3_DEPLOY_JOINT_NAMES,
  N3_JOINT_NAMES,
)
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

ONNX_OPSET_VERSION = 14

# Isaac Lab / Gazebo N3 URDF joint order (n3.py ``N3_29DOF_JOINT_NAMES``).
DEPLOY_JOINT_NAMES: tuple[str, ...] = N3_DEPLOY_JOINT_NAMES

JointOrder = Literal["deploy", "mjlab"]


def resolve_deploy_joint_names(train_joint_names: Sequence[str]) -> tuple[str, ...]:
  """Pick Gazebo/Isaac deploy order for a training joint list."""
  if set(train_joint_names) == set(N3_JOINT_NAMES):
    return DEPLOY_JOINT_NAMES
  # Unknown robot: identity (deploy == train).
  return tuple(train_joint_names)


def _permutation_indices(
  src_names: Sequence[str], dst_names: Sequence[str]
) -> list[int]:
  """Indices such that ``dst_vec = src_vec[indices]``."""
  src_index = {n: i for i, n in enumerate(src_names)}
  missing = [n for n in dst_names if n not in src_index]
  if missing:
    raise ValueError(f"Joint names missing from source order: {missing}")
  return [src_index[n] for n in dst_names]


def _reorder_list(values: list, indices: list[int]) -> list:
  return [values[i] for i in indices]


class _OnnxAmpHimPolicy(nn.Module):
  """HIM actor export with optional deploy↔mjlab joint remapping."""

  def __init__(
    self,
    policy: nn.Module,
    *,
    train_joint_names: Sequence[str],
    deploy_joint_names: Sequence[str] | None = None,
    joint_order: JointOrder = "deploy",
  ):
    super().__init__()
    if not hasattr(policy, "him_estimator"):
      raise ValueError("Policy must have him_estimator for AMP HIM ONNX export.")

    num_joints = len(train_joint_names)
    one_step = int(policy.num_one_step_obs)
    # Layout: [prefix | joint_pos | joint_vel | last_action]
    if one_step < 3 * num_joints:
      raise ValueError(f"one-step obs dim {one_step} too small for {num_joints} joints")
    prefix_dim = one_step - 3 * num_joints

    if deploy_joint_names is None:
      deploy_joint_names = resolve_deploy_joint_names(train_joint_names)
    if len(deploy_joint_names) != num_joints:
      raise ValueError(
        f"deploy joint count {len(deploy_joint_names)} != train {num_joints}"
      )
    if set(deploy_joint_names) != set(train_joint_names):
      raise ValueError("deploy and train joint name sets must match")

    self.actor = copy.deepcopy(policy.actor)
    self.him_estimator = copy.deepcopy(policy.him_estimator)
    self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)
    self.num_one_step_obs = one_step
    self.prefix_dim = prefix_dim
    self.temporal_steps = int(getattr(policy.him_estimator, "temporal_steps", 1))
    self.num_actor_obs = self.num_one_step_obs * self.temporal_steps
    self.num_joints = num_joints
    self.joint_order = joint_order
    self.deploy_joint_names = tuple(deploy_joint_names)
    self.train_joint_names = tuple(train_joint_names)

    train_from_deploy = _permutation_indices(deploy_joint_names, train_joint_names)
    deploy_from_train = _permutation_indices(train_joint_names, deploy_joint_names)
    self.register_buffer(
      "train_from_deploy",
      torch.tensor(train_from_deploy, dtype=torch.long),
    )
    self.register_buffer(
      "deploy_from_train",
      torch.tensor(deploy_from_train, dtype=torch.long),
    )

  def _permute_joint_vec(self, vec: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return vec.index_select(-1, perm)

  def _remap_one_step_deploy_to_train(self, frame: torch.Tensor) -> torch.Tensor:
    """Remap joint_pos / joint_vel / last_action from deploy → train order."""
    n = self.num_joints
    p = self.prefix_dim
    prefix = frame[..., :p]
    jpos = self._permute_joint_vec(frame[..., p : p + n], self.train_from_deploy)
    jvel = self._permute_joint_vec(
      frame[..., p + n : p + 2 * n], self.train_from_deploy
    )
    lact = self._permute_joint_vec(
      frame[..., p + 2 * n : p + 3 * n], self.train_from_deploy
    )
    return torch.cat((prefix, jpos, jvel, lact), dim=-1)

  def _remap_history_deploy_to_train(self, obs: torch.Tensor) -> torch.Tensor:
    frames = obs.view(-1, self.temporal_steps, self.num_one_step_obs)
    remapped = self._remap_one_step_deploy_to_train(frames)
    return remapped.reshape(obs.shape[0], -1)

  def _actions_to_export_order(self, actions: torch.Tensor) -> torch.Tensor:
    if self.joint_order == "mjlab":
      return actions
    return self._permute_joint_vec(actions, self.deploy_from_train)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    if self.joint_order == "deploy":
      obs = self._remap_history_deploy_to_train(obs)
    x = self.normalizer(obs)
    vel, latent = self.him_estimator(x)
    actor_input = torch.cat((x[:, -self.num_one_step_obs :], vel, latent), dim=-1)
    actions_train = self.actor(actor_input)
    return self._actions_to_export_order(actions_train)


def _metadata_for_order(
  metadata: dict,
  train_names: Sequence[str],
  deploy_names: Sequence[str],
  joint_order: JointOrder,
) -> dict:
  if joint_order == "mjlab":
    metadata["joint_names"] = list(train_names)
    return metadata

  deploy_from_train = _permutation_indices(train_names, deploy_names)
  metadata["joint_names"] = list(deploy_names)
  for key in ("joint_stiffness", "joint_damping", "default_joint_pos", "action_scale"):
    values = metadata.get(key)
    if isinstance(values, list) and len(values) == len(train_names):
      metadata[key] = _reorder_list(values, deploy_from_train)
  return metadata


def export_amp_him_policy_as_onnx(
  policy: nn.Module,
  path: str,
  filename: str = "policy.onnx",
  verbose: bool = False,
  *,
  env: ManagerBasedRlEnv | None = None,
  run_path: str = "",
  train_joint_names: Sequence[str] = N3_JOINT_NAMES,
  deploy_joint_names: Sequence[str] | None = None,
  joint_order: JointOrder = "deploy",
) -> str:
  """Export AMP HIM policy to ONNX.

  Returns:
    Absolute path to the written ONNX file.
  """
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

  exporter = _OnnxAmpHimPolicy(
    policy,
    train_joint_names=train_joint_names,
    deploy_joint_names=deploy_joint_names,
    joint_order=joint_order,
  )
  exporter.to("cpu")
  exporter.eval()

  dummy = torch.zeros(1, exporter.num_actor_obs)
  out_path = os.path.join(path, filename)
  torch.onnx.export(
    exporter,
    dummy,
    out_path,
    export_params=True,
    opset_version=ONNX_OPSET_VERSION,
    do_constant_folding=True,
    verbose=verbose,
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={},
    dynamo=False,
  )

  if env is not None:
    metadata = get_base_metadata(env, run_path)
    metadata = _metadata_for_order(
      metadata,
      list(train_joint_names),
      list(exporter.deploy_joint_names),
      joint_order,
    )
    metadata["policy_type"] = "amp_him"
    metadata["joint_order"] = joint_order
    metadata["obs_dim"] = str(exporter.num_actor_obs)
    metadata["history_length"] = str(exporter.temporal_steps)
    metadata["one_step_obs_dim"] = str(exporter.num_one_step_obs)
    attach_metadata_to_onnx(out_path, metadata)

  return out_path
