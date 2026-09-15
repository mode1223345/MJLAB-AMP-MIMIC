"""AMP reset / domain-randomization events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def randomize_joint_default_pos(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  pos_distribution_params: tuple[float, float] = (-0.02, 0.02),
  operation: Literal["add"] = "add",
  action_name: str = "joint_pos",
) -> None:
  """Randomize default joint positions (calibration zero-point error).

  Port of Noetix ``randomize_joint_default_pos``: offsets
  ``default_joint_pos`` and mirrors the change into the joint-position
  action term offset so relative actions stay consistent.
  """
  if operation != "add":
    raise ValueError("randomize_joint_default_pos only supports operation='add'")

  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice):
    n_joints = asset.data.default_joint_pos.shape[-1]
    joint_ids_t = torch.arange(n_joints, device=env.device, dtype=torch.long)
  else:
    joint_ids_t = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)

  noise = sample_uniform(
    pos_distribution_params[0],
    pos_distribution_params[1],
    (len(env_ids), len(joint_ids_t)),
    device=env.device,
  )
  asset.data.default_joint_pos[env_ids[:, None], joint_ids_t] = (
    asset.data.default_joint_pos[env_ids[:, None], joint_ids_t] + noise
  )

  # Keep action offset in sync when using default-offset joint position control.
  try:
    action_term = env.action_manager.get_term(action_name)
  except KeyError:
    return
  if not isinstance(action_term, JointPositionAction):
    return
  if not isinstance(action_term._offset, torch.Tensor):
    return

  # Map asset joint ids -> action target columns.
  target_ids = action_term._target_ids
  action_term._offset[env_ids] = asset.data.default_joint_pos[env_ids][:, target_ids]


def reset_root_state_amp(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  root_pos_range: dict[str, tuple[float, float]],
  root_vel_range: dict[str, tuple[float, float]],
  joint_pos_range: tuple[float, float],
  joint_vel_range: tuple[float, float],
  reference_state_initialization: bool = False,
  prob_rsi: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  """Reset root/joint state, optionally from AMP motion frames (RSI)."""
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  motion_loader = getattr(env.unwrapped, "motion_loader", None)

  use_rsi = (
    motion_loader is not None
    and reference_state_initialization
    and float(torch.rand(1, device=env.device)) < prob_rsi
  )

  if use_rsi:
    frames = motion_loader.get_full_frame_batch(len(env_ids))
    positions = motion_loader.get_root_pos_batch(frames)
    positions[:, :2] = 0.0
    positions += env.scene.env_origins[env_ids]
    positions[:, 2] += 0.05
    quat_xyzw = motion_loader.get_root_rot_batch(frames)
    orientations = torch.cat(
      (quat_xyzw[:, -1].unsqueeze(1), quat_xyzw[:, :-1]), dim=1
    )  # xyzw -> wxyz

    base_lin_vel = motion_loader.get_linear_vel_batch(frames)
    base_ang_vel = motion_loader.get_angular_vel_batch(frames)
    lin_vel = quat_apply(orientations, base_lin_vel)
    ang_vel = quat_apply(orientations, base_ang_vel)
    velocities = torch.cat([lin_vel, ang_vel], dim=-1)

    joint_pos = motion_loader.get_joint_pose_batch(frames)[
      ..., motion_loader.joint_mapping["motion2sim"]
    ]
    joint_vel = motion_loader.get_joint_vel_batch(frames)[
      ..., motion_loader.joint_mapping["motion2sim"]
    ]
  else:
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()

    range_list = [
      root_pos_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
    )
    positions = (
      root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    )
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    vel_list = [
      root_vel_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    vel_ranges = torch.tensor(vel_list, device=env.device)
    vel_samples = sample_uniform(
      vel_ranges[:, 0],
      vel_ranges[:, 1],
      (len(env_ids), 6),
      device=env.device,
    )
    velocities = root_states[:, 7:13] + vel_samples

    default_joint_pos = asset.data.default_joint_pos
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_pos is not None and default_joint_vel is not None
    joint_pos = default_joint_pos[env_ids].clone()
    joint_vel = default_joint_vel[env_ids].clone()
    joint_pos += sample_uniform(*joint_pos_range, joint_pos.shape, joint_pos.device)
    joint_vel += sample_uniform(*joint_vel_range, joint_vel.shape, joint_vel.device)

  soft_pos_limits = asset.data.soft_joint_pos_limits
  if soft_pos_limits is not None:
    limits = soft_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])

  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)
