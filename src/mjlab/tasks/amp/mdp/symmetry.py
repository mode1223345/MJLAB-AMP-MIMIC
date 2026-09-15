"""Sagittal-mirror symmetry engine for AMP tasks.

A single generic implementation parameterized by :class:`SymmetrySpec`;
per-robot modules (currently ``symmetry_n3``) bind it to their joint
tables.

Actor obs layout (single frame):
  euler:  commands(3), ang_vel(3), euler_roll_pitch(2), dof_pos(N),
          dof_vel(N), actions(N)
  gravity: commands(3), ang_vel(3), projected_gravity(3), dof_pos(N),
           dof_vel(N), actions(N)
Critic obs layout: actor prefix, base_lin_vel(3), feet_contact(2), tail
(height scan when present).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch
from tensordict import TensorDict

from mjlab.sensor import GridPatternCfg, RayCastSensorCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _opposite_joint(name: str) -> str:
  if name.startswith("l_"):
    return "r_" + name[2:]
  if name.startswith("r_"):
    return "l_" + name[2:]
  return name


@dataclass(frozen=True)
class SymmetrySpec:
  """Robot-specific symmetry parameters.

  Args:
    joint_names: Joint names in training (MJCF/entity) order. Must be
      left-right symmetric: every ``l_*`` joint has an ``r_*`` counterpart
      at the mirrored position in the ordering.
    orientation: Actor prefix flavor -- ``"euler"`` (8-dim roll/pitch) or
      ``"gravity"`` (9-dim projected gravity).
    critic_tail: How the critic tail beyond contacts is mirrored --
      ``"copy"`` leaves it as-is, ``"grid_flip_dynamic"`` mirrors a ray-cast
      height scan grid read from the env's ``terrain_scan`` sensor cfg.
    critic_tail_dim: Per-frame tail width for ``"copy"`` (e.g. 96 for a
      height scan); the frame layout is prefix + 5 + tail. Ignored for
      ``"grid_flip_dynamic"`` (derived from the sensor cfg).
  """

  joint_names: tuple[str, ...]
  orientation: Literal["euler", "gravity"]
  critic_tail: Literal["copy", "grid_flip_dynamic"] = "copy"
  critic_tail_dim: int = 0


_PREFIX_SIGNS = {
  "euler": (1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0),
  "gravity": (1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
}


class Mirror:
  """Sagittal reflection for one robot, derived from its joint names."""

  def __init__(self, spec: SymmetrySpec):
    self.spec = spec
    self.num_joints = n = len(spec.joint_names)
    self._permutation = tuple(
      spec.joint_names.index(_opposite_joint(name)) for name in spec.joint_names
    )
    # Yaw/roll axes reverse under left-right reflection; pitch keeps sign.
    self._signs = tuple(
      -1.0 if "_roll_" in name or "_yaw_" in name else 1.0 for name in spec.joint_names
    )
    self.prefix_dim = 8 if spec.orientation == "euler" else 9
    self.actor_dim = self.prefix_dim + 3 * n

  # ------------------------------------------------------------------
  # Primitive flips
  # ------------------------------------------------------------------

  def flip_dof(self, dof: torch.Tensor) -> torch.Tensor:
    """Swap limbs; yaw/roll axes reverse, pitch axes keep their sign."""
    if dof.shape[-1] != self.num_joints:
      raise ValueError(
        f"Symmetry expects {self.num_joints} joints, got {dof.shape[-1]}"
      )
    return dof[..., list(self._permutation)] * dof.new_tensor(self._signs)

  def flip_actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
    """Mirror commands, angular velocity, orientation, joints, and actions.

    Works with one frame, explicit history, or flattened actor history.
    """
    if obs is None:
      return obs
    frames = obs.unflatten(-1, (-1, self.actor_dim))
    flipped = frames.clone()
    flipped[..., : self.prefix_dim] *= obs.new_tensor(
      _PREFIX_SIGNS[self.spec.orientation]
    )
    for start in (
      self.prefix_dim,
      self.prefix_dim + self.num_joints,
      self.prefix_dim + 2 * self.num_joints,
    ):
      flipped[..., start : start + self.num_joints] = self.flip_dof(
        frames[..., start : start + self.num_joints]
      )
    return flipped.reshape_as(obs)

  def _scan_grid(self, env: ManagerBasedRlEnv) -> tuple[int, int]:
    scan_cfg = next(
      sensor
      for sensor in env.unwrapped.cfg.scene.sensors
      if sensor.name == "terrain_scan"
    )
    assert isinstance(scan_cfg, RayCastSensorCfg)
    pattern = scan_cfg.pattern
    assert isinstance(pattern, GridPatternCfg)
    num_x, num_y = (round(size / pattern.resolution) + 1 for size in pattern.size)
    return num_x, num_y

  def flip_critic_obs(
    self, obs: torch.Tensor, env: ManagerBasedRlEnv | None = None
  ) -> torch.Tensor:
    """Mirror actor prefix, linear velocity, contacts, and the tail."""
    if obs is None:
      return obs
    num_x = num_y = 0
    if self.spec.critic_tail == "grid_flip_dynamic":
      assert env is not None, "grid_flip_dynamic requires the env sensor cfg"
      num_x, num_y = self._scan_grid(env)
      tail = num_x * num_y
    else:
      tail = self.spec.critic_tail_dim
    critic_dim = self.actor_dim + 5 + tail
    frames = obs.unflatten(-1, (-1, critic_dim))
    flipped = frames.clone()
    flipped[..., : self.actor_dim] = self.flip_actor_obs(frames[..., : self.actor_dim])
    flipped[..., self.actor_dim + 1] *= -1
    flipped[..., self.actor_dim + 3 : self.actor_dim + 5] = frames[
      ..., self.actor_dim + 3 : self.actor_dim + 5
    ].flip(-1)
    if self.spec.critic_tail == "grid_flip_dynamic" and tail:
      base = self.actor_dim + 5
      # GridPatternCfg uses meshgrid(indexing="xy"): x varies fastest.
      scan = frames[..., base:].unflatten(-1, (num_y, num_x))
      flipped[..., base:] = scan.flip(-2).flatten(-2)
    # "copy" tail is preserved by the clone above.
    return flipped.reshape_as(obs)

  # ------------------------------------------------------------------
  # PPO data augmentation
  # ------------------------------------------------------------------

  @torch.no_grad()
  def data_augmentation_func(
    self,
    env: ManagerBasedRlEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
  ) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append mirrored PPO/HIM observations and actions to the batch."""
    obs_aug = None
    if obs is not None:
      batch_size = obs.batch_size[0]
      obs_aug = cast(TensorDict, TensorDict.cat((obs, obs), dim=0))
      for key in ("actor", "policy", "next_obs"):
        if key in obs:
          obs_aug[key][batch_size:] = self.flip_actor_obs(cast(torch.Tensor, obs[key]))
      if "critic" in obs:
        obs_aug["critic"][batch_size:] = self.flip_critic_obs(
          cast(torch.Tensor, obs["critic"]), env
        )
    actions_aug = None
    if actions is not None:
      actions_aug = torch.cat((actions, self.flip_dof(actions)), dim=0)
    return obs_aug, actions_aug
