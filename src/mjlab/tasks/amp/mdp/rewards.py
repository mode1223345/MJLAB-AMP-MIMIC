"""AMP task rewards (ported formulas, mjlab Entity APIs)."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_apply_inverse,
  yaw_quat,
)

from .terminations import get_delay_env_mask

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def delay_masked(func: Callable) -> Callable:
  """Zero a reward term for delay envs (fall-recovery training).

  Wraps a reward function so that environments flagged as delay envs (see
  :class:`DelayedTerminationManager`) receive 0 instead of the term value.
  With no delayed termination installed the wrapper is a pass-through.
  """

  @functools.wraps(func)
  def wrapped(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    reward = func(env, **kwargs)
    mask = get_delay_env_mask(env)
    if mask is None:
      return reward
    return torch.where(mask, torch.zeros_like(reward), reward)

  return wrapped


def track_root_height(
  env: ManagerBasedRlEnv,
  std: float,
  delay_env_rew_ratio: float = 3.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Get-up reward: track the default standing root height.

  Delay envs only (delay-mask): ``delay_env_rew_ratio``-scaled exponential
  height-tracking reward — the primary learning signal for standing back up.
  Returns zeros when delayed termination is not installed.
  """
  asset: Entity = env.scene[asset_cfg.name]
  desired_height = asset.data.default_root_state[:, 2]
  height = asset.data.root_link_pos_w[:, 2]
  reward = torch.exp(-torch.square(desired_height - height) / std**2)
  mask = get_delay_env_mask(env)
  if mask is None:
    return torch.zeros_like(reward)
  return torch.where(mask, delay_env_rew_ratio * reward, torch.zeros_like(reward))


def is_terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 for envs terminated by a bad (non-timeout) termination this step.

  Note: with delayed termination installed, suppressed falls do not count
  (the buffer already reflects the suppression), so lying delay envs are not
  penalized — only actually-released resets and normal-env falls are.
  """
  tm = env.termination_manager
  return (tm.terminated & ~tm.time_outs).float()


def root_height_out_of_range(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  maximum_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared root-height error outside a band above the environment origin."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  below = (minimum_height - height).clamp(min=0.0)
  above = (height - maximum_height).clamp(min=0.0)
  return below.square() + above.square()


def both_feet_air(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  command_name: str | None = None,
  speed_threshold: float = 1.5,
) -> torch.Tensor:
  """Return one when neither foot contacts the ground.

  With ``command_name`` set, fast commands (planar command norm above
  ``speed_threshold``) are exempt: running gaits have a legitimate flight
  phase, so the penalty only guards against hopping at walking speeds.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  both_air = (sensor.data.found == 0).all(dim=-1).float()
  if command_name is None:
    return both_air
  command = env.command_manager.get_command(command_name)
  assert command is not None
  slow_command = torch.norm(command[:, :2], dim=1) <= speed_threshold
  return both_air * slow_command.float()


def _pure_forward_command_mask(
  command: torch.Tensor,
  cmd_threshold: float,
  lateral_cmd_threshold: float,
  yaw_cmd_threshold: float,
) -> torch.Tensor:
  """True when twist is a pure forward command (+vx, small vy/wz)."""
  return (
    (command[:, 0] > cmd_threshold)
    & (torch.abs(command[:, 1]) < lateral_cmd_threshold)
    & (torch.abs(command[:, 2]) < yaw_cmd_threshold)
  )


def _yaw_frame_linear_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Actual and commanded horizontal velocity in the yaw frame."""
  asset: Entity = env.scene[asset_cfg.name]
  actual = quat_apply_inverse(
    yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w[:, :3]
  )[:, :2]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return actual, command[:, :2]


def adaptive_cauchy_velocity_precision(
  actual: torch.Tensor,
  target: torch.Tensor,
  absolute_scale: float,
  relative_scale: float,
  reduction_dim: int = -1,
) -> torch.Tensor:
  """Command-adaptive Cauchy: 1 / (1 + (||e|| / (abs + rel*||c||))^2)."""
  error = torch.linalg.vector_norm(actual - target, dim=reduction_dim)
  target_mag = torch.linalg.vector_norm(target, dim=reduction_dim)
  scale = absolute_scale + relative_scale * target_mag
  return torch.reciprocal(1.0 + torch.square(error / scale.clamp_min(1.0e-8)))


def bidirectional_velocity_progress(
  actual: torch.Tensor,
  target: torch.Tensor,
  activity_threshold: float,
  reduction_dim: int = -1,
  denominator_epsilon: float = 1.0e-8,
) -> torch.Tensor:
  """Signed progress toward command; peaks at exact match, gated by activity."""
  projection = torch.sum(actual * target, dim=reduction_dim)
  target_squared = torch.sum(torch.square(target), dim=reduction_dim)
  ratio = projection / torch.clamp(target_squared, min=denominator_epsilon)
  triangular = torch.clamp(1.0 - torch.abs(ratio - 1.0), min=-1.0, max=1.0)
  activity = torch.clamp(
    torch.sqrt(target_squared) / activity_threshold, min=0.0, max=1.0
  )
  return activity * triangular


def track_lin_vel_xy_yaw_frame_cauchy(
  env: ManagerBasedRlEnv,
  absolute_scale: float,
  relative_scale: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track horizontal velocity with a command-adaptive Cauchy reward."""
  actual, target = _yaw_frame_linear_velocity(env, asset_cfg, command_name)
  return adaptive_cauchy_velocity_precision(
    actual,
    target,
    absolute_scale=absolute_scale,
    relative_scale=relative_scale,
    reduction_dim=-1,
  )


def track_lin_vel_xy_yaw_frame_progress(
  env: ManagerBasedRlEnv,
  activity_threshold: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward continuous progress toward the commanded horizontal velocity."""
  actual, target = _yaw_frame_linear_velocity(env, asset_cfg, command_name)
  return bidirectional_velocity_progress(
    actual,
    target,
    activity_threshold=activity_threshold,
    reduction_dim=-1,
  )


def track_lin_vel_xy_yaw_frame_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  vel_yaw = quat_apply_inverse(
    yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w[:, :3]
  )
  command = env.command_manager.get_command(command_name)
  assert command is not None
  lin_vel_error = torch.sum(torch.square(command[:, :2] - vel_yaw[:, :2]), dim=1)
  return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  ang_vel_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_w[:, 2])
  return torch.exp(-ang_vel_error / std**2)


def flat_orientation_exp(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.exp(-torch.norm(asset.data.projected_gravity_b[:, :2], dim=1) / std**2)


def feet_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "twist",
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  feet_contact_flag = sensor.data.found > 0  # (N, num_feet)

  if not hasattr(env.unwrapped, "feet_both_contact_time"):
    env.unwrapped.feet_both_contact_time = torch.zeros(
      env.num_envs, dtype=torch.float, device=env.device
    )

  both_feet_contact = torch.all(feet_contact_flag, dim=1)
  env.unwrapped.feet_both_contact_time[both_feet_contact] += env.step_dt
  env.unwrapped.feet_both_contact_time[~both_feet_contact] = 0

  single_feet_contact = torch.sum(feet_contact_flag.float(), dim=1) == 1
  command = env.command_manager.get_command(command_name)
  assert command is not None
  reward = (
    single_feet_contact
    | (env.unwrapped.feet_both_contact_time < 0.2)
    | ((torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < 0.1)
  ).float()
  return reward


def feet_air_time_positive_biped(
  env: ManagerBasedRlEnv,
  threshold: float = 0.5,
  min_threshold: float | None = None,
  speed_max: float = 2.5,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "twist",
) -> torch.Tensor:
  """Reward single-stance mode time (air or contact), capped at ``threshold``.

  The cap is the target phase time: reward grows with the shorter of the two
  feet's current phase times and saturates at the cap. With ``min_threshold``
  set, the cap decays linearly with commanded speed, from ``threshold`` at
  standstill to ``min_threshold`` at ``speed_max`` and above, so fast commands
  target short phases. Stance times above the cap at running speed leave the
  term saturated (constant, no gradient) rather than pushing toward a low
  cadence.
  """
  if min_threshold is not None and speed_max <= 0.0:
    raise ValueError("speed_max must be positive when min_threshold is set")
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_air_time is not None
  assert sensor.data.current_contact_time is not None
  air_time = sensor.data.current_air_time
  contact_time = sensor.data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.sum(in_contact.int(), dim=1) == 1
  reward = torch.min(
    torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
  )[0]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  if min_threshold is not None:
    speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    blend = torch.clamp(speed / speed_max, max=1.0)
    reward = torch.minimum(reward, threshold + (min_threshold - threshold) * blend)
  else:
    reward = torch.clamp(reward, max=threshold)
  reward *= (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > 0.1
  return reward


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  contacts = sensor.data.found > 0
  asset: Entity = env.scene[asset_cfg.name]
  # Prefer body ids from asset_cfg; fall back to contact sensor primary bodies.
  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice) or (
    isinstance(body_ids, (list, tuple)) and len(body_ids) == 0
  ):
    # Match ankle_roll bodies.
    _, names = asset.find_bodies(".*_ankle_roll_link")
    body_ids = [asset.body_names.index(n) for n in names]
  body_vel = asset.data.body_link_lin_vel_w[:, body_ids, :2]
  return torch.sum(body_vel.norm(dim=-1) * contacts.float(), dim=1)


def feet_force(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  threshold: float = 100.0,
  max_reward: float = 800.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  # force: [B, num_feet, 3]
  force_z = torch.abs(sensor.data.force[..., 2])
  reward = force_z.norm(dim=-1)
  reward = torch.where(reward < threshold, torch.zeros_like(reward), reward - threshold)
  return reward.clamp(min=0, max=max_reward)


def feet_flat_orientation_when_loaded(
  env: ManagerBasedRlEnv,
  max_tilt: float = 0.26,
  force_threshold: float = 50.0,
  sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """着地且受力时，惩罚脚板相对地面倾角过大（抑脚尖点地）。

  ``max_tilt`` 为死区（允许自然着地/蹬地）；超过后按超额 ``sin(tilt)`` 平方惩罚。
  """
  if not 0.0 <= max_tilt < 0.5 * math.pi:
    raise ValueError("max_tilt must be in [0, pi/2)")
  if force_threshold < 0.0:
    raise ValueError("force_threshold must be non-negative")

  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice) or (
    isinstance(body_ids, (list, tuple)) and len(body_ids) == 0
  ):
    body_ids, _ = asset.find_bodies(
      ("l_leg_knee_pitch_link", "r_leg_knee_pitch_link"),
      preserve_order=True,
    )

  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  # force: [B, num_feet, 3]（与足 body 顺序一致）
  if sensor.data.force.shape[1] != len(body_ids):
    raise ValueError(
      "feet_flat_orientation_when_loaded: foot body count must match "
      "contact sensor primary slots"
    )

  foot_quat_w = asset.data.body_link_quat_w[:, body_ids, :]
  local_up = torch.zeros(
    foot_quat_w.shape[0],
    foot_quat_w.shape[1],
    3,
    device=foot_quat_w.device,
    dtype=foot_quat_w.dtype,
  )
  local_up[..., 2] = 1.0
  foot_up_w = quat_apply(foot_quat_w, local_up)
  tilt_sine = torch.linalg.vector_norm(foot_up_w[..., :2], dim=-1)
  excess_tilt = torch.clamp(tilt_sine - math.sin(max_tilt), min=0.0)

  contact_force = torch.linalg.vector_norm(sensor.data.force, dim=-1)
  loaded = contact_force > force_threshold
  return torch.sum(torch.square(excess_tilt) * loaded.float(), dim=1)


def feet_contact_roll_penalty(
  env: ManagerBasedRlEnv,
  threshold: float = 0.08,
  force_threshold: float = 1.0,
  sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize an inverted/everted foot beyond ``threshold`` while loaded.

  Linear (not squared) in the excess roll, summed over the feet whose net
  contact force exceeds ``force_threshold``. Roll is the world-frame xyz-Euler
  component: a flat foot reads zero at any heading, but a tilted foot also
  picks up pitch when the robot faces off x. Ported from the Noetix walkrun
  task, which used weight -1.0 and the same 0.08 rad (4.6 deg) deadzone.
  """
  if threshold < 0.0:
    raise ValueError("threshold must be non-negative")
  if force_threshold < 0.0:
    raise ValueError("force_threshold must be non-negative")

  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice) or (
    isinstance(body_ids, (list, tuple)) and len(body_ids) == 0
  ):
    body_ids, _ = asset.find_bodies(".*_ankle_roll_link")

  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  # force: [B, num_feet, 3]（与足 body 顺序一致）
  if sensor.data.force.shape[1] != len(body_ids):
    raise ValueError(
      "feet_contact_roll_penalty: foot body count must match contact sensor "
      "primary slots"
    )

  contact_force = torch.linalg.vector_norm(sensor.data.force, dim=-1)
  loaded = contact_force > force_threshold
  foot_quat_w = asset.data.body_link_quat_w[:, body_ids, :]
  num_feet = foot_quat_w.shape[1]
  roll = euler_xyz_from_quat(foot_quat_w.reshape(-1, 4))[0].view(-1, num_feet)
  excess_roll = torch.clamp(torch.abs(roll) - threshold, min=0.0)
  return torch.sum(excess_roll * loaded.float(), dim=1)


def feet_too_near_humanoid(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  threshold: float = 0.2,
) -> torch.Tensor:
  """Penalize when the two feet are closer than ``threshold`` (positive cost)."""
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice) or (
    isinstance(body_ids, (list, tuple)) and len(body_ids) == 0
  ):
    body_ids, _ = asset.find_bodies(
      ("l_leg_knee_pitch_link", "r_leg_knee_pitch_link"),
      preserve_order=True,
    )
  if len(body_ids) != 2:
    raise ValueError("feet_too_near_humanoid requires exactly two foot bodies")
  feet_pos = asset.data.body_link_pos_w[:, body_ids, :]
  distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
  return (threshold - distance).clamp(min=0.0)


def hip_roll_angle_limits(
  env: ManagerBasedRlEnv,
  left_range: tuple[float, float] = (-0.1, 0.12),
  right_range: tuple[float, float] = (-0.12, 0.1),
  violation_scale: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic penalty for left/right hip-roll outside allowed ranges."""
  if left_range[0] >= left_range[1]:
    raise ValueError("left_range lower bound must be smaller than its upper bound")
  if right_range[0] >= right_range[1]:
    raise ValueError("right_range lower bound must be smaller than its upper bound")
  if violation_scale <= 0.0:
    raise ValueError("violation_scale must be positive")

  asset: Entity = env.scene[asset_cfg.name]
  try:
    left_joint_id = asset.joint_names.index("l_leg_hip_roll_joint")
    right_joint_id = asset.joint_names.index("r_leg_hip_roll_joint")
  except ValueError as exc:
    raise ValueError(
      "hip_roll_angle_limits requires l_leg_hip_roll_joint and r_leg_hip_roll_joint"
    ) from exc

  left_angle = asset.data.joint_pos[:, left_joint_id]
  right_angle = asset.data.joint_pos[:, right_joint_id]
  left_violation = torch.relu(left_range[0] - left_angle) + torch.relu(
    left_angle - left_range[1]
  )
  right_violation = torch.relu(right_range[0] - right_angle) + torch.relu(
    right_angle - right_range[1]
  )
  return torch.square(left_violation / violation_scale) + torch.square(
    right_violation / violation_scale
  )


def motors_power_square(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize squared mechanical power (actuator force * joint velocity)."""
  asset: Entity = env.scene[asset_cfg.name]
  force = asset.data.actuator_force
  joint_vel = asset.data.joint_vel
  n = min(force.shape[-1], joint_vel.shape[-1])
  power_j = force[:, :n] * joint_vel[:, :n]
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice):
    selected = power_j
  else:
    selected = power_j[:, joint_ids]
  return torch.sum(torch.square(selected), dim=-1)


def feet_distance_when_standing(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  min_distance: float = 0.12,
  max_distance: float = 0.30,
  command_name: str = "twist",
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice):
    _, names = asset.find_bodies(".*_ankle_roll_link")
    body_ids = [asset.body_names.index(n) for n in names]
  assert len(body_ids) == 2
  feet_pos = asset.data.body_link_pos_w[:, body_ids, :]
  distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  standing = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < 0.1
  too_near = (min_distance - distance).clamp(min=0.0)
  too_far = (distance - max_distance).clamp(min=0.0)
  return (too_near + too_far) * standing.float()


def joint_deviation_l1(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(angle), dim=1)


def joint_deviation_l1_with_zero_flag(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str = "twist",
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  zero_flag = torch.logical_or(
    torch.norm(command[:, 1:], dim=1) > 0.1,
    (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < 0.1,
  )
  return torch.sum(torch.abs(angle), dim=1) * (~zero_flag).float()


def joint_deviation_l1_when_standing(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str = "twist",
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  zero_flag = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < 0.1
  return torch.sum(torch.abs(angle), dim=1) * zero_flag.float()


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize residual base motion when the commanded twist is near zero.

  Targets pitch/roll rocking from a heavy head: horizontal lin-vel plus
  roll/pitch rates (yaw rate left free for in-place turn commands).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  standing = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < 0.1
  lin_xy = torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1)
  ang_xy = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
  return (lin_xy + ang_xy) * standing.float()


def _action_rate_sq(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared action rate on raw policy output (before per-term scale/offset)."""
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def action_rate_l2_when_moving(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  stand_cmd_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize action rate only when the env is commanded to move.

  Port of the Isaac-Lab N3 ``action_rate_walk`` gate: moving means
  ``‖cmd_xy‖ + |cmd_wz| >= stand_cmd_threshold`` (Isaac used 0.05). Below
  that the (usually harsher) standing tier takes over instead.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None
  moving = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) >= (
    stand_cmd_threshold
  )
  return _action_rate_sq(env) * moving.float()


def action_rate_l2_when_standing(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  stand_cmd_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize action rate only when the commanded twist is near zero.

  Isaac-Lab N3 ``action_rate_stand``: same gate as
  :func:`action_rate_l2_when_moving`, inverted. Standing jitter is pure noise
  on the real robot, hence a heavier weight than the moving tier.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None
  standing = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) < (
    stand_cmd_threshold
  )
  return _action_rate_sq(env) * standing.float()


def joint_vel_limit_margin_penalty(
  env: ManagerBasedRlEnv,
  velocity_limits: dict[str, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ratio: float = 0.9,
) -> torch.Tensor:
  """Penalize joint speeds inside the top margin of their rated limits.

  For each selected joint: (|q̇| / limit − ratio)₊², summed. ``velocity_limits``
  maps joint name → rated speed [rad/s]; joints missing from the table get a
  1e9 limit (never penalized). Isaac-N3 port (``joint_vel_limit`` term).
  """
  asset: Entity = env.scene[asset_cfg.name]
  cache = getattr(env, "_amp_joint_vel_limits", None)
  if cache is None:
    device = asset.data.joint_pos.device
    cache = torch.tensor(
      [velocity_limits.get(name, 1e9) for name in asset.joint_names],
      dtype=torch.float32,
      device=device,
    )
    env._amp_joint_vel_limits = cache
  limit = cache[asset_cfg.joint_ids]
  exceed = asset.data.joint_vel[:, asset_cfg.joint_ids].abs() / limit - ratio
  return torch.sum(torch.square(exceed.clamp(min=0.0)), dim=1)


def joint_effort_limit_margin_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ratio: float = 0.9,
) -> torch.Tensor:
  """Penalize actuator forces inside the top margin of their effort limits.

  Computed in actuator space (one position actuator per joint on the AMP
  robots): (|τ| / effort_limit − ratio)₊² summed over the selected actuators.
  Limits come from the actuator configs (``effort_limit``); unlimited
  actuators are skipped. Isaac-N3 port (``joint_effort_limit`` term).
  """
  asset: Entity = env.scene[asset_cfg.name]
  force = asset.data.actuator_force
  cache = getattr(env, "_amp_actuator_effort_limits", None)
  if cache is None:
    cache = torch.full((force.shape[1],), 1e9, device=force.device)
    for act in asset.actuators:
      effort = getattr(act.cfg, "effort_limit", None)
      if effort is None:
        continue
      cache[act.ctrl_ids] = float(effort)
    env._amp_actuator_effort_limits = cache
  exceed = force.abs() / cache - ratio
  return torch.sum(torch.square(exceed.clamp(min=0.0)), dim=1)


def energy(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  # Prefer actuator forces * joint vel as a proxy for electrical energy.
  force = asset.data.actuator_force
  joint_vel = asset.data.joint_vel
  n = min(force.shape[-1], joint_vel.shape[-1])
  return torch.norm(torch.abs(force[:, :n] * joint_vel[:, :n]), dim=-1)


def forward_startup_from_stand(
  env: ManagerBasedRlEnv,
  cmd_threshold: float = 0.2,
  lateral_cmd_threshold: float = 0.1,
  yaw_cmd_threshold: float = 0.1,
  speed_ratio: float = 0.5,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward cold-start forward walking from standstill.

  Active only for pure forward commands (``cmd_x > threshold``, small
  ``cmd_y`` / ``cmd_wz``). While yaw-frame ``vx`` stays below
  ``speed_ratio * cmd_x``, reward same-sign forward progress and
  single-foot stance (encourages lifting a foot to begin stepping).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  vel_yaw = quat_apply_inverse(
    yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w[:, :3]
  )
  actual_x = vel_yaw[:, 0]
  cmd_x = command[:, 0]

  forward_only = _pure_forward_command_mask(
    command, cmd_threshold, lateral_cmd_threshold, yaw_cmd_threshold
  )
  under_speed = actual_x < speed_ratio * cmd_x
  needs_startup = forward_only & under_speed

  progress = torch.clamp(actual_x / torch.clamp(cmd_x, min=1e-6), min=0.0, max=1.0)
  progress = torch.where(actual_x > 0.0, progress, torch.zeros_like(progress))

  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_contact_time is not None
  in_contact = sensor.data.current_contact_time > 0.0
  single_stance = torch.sum(in_contact.int(), dim=1) == 1

  return needs_startup.float() * torch.clamp(progress + single_stance.float(), max=1.0)


def forward_stride_knee(
  env: ManagerBasedRlEnv,
  knee_target: float = 1.0,
  cmd_threshold: float = 0.2,
  lateral_cmd_threshold: float = 0.1,
  yaw_cmd_threshold: float = 0.1,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward swing-knee flexion during forward walk single stance.

  Active only for pure forward commands. When exactly one foot is loaded,
  rewards the swing leg knee angle up to ``knee_target`` (rad).
  """
  if knee_target <= 0.0:
    raise ValueError("knee_target must be positive")

  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  forward_only = _pure_forward_command_mask(
    command, cmd_threshold, lateral_cmd_threshold, yaw_cmd_threshold
  )

  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_contact_time is not None
  in_contact = sensor.data.current_contact_time > 0.0
  if in_contact.shape[1] != 2:
    raise ValueError("forward_stride_knee expects exactly two foot contacts")

  left_contact = in_contact[:, 0]
  right_contact = in_contact[:, 1]
  single_left = left_contact & ~right_contact
  single_right = right_contact & ~left_contact
  single_stance = single_left | single_right

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice) or len(joint_ids) != 2:
    joint_ids, _ = asset.find_joints(
      ("l_leg_knee_pitch_joint", "r_leg_knee_pitch_joint"),
      preserve_order=True,
    )
  knee_pos = asset.data.joint_pos[:, joint_ids]
  swing_knee = torch.where(
    single_left,
    knee_pos[:, 1],
    torch.where(single_right, knee_pos[:, 0], torch.zeros_like(knee_pos[:, 0])),
  )
  knee_ratio = torch.clamp(swing_knee / knee_target, min=0.0, max=1.0)

  return forward_only.float() * single_stance.float() * knee_ratio


def low_speed(
  env: ManagerBasedRlEnv,
  min_cmd_vel: float = 0.2,
  low_speed_threshold: float = 0.8,
  high_speed_threshold: float = 1.2,
  command_name: str = "twist",
) -> torch.Tensor:
  """Band reward on body-frame ``vx`` relative to ``cmd_x``, along the command.

  The band is applied to the signed ratio ``vx / cmd_x``, so moving against the
  command gives a negative ratio and lands in the "too slow" branch. Comparing
  ``|vx|`` against ``|cmd_x|`` cannot tell forward from backward motion and
  hands the band reward to a robot walking the wrong way. Envs with
  ``|cmd_x| <= min_cmd_vel`` are inactive and return 0.
  """
  base_lin_vel = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
  commands = env.command_manager.get_command(command_name)
  assert commands is not None
  commands = commands[:, 0]
  active = torch.abs(commands) > min_cmd_vel
  # Inactive envs are zeroed below; this keeps the division finite.
  safe_command = torch.where(active, commands, torch.ones_like(commands))
  speed_ratio = base_lin_vel / safe_command
  speed_too_low = speed_ratio < low_speed_threshold
  speed_too_high = speed_ratio > high_speed_threshold
  reward = torch.zeros_like(base_lin_vel)
  reward[speed_too_low] = -1.0
  reward[speed_too_high] = 0.0
  reward[~(speed_too_low | speed_too_high)] = 1.2
  return reward * active.float()


def track_ang_vel_stand_world_exp(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  lin_vel_threshold: float = 1.0,
  command_name: str = "twist",
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  is_pure_rotation = (torch.abs(command[:, 0]) < 1e-6) & (
    torch.abs(command[:, 2]) > 1e-6
  )
  lin_vel_penalty = torch.norm(asset.data.root_link_lin_vel_w[:, :2], dim=1)
  ang_vel_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_w[:, 2])
  reward = torch.exp(-ang_vel_error / std**2)
  return is_pure_rotation.float() * (reward - lin_vel_threshold * lin_vel_penalty)


def waist_com_feet_support_x_error_exp(
  env: ManagerBasedRlEnv,
  std: float,
  target_x: float = 0.0,
  stand_cmd_threshold: float = 0.05,
  waist_body_name: str = "waist_yaw_link",
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalise the waist link COM drifting off the feet centre along x.

  Ported from the IsaacLab walkrun task's ``waist_com_feet_support_x_error_exp``.
  The offset from the ankle midpoint to the waist COM is expressed in the root
  yaw frame and only its x component is scored, so ``target_x=0`` means the waist
  projects onto the middle of the support. Only active for commands with no
  translation component (stand and in-place turn): while translating, the COM
  has to lead the feet and this term would fight that.
  """
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice) or len(body_ids) != 2:
    raise ValueError("waist_com_feet_support_x_error_exp requires exactly two feet")
  waist_id = asset.body_names.index(waist_body_name)

  feet_center_w = 0.5 * asset.data.body_link_pos_w[:, body_ids, :].sum(dim=1)
  waist_from_feet_yaw = quat_apply_inverse(
    yaw_quat(asset.data.root_link_quat_w),
    asset.data.body_com_pos_w[:, waist_id] - feet_center_w,
  )
  error = torch.square(waist_from_feet_yaw[:, 0] - target_x)

  command = env.command_manager.get_command(command_name)
  assert command is not None
  stationary = torch.norm(command[:, :2], dim=1) < stand_cmd_threshold
  return torch.exp(-error / std**2) * stationary.float()


def track_ang_vel_run_world_exp(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str = "twist",
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  is_run = (torch.abs(command[:, 0]) > 1e-6) & (torch.abs(command[:, 2]) > 1e-6)
  ang_vel_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_w[:, 2])
  return is_run.float() * torch.exp(-ang_vel_error / std**2)


def back_forward_velocity_x(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  run_command_threshold: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward body-frame -vx while cmd_x is a binary run flag (> threshold)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  is_running = command[:, 0] > run_command_threshold
  return -asset.data.root_link_lin_vel_b[:, 0] * is_running.float()


def lateral_velocity_y_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize squared body-frame lateral velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 1])


def stand_still_velocity_xy_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward near-zero horizontal speed when the twist command is near zero."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  is_standing = torch.linalg.vector_norm(command, dim=1) < 0.1
  horizontal_speed_sq = torch.sum(
    torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  return torch.exp(-horizontal_speed_sq / std**2) * is_standing.float()


def base_euler_xy_exp(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward small base roll/pitch via an exponential kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  roll, pitch, _yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
  euler_xy = torch.stack([roll, pitch], dim=1)
  return torch.exp(-torch.linalg.vector_norm(euler_xy, dim=1) / std**2)


def track_robot_height_span_exp(
  env: ManagerBasedRlEnv,
  target_height: float = 1.4,
  tolerance: float = 0.05,
  std: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward max(body_z) - min(body_z) near ``target_height`` (span + tol)."""
  asset: Entity = env.scene[asset_cfg.name]
  body_heights = asset.data.body_link_pos_w[:, :, 2]
  robot_height = body_heights.max(dim=1).values - body_heights.min(dim=1).values
  height_error = torch.abs(robot_height - target_height)
  outside_tolerance = torch.clamp(height_error - tolerance, min=0.0)
  return torch.exp(-torch.square(outside_tolerance) / std**2)


def feet_lateral_distance_yaw_frame(
  env: ManagerBasedRlEnv,
  minimum_distance: float,
  maximum_distance: float,
  too_wide_scale: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize yaw-frame lateral foot separation outside ``[min, max]``.

  ``asset_cfg.body_names`` must be ordered left foot, right foot.
  """
  if minimum_distance <= 0.0:
    raise ValueError("minimum_distance must be positive")
  if maximum_distance <= minimum_distance:
    raise ValueError("maximum_distance must be greater than minimum_distance")
  if too_wide_scale < 0.0:
    raise ValueError("too_wide_scale must be non-negative")

  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice):
    raise ValueError("feet_lateral_distance_yaw_frame requires exactly two feet")
  if len(body_ids) != 2:
    raise ValueError("feet_lateral_distance_yaw_frame requires exactly two feet")

  feet_pos_w = asset.data.body_link_pos_w[:, body_ids, :]
  left_to_right_w = feet_pos_w[:, 0] - feet_pos_w[:, 1]
  left_to_right_yaw = quat_apply_inverse(
    yaw_quat(asset.data.root_link_quat_w), left_to_right_w
  )
  signed_lateral_distance = left_to_right_yaw[:, 1]
  too_narrow = torch.clamp(
    (minimum_distance - signed_lateral_distance) / minimum_distance,
    min=0.0,
    max=1.0,
  )
  too_wide = torch.clamp(
    (signed_lateral_distance - maximum_distance) / maximum_distance,
    min=0.0,
    max=1.0,
  )
  return torch.square(too_narrow) + too_wide_scale * torch.square(too_wide)
