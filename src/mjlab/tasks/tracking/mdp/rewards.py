from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


# ---------------- Isaac mimic_noetix_n3_mha reward ports ----------------


def motion_anchor_height_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """exp(−(anchor_z − robot_anchor_z)²/std²): 腾空/下蹲高度跟踪。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.square(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
  return torch.exp(-error / std**2)


def motion_global_anchor_gravity_error_exp(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """exp(−‖g_ref_b − g_robot_b‖²/threshold²): 锚点倾斜方向(含横滚/俯仰)跟踪。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, command.robot.data.gravity_vec_w
  )
  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, command.robot.data.gravity_vec_w
  )
  error = torch.sum(
    torch.square(motion_projected_gravity_b - robot_projected_gravity_b), dim=-1
  )
  return torch.exp(-error / threshold**2)


def motion_anchor_to_support_foot_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  ref_contact_z: float,
  foot_body_names: tuple[str, ...],
) -> torch.Tensor:
  """单支撑相(参考一脚 z < ref_contact_z)时, 锚点对支撑脚的水平距离误差。

  比较的是"锚点→脚"向量在参考与机器人之间的一致性(平移漂移抵消)。
  非单支撑相位给 0。
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  foot_indexes = _get_body_indexes(command, foot_body_names)

  ref_contact = command.body_pos_w[:, foot_indexes, 2] < ref_contact_z
  left_single_support = ref_contact[:, 0] & ~ref_contact[:, 1]
  right_single_support = ref_contact[:, 1] & ~ref_contact[:, 0]
  support_weights = torch.stack(
    [left_single_support, right_single_support], dim=1
  ).float()

  ref_anchor_to_foot_xy = (
    command.anchor_pos_w[:, None, :2] - command.body_pos_w[:, foot_indexes, :2]
  )
  robot_anchor_to_foot_xy = (
    command.robot_anchor_pos_w[:, None, :2]
    - command.robot_body_pos_w[:, foot_indexes, :2]
  )
  error = torch.sum(
    torch.square(ref_anchor_to_foot_xy - robot_anchor_to_foot_xy), dim=-1
  )
  error = torch.sum(error * support_weights, dim=1)
  single_support = torch.sum(support_weights, dim=1)

  return single_support * torch.exp(-error / std**2)


def motion_feet_z_under_lift_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """参考脚已抬起的额外高度没跟上 → 罚(单向: 机器人脚低于参考脚才算)。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  ref_z_rel = command.body_pos_w[:, body_indexes, 2] - command.anchor_pos_w[:, None, 2]
  robot_z_rel = (
    command.robot_body_pos_w[:, body_indexes, 2]
    - command.robot_anchor_pos_w[:, None, 2]
  )
  under_lift = torch.clamp(ref_z_rel - robot_z_rel, min=0.0)
  return torch.exp(-torch.square(under_lift).mean(-1) / std**2)


def motion_anchor_z_under_lift_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """锚点垂直高度欠跟(单向): 机器人锚点低于参考锚点才算误差。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  under_lift = torch.clamp(
    command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2], min=0.0
  )
  return torch.exp(-torch.square(under_lift) / std**2)


def motion_swing_feet_height_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  contact_height: float = 0.08,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """摆动脚(参考脚 z > contact_height)的高度误差, 仅对摆动脚平均。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)

  ref_z = command.body_pos_relative_w[:, body_indexes, 2]
  robot_z = command.robot_body_pos_w[:, body_indexes, 2]
  swing_mask = ref_z > contact_height

  error = torch.square(ref_z - robot_z) * swing_mask.float()
  denom = swing_mask.float().sum(dim=-1).clamp(min=1.0)
  error = error.sum(dim=-1) / denom

  return torch.exp(-error / std**2)


def motion_root_vertical_velocity_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """根部垂直速度 vz 跟踪(决定腾空时长/起跳时机)。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  ref_root_vz = command.body_lin_vel_w[:, 0, 2]
  robot_root_vz = command.robot_body_lin_vel_w[:, 0, 2]
  error = torch.square(ref_root_vz - robot_root_vz)
  return torch.exp(-error / std**2)


def _get_joint_indexes(
  command: MotionCommand, joint_names: tuple[str, ...] | None
) -> torch.Tensor | None:
  """Indexes into the motion joint axis (== robot joint order) by name."""
  if joint_names is None:
    return None
  cache = getattr(command, "_joint_index_cache", None)
  if cache is None:
    cache = command._joint_index_cache = {}
  key = tuple(joint_names)
  if key not in cache:
    cache[key] = torch.tensor(
      command.robot.find_joints(list(joint_names), preserve_order=True)[0],
      dtype=torch.long,
      device=command.joint_pos.device,
    )
  return cache[key]


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  joint_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """参考关节角跟踪 exp 核; joint_names=None 时全部关节。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  joint_indexes = _get_joint_indexes(command, joint_names)
  if joint_indexes is None:
    error = torch.square(command.joint_pos - command.robot_joint_pos)
  else:
    error = torch.square(
      command.joint_pos[:, joint_indexes] - command.robot_joint_pos[:, joint_indexes]
    )
  return torch.exp(-error.mean(-1) / std**2)


def motion_joint_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  joint_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """参考关节速度跟踪 exp 核(空翻收腿/鞭腿的时序)。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  joint_indexes = _get_joint_indexes(command, joint_names)
  if joint_indexes is None:
    error = torch.square(command.joint_vel - command.robot_joint_vel)
  else:
    error = torch.square(
      command.joint_vel[:, joint_indexes] - command.robot_joint_vel[:, joint_indexes]
    )
  return torch.exp(-error.mean(-1) / std**2)


def _sensor_primary_indexes(
  sensor: ContactSensor, body_names: tuple[str, ...]
) -> list[int]:
  """Sensor-primary index for each body name (order-safe pairing)."""
  names = sensor.primary_names
  return [names.index(name) for name in body_names]


def _max_force_over_history(sensor: ContactSensor) -> torch.Tensor:
  """[B, P] max net-contact-force norm over the sensor force-history window."""
  assert sensor.data.force_history is not None, (
    f"Sensor '{sensor.cfg.name}' needs history_length > 0 for this reward."
  )
  # force_history: [B, N, H, 3]
  return torch.norm(sensor.data.force_history, dim=-1).max(dim=-1).values


def motion_feet_air_contact_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  ref_air_z: float,
  ref_air_margin: float,
  contact_threshold: float,
  contact_force_margin: float,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  """参考脚在空中(摆动相)时机器人同侧脚却触地 → 软惩罚。

  双斜坡门(参考脚高度门 × 接触力门)相乘, 处处连续有梯度。
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  sensor: ContactSensor = env.scene[sensor_name]
  body_indexes = _get_body_indexes(command, body_names)
  sensor_indexes = _sensor_primary_indexes(sensor, body_names)

  ref_foot_z = command.body_pos_w[:, body_indexes, 2]
  ref_air_weight = torch.clamp(
    (ref_foot_z - ref_air_z) / ref_air_margin, min=0.0, max=1.0
  )

  contact_force = _max_force_over_history(sensor)[:, sensor_indexes]
  contact_weight = torch.clamp(
    (contact_force - contact_threshold) / contact_force_margin, min=0.0, max=1.0
  )

  return torch.mean(ref_air_weight * contact_weight, dim=1)


def motion_support_foot_slide_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  ref_contact_z: float,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  """支撑脚(参考脚 z < ref_contact_z)的水平打滑速度均值。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)

  ref_contact = command.body_pos_w[:, body_indexes, 2] < ref_contact_z
  foot_xy_vel = torch.norm(command.robot_body_lin_vel_w[:, body_indexes, :2], dim=-1)
  num_support_feet = torch.clamp(ref_contact.float().sum(dim=1), min=1.0)

  return torch.sum(ref_contact.float() * foot_xy_vel, dim=1) / num_support_feet


def feet_slide_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Isaac mimic 版 feet_slide: 接触判定取力史窗口内最大净力 > 1 N。"""
  sensor: ContactSensor = env.scene[sensor_name]
  contacts = _max_force_over_history(sensor) > 1.0
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice) or (
    isinstance(body_ids, (list, tuple)) and len(body_ids) == 0
  ):
    _, names = asset.find_bodies(".*_ankle_roll_link")
    body_ids = [asset.body_names.index(n) for n in names]
    sensor_indexes = _sensor_primary_indexes(
      sensor, tuple(cast("tuple[str, ...]", names))
    )
  else:
    body_names = tuple(asset.body_names[i] for i in body_ids)
    sensor_indexes = _sensor_primary_indexes(sensor, body_names)
  body_vel = asset.data.body_link_lin_vel_w[:, body_ids, :2]
  return torch.sum(body_vel.norm(dim=-1) * contacts[:, sensor_indexes].float(), dim=1)


def undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 1.0,
) -> torch.Tensor:
  """非脚部位触地惩罚: 传感器(只含非脚碰撞体)上净力超阈值的部位计数。"""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  body_forces = torch.norm(sensor.data.force, dim=-1)  # [B, P]
  return (body_forces > threshold).sum(dim=1).float()


def joint_vel_limit_margin_penalty(
  env: ManagerBasedRlEnv,
  velocity_limits: dict[str, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ratio: float = 0.9,
) -> torch.Tensor:
  """(|q̇|/额定转速 − ratio)₊² 求和: 逼策略留在 90% 限速内(sim2real 余量)。"""
  asset: Entity = env.scene[asset_cfg.name]
  cache = getattr(env, "_mimic_joint_vel_limits", None)
  if cache is None:
    device = asset.data.joint_pos.device
    cache = torch.tensor(
      [velocity_limits.get(name, 1e9) for name in asset.joint_names],
      dtype=torch.float32,
      device=device,
    )
    env._mimic_joint_vel_limits = cache
  limit = cache[asset_cfg.joint_ids]
  exceed = asset.data.joint_vel[:, asset_cfg.joint_ids].abs() / limit - ratio
  return torch.sum(torch.square(exceed.clamp(min=0.0)), dim=1)


def joint_effort_limit_margin_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ratio: float = 0.9,
) -> torch.Tensor:
  """(|τ|/额定力矩 − ratio)₊² 求和, 执行器空间取限位(sim2real 余量)。"""
  asset: Entity = env.scene[asset_cfg.name]
  force = asset.data.actuator_force
  cache = getattr(env, "_mimic_actuator_effort_limits", None)
  if cache is None:
    cache = torch.full((force.shape[1],), 1e9, device=force.device)
    for act in asset.actuators:
      effort = getattr(act.cfg, "effort_limit", None)
      if effort is None:
        continue
      cache[act.ctrl_ids] = float(effort)
    env._mimic_actuator_effort_limits = cache
  exceed = force.abs() / cache - ratio
  return torch.sum(torch.square(exceed.clamp(min=0.0)), dim=1)
