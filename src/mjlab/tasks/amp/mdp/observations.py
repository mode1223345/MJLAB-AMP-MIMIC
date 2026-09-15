"""AMP observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def current_feet_contact(
  env: ManagerBasedRlEnv, sensor_name: str = "feet_ground_contact"
) -> torch.Tensor:
  """Binary feet contact flags from a ContactSensor (shape ``[B, num_feet]``)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float()


def euler_angles(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Root roll/pitch from world quaternion (Noetix AMP policy obs, shape ``[B, 2]``)."""
  asset: Entity = env.scene[asset_cfg.name]
  roll, pitch, _yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
  return torch.stack([roll, pitch], dim=1)


def amp_joint_pos(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Joint positions remapped into motion-file joint order when possible."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  motion_loader = getattr(env.unwrapped, "motion_loader", None)
  if motion_loader is not None:
    joint_pos = joint_pos[..., motion_loader.joint_mapping["sim2motion"]]
  return joint_pos


def amp_joint_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Joint velocities remapped into motion-file joint order when possible."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  motion_loader = getattr(env.unwrapped, "motion_loader", None)
  if motion_loader is not None:
    joint_vel = joint_vel[..., motion_loader.joint_mapping["sim2motion"]]
  return joint_vel


def amp_key_pos(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Key body positions in the root frame, flattened (N, num_bodies * 3)."""
  asset: Entity = env.scene[asset_cfg.name]
  root_pos = asset.data.root_link_pos_w[:, :3].unsqueeze(1)
  key_pos = asset.data.body_link_pos_w.clone()
  motion_loader = getattr(env.unwrapped, "motion_loader", None)
  if motion_loader is not None:
    key_pos = key_pos[:, motion_loader.key_pos_mapping["sim2motion"], :]
  return (key_pos - root_pos).flatten(1, 2)
