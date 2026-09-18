from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)


def motion_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Clip boundary (Isaac mimic): within 2 frames of the assigned clip's end.

  Marked time_out=True in the cfg so the bootstrapped value target treats a
  finished motion as a truncation, not a failure.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.time_steps >= command.motion_end_steps - 2


def bad_global_anchor_ori(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  """World-frame anchor heading/attitude error (rad) exceeds threshold.

  Unlike :func:`bad_anchor_ori` (gravity-z delta, tilt only), this also fires
  on yaw drift — flips/spins make it the meaningful check.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
  return error > threshold


def bad_motion_body_pos_z_only_in_base(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Z error of tracked bodies measured in their own base frame (Isaac variant).

  Both reference and robot body positions are expressed relative to their own
  base_link before the z comparison, so world-z drift of the anchor does not
  leak into the check (matters for motions with large height swings).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  base_body_index = command.cfg.body_names.index("base_link")

  motion_base_pos_w = command.body_pos_w[:, base_body_index]
  motion_base_quat_w = command.body_quat_w[:, base_body_index]
  robot_base_pos_w = command.robot_body_pos_w[:, base_body_index]
  robot_base_quat_w = command.robot_body_quat_w[:, base_body_index]

  motion_body_pos_b = quat_apply_inverse(
    motion_base_quat_w[:, None, :].expand(-1, len(body_indexes), -1),
    command.body_pos_w[:, body_indexes] - motion_base_pos_w[:, None, :],
  )
  robot_body_pos_b = quat_apply_inverse(
    robot_base_quat_w[:, None, :].expand(-1, len(body_indexes), -1),
    command.robot_body_pos_w[:, body_indexes] - robot_base_pos_w[:, None, :],
  )

  error = torch.abs(motion_body_pos_b[:, :, -1] - robot_body_pos_b[:, :, -1])
  return torch.any(error > threshold, dim=-1)
