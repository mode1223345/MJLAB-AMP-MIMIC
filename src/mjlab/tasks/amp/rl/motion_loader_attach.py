"""Shared MotionLoader construction from a live environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .motion_loader import MotionLoader

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def build_motion_loader(
  env: ManagerBasedRlEnv,
  motion_files: list[str],
  reference_observation_horizon: int,
  num_preload_transitions: int,
) -> MotionLoader:
  """Build a MotionLoader bound to the env's robot joint/body order."""
  unwrapped = env.unwrapped
  robot = unwrapped.scene["robot"]
  return MotionLoader(
    device=str(unwrapped.device),
    time_between_frames=float(unwrapped.step_dt),
    reference_observation_horizon=reference_observation_horizon,
    num_preload_transitions=num_preload_transitions,
    joint_pos_size=len(robot.joint_names),
    key_pos_local_size=len(robot.body_names) * 3,
    motion_files=motion_files,
    sim_joint_names=list(robot.joint_names),
    sim_key_pos_names=list(robot.body_names),
  )
