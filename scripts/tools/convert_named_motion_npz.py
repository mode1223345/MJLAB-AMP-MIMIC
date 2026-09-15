"""Convert Isaac-Lab-style named motion NPZ into mjlab MuJoCo-indexed NPZ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tyro
from mjlab.asset_zoo.robots.bumi3_4340.bumi3_constants import get_bumi3_robot_cfg

import mjlab
from mjlab.entity import Entity


def convert_named_npz_to_mjlab(
  input_file: str,
  output_file: str,
) -> None:
  """Reorder joints/bodies by name into mjlab Entity index order.

  Isaac Lab NPZs store arrays ordered by PhysX body/joint discovery order and
  include ``joint_names`` / ``body_names``. mjlab's MotionLoader indexes body
  arrays by MuJoCo entity body index, so we expand/reorder into that layout.
  """
  src = np.load(input_file, allow_pickle=True)
  required = (
    "fps",
    "joint_names",
    "body_names",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  )
  missing = [k for k in required if k not in src.files]
  if missing:
    raise KeyError(f"Input NPZ missing keys: {missing}")

  src_joint_names = [str(n) for n in src["joint_names"].tolist()]
  src_body_names = [str(n) for n in src["body_names"].tolist()]

  robot = Entity(get_bumi3_robot_cfg())
  dst_joint_names = list(robot.joint_names)
  dst_body_names = list(robot.body_names)

  missing_joints = [n for n in dst_joint_names if n not in src_joint_names]
  missing_bodies = [n for n in dst_body_names if n not in src_body_names]
  if missing_joints:
    raise ValueError(f"Source NPZ missing joints: {missing_joints}")
  if missing_bodies:
    raise ValueError(f"Source NPZ missing bodies: {missing_bodies}")

  num_frames = src["joint_pos"].shape[0]
  joint_pos = np.zeros((num_frames, len(dst_joint_names)), dtype=np.float32)
  joint_vel = np.zeros_like(joint_pos)
  for i, name in enumerate(dst_joint_names):
    j = src_joint_names.index(name)
    joint_pos[:, i] = src["joint_pos"][:, j]
    joint_vel[:, i] = src["joint_vel"][:, j]

  body_pos_w = np.zeros((num_frames, len(dst_body_names), 3), dtype=np.float32)
  body_quat_w = np.zeros((num_frames, len(dst_body_names), 4), dtype=np.float32)
  body_quat_w[..., 0] = 1.0
  body_lin_vel_w = np.zeros_like(body_pos_w)
  body_ang_vel_w = np.zeros_like(body_pos_w)
  for i, name in enumerate(dst_body_names):
    j = src_body_names.index(name)
    body_pos_w[:, i] = src["body_pos_w"][:, j]
    body_quat_w[:, i] = src["body_quat_w"][:, j]
    body_lin_vel_w[:, i] = src["body_lin_vel_w"][:, j]
    body_ang_vel_w[:, i] = src["body_ang_vel_w"][:, j]

  fps = np.asarray(src["fps"]).reshape(-1)
  out = Path(output_file)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    out,
    fps=fps,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
    # Keep names so downstream tools (JSON/ONNX) can remap without guessing.
    joint_names=np.asarray(dst_joint_names),
    body_names=np.asarray(dst_body_names),
  )
  print(f"[OK] Wrote mjlab motion NPZ: {out}")
  print(
    f"  frames={num_frames}, joints={len(dst_joint_names)}, bodies={len(dst_body_names)}"
  )
  print(f"  joint order: {dst_joint_names}")
  print(f"  body order: {dst_body_names}")


def main(
  input_file: str = (
    "/home/user/legged_lab/source/NoetixRobot/NoetixRobot/assets/"
    "datasets/mimic_data/4340/0706/3牛仔_1.npz"
  ),
  output_file: str = "motions/bumi3_cowboy_mjlab.npz",
) -> None:
  convert_named_npz_to_mjlab(input_file, output_file)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
