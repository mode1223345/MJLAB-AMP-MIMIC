#!/usr/bin/env python3
"""在同一个 MuJoCo 原生 viewer 中连续回放一个或多个 AMP 运动 JSON。

Example::

  uv run python scripts/tools/replay_amp_motion_json.py \\
    --motion motions/amp_data/labubu_v1/walk_lh_head/labubu_walk_y.json

  uv run python scripts/tools/replay_amp_motion_json.py \\
    --motion motions/amp_data/f5_11dof_walk/f5_walk_f_30fps.json \\
    --speed 0.5 --loop

  uv run python scripts/tools/replay_amp_motion_json.py \\
    --motion motions/amp_data/N3/json \\
    --xml src/mjlab/asset_zoo/robots/N3/xmls/N3.xml --no-loop
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro

import mjlab
from mjlab.asset_zoo.robots.N3.constants import N3_JOINT_NAMES, N3_XML


@dataclass
class ReplayConfig:
  motion: list[str]
  """一个或多个 AMP JSON 文件或目录；目录按文件名展开所有 *.json。"""
  xml: str | None = None
  """机器人 MJCF；默认 N3 0905。"""
  speed: float = 1.0
  """回放倍速（1.0 = 实时）。"""
  loop: bool = True
  """整个动作列表播放结束后是否从头循环。"""
  zero_xy: bool = True
  """将 root 的 x/y 固定为 0，方便在原点观察。"""
  start_frame: int = 0
  """每个动作的起始帧。"""


@dataclass
class _MotionClip:
  path: Path
  frames: np.ndarray
  joint_names: list[str]
  joint_addrs: list[tuple[str, int]]
  dt: float
  start: int


def _resolve_motion_paths(sources: list[str]) -> list[Path]:
  """Preserve explicit file order and sort JSON files within each directory."""
  paths: list[Path] = []
  for source in sources:
    path = Path(source).expanduser().resolve()
    if path.is_dir():
      files = sorted(p for p in path.glob("*.json") if p.is_file())
      if not files:
        raise ValueError(f"No motion JSON files found in {path}")
      paths.extend(files)
    elif path.is_file():
      paths.append(path)
    else:
      raise FileNotFoundError(path)
  if not paths:
    raise ValueError("Provide at least one motion JSON file or directory")
  return paths


def _load_motion(path: Path) -> tuple[dict, np.ndarray]:
  meta = json.loads(path.read_text())
  frames = np.asarray(meta["Frames"], dtype=np.float64)
  if frames.ndim != 2 or frames.shape[0] == 0 or frames.shape[1] < 7:
    raise ValueError(f"Invalid Frames shape: {frames.shape}")
  return meta, frames


def _default_xml_for_joints(joint_names: list[str]) -> Path:
  """Pick MJCF from motion joint names when --xml is omitted."""
  if set(joint_names) == set(N3_JOINT_NAMES):
    return N3_XML
  return N3_XML


def _build_model(xml_path: Path) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(xml_path))
  # 地面，便于判断脚底高度。
  ground = spec.worldbody.add_geom()
  ground.name = "replay_ground"
  ground.type = mujoco.mjtGeom.mjGEOM_PLANE
  ground.size[:] = (5.0, 5.0, 0.1)
  ground.rgba[:] = (0.45, 0.5, 0.55, 1.0)
  return spec.compile()


def _joint_qpos_addrs(
  model: mujoco.MjModel, joint_names: list[str]
) -> list[tuple[str, int]]:
  addrs: list[tuple[str, int]] = []
  for name in joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
      raise KeyError(f"Joint {name!r} not found in MJCF")
    addrs.append((name, int(model.jnt_qposadr[jid])))
  return addrs


def _apply_frame(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  frame: np.ndarray,
  joint_names: list[str],
  joint_addrs: list[tuple[str, int]],
  *,
  zero_xy: bool,
) -> tuple[float, float, float]:
  """把一帧写入 qpos，返回 (vx, vy, wz)。"""
  n_joints = len(joint_names)
  root_pos = frame[0:3].copy()
  quat_xyzw = frame[3:7]
  joint_pos = frame[7 : 7 + n_joints]

  # key_pos (n_keys*3) then lin_vel(3) ang_vel(3) ...
  # 速度偏移不依赖 KeyPosNames 长度时用剩余布局；此处用 JointNames 后紧跟
  # 的标准 AMP 布局：joint | key*3 | lin3 | ang3 | jvel | contact
  # Key 维数 = (frame_dim - 7 - 2*n_joints - 3 - 3 - 2) ，必须能被 3 整除。
  rem = frame.shape[0] - 7 - 2 * n_joints - 3 - 3 - 2
  if rem < 0 or rem % 3 != 0:
    raise ValueError(f"Frame dim {frame.shape[0]} inconsistent with {n_joints} joints")
  lin_start = 7 + n_joints + rem
  lin_vel = frame[lin_start : lin_start + 3]
  ang_vel = frame[lin_start + 3 : lin_start + 6]

  if zero_xy:
    root_pos[0] = 0.0
    root_pos[1] = 0.0

  # freejoint: pos(3) + quat wxyz(4)
  data.qpos[0:3] = root_pos
  data.qpos[3] = quat_xyzw[3]
  data.qpos[4:7] = quat_xyzw[0:3]

  for i, (_name, adr) in enumerate(joint_addrs):
    data.qpos[adr] = joint_pos[i]

  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)
  return float(lin_vel[0]), float(lin_vel[1]), float(ang_vel[2])


def replay(cfg: ReplayConfig) -> None:
  if not np.isfinite(cfg.speed) or cfg.speed <= 0:
    raise ValueError("--speed must be > 0")

  motion_paths = _resolve_motion_paths(cfg.motion)
  loaded = [(path, *_load_motion(path)) for path in motion_paths]
  first_joint_names = list(loaded[0][1]["JointNames"])
  xml_path = (
    Path(cfg.xml).expanduser().resolve()
    if cfg.xml
    else _default_xml_for_joints(first_joint_names)
  )

  model = _build_model(xml_path)
  data = mujoco.MjData(model)
  clips: list[_MotionClip] = []
  for path, meta, frames in loaded:
    joint_names = list(meta["JointNames"])
    if set(joint_names) != set(first_joint_names):
      raise ValueError(
        f"Motion {path} has a different joint set from {motion_paths[0]}"
      )
    dt = float(meta["FrameDuration"])
    if not np.isfinite(dt) or dt <= 0:
      raise ValueError(f"Motion {path}: FrameDuration must be > 0")
    clips.append(
      _MotionClip(
        path=path,
        frames=frames,
        joint_names=joint_names,
        joint_addrs=_joint_qpos_addrs(model, joint_names),
        dt=dt,
        start=max(0, min(cfg.start_frame, len(frames) - 1)),
      )
    )

  print(f"Replay {len(clips)} motion(s), speed={cfg.speed}x")
  for i, clip in enumerate(clips, start=1):
    print(
      f"  [{i}/{len(clips)}] {clip.path.name}: {len(clip.frames)} frames, "
      f"dt={clip.dt:.4f}s, T={(len(clip.frames) - 1) * clip.dt:.2f}s"
    )
  print(f"  xml: {xml_path}")

  clip_i = 0
  frame_i = clips[0].start
  with mujoco.viewer.launch_passive(model, data) as viewer:
    # 稍抬相机，看全身（F5 更高，拉远一点）。
    viewer.cam.distance = 4.0 if "f5" in xml_path.as_posix().lower() else 2.5
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 120
    while viewer.is_running():
      clip = clips[clip_i]
      t0 = time.perf_counter()
      vx, vy, wz = _apply_frame(
        model,
        data,
        clip.frames[frame_i],
        clip.joint_names,
        clip.joint_addrs,
        zero_xy=cfg.zero_xy,
      )
      viewer.sync()
      print(
        f"\r[{clip_i + 1}/{len(clips)}] {clip.path.name}  "
        f"frame {frame_i:4d}/{len(clip.frames) - 1}  "
        f"t={frame_i * clip.dt:6.2f}s  vx={vx:+.3f} vy={vy:+.3f} wz={wz:+.3f}",
        end="",
        flush=True,
      )
      elapsed = time.perf_counter() - t0
      sleep_dt = clip.dt / cfg.speed
      if sleep_dt > elapsed:
        time.sleep(sleep_dt - elapsed)
      frame_i += 1
      if frame_i >= len(clip.frames):
        clip_i += 1
        if clip_i == len(clips):
          if not cfg.loop:
            break
          clip_i = 0
        frame_i = clips[clip_i].start
        print()
  print()


if __name__ == "__main__":
  replay(tyro.cli(ReplayConfig, config=mjlab.TYRO_FLAGS))
