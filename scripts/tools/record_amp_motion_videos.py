"""Record AMP motion JSONs to MP4 videos with a following camera.

Reuses the loader/model helpers from ``replay_amp_motion_json``; renders
offscreen with a free camera tracking the root so the robot never leaves
the frame. Example:

  uv run python scripts/tools/record_amp_motion_videos.py \
    --motion motions/amp_data/N3/json,motions/amp_data/N3/recovery
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import tyro
from replay_amp_motion_json import (
  _apply_frame,
  _build_model,
  _joint_qpos_addrs,
  _load_motion,
  _resolve_motion_paths,
)

import mjlab


@dataclass
class RecordConfig:
  motion: str
  """Comma-separated AMP JSON files or directories (dirs expand to *.json)."""

  output_dir: str = "motions/amp_data/N3/videos"
  xml: str | None = None
  """Robot MJCF; defaults to the N3 0905 asset."""

  width: int = 1280
  height: int = 720
  distance: float = 2.8
  """Camera distance to the robot [m]."""

  azimuth: float = 90.0
  """Camera azimuth [deg]; 90 = side view along +x travel."""

  elevation: float = -8.0
  min_seconds: float = 6.0
  """Loop short clips until at least this long."""

  max_loops: int = 2


def _record_one(
  cfg: RecordConfig,
  path: Path,
  meta: dict,
  frames: np.ndarray,
) -> Path:
  # 地面给足半边长：跟拍时机器人走多远都不露边。
  model = _build_model(
    Path(cfg.xml) if cfg.xml else _default_xml(), ground_half_extent=50.0
  )
  # Offscreen framebuffer defaults to 640x480; grow it to fit the requested size.
  model.vis.global_.offwidth = max(model.vis.global_.offwidth, cfg.width)
  model.vis.global_.offheight = max(model.vis.global_.offheight, cfg.height)
  data = mujoco.MjData(model)
  joint_names = list(meta["JointNames"])
  joint_addrs = _joint_qpos_addrs(model, joint_names)
  dt = float(meta["FrameDuration"])
  fps = int(round(1.0 / dt))

  loops = max(1, min(cfg.max_loops, math.ceil(cfg.min_seconds / (len(frames) * dt))))

  renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  lookat = np.zeros(3)

  out_path = Path(cfg.output_dir)
  out_path.mkdir(parents=True, exist_ok=True)
  video_path = out_path / (path.stem + ".mp4")
  writer = imageio.get_writer(
    video_path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p"
  )
  for _ in range(loops):
    for frame in frames:
      _apply_frame(model, data, frame, joint_names, joint_addrs, zero_xy=False)
      # Follow camera: track the root, side view.
      lookat[:] = data.qpos[0:3]
      lookat[2] += 0.05
      cam.lookat[:] = lookat
      cam.distance = cfg.distance
      cam.azimuth = cfg.azimuth
      cam.elevation = cfg.elevation
      renderer.update_scene(data, camera=cam)
      writer.append_data(renderer.render())
  writer.close()
  renderer.close()
  return video_path


def _default_xml() -> Path:
  from mjlab.asset_zoo.robots.N3.constants import N3_XML

  return N3_XML


def main() -> None:
  cfg = tyro.cli(RecordConfig, config=mjlab.TYRO_FLAGS)
  paths = _resolve_motion_paths([m.strip() for m in cfg.motion.split(",")])
  print(f"Recording {len(paths)} motion(s) -> {cfg.output_dir}")
  for i, path in enumerate(paths, start=1):
    meta, frames = _load_motion(path)
    video = _record_one(cfg, path, meta, frames)
    dur = len(frames) * float(meta["FrameDuration"])
    print(f"  [{i}/{len(paths)}] {path.stem}: {dur:.1f}s -> {video}")
  print("Done.")


if __name__ == "__main__":
  main()
