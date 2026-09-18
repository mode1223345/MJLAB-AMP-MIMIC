"""Record mimic npz motions to MP4 videos with a following camera.

复用 replay_amp_motion_json 的场景构建（渐变天空盒 + 双灯 + 棋盘格地面——
N3.xml 本身无灯无天空无地面）与跟拍相机：自由相机逐帧 lookat 根位置，
方位/仰角/距离恒定，机器人翻多远都在画面里。npz 的 joint/body 轴按名字
重排到 MJCF 顺序（与 MotionLoader 同规则）。Example::

  uv run python scripts/tools/record_mimic_npz_videos.py \
    --npz motions/mimic_data/N3/npz_baselink
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import tyro
from replay_amp_motion_json import _build_model

import mjlab


@dataclass
class RecordConfig:
  npz: str
  """mimic npz 文件或目录（目录 = 全部 *.npz 逐个录）。"""

  output_dir: str = "motions/mimic_data/N3/videos"
  xml: str | None = None
  """机器人 MJCF；默认 N3 0905。"""

  fps: int = 50
  """数据帧率（n3_29dof 库全部为 50，文件名里的 30fps 是历史命名）。"""

  width: int = 1280
  height: int = 720
  distance: float = 3.2
  """跟拍距离 [m]；空翻腾空弧度大，比 AMP 行走(2.8)稍远。"""

  azimuth: float = 310.0
  """相机方位角 [deg]；310 = 310-180=130 的正面机位（130 实拍是背面），3/4 侧前。"""

  elevation: float = -12.0


def _default_xml() -> Path:
  from mjlab.asset_zoo.robots.N3.constants import N3_XML

  return N3_XML


def _record_one(cfg: RecordConfig, path: Path, model: mujoco.MjModel) -> Path:
  data_npz = np.load(path)
  npz_joints = [str(n) for n in data_npz["joint_names"]]
  npz_bodies = [str(n) for n in data_npz["body_names"]]
  base_b = npz_bodies.index("base_link")

  # 名字 → MJCF qpos 地址：npz 关节轴保持自身顺序，逐名查模型地址写入
  # （npz 顺序 ≠ MJCF，不能按位置直拷）。
  addrs: list[int] = []
  for name in npz_joints:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
      raise KeyError(f"Joint {name!r} not found in MJCF")
    addrs.append(int(model.jnt_qposadr[jid]))

  joint_pos = np.asarray(data_npz["joint_pos"], dtype=np.float64)
  body_pos = np.asarray(data_npz["body_pos_w"], dtype=np.float64)[:, base_b]
  body_quat = np.asarray(data_npz["body_quat_w"], dtype=np.float64)[:, base_b]

  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE

  out_path = Path(cfg.output_dir)
  out_path.mkdir(parents=True, exist_ok=True)
  video_path = out_path / (path.stem + ".mp4")
  writer = imageio.get_writer(
    video_path, fps=cfg.fps, codec="libx264", quality=8, pixelformat="yuv420p"
  )
  for t in range(joint_pos.shape[0]):
    data.qpos[0:3] = body_pos[t]
    data.qpos[3:7] = body_quat[t]  # npz 四元数即 wxyz，与 MuJoCo 同序
    for adr, val in zip(addrs, joint_pos[t], strict=True):
      data.qpos[adr] = val
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    # Follow camera: track the root, fixed azimuth/elevation/distance.
    cam.lookat[:] = data.qpos[0:3]
    cam.lookat[2] += 0.05
    cam.distance = cfg.distance
    cam.azimuth = cfg.azimuth
    cam.elevation = cfg.elevation
    renderer.update_scene(data, camera=cam)
    writer.append_data(renderer.render())
  writer.close()
  renderer.close()
  return video_path


def main() -> None:
  cfg = tyro.cli(RecordConfig, config=mjlab.TYRO_FLAGS)
  root = Path(cfg.npz)
  paths = sorted(root.glob("*.npz")) if root.is_dir() else [root]
  if not paths:
    raise FileNotFoundError(f"No npz found in {cfg.npz}")

  model = _build_model(
    Path(cfg.xml).expanduser() if cfg.xml else _default_xml(),
    ground_half_extent=50.0,  # 跟拍时地面给足，机器人走多远都不露边
  )
  model.vis.global_.offwidth = max(model.vis.global_.offwidth, cfg.width)
  model.vis.global_.offheight = max(model.vis.global_.offheight, cfg.height)

  print(f"Recording {len(paths)} motion(s) -> {cfg.output_dir}")
  for i, path in enumerate(paths, start=1):
    video = _record_one(cfg, path, model)
    print(f"  [{i}/{len(paths)}] {path.stem} -> {video}")
  print("Done.")


if __name__ == "__main__":
  main()
