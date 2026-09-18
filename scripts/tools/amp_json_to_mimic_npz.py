"""Convert an AMP motion json to the mimic npz format (baselink).

AMP json 帧布局（见 csv_to_amp_json.py 头注释）：root_pos(3) ‖ root_quat xyzw(4) ‖
joint_pos(29) ‖ key_pos(90) ‖ root_lin_vel(3,root系) ‖ root_ang_vel(3,root系) ‖
joint_vel(29) ‖ feet_contact(2)。转 mimic npz：root/joint 直取，joint_vel 用 json
现成值，30 体世界系位姿用本地 N3.xml FK，体速度前向差分（与 csv_to_mimic_npz 同
约定）。输出名字顺序照抄库参考 npz，float32 十键。Example::

  uv run python scripts/tools/amp_json_to_mimic_npz.py \
    --json motions/amp_data/N3/recovery/n3_起身_50hz.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro
from csv_to_mimic_npz import _ang_vel_forward, _vel_forward

import mjlab


@dataclass
class ConvertConfig:
  json: str
  """AMP json 文件或目录（目录 = 全部 *.json 逐个转）。"""

  output_dir: str = "motions/mimic_data/N3/npz_baselink"
  library_ref: str = "motions/mimic_data/N3/npz_baselink/n3_侧空翻_30fps.npz"
  """库参考 npz（只读，照抄 joint/body 名字顺序）。"""


def _convert_one(
  cfg: ConvertConfig, json_path: Path, model: mujoco.MjModel
) -> tuple[Path, int]:
  with json_path.open() as f:
    d = json.load(f)
  frames = np.array(d["Frames"], dtype=np.float64)
  fps = int(round(1.0 / d["FrameDuration"]))
  json_joint_names = [str(n) for n in d["JointNames"]]

  root_pos = frames[:, 0:3]
  root_quat_wxyz = frames[:, 3:7][:, [3, 0, 1, 2]]  # json 是 xyzw
  joint_pos = frames[:, 7:36]
  joint_vel = frames[:, 132:161]

  ref = np.load(cfg.library_ref)
  out_joint_names = [str(s) for s in ref["joint_names"]]
  out_body_names = [str(s) for s in ref["body_names"]]

  addrs: list[int] = []
  for name in json_joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
      raise KeyError(f"Joint {name!r} not found in MJCF")
    addrs.append(int(model.jnt_qposadr[jid]))
  body_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in out_body_names
  ]
  data = mujoco.MjData(model)
  T = len(frames)
  body_pos = np.empty((T, len(body_ids), 3))
  body_quat = np.empty((T, len(body_ids), 4))
  for t in range(T):
    data.qpos[0:3] = root_pos[t]
    data.qpos[3:7] = root_quat_wxyz[t]
    for adr, val in zip(addrs, joint_pos[t], strict=True):
      data.qpos[adr] = val
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for b, bid in enumerate(body_ids):
      body_pos[t, b] = data.xpos[bid]
      body_quat[t, b] = data.xquat[bid]

  # json 关节轴按名字重排到库顺序；joint_vel 同序取现成值。
  reorder = [json_joint_names.index(n) for n in out_joint_names]
  dt = 1.0 / fps
  out = {
    "fps": np.array([fps], dtype=np.int64),
    "joint_names": np.array(out_joint_names),
    "body_names": np.array(out_body_names),
    "joint_pos": joint_pos[:, reorder].astype(np.float32),
    "joint_vel": joint_vel[:, reorder].astype(np.float32),
    "body_pos_w": body_pos.astype(np.float32),
    "body_quat_w": body_quat.astype(np.float32),
    "body_lin_vel_w": _vel_forward(body_pos, dt).astype(np.float32),
    "body_ang_vel_w": _ang_vel_forward(body_quat, dt).astype(np.float32),
  }
  out_path = Path(cfg.output_dir) / (json_path.stem + ".npz")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out_path, **out)
  return out_path, T


def main() -> None:
  cfg = tyro.cli(ConvertConfig, config=mjlab.TYRO_FLAGS)
  root = Path(cfg.json)
  paths = sorted(root.glob("*.json")) if root.is_dir() else [root]
  if not paths:
    raise FileNotFoundError(f"No json found in {cfg.json}")

  from mjlab.asset_zoo.robots.N3.constants import N3_XML

  model = mujoco.MjModel.from_xml_path(str(N3_XML))
  print(f"Converting {len(paths)} motion(s) -> {cfg.output_dir}")
  for i, path in enumerate(paths, start=1):
    out_path, T = _convert_one(cfg, path, model)
    print(f"  [{i}/{len(paths)}] {path.stem} ({T} frames) -> {out_path}")
  print("Done.")


if __name__ == "__main__":
  main()
