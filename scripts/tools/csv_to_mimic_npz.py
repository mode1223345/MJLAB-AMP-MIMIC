"""Regenerate mimic npz from the original retarget CSVs with the local N3 MJCF.

数据源是动捕重定向的原始 CSV（root 位姿 + 29 关节，全精度），机器人用本地
asset_zoo 的 N3.xml（最新机器人数据）做 FK——anchor 即 CSV root = base_link。
输出与现有动作库同格式（10 键 float32，joint/body 名字顺序照抄库里同名 npz，
保持 MotionLoader/部署消费端 drop-in 兼容）。速度 = 前向差分 @fps，末帧重复；
角速度由四元数旋转差 log 求得。Example::

  uv run python scripts/tools/csv_to_mimic_npz.py \
    --csv "/path/to/motion-csv/高动态/n3" \
    --library motions/mimic_data/N3/npz_baselink
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

import mjlab


@dataclass
class RegenerateConfig:
  csv: str
  """原始 CSV 文件或目录（目录 = 全部 *.csv 逐个转）。"""

  output_dir: str = "motions/mimic_data/N3/npz_baselink"
  library: str = "motions/mimic_data/N3/npz_baselink"
  """现有 npz 动作库（只读，用来照抄 joint/body 名字顺序）。"""

  fps: int = 50
  """数据帧率（文件名里的 30fps 是历史命名）。"""


def _mcj(col: str) -> str:
  """CSV 关节列名 -> MJCF 关节名（hip.pitch.l -> l_hip_pitch_joint）。"""
  p = col.split(".")
  return f"{p[2]}_{p[0]}_{p[1]}_joint" if len(p) == 3 else f"{p[0]}_{p[1]}_joint"


def _vel_forward(x: np.ndarray, dt: float) -> np.ndarray:
  """前向差分速度，末帧重复（库 jv[0]==前向差分[0] 的同款约定）。"""
  v = np.empty_like(x)
  v[:-1] = (x[1:] - x[:-1]) / dt
  v[-1] = v[-2]
  return v


def _ang_vel_forward(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
  """(T,B,4) wxyz -> 世界系角速度 (T,B,3)：ω = log(R[t+1]·R[t]ᵀ)/dt。"""
  q = quat_wxyz / np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
  R = Rotation.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()
  R_rel = R[1:] @ np.swapaxes(R[:-1], -1, -2)
  w = Rotation.from_matrix(R_rel).as_rotvec() / dt
  return np.concatenate([w, w[-1:]], axis=0)


def _regenerate_one(
  cfg: RegenerateConfig, csv_path: Path, model: mujoco.MjModel
) -> tuple[Path, int]:
  import csv as csv_mod

  rows = list(csv_mod.DictReader(csv_path.open()))
  cols = list(rows[0].keys())
  joint_cols = cols[7:]  # 前 7 列 = root_pos_xyz + root_rot_xyzw
  csv_joint_names = [_mcj(c) for c in joint_cols]

  root_pos = np.array(
    [[float(r[f"root_pos_{a}"]) for a in "xyz"] for r in rows], dtype=np.float64
  )
  root_quat_xyzw = np.array(
    [[float(r[f"root_rot_{a}"]) for a in "xyzw"] for r in rows], dtype=np.float64
  )
  joint_pos = np.array(
    [[float(r[c]) for c in joint_cols] for r in rows], dtype=np.float64
  )

  # 名字顺序照抄库里同名 npz（drop-in 兼容）。
  ref = np.load(Path(cfg.library) / (csv_path.stem + ".npz"))
  out_joint_names = [str(s) for s in ref["joint_names"]]
  out_body_names = [str(s) for s in ref["body_names"]]

  # FK：qpos = root + 29 关节，读全 30 体世界位姿。
  addrs: list[int] = []
  for name in csv_joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
      raise KeyError(f"Joint {name!r} not found in MJCF")
    addrs.append(int(model.jnt_qposadr[jid]))
  body_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in out_body_names
  ]
  data = mujoco.MjData(model)
  T = len(rows)
  body_pos = np.empty((T, len(body_ids), 3))
  body_quat = np.empty((T, len(body_ids), 4))
  for t in range(T):
    data.qpos[0:3] = root_pos[t]
    data.qpos[3:7] = root_quat_xyzw[t][[3, 0, 1, 2]]  # xyzw -> wxyz
    for adr, val in zip(addrs, joint_pos[t], strict=True):
      data.qpos[adr] = val
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for b, bid in enumerate(body_ids):
      body_pos[t, b] = data.xpos[bid]
      body_quat[t, b] = data.xquat[bid]

  # 输出按库里顺序重排关节轴；速度前向差分；float32 与库一致。
  reorder = [csv_joint_names.index(n) for n in out_joint_names]
  dt = 1.0 / cfg.fps
  out = {
    "fps": np.array([cfg.fps], dtype=np.int64),
    "joint_names": np.array(out_joint_names),
    "body_names": np.array(out_body_names),
    "joint_pos": (joint_pos[:, reorder]).astype(np.float32),
    "joint_vel": _vel_forward(joint_pos[:, reorder], dt).astype(np.float32),
    "body_pos_w": body_pos.astype(np.float32),
    "body_quat_w": body_quat.astype(np.float32),
    "body_lin_vel_w": _vel_forward(body_pos, dt).astype(np.float32),
    "body_ang_vel_w": _ang_vel_forward(body_quat, dt).astype(np.float32),
  }
  out_path = Path(cfg.output_dir) / (csv_path.stem + ".npz")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out_path, **out)
  return out_path, T


def main() -> None:
  cfg = tyro.cli(RegenerateConfig, config=mjlab.TYRO_FLAGS)
  root = Path(cfg.csv)
  paths = sorted(root.glob("*.csv")) if root.is_dir() else [root]
  if not paths:
    raise FileNotFoundError(f"No csv found in {cfg.csv}")

  from mjlab.asset_zoo.robots.N3.constants import N3_XML

  model = mujoco.MjModel.from_xml_path(str(N3_XML))
  print(f"Regenerating {len(paths)} motion(s) -> {cfg.output_dir}")
  for i, path in enumerate(paths, start=1):
    out_path, T = _regenerate_one(cfg, path, model)
    print(f"  [{i}/{len(paths)}] {path.stem} ({T} frames) -> {out_path}")
  print("Done.")


if __name__ == "__main__":
  main()
