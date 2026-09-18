"""Convert mimic npz motions to the deployment baselink JSON format.

输出格式逐键复刻部署文件 ``n3_tomas_50fps_baselink.json``（帧×关节/体，float64 全精度，
anchor 四元数 = waist_yaw_link 姿态去掉腰部偏航关节旋转后的 base 系姿态，wxyz）。
npz 的 joint/body 轴保持自身名字顺序原样输出（与部署文件一致，不做 MJCF 重排）::

  uv run python scripts/tools/npz_to_baselink_json.py --npz motions/mimic_data/N3/npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import mjlab


@dataclass
class ConvertConfig:
  npz: str
  """mimic npz 文件或目录（目录 = 全部 *.npz 逐个转）。"""

  output_dir: str = "motions/mimic_data/N3/json_baselink"
  fps: int = 50
  """数据帧率（n3_29dof 库全部为 50，文件名里的 30fps 是历史命名）。"""


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
  """(T,4) wxyz → (T,3,3)。npz 存 float32（范数差 ~1e-7），先归一化。"""
  from scipy.spatial.transform import Rotation

  q = q / np.linalg.norm(q, axis=-1, keepdims=True)
  return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
  """(T,3,3) → (T,4) wxyz，严格单位（部署文件范数误差 2e-16），w ≥ 0 规范符号。"""
  from scipy.spatial.transform import Rotation

  q = Rotation.from_matrix(R).as_quat()  # xyzw
  q = np.concatenate([q[:, 3:4], q[:, 0:3]], axis=-1)  # wxyz
  q[q[:, 0] < 0] *= -1
  return q


def _anchor_quat(
  body_quat_w: np.ndarray, waist_b: int, q_yaw: np.ndarray
) -> np.ndarray:
  """部署约定：base_R = waist_R · Rz(−q_waist_yaw_joint)。"""
  R_waist = _quat_to_mat(body_quat_w[:, waist_b, :])
  c, s = np.cos(-q_yaw), np.sin(-q_yaw)
  zero, one = np.zeros_like(c), np.ones_like(c)
  Rz = np.stack([c, -s, zero, s, c, zero, zero, zero, one], axis=-1).reshape(-1, 3, 3)
  return _mat_to_quat(R_waist @ Rz)


def _convert_one(cfg: ConvertConfig, path: Path) -> Path:
  import json

  z = np.load(path)
  joint_names = [str(n) for n in z["joint_names"]]
  body_names = [str(n) for n in z["body_names"]]
  joint_pos = np.asarray(z["joint_pos"], dtype=np.float64)
  waist_b = body_names.index("waist_yaw_link")
  q_yaw = joint_pos[:, joint_names.index("waist_yaw_joint")]

  # 键顺序与部署文件逐键一致。
  out = {
    "metadata": {
      "original_file": str(path),
      "frames_count": int(joint_pos.shape[0]),
      "joints_count": len(joint_names),
      "anchor_body_name": "waist_yaw_link",
      "body_quat_w_source": f"body_quat_w[:, {waist_b}, :]",
      "anchor_converted_from": "waist_yaw_link",
      "conversion": "base_R = waist_R * Rz(-q_waist_yaw_joint)",
    },
    "joint_pos": joint_pos.tolist(),
    "joint_vel": np.asarray(z["joint_vel"], dtype=np.float64).tolist(),
    "body_quat_w": _anchor_quat(
      np.asarray(z["body_quat_w"], dtype=np.float64), waist_b, q_yaw
    ).tolist(),
    "fps": [cfg.fps],
    "body_pos_w": np.asarray(z["body_pos_w"], dtype=np.float64).tolist(),
    "body_lin_vel_w": np.asarray(z["body_lin_vel_w"], dtype=np.float64).tolist(),
    "body_ang_vel_w": np.asarray(z["body_ang_vel_w"], dtype=np.float64).tolist(),
    "body_names": body_names,
    "joint_names": joint_names,
    "anchor_body_name": ["base_link"],
  }
  out_path = Path(cfg.output_dir) / (path.stem + ".json")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w") as f:
    json.dump(out, f)
  return out_path


def main() -> None:
  cfg = tyro.cli(ConvertConfig, config=mjlab.TYRO_FLAGS)
  root = Path(cfg.npz)
  paths = sorted(root.glob("*.npz")) if root.is_dir() else [root]
  if not paths:
    raise FileNotFoundError(f"No npz found in {cfg.npz}")

  print(f"Converting {len(paths)} motion(s) -> {cfg.output_dir}")
  for i, path in enumerate(paths, start=1):
    out = _convert_one(cfg, path)
    print(f"  [{i}/{len(paths)}] {path.stem} -> {out}")
  print("Done.")


if __name__ == "__main__":
  main()
