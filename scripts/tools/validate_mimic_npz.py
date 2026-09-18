"""校验 mimic npz 是否满足 tracking 任务 MotionLoader 的硬要求。

移植自 Isaac 端 legged_lab/scripts/motion/validate_mimic_npz.py，只依赖 numpy
（关节表从 mjlab asset_zoo 导入）。用 `uv run python scripts/tools/validate_mimic_npz.py
--npz <file_or_dir>`。

检查(ERROR, 任一即退出码 1):
  1. 9 个必需键齐全: fps, joint_names, body_names, joint_pos, joint_vel,
     body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
  2. joint_names 覆盖机器人全部 29 关节且无重复
  3. body_names 覆盖任务 21 个 tracking body 且无重复
  4. fps == 50 (任务按 50Hz 控制率逐帧步进, 非 50fps 数据速度全错)
  5. 形状: joint_pos/joint_vel [T, Nj]; body_pos_w/lin_vel/ang_vel [T, B, 3];
     body_quat_w [T, B, 4] (wxyz); T >= 2; Nj == len(joint_names); B == len(body_names)
  6. 所有数组有限(无 NaN/Inf)
  7. 四元数模长 ≈ 1 (atol 1e-3)

检查(WARN, 不失败): 根平均 z 不在站立带 / 踝穿地 / joint_vel 与差分不一致 / 时长 < 10s。
"""

import argparse
import os
import sys

import numpy as np

from mjlab.asset_zoo.robots.N3.constants import N3_JOINT_NAMES

# 21 tracking body ← tasks/tracking/config/N3/env_cfg.py (N3_MIMIC_BODY_NAMES)。
TASK_BODY_NAMES = [
  "base_link",
  "waist_yaw_link",
  "l_arm_roll_link",
  "l_elbow_pitch_link",
  "l_hand_roll_link",
  "r_arm_roll_link",
  "r_elbow_pitch_link",
  "r_hand_roll_link",
  "l_hip_pitch_link",
  "l_hip_roll_link",
  "l_hip_yaw_link",
  "l_knee_pitch_link",
  "l_ankle_pitch_link",
  "l_ankle_roll_link",
  "r_hip_pitch_link",
  "r_hip_roll_link",
  "r_hip_yaw_link",
  "r_knee_pitch_link",
  "r_ankle_pitch_link",
  "r_ankle_roll_link",
  "head_yaw_link",
]

REQUIRED_KEYS = [
  "fps",
  "joint_names",
  "body_names",
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
]

ROBOT_TABLES = {
  "n3": {
    "joints": N3_JOINT_NAMES,
    "bodies": TASK_BODY_NAMES,
    "root_z_band": (0.55, 0.95),
  },
}


def _as_name_list(arr) -> list[str]:
  return [str(n) for n in np.asarray(arr).ravel()]


def validate_npz(path: str, robot: str) -> bool:
  tables = ROBOT_TABLES[robot]
  print(f"\n===== {path} =====")
  data = np.load(path, allow_pickle=False)

  errors: list[str] = []
  warnings: list[str] = []

  missing = [k for k in REQUIRED_KEYS if k not in data.files]
  if missing:
    print(f"[ERROR] 缺少必需键: {missing} (拥有: {data.files})")
    return False

  joint_names = _as_name_list(data["joint_names"])
  body_names = _as_name_list(data["body_names"])

  miss_j = [n for n in tables["joints"] if n not in joint_names]
  if miss_j:
    errors.append(f"joint_names 缺少关节: {miss_j}")
  if len(set(joint_names)) != len(joint_names):
    errors.append("joint_names 存在重复")

  miss_b = [n for n in tables["bodies"] if n not in body_names]
  if miss_b:
    errors.append(f"body_names 缺少 body: {miss_b}")
  if len(set(body_names)) != len(body_names):
    errors.append("body_names 存在重复")

  fps_raw = np.asarray(data["fps"]).ravel()
  fps = float(fps_raw[0]) if fps_raw.size else -1.0
  if fps != 50.0:
    errors.append(f"fps = {fps}, 任务要求 50 (控制率 50Hz 逐帧步进)")

  T = data["joint_pos"].shape[0]
  Nj, B = len(joint_names), len(body_names)
  if T < 2:
    errors.append(f"帧数 T = {T} < 2")
  if data["joint_pos"].shape != (T, Nj):
    errors.append(f"joint_pos 形状 {data['joint_pos'].shape} != ({T}, {Nj})")
  if data["joint_vel"].shape != (T, Nj):
    errors.append(f"joint_vel 形状 {data['joint_vel'].shape} != ({T}, {Nj})")
  for key, last in (
    ("body_pos_w", 3),
    ("body_lin_vel_w", 3),
    ("body_ang_vel_w", 3),
    ("body_quat_w", 4),
  ):
    shape = data[key].shape
    if len(shape) != 3 or shape != (T, B, last):
      errors.append(f"{key} 形状 {shape} != ({T}, {B}, {last})")

  for key in REQUIRED_KEYS[3:]:
    if not np.isfinite(data[key]).all():
      errors.append(f"{key} 含 NaN/Inf")

  quat_norm = np.linalg.norm(data["body_quat_w"], axis=-1)
  if quat_norm.max() > 1 + 1e-3 or quat_norm.min() < 1 - 1e-3:
    errors.append(
      f"body_quat_w 模长偏离 1: min={quat_norm.min():.4f}, max={quat_norm.max():.4f}"
    )

  # ---- 警告项 ----
  root_idx = 0
  try:
    root_idx = body_names.index("base_link")
  except ValueError:
    pass
  root_z = data["body_pos_w"][:, root_idx, 2]
  lo, hi = tables["root_z_band"]
  if not (lo <= root_z.mean() <= hi):
    warnings.append(
      f"根平均 z = {root_z.mean():.3f} 不在站立带 [{lo}, {hi}] (根帧/重定向高度可疑)"
    )

  for ankle in ("l_ankle_roll_link", "r_ankle_roll_link"):
    if ankle in body_names:
      min_z = data["body_pos_w"][:, body_names.index(ankle), 2].min()
      if min_z < -0.05:
        warnings.append(
          f"{ankle} 最低 z = {min_z:.3f} < -0.05 (脚底穿地, 转换时用 --root_z_offset 校正)"
        )

  dt = 1.0 / fps if fps > 0 else 0.02
  fd_vel = np.gradient(data["joint_pos"], dt, axis=0)
  med_fd = np.median(np.abs(fd_vel))
  med_vel = np.median(np.abs(data["joint_vel"]))
  if med_fd > 1e-6 and abs(med_vel - med_fd) > 3 * max(med_fd, 1e-6):
    warnings.append(
      f"joint_vel 中位幅值 {med_vel:.3f} 与位置差分 {med_fd:.3f} 差 > 3x (fps 或速度来源可疑)"
    )

  duration = (T - 1) * dt
  if duration < 10.0:
    warnings.append(f"时长 {duration:.1f}s < 10s (疑似合成占位/截断数据)")

  print(f"  T={T}, 关节={Nj}, body={B}, fps={fps}, 时长={duration:.1f}s")
  print(
    f"  joint_names 覆盖: {'OK' if not miss_j else 'MISS ' + str(miss_j)};"
    f" body_names 覆盖: {'OK' if not miss_b else 'MISS ' + str(miss_b)}"
  )
  for w in warnings:
    print(f"  [WARN] {w}")
  if errors:
    for e in errors:
      print(f"  [ERROR] {e}")
    print("  结果: 不通过 ✗")
    return False
  print("  结果: 通过 ✓")
  return True


def main():
  parser = argparse.ArgumentParser(
    description="Validate mimic npz against MotionLoader requirements."
  )
  parser.add_argument(
    "--npz", type=str, required=True, help="npz 文件或目录(逐个校验)。"
  )
  parser.add_argument(
    "--robot", type=str, default="n3", choices=list(ROBOT_TABLES.keys())
  )
  args = parser.parse_args()

  target = args.npz
  if os.path.isdir(target):
    files = sorted(
      os.path.join(target, f) for f in os.listdir(target) if f.endswith(".npz")
    )
    if not files:
      print(f"[ERROR] 目录 {target} 内没有 .npz 文件")
      sys.exit(1)
  else:
    files = [target]

  results = [validate_npz(f, args.robot) for f in files]
  n_pass = sum(results)
  print(f"\n===== 汇总: {n_pass}/{len(results)} 通过 =====")
  sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
  main()
