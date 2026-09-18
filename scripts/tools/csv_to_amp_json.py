"""Convert N3 motion CSVs into mjlab AMP expert JSON files.

CSV columns (see N3-Motion/*.csv):
  root_pos(3), root_rot xyzw(4), 29 joint angles [rad] in deploy order.

Output JSON frame layout (163 dims, matches MotionLoader):
  root_pos(3) root_quat xyzw(4) joint_pos(29) key_pos(90, world axes rel. root)
  root_lin_vel(3, root body frame) root_ang_vel(3, root body frame)
  joint_vel(29) feet_contact(2)

Velocities are central differences of the resampled trajectories, rotated
into the root body frame (verified against the legacy Noetix converter).

Example:
  uv run python scripts/tools/csv_to_amp_json.py \
    --input N3-Motion --output-dir motions/amp_data/N3/json
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation, Slerp

import mjlab

# CSV joint column order (deploy chain order) -> mjlab joint names.
CSV_JOINT_NAMES: tuple[str, ...] = (
  "l_hip_pitch_joint",
  "l_hip_roll_joint",
  "l_hip_yaw_joint",
  "l_knee_pitch_joint",
  "l_ankle_pitch_joint",
  "l_ankle_roll_joint",
  "r_hip_pitch_joint",
  "r_hip_roll_joint",
  "r_hip_yaw_joint",
  "r_knee_pitch_joint",
  "r_ankle_pitch_joint",
  "r_ankle_roll_joint",
  "waist_yaw_joint",
  "l_arm_pitch_joint",
  "l_arm_roll_joint",
  "l_arm_yaw_joint",
  "l_elbow_pitch_joint",
  "l_hand_yaw_joint",
  "l_hand_roll_joint",
  "l_hand_pitch_joint",
  "r_arm_pitch_joint",
  "r_arm_roll_joint",
  "r_arm_yaw_joint",
  "r_elbow_pitch_joint",
  "r_hand_yaw_joint",
  "r_hand_roll_joint",
  "r_hand_pitch_joint",
  "head_pitch_joint",
  "head_yaw_joint",
)

# Key-body order (MJCF depth-first chain order, matches legacy JSONs).
KEY_POS_NAMES: tuple[str, ...] = (
  "base_link",
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
  "waist_yaw_link",
  "l_arm_pitch_link",
  "l_arm_roll_link",
  "l_arm_yaw_link",
  "l_elbow_pitch_link",
  "l_hand_yaw_link",
  "l_hand_roll_link",
  "l_hand_pitch_link",
  "r_arm_pitch_link",
  "r_arm_roll_link",
  "r_arm_yaw_link",
  "r_elbow_pitch_link",
  "r_hand_yaw_link",
  "r_hand_roll_link",
  "r_hand_pitch_link",
  "head_pitch_link",
  "head_yaw_link",
)

FOOT_GEOM_NAMES: tuple[str, ...] = ("l_foot_collision", "r_foot_collision")
# Legacy N3 model names the foot geoms differently.
FOOT_GEOM_NAMES_FALLBACK: tuple[str, ...] = (
  "l_ankle_roll_link_collision",
  "r_ankle_roll_link_collision",
)

_DEFAULT_MODEL = (
  Path(__file__).resolve().parents[2] / "src/mjlab/asset_zoo/robots/N3/xmls/N3.xml"
)


@dataclass
class CsvToAmpJsonArgs:
  input: str
  """CSV file or directory of CSVs to convert."""

  output_dir: str
  """Destination directory for the JSON files."""

  model: str = str(_DEFAULT_MODEL)
  """MJCF used for FK (key body positions, foot contacts)."""

  input_fps: float = 30.0
  output_fps: float = 50.0
  motion_weight: float = 0.5
  contact_tolerance: float = 0.01
  """Foot sole height below which a foot counts as in contact [m]."""

  exclude: list[str] = field(default_factory=list)
  """Filename substrings to skip (e.g. run_fast)."""


def _slerp_resample(
  quat_xyzw: np.ndarray, t_old: np.ndarray, t_new: np.ndarray
) -> np.ndarray:
  rot = Rotation.from_quat(quat_xyzw)
  slerp = Slerp(t_old, rot)
  return slerp(t_new).as_quat()


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  data = np.loadtxt(path, delimiter=",", skiprows=1)
  return data[:, 0:3], data[:, 3:7], data[:, 7:]


def convert_one(
  csv_path: Path,
  args: CsvToAmpJsonArgs,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_qpos_adrs: np.ndarray,
  body_ids: dict[str, int],
  foot_meshes: dict[str, tuple[np.ndarray, int]],
) -> dict[str, float]:
  pos, quat, joints = _load_csv(csv_path)

  # --- resample to output fps ---
  n_old = len(pos)
  t_old = np.arange(n_old) / args.input_fps
  dur = t_old[-1]
  t_new = np.arange(0, dur + 1e-9, 1.0 / args.output_fps)
  t_new = np.clip(t_new, 0.0, dur)
  pos_n = np.stack([np.interp(t_new, t_old, pos[:, k]) for k in range(3)], axis=1)
  joints_n = np.stack(
    [np.interp(t_new, t_old, joints[:, k]) for k in range(29)], axis=1
  )
  quat_n = _slerp_resample(quat, t_old, t_new)
  quat_n /= np.linalg.norm(quat_n, axis=1, keepdims=True)
  n = len(t_new)
  dt = 1.0 / args.output_fps

  # --- FK per frame ---
  key_pos = np.zeros((n, 90))
  contact = np.zeros((n, 2))
  qpos = np.zeros(model.nq)
  for i in range(n):
    qpos[0:3] = pos_n[i]
    qpos[3:7] = (quat_n[i][3], quat_n[i][0], quat_n[i][1], quat_n[i][2])
    qpos[joint_qpos_adrs] = joints_n[i]
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    root = data.xpos[body_ids["base_link"]].copy()
    for b, name in enumerate(KEY_POS_NAMES):
      key_pos[i, 3 * b : 3 * b + 3] = data.xpos[body_ids[name]] - root
    for f, gname in enumerate(foot_meshes):
      verts, gid = foot_meshes[gname]
      R = data.geom_xmat[gid].reshape(3, 3)
      p = data.geom_xpos[gid]
      sole_z = (verts @ R.T + p)[:, 2].min()
      contact[i, f] = 1.0 if sole_z < args.contact_tolerance else 0.0

  # --- velocities: central diff, rotated into the root body frame ---
  v_world = np.gradient(pos_n, dt, axis=0)
  R = Rotation.from_quat(quat_n).as_matrix()
  lin_vel = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), v_world)

  # omega_world = 2 * conj(q) * dq (vec part), central diff on components.
  dq = np.gradient(quat_n, dt, axis=0)
  w = np.zeros((n, 4))
  w[:, 0] = (
    dq[:, 3] * quat_n[:, 0]
    + dq[:, 0] * quat_n[:, 3]
    + dq[:, 1] * quat_n[:, 2]
    - dq[:, 2] * quat_n[:, 1]
  )
  w[:, 1] = (
    dq[:, 3] * quat_n[:, 1]
    - dq[:, 0] * quat_n[:, 2]
    + dq[:, 1] * quat_n[:, 3]
    + dq[:, 2] * quat_n[:, 0]
  )
  w[:, 2] = (
    dq[:, 3] * quat_n[:, 2]
    + dq[:, 0] * quat_n[:, 1]
    - dq[:, 1] * quat_n[:, 0]
    + dq[:, 2] * quat_n[:, 3]
  )
  ang_vel = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), 2.0 * w[:, :3])
  joint_vel = np.gradient(joints_n, dt, axis=0)

  frames = np.concatenate(
    [pos_n, quat_n, joints_n, key_pos, lin_vel, ang_vel, joint_vel, contact],
    axis=1,
  )
  assert frames.shape[1] == 163

  out = {
    "FrameDuration": 1.0 / args.output_fps,
    "JointNames": list(CSV_JOINT_NAMES),
    "KeyPosNames": list(KEY_POS_NAMES),
    "MotionWeight": args.motion_weight,
    "KeyPosFrame": "root",
    "RootLinearVelocityFrame": "root",
    "RootAngularVelocityFrame": "root",
    "ContactMaskMethod": "foot_support_surface_height",
    "ContactTolerance": args.contact_tolerance,
    "SourceCsv": csv_path.name,
    "SourceFps": args.input_fps,
    "SourceModel": args.model,
    "SourceModelSha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
    "Frames": frames.tolist(),
  }
  out_path = Path(args.output_dir)
  out_path.mkdir(parents=True, exist_ok=True)
  # Name the clip by its OUTPUT rate: n3_walk_30fps.csv -> n3_walk_50hz.json
  # (the source rate is recorded in the metadata).
  stem = re.sub(r"_\d+fps$", "", csv_path.stem)
  name = f"{stem}_{int(args.output_fps)}hz.json"
  with open(out_path / name, "w") as f:
    json.dump(out, f)

  vx_body = lin_vel[:, 0]
  return {
    "frames": n,
    "dur_s": dur,
    "vx_min": float(np.percentile(vx_body, 2)),
    "vx_max": float(np.percentile(vx_body, 98)),
    "contact_pct": float(contact.mean()),
  }


def main():
  args = tyro.cli(CsvToAmpJsonArgs, config=mjlab.TYRO_FLAGS)

  model = mujoco.MjModel.from_xml_path(args.model)
  data = mujoco.MjData(model)
  joint_qpos_adrs = np.array(
    [
      model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
      for name in CSV_JOINT_NAMES
    ]
  )
  body_ids = {
    name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    for name in KEY_POS_NAMES
  }
  assert all(v >= 0 for v in body_ids.values()), "missing body in MJCF"
  foot_meshes = {}
  if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FOOT_GEOM_NAMES[0]) < 0:
    foot_geom_names = FOOT_GEOM_NAMES_FALLBACK
  else:
    foot_geom_names = FOOT_GEOM_NAMES
  for gname in foot_geom_names:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
    assert gid >= 0, f"missing foot geom {gname}"
    mid = model.geom_dataid[gid]
    foot_meshes[gname] = (
      model.mesh_vert[
        model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]
      ].astype(np.float64),
      gid,
    )

  src = Path(args.input)
  csvs = sorted(src.glob("*.csv")) if src.is_dir() else [src]
  csvs = [c for c in csvs if not any(x in c.name for x in args.exclude)]
  print(f"Converting {len(csvs)} clip(s) with model {args.model}")
  for c in csvs:
    stats = convert_one(c, args, model, data, joint_qpos_adrs, body_ids, foot_meshes)
    print(
      f"  {c.stem:<18} {stats['frames']:>4} fr {stats['dur_s']:>5.1f}s "
      f"vx[{stats['vx_min']:+.2f},{stats['vx_max']:+.2f}] "
      f"contact {stats['contact_pct']:.0%}"
    )


if __name__ == "__main__":
  main()
