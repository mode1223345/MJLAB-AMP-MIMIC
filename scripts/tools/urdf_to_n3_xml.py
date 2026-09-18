"""Generate the N3 MJCF from the authoritative URDF.

Kinematics, inertials, and joint limits come from ``urdf/N3.urdf`` (54.968 kg
total). Collision layout per user decision (2026-09-17):

- Feet: mesh collision (``*_foot_collision`` uses the ankle-roll STL), with
  the contact parameters proven on the colleague's n3_0905 mesh feet
  (condim 3, friction 1.0/0.02/0.005, solref 0.002 1, solimp 0.99 0.999 1e-5).
- Every other body: slim capsules (class ``n3_collision``) left at MuJoCo
  defaults for solver params so lying-on-ground contacts stay soft. Capsule
  radii are deliberately under-sized vs the visual meshes to avoid
  self-penetration across expert poses (the failure mode of the previous
  full-collision attempt); ``--scan`` verifies against the expert JSONs.

Body/joint ordering is pinned to the legacy XML (legs first, then the waist
subtree) because ``body_names`` feeds the AMP expert ``KeyPosNames`` and
joint order feeds ``N3_JOINT_NAMES``; changing it invalidates the 9 expert
JSONs and the discriminator mask dims.

Usage:
  uv run python scripts/tools/urdf_to_n3_xml.py           # write + validate
  uv run python scripts/tools/urdf_to_n3_xml.py --scan    # also run the
      expert-frame self-penetration scan (motions/amp_data/N3/json)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_DIR = REPO_ROOT / "src/mjlab/asset_zoo/robots/N3"
URDF_PATH = ROBOT_DIR / "urdf/N3.urdf"
XML_PATH = ROBOT_DIR / "xmls/N3.xml"
MOTIONS_DIR = REPO_ROOT / "motions/amp_data/N3/json"

# Depth-first body order of the legacy XML. MUST stay stable: it defines
# body_names (= AMP expert KeyPosNames order) and, via joints, N3_JOINT_NAMES.
CHILD_ORDER = {
  "base_link": (
    "l_hip_pitch_joint",
    "r_hip_pitch_joint",
    "waist_yaw_joint",
  ),
  "waist_yaw_link": (
    "head_pitch_joint",
    "l_arm_pitch_joint",
    "r_arm_pitch_joint",
  ),
}

# Body-frame capsules: body name -> list of (radius, from, to).
# Sized from mesh AABBs (see scripts/tools printout) minus a slim margin, so
# mocap arm-swing and hanging-arm poses do not self-intersect.
CAPSULES: dict[
  str, list[tuple[float, tuple[float, float, float], tuple[float, float, float]]]
] = {
  # Pelvis: horizontal capsule across the hips, slightly under mesh (y +-0.102).
  "base_link": [(0.055, (0, -0.06, 0.09), (0, 0.06, 0.09))],
  # Hip pitch bracket reaching out to the roll joint (mesh y 0.001..0.106).
  "l_hip_pitch_link": [(0.03, (0, 0.01, -0.02), (0, 0.05, -0.09))],
  "r_hip_pitch_link": [(0.03, (0, -0.01, -0.02), (0, -0.05, -0.09))],
  # Hip roll bracket down to the yaw joint (mesh z -0.143..0.049).
  "l_hip_roll_link": [(0.045, (0, 0, -0.01), (0, 0, -0.13))],
  "r_hip_roll_link": [(0.045, (0, 0, -0.01), (0, 0, -0.13))],
  # Thigh proper, follows hip yaw (knee joint at z -0.1631).
  "l_hip_yaw_link": [(0.05, (0, 0, -0.01), (0, 0, -0.16))],
  "r_hip_yaw_link": [(0.05, (0, 0, -0.01), (0, 0, -0.16))],
  # Shin (ankle joints at z -0.34); slim so hands can pass in mocap poses.
  "l_knee_pitch_link": [(0.042, (0, 0, -0.02), (0, 0, -0.32))],
  "r_knee_pitch_link": [(0.042, (0, 0, -0.02), (0, 0, -0.32))],
  # Torso: three horizontal capsules; biased 1 cm back for back-fall contact.
  # Mesh: x -0.118..0.097, y +-0.147, z -0.042..0.404 (URDF box 0.18x0.25x0.30
  # at z 0.15).
  "waist_yaw_link": [
    (0.078, (-0.01, -0.065, 0.05), (-0.01, 0.065, 0.05)),
    (0.08, (-0.01, -0.065, 0.15), (-0.01, 0.065, 0.15)),
    (0.075, (-0.01, -0.06, 0.25), (-0.01, 0.06, 0.25)),
  ],
  # Head shell only; the rear neck yoke running down the back stays collision-
  # free on purpose (rotates with head pitch, would sweep the torso).
  "head_yaw_link": [(0.05, (0.02, 0, 0.0), (0.04, 0, 0.09))],
  # Shoulder bracket (mesh y 0.015..0.108).
  "l_arm_pitch_link": [(0.032, (0, 0.03, 0), (0, 0.08, 0))],
  "r_arm_pitch_link": [(0.032, (0, -0.03, 0), (0, -0.08, 0))],
  # Upper arm (elbow chain continues at z -0.2033).
  "l_arm_roll_link": [(0.04, (0, 0, 0.02), (0, 0, -0.19))],
  "r_arm_roll_link": [(0.04, (0, 0, 0.02), (0, 0, -0.19))],
  # Elbow bracket of the upper arm.
  "l_arm_yaw_link": [(0.036, (0, 0, -0.02), (0, 0, -0.06))],
  "r_arm_yaw_link": [(0.036, (0, 0, -0.02), (0, 0, -0.06))],
  # Forearm (hand chain at z -0.17).
  "l_elbow_pitch_link": [(0.035, (0, 0, 0.02), (0, 0, -0.16))],
  "r_elbow_pitch_link": [(0.035, (0, 0, 0.02), (0, 0, -0.16))],
  # Wrist bracket.
  "l_hand_yaw_link": [(0.03, (0, 0, -0.02), (0, 0, -0.05))],
  "r_hand_yaw_link": [(0.03, (0, 0, -0.02), (0, 0, -0.05))],
  # Palm (fingers reach z -0.132).
  "l_hand_pitch_link": [(0.032, (0, 0, -0.01), (0, 0, -0.11))],
  "r_hand_pitch_link": [(0.032, (0, 0, -0.01), (0, 0, -0.11))],
}

# Parent-child and near-neighbor pairs that must not collide. Adjacent pairs
# because their capsules overlap by construction; cross pairs because the
# home pose / expert poses bring them within capsule reach.
EXCLUDES: tuple[tuple[str, str], ...] = (
  ("base_link", "waist_yaw_link"),
  ("base_link", "l_hip_pitch_link"),
  ("base_link", "r_hip_pitch_link"),
  ("base_link", "l_hip_roll_link"),
  ("base_link", "r_hip_roll_link"),
  ("waist_yaw_link", "l_hip_pitch_link"),
  ("waist_yaw_link", "r_hip_pitch_link"),
  ("waist_yaw_link", "head_yaw_link"),
  ("l_hip_pitch_link", "l_hip_roll_link"),
  ("l_hip_roll_link", "l_knee_pitch_link"),
  ("l_knee_pitch_link", "l_ankle_roll_link"),
  ("r_hip_pitch_link", "r_hip_roll_link"),
  ("r_hip_roll_link", "r_knee_pitch_link"),
  ("r_knee_pitch_link", "r_ankle_roll_link"),
  ("waist_yaw_link", "l_arm_roll_link"),
  ("l_arm_pitch_link", "l_arm_roll_link"),
  ("l_arm_roll_link", "l_arm_yaw_link"),
  ("l_arm_yaw_link", "l_elbow_pitch_link"),
  # Upper-arm and forearm capsules overlap across the hinge at full extension.
  ("l_arm_roll_link", "l_elbow_pitch_link"),
  ("l_elbow_pitch_link", "l_hand_yaw_link"),
  ("l_hand_yaw_link", "l_hand_roll_link"),
  ("l_hand_yaw_link", "l_hand_pitch_link"),
  ("l_elbow_pitch_link", "l_hand_pitch_link"),
  ("waist_yaw_link", "r_arm_roll_link"),
  ("r_arm_pitch_link", "r_arm_roll_link"),
  ("r_arm_roll_link", "r_arm_yaw_link"),
  ("r_arm_yaw_link", "r_elbow_pitch_link"),
  ("r_arm_roll_link", "r_elbow_pitch_link"),
  ("r_elbow_pitch_link", "r_hand_yaw_link"),
  ("r_hand_yaw_link", "r_hand_roll_link"),
  ("r_hand_yaw_link", "r_hand_pitch_link"),
  ("r_elbow_pitch_link", "r_hand_pitch_link"),
  # Thigh capsule (hip_yaw) meets the shin capsule (knee) across the hinge.
  ("l_hip_yaw_link", "l_knee_pitch_link"),
  ("r_hip_yaw_link", "r_knee_pitch_link"),
)


def rpy_to_quat(rpy: tuple[float, ...]) -> str:
  """URDF xyz-roll-pitch-yaw (fixed-axis RPY) -> MJCF wxyz quaternion."""
  cr, cp, cy = (math.cos(v / 2) for v in rpy)
  sr, sp, sy = (math.sin(v / 2) for v in rpy)
  w = cr * cp * cy + sr * sp * sy
  x = sr * cp * cy - cr * sp * sy
  y = cr * sp * cy + sr * cp * sy
  z = cr * cp * sy - sr * sp * cy
  return f"{w:.6g} {x:.6g} {y:.6g} {z:.6g}"


def fmt(values) -> str:
  return " ".join(f"{v:.6g}" for v in values)


def attr(element: ET.Element, name: str) -> str:
  """Attribute text with None ruled out (the URDF schema guarantees it)."""
  value = element.get(name)
  assert value is not None, f"<{element.tag}> missing @{name}"
  return value


def sub(element: ET.Element, tag: str) -> ET.Element:
  """First child element with None ruled out."""
  found = element.find(tag)
  assert found is not None, f"<{element.tag}> missing <{tag}>"
  return found


def floats(text: str) -> tuple[float, ...]:
  return tuple(float(v) for v in text.split())


class Urdf:
  def __init__(self, path: Path):
    root = ET.parse(path).getroot()
    self.links = {attr(lk, "name"): lk for lk in root.findall("link")}
    self.joints = {attr(j, "name"): j for j in root.findall("joint")}
    self.child_joint: dict[str, str] = {}  # link -> joint connecting to parent
    self.children: dict[str, list[str]] = {}
    for name, j in self.joints.items():
      parent = attr(sub(j, "parent"), "link")
      child = attr(sub(j, "child"), "link")
      self.child_joint[child] = name
      self.children.setdefault(parent, []).append(name)

  def order_children(self, link: str) -> list[str]:
    order = CHILD_ORDER.get(link)
    if order is not None:
      missing = set(self.children[link]) - set(order)
      assert not missing, f"CHILD_ORDER incomplete for {link}: {missing}"
      return list(order)
    return list(self.children.get(link, []))


def emit(urdf: Urdf) -> str:
  lines: list[str] = []
  w = lines.append

  w('<mujoco model="n3">')
  w("  <!-- Generated by scripts/tools/urdf_to_n3_xml.py from urdf/N3.urdf.")
  w("       Kinematics/inertials/limits are URDF-faithful (54.968 kg). Feet")
  w("       collide as meshes (colleague-proven contact params); other bodies")
  w("       are slim capsules for fall recovery. Body order is load-bearing:")
  w("       it feeds body_names (AMP KeyPosNames) and N3_JOINT_NAMES. -->")
  w('  <compiler angle="radian" meshdir="../meshes"/>')
  w("  <default>")
  w('    <joint damping="0.001" armature="0.01" frictionloss="0.1"/>')
  w('    <default class="n3_collision">')
  w(
    '      <geom group="3" rgba=".2 .6 .2 .3" type="capsule" contype="1" conaffinity="1"/>'
  )
  w("    </default>")
  w('    <default class="foot_collision">')
  w('      <geom type="mesh" group="3" contype="1" conaffinity="1" condim="3"')
  w(
    '            friction="1.0 0.02 0.005" solref="0.002 1" solimp="0.99 0.999 0.00001"'
  )
  w('            rgba="0.1 0.7 0.1 0.35"/>')
  w("    </default>")
  w("  </default>")
  w("")
  w("  <asset>")
  for link in sorted(urdf.links):
    w(f'    <mesh name="{link}" file="{link}.STL"/>')
  w("  </asset>")
  w("")
  w("  <worldbody>")

  def emit_body(link: str, depth: int) -> None:
    pad = "  " * (depth + 2)
    element = urdf.links[link]
    if link == "base_link":
      w(f'{pad}<body name="{link}">')
    else:
      joint = urdf.joints[urdf.child_joint[link]]
      origin = joint.find("origin")
      xyz = floats(attr(origin, "xyz")) if origin is not None else (0.0, 0.0, 0.0)
      rpy = floats(attr(origin, "rpy")) if origin is not None else (0.0, 0.0, 0.0)
      attrs = f'name="{link}" pos="{fmt(xyz)}"'
      quat = rpy_to_quat(rpy)
      if quat.split()[0] != "1" or any(float(v) != 0 for v in quat.split()[1:]):
        attrs += f' quat="{quat}"'
      w(f"{pad}<body {attrs}>")
      axis = attr(sub(joint, "axis"), "xyz")
      limit = sub(joint, "limit")
      jattrs = f'name="{urdf.child_joint[link]}" pos="0 0 0" axis="{axis}"'
      jattrs += (
        f' range="{float(attr(limit, "lower")):.6g} {float(attr(limit, "upper")):.6g}"'
      )
      w(f"{pad}  <joint {jattrs}/>")

    inertial = sub(element, "inertial")
    origin = inertial.find("origin")
    ipos = attr(origin, "xyz") if origin is not None else "0 0 0"
    i = sub(inertial, "inertia")
    full = [attr(i, k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
    mass = attr(sub(inertial, "mass"), "value")
    w(
      f'{pad}  <inertial pos="{ipos}" mass="{mass}" fullinertia="{fmt(float(v) for v in full)}"/>'
    )

    if link == "base_link":
      w(f'{pad}  <joint name="root" type="free"/>')

    w(
      f'{pad}  <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" mesh="{link}"/>'
    )

    if link.endswith("ankle_roll_link"):
      side = link[0]
      w(
        f'{pad}  <geom name="{side}_foot_collision" class="foot_collision" mesh="{link}"/>'
      )
    else:
      for index, (radius, start, end) in enumerate(CAPSULES.get(link, []), start=1):
        name = f"{link}_collision" if index == 1 else f"{link}_collision{index}"
        w(
          f'{pad}  <geom name="{name}" class="n3_collision" '
          f'size="{radius}" fromto="{fmt(start)} {fmt(end)}"/>'
        )

    for joint_name in urdf.order_children(link):
      child = attr(sub(urdf.joints[joint_name], "child"), "link")
      emit_body(child, depth + 1)
    w(f"{pad}</body>")

  emit_body("base_link", 0)
  w("  </worldbody>")
  w("  <contact>")
  for a, b in EXCLUDES:
    w(f'    <exclude body1="{a}" body2="{b}"/>')
  w("  </contact>")
  w("</mujoco>")
  return "\n".join(lines) + "\n"


def validate(xml_text: str, write_path: Path | None) -> mujoco.MjModel:
  """Compile and check order/mass; write the file only if all checks pass."""
  # Temp must sit in xmls/ so meshdir="../meshes" resolves during compile.
  tmp = (write_path or XML_PATH).with_suffix(".tmp.xml")
  tmp.write_text(xml_text)
  try:
    model = mujoco.MjModel.from_xml_path(str(tmp))
  except Exception as ex:
    print(f"COMPILE FAILED: {ex}")
    sys.exit(1)

  joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    for j in range(model.njnt)
    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
  ]
  body_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, model.nbody)
  ]
  from mjlab.asset_zoo.robots.N3.constants import N3_JOINT_NAMES  # noqa: E402

  assert tuple(joint_names) == N3_JOINT_NAMES, (
    f"joint order mismatch:\n  got      {joint_names}\n  expected {list(N3_JOINT_NAMES)}"
  )
  assert len(body_names) == 30, f"expected 30 bodies, got {len(body_names)}"

  if MOTIONS_DIR.is_dir():
    # The motion loader maps key positions by name (create_index_mapping), so
    # only the name SET must match; order may differ between JSON and XML.
    sample = sorted(MOTIONS_DIR.glob("*.json"))[0]
    key_names = json.loads(sample.read_text())["KeyPosNames"]
    assert set(key_names) == set(body_names), (
      f"body name set no longer matches expert {sample.name} KeyPosNames: "
      f"{set(key_names) ^ set(body_names)}"
    )

  total_mass = float(model.body_mass.sum())
  print(f"nbody {model.nbody} nq {model.nq} total mass {total_mass:.3f} kg")
  print(
    f"foot geoms: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom) if 'foot_collision' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '')]}"
  )
  assert abs(total_mass - 54.968) < 0.01, f"mass drifted: {total_mass}"

  if write_path is not None:
    tmp.replace(write_path)
    print(f"wrote {write_path}")
  return model


def scan_expert_frames(model: mujoco.MjModel) -> None:
  """Self-penetration check of the collision set over expert frames."""
  data = mujoco.MjData(model)
  excluded = set()
  for a, b in EXCLUDES:
    b1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, a)
    b2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
    assert b1 >= 0 and b2 >= 0, f"exclude references missing body {a}/{b}"
    excluded.add(frozenset((b1, b2)))
  collision_geoms = np.flatnonzero(model.geom_contype | model.geom_conaffinity)
  joint_adr = np.array(
    [
      model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
      for name in (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
      )
    ]
  )
  files = sorted(MOTIONS_DIR.glob("*.json"))
  files += sorted((MOTIONS_DIR.parent / "recovery").glob("*.json"))
  n_bad = 0
  for path in files:
    payload = json.loads(path.read_text())
    sim_idx = [
      payload["JointNames"].index(n)
      for n in (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
      )
    ]
    worst: dict[frozenset, float] = {}
    for frame in payload["Frames"]:
      qpos = np.zeros(model.nq)
      qpos[0:3] = frame[0:3]
      qpos[3:7] = (frame[6], frame[3], frame[4], frame[5])
      qpos[joint_adr] = np.array(frame[7:36])[sim_idx]
      data.qpos[:] = qpos
      mujoco.mj_forward(model, data)
      for c in range(data.ncon):
        g1, g2 = data.contact.geom1[c], data.contact.geom2[c]
        if g1 not in collision_geoms and g2 not in collision_geoms:
          continue
        if frozenset((model.geom_bodyid[g1], model.geom_bodyid[g2])) in excluded:
          continue
        pair = frozenset(
          (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g1]),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g2]),
          )
        )
        worst[pair] = min(worst.get(pair, 0.0), float(data.contact.dist[c]))
    if worst:
      n_bad += 1
      detail = ", ".join(
        f"{'<->'.join(sorted(p))} {d * 1000:.1f}mm"
        for p, d in sorted(worst.items(), key=lambda kv: kv[1])[:6]
      )
      print(f"  {path.name}: {detail}")
  print(
    f"scan: {n_bad}/{len(files)} files have non-excluded contacts (see pairs above)"
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--scan", action="store_true", help="run expert-frame penetration scan"
  )
  parser.add_argument("--dry-run", action="store_true", help="validate without writing")
  args = parser.parse_args()

  urdf = Urdf(URDF_PATH)
  xml_text = emit(urdf)
  model = validate(xml_text, None if args.dry_run else XML_PATH)
  if args.scan:
    scan_expert_frames(model)


if __name__ == "__main__":
  main()
