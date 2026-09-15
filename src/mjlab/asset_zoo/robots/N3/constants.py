"""Noetix N3 (0905) constants for mjlab AMP walk."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

N3_XML: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "N3" / "xmls" / "N3.xml"
assert N3_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(N3_XML))


##
# Joint names (MJCF depth-first under base_link).
##

N3_JOINT_NAMES: tuple[str, ...] = (
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
  "head_pitch_joint",
  "head_yaw_joint",
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
)

# Isaac Lab / Gazebo order from n3.py ``N3_29DOF_JOINT_NAMES``.
N3_DEPLOY_JOINT_NAMES: tuple[str, ...] = (
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
  "l_hand_pitch_joint",
  "l_hand_roll_joint",
  "r_arm_pitch_joint",
  "r_arm_roll_joint",
  "r_arm_yaw_joint",
  "r_elbow_pitch_joint",
  "r_hand_yaw_joint",
  "r_hand_pitch_joint",
  "r_hand_roll_joint",
  "head_pitch_joint",
  "head_yaw_joint",
)

##
# Actuator config (ported from Noetix n3.py N3_29DOF_CFG).
# delay_max_lag=4 matches DelayedImplicitActuatorCfg(min_delay=0, max_delay=4).
##

N3_ACTUATOR_LEGS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_pitch_joint",
  ),
  stiffness=80.0,
  damping=5.0,
  effort_limit=150.0,
  delay_min_lag=0,
  delay_max_lag=4,
)
N3_ACTUATOR_FEET = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=40.0,
  damping=5.0,
  effort_limit=50.0,
  armature=0.01,
  frictionloss=0.01,
  delay_min_lag=0,
  delay_max_lag=4,
)
N3_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=80.0,
  damping=3.0,
  effort_limit=150.0,
  delay_min_lag=0,
  delay_max_lag=4,
)
N3_ACTUATOR_ARMS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_arm_pitch_joint",
    ".*_arm_roll_joint",
    ".*_arm_yaw_joint",
    ".*_elbow_pitch_joint",
  ),
  stiffness=50.0,
  damping=3.0,
  effort_limit=50.0,
  armature=0.01,
  delay_min_lag=0,
  delay_max_lag=4,
)
N3_ACTUATOR_HANDS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hand_yaw_joint",
    ".*_hand_pitch_joint",
    ".*_hand_roll_joint",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=20.0,
  armature=0.01,
  delay_min_lag=0,
  delay_max_lag=4,
)
N3_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=("head_pitch_joint", "head_yaw_joint"),
  stiffness=20.0,
  damping=3.0,
  effort_limit=20.0,
  delay_min_lag=0,
  delay_max_lag=4,
)

##
# Keyframe / collision.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  # Home pose with bent legs and mesh foot collisions.
  pos=(0.0, 0.0, 0.724),
  joint_pos={
    "l_hip_pitch_joint": -0.1495,
    "r_hip_pitch_joint": -0.1495,
    "l_knee_pitch_joint": 0.3215,
    "r_knee_pitch_joint": 0.3215,
    "l_ankle_pitch_joint": -0.1720,
    "r_ankle_pitch_joint": -0.1720,
    "l_arm_roll_joint": 0.2,
    "r_arm_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

_FOOT_COLLISION = ".*_foot_collision.*"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(
    _FOOT_COLLISION,
    "base_collision",
  ),
  contype=1,
  conaffinity=1,
  condim={
    _FOOT_COLLISION: 3,
    "base_collision": 1,
  },
  priority={_FOOT_COLLISION: 1, ".*": 0},
  friction={_FOOT_COLLISION: (1.0,), ".*": (0.5,)},
  solref=(0.002, 1.0),
  solimp=(0.99, 0.999, 0.00001),
)

N3_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    N3_ACTUATOR_LEGS,
    N3_ACTUATOR_FEET,
    N3_ACTUATOR_WAIST,
    N3_ACTUATOR_ARMS,
    N3_ACTUATOR_HANDS,
    N3_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_n3_robot_cfg() -> EntityCfg:
  """Fresh N3 (0905) robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=N3_ARTICULATION,
  )


N3_ACTION_SCALE: dict[str, float] = {}
for a in N3_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    N3_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_n3_robot_cfg())
  viewer.launch(robot.spec.compile())
