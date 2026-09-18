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
# delay_max_lag=6 matches DelayedImplicitActuatorCfg(min_delay=0, max_delay=6)
# from the Isaac Lab N3 config: 0-6 control steps at 50 Hz.
#
# 踝 pitch/roll 拆成两组（80/4 与 50/2，源自 noetix_mjlab 的 n3_0905，对齐其
# URDF/XML 更新）。注意 N3_ACTION_SCALE 由 effort/stiffness 推出，所以这会把踝
# action scale 从 0.3125 改成 pitch 0.15625 / roll 0.25（每组满动作的力矩上限仍是
# 0.25*effort = 12.5 N·m，变的是"每个单位动作能掰多远"）；旧 checkpoint 的踝动作
# 含义随之改变。
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
  delay_max_lag=6,
)
N3_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=80.0,
  damping=4.0,
  effort_limit=50.0,
  armature=0.01,
  frictionloss=0.01,
  delay_min_lag=0,
  delay_max_lag=6,
)
N3_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=50.0,
  damping=2.0,
  effort_limit=50.0,
  armature=0.01,
  frictionloss=0.01,
  delay_min_lag=0,
  delay_max_lag=6,
)
N3_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=80.0,
  damping=3.0,
  effort_limit=150.0,
  delay_min_lag=0,
  delay_max_lag=6,
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
  delay_max_lag=6,
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
  delay_max_lag=6,
)
N3_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=("head_pitch_joint", "head_yaw_joint"),
  stiffness=20.0,
  damping=3.0,
  effort_limit=20.0,
  delay_min_lag=0,
  delay_max_lag=6,
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

# 脚碰撞：mesh（l/r_foot_collision，见 xmls/N3.xml 的 foot_collision 类，接触参数
# 沿用 noetix n3_0905 验证过的组合，写在 XML 类里，这里不覆写）。其余体节：瘦身胶囊
# （见 N3.xml）， MuJoCo 默认解算参数（软接触，躺地不起跳）。
_FOOT_COLLISION = ".*_foot_collision"

# 全套碰撞（mesh 脚 + 全身胶囊 + 骨盆胶囊），为摔倒/起身 recovery 服务。胶囊半径
# 全部小于视觉 mesh 包络，且 20+ 条 <exclude> 排掉关节相邻对；urdf_to_n3_xml.py
# --scan 对 9 个走跑片段 + 起身片段逐帧验证 0 自穿透（旧胶囊方案是 617/2428 帧穿透、
# 最深 −83.7 mm，起跳弹开的根源）。
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={_FOOT_COLLISION: 3, ".*": 1},
  priority={_FOOT_COLLISION: 1, ".*": 0},
  friction={_FOOT_COLLISION: (1.0, 0.02, 0.005)},
)

N3_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    N3_ACTUATOR_LEGS,
    N3_ACTUATOR_ANKLE_PITCH,
    N3_ACTUATOR_ANKLE_ROLL,
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
