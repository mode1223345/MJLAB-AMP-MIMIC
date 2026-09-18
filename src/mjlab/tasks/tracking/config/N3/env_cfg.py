"""N3 (0905) 全身动作跟踪 (mimic, MHA+HIM) 任务配置。

移植 Isaac 端 ``mimic_noetix_n3_mha``（用户真机验证过的配置）。与 Isaac 的有意差异：
- 机器人资产用本地 ``asset_zoo/robots/N3``（N3.xml + constants.py，最新机器人数据）；
  Isaac mimic 端踝关节 kp 40/kd 5，本地 pitch 80/4、roll 50/2 按用户决定保留。
- ``physics_material`` 事件只随机摩擦（mjlab 无 geom restitution 域随机化，
  Isaac 的 restitution 0–0.5 未移植）。
- terminal obs 取 pre-step 近似（fork 既有约定，见 rl/vecenv_wrapper.py）。

观测布局（Isaac 同序，顺序即拼接顺序，测试钉死）：
actor/步 154 = command(58) ‖ anchor_ori_b(6) ‖ base_ang_vel(3) ‖ joint_pos(29)
‖ joint_vel(29) ‖ actions(29)，history 5 → 770，command 在每块前 58。
critic 349 = actor 同序无噪声 ‖ base_lin_vel(3) ‖ anchor_pos_b(3) ‖ body_pos(63)
‖ body_ori(126)。critic[154:157] 必须是 base_lin_vel（HIM 估计目标切片）。
"""

from __future__ import annotations

from mjlab.asset_zoo.robots.N3.constants import (
  N3_ACTION_SCALE,
  get_n3_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.amp.config.N3.env_cfg import N3_JOINT_VELOCITY_LIMITS
from mjlab.tasks.amp.mdp import randomize_joint_default_pos
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# reset 时根/基座速度扰动范围（push_robot 与 RSI 共用，Isaac 同款）。
VELOCITY_RANGE = {
  "x": (-0.1, 0.1),
  "y": (-0.1, 0.1),
  "z": (-0.05, 0.05),
  "roll": (-0.1, 0.1),
  "pitch": (-0.1, 0.1),
  "yaw": (-0.2, 0.2),
}

# 全身跟踪的 21 个 tracking body（奖励/终止/特权观测共用，Isaac 同名单）。
N3_MIMIC_BODY_NAMES = (
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
)

FEET_BODY_NAMES = ("l_ankle_roll_link", "r_ankle_roll_link")
LOWER_LEG_BODY_NAMES = (
  "l_knee_pitch_link",
  "r_knee_pitch_link",
  "l_ankle_roll_link",
  "r_ankle_roll_link",
)
LEG_JOINT_NAMES = (
  ".*_hip_pitch_joint",
  ".*_hip_roll_joint",
  ".*_hip_yaw_joint",
  ".*_knee_pitch_joint",
  ".*_ankle_pitch_joint",
  ".*_ankle_roll_joint",
)
ARM_JOINT_NAMES = (
  ".*_arm_pitch_joint",
  ".*_arm_roll_joint",
  ".*_arm_yaw_joint",
  ".*_elbow_pitch_joint",
)

# 动作库：32 个厂商高动态动作（文件名里的 30fps 是历史命名，数据实为 50fps=控制频率）。
# npz_baselink = 从原始 CSV 用本地 N3.xml 重新 FK 生成（anchor=base_link，与部署
# json 同源，csv_to_mimic_npz.py）；旧的 Isaac 再处理版 npz 已删。训练默认单动作
# （Isaac 端同款用法，每个动作单独训一个策略），换动作用
# --env.commands.motion.motion-file 覆盖；指向 N3_MIMIC_MOTION_DIR 目录则是
# 多片段混训（MotionLoader 逐文件按名字重排后拼接）。命令在仓库根目录执行。
N3_MIMIC_MOTION_DIR = "motions/mimic_data/N3/npz_baselink"
N3_MIMIC_MOTION_FILE = f"{N3_MIMIC_MOTION_DIR}/n3_侧空翻_30fps.npz"


def _n3_mimic_observations() -> dict[str, ObservationGroupCfg]:
  """Isaac 观测表：actor 带噪 6 项 + history 5，critic 无噪 10 项（无 history）。"""
  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
  }
  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=5,
      flatten_history_dim=False,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def _n3_mimic_events() -> dict[str, EventTermCfg]:
  """Isaac 事件表：startup 物理参数 + reset 增益 + interval 推扰。"""
  return {
    "physics_material": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
        "operation": "abs",
        "ranges": (0.3, 1.0),
        "shared_random": False,
      },
    ),
    "add_joint_default_pos": EventTermCfg(
      mode="startup",
      func=randomize_joint_default_pos,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "pos_distribution_params": (-0.02, 0.02),
        "operation": "add",
      },
    ),
    "add_base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "operation": "add",
        "ranges": (-2.0, 5.0),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.02, 0.02),
          2: (-0.05, 0.05),
        },
      },
    ),
    "randomize_actuator_gains": EventTermCfg(
      mode="reset",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.7, 1.2),
        "kd_range": (0.7, 1.2),
        "operation": "scale",
      },
    ),
    "randomize_ankle_armature": EventTermCfg(
      mode="reset",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_ankle.*joint",)),
        "ranges": (0.8, 1.2),
        "operation": "scale",
      },
    ),
    "randomize_ankle_friction": EventTermCfg(
      mode="reset",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_ankle.*joint",)),
        "ranges": (0.8, 1.2),
        "operation": "scale",
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(6.0, 10.0),
      params={"velocity_range": dict(VELOCITY_RANGE)},
    ),
  }


def _n3_mimic_rewards() -> dict[str, RewardTermCfg]:
  """Isaac 全奖励表（weight/std 逐项照抄，2026-09-14 转速包最终版）。"""
  return {
    "motion_anchor_height": RewardTermCfg(
      func=mdp.motion_anchor_height_error_exp,
      weight=2.5,
      params={"command_name": "motion", "std": 0.15},
    ),
    "motion_global_anchor_gravity": RewardTermCfg(
      func=mdp.motion_global_anchor_gravity_error_exp,
      weight=1.5,
      params={"command_name": "motion", "threshold": 0.5},
    ),
    "motion_global_anchor_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=1.8,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=3.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_feet_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=2.5,
      params={"command_name": "motion", "std": 0.15, "body_names": FEET_BODY_NAMES},
    ),
    "motion_swing_feet_height": RewardTermCfg(
      func=mdp.motion_swing_feet_height_error_exp,
      weight=0.2,
      params={
        "command_name": "motion",
        "std": 0.15,
        "contact_height": 0.08,
        "body_names": FEET_BODY_NAMES,
      },
    ),
    "motion_feet_z_under_lift": RewardTermCfg(
      func=mdp.motion_feet_z_under_lift_error_exp,
      weight=3.0,
      params={"command_name": "motion", "std": 0.05, "body_names": FEET_BODY_NAMES},
    ),
    "motion_anchor_to_support_foot": RewardTermCfg(
      func=mdp.motion_anchor_to_support_foot_error_exp,
      weight=1.0,
      params={
        "command_name": "motion",
        "std": 0.12,
        "ref_contact_z": 0.065,
        "foot_body_names": FEET_BODY_NAMES,
      },
    ),
    "motion_anchor_z_under_lift": RewardTermCfg(
      func=mdp.motion_anchor_z_under_lift_error_exp,
      weight=2.0,
      params={"command_name": "motion", "std": 0.05},
    ),
    "motion_lower_leg_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=2.5,
      params={
        "command_name": "motion",
        "std": 0.20,
        "body_names": LOWER_LEG_BODY_NAMES,
      },
    ),
    "motion_lower_leg_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.8,
      params={
        "command_name": "motion",
        "std": 0.35,
        "body_names": LOWER_LEG_BODY_NAMES,
      },
    ),
    "motion_feet_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.5,
      params={"command_name": "motion", "std": 0.30, "body_names": FEET_BODY_NAMES},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=2.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_root_vertical_velocity": RewardTermCfg(
      func=mdp.motion_root_vertical_velocity_error_exp,
      weight=1.2,
      params={"command_name": "motion", "std": 0.25},
    ),
    "motion_feet_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.5,
      params={
        "command_name": "motion",
        "std": 0.4,
        "body_names": FEET_BODY_NAMES,
      },
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.8,
      params={"command_name": "motion", "std": 1.2},
    ),
    "motion_joint_pos": RewardTermCfg(
      func=mdp.motion_joint_position_error_exp,
      weight=1.8,
      params={"command_name": "motion", "std": 0.30},
    ),
    "motion_leg_joint_pos": RewardTermCfg(
      func=mdp.motion_joint_position_error_exp,
      weight=1.5,
      params={
        "command_name": "motion",
        "std": 0.15,
        "joint_names": LEG_JOINT_NAMES,
      },
    ),
    "motion_arm_joint_pos": RewardTermCfg(
      func=mdp.motion_joint_position_error_exp,
      weight=1.0,
      params={
        "command_name": "motion",
        "std": 0.25,
        "joint_names": ARM_JOINT_NAMES,
      },
    ),
    "motion_joint_vel": RewardTermCfg(
      func=mdp.motion_joint_velocity_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 1.5},
    ),
    "motion_feet_air_contact": RewardTermCfg(
      func=mdp.motion_feet_air_contact_penalty,
      weight=-2.0,
      params={
        "command_name": "motion",
        "sensor_name": "feet_contact",
        "ref_air_z": 0.065,
        "ref_air_margin": 0.04,
        "contact_threshold": 5.0,
        "contact_force_margin": 20.0,
        "body_names": FEET_BODY_NAMES,
      },
    ),
    "motion_support_feet_slide": RewardTermCfg(
      func=mdp.motion_support_foot_slide_penalty,
      weight=-0.2,
      params={
        "command_name": "motion",
        "ref_contact_z": 0.065,
        "body_names": FEET_BODY_NAMES,
      },
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.feet_slide_contact_force,
      weight=-0.05,
      params={
        "sensor_name": "feet_contact",
        "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES),
      },
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts,
      weight=-0.5,
      params={"sensor_name": "self_collision", "threshold": 1.0},
    ),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(r"^(?!.*_ankle_.*joint$).*$",),
        )
      },
    ),
    "ankle_joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-0.5,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
        )
      },
    ),
    "joint_vel_limit": RewardTermCfg(
      func=mdp.joint_vel_limit_margin_penalty,
      weight=-10.0,
      params={
        "velocity_limits": N3_JOINT_VELOCITY_LIMITS,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ratio": 0.9,
      },
    ),
    "joint_effort_limit": RewardTermCfg(
      func=mdp.joint_effort_limit_margin_penalty,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)), "ratio": 0.9},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
  }


def _n3_mimic_terminations() -> dict[str, TerminationTermCfg]:
  """Isaac 终止表：全局锚点位置类禁用，ee 为 base 系 z-only 变体。"""
  return {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # 动作播完即截断（time_out=True 有 value bootstrap，防 5 帧历史跨相位污染）。
    "motion_end": TerminationTermCfg(
      func=mdp.motion_end,
      time_out=True,
      params={"command_name": "motion"},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "global_anchor_ori": TerminationTermCfg(
      func=mdp.bad_global_anchor_ori,
      params={"command_name": "motion", "threshold": 0.8},
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only_in_base,
      params={
        "command_name": "motion",
        "threshold": 0.50,
        "body_names": FEET_BODY_NAMES,
      },
    ),
  }


def n3_mimic_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the N3 mimic (MHA+HIM) tracking env config."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_n3_robot_cfg()}
  cfg.scene.num_envs = 4096
  cfg.scene.extent = 2.0
  # feet_contact: 双脚 subtree 对任意接触（Isaac 全体传感器语义，含翻转时脚-身接触）；
  # history 3 供 force-history 型奖励（feet_air/feet_slide）。
  feet_contact = ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(
      mode="subtree", pattern=r"^(l_ankle_roll_link|r_ankle_roll_link)$", entity="robot"
    ),
    secondary=None,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=3,
  )
  # self_collision: 非脚碰撞 geom（25 个 *_link_collision）对任意接触，
  # 供 undesired_contacts 计数（阈值 1N）。
  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="geom", pattern=r".*_link_collision", entity="robot"),
    secondary=None,
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_contact, self_collision)

  cfg.observations = _n3_mimic_observations()

  joint_pos_action = cfg.actions["joint_pos"]
  joint_pos_action.scale = N3_ACTION_SCALE

  cfg.commands = {
    "motion": MotionCommandCfg(
      entity_name="robot",
      motion_file=N3_MIMIC_MOTION_FILE,
      anchor_body_name="base_link",
      body_names=N3_MIMIC_BODY_NAMES,
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=False,
      pose_range={
        "x": (-0.02, 0.02),
        "y": (-0.02, 0.02),
        "z": (0.0, 0.01),
        "roll": (-0.03, 0.03),
        "pitch": (-0.03, 0.03),
        "yaw": (-0.05, 0.05),
      },
      velocity_range=dict(VELOCITY_RANGE),
      joint_position_range=(-0.01, 0.01),
      adaptive_kernel_size=5,
      adaptive_uniform_ratio=0.05,
      adaptive_alpha=0.005,
    )
  }

  cfg.events = _n3_mimic_events()
  cfg.rewards = _n3_mimic_rewards()
  cfg.terminations = _n3_mimic_terminations()

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 3.0

  cfg.sim.nconmax = 80
  cfg.sim.njmax = 400
  cfg.decimation = 4
  cfg.episode_length_s = 20.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
    motion_cmd.debug_vis = True

  return cfg
