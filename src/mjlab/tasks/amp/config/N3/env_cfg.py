"""N3 (0905) flat-terrain AMP locomotion env config.

Robot asset: ``asset_zoo/robots/N3`` (29 DoF). Assembly lives in
:mod:`mjlab.tasks.amp.amp_env_cfg`; this module only declares the N3 spec
and its reward set.

2026-09-18 起对齐 Isaac 端 ``noetix_n3_29dof_full_amp2``（用户真机验证过的
配置）。与 Isaac 的差异仅剩 mjlab 侧的 recovery 机制：30% delay env 的
``delay_masked`` 包装、``track_root_height`` 起身信号、``is_terminated``
重惩罚（由 factory 追加，不在此文件）。
"""

from __future__ import annotations

import re

from mjlab.asset_zoo.robots.N3.constants import (
  N3_ACTION_SCALE,
  N3_JOINT_NAMES,
  get_n3_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.amp import mdp
from mjlab.tasks.amp.amp_env_cfg import (
  AmpCommandSpec,
  AmpRobotSpec,
  make_amp_flat_env_cfg,
)

# 电机额定转速 [rad/s]（Isaac N3 配置：8522→18.58、5014→13.85、4308→16.02、
# 臂统一 10.0）。同时用于 sim 内逐控制步速度钳制和 90% 余量惩罚。
_N3_JOINT_VELOCITY_LIMITS_SPEC: dict[tuple[str, ...], float] = {
  (
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_pitch_joint",
    "waist_yaw_joint",
  ): 18.58,
  (".*_ankle_pitch_joint", ".*_ankle_roll_joint"): 13.85,
  (
    ".*_arm_pitch_joint",
    ".*_arm_roll_joint",
    ".*_arm_yaw_joint",
    ".*_elbow_pitch_joint",
    ".*_hand_yaw_joint",
    ".*_hand_roll_joint",
    ".*_hand_pitch_joint",
  ): 10.0,
  ("head_pitch_joint", "head_yaw_joint"): 16.02,
}

N3_JOINT_VELOCITY_LIMITS: dict[str, float] = {
  name: limit
  for patterns, limit in _N3_JOINT_VELOCITY_LIMITS_SPEC.items()
  for name in N3_JOINT_NAMES
  if any(re.fullmatch(p, name) for p in patterns)
}


def n3_rewards() -> dict[str, RewardTermCfg]:
  """N3 AMP task rewards（对齐 Isaac noetix_n3_29dof_full_amp2 权重/std）。

  每步大小 = weight × raw × 0.02（RewardManager 的 dt 缩放）；AMP style 不参与
  该缩放。调权重先看 TB 的 ``Episode_Reward/<项>``（= weight × raw）。
  ``delay_masked`` 项对 30% 的 delay（recovery）环境恒为 0。
  """
  foot_asset = SceneEntityCfg(
    "robot",
    body_names=("l_ankle_roll_link", "r_ankle_roll_link"),
    preserve_order=True,
  )
  # fmt: off
  return {
    # 线速度跟踪（yaw 系）：exp(−‖v_cmd−v‖²/std²)，std=0.4 → 误差 0.4 m/s 时 0.37。
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=mdp.delay_masked(mdp.track_lin_vel_xy_yaw_frame_exp),
      weight=5.0,
      params={"std": 0.4, "command_name": "twist"},
    ),

    # 角速度跟踪（世界系 wz）：exp(−Δωz²/std²)，std=0.4。全指令生效：零 wz 指令
    # 下等价于抑制乱转。
    "track_ang_vel_z_exp": RewardTermCfg(
      func=mdp.delay_masked(mdp.track_ang_vel_z_world_exp),
      weight=3.0,
      params={"std": 0.4, "command_name": "twist"},
    ),

    # 前向速度带：带符号比值 vx/cmd_x ∈ [0.5, 1.2] 给 +1.2（反向必落低速支给
    # −1.0），过快给 0。死区 |cmd_x| ≤ 0.3（min_cmd_vel）内不激活。
    "low_speed": RewardTermCfg(
      func=mdp.delay_masked(mdp.low_speed),
      weight=1.2,
      params={
        "min_cmd_vel": 0.3,
        "low_speed_threshold": 0.5,
        "high_speed_threshold": 1.2,
        "command_name": "twist",
      },
    ),

    # 纯旋转指令（|cmd_x|<1e-6 且 |cmd_z|>1e-6）：wz 跟踪 exp(−Δωz²/std²)
    # 减 2.0×‖v_xy‖_w（原地转时不许平移）。
    "track_ang_vel_stand_exp": RewardTermCfg(
      func=mdp.delay_masked(mdp.track_ang_vel_stand_world_exp),
      weight=2.0,
      params={"std": 0.4, "lin_vel_threshold": 2.0, "command_name": "twist"},
    ),

    # 躯干直立：exp(−‖g_xy‖/std²)，std=0.4（倾 0.4 rad 时 0.37）。
    "flat_orientation_exp": RewardTermCfg(
      func=mdp.flat_orientation_exp, weight=1.0, params={"std": 0.4}
    ),

    # 质心支撑：waist COM 相对两踝中点（yaw 系）的 x 偏差 exp(−e²/std²)，
    # std=0.2 m（偏 0.2 m 掉 63%）。站立门控 ‖cmd_xy‖ < 0.05。
    "waist_feet_support_x": RewardTermCfg(
      func=mdp.delay_masked(mdp.waist_com_feet_support_x_error_exp),
      weight=1.5,
      params={"std": 0.2, "target_x": 0.0, "asset_cfg": foot_asset},
    ),

    # 双支撑上限：双支撑持续 ≥0.2 s 归零，其余（单脚触地/飞行/双支撑早期）+1。
    "feet_contact": RewardTermCfg(
      func=mdp.delay_masked(mdp.feet_contact),
      weight=1.5,
      params={"sensor_name": "feet_ground_contact"},
    ),

    # 脚底打滑：触地脚水平速度之和 [m/s]。
    "feet_slide": RewardTermCfg(
      func=mdp.delay_masked(mdp.feet_slide),
      weight=-2.5,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*_ankle_roll_link",)),
      },
    ),

    # 单支撑相时长：min(两脚当前相时长) 截断于 0.25 s。门控 |cmd_xy|+|wz| > 0.1。
    "feet_air_time": RewardTermCfg(
      func=mdp.delay_masked(mdp.feet_air_time_positive_biped),
      weight=0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold": 0.25,
        "command_name": "twist",
      },
    ),

    # 足底冲击：两脚 z 向接触力 L2 超过 threshold=580 N 按超出量罚，上限 800 N。
    "feet_force": RewardTermCfg(
      func=mdp.delay_masked(mdp.feet_force),
      weight=-1e-2,
      params={"sensor_name": "feet_ground_contact", "threshold": 580,
              "max_reward": 800},
    ),

    # 触地脚内/外翻：受力脚（净力 >1 N）|roll| 超 0.08 rad（4.6°）按超额线性罚。
    "feet_contact_roll": RewardTermCfg(
      func=mdp.delay_masked(mdp.feet_contact_roll_penalty),
      weight=-1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*_ankle_roll_link",)),
        "threshold": 0.08,
        "force_threshold": 1.0,
      },
    ),

    # 髋 yaw/roll + 踝 roll 偏离默认位形 L1。生效 |vy|、|wz| ≤ 0.1 且指令非零。
    "joint_deviation_leg": RewardTermCfg(
      func=mdp.delay_masked(mdp.joint_deviation_l1_with_zero_flag),
      weight=-1.2,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_ankle_roll_joint",
          ),
        )
      },
    ),

    # 臂 roll 偏离默认位形 L1。全指令生效。
    "joint_deviation_upbody": RewardTermCfg(
      func=mdp.delay_masked(mdp.joint_deviation_l1),
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot",
                                          joint_names=(".*_arm_roll_joint",))},
    ),

    # 站立时上肢（臂 pitch/roll + 肘 pitch）偏离默认位形 L1。
    # 门控 ‖cmd_xy‖+|wz| < 0.1。
    "joint_deviation_standing": RewardTermCfg(
      func=mdp.delay_masked(mdp.joint_deviation_l1_when_standing),
      weight=-1.5,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            ".*_arm_pitch_joint",
            ".*_arm_roll_joint",
            ".*_elbow_pitch_joint",
          ),
        )
      },
    ),

    # 关节限位越界 L1。
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),

    # 机械能 ‖|τ·q̇|‖（电功率代理）。
    "energy": RewardTermCfg(func=mdp.energy, weight=-1e-2),

    # 关节加速度 L2（抑抖动）。
    "dof_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-3e-6),

    # 动作变化率 L2 双档：移动档 / 站立档，门控 ‖cmd_xy‖+|wz| 与 0.05 比较。
    # 站立档 delay_masked（躺地 recovery env 指令多为 0，不该吃起身惩罚）。
    "action_rate_walk": RewardTermCfg(
      func=mdp.action_rate_l2_when_moving,
      weight=-0.25,
      params={"command_name": "twist", "stand_cmd_threshold": 0.05},
    ),
    "action_rate_stand": RewardTermCfg(
      func=mdp.delay_masked(mdp.action_rate_l2_when_standing),
      weight=-0.5,
      params={"command_name": "twist", "stand_cmd_threshold": 0.05},
    ),

    # 限位余量惩罚：|q̇|/额定转速 或 |τ|/额定力矩 超过 0.9 后按平方余量罚
    # （配合 sim 内硬钳制，把策略推离饱和区）。
    "joint_vel_limit": RewardTermCfg(
      func=mdp.joint_vel_limit_margin_penalty,
      weight=-10.0,
      params={"velocity_limits": N3_JOINT_VELOCITY_LIMITS, "ratio": 0.9},
    ),
    "joint_effort_limit": RewardTermCfg(
      func=mdp.joint_effort_limit_margin_penalty,
      weight=-10.0,
      params={"ratio": 0.9},
    ),
  }
  # fmt: on


_N3_SPEC = AmpRobotSpec(
  robot_cfg_factory=get_n3_robot_cfg,
  action_scale=N3_ACTION_SCALE,
  foot_body_regex=r"^(l_ankle_roll_link|r_ankle_roll_link)$",
  orientation_term="projected_gravity",
  # 观测噪声（逐帧均匀分布），对齐 Isaac。
  joint_pos_noise=(-0.05, 0.05),
  joint_vel_noise=(-0.5, 0.5),
  motion_dir=("amp_data", "N3", "json"),
  command=AmpCommandSpec(
    # Isaac 端：3~5 s 重采样、8% 站立、单模式采样（前进 0.30 / 后退 0.25 /
    # 侧移 0.25 / 纯转 0.20，每次只激活一根轴；command_mode_prob 覆盖 zero_prob）。
    # vx 上限 2.5（用户需求，真机最大速度）：专家覆盖 walk≤0.87 / run_slow≤2.33 /
    # walk2run≤2.24 / 后退 walk_b≤0.90，2.3~2.5 为少量外推。vy 专家只到 0.54，
    # ±0.8 留外推余量。模式采样自动取正/负半轴。
    resampling_time_range=(3.0, 5.0),
    rel_standing_envs=0.08,
    lin_vel_x=(-1.0, 2.5),
    lin_vel_y=(-0.8, 0.8),
    ang_vel_z=(-1.0, 1.0),
    zero_prob=(0.4, 0.2, 0.2),  # 仅声明对齐；被 command_mode_prob 覆盖
    command_mode_prob=(0.30, 0.25, 0.25, 0.20),
    single_axis_prob=None,
  ),
  # 【关闭】每回合恒定的 IMU 安装偏置随机化：factory 默认 0.15 作用在
  # projected_gravity 上 ≈ 8.6° 恒定倾斜误差（10× 单位错误），Isaac 端没有。
  imu_orientation_bias_std=None,
  imu_ang_vel_bias_std=None,
  com_body_name="waist_yaw_link",
  com_ranges={0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.025, 0.025)},
  mass_body_name="waist_yaw_link",
  mass_range=(-2.5, 2.5),
  # 周期推扰，对齐 Isaac ±0.5 m/s（角 ±0.5 rad/s）、间隔 3~6 s。
  push_velocity_range={
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.05, 0.05),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.5, 0.5),
  },
  rewards_factory=n3_rewards,
  # 关节速度硬钳制（逐控制步按额定转速截断）；执行器延迟见 constants.py
  # （全部执行器组 delay 0~6 控制步 = 0~120 ms，Isaac 同款）。
  joint_velocity_limits=_N3_JOINT_VELOCITY_LIMITS_SPEC,
  # Fall-recovery：30% env 终止抑制 ≤250 步并重置到起身片段帧；起身信号
  # track_root_height（×3.5，仅 delay env）。None = 纯走跑。
  recovery_motion_dir=("amp_data", "N3", "recovery"),
  delay_reset_env_ratio=0.3,
  max_delay_steps=250,
  with_height_scan=True,
  motion_required=False,
  reset_prob_rsi=0.2,
  reset_joint_vel_range=(-0.5, 0.5),
  friction_geom_pattern=".*_foot.*_collision",
  friction_range=(0.3, 1.6),
  pd_gain_range=(0.7, 1.0),
  with_ankle_randomization=True,
  # 地形相对终止：root 低于脚下扫描地面 0.35 m（平地与绝对 0.35 等价；
  # Isaac 用 0.6 但无 recovery 躺姿容忍需求）。
  fall_down_height=0.35,
  bad_orientation_limit=1.0,
  extent=2.0,
  viewer_distance=3.0,
  nconmax=80,
  njmax=400,
)


def n3_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the N3 flat-terrain AMP walk/run env config."""
  return make_amp_flat_env_cfg(_N3_SPEC, play=play)
