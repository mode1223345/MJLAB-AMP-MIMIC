"""N3 (0905) flat-terrain AMP locomotion env config.

Robot asset: ``asset_zoo/robots/N3`` (29 DoF). Assembly lives in
:mod:`mjlab.tasks.amp.amp_env_cfg`; this module only declares the N3 spec
and its reward set.
"""

from __future__ import annotations

from mjlab.asset_zoo.robots.N3.constants import (
  N3_ACTION_SCALE,
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


def n3_rewards() -> dict[str, RewardTermCfg]:
  """N3 AMP task rewards (kept verbatim from the pre-refactor config)."""
  foot_asset = SceneEntityCfg(
    "robot",
    body_names=("l_ankle_roll_link", "r_ankle_roll_link"),
    preserve_order=True,
  )
  return {
    "track_lin_vel_precision": RewardTermCfg(
      func=mdp.track_lin_vel_xy_yaw_frame_cauchy,
      weight=8.0,
      params={
        "absolute_scale": 0.2,
        "relative_scale": 0.06,
        "command_name": "twist",
      },
    ),
    "track_lin_vel_progress": RewardTermCfg(
      func=mdp.track_lin_vel_xy_yaw_frame_progress,
      weight=5.0,
      params={"activity_threshold": 0.3, "command_name": "twist"},
    ),
    "track_ang_height_vel_z_exp": RewardTermCfg(
      func=mdp.track_ang_vel_z_world_exp,
      weight=3.0,
      params={"std": 1.0, "command_name": "twist"},
    ),
    "track_ang_low_vel_z_exp": RewardTermCfg(
      func=mdp.track_ang_vel_z_world_exp,
      weight=5.0,
      params={"std": 0.4, "command_name": "twist"},
    ),
    "low_speed": RewardTermCfg(
      func=mdp.low_speed,
      weight=2.0,
      params={
        "low_speed_threshold": 0.8,
        "high_speed_threshold": 1.2,
        "command_name": "twist",
      },
    ),
    "track_ang_vel_stand_exp": RewardTermCfg(
      func=mdp.track_ang_vel_stand_world_exp,
      weight=2.0,
      params={"std": 0.5, "lin_vel_threshold": 2.0, "command_name": "twist"},
    ),
    "track_ang_vel_run_exp": RewardTermCfg(
      func=mdp.track_ang_vel_run_world_exp,
      weight=6.0,
      params={"std": 0.5, "command_name": "twist"},
    ),
    "flat_orientation_exp": RewardTermCfg(
      func=mdp.flat_orientation_exp, weight=1.5, params={"std": 0.25}
    ),
    # Soft band around n3 expert root z (~0.69–0.73 walk, 0.724 stand).
    "root_height": RewardTermCfg(
      func=mdp.root_height_out_of_range,
      weight=-5.0,
      params={"minimum_height": 0.65, "maximum_height": 0.78},
    ),
    "stand_still": RewardTermCfg(func=mdp.stand_still, weight=-1.0),
    "feet_contact": RewardTermCfg(
      func=mdp.feet_contact,
      weight=0.5,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.feet_slide,
      weight=-0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*_ankle_roll_link",)),
      },
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time_positive_biped,
      weight=4.0,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold": 0.5,
        "command_name": "twist",
      },
    ),
    "both_feet_air": RewardTermCfg(
      func=mdp.both_feet_air,
      weight=-4.0,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "feet_force": RewardTermCfg(
      func=mdp.feet_force,
      weight=-1e-2,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold": 580,
        "max_reward": 1200,
      },
    ),
    "feet_too_near": RewardTermCfg(
      func=mdp.feet_too_near_humanoid,
      weight=-4.0,
      params={"threshold": 0.15, "asset_cfg": foot_asset},
    ),
    "feet_lateral_distance": RewardTermCfg(
      func=mdp.feet_lateral_distance_yaw_frame,
      weight=-0.2,
      params={
        "minimum_distance": 0.15,
        "maximum_distance": 0.40,
        "too_wide_scale": 0.5,
        "asset_cfg": foot_asset,
      },
    ),
    "feet_distance_standing": RewardTermCfg(
      func=mdp.feet_distance_when_standing,
      weight=-2.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*_ankle_roll_link",)),
        "min_distance": 0.15,
        "max_distance": 0.40,
      },
    ),
    "joint_deviation_leg": RewardTermCfg(
      func=mdp.joint_deviation_l1_with_zero_flag,
      weight=-1.0,
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
    "joint_deviation_waist": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("waist_yaw_joint",))},
    ),
    "joint_deviation_standing": RewardTermCfg(
      func=mdp.joint_deviation_l1_when_standing,
      weight=-2.0,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            ".*_hip_pitch_joint",
            ".*_knee_pitch_joint",
            ".*_arm_pitch_joint",
            ".*_arm_roll_joint",
            ".*_arm_yaw_joint",
            ".*_elbow_pitch_joint",
            ".*_hand_yaw_joint",
            ".*_hand_roll_joint",
            ".*_hand_pitch_joint",
            "waist_yaw_joint",
            "head_pitch_joint",
            "head_yaw_joint",
          ),
        )
      },
    ),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "energy": RewardTermCfg(func=mdp.energy, weight=-1e-4),
    "dof_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.15),
  }


_N3_SPEC = AmpRobotSpec(
  robot_cfg_factory=get_n3_robot_cfg,
  action_scale=N3_ACTION_SCALE,
  foot_body_regex=r"^(l_ankle_roll_link|r_ankle_roll_link)$",
  orientation_term="projected_gravity",
  joint_pos_noise=(-0.1, 0.1),
  joint_vel_noise=(-1.0, 1.0),
  motion_dir=("amp_data", "N3", "json"),
  command=AmpCommandSpec(
    resampling_time_range=(5.0, 10.0),
    rel_standing_envs=0.2,
    # Match N3 motion envelope (|vx|≲1.55, |vy|≲0.91, |ωz|≲1.25).
    lin_vel_x=(-1.0, 1.0),
    lin_vel_y=(-0.8, 0.8),
    ang_vel_z=(-1.0, 1.0),
    zero_prob=(0.2, 0.2, 0.2),
  ),
  com_body_name="waist_yaw_link",
  com_ranges={0: (-0.02, 0.02), 1: (-0.02, 0.02), 2: (-0.02, 0.02)},
  mass_body_name="waist_yaw_link",
  mass_range=(-3.0, 3.0),
  push_velocity_range={
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (-0.05, 0.05),
    "roll": (-0.15, 0.15),
    "pitch": (-0.15, 0.15),
    "yaw": (-0.15, 0.15),
  },
  rewards_factory=n3_rewards,
  with_height_scan=True,
  motion_required=False,
  reset_prob_rsi=0.6,
  friction_geom_pattern=".*_foot_collision.*",
  friction_range=(0.6, 1.3),
  pd_gain_range=(0.7, 1.0),
  with_ankle_randomization=True,
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
