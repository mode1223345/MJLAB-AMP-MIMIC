"""Factory for flat-terrain AMP locomotion env configs.

All AMP env configs (F5 variants, Labubu, N3) share the same assembly:
ContactSensor + actor/critic/discriminator observation groups + joint
position actions + a twist command + the standard event/termination/scene
scaffolding. Only per-robot values differ, captured in
:class:`AmpRobotSpec`; task-specific reward dicts are provided by each
robot config module via ``rewards_factory``.

The factory reproduces the pre-refactor per-robot configs field-for-field
(term dicts included, insertion order included); equivalence is pinned by
the golden tests.
"""

from __future__ import annotations

import glob
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.amp import mdp
from mjlab.tasks.amp.mdp.commands import UniformVelocityWithZeroCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# src/mjlab/tasks/amp/amp_env_cfg.py -> repository root.
_REPO_ROOT = Path(__file__).resolve().parents[4]

_ROOT_POS_RANGE = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)}
_ROOT_VEL_RANGE = {
  "x": (-0.05, 0.05),
  "y": (-0.05, 0.05),
  "z": (-0.05, 0.05),
  "roll": (-0.05, 0.05),
  "pitch": (-0.05, 0.05),
  "yaw": (-0.05, 0.05),
}


@dataclass(frozen=True)
class AmpCommandSpec:
  """Twist command sampling ranges."""

  resampling_time_range: tuple[float, float]
  rel_standing_envs: float
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  ang_vel_z: tuple[float, float]
  zero_prob: tuple[float, float, float]
  # Isaac-N3 mode sampling (overrides zero_prob when set): probabilities of
  # (forward, backward, lateral, pure-turn); one axis active per resample.
  command_mode_prob: tuple[float, float, float, float] | None = None
  # Axis-mode sampling (overrides zero_prob when set): P(single active axis)
  # / P(two active axes); remainder is three-axis. axis_weights orders axis
  # importance (vx, vy, wz).
  single_axis_prob: float | None = None
  double_axis_prob: float | None = None
  axis_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)


@dataclass(frozen=True)
class AmpRobotSpec:
  """Per-robot values for :func:`make_amp_flat_env_cfg`."""

  robot_cfg_factory: Callable[[], EntityCfg]
  action_scale: dict[str, float]
  foot_body_regex: str  # ContactSensor primary pattern.
  orientation_term: Literal["euler_angles", "projected_gravity"]
  joint_pos_noise: tuple[float, float]
  joint_vel_noise: tuple[float, float]
  motion_dir: tuple[str, ...]  # relative to motions/
  command: AmpCommandSpec
  com_body_name: str
  com_ranges: Mapping[int, tuple[float, float]]
  mass_body_name: str
  mass_range: tuple[float, float]
  push_velocity_range: Mapping[str, tuple[float, float]]
  rewards_factory: Callable[[], dict[str, RewardTermCfg]]
  # Per-episode constant IMU bias stds (mounting misalignment + gyro drift).
  # None falls back to plain per-frame noise without bias. Defaults are None:
  # 0.03/0.15 turned out to be a ~10x unit error (0.15 on the projected-gravity
  # unit vector is a constant ~8.6 deg tilt, not the claimed 0.9 deg) and the
  # proven reference config has no such randomization. Opt in per task.
  imu_ang_vel_bias_std: float | None = None
  imu_orientation_bias_std: float | None = None
  # Joint velocity limits [rad/s] keyed by joint-name regex tuples. Wired as
  # per-control-step clamp events (motor rated-speed envelope). None disables.
  joint_velocity_limits: Mapping[tuple[str, ...], float] | None = None
  # Fall-recovery training (delayed termination + recovery-frame resets).
  # recovery_motion_dir enables the whole mechanism; delay envs get their fall
  # terminations suppressed for max_delay_steps and are re-initialized from
  # the recovery motion set. Set to None for pure locomotion.
  recovery_motion_dir: tuple[str, ...] | None = None
  delay_reset_env_ratio: float = 0.3
  max_delay_steps: int = 250
  recovery_height_std: float = 0.3
  recovery_height_reward_scale: float = 3.5
  terminated_penalty_weight: float = -50.0
  # Kept small: the startup z-lift precompute loops over every preloaded
  # recovery frame once (~1 s per 10k frames on CPU).
  recovery_num_preload_transitions: int = 20_000
  with_height_scan: bool = False
  motion_required: bool = True  # False reproduces empty-dir-returns-empty.
  amp_horizon: int = 4
  reset_prob_rsi: float = 0.5
  reset_joint_pos_range: tuple[float, float] = (-0.2, 0.2)
  reset_joint_vel_range: tuple[float, float] = (-1.0, 1.0)
  friction_geom_pattern: str = ".*_foot.*_collision"
  friction_range: tuple[float, float] = (0.3, 1.6)
  pd_gain_range: tuple[float, float] = (0.8, 1.2)
  push_interval_s: tuple[float, float] = (3.0, 6.0)
  with_ankle_randomization: bool = False
  ankle_randomization_range: tuple[float, float] = (0.7, 1.5)
  fall_down_height: float | None = 0.5  # None: no fall_down termination.
  bad_orientation_limit: float = 0.6
  extent: float = 3.0
  viewer_distance: float = 4.0
  nconmax: int = 50
  njmax: int = 300
  episode_length_s: float = 20.0
  num_envs: int = 4096
  amp_num_preload_transitions: int = 200_000


def _default_amp_motion_files(motion_dir: tuple[str, ...], required: bool) -> list[str]:
  motion_path = _REPO_ROOT.joinpath("motions", *motion_dir)
  files = sorted(glob.glob(str(motion_path / "*.json")))
  if required and not files:
    raise FileNotFoundError(f"No AMP motions found in {motion_path}")
  return files


def _terrain_scan_cfg() -> RayCastSensorCfg:
  return RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base_link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.1, 0.7), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
  )


def _feet_ground_cfg(foot_body_regex: str) -> ContactSensorCfg:
  return ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=foot_body_regex,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )


def _imu_noise(
  bias_std: float | None, n_min: float, n_max: float
) -> Unoise | NoiseModelWithAdditiveBiasCfg:
  """Per-frame noise, optionally with a per-episode constant IMU bias.

  The bias approximates a small fixed IMU mounting rotation (orientation
  error ~ |g| * sin(angle), e.g. std 0.15 ~= 0.9 deg) sampled per env and
  held constant for the episode, on top of the usual per-frame jitter.
  """
  per_frame = Unoise(n_min=n_min, n_max=n_max)
  if bias_std is None:
    return per_frame
  return NoiseModelWithAdditiveBiasCfg(
    noise_cfg=per_frame,
    bias_noise_cfg=GaussianNoiseCfg(std=bias_std),
  )


def make_amp_flat_env_cfg(
  spec: AmpRobotSpec, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Assemble the flat-terrain AMP env cfg for one robot spec.

  Args:
    spec: Per-robot values; see :class:`AmpRobotSpec`.
    play: True disables observation noise and push events and lengthens
      episodes for visualization.
  """
  terrain_scan = _terrain_scan_cfg()

  actor_terms = {
    "velocity_commands": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel,
      noise=_imu_noise(spec.imu_ang_vel_bias_std, -0.2, 0.2),
    ),
  }
  if spec.orientation_term == "euler_angles":
    actor_terms["euler_angles"] = ObservationTermCfg(
      func=mdp.euler_angles,
      noise=_imu_noise(spec.imu_orientation_bias_std, -0.05, 0.05),
    )
  else:
    actor_terms["projected_gravity"] = ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=_imu_noise(spec.imu_orientation_bias_std, -0.05, 0.05),
    )
  actor_terms["joint_pos"] = ObservationTermCfg(
    func=mdp.joint_pos_rel,
    noise=Unoise(*spec.joint_pos_noise),
  )
  actor_terms["joint_vel"] = ObservationTermCfg(
    func=mdp.joint_vel_rel,
    noise=Unoise(*spec.joint_vel_noise),
  )
  actor_terms["actions"] = ObservationTermCfg(func=mdp.last_action)

  orientation_name = spec.orientation_term
  critic_terms = {
    "velocity_commands": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
    orientation_name: ObservationTermCfg(func=getattr(mdp, orientation_name)),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "feet_contact": ObservationTermCfg(
      func=mdp.current_feet_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }
  if spec.with_height_scan:
    critic_terms["height_scan"] = ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      clip=(-1.0, 1.0),
      scale=1 / terrain_scan.max_distance,
    )

  disc_terms = {
    "joint_pos": ObservationTermCfg(func=mdp.amp_joint_pos),
    "key_pos": ObservationTermCfg(func=mdp.amp_key_pos),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
    "joint_vel": ObservationTermCfg(func=mdp.amp_joint_vel),
    "feet_contact": ObservationTermCfg(
      func=mdp.current_feet_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
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
    "discriminator": ObservationGroupCfg(
      terms=disc_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=spec.amp_horizon,
      flatten_history_dim=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=spec.action_scale,
      use_default_offset=True,
    )
  }

  cmd = spec.command
  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityWithZeroCommandCfg(
      entity_name="robot",
      resampling_time_range=cmd.resampling_time_range,
      rel_standing_envs=cmd.rel_standing_envs,
      debug_vis=True,
      ranges=UniformVelocityWithZeroCommandCfg.Ranges(
        lin_vel_x=cmd.lin_vel_x,
        lin_vel_y=cmd.lin_vel_y,
        ang_vel_z=cmd.ang_vel_z,
        zero_prob=cmd.zero_prob,
        command_mode_prob=cmd.command_mode_prob,
        single_axis_prob=cmd.single_axis_prob,
        double_axis_prob=cmd.double_axis_prob,
        axis_weights=cmd.axis_weights,
      ),
    )
  }

  events = {
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          geom_names=(spec.friction_geom_pattern,),
        ),
        "operation": "abs",
        "ranges": spec.friction_range,
        "shared_random": True,
      },
    ),
    "add_joint_default_pos": EventTermCfg(
      mode="startup",
      func=mdp.randomize_joint_default_pos,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "pos_distribution_params": (-0.02, 0.02),
        "operation": "add",
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(spec.com_body_name,)),
        "operation": "add",
        "ranges": dict(spec.com_ranges),
      },
    ),
    "add_base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(spec.mass_body_name,)),
        "operation": "add",
        "ranges": spec.mass_range,
      },
    ),
    "reset_robot_states": EventTermCfg(
      func=mdp.reset_root_state_amp,
      mode="reset",
      params={
        "reference_state_initialization": True,
        "prob_rsi": spec.reset_prob_rsi,
        "use_recovery_for_delay_envs": spec.recovery_motion_dir is not None,
        "root_pos_range": dict(_ROOT_POS_RANGE),
        "root_vel_range": dict(_ROOT_VEL_RANGE),
        "joint_pos_range": spec.reset_joint_pos_range,
        "joint_vel_range": spec.reset_joint_vel_range,
      },
    ),
    "randomize_actuator_gains": EventTermCfg(
      mode="reset",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": spec.pd_gain_range,
        "kd_range": spec.pd_gain_range,
        "operation": "scale",
      },
    ),
  }
  if spec.with_ankle_randomization:
    events["randomize_ankle_armature"] = EventTermCfg(
      mode="reset",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_ankle.*joint",)),
        "ranges": spec.ankle_randomization_range,
        "operation": "scale",
      },
    )
    events["randomize_ankle_friction"] = EventTermCfg(
      mode="reset",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_ankle.*joint",)),
        "ranges": spec.ankle_randomization_range,
        "operation": "scale",
      },
    )
  events["push_robot"] = EventTermCfg(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=spec.push_interval_s,
    params={"velocity_range": dict(spec.push_velocity_range)},
  )
  if spec.joint_velocity_limits:
    for index, (patterns, limit) in enumerate(spec.joint_velocity_limits.items()):
      events[f"clamp_joint_velocity_{index}"] = EventTermCfg(
        func=mdp.clamp_joint_velocity,
        mode="step",
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=patterns),
          "velocity_limit": limit,
        },
      )

  delay_ratio = spec.delay_reset_env_ratio
  if spec.recovery_motion_dir is not None:
    # NOTE: play keeps the training ratio. Forcing 1.0 here made every play
    # env a delay env, zeroing all delay-masked rewards in the viewer.
    events["install_delayed_recovery"] = EventTermCfg(
      func=mdp.install_delayed_recovery,
      mode="startup",
      params={
        "recovery_motion_files": _default_amp_motion_files(
          spec.recovery_motion_dir, required=True
        ),
        "delay_reset_env_ratio": delay_ratio,
        "max_delay_steps": spec.max_delay_steps,
        "reference_observation_horizon": spec.amp_horizon,
        "num_preload_transitions": spec.recovery_num_preload_transitions,
      },
    )

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }
  if spec.fall_down_height is not None:
    if spec.with_height_scan:
      # Terrain-relative (Isaac N3 [ADD 2026-09-04]): the absolute check goes
      # inert/false-fires on elevated or sloped terrain; with a height scan,
      # measure the root against the ground directly under it. The function
      # itself falls back to the absolute check when the sensor is missing.
      terminations["fall_down"] = TerminationTermCfg(
        func=mdp.terminations.root_height_below_terrain,
        params={
          "minimum_height": spec.fall_down_height,
          "sensor_name": "terrain_scan",
        },
      )
    else:
      terminations["fall_down"] = TerminationTermCfg(
        func=envs_mdp.root_height_below_minimum,
        params={"minimum_height": spec.fall_down_height},
      )
  terminations["bad_orientation"] = TerminationTermCfg(
    func=mdp.bad_orientation, params={"limit_angle": spec.bad_orientation_limit}
  )

  sensors: tuple = (feet := _feet_ground_cfg(spec.foot_body_regex),)
  if spec.with_height_scan:
    sensors = (terrain_scan, feet)

  rewards = spec.rewards_factory()
  if spec.recovery_motion_dir is not None:
    # Recovery learning signals (delay envs only for the height term).
    rewards["track_root_height"] = RewardTermCfg(
      func=mdp.track_root_height,
      weight=1.0,
      params={
        "std": spec.recovery_height_std,
        "delay_env_rew_ratio": spec.recovery_height_reward_scale,
      },
    )
    rewards["is_terminated"] = RewardTermCfg(
      func=mdp.is_terminated,
      weight=spec.terminated_penalty_weight,
    )

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": spec.robot_cfg_factory()},
      sensors=sensors,
      num_envs=spec.num_envs,
      extent=spec.extent,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=spec.viewer_distance,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=spec.nconmax,
      njmax=spec.njmax,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=spec.episode_length_s,
    amp_motion_files=_default_amp_motion_files(spec.motion_dir, spec.motion_required),
    amp_reference_observation_horizon=spec.amp_horizon,
    amp_num_preload_transitions=spec.amp_num_preload_transitions,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg
