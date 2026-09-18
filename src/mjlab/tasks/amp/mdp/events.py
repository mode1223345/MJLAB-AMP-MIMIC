"""AMP reset / domain-randomization events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def randomize_joint_default_pos(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  pos_distribution_params: tuple[float, float] = (-0.02, 0.02),
  operation: Literal["add"] = "add",
  action_name: str = "joint_pos",
) -> None:
  """Randomize default joint positions (calibration zero-point error).

  Port of Noetix ``randomize_joint_default_pos``: offsets
  ``default_joint_pos`` and mirrors the change into the joint-position
  action term offset so relative actions stay consistent.
  """
  if operation != "add":
    raise ValueError("randomize_joint_default_pos only supports operation='add'")

  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice):
    n_joints = asset.data.default_joint_pos.shape[-1]
    joint_ids_t = torch.arange(n_joints, device=env.device, dtype=torch.long)
  else:
    joint_ids_t = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)

  noise = sample_uniform(
    pos_distribution_params[0],
    pos_distribution_params[1],
    (len(env_ids), len(joint_ids_t)),
    device=env.device,
  )
  asset.data.default_joint_pos[env_ids[:, None], joint_ids_t] = (
    asset.data.default_joint_pos[env_ids[:, None], joint_ids_t] + noise
  )

  # Keep action offset in sync when using default-offset joint position control.
  try:
    action_term = env.action_manager.get_term(action_name)
  except KeyError:
    return
  if not isinstance(action_term, JointPositionAction):
    return
  if not isinstance(action_term._offset, torch.Tensor):
    return

  # Map asset joint ids -> action target columns.
  target_ids = action_term._target_ids
  action_term._offset[env_ids] = asset.data.default_joint_pos[env_ids][:, target_ids]


def clamp_joint_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  velocity_limit: float = 10.0,
) -> None:
  """Clamp joint velocities to a motor rated speed, per control step.

  Approximates PhysX-style ``velocity_limit_sim``: joint velocities are read
  after the decimation loop, clamped to +/- ``velocity_limit``, and written
  back so the next control step starts within the motor's speed envelope.
  Wire as an event term with ``mode="step"``.
  """
  del env_ids  # Step-mode events always apply to all envs.
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  joint_vel = asset.data.joint_vel[:, joint_ids].clone()
  clamped = joint_vel.clamp(-velocity_limit, velocity_limit)
  if torch.equal(clamped, joint_vel):
    return
  asset.write_joint_velocity_to_sim(clamped, joint_ids=joint_ids)


def _write_motion_frame_state(
  env: ManagerBasedRlEnv,
  asset: Entity,
  env_ids: torch.Tensor,
  motion_loader,
  frames: torch.Tensor | None = None,
  z_lift: torch.Tensor | None = None,
) -> None:
  """Write a sampled motion frame (root pose/vel + joints) to the sim.

  Args:
    frames: pre-sampled full frames; sampled internally when None.
    z_lift: per-frame extra root-z lift [m] (deep-penetration compensation
      for lying recovery frames; 0 for standing frames).
  """
  if frames is None:
    frames = motion_loader.get_full_frame_batch(len(env_ids))
  positions = motion_loader.get_root_pos_batch(frames)
  positions[:, :2] = 0.0
  positions += env.scene.env_origins[env_ids]
  positions[:, 2] += 0.05
  if z_lift is not None:
    positions[:, 2] += z_lift
  quat_xyzw = motion_loader.get_root_rot_batch(frames)
  orientations = torch.cat(
    (quat_xyzw[:, -1].unsqueeze(1), quat_xyzw[:, :-1]), dim=1
  )  # xyzw -> wxyz

  base_lin_vel = motion_loader.get_linear_vel_batch(frames)
  base_ang_vel = motion_loader.get_angular_vel_batch(frames)
  lin_vel = quat_apply(orientations, base_lin_vel)
  ang_vel = quat_apply(orientations, base_ang_vel)
  velocities = torch.cat([lin_vel, ang_vel], dim=-1)

  joint_pos = motion_loader.get_joint_pose_batch(frames)[
    ..., motion_loader.joint_mapping["motion2sim"]
  ]
  joint_vel = motion_loader.get_joint_vel_batch(frames)[
    ..., motion_loader.joint_mapping["motion2sim"]
  ]

  soft_pos_limits = asset.data.soft_joint_pos_limits
  if soft_pos_limits is not None:
    limits = soft_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])

  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def _precompute_recovery_z_lift(env: ManagerBasedRlEnv, motion_loader) -> None:
  """Per-preloaded-frame root-z lift so no collision geom starts underground.

  Recovery mocap was retargeted assuming full-body contact; our reduced
  collision set (torso box + capsules) sits deeper when lying down. For each
  preloaded frame, FK on a CPU copy of the robot (+ ground plane) measures
  the deepest collision penetration and stores the extra lift that puts the
  lowest geom ~1 cm above ground after the standard +5 cm offset.
  """
  import time as _time

  import mujoco

  t0 = _time.time()
  spec = env.scene["robot"].spec.copy()
  spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0.0, 0.0, 0.05],
    contype=1,
    conaffinity=1,
    condim=3,
  )
  model = spec.compile()
  data = mujoco.MjData(model)

  qpos_adr = torch.tensor(
    [
      model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
      for name in env.scene["robot"].joint_names
    ]
  )
  motion2sim = torch.tensor(motion_loader.joint_mapping["motion2sim"], dtype=torch.long)
  collision_geoms = np.flatnonzero((model.geom_contype | model.geom_conaffinity) != 0)

  n = motion_loader.num_preload_transitions
  lift = torch.zeros(n)
  frames_cpu = motion_loader.preloaded_states[:, 0].cpu()
  for i in range(n):
    frame = frames_cpu[i]
    qpos = np.zeros(model.nq)
    qpos[0:3] = frame[0:3]
    qpos[3:7] = (frame[6], frame[3], frame[4], frame[5])  # xyzw -> wxyz
    joints_sim = frame[7:36][motion2sim.numpy()]
    qpos[qpos_adr.numpy()] = joints_sim
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    min_dist = 0.0
    for c in range(data.ncon):
      g1, g2 = data.contact.geom1[c], data.contact.geom2[c]
      if g1 in collision_geoms or g2 in collision_geoms:
        # Robot-vs-plane pairs only exist for collision geoms here.
        min_dist = min(min_dist, float(data.contact.dist[c]))
    # Lowest geom ends at min_dist + 0.05 after the standard offset; lift
    # the extra amount to clear ~1 cm.
    lift[i] = max(0.0, -(min_dist + 0.05) + 0.01)
  motion_loader.z_lift_table = lift.to(motion_loader.device)
  print(
    "[install_delayed_recovery] z-lift table: "
    f"{int((lift > 0).sum())}/{n} frames lifted, max {lift.max():.3f} m, "
    f"{_time.time() - t0:.1f} s"
  )


def install_delayed_recovery(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  recovery_motion_files: list[str],
  delay_reset_env_ratio: float = 0.0,
  max_delay_steps: int = 0,
  reference_observation_horizon: int = 4,
  num_preload_transitions: int = 100_000,
):
  """Startup event: attach the recovery MotionLoader + delayed termination.

  Loads the fall-recovery motion set as ``motion_loader_recovery`` (never
  fed to the discriminator — recovery style is task-reward-driven) and, when
  ``delay_reset_env_ratio > 0``, flags that fraction of envs as delay envs
  whose fall terminations are suppressed for ``max_delay_steps``.
  """
  del env_ids  # Startup events run once over all envs.
  unwrapped = env.unwrapped
  if (
    recovery_motion_files and getattr(unwrapped, "motion_loader_recovery", None) is None
  ):
    from ..rl.motion_loader_attach import build_motion_loader

    object.__setattr__(
      unwrapped,
      "motion_loader_recovery",
      build_motion_loader(
        env,
        recovery_motion_files,
        reference_observation_horizon,
        num_preload_transitions,
      ),
    )
    _precompute_recovery_z_lift(env, unwrapped.motion_loader_recovery)

  num_delay = int(env.num_envs * delay_reset_env_ratio)
  if num_delay > 0 and max_delay_steps > 0:
    from .terminations import DelayedTerminationManager

    delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    delay_indices = torch.randperm(env.num_envs, device=env.device)[:num_delay]
    delay_mask[delay_indices] = True
    env.termination_manager = DelayedTerminationManager(
      base=env.termination_manager,
      delay_env_mask=delay_mask,
      max_delay_steps=max_delay_steps,
    )
    print(
      "[install_delayed_recovery] DelayedTerminationManager installed: "
      f"{num_delay}/{env.num_envs} envs, max_delay_steps={max_delay_steps}"
    )


def reset_root_state_amp(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  root_pos_range: dict[str, tuple[float, float]],
  root_vel_range: dict[str, tuple[float, float]],
  joint_pos_range: tuple[float, float],
  joint_vel_range: tuple[float, float],
  reference_state_initialization: bool = False,
  prob_rsi: float = 0.0,
  use_recovery_for_delay_envs: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  """Reset root/joint state, optionally from AMP motion frames (RSI).

  With ``use_recovery_for_delay_envs`` and delayed termination installed,
  delay envs are always initialized from random recovery-motion frames
  (fallen / get-up poses); the remaining envs follow the normal per-env RSI
  coin flip.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  motion_loader = getattr(env.unwrapped, "motion_loader", None)

  # Split off delay envs when recovery initialization is enabled.
  delay_ids: torch.Tensor = env_ids[:0]
  normal_ids = env_ids
  if use_recovery_for_delay_envs:
    from .terminations import get_delay_env_mask

    mask = get_delay_env_mask(env)
    if (
      mask is not None
      and getattr(env.unwrapped, "motion_loader_recovery", None) is not None
    ):
      is_delay = mask[env_ids]
      delay_ids = env_ids[is_delay]
      normal_ids = env_ids[~is_delay]

  if len(delay_ids) > 0:
    recovery_loader = env.unwrapped.motion_loader_recovery
    lift_table = getattr(recovery_loader, "z_lift_table", None)
    if lift_table is not None:
      idxs = torch.randint(
        0, recovery_loader.num_preload_transitions, (len(delay_ids),)
      )
      frames = recovery_loader.preloaded_states[idxs, 0]
      z_lift = lift_table[idxs]
      _write_motion_frame_state(
        env, asset, delay_ids, recovery_loader, frames=frames, z_lift=z_lift
      )
    else:
      _write_motion_frame_state(env, asset, delay_ids, recovery_loader)

  if len(normal_ids) == 0:
    return

  env_ids = normal_ids
  if motion_loader is not None and reference_state_initialization:
    rsi_mask = torch.rand(len(env_ids), device=env.device) < prob_rsi
  else:
    rsi_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)

  if rsi_mask.any():
    assert motion_loader is not None  # guarded by rsi_mask construction.
    rsi_ids = env_ids[rsi_mask]
    _write_motion_frame_state(env, asset, rsi_ids, motion_loader)
    env_ids = env_ids[~rsi_mask]
    if len(env_ids) == 0:
      return

  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  range_list = [
    root_pos_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  rand_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  positions = (
    root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
  )
  orientations_delta = quat_from_euler_xyz(
    rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
  )
  orientations = quat_mul(root_states[:, 3:7], orientations_delta)

  vel_list = [
    root_vel_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  vel_ranges = torch.tensor(vel_list, device=env.device)
  vel_samples = sample_uniform(
    vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  velocities = root_states[:, 7:13] + vel_samples

  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_pos is not None and default_joint_vel is not None
  joint_pos = default_joint_pos[env_ids].clone()
  joint_vel = default_joint_vel[env_ids].clone()
  joint_pos += sample_uniform(*joint_pos_range, joint_pos.shape, joint_pos.device)
  joint_vel += sample_uniform(*joint_vel_range, joint_vel.shape, joint_vel.device)

  soft_pos_limits = asset.data.soft_joint_pos_limits
  if soft_pos_limits is not None:
    limits = soft_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])

  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)
