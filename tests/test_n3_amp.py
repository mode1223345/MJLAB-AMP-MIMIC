"""N3 asset, symmetry, rewards, and AMP training integration checks."""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from mjlab.asset_zoo.robots.N3.constants import (
  N3_JOINT_NAMES,
  get_n3_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import GridPatternCfg, RayCastSensorCfg
from mjlab.tasks.amp.config.N3.rl_cfg import AmpHimPpoRunnerCfg
from mjlab.tasks.amp.mdp.rewards import both_feet_air, root_height_out_of_range
from mjlab.tasks.amp.mdp.symmetry_n3 import (
  data_augmentation_func,
  flip_actor_obs,
  flip_critic_obs,
  flip_dof,
)
from mjlab.tasks.amp.rl import AmpHimOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Amp-Flat-N3-Walk"


@pytest.fixture(scope="module")
def n3_robot() -> tuple[Entity, mujoco.MjModel]:
  robot = Entity(get_n3_robot_cfg())
  return robot, robot.compile()


def test_n3_asset_and_task(n3_robot):
  robot, model = n3_robot
  assert robot.joint_names == N3_JOINT_NAMES
  assert (model.nq, model.nv, model.nu) == (36, 35, 29)
  # Joint-position actions resolve actuator targets in entity joint order.
  _, action_joints = robot.find_joints_by_actuator_names((".*",))
  assert tuple(action_joints) == N3_JOINT_NAMES
  assert load_runner_cls(TASK) is AmpHimOnPolicyRunner
  cfg = load_env_cfg(TASK)
  play = load_env_cfg(TASK, play=True)
  assert cfg.observations["actor"].enable_corruption
  assert not play.observations["actor"].enable_corruption
  assert "push_robot" not in play.events
  assert cfg.sim.mujoco.timestep * cfg.decimation == 0.02


def test_joint_mirror_matches_model_kinematics(n3_robot):
  """Check mirror signs/order against MJCF axes and body transforms."""
  robot, model = n3_robot
  data = mujoco.MjData(model)
  data.qpos[:] = model.key_qpos[0]
  q = torch.linspace(-0.15, 0.15, 29, dtype=torch.float64)
  data.qpos[7:] = q.numpy()
  mujoco.mj_forward(model, data)
  original = data.xpos.copy()
  data.qpos[7:] = flip_dof(q).numpy()
  mujoco.mj_forward(model, data)
  for name in robot.body_names:
    opposite = (
      "r_" + name[2:]
      if name.startswith("l_")
      else "l_" + name[2:]
      if name.startswith("r_")
      else name
    )
    expected = original[model.body(opposite).id] * np.array([1, -1, 1])
    np.testing.assert_allclose(data.xpos[model.body(name).id], expected, atol=1e-5)
  torch.testing.assert_close(flip_dof(flip_dof(q)), q)


@pytest.mark.parametrize("history_shape", [(2, 96), (2, 5, 96), (2, 480)])
def test_actor_symmetry_history(history_shape):
  obs = torch.randn(history_shape)
  mirrored = flip_actor_obs(obs)
  torch.testing.assert_close(flip_actor_obs(mirrored), obs)
  frames = obs.reshape(2, -1, 96)
  flipped = mirrored.reshape(2, -1, 96)
  torch.testing.assert_close(flipped[..., 1], -frames[..., 1])
  torch.testing.assert_close(flipped[..., 7], -frames[..., 7])
  # Left knee receives right knee; central head pitch keeps its sign.
  torch.testing.assert_close(flipped[..., 12], frames[..., 18])
  torch.testing.assert_close(flipped[..., 22], frames[..., 22])


def test_critic_symmetry_tracks_ray_coordinates():
  env = MagicMock()
  env.unwrapped.cfg = load_env_cfg(TASK)
  scan = next(s for s in env.unwrapped.cfg.scene.sensors if s.name == "terrain_scan")
  assert isinstance(scan, RayCastSensorCfg)
  assert isinstance(scan.pattern, GridPatternCfg)
  offsets, _ = scan.pattern.generate_rays(None, "cpu")
  obs = torch.randn(2, 101 + len(offsets))
  obs[:, 101:] = offsets[:, 0] + 10 * offsets[:, 1]
  flipped = flip_critic_obs(obs, env)
  expected_scan = offsets[:, 0] - 10 * offsets[:, 1]
  torch.testing.assert_close(flipped[:, 101:], expected_scan.expand(2, -1))
  torch.testing.assert_close(flipped[:, 99:101], obs[:, [100, 99]])
  torch.testing.assert_close(flip_critic_obs(flipped, env), obs)
  batch = TensorDict(
    {"actor": torch.randn(2, 480), "critic": obs, "next_obs": torch.randn(2, 96)},
    batch_size=[2],
  )
  actions = torch.randn(2, 29)
  augmented, augmented_actions = data_augmentation_func(env, batch, actions)
  assert augmented is not None and augmented_actions is not None
  assert augmented.batch_size == (4,)
  torch.testing.assert_close(augmented["actor"][:2], batch["actor"])
  torch.testing.assert_close(augmented["critic"][2:], flipped)
  torch.testing.assert_close(augmented_actions[2:], flip_dof(actions))
  assert data_augmentation_func(env) == (None, None)


def test_n3_height_and_air_rewards():
  env = MagicMock()
  robot = MagicMock()
  sensor = MagicMock()
  env.scene.__getitem__.side_effect = {
    "robot": robot,
    "feet_ground_contact": sensor,
  }.__getitem__
  env.scene.env_origins = torch.tensor([[0, 0, 2.0]]).expand(4, -1)
  robot.data.root_link_pos_w = torch.tensor(
    [[0, 0, 2.5], [0, 0, 2.65], [0, 0, 2.78], [0, 0, 2.9]]
  )
  torch.testing.assert_close(
    root_height_out_of_range(env, 0.65, 0.78),
    torch.tensor([0.15**2, 0, 0, 0.12**2]),
  )
  sensor.data.found = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
  torch.testing.assert_close(both_feet_air(env), torch.tensor([1.0, 0, 0, 0]))


@pytest.mark.slow
def test_n3_amp_training_step(n3_robot, tmp_path: Path):
  """Exercise resets, sensors, symmetry, AMP/HIM updates, and checkpoint I/O.

  The temporary static motion is a test fixture, not walking expert data.
  """
  robot, model = n3_robot
  data = mujoco.MjData(model)
  data.qpos[:] = model.key_qpos[0]
  mujoco.mj_forward(model, data)
  frame = np.concatenate(
    (
      data.qpos[:3],
      data.qpos[[4, 5, 6, 3]],
      data.qpos[7:],
      (data.xpos[1:] - data.qpos[:3]).ravel(),
      np.zeros(6 + 29),
      np.ones(2),
    )
  )
  motion = tmp_path / "synthetic_test_pose.json"
  motion.write_text(
    json.dumps(
      {
        "JointNames": list(robot.joint_names),
        "KeyPosNames": list(robot.body_names),
        "Frames": [frame.tolist()] * 20,
        "FrameDuration": 0.02,
        "MotionWeight": 1.0,
      }
    )
  )
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 4
  cfg.episode_length_s = 0.04
  cfg.amp_motion_files = [str(motion)]
  cfg.amp_num_preload_transitions = 32
  cfg.events["reset_robot_states"].params["prob_rsi"] = 1.0
  agent = load_rl_cfg(TASK)
  assert isinstance(agent, AmpHimPpoRunnerCfg)
  agent.num_steps_per_env = 4
  agent.algorithm.num_learning_epochs = 1
  agent.algorithm.num_mini_batches = 2
  agent.algorithm.discriminator_num_mini_batches = 1
  agent.algorithm.amp_replay_buffer_size = 32
  previous_threads = torch.get_num_threads()
  torch.set_num_threads(2)
  env = None
  runner = None
  try:
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    runner = AmpHimOnPolicyRunner(env, asdict(agent), str(tmp_path), device="cpu")
    obs = runner.env.get_observations()
    assert obs["actor"].shape == (4, 480)
    assert obs["critic"].shape == (4, 197)
    assert obs["discriminator"].shape == (4, 4, 156)
    before = next(runner.alg.policy.actor.parameters()).detach().clone()
    runner.learn(1)
    after = next(runner.alg.policy.actor.parameters()).detach()
    assert not torch.equal(before, after)
    assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
    assert (tmp_path / "model_0.pt").is_file()
    env.close()
    play_cfg = load_env_cfg(TASK, play=True)
    play_cfg.scene.num_envs = 4
    play_cfg.amp_motion_files = [str(motion)]
    play_cfg.amp_num_preload_transitions = 32
    env = ManagerBasedRlEnv(cfg=play_cfg, device="cpu")
    restored = AmpHimOnPolicyRunner(env, asdict(agent), device="cpu")
    restored.load(str(tmp_path / "model_0.pt"), map_location="cpu")
    with torch.inference_mode():
      actions = restored.get_inference_policy()(restored.env.get_observations())
    assert actions.shape == (4, 29)
    assert torch.isfinite(actions).all()
  finally:
    if runner is not None and runner.writer is not None:
      runner.writer.close()
    if env is not None:
      env.close()
    torch.set_num_threads(previous_threads)
