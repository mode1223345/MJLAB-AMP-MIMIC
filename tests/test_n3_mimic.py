"""N3 mimic (MHA+HIM) loader, config, obs layout, and training checks."""

from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots.N3.constants import N3_JOINT_NAMES, get_n3_robot_cfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.registry import (
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)
from mjlab.tasks.tracking.config.N3.env_cfg import (
  N3_MIMIC_BODY_NAMES,
  n3_mimic_env_cfg,
)
from mjlab.tasks.tracking.config.N3.rl_cfg import MhaHimPpoRunnerCfg
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.tasks.tracking.rl import MimicHimOnPolicyRunner

TASK = "Mjlab-Tracking-Flat-N3-Mimic"
REPO_ROOT = Path(__file__).parents[1]
CLIP_FRAMES = 12  # 0.24 s at 50 fps; motion_end fires at frame 10.


@pytest.fixture(scope="module")
def n3_robot() -> tuple[Entity, mujoco.MjModel]:
  robot = Entity(get_n3_robot_cfg())
  return robot, robot.compile()


def _keyframe_state(model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  data = mujoco.MjData(model)
  data.qpos[:] = model.key_qpos[0]
  mujoco.mj_forward(model, data)
  return data.qpos[7:].copy(), data.xpos[1:].copy(), data.xquat[1:].copy()


def _synthetic_motion_dir(root: Path, model) -> Path:
  """Two clips whose npz name axes are SHUFFLED (values encode the layout)."""
  joint_pos0, body_pos0, body_quat0 = _keyframe_state(model)
  rng = np.random.default_rng(0)
  joint_order = rng.permutation(len(N3_JOINT_NAMES))
  all_bodies = [model.body(i).name for i in range(1, model.nbody)]
  body_order = rng.permutation(len(all_bodies))
  num_bodies = len(all_bodies)
  motion_dir = root / "npz"
  motion_dir.mkdir(parents=True)
  for clip in range(2):
    t = np.arange(CLIP_FRAMES, dtype=np.float32)
    wiggle = 0.01 * np.sin(0.5 * t[:, None] + clip)
    np.savez(
      motion_dir / f"clip{clip}.npz",
      joint_names=np.array([N3_JOINT_NAMES[i] for i in joint_order]),
      body_names=np.array([all_bodies[i] for i in body_order]),
      joint_pos=(joint_pos0 + wiggle)[:, joint_order],
      joint_vel=np.zeros((CLIP_FRAMES, len(joint_order))),
      body_pos_w=np.tile(body_pos0, (CLIP_FRAMES, 1, 1))[:, body_order],
      body_quat_w=np.tile(body_quat0, (CLIP_FRAMES, 1, 1))[:, body_order],
      body_lin_vel_w=np.zeros((CLIP_FRAMES, num_bodies, 3))[:, body_order],
      body_ang_vel_w=np.zeros((CLIP_FRAMES, num_bodies, 3))[:, body_order],
    )
  return motion_dir


@pytest.fixture(scope="module")
def motion_dir(n3_robot, tmp_path_factory) -> Path:
  _, model = n3_robot
  return _synthetic_motion_dir(tmp_path_factory.mktemp("n3_mimic_motions"), model)


def test_motion_loader_remaps_shuffled_names(n3_robot, motion_dir):
  """N3 npz ordering differs from the MJCF; remap must follow names, not slots."""
  robot, model = n3_robot
  loader = MotionLoader(
    str(motion_dir),
    joint_names=tuple(robot.joint_names),
    body_names=N3_MIMIC_BODY_NAMES,
  )
  assert loader.time_step_total == 2 * CLIP_FRAMES
  assert loader.motion_lengths.tolist() == [CLIP_FRAMES, CLIP_FRAMES]
  assert loader.motion_start_steps.tolist() == [0, CLIP_FRAMES]
  assert loader.body_pos_w.shape == (2 * CLIP_FRAMES, len(N3_MIMIC_BODY_NAMES), 3)

  joint_pos0, body_pos0, _ = _keyframe_state(model)
  # Clip 0 frame 0 has no wiggle: rows must equal the keyframe in MJCF order.
  np.testing.assert_allclose(loader.joint_pos[0].numpy(), joint_pos0, atol=1e-6)
  # Body axis follows N3_MIMIC_BODY_NAMES (base_link first).
  base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
  np.testing.assert_allclose(
    loader.body_pos_w[0, 0].numpy(), body_pos0[base_id - 1], atol=1e-6
  )

  with pytest.raises(ValueError, match="missing entry"):
    MotionLoader(
      str(motion_dir),
      joint_names=tuple(robot.joint_names),
      body_names=("nonexistent_link",),
    )


def test_n3_mimic_task_cfg():
  assert load_runner_cls(TASK) is MimicHimOnPolicyRunner
  cfg = load_env_cfg(TASK)
  play = load_env_cfg(TASK, play=True)

  assert cfg.sim.mujoco.timestep * cfg.decimation == 0.02
  assert cfg.scene.num_envs == 4096
  assert cfg.observations["actor"].history_length == 5
  assert not cfg.observations["actor"].flatten_history_dim
  assert cfg.terminations["motion_end"].time_out

  sensors = {s.name: s for s in cfg.scene.sensors}
  assert isinstance(sensors["feet_contact"], ContactSensorCfg)
  assert sensors["feet_contact"].history_length == 3
  assert sensors["self_collision"].primary.mode == "geom"

  # Reward weight spot checks against the Isaac table.
  assert cfg.rewards["motion_body_pos"].weight == 3.0
  assert cfg.rewards["motion_feet_air_contact"].weight == -2.0
  assert cfg.rewards["joint_vel_limit"].weight == -10.0
  assert cfg.rewards["action_rate_l2"].weight == -0.05

  assert play.observations["actor"].enable_corruption is False
  assert "push_robot" not in play.events
  motion_cmd = play.commands["motion"]
  assert motion_cmd.sampling_mode == "start"
  assert motion_cmd.debug_vis

  # 默认单动作训练（每个动作单独训一个策略）；目录模式（多片段混训）仍受支持。
  motion_file = REPO_ROOT / cfg.commands["motion"].motion_file
  assert motion_file.is_file()
  # 32 个高动态动作 + n3_起身_50hz（AMP recovery json 转换）。
  assert len(list(motion_file.parent.glob("*.npz"))) == 33

  agent = load_rl_cfg(TASK)
  assert isinstance(agent, MhaHimPpoRunnerCfg)
  assert agent.policy.command_dim == 58
  assert agent.policy.attention_nhead == 4
  assert agent.algorithm.entropy_coef == 0.001
  assert agent.algorithm.learning_rate == 5.0e-4
  assert agent.clip_actions == 18.0
  assert agent.logger == "tensorboard"


def test_n3_mimic_obs_layout_and_motion_end(motion_dir):
  """Pin the HIM-critical layout: command block and critic[154:157]."""
  cfg = n3_mimic_env_cfg()
  cfg.scene.num_envs = 4
  cfg.scene.extent = 1.0
  cfg.commands["motion"].motion_file = str(motion_dir)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (4, 5, 154)
    assert obs["critic"].shape == (4, 349)

    command = env.command_manager.get_term("motion").command
    torch.testing.assert_close(obs["actor"][:, -1, :58], command)
    torch.testing.assert_close(obs["critic"][:, :58], command)
    lin_vel = env.scene["robot"].data.root_link_lin_vel_b
    torch.testing.assert_close(obs["critic"][:, 154:157], lin_vel)

    # 12-frame clips: motion_end must truncate every env and count as time-out.
    saw_motion_end = torch.zeros(4, dtype=torch.bool)
    saw_time_out = torch.zeros(4, dtype=torch.bool)
    for _ in range(3 * CLIP_FRAMES):
      obs, _, _, truncated, _ = env.step(torch.zeros(4, 29))
      saw_motion_end |= env.termination_manager.get_term("motion_end").cpu()
      saw_time_out |= truncated.cpu()
      if saw_motion_end.all():
        break
    assert saw_motion_end.all()
    assert torch.equal(saw_motion_end, saw_time_out)
  finally:
    env.close()


@pytest.mark.slow
def test_n3_mimic_training_step(motion_dir, tmp_path: Path):
  """One learn() iteration through the MHA+HIM stack + play-style reload."""
  cfg = n3_mimic_env_cfg()
  cfg.scene.num_envs = 4
  cfg.scene.extent = 1.0
  cfg.episode_length_s = 1.0
  cfg.commands["motion"].motion_file = str(motion_dir)
  agent = load_rl_cfg(TASK)
  assert isinstance(agent, MhaHimPpoRunnerCfg)
  agent.num_steps_per_env = 4
  agent.algorithm.num_learning_epochs = 1
  agent.algorithm.num_mini_batches = 2  # minibatch of 8 samples

  previous_threads = torch.get_num_threads()
  torch.set_num_threads(2)
  env = None
  runner = None
  try:
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    # train.py hands the runner an RslRlVecEnvWrapper; it must re-wrap.
    runner = MimicHimOnPolicyRunner(
      RslRlVecEnvWrapper(env), asdict(agent), str(tmp_path), "cpu"
    )
    obs = runner.env.get_observations()
    assert obs["actor"].shape == (4, 770)  # wrapper flattens (4, 5, 154)
    assert obs["critic"].shape == (4, 349)

    before = next(runner.alg.policy.actor.parameters()).detach().clone()
    runner.learn(1)
    after = next(runner.alg.policy.actor.parameters()).detach()
    assert not torch.equal(before, after)
    assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
    checkpoint = tmp_path / "model_0.pt"
    assert checkpoint.is_file()
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert "estimator_state_dict" in saved
  finally:
    if runner is not None and runner.writer is not None:
      runner.writer.close()
    if env is not None:
      env.close()
    torch.set_num_threads(previous_threads)

  play_cfg = n3_mimic_env_cfg(play=True)
  play_cfg.scene.num_envs = 4
  play_cfg.scene.extent = 1.0
  play_cfg.commands["motion"].motion_file = str(motion_dir)
  env = None
  try:
    env = ManagerBasedRlEnv(cfg=play_cfg, device="cpu")
    restored = MimicHimOnPolicyRunner(env, asdict(agent), device="cpu")
    restored.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu"
    )
    with torch.inference_mode():
      actions = restored.get_inference_policy()(restored.env.get_observations())
    assert actions.shape == (4, 29)
    assert torch.isfinite(actions).all()
  finally:
    env.close()
