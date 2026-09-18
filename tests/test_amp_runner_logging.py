"""Finite-value guards and task/style logging in the AMP runner."""

from unittest.mock import MagicMock, Mock

import torch

from mjlab.tasks.amp.rl.discriminator import Discriminator
from mjlab.tasks.amp.rl.normalizer import Normalizer
from mjlab.tasks.amp.rl.runner import (
  AmpHimOnPolicyRunner,
  _finite_mean,
  _safe_ratio,
)


def test_finite_mean_skips_non_finite_and_unconvertible():
  assert _finite_mean([]) == 0.0
  assert _finite_mean([float("nan"), float("inf"), None, "x"]) == 0.0
  assert _finite_mean([1.0, float("nan"), float("inf"), None, "x", 3.0]) == 2.0


def test_safe_ratio_guards_bad_denominators():
  assert _safe_ratio(3.0, 4.0) == 0.75
  assert _safe_ratio(1.0, 0.0) == 0.0
  assert _safe_ratio(0.0, 0.0) == 0.0
  assert _safe_ratio(float("inf"), 1.0) == 0.0
  assert _safe_ratio(1.0, float("nan")) == 0.0
  assert _safe_ratio(1.0, 1e-12) == 0.0
  assert _safe_ratio(1.0, 0.0, default=-1.0) == -1.0


def test_add_scalar_normalizes_and_drops_bad_values():
  runner = object.__new__(AmpHimOnPolicyRunner)
  runner.writer = Mock()
  runner.logger_type = "tensorboard"

  runner._add_scalar("one", torch.tensor(2.5), 0)
  runner._add_scalar("many", torch.tensor([1.0, 3.0]), 0)
  runner._add_scalar("nan", torch.tensor(float("nan")), 0)
  runner._add_scalar("text", "abc", 0)

  written = {
    call.args[0]: call.args[1] for call in runner.writer.add_scalar.call_args_list
  }
  assert written == {"one": 2.5, "many": 2.0, "nan": 0.0}


def test_log_emits_task_and_style_scalars():
  runner = object.__new__(AmpHimOnPolicyRunner)
  runner.writer = Mock()
  runner.logger_type = "tensorboard"
  runner.device = "cpu"
  runner.gpu_world_size = 1
  runner.tot_timesteps = 0
  runner.tot_time = 0.0
  runner.num_steps_per_env = 4
  runner.env = Mock(num_envs=2)
  runner.alg = MagicMock()
  runner.alg.policy.action_std = torch.tensor([0.5])

  runner.log(
    {
      "it": 0,
      "tot_iter": 1,
      "collection_time": 0.1,
      "learn_time": 0.1,
      "ep_infos": [],
      "loss_dict": {},
      "rewbuffer": [10.0, 20.0],
      "srewbuffer": [4.0, 6.0],
      "trewbuffer": [15.0],
      "lenbuffer": [100.0, 200.0],
    }
  )

  tags = {
    call.args[0]: call.args[1] for call in runner.writer.add_scalar.call_args_list
  }
  assert tags["Train/mean_task_reward"] == 15.0
  assert tags["Train/mean_reward"] == 15.0
  assert tags["Train/mean_style_reward"] == 5.0
  assert tags["Train/style_reward_fraction"] == 1 / 3


def test_log_falls_back_to_reward_minus_style_without_trewbuffer():
  runner = object.__new__(AmpHimOnPolicyRunner)
  runner.writer = Mock()
  runner.logger_type = "tensorboard"
  runner.device = "cpu"
  runner.gpu_world_size = 1
  runner.tot_timesteps = 0
  runner.tot_time = 0.0
  runner.num_steps_per_env = 4
  runner.env = Mock(num_envs=1)
  runner.alg = MagicMock()
  runner.alg.policy.action_std = torch.tensor([0.5])

  runner.log(
    {
      "it": 0,
      "tot_iter": 1,
      "collection_time": 0.1,
      "learn_time": 0.1,
      "ep_infos": [],
      "loss_dict": {},
      "rewbuffer": [10.0],
      "srewbuffer": [4.0],
      "lenbuffer": [100.0],
    }
  )

  tags = {
    call.args[0]: call.args[1] for call in runner.writer.add_scalar.call_args_list
  }
  assert tags["Train/mean_task_reward"] == 6.0
  assert tags["Train/style_reward_fraction"] == 0.4


def _make_discriminator(style_reward_normalizer):
  disc = Discriminator(
    observation_dim=8,
    observation_horizon=2,
    device="cpu",
    reward_coef=0.8,
    reward_lerp=0.7,
    shape=(16, 8),
    style_reward_function="wasserstein_mapping",
  )
  return disc


def test_predict_amp_reward_returns_task_and_style_split():
  normalizer = Normalizer(1)
  disc = _make_discriminator(normalizer)
  state_buf = torch.randn(4, 2, 8)
  task = torch.full((4,), 2.0)

  reward, style, task_out = disc.predict_amp_reward(state_buf, task, None, normalizer)

  assert reward.shape == style.shape == task_out.shape == (4,)
  assert torch.isfinite(reward).all()
  assert torch.allclose(task_out, task * disc.reward_lerp)
  assert torch.allclose(reward, style + task_out)
  assert disc.training


def test_predict_amp_reward_can_skip_normalizer_update():
  normalizer = Normalizer(1)
  disc = _make_discriminator(normalizer)
  state_buf = torch.randn(4, 2, 8)
  task = torch.ones(4)
  before = normalizer.count

  disc.predict_amp_reward(state_buf, task, None, normalizer)
  assert normalizer.count > before

  frozen = normalizer.count
  disc.predict_amp_reward(
    state_buf, task, None, normalizer, update_style_normalizer=False
  )
  assert normalizer.count == frozen


def test_predict_amp_reward_zeroes_non_finite_rewards():
  disc = _make_discriminator(None)
  state_buf = torch.randn(4, 2, 8)

  reward, style, task_out = disc.predict_amp_reward(
    state_buf, torch.full((4,), float("nan")), None, None
  )

  assert torch.isfinite(reward).all()
  assert (task_out == 0.0).all()
  assert torch.allclose(reward, style)


def test_predict_amp_reward_preserves_eval_mode():
  disc = _make_discriminator(None)
  disc.eval()

  disc.predict_amp_reward(torch.randn(2, 2, 8), torch.ones(2), None, None)

  assert not disc.training
