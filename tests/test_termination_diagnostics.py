"""Termination diagnostics and the last-episode reason snapshot."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mjlab.envs.mdp.terminations import (
  bad_orientation,
  root_height_below_minimum,
  time_out,
)
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlab.tasks.amp.mdp.terminations import DelayedTerminationManager
from mjlab.viewer.viser.termination_diagnostics import (
  term_error_lower_is_bad,
  term_error_value,
  term_threshold_value,
)
from mjlab.viewer.viser.termination_panel import TerminationPanel


class _FakeEnv:
  """Minimal env: what TerminationManager and the panel actually touch."""

  def __init__(self, num_envs: int = 2, root_z: float = 0.5, gravity_z: float = -1.0):
    height = torch.full((num_envs,), root_z)
    gravity = torch.zeros(num_envs, 3)
    gravity[:, 2] = gravity_z
    robot = SimpleNamespace(
      data=SimpleNamespace(
        root_link_pos_w=torch.stack(
          [torch.zeros(num_envs), torch.zeros(num_envs), height], dim=1
        ),
        projected_gravity_b=gravity,
      )
    )
    self.num_envs = num_envs
    self.device = "cpu"
    self.scene = {"robot": robot}
    self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    self.max_episode_length = 500


def _flaky(fires: list[bool]):
  """Term func firing on the envs where ``fires`` is True."""

  def func(env):
    return torch.tensor(fires, dtype=torch.bool)

  return func


def _make_manager(env, fires):
  cfg = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "fall_down": TerminationTermCfg(
      func=root_height_below_minimum, params={"minimum_height": 0.4}
    ),
    "bad_orientation": TerminationTermCfg(
      func=bad_orientation, params={"limit_angle": 0.8}
    ),
    "fail": TerminationTermCfg(func=_flaky(fires)),
  }
  return TerminationManager(cfg, env)


def test_last_episode_term_records_only_reset_envs():
  env = _FakeEnv(num_envs=3)
  tm = _make_manager(env, [True, True, False])

  # A step that fires never records on its own: the env has not reset yet.
  tm.compute()
  assert not tm.get_last_episode_term("fail").any()

  tm.reset(env_ids=torch.tensor([0, 1]))
  last = tm.get_last_episode_term("fail")
  assert last.tolist() == [True, True, False]
  # Terms that did not fire stay False for the same episodes.
  assert tm.get_last_episode_term("bad_orientation").tolist() == [False] * 3


def test_reset_logs_termination_share_as_fraction():
  env = _FakeEnv(num_envs=3)
  tm = _make_manager(env, [True, True, False])
  tm.compute()

  # Both episodes that just ended ended by ``fail``: all of the resets.
  extras = tm.reset(env_ids=torch.tensor([0, 1]))
  assert extras["Episode_Termination/fail"] == 1.0
  assert extras["Episode_Termination/time_out"] == 0.0

  # A full reset (``env.reset()``) puts a clean env in the denominator.
  extras = tm.reset(env_ids=None)
  assert extras["Episode_Termination/fail"] == pytest.approx(2 / 3)


def test_reset_attributes_overlapping_terms_to_one_reason():
  env = _FakeEnv(num_envs=3)
  tm = _make_manager(env, [True, True, True])
  env.episode_length_buf[:] = env.max_episode_length  # Every env also times out.
  tm.compute()

  extras = tm.reset(env_ids=None)
  # Overlapping flavor: both terms claim every episode, so it sums past 1.
  assert extras["Episode_Termination_any/time_out"] == 1.0
  assert extras["Episode_Termination_any/fail"] == 1.0
  # Exclusive flavor credits the time out (highest priority) and still sums to 1.
  assert extras["Episode_Termination/time_out"] == 1.0
  assert extras["Episode_Termination/fail"] == 0.0
  exclusive = [v for k, v in extras.items() if k.startswith("Episode_Termination/")]
  assert sum(exclusive) == pytest.approx(1.0)


def test_delayed_termination_keeps_previous_reasons():
  env = _FakeEnv(num_envs=2)
  base = _make_manager(env, [True, False])
  delayed = DelayedTerminationManager(
    base, delay_env_mask=torch.tensor([True, False]), max_delay_steps=2
  )

  # First suppressed step: the env never resets, so nothing is recorded.
  delayed.compute()
  assert not delayed.terminated[0]
  delayed.reset(env_ids=torch.tensor([], dtype=torch.long))
  assert not delayed.get_last_episode_term("fail").any()

  # Second step reaches the limit and releases the reset, which is recorded.
  delayed.compute()
  assert delayed.terminated[0]
  delayed.reset(env_ids=torch.tensor([0]))
  assert delayed.get_last_episode_term("fail").tolist() == [True, False]


def test_term_error_value_reads_the_term_metric():
  env = _FakeEnv(num_envs=2, root_z=0.42, gravity_z=-0.5)
  env.episode_length_buf = torch.tensor([7, 9])
  tm = _make_manager(env, [False, False])

  assert term_error_value(env, tm.get_term_cfg("time_out"), 1) == 9.0
  assert term_error_value(env, tm.get_term_cfg("fall_down"), 0) == pytest.approx(0.42)
  # acos(-(-0.5)) = 60 degrees, in radians to match ``limit_angle``.
  assert term_error_value(env, tm.get_term_cfg("bad_orientation"), 0) == pytest.approx(
    math.pi / 3
  )
  # Unmapped terms fall back to no value.
  assert term_error_value(env, tm.get_term_cfg("fail"), 0) is None


def test_term_thresholds_and_direction():
  env = _FakeEnv()
  tm = _make_manager(env, [False, False])

  assert term_threshold_value(tm.get_term_cfg("fall_down")) == 0.4
  assert term_threshold_value(tm.get_term_cfg("bad_orientation")) == 0.8
  assert term_threshold_value(tm.get_term_cfg("time_out")) == float("inf")
  assert term_threshold_value(tm.get_term_cfg("fail")) is None

  assert term_error_lower_is_bad(tm.get_term_cfg("fall_down"))
  assert not term_error_lower_is_bad(tm.get_term_cfg("bad_orientation"))


def test_panel_marks_the_violated_side():
  env = _FakeEnv(num_envs=1, root_z=0.2)
  tm = _make_manager(env, [False])
  wrapper = SimpleNamespace(
    unwrapped=SimpleNamespace(
      termination_manager=tm,
      episode_length_buf=env.episode_length_buf,
      max_episode_length=env.max_episode_length,
      scene=env.scene,
    )
  )
  panel = TerminationPanel(Mock(), wrapper, get_env_idx=lambda: 0)
  panel.update()

  html = panel._html.content
  # 0.200 m is below the 0.400 m floor: red, and the limit reads "< 0.400".
  assert "#dc2626" in html
  assert "&lt; 0.400" in html
  assert set(dict(panel.last_terms)) == {
    "time_out",
    "fall_down",
    "bad_orientation",
    "fail",
  }


def test_panel_redraws_empty_without_a_manager():
  env = SimpleNamespace(unwrapped=SimpleNamespace())
  panel = TerminationPanel(Mock(), env, get_env_idx=lambda: 0)

  panel.update()

  assert "No termination manager" in panel._html.content
  assert panel.term_names == []
