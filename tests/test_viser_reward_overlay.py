"""AMP style/task terms in the Viser reward bar overlay."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, Mock

import numpy as np

from mjlab.viewer.viser.overlays import ViserTermOverlays
from mjlab.viewer.viser.reward_bar_panel import RewardBarPanel

_NUM_BASE_TERMS = 27


class _DummyEnv:
  def __init__(self, unwrapped):
    self._unwrapped = unwrapped

  @property
  def unwrapped(self):
    return self._unwrapped


def _env(unwrapped):
  return _DummyEnv(unwrapped)


def _unwrapped(**extra):
  reward_manager = Mock()
  reward_manager.get_active_iterable_terms.return_value = [
    (f"term_{i}", [0.0]) for i in range(_NUM_BASE_TERMS)
  ]
  return SimpleNamespace(reward_manager=reward_manager, **extra)


def _overlays(unwrapped, **kwargs):
  scene = cast(Any, SimpleNamespace(env_idx=0))
  return ViserTermOverlays(
    MagicMock(), _env(unwrapped), scene, frame_time=1 / 30, **kwargs
  )


def _bar_panel(overlays: ViserTermOverlays) -> RewardBarPanel:
  """Set up the tabs and narrow the optional panel field."""
  overlays.setup_tabs(MagicMock())
  assert overlays.reward_bar_panel is not None
  return overlays.reward_bar_panel


def test_reward_bar_lists_amp_terms_and_every_reward_term():
  probe = SimpleNamespace(get_amp_reward_panel_terms=lambda idx: [])
  overlays = _overlays(_unwrapped(amp_reward_probe=probe))

  names = _bar_panel(overlays)._term_names
  # ``reward_bar_max_terms`` (20) is a floor: all 27 terms plus the AMP pair.
  assert names[:2] == ["style_reward", "task_reward"]
  assert len(names) == _NUM_BASE_TERMS + 2


def test_reward_bar_without_a_probe_keeps_only_reward_terms():
  overlays = _overlays(_unwrapped())

  names = _bar_panel(overlays)._term_names
  assert names == [f"term_{i}" for i in range(_NUM_BASE_TERMS)]


def test_amp_terms_precede_reward_terms_and_sanitize_values():
  probe = SimpleNamespace(
    get_amp_reward_panel_terms=lambda idx: [
      ("style_reward", [float("nan")]),
      ("task_reward", [3.5]),
    ]
  )
  env = _env(_unwrapped(amp_reward_probe=probe))
  overlays = _overlays(env.unwrapped)
  panel = _bar_panel(overlays)

  overlays.update(paused=False)

  # NaN is zeroed; the AMP entries come first so the bars stay adjacent.
  assert list(panel._histories["style_reward"]) == [0.0]
  assert list(panel._histories["task_reward"]) == [3.5]


def test_amp_terms_are_dropped_when_the_probe_raises():
  def boom(idx):
    raise RuntimeError("no obs buffer")

  probe = SimpleNamespace(get_amp_reward_panel_terms=boom)
  env = _env(_unwrapped(amp_reward_probe=probe))
  overlays = _overlays(env.unwrapped)
  panel = _bar_panel(overlays)

  assert overlays._amp_reward_terms(env, 0) == []
  # The panel still updates the base terms.
  overlays.update(paused=False)
  assert list(panel._histories["term_0"]) == [0.0]


def test_env_switch_clears_amp_history():
  probe = SimpleNamespace(
    get_amp_reward_panel_terms=lambda idx: [("task_reward", np.array([1.0]))]
  )
  overlays = _overlays(_unwrapped(amp_reward_probe=probe))
  panel = _bar_panel(overlays)
  overlays.update(paused=False)
  assert list(panel._histories["task_reward"]) == [1.0]

  overlays.on_env_switch()

  assert list(panel._histories["task_reward"]) == []
