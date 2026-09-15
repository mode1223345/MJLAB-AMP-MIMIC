"""Tests for the Viser foot / ground friction panel helpers."""

from unittest.mock import MagicMock, Mock, patch

from mjlab.viewer.viser.foot_friction_panel import (
  FootFrictionPanel,
  _resolve_global_geom_ids,
  apply_foot_ground_friction,
)


def test_resolve_geom_ids_uses_robot_and_terrain():
  robot = Mock()
  robot.geom_names = ("l_foot_collision", "r_foot_collision", "other")
  robot.find_geoms.return_value = ([0, 1], ["l_foot_collision", "r_foot_collision"])
  robot.indexing.geom_ids = MagicMock()
  robot.indexing.geom_ids.__getitem__ = Mock(
    side_effect=lambda i: MagicMock(item=lambda: 10 + i)
  )

  mj = Mock()
  mj.ngeom = 3

  def _id2name(_m, _obj, gid):
    return {0: "terrain", 1: "other", 2: "x"}[gid]

  env = Mock()
  env.unwrapped = env
  env.scene.entities = {"robot": robot}
  env.sim.mj_model = mj

  with patch(
    "mjlab.viewer.viser.foot_friction_panel.mujoco.mj_id2name",
    side_effect=_id2name,
  ):
    ids = _resolve_global_geom_ids(
      env,
      foot_geom_patterns=(".*_foot_collision",),
      include_terrain=True,
    )
  assert ids == [0, 10, 11]


def test_apply_sets_friction_axis0():
  sim = Mock()
  sim.expanded_fields = set()
  sim.expand_model_fields = Mock()
  friction = MagicMock()
  sim.model.geom_friction = friction

  env = Mock()
  env.unwrapped = env
  env.sim = sim

  with patch(
    "mjlab.viewer.viser.foot_friction_panel._resolve_global_geom_ids",
    return_value=[4, 5],
  ):
    n = apply_foot_ground_friction(env, friction=0.7, all_envs=True)
  assert n == 2
  sim.expand_model_fields.assert_called_once_with(("geom_friction",))
  assert friction.__setitem__.call_count == 2


def test_panel_queues_foot_friction_action():
  server = Mock()
  folder = Mock()
  folder.__enter__ = Mock(return_value=None)
  folder.__exit__ = Mock(return_value=False)
  server.gui.add_folder.return_value = folder
  # sliders / checkbox / button / markdown
  slider = Mock(value=1.0, min=0.3, max=1.6)
  server.gui.add_slider.return_value = slider
  server.gui.add_checkbox.return_value = Mock(value=True)
  server.gui.add_button.return_value = Mock()
  server.gui.add_markdown.return_value = Mock()

  env = Mock()
  env.unwrapped.scene.entities = {}
  actions: list = []

  with patch.object(FootFrictionPanel, "_nominal_friction", return_value=1.0):
    panel = FootFrictionPanel(
      server,
      env,
      get_env_idx=lambda: 0,
      request_action=lambda kind, payload: actions.append((kind, payload)),
    )
    panel._queue_apply()

  assert actions
  assert actions[-1][0] == "custom"
  assert actions[-1][1]["type"] == "foot_friction"
  assert "friction" in actions[-1][1]
