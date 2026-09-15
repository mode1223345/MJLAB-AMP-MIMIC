"""Tests for the Viser joint state panel helpers."""

from unittest.mock import Mock

from mjlab.viewer.viser.joint_panel import (
  JointStatePanel,
  action_series_name,
  pos_series_name,
  vel_series_name,
)


def test_series_name_helpers():
  assert action_series_name("l_leg_knee_pitch_joint") == "a:l_leg_knee_pitch_joint"
  assert pos_series_name("l_leg_knee_pitch_joint") == "q:l_leg_knee_pitch_joint"
  assert vel_series_name("l_leg_knee_pitch_joint") == "dq:l_leg_knee_pitch_joint"


def test_build_markup_table_columns():
  server = Mock()
  folder = Mock()
  folder.__enter__ = Mock(return_value=None)
  folder.__exit__ = Mock(return_value=False)
  server.gui.add_folder.return_value = folder
  server.gui.add_checkbox.return_value = Mock(value=True)
  html_handle = Mock()
  server.gui.add_html.return_value = html_handle

  env = Mock()
  env.unwrapped.scene.entities = {}
  panel = JointStatePanel(server, env, get_env_idx=lambda: 0)
  panel._show_action.value = True
  panel._show_pos.value = True
  panel._show_vel.value = True
  markup = panel._build_markup(
    [("l_leg_knee_pitch_joint", 0.1, -0.2, 1.5)],
    has_action=True,
  )
  assert "Joint state" in markup
  assert "l_leg_knee_pitch_joint" in markup
  assert ">a<" in markup or ">a</th>" in markup
  assert "0.10" in markup
  assert "-0.20" in markup
  assert "1.50" in markup
