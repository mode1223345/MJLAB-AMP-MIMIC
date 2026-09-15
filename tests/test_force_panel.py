"""Tests for the Viser force panel helpers."""

from unittest.mock import Mock

from mjlab.viewer.viser.force_panel import (
  ForcePanel,
  _fmt,
  _resolve_robot_entity,
  contact_display_name,
)


def test_fmt_scales():
  assert _fmt(0.0) == "0.00"
  assert _fmt(12.345) == "12.35"
  assert "e" in _fmt(1e-4)


def test_contact_display_name_feet():
  assert contact_display_name("l_ankle_roll_link") == "L foot↔terrain |F|"
  assert contact_display_name("r_ankle_roll_link") == "R foot↔terrain |F|"


def test_resolve_robot_prefers_robot_key():
  robot = Mock()
  robot.joint_names = ("a",)
  other = Mock()
  other.joint_names = ("b",)
  scene = Mock()
  scene.entities = {"other": other, "robot": robot}
  assert _resolve_robot_entity(scene) is robot


def test_build_markup_contains_sections():
  server = Mock()
  folder = Mock()
  folder.__enter__ = Mock(return_value=None)
  folder.__exit__ = Mock(return_value=False)
  server.gui.add_folder.return_value = folder
  server.gui.add_checkbox.return_value = Mock(value=False)
  html_handle = Mock()
  server.gui.add_html.return_value = html_handle

  env = Mock()
  env.unwrapped.scene.entities = {}
  env.unwrapped.scene.sensors = {}
  panel = ForcePanel(server, env, get_env_idx=lambda: 0)
  panel._show_contact_xyz.value = True
  markup = panel._build_markup(
    [("l_hip_pitch_joint", 1.5), ("r_hip_pitch_joint", -0.5)],
    [("feet_ground_contact", "L foot↔terrain |F|", 120.0, (1.0, 2.0, 119.0))],
  )
  assert "Joint torques" in markup
  assert "l_hip_pitch_joint" in markup
  assert "Contact forces" in markup
  assert "L foot↔terrain |F|" in markup
  assert "ankle_roll_link" in markup  # explanatory note
  assert "119.00" in markup
