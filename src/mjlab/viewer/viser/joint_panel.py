"""Live joint action / position / velocity panel for the Viser viewer."""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import viser

from mjlab.viewer.viser.force_panel import _fmt, _resolve_robot_entity


def action_series_name(joint_name: str) -> str:
  return f"a:{joint_name}"


def pos_series_name(joint_name: str) -> str:
  return f"q:{joint_name}"


def vel_series_name(joint_name: str) -> str:
  return f"dq:{joint_name}"


class JointStatePanel:
  """HTML panel listing per-joint policy action, position, and velocity."""

  def __init__(
    self,
    server: viser.ViserServer,
    env: Any,
    get_env_idx: Callable[[], int],
    *,
    leg_joint_pattern: str = r".*(hip|knee|ankle|waist).*",
  ) -> None:
    self._server = server
    self._env = env
    self._get_env_idx = get_env_idx
    self._leg_re = re.compile(leg_joint_pattern)
    self._joint_names = self._discover_joints()
    self.joint_names = list(self._joint_names)
    self.has_matching_action = self._action_dim_matches(len(self._joint_names))
    self.term_names = []
    for jn in self._joint_names:
      if self.has_matching_action:
        self.term_names.append(action_series_name(jn))
      self.term_names.append(pos_series_name(jn))
      self.term_names.append(vel_series_name(jn))
    self.last_terms: list[tuple[str, np.ndarray]] = []

    with server.gui.add_folder("Display", expand_by_default=True):
      self._legs_only = server.gui.add_checkbox(
        "Legs / waist only",
        initial_value=True,
      )
      self._show_action = server.gui.add_checkbox(
        "Show action",
        initial_value=self.has_matching_action,
      )
      self._show_pos = server.gui.add_checkbox("Show pos", initial_value=True)
      self._show_vel = server.gui.add_checkbox("Show vel", initial_value=True)

    self._html = server.gui.add_html("")
    self._render_empty("Waiting for data…")

  def _discover_joints(self) -> list[str]:
    env = self._env.unwrapped
    robot = _resolve_robot_entity(env.scene)
    return list(robot.joint_names) if robot is not None else []

  def _action_dim_matches(self, n_joints: int) -> bool:
    action_manager = getattr(self._env.unwrapped, "action_manager", None)
    if action_manager is None or n_joints == 0:
      return False
    return int(action_manager.total_action_dim) == n_joints

  def update(self) -> None:
    """Read sim / action state for the selected env and refresh the panel."""
    env = self._env.unwrapped
    env_idx = int(self._get_env_idx())
    robot = _resolve_robot_entity(env.scene)
    if robot is None:
      self._render_empty("No articulated robot entity found.")
      self.last_terms = []
      return

    joint_names = list(robot.joint_names)
    n_joints = len(joint_names)
    q = robot.data.joint_pos[env_idx].detach()
    dq = robot.data.joint_vel[env_idx].detach()
    if q.ndim != 1 or q.shape[0] != n_joints:
      self._render_empty(f"Pos dim mismatch: joints={n_joints}, q={tuple(q.shape)}")
      self.last_terms = []
      return
    if dq.ndim != 1 or dq.shape[0] != n_joints:
      self._render_empty(f"Vel dim mismatch: joints={n_joints}, dq={tuple(dq.shape)}")
      self.last_terms = []
      return

    actions = self._read_actions(env, env_idx, n_joints)

    rows_all = [
      (
        name,
        None if actions is None else float(actions[i]),
        float(q[i].item()),
        float(dq[i].item()),
      )
      for i, name in enumerate(joint_names)
    ]
    if self._legs_only.value:
      rows_html = [r for r in rows_all if self._leg_re.search(r[0])]
    else:
      rows_html = rows_all

    self._html.content = self._build_markup(rows_html, has_action=actions is not None)

    terms: list[tuple[str, np.ndarray]] = []
    for name, act, pos, vel in rows_all:
      if self.has_matching_action:
        terms.append(
          (
            action_series_name(name),
            np.asarray([0.0 if act is None else act], dtype=np.float64),
          )
        )
      terms.append((pos_series_name(name), np.asarray([pos], dtype=np.float64)))
      terms.append((vel_series_name(name), np.asarray([vel], dtype=np.float64)))
    self.last_terms = terms

  def cleanup(self) -> None:
    self._html.remove()

  def _read_actions(self, env: Any, env_idx: int, n_joints: int) -> np.ndarray | None:
    """Return per-joint raw policy actions, or None if unavailable / mismatched."""
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
      return None
    action = action_manager.action[env_idx].detach()
    if action.ndim != 1:
      return None
    if int(action.shape[0]) != n_joints:
      # Still plot pos/vel; action column omitted when dims disagree.
      return None
    return action.cpu().numpy().astype(np.float64, copy=False)

  def _render_empty(self, msg: str) -> None:
    safe = html.escape(msg)
    self._html.content = (
      f'<div style="padding:0.5em;color:#999;font-size:0.85em;">{safe}</div>'
    )

  def _build_markup(
    self,
    rows: list[tuple[str, float | None, float, float]],
    *,
    has_action: bool,
  ) -> str:
    show_a = self._show_action.value and has_action
    show_q = self._show_pos.value
    show_dq = self._show_vel.value
    parts: list[str] = [
      '<div style="padding:0.3em 0.5em;font-family:monospace;font-size:0.8em;">',
      self._section_title(f"Joint state · {len(rows)}"),
      self._muted(
        "a = raw policy action (action_manager.action); "
        "q = joint position (rad); dq = joint velocity (rad/s)."
      ),
    ]
    if not has_action:
      parts.append(
        self._muted(
          "Action omitted: action dim does not match joint count "
          "(or no action_manager)."
        )
      )
    if not rows:
      parts.append(self._muted("No joints match the current filter."))
    else:
      headers = ["joint"]
      if show_a:
        headers.append("a")
      if show_q:
        headers.append("q")
      if show_dq:
        headers.append("dq")
      parts.append(self._table_header(headers))
      for name, act, pos, vel in rows:
        cells = [html.escape(name)]
        if show_a:
          cells.append(_fmt(0.0 if act is None else act))
        if show_q:
          cells.append(_fmt(pos))
        if show_dq:
          cells.append(_fmt(vel))
        parts.append(self._table_row(cells, emphasize_first=True))
      parts.append("</tbody></table>")
    parts.append("</div>")
    return "".join(parts)

  @staticmethod
  def _section_title(text: str) -> str:
    safe = html.escape(text)
    return (
      f'<div style="margin:0.6em 0 0.25em;font-weight:600;color:#222;'
      f'border-bottom:1px solid #bbb;padding-bottom:2px;">{safe}</div>'
    )

  @staticmethod
  def _muted(text: str) -> str:
    return (
      f'<div style="color:#555;font-size:0.9em;margin:0.2em 0;">'
      f"{html.escape(text)}</div>"
    )

  @staticmethod
  def _table_header(cols: list[str]) -> str:
    cells = "".join(
      f'<th style="text-align:{"left" if i == 0 else "right"};'
      f'padding:2px 6px;color:#444;border-bottom:1px solid #ccc;">'
      f"{html.escape(c)}</th>"
      for i, c in enumerate(cols)
    )
    return (
      f'<table style="width:100%;border-collapse:collapse;margin-top:0.3em;">'
      f"<thead><tr>{cells}</tr></thead><tbody>"
    )

  @staticmethod
  def _table_row(cells: list[str], *, emphasize_first: bool) -> str:
    tds: list[str] = []
    for i, c in enumerate(cells):
      weight = "600" if emphasize_first and i == 0 else "400"
      align = "left" if i == 0 else "right"
      color = "#222" if i == 0 else "#111"
      tds.append(
        f'<td style="text-align:{align};padding:2px 6px;font-weight:{weight};'
        f"color:{color};white-space:nowrap;overflow:hidden;"
        f'text-overflow:ellipsis;max-width:180px;">{c}</td>'
      )
    return f"<tr>{''.join(tds)}</tr>"
