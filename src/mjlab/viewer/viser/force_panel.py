"""Live joint-torque / contact-force numeric panel for the Viser viewer."""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import viser

from mjlab.sensor.contact_sensor import ContactSensor


def _fmt(val: float) -> str:
  if abs(val) >= 1000 or (0.0 < abs(val) < 1e-3):
    return f"{val:.2e}"
  return f"{val:.2f}"


def _resolve_robot_entity(scene: Any) -> Any | None:
  """Prefer ``robot``, else first entity that exposes articulated joints."""
  entities = getattr(scene, "entities", None)
  if not entities:
    return None
  if "robot" in entities:
    return entities["robot"]
  for ent in entities.values():
    if getattr(ent, "joint_names", ()):
      return ent
  return None


def contact_display_name(primary: str) -> str:
  """Human-readable foot/terrain contact label from a primary body name."""
  side = "L" if primary.startswith("l_") else "R" if primary.startswith("r_") else ""
  if "ankle" in primary or "foot" in primary:
    if side:
      return f"{side} foot↔terrain |F|"
    return f"{primary}↔terrain |F|"
  if side:
    return f"{side} {primary} |F|"
  return f"{primary} |F|"


def torque_series_name(joint_name: str) -> str:
  return f"τ:{joint_name}"


class ForcePanel:
  """HTML panel listing joint torques and contact-sensor forces."""

  def __init__(
    self,
    server: viser.ViserServer,
    env: Any,
    get_env_idx: Callable[[], int],
    *,
    leg_joint_pattern: str = r".*(hip|knee|ankle).*",
  ) -> None:
    self._server = server
    self._env = env
    self._get_env_idx = get_env_idx
    self._leg_re = re.compile(leg_joint_pattern)
    self._joint_names, self._contact_meta = self._discover()
    self.term_names = [torque_series_name(n) for n in self._joint_names] + [
      label for _, label, _ in self._contact_meta
    ]
    self.contact_term_names = [label for _, label, _ in self._contact_meta]
    self.last_terms: list[tuple[str, np.ndarray]] = []

    with server.gui.add_folder("Display", expand_by_default=True):
      self._legs_only = server.gui.add_checkbox(
        "Legs only (hip/knee/ankle)",
        initial_value=True,
      )
      self._show_contact_xyz = server.gui.add_checkbox(
        "Contact force XYZ",
        initial_value=True,
      )

    self._html = server.gui.add_html("")
    self._render_empty("Waiting for data…")

  def _discover(
    self,
  ) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Return (joint_names, contact_meta).

    ``contact_meta`` entries are ``(sensor_name, display_label, primary)``.
    """
    env = self._env.unwrapped
    robot = _resolve_robot_entity(env.scene)
    joint_names = list(robot.joint_names) if robot is not None else []
    contacts: list[tuple[str, str, str]] = []
    sensors = getattr(env.scene, "sensors", {}) or {}
    for sensor in sensors.values():
      if not isinstance(sensor, ContactSensor):
        continue
      if "force" not in sensor.cfg.fields:
        continue
      for primary in sensor.primary_names:
        contacts.append((sensor.cfg.name, contact_display_name(primary), primary))
    return joint_names, contacts

  def update(self) -> None:
    """Read sim state for the selected env and refresh the HTML panel."""
    env = self._env.unwrapped
    env_idx = int(self._get_env_idx())
    robot = _resolve_robot_entity(env.scene)
    if robot is None:
      self._render_empty("No articulated robot entity found.")
      self.last_terms = []
      return

    joint_names = list(robot.joint_names)
    tau = robot.data.qfrc_actuator[env_idx].detach()
    if tau.ndim != 1 or tau.shape[0] != len(joint_names):
      self._render_empty(
        f"Torque dim mismatch: joints={len(joint_names)}, qfrc={tuple(tau.shape)}"
      )
      self.last_terms = []
      return

    joint_pairs_all = [(n, float(tau[i].item())) for i, n in enumerate(joint_names)]
    if self._legs_only.value:
      joint_pairs_html = [(n, v) for n, v in joint_pairs_all if self._leg_re.search(n)]
    else:
      joint_pairs_html = joint_pairs_all

    contact_rows = self._collect_contacts(env.scene, env_idx)
    self._html.content = self._build_markup(joint_pairs_html, contact_rows)

    terms: list[tuple[str, np.ndarray]] = [
      (torque_series_name(n), np.asarray([v], dtype=np.float64))
      for n, v in joint_pairs_all
    ]
    for _sensor, label, mag, _fvec in contact_rows:
      terms.append((label, np.asarray([mag], dtype=np.float64)))
    self.last_terms = terms

  def cleanup(self) -> None:
    self._html.remove()

  def _collect_contacts(
    self, scene: Any, env_idx: int
  ) -> list[tuple[str, str, float, tuple[float, float, float]]]:
    """Return ``(sensor_name, display_label, |F|, (fx,fy,fz))`` rows."""
    rows: list[tuple[str, str, float, tuple[float, float, float]]] = []
    sensors = getattr(scene, "sensors", {}) or {}
    for sensor in sensors.values():
      if not isinstance(sensor, ContactSensor):
        continue
      data = sensor.data
      if data.force is None:
        continue
      force = data.force[env_idx].detach()  # [N, 3]
      found = data.found[env_idx].detach() if data.found is not None else None
      names = sensor.primary_names
      n_slots = max(1, int(sensor.cfg.num_slots))
      for p_i, primary in enumerate(names):
        slot = p_i * n_slots
        if slot >= force.shape[0]:
          continue
        if found is not None and float(found[slot].item()) <= 0:
          fvec = (0.0, 0.0, 0.0)
          mag = 0.0
        else:
          fx, fy, fz = (float(x) for x in force[slot].tolist())
          fvec = (fx, fy, fz)
          mag = (fx * fx + fy * fy + fz * fz) ** 0.5
        rows.append((sensor.cfg.name, contact_display_name(primary), mag, fvec))
    return rows

  def _render_empty(self, msg: str) -> None:
    safe = html.escape(msg)
    self._html.content = (
      f'<div style="padding:0.5em;color:#999;font-size:0.85em;">{safe}</div>'
    )

  def _build_markup(
    self,
    joint_pairs: list[tuple[str, float]],
    contacts: list[tuple[str, str, float, tuple[float, float, float]]],
  ) -> str:
    show_xyz = self._show_contact_xyz.value
    parts: list[str] = [
      '<div style="padding:0.3em 0.5em;font-family:monospace;font-size:0.8em;">',
      self._section_title(f"Joint torques (Nm) · {len(joint_pairs)}"),
    ]
    if not joint_pairs:
      parts.append(self._muted("No joints match the current filter."))
    else:
      max_abs = max((abs(v) for _, v in joint_pairs), default=1.0) or 1.0
      for name, val in joint_pairs:
        parts.append(self._bar_row(name, val, max_abs, mode="signed"))

    parts.append(self._section_title(f"Contact forces (N) · {len(contacts)}"))
    parts.append(
      self._muted(
        "Net contact force magnitude between each foot "
        "(ankle_roll_link) and the terrain — not joint torques. "
        "XYZ is the 3D force vector (N)."
      )
    )
    if not contacts:
      parts.append(self._muted("No ContactSensor with force field in scene."))
    else:
      max_f = max((c[2] for c in contacts), default=1.0) or 1.0
      for sensor_name, label, mag, fvec in contacts:
        extra = ""
        if show_xyz:
          extra = (
            f' <span style="color:#aaa;">'
            f"[{_fmt(fvec[0])}, {_fmt(fvec[1])}, {_fmt(fvec[2])}]"
            f"</span>"
          )
        parts.append(
          self._bar_row(
            label,
            mag,
            max_f,
            mode="magnitude",
            title=f"{sensor_name}",
            suffix=extra,
          )
        )

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
  def _bar_row(
    name: str,
    val: float,
    max_abs: float,
    *,
    mode: Literal["signed", "magnitude"],
    title: str | None = None,
    suffix: str = "",
  ) -> str:
    pct = min(100.0, abs(val) / max(max_abs, 1e-12) * 100.0)
    if mode == "magnitude":
      color = "#42a5f5"
    elif val < 0:
      color = "#f44336"
    else:
      color = "#4caf50"
    text_color = "#fff" if pct > 25 else "#ccc"
    val_str = _fmt(val)
    safe_name = html.escape(name, quote=True)
    tip = html.escape(title or name, quote=True)
    return (
      f'<div style="display:flex;align-items:center;margin:2px 0;">'
      f'<span style="min-width:150px;font-size:0.78em;text-align:right;'
      f"padding-right:6px;color:#222;font-weight:600;white-space:nowrap;"
      f'overflow:hidden;text-overflow:ellipsis;" title="{tip}">{safe_name}</span>'
      f'<div style="flex:1;background:#333;border-radius:3px;height:18px;'
      f'position:relative;overflow:hidden;">'
      f'<div style="width:{pct:.1f}%;height:100%;background:{color};'
      f'border-radius:3px;"></div>'
      f'<span style="position:absolute;left:4px;top:0;line-height:18px;'
      f'font-size:0.72em;color:{text_color};">{val_str}{suffix}</span>'
      f"</div></div>"
    )
