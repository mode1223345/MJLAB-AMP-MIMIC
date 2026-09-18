"""Live termination status panel for the Viser viewer."""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

import viser

from mjlab.viewer.viser.termination_diagnostics import (
  term_error_lower_is_bad,
  term_error_value,
  term_threshold_value,
)


def _bool_badge(
  active: bool, *, true_label: str = "YES", false_label: str = "no"
) -> str:
  if active:
    return (
      f'<span style="color:#dc2626;font-weight:600;">{html.escape(true_label)}</span>'
    )
  return f'<span style="color:#6b7280;">{html.escape(false_label)}</span>'


def _term_type_badge(is_timeout: bool) -> str:
  label = "timeout" if is_timeout else "failure"
  color = "#2563eb" if is_timeout else "#b45309"
  return f'<span style="color:{color};font-size:0.85em;">{html.escape(label)}</span>'


def _fmt_error(
  error: float | None, threshold: float | None, *, lower_is_bad: bool
) -> str:
  if error is None:
    return '<span style="color:#9ca3af;">—</span>'
  text = f"{error:.3f}"
  if threshold is not None:
    violated = error < threshold if lower_is_bad else error > threshold
    if violated:
      return f'<span style="color:#dc2626;font-weight:600;">{html.escape(text)}</span>'
  return html.escape(text)


def _fmt_threshold(
  threshold: float | None, *, is_timeout: bool, lower_is_bad: bool
) -> str:
  if is_timeout and threshold is not None:
    return html.escape(f"≥ {int(threshold)}")
  if threshold is None:
    return '<span style="color:#9ca3af;">—</span>'
  sign = "<" if lower_is_bad else ">"
  return html.escape(f"{sign} {threshold:.3f}")


class TerminationPanel:
  """HTML panel listing episode termination state for the selected env."""

  def __init__(
    self,
    server: viser.ViserServer,
    env: Any,
    get_env_idx: Callable[[], int],
  ) -> None:
    self._server = server
    self._env = env
    self._get_env_idx = get_env_idx
    self.term_names: list[str] = []
    self.last_terms: list[tuple[str, float]] = []
    self._html = server.gui.add_html("")
    self._render_empty("Waiting for data…")

  def update(self) -> None:
    """Refresh panel from the selected environment's termination manager."""
    env = self._env.unwrapped
    if not hasattr(env, "termination_manager"):
      self._render_empty("No termination manager on this environment.")
      self.term_names = []
      self.last_terms = []
      return

    tm = env.termination_manager
    env_idx = int(self._get_env_idx())
    self.term_names = list(tm.active_terms)
    self.last_terms = [
      (name, float(vals[0])) for name, vals in tm.get_active_iterable_terms(env_idx)
    ]

    ep_len = int(env.episode_length_buf[env_idx].item())
    max_len = int(env.max_episode_length)
    terminated = bool(tm.terminated[env_idx].item())
    truncated = bool(tm.time_outs[env_idx].item())

    rows: list[tuple[str, bool, bool, bool, float | None, float | None, bool]] = []
    for name in tm.active_terms:
      cfg = tm.get_term_cfg(name)
      current = bool(tm.get_term(name)[env_idx].item())
      last = bool(tm.get_last_episode_term(name)[env_idx].item())
      error = term_error_value(env, cfg, env_idx)
      threshold = term_threshold_value(cfg)
      rows.append(
        (
          name,
          cfg.time_out,
          current,
          last,
          error,
          threshold,
          term_error_lower_is_bad(cfg),
        )
      )

    self._html.content = self._build_markup(
      ep_len=ep_len,
      max_len=max_len,
      terminated=terminated,
      truncated=truncated,
      rows=rows,
    )

  def cleanup(self) -> None:
    self._html.remove()

  def _render_empty(self, message: str) -> None:
    self._html.content = (
      '<div style="padding:0.5em;color:#555;font-size:0.85em;">'
      f"{html.escape(message)}</div>"
    )

  def _build_markup(
    self,
    *,
    ep_len: int,
    max_len: int,
    terminated: bool,
    truncated: bool,
    rows: list[tuple[str, bool, bool, bool, float | None, float | None, bool]],
  ) -> str:
    """Build HTML for episode summary and per-term status."""
    header = (
      '<div style="padding:0.5em;font-size:0.85em;line-height:1.5;">'
      "<strong>Episode</strong><br/>"
      f"Step {ep_len} / {max_len}<br/>"
      f"Terminated (failure): {_bool_badge(terminated)}<br/>"
      f"Truncated (timeout): {_bool_badge(truncated)}"
      "</div>"
    )

    if not rows:
      return header + (
        '<div style="padding:0.5em;color:#555;font-size:0.85em;">'
        "No active termination terms.</div>"
      )

    table_rows = []
    for name, is_timeout, current, last, error, threshold, lower_is_bad in rows:
      display_error = error
      display_threshold = threshold
      if is_timeout:
        display_error = float(ep_len)
        display_threshold = float(max_len)
      error_html = _fmt_error(
        display_error, display_threshold, lower_is_bad=lower_is_bad
      )
      limit_html = _fmt_threshold(
        display_threshold, is_timeout=is_timeout, lower_is_bad=lower_is_bad
      )
      table_rows.append(
        "<tr>"
        f'<td style="padding:2px 6px;">{html.escape(name)}</td>'
        f'<td style="padding:2px 6px;">{_term_type_badge(is_timeout)}</td>'
        f'<td style="padding:2px 6px;text-align:right;">{error_html}</td>'
        f'<td style="padding:2px 6px;text-align:right;">{limit_html}</td>'
        f'<td style="padding:2px 6px;text-align:center;">'
        f"{_bool_badge(current)}</td>"
        f'<td style="padding:2px 6px;text-align:center;">'
        f"{_bool_badge(last, true_label='ended', false_label='—')}</td>"
        "</tr>"
      )

    table = (
      '<table style="width:100%;border-collapse:collapse;font-size:0.85em;'
      'margin-top:0.25em;">'
      "<thead><tr>"
      '<th style="text-align:left;padding:2px 6px;">Term</th>'
      '<th style="text-align:left;padding:2px 6px;">Type</th>'
      '<th style="text-align:right;padding:2px 6px;">Error</th>'
      '<th style="text-align:right;padding:2px 6px;">Limit</th>'
      '<th style="text-align:center;padding:2px 6px;">Now</th>'
      '<th style="text-align:center;padding:2px 6px;">Last end</th>'
      "</tr></thead>"
      f"<tbody>{''.join(table_rows)}</tbody></table>"
      '<div style="padding:0.5em 0.25em;color:#6b7280;font-size:0.8em;">'
      "Error = raw metric (root height in m, tilt in rad). Limit = termination "
      "threshold; a violated error is red. Now = triggered this step. "
      "Last end = reason when this env last reset (— if it never has).</div>"
    )
    return header + table
