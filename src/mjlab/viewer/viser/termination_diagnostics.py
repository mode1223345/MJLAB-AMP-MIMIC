"""Compute raw termination errors for Viser diagnostics.

Maps the termination terms this repo actually uses to the raw metric behind
them, so the Viser termination panel can show "current value vs limit"
instead of a bare bool. Unmapped terms render as "—" in the panel.
Ported from ``noetix_mjlab`` and trimmed to the terms configured in
:mod:`mjlab.tasks.amp` (``time_out``, ``fall_down``, ``bad_orientation``).
"""

from __future__ import annotations

import math
from typing import Any

from mjlab.managers.termination_manager import TerminationTermCfg


def _func_name(func: Any) -> str:
  return getattr(func, "__name__", type(func).__name__)


def _asset(env: Any, term_cfg: TerminationTermCfg) -> Any:
  """Entity the term watches: its ``asset_cfg`` param, else the robot."""
  asset_cfg = term_cfg.params.get("asset_cfg")
  return env.scene[asset_cfg.name if asset_cfg is not None else "robot"]


def term_error_value(
  env: Any, term_cfg: TerminationTermCfg, env_idx: int
) -> float | None:
  """Return the diagnostic error for one termination term, if available."""
  name = _func_name(term_cfg.func)

  if name == "time_out":
    return float(env.episode_length_buf[env_idx].item())

  if name == "root_height_below_minimum":
    return float(_asset(env, term_cfg).data.root_link_pos_w[env_idx, 2].item())

  if name == "bad_orientation":
    # Same metric as the term: angle between -z_world and the base z axis.
    gravity_z = float(_asset(env, term_cfg).data.projected_gravity_b[env_idx, 2])
    return math.acos(max(-1.0, min(1.0, -gravity_z)))

  return None


def term_threshold_value(term_cfg: TerminationTermCfg) -> float | None:
  """Return the active threshold for a term, when applicable."""
  for key in ("minimum_height", "limit_angle"):
    if key in term_cfg.params:
      return float(term_cfg.params[key])
  if _func_name(term_cfg.func) == "time_out":
    return float("inf")
  return None


def term_error_lower_is_bad(term_cfg: TerminationTermCfg) -> bool:
  """Whether the term fires when the metric drops *below* its limit."""
  return _func_name(term_cfg.func) == "root_height_below_minimum"
