"""Viser controls to live-edit foot / ground tangential friction during play."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import mujoco
import viser

# Match Labubu / F5 AMP foot_friction DR targets (optional per pattern).
_DEFAULT_FOOT_GEOM_PATTERNS: tuple[str, ...] = (
  ".*_foot.*_collision",
  ".*_ankle_roll_collision",
)


def _matched_geom_names(robot: Any, patterns: Sequence[str]) -> list[str]:
  """Return geom names matching any pattern (patterns need not all match)."""
  matched: list[str] = []
  seen: set[str] = set()
  for name in robot.geom_names:
    if not name or name in seen:
      continue
    if any(re.fullmatch(pat, name) for pat in patterns):
      matched.append(name)
      seen.add(name)
  return matched


def _resolve_global_geom_ids(
  env: Any,
  *,
  foot_geom_patterns: Sequence[str],
  include_terrain: bool,
) -> list[int]:
  """Collect global geom ids for foot collisions and optional terrain."""
  unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
  sim = unwrapped.sim
  ids: list[int] = []

  robot = unwrapped.scene.entities.get("robot")
  if robot is not None:
    matched = _matched_geom_names(robot, foot_geom_patterns)
    if matched:
      local_ids, _ = robot.find_geoms(matched)
      if local_ids:
        geom_ids = robot.indexing.geom_ids
        ids.extend(int(geom_ids[i].item()) for i in local_ids)

  if include_terrain:
    mj = sim.mj_model
    for gid in range(int(mj.ngeom)):
      name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
      if name == "terrain" or name.endswith("/terrain"):
        ids.append(gid)

  return sorted(set(ids))


def apply_foot_ground_friction(
  env: Any,
  *,
  friction: float,
  env_idx: int | None = None,
  all_envs: bool = True,
  foot_geom_patterns: Sequence[str] = _DEFAULT_FOOT_GEOM_PATTERNS,
  include_terrain: bool = True,
  axis: int = 0,
) -> int:
  """Set tangential friction on foot (+ terrain) geoms.

  Sets both foot collision geoms and the terrain geom to the same μ so the
  MuJoCo element-wise-max contact friction equals ``friction``.

  Returns the number of geoms updated (0 if none found).
  """
  unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
  sim = unwrapped.sim
  geom_ids = _resolve_global_geom_ids(
    unwrapped,
    foot_geom_patterns=foot_geom_patterns,
    include_terrain=include_terrain,
  )
  if not geom_ids:
    return 0

  if "geom_friction" not in sim.expanded_fields:
    sim.expand_model_fields(("geom_friction",))

  mu = max(float(friction), 1e-4)
  env_ids: Any = slice(None) if all_envs or env_idx is None else int(env_idx)
  for gid in geom_ids:
    sim.model.geom_friction[env_ids, gid, axis] = mu
  return len(geom_ids)


class FootFrictionPanel:
  """Sliders that pin foot/ground tangential friction at runtime."""

  def __init__(
    self,
    server: viser.ViserServer,
    env: Any,
    get_env_idx: Callable[[], int],
    request_action: Callable[[str, Any], None],
    *,
    friction_range: tuple[float, float] = (0.3, 1.6),
    friction_bound_limits: tuple[float, float] = (0.05, 3.0),
    foot_geom_patterns: Sequence[str] = _DEFAULT_FOOT_GEOM_PATTERNS,
    include_terrain: bool = True,
  ) -> None:
    self._env = env
    self._get_env_idx = get_env_idx
    self._request_action = request_action
    self._foot_geom_patterns = tuple(foot_geom_patterns)
    self._include_terrain = include_terrain

    fr_lo, fr_hi = float(friction_range[0]), float(friction_range[1])
    if fr_lo > fr_hi:
      fr_lo, fr_hi = fr_hi, fr_lo
    bound_lo, bound_hi = (
      float(friction_bound_limits[0]),
      float(friction_bound_limits[1]),
    )
    initial = self._nominal_friction()
    if initial is None:
      initial = 1.0
    initial = min(max(initial, fr_lo), fr_hi)

    with server.gui.add_folder("Foot / ground friction (live)", expand_by_default=True):
      server.gui.add_markdown(
        "<small>Sets tangential μ on matching foot collision geoms "
        "(e.g. <code>*_foot_collision</code>) "
        "and the <code>terrain</code> geom (same value) so contact friction "
        "tracks the slider. Default range matches training DR (0.3..1.6).</small>"
      )
      self._all_envs = server.gui.add_checkbox(
        "Apply to all envs",
        initial_value=True,
      )
      self._fr_min = server.gui.add_slider(
        "Min friction",
        min=bound_lo,
        max=bound_hi,
        step=0.05,
        initial_value=fr_lo,
      )
      self._fr_max = server.gui.add_slider(
        "Max friction",
        min=bound_lo,
        max=bound_hi,
        step=0.05,
        initial_value=fr_hi,
      )
      self._friction = server.gui.add_slider(
        "Friction μ",
        min=fr_lo,
        max=fr_hi,
        step=0.05,
        initial_value=initial,
      )
      reset_btn = server.gui.add_button("Reset to nominal")

      def _sync_bounds() -> None:
        lo = float(self._fr_min.value)
        hi = float(self._fr_max.value)
        if lo > hi:
          lo, hi = hi, lo
          self._fr_min.value = lo
          self._fr_max.value = hi
        self._friction.min = lo
        self._friction.max = hi
        value = float(self._friction.value)
        if value < lo:
          self._friction.value = lo
        elif value > hi:
          self._friction.value = hi

      @self._fr_min.on_update
      def _(_) -> None:
        _sync_bounds()
        self._queue_apply()

      @self._fr_max.on_update
      def _(_) -> None:
        _sync_bounds()
        self._queue_apply()

      @self._friction.on_update
      def _(_) -> None:
        self._queue_apply()

      @self._all_envs.on_update
      def _(_) -> None:
        self._queue_apply()

      @reset_btn.on_click
      def _(_) -> None:
        nominal = self._nominal_friction()
        self._friction.value = 1.0 if nominal is None else float(nominal)
        self._queue_apply()

    self._status = server.gui.add_markdown(self._status_text(initial, 0))

  def _nominal_friction(self) -> float | None:
    unwrapped = self._env.unwrapped
    robot = unwrapped.scene.entities.get("robot")
    if robot is None:
      return None
    matched = _matched_geom_names(robot, self._foot_geom_patterns)
    if not matched:
      return None
    local_ids, _ = robot.find_geoms(matched)
    if not local_ids:
      return None
    gid = int(robot.indexing.geom_ids[local_ids[0]].item())
    default = unwrapped.sim.get_default_field("geom_friction")
    return float(default[gid, 0].item())

  def _status_text(self, friction: float, n_geoms: int) -> str:
    nominal = self._nominal_friction()
    nom_s = f"{nominal:.2f}" if nominal is not None else "?"
    geoms = f"{n_geoms} geoms" if n_geoms else "pending apply"
    return (
      f"<small>nominal μ≈{nom_s} → "
      f"live=<b>{friction:.2f}</b> ({geoms}; foot+terrain)</small>"
    )

  def _queue_apply(self) -> None:
    friction = float(self._friction.value)
    self._status.content = self._status_text(friction, 0)
    self._request_action(
      "custom",
      {
        "type": "foot_friction",
        "friction": friction,
        "all_envs": bool(self._all_envs.value),
        "env_idx": int(self._get_env_idx()),
        "foot_geom_patterns": list(self._foot_geom_patterns),
        "include_terrain": self._include_terrain,
      },
    )

  def apply(self, payload: dict[str, Any]) -> None:
    """Apply slider values on the sim thread."""
    patterns = payload.get("foot_geom_patterns", self._foot_geom_patterns)
    n = apply_foot_ground_friction(
      self._env,
      friction=float(payload["friction"]),
      env_idx=int(payload.get("env_idx", 0)),
      all_envs=bool(payload.get("all_envs", True)),
      foot_geom_patterns=tuple(patterns),
      include_terrain=bool(payload.get("include_terrain", self._include_terrain)),
    )
    self._status.content = self._status_text(float(payload["friction"]), n)
    if n == 0:
      print("[WARN] FootFrictionPanel: no foot/terrain geoms found")
