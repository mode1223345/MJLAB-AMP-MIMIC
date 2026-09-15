"""Viser controls to live-edit head/torso mass / COM during play."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import torch
import viser

from mjlab.managers.event_manager import RecomputeLevel

# Prefer Labubu head; fall back to F5 waist (upper mass DR target).
_DEFAULT_MASS_BODY_CANDIDATES: tuple[str, ...] = (
  "head_pitch_link",
  "waist_yaw_link",
)


def resolve_mass_body_name(
  env: Any,
  candidates: Sequence[str] = _DEFAULT_MASS_BODY_CANDIDATES,
) -> str | None:
  """Return the first candidate body present on the robot, else None."""
  unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
  robot = unwrapped.scene.entities.get("robot")
  if robot is None:
    return None
  for cand in candidates:
    for n in robot.body_names:
      if n == cand or re.fullmatch(cand, n):
        return n
  return None


def apply_head_pitch_props(
  env: Any,
  *,
  mass_add: float,
  com_offset: tuple[float, float, float],
  env_idx: int | None = None,
  all_envs: bool = True,
  body_name: str = "head_pitch_link",
) -> bool:
  """Set body mass/COM relative to MJCF defaults.

  Returns False if the body is missing.
  """
  unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
  robot = unwrapped.scene.entities.get("robot")
  if robot is None:
    return False
  matched = [
    n for n in robot.body_names if n == body_name or re.fullmatch(body_name, n)
  ]
  if not matched:
    return False
  ids, _ = robot.find_bodies(matched[0])
  if not ids:
    return False
  body_id = int(robot.indexing.body_ids[ids[0]].item())
  sim = unwrapped.sim

  needed = tuple(f for f in ("body_mass", "body_ipos") if f not in sim.expanded_fields)
  if needed:
    sim.expand_model_fields(needed)

  default_mass = float(sim.get_default_field("body_mass")[body_id].item())
  default_ipos = sim.get_default_field("body_ipos")[body_id].to(
    device=sim.device, dtype=torch.float32
  )
  offset = torch.tensor(com_offset, dtype=torch.float32, device=sim.device)
  # Keep a tiny positive mass so MuJoCo mass-matrix stays valid.
  new_mass = max(default_mass + float(mass_add), 1e-3)
  new_ipos = default_ipos + offset

  env_ids: Any = slice(None) if all_envs or env_idx is None else int(env_idx)
  sim.model.body_mass[env_ids, body_id] = new_mass
  sim.model.body_ipos[env_ids, body_id] = new_ipos
  sim.recompute_constants(RecomputeLevel.set_const)
  return True


class HeadMassPanel:
  """Sliders that pin a link's mass add and COM offset at runtime."""

  def __init__(
    self,
    server: viser.ViserServer,
    env: Any,
    get_env_idx: Callable[[], int],
    request_action: Callable[[str, Any], None],
    *,
    body_name: str = "head_pitch_link",
    mass_add_range: tuple[float, float] = (-5.0, 8.0),
    mass_bound_limits: tuple[float, float] = (-20.0, 20.0),
    com_range: float = 0.10,
    folder_title: str | None = None,
  ) -> None:
    self._env = env
    self._get_env_idx = get_env_idx
    self._request_action = request_action
    self._body_name = body_name
    mass_lo, mass_hi = float(mass_add_range[0]), float(mass_add_range[1])
    if mass_lo > mass_hi:
      mass_lo, mass_hi = mass_hi, mass_lo
    bound_lo, bound_hi = float(mass_bound_limits[0]), float(mass_bound_limits[1])
    title = folder_title or f"{body_name} mass (live)"

    with server.gui.add_folder(title, expand_by_default=True):
      server.gui.add_markdown(
        f"<small>Edits <code>{body_name}</code> relative to MJCF "
        "nominal. Applies immediately (triggers mass-matrix recompute). "
        "Total mass is clamped above zero. Min/Max below widen the Mass "
        "add slider.</small>"
      )
      self._all_envs = server.gui.add_checkbox(
        "Apply to all envs",
        initial_value=True,
      )
      self._mass_min = server.gui.add_slider(
        "Min mass add",
        min=bound_lo,
        max=bound_hi,
        step=0.1,
        initial_value=mass_lo,
      )
      self._mass_max = server.gui.add_slider(
        "Max mass add",
        min=bound_lo,
        max=bound_hi,
        step=0.1,
        initial_value=mass_hi,
      )
      self._mass_add = server.gui.add_slider(
        "Mass add (kg)",
        min=mass_lo,
        max=mass_hi,
        step=0.1,
        initial_value=0.0,
      )
      self._com_x = server.gui.add_slider(
        "COM offset x (m)",
        min=-com_range,
        max=com_range,
        step=0.005,
        initial_value=0.0,
      )
      self._com_y = server.gui.add_slider(
        "COM offset y (m)",
        min=-com_range,
        max=com_range,
        step=0.005,
        initial_value=0.0,
      )
      self._com_z = server.gui.add_slider(
        "COM offset z (m)",
        min=-com_range,
        max=com_range,
        step=0.005,
        initial_value=0.0,
      )
      reset_btn = server.gui.add_button("Reset to nominal")

      def _sync_mass_bounds() -> None:
        lo = float(self._mass_min.value)
        hi = float(self._mass_max.value)
        if lo > hi:
          lo, hi = hi, lo
          self._mass_min.value = lo
          self._mass_max.value = hi
        self._mass_add.min = lo
        self._mass_add.max = hi
        value = float(self._mass_add.value)
        if value < lo:
          self._mass_add.value = lo
        elif value > hi:
          self._mass_add.value = hi

      @self._mass_min.on_update
      def _(_) -> None:
        _sync_mass_bounds()
        self._queue_apply()

      @self._mass_max.on_update
      def _(_) -> None:
        _sync_mass_bounds()
        self._queue_apply()

      @self._mass_add.on_update
      def _(_) -> None:
        self._queue_apply()

      @self._com_x.on_update
      def _(_) -> None:
        self._queue_apply()

      @self._com_y.on_update
      def _(_) -> None:
        self._queue_apply()

      @self._com_z.on_update
      def _(_) -> None:
        self._queue_apply()

      @self._all_envs.on_update
      def _(_) -> None:
        self._queue_apply()

      @reset_btn.on_click
      def _(_) -> None:
        self._mass_add.value = 0.0
        self._com_x.value = 0.0
        self._com_y.value = 0.0
        self._com_z.value = 0.0
        self._queue_apply()

    self._status = server.gui.add_markdown(self._status_text(0.0, (0.0, 0.0, 0.0)))

  def _nominal_mass(self) -> float | None:
    unwrapped = self._env.unwrapped
    robot = unwrapped.scene.entities.get("robot")
    if robot is None:
      return None
    matched = [
      n
      for n in robot.body_names
      if n == self._body_name or re.fullmatch(self._body_name, n)
    ]
    if not matched:
      return None
    ids, _ = robot.find_bodies(matched[0])
    if not ids:
      return None
    body_id = int(robot.indexing.body_ids[ids[0]].item())
    return float(unwrapped.sim.get_default_field("body_mass")[body_id].item())

  def _status_text(self, mass_add: float, com: tuple[float, float, float]) -> str:
    nominal = self._nominal_mass()
    if nominal is None:
      return f"<small>Body <code>{self._body_name}</code> not found.</small>"
    total = nominal + mass_add
    return (
      f"<small>nominal={nominal:.3f} kg → "
      f"total=<b>{total:.3f} kg</b> "
      f"({mass_add:+.2f}); "
      f"COM Δ=({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:+.3f}) m</small>"
    )

  def _queue_apply(self) -> None:
    mass_add = float(self._mass_add.value)
    com = (
      float(self._com_x.value),
      float(self._com_y.value),
      float(self._com_z.value),
    )
    self._status.content = self._status_text(mass_add, com)
    self._request_action(
      "custom",
      {
        "type": "head_mass",
        "mass_add": mass_add,
        "com_offset": com,
        "all_envs": bool(self._all_envs.value),
        "env_idx": int(self._get_env_idx()),
        "body_name": self._body_name,
      },
    )

  def apply(self, payload: dict[str, Any]) -> None:
    """Apply slider values on the sim thread."""
    com_raw = payload["com_offset"]
    com = (float(com_raw[0]), float(com_raw[1]), float(com_raw[2]))
    ok = apply_head_pitch_props(
      self._env,
      mass_add=float(payload["mass_add"]),
      com_offset=com,
      env_idx=int(payload.get("env_idx", 0)),
      all_envs=bool(payload.get("all_envs", True)),
      body_name=str(payload.get("body_name", self._body_name)),
    )
    if not ok:
      print(f"[WARN] HeadMassPanel: body {self._body_name!r} not found")
