"""Delayed termination for fall-recovery training.

A subset of envs ("delay envs") has fall terminations suppressed for up to
``max_delay_steps`` consecutive terminated steps, giving the policy time to
get up instead of being reset immediately. Ported from the Noetix AMP stack
(``AMP_mjlab``); see ``DelayedTerminationManager`` there for the original.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.termination_manager import TerminationManager

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class DelayedTerminationManager(TerminationManager):
  """Suppresses resets for delay envs until the counter reaches the limit.

  For delay envs, each step with an active termination increments a counter
  and the ``terminated`` flag is suppressed. After ``max_delay_steps``
  consecutive terminated steps the reset is released and the counter clears.
  If the env recovers on its own (the termination condition goes away), the
  counter resets and the env keeps running. Time-outs always pass through.
  """

  def __init__(
    self,
    base: TerminationManager,
    delay_env_mask: torch.Tensor,
    max_delay_steps: int,
  ):
    # Take over the base manager's state in place (buffers, term cfgs, env
    # reference) so the env can swap us in without re-initializing anything.
    self.__dict__.update(base.__dict__)
    self.delay_env_mask = delay_env_mask  # (num_envs,) bool
    self._delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
    self._max_delay_steps = max_delay_steps

  def compute(self) -> torch.Tensor:
    dones = super().compute()  # fills _truncated_buf / _terminated_buf

    if self._max_delay_steps <= 0:
      return dones

    # Delay envs that just got a done signal increment their counter.
    delay_and_done = self.delay_env_mask & dones
    self._delay_counters[delay_and_done] += 1

    # Not yet at the limit: suppress the reset.
    not_ready = delay_and_done & (self._delay_counters < self._max_delay_steps)
    self._terminated_buf[not_ready] = False

    # Reached the limit: allow the reset, clear the counter.
    ready = delay_and_done & (self._delay_counters >= self._max_delay_steps)
    self._delay_counters[ready] = 0

    # Env recovered on its own: clear the counter and keep running.
    self._delay_counters[self.delay_env_mask & ~dones] = 0

    return self._truncated_buf | self._terminated_buf


def get_delay_env_mask(env: ManagerBasedRlEnv) -> torch.Tensor | None:
  """Return the delay-env mask, or None when delayed termination is off."""
  tm = env.termination_manager
  if isinstance(tm, DelayedTerminationManager):
    return tm.delay_env_mask
  return None


def root_height_below_terrain(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  sensor_name: str = "terrain_scan",
) -> torch.Tensor:
  """Terminate when the root drops below a height above the local terrain.

  Port of the Isaac-Lab N3 termination [ADD 2026-09-04]: an absolute
  world-height check goes inert (or false-fires downhill) once terrains are
  elevated, so the local ground is taken from the height scan — the ray hit
  closest to the root in the xy plane approximates the terrain directly under
  the robot (grid spacing 0.1 m → within ~7 cm). Height-relative to
  ``env_origins`` would not work either: walking down a slope changes the
  origin-relative height by the full slope drop while the robot stays upright.

  Rays that miss (``distances`` < 0) are excluded from the nearest-hit pick.
  If the sensor is absent or no ray hits, fall back to the absolute world
  height check, which is exact on flat terrain.
  """
  asset = env.scene["robot"]
  root_pos_w = asset.data.root_link_pos_w

  if sensor_name not in env.scene.sensors:
    return root_pos_w[:, 2] < minimum_height
  sensor = env.scene.sensors[sensor_name]
  distances = sensor.data.distances
  if distances is None or sensor.data.hit_pos_w is None:
    return root_pos_w[:, 2] < minimum_height

  valid = distances > 0
  dist_xy = torch.norm(sensor.data.hit_pos_w[..., :2] - root_pos_w[:, None, :2], dim=-1)
  dist_xy = torch.where(valid, dist_xy, torch.full_like(dist_xy, torch.inf))
  nearest = dist_xy.argmin(dim=1)
  env_ids = torch.arange(root_pos_w.shape[0], device=root_pos_w.device)
  terrain_z = sensor.data.hit_pos_w[env_ids, nearest, 2]
  # No hit at all (e.g. ray budget exhausted): terrain height 0 ≡ absolute.
  has_ground = valid.any(dim=1)
  terrain_z = torch.where(has_ground, terrain_z, torch.zeros_like(terrain_z))
  return (root_pos_w[:, 2] - terrain_z) < minimum_height
