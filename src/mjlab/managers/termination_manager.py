"""Termination manager for computing done signals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class TerminationTermCfg(ManagerTermBaseCfg):
  """Configuration for a termination term."""

  time_out: bool = False
  """Whether the term contributes towards episodic timeouts."""


class TerminationManager(ManagerBase):
  """Manages termination conditions for the environment.

  The termination manager aggregates multiple termination terms to compute
  episode done signals. Terms can be either truncations (time-based) or
  terminations (failure conditions).
  """

  _env: ManagerBasedRlEnv

  def __init__(self, cfg: dict[str, TerminationTermCfg], env: ManagerBasedRlEnv):
    self._term_names: list[str] = list()
    self._term_cfgs: list[TerminationTermCfg] = list()
    self._class_term_cfgs: list[TerminationTermCfg] = list()

    self.cfg = deepcopy(cfg)
    super().__init__(env)

    self._term_dones = dict()
    self._last_episode_dones = dict()
    for term_name in self._term_names:
      self._term_dones[term_name] = torch.zeros(
        self.num_envs, device=self.device, dtype=torch.bool
      )
      self._last_episode_dones[term_name] = torch.zeros_like(
        self._term_dones[term_name]
      )
    self._truncated_buf = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._terminated_buf = torch.zeros_like(self._truncated_buf)

  def __str__(self) -> str:
    msg = f"<TerminationManager> contains {len(self._term_names)} active terms.\n"
    table = PrettyTable()
    table.title = "Active Termination Terms"
    table.field_names = ["Index", "Name", "Time Out"]
    table.align["Name"] = "l"
    for index, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      table.add_row([index, name, term_cfg.time_out])
    msg += table.get_string()
    msg += "\n"
    return msg

  # Properties.

  @property
  def active_terms(self) -> list[str]:
    return self._term_names

  @property
  def dones(self) -> torch.Tensor:
    return self._truncated_buf | self._terminated_buf

  @property
  def time_outs(self) -> torch.Tensor:
    return self._truncated_buf

  @property
  def terminated(self) -> torch.Tensor:
    return self._terminated_buf

  # Methods.

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    if env_ids is None:
      env_ids = slice(None)
    # Snapshot the reasons for the episodes that are ending now. This runs for
    # the envs that actually reset, so an env whose termination was suppressed
    # (e.g. by ``DelayedTerminationManager``) keeps its previous episode's
    # reasons instead of recording a reset that never happened.
    for key in self._term_dones.keys():
      self._last_episode_dones[key][env_ids] = self._term_dones[key][env_ids]
    # Share of the episodes that ended in this reset call, as a fraction
    # (Isaac Lab's ``Episode_Termination`` convention). In training every reset
    # ends an episode, so the denominator is the number of episodes just ended.
    # Terms are not mutually exclusive -- a robot on the ground trips the height
    # and the tilt checks in the same step -- so two flavors are logged:
    # ``Episode_Termination_any/*`` is the overlapping share (may sum past 1),
    # while ``Episode_Termination/*`` attributes each episode to a single
    # reason by priority (time outs first, then declaration order), so those
    # sum to exactly 1 and read as "what ended the episode".
    order = self._priority_ordered_names()
    extras: dict[str, float] = {}
    if order:
      num_resets = self.num_envs if isinstance(env_ids, slice) else int(env_ids.numel())
      fired = {key: self._term_dones[key][env_ids] for key in order}
      exclusive = {}
      unassigned = torch.zeros_like(self._truncated_buf[env_ids])
      for key in order:
        exclusive[key] = fired[key] & ~unassigned
        unassigned |= fired[key]
      # Stack everything and sync once, so a reset step costs a single .tolist().
      shares = (
        torch.stack(
          [
            torch.stack(
              [
                torch.count_nonzero(fired[key]).float(),
                torch.count_nonzero(exclusive[key]).float(),
              ]
            )
            for key in order
          ]
        )
        / max(num_resets, 1)
      ).tolist()
      for key, (any_share, exclusive_share) in zip(order, shares, strict=True):
        extras["Episode_Termination_any/" + key] = any_share
        extras["Episode_Termination/" + key] = exclusive_share
    for term_cfg in self._class_term_cfgs:
      term_cfg.func.reset(env_ids=env_ids)
    return extras

  def compute(self) -> torch.Tensor:
    self._truncated_buf[:] = False
    self._terminated_buf[:] = False
    for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=False):
      value = term_cfg.func(self._env, **term_cfg.params)
      self._check_term_shape(name, value)
      if term_cfg.time_out:
        self._truncated_buf |= value
      else:
        self._terminated_buf |= value
      self._term_dones[name][:] = value
    return self._truncated_buf | self._terminated_buf

  def get_term(self, name: str) -> torch.Tensor:
    return self._term_dones[name]

  def get_last_episode_term(self, name: str) -> torch.Tensor:
    """Per-env bool mask of whether ``name`` ended the last episode."""
    return self._last_episode_dones[name]

  def get_term_cfg(self, term_name: str) -> TerminationTermCfg:
    if term_name not in self._term_names:
      raise ValueError(f"Term '{term_name}' not found in active terms.")
    return self._term_cfgs[self._term_names.index(term_name)]

  def _priority_ordered_names(self) -> list[str]:
    """Term names for exclusive attribution: time outs first, then decl order.

    An episode that trips several terms at once is credited to the first name
    in this order, so the per-term shares in ``reset`` add up to 1. Time outs
    lead because reaching the horizon is not a fall.
    """
    pairs = list(zip(self._term_names, self._term_cfgs, strict=False))
    return [name for name, cfg in pairs if cfg.time_out] + [
      name for name, cfg in pairs if not cfg.time_out
    ]

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    terms = []
    for key in self._term_dones.keys():
      terms.append((key, [self._term_dones[key][env_idx].float().cpu().item()]))
    return terms

  def _prepare_terms(self):
    for term_name, term_cfg in self.cfg.items():
      term_cfg: TerminationTermCfg | None
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      self._resolve_common_term_cfg(term_name, term_cfg)
      self._term_names.append(term_name)
      self._term_cfgs.append(term_cfg)
      if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
        self._class_term_cfgs.append(term_cfg)
