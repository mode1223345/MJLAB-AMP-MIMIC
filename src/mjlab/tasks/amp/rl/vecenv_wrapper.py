# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# SPDX-License-Identifier: BSD-3-Clause

"""VecEnv wrapper for AMP: terminal obs capture + motion loader attach."""

from __future__ import annotations

import glob
from pathlib import Path

import torch
from tensordict import TensorDict

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from .motion_loader_attach import build_motion_loader


class AmpHimVecEnvWrapper(RslRlVecEnvWrapper):
  """Flatten actor history, expose terminal actor/amp obs, attach MotionLoader."""

  def __init__(self, env, clip_actions: float | None = None):
    super().__init__(env, clip_actions=clip_actions)
    attached_loader = self._ensure_motion_loader()
    self._ensure_amp_buffers()
    # Env construction may reset before MotionLoader exists; re-reset so RSI
    # (prob_rsi / reference frames) actually runs with the loader attached.
    if attached_loader:
      self.reset()
    self._last_actor_obs: torch.Tensor | None = None
    self._last_amp_obs: torch.Tensor | None = None
    obs = self.get_observations()
    self._last_actor_obs = obs["actor"].clone()
    if "discriminator" in obs:
      self._last_amp_obs = obs["discriminator"].clone()

  def _ensure_amp_buffers(self) -> None:
    unwrapped = self.unwrapped
    if not hasattr(unwrapped, "feet_both_contact_time"):
      object.__setattr__(
        unwrapped,
        "feet_both_contact_time",
        torch.zeros(
          unwrapped.num_envs,
          dtype=torch.float,
          device=unwrapped.device,
          requires_grad=False,
        ),
      )

  def _ensure_motion_loader(self) -> bool:
    """Attach MotionLoader if missing. Returns True when a new loader was created."""
    unwrapped = self.unwrapped
    if getattr(unwrapped, "motion_loader", None) is not None:
      return False

    cfg = unwrapped.cfg
    motion_files = list(getattr(cfg, "amp_motion_files", None) or [])
    if isinstance(motion_files, (str, Path)):
      motion_files = sorted(glob.glob(str(motion_files)))
    if not motion_files:
      raise ValueError(
        "AMP env cfg.amp_motion_files is empty. Set expert JSON paths on the "
        "task env config (tyro drops non-dataclass attrs)."
      )
    horizon = int(getattr(cfg, "amp_reference_observation_horizon", 1))
    num_preload = int(getattr(cfg, "amp_num_preload_transitions", 200000))

    object.__setattr__(
      unwrapped,
      "motion_loader",
      build_motion_loader(self, motion_files, horizon, num_preload),
    )
    return True

  @property
  def num_one_step_actor_obs(self) -> int:
    actor_space = self.unwrapped.single_observation_space.spaces["actor"]
    return int(actor_space.shape[-1])

  def _flatten_actor(self, obs: TensorDict) -> TensorDict:
    actor = obs["actor"]
    if actor.ndim == 3:
      obs["actor"] = actor.flatten(1, 2)
    return obs

  def get_observations(self) -> TensorDict:
    obs = super().get_observations()
    obs = self._flatten_actor(obs)
    self._last_actor_obs = obs["actor"].clone()
    if "discriminator" in obs:
      amp = obs["discriminator"]
      # Keep disc history as (N, H, D) for AMP buffer; squeeze H=1 later if needed.
      self._last_amp_obs = amp.clone()
    return obs

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    last_actor = self._last_actor_obs
    last_amp = self._last_amp_obs
    obs, rew, dones, extras = super().step(actions)

    done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
    # Reset both-feet contact timer on episode ends.
    contact_time = getattr(self.unwrapped, "feet_both_contact_time", None)
    if len(done_ids) > 0 and isinstance(contact_time, torch.Tensor):
      contact_time[done_ids] = 0.0

    term_actor = (
      last_actor[done_ids] if last_actor is not None else obs["actor"][done_ids]
    )
    if isinstance(term_actor, torch.Tensor) and term_actor.ndim == 3:
      term_actor = term_actor.flatten(1, 2)

    if last_amp is not None:
      term_amp = last_amp[done_ids]
    elif "discriminator" in obs:
      term_amp = obs["discriminator"][done_ids]
    else:
      term_amp = torch.empty(0)

    extras["terminal_observations"] = {
      "env_ids": done_ids,
      "actor": term_actor,
      "policy": term_actor,  # alias for source runner naming
      "amp": term_amp,
    }

    obs = self._flatten_actor(obs)
    self._last_actor_obs = obs["actor"].clone()
    if "discriminator" in obs:
      self._last_amp_obs = obs["discriminator"].clone()
    return obs, rew, dones, extras
