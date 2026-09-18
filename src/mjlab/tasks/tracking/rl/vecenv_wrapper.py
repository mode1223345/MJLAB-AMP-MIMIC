"""VecEnv wrapper for N3 mimic: actor-history flattening + terminal obs."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MimicHimVecEnvWrapper(RslRlVecEnvWrapper):
  """Flatten actor history (N, H, D) → (N, H·D), expose terminal actor obs.

  Terminal observations use the pre-step actor obs: mjlab resets terminated
  envs inside ``step()``, so the post-step obs of a done env is already the
  fresh episode's first frame (fork convention, slight difference from Isaac
  which captures the true final state before reset).
  """

  def __init__(self, env, clip_actions: float | None = None):
    super().__init__(env, clip_actions=clip_actions)
    self._last_actor_obs: torch.Tensor | None = None
    obs = self.get_observations()
    self._last_actor_obs = obs["actor"].clone()

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
    return obs

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    last_actor = self._last_actor_obs
    obs, rew, dones, extras = super().step(actions)

    done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
    term_actor = (
      last_actor[done_ids] if last_actor is not None else obs["actor"][done_ids]
    )
    if isinstance(term_actor, torch.Tensor) and term_actor.ndim == 3:
      term_actor = term_actor.flatten(1, 2)

    extras["terminal_observations"] = {
      "env_ids": done_ids,
      "actor": term_actor,
      "policy": term_actor,  # alias for source runner naming
    }

    obs = self._flatten_actor(obs)
    self._last_actor_obs = obs["actor"].clone()
    return obs, rew, dones, extras
