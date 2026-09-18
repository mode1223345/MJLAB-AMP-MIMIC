"""HIM PPO for N3 mimic (port of Isaac HIMPPO, AMP/RND/symmetry removed).

The HIM estimator keeps its own optimizer inside itself; the PPO optimizer
excludes every ``him_estimator.*`` parameter so PPO gradients never touch it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from mjlab.tasks.amp.rl.him_rollout_storage import HimRolloutStorage


class MimicHIMPPO:
  """Proximal Policy Optimization with a per-minibatch HIM update."""

  policy: torch.nn.Module

  def __init__(
    self,
    policy,
    num_learning_epochs=5,
    num_mini_batches=4,
    clip_param=0.2,
    gamma=0.99,
    lam=0.95,
    value_loss_coef=1.0,
    entropy_coef=0.01,
    learning_rate=1e-3,
    max_grad_norm=1.0,
    use_clipped_value_loss=True,
    schedule="adaptive",
    desired_kl=0.01,
    device="cpu",
    normalize_advantage_per_mini_batch=False,
  ):
    self.device = device
    self.policy = policy
    self.policy.to(self.device)

    # PPO optimizer: HIM estimator is trained by its own internal Adam.
    ppo_params = [
      param
      for name, param in self.policy.named_parameters()
      if not name.startswith("him_estimator.")
    ]
    self.optimizer = optim.Adam(ppo_params, lr=learning_rate)

    self.storage: HimRolloutStorage = None  # type: ignore[assignment]
    self.transition = HimRolloutStorage.Transition()

    self.clip_param = clip_param
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss
    self.desired_kl = desired_kl
    self.schedule = schedule
    self.learning_rate = learning_rate
    self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

  def init_storage(
    self, training_type, num_envs, num_transitions_per_env, obs, actions_shape
  ):
    self.storage = HimRolloutStorage(
      training_type,
      num_envs,
      num_transitions_per_env,
      obs,
      actions_shape,
      self.device,
    )

  def act(self, obs):
    self.transition.actions = self.policy.act(obs).detach()
    self.transition.values = self.policy.evaluate(obs).detach()
    self.transition.actions_log_prob = self.policy.get_actions_log_prob(
      self.transition.actions
    ).detach()
    self.transition.action_mean = self.policy.action_mean.detach()
    self.transition.action_sigma = self.policy.action_std.detach()
    # Record obs before env.step().
    self.transition.observations = obs
    return self.transition.actions

  def process_env_step(self, obs, rewards, dones, extras, next_obs):
    self.policy.update_normalization(obs)

    # Cloned: rewards are bootstrapped below based on timeouts.
    self.transition.next_observations = next_obs.clone()
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones

    # Bootstrapping on time outs (motion_end counts as a truncation).
    if "time_outs" in extras:
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
      )

    self.storage.add_transitions(self.transition)
    self.transition.clear()
    self.policy.reset(dones)

  def compute_returns(self, obs):
    last_values = self.policy.evaluate(obs).detach()
    self.storage.compute_returns(
      last_values,
      self.gamma,
      self.lam,
      normalize_advantage=not self.normalize_advantage_per_mini_batch,
    )

  def update(self):  # noqa: C901
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_estimate_loss = 0.0
    mean_barlow_twin_loss = 0.0

    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for (
      obs_batch,
      actions_batch,
      next_obs_batch,
      target_values_batch,
      advantages_batch,
      returns_batch,
      old_actions_log_prob_batch,
      old_mu_batch,
      old_sigma_batch,
      _hid_states_batch,
      _masks_batch,
    ) in generator:
      obs_batch["next_obs"] = next_obs_batch

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
          )

      self.policy.act(obs_batch)
      actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
      value_batch = self.policy.evaluate(obs_batch)
      mu_batch = self.policy.action_mean
      sigma_batch = self.policy.action_std
      entropy_batch = self.policy.entropy

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            axis=-1,
          )
          kl_mean = torch.mean(kl)
          if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(
        actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
      )
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy_batch.mean()
      )

      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(
        [
          p
          for n, p in self.policy.named_parameters()
          if not n.startswith("him_estimator.")
        ],
        self.max_grad_norm,
      )
      self.optimizer.step()

      # HIM update after the PPO step (own optimizer, follows PPO lr).
      actor_obs_batch = self.policy.get_actor_obs(obs_batch)
      actor_obs_batch = self.policy.actor_obs_normalizer(actor_obs_batch)
      # Unnormalized critic obs: the state slice indexes raw layout.
      critic_obs_batch_raw = self.policy.get_critic_obs(obs_batch)
      next_obs_batch_norm = self.policy.actor_obs_normalizer(obs_batch["next_obs"])
      estimation_loss, barlow_twin_loss = self.policy.him_estimator.update(
        actor_obs_batch,
        critic_obs_batch_raw,
        next_obs_batch_norm,
        lr=self.learning_rate,
      )

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy_batch.mean().item()
      mean_estimate_loss += estimation_loss
      mean_barlow_twin_loss += barlow_twin_loss

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_estimate_loss /= num_updates
    mean_barlow_twin_loss /= num_updates
    self.storage.clear()

    return {
      "value_function": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "estimate": mean_estimate_loss,
      "barlow_twin": mean_barlow_twin_loss,
    }
