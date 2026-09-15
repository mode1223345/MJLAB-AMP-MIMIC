# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# SPDX-License-Identifier: BSD-3-Clause

"""HIM actor-critic for AMP locomotion (non-MHA)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .him_estimator import HimEstimator
from .networks import MLP, EmpiricalNormalization


class HimActorCritic(nn.Module):
  is_recurrent = False

  def __init__(
    self,
    obs,
    obs_groups,
    num_actions,
    num_one_step_obs,
    actor_obs_normalization=False,
    critic_obs_normalization=False,
    actor_hidden_dims=None,
    critic_hidden_dims=None,
    encoder_hidden_dims=None,
    projector_hidden_dims=None,
    projector_output_dim=256,
    command_dim=3,
    estimate_dim=3,
    activation="elu",
    init_noise_std=1.0,
    noise_std_type: str = "scalar",
    state_dependent_std=False,
    **kwargs,
  ):
    if kwargs:
      print(
        "HimActorCritic.__init__ got unexpected arguments, which will be ignored: "
        + str([key for key in kwargs.keys()])
      )
    super().__init__()
    if actor_hidden_dims is None:
      actor_hidden_dims = [256, 256, 256]
    if critic_hidden_dims is None:
      critic_hidden_dims = [256, 256, 256]
    if encoder_hidden_dims is None:
      encoder_hidden_dims = [256, 128, 64]
    if projector_hidden_dims is None:
      projector_hidden_dims = [256, 256]

    assert command_dim >= 0 and estimate_dim >= 0

    self.obs_groups = obs_groups
    num_actor_obs = 0
    for obs_group in obs_groups["actor"]:
      assert len(obs[obs_group].shape) == 2, (
        "The ActorCritic module only supports 1D observations."
      )
      num_actor_obs += obs[obs_group].shape[-1]
    num_critic_obs = 0
    for obs_group in obs_groups["critic"]:
      assert len(obs[obs_group].shape) == 2, (
        "The ActorCritic module only supports 1D observations."
      )
      num_critic_obs += obs[obs_group].shape[-1]

    self.num_one_step_obs = num_one_step_obs
    history_size = int(num_actor_obs / num_one_step_obs)
    encoder_latent_dim = encoder_hidden_dims[-1]

    self.actor = MLP(
      self.num_one_step_obs + estimate_dim + encoder_latent_dim,
      num_actions,
      actor_hidden_dims,
      activation,
    )
    self.actor_obs_normalization = actor_obs_normalization
    if actor_obs_normalization:
      self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
    else:
      self.actor_obs_normalizer = torch.nn.Identity()
    print(f"Actor MLP: {self.actor}")

    self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
    self.critic_obs_normalization = critic_obs_normalization
    if critic_obs_normalization:
      self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
    else:
      self.critic_obs_normalizer = torch.nn.Identity()
    print(f"Critic MLP: {self.critic}")

    self.him_estimator = HimEstimator(
      temporal_steps=history_size,
      num_one_step_obs=self.num_one_step_obs,
      num_one_step_priveleged_obs=num_critic_obs,
      enc_hidden_dims=encoder_hidden_dims,
      proj_hidden_dims=projector_hidden_dims,
      command_dim=command_dim,
      estimate_dim=estimate_dim,
      projector_output_dim=projector_output_dim,
    )
    print(f"Estimator Encoder: {self.him_estimator.encoder}")
    print(f"Estimator Projector: {self.him_estimator.projector}")

    self.noise_std_type = noise_std_type
    if self.noise_std_type == "scalar":
      self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    elif self.noise_std_type == "log":
      self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
    else:
      raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

    self.distribution = None
    Normal.set_default_validate_args(False)

  def reset(self, dones=None):
    pass

  def forward(self):
    raise NotImplementedError

  @property
  def action_mean(self):
    return self.distribution.mean

  @property
  def action_std(self):
    return self.distribution.stddev

  @property
  def entropy(self):
    return self.distribution.entropy().sum(dim=-1)

  def update_distribution(self, obs):
    with torch.no_grad():
      vel, latent = self.him_estimator(obs)
    actor_input = torch.cat((obs[:, -self.num_one_step_obs :], vel, latent), dim=-1)
    mean = self.actor(actor_input)
    if self.noise_std_type == "scalar":
      std = self.std.expand_as(mean)
    elif self.noise_std_type == "log":
      std = torch.exp(self.log_std).expand_as(mean)
    else:
      raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
    self.distribution = Normal(mean, std)

  def act(self, obs, **kwargs):
    obs = self.get_actor_obs(obs)
    obs = self.actor_obs_normalizer(obs)
    self.update_distribution(obs)
    return self.distribution.sample()

  def act_inference(self, obs):
    obs = self.get_actor_obs(obs)
    obs = self.actor_obs_normalizer(obs)
    vel, latent = self.him_estimator(obs)
    actor_input = torch.cat((obs[:, -self.num_one_step_obs :], vel, latent), dim=-1)
    return self.actor(actor_input)

  def evaluate(self, obs, **kwargs):
    obs = self.get_critic_obs(obs)
    obs = self.critic_obs_normalizer(obs)
    return self.critic(obs)

  def get_actor_obs(self, obs):
    obs_list = [obs[obs_group] for obs_group in self.obs_groups["actor"]]
    return torch.cat(obs_list, dim=-1)

  def get_critic_obs(self, obs):
    obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
    return torch.cat(obs_list, dim=-1)

  def get_actions_log_prob(self, actions):
    return self.distribution.log_prob(actions).sum(dim=-1)

  def update_normalization(self, obs):
    if self.actor_obs_normalization:
      actor_obs = self.get_actor_obs(obs)
      self.actor_obs_normalizer.update(actor_obs)
    if self.critic_obs_normalization:
      critic_obs = self.get_critic_obs(obs)
      self.critic_obs_normalizer.update(critic_obs)

  def load_state_dict(self, state_dict, strict=True):
    super().load_state_dict(state_dict, strict=strict)
    return True
