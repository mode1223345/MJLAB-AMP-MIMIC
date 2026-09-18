# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# SPDX-License-Identifier: BSD-3-Clause

"""AMP discriminator network."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd

DISC_LOGIT_INIT_SCALE = 1.0


class Discriminator(nn.Module):
  def __init__(
    self,
    observation_dim,
    observation_horizon,
    device,
    reward_coef=0.1,
    reward_lerp=0.3,
    shape=None,
    style_reward_function="quad_mapping",
    mask_dims=None,
    mask_joint_names=None,
    joint_names=None,
    joint_pos_size: int | None = None,
    joint_vel_start_relative: int | None = None,
    **kwargs,
  ):
    if kwargs:
      print(
        "Discriminator.__init__ got unexpected arguments, which will be ignored: "
        + str([key for key in kwargs.keys()])
      )
    super().__init__()
    if shape is None:
      shape = [1024, 512]
    self.observation_dim = observation_dim
    self.observation_horizon = observation_horizon
    self.input_dim = observation_dim * observation_horizon
    self.device = device
    self.reward_coef = reward_coef
    self.reward_lerp = reward_lerp
    self.style_reward_function = style_reward_function
    self.shape = shape

    self.mask_dims = mask_dims
    self.mask_joint_names = mask_joint_names
    self.joint_names = joint_names
    # Relative layout inside AMP obs (starts at joint pose).
    self.joint_pos_size = (
      joint_pos_size
      if joint_pos_size is not None
      else (len(joint_names) if joint_names else 0)
    )
    # joint_pos | key_pos | lin_vel(3) | ang_vel(3) | joint_vel | contact
    if joint_vel_start_relative is None:
      # Infer when key_pos local size = observation_dim - 2*joint - 3 - 3 - 2.
      key_pos = observation_dim - 2 * self.joint_pos_size - 3 - 3 - 2
      self.joint_vel_start_relative = self.joint_pos_size + max(key_pos, 0) + 6
    else:
      self.joint_vel_start_relative = joint_vel_start_relative
    self._setup_mask()

    discriminator_layers = []
    curr_in_dim = self.input_dim
    for hidden_dim in self.shape:
      discriminator_layers.append(nn.Linear(curr_in_dim, hidden_dim))
      discriminator_layers.append(nn.LeakyReLU())
      curr_in_dim = hidden_dim
    self.architecture = nn.Sequential(*discriminator_layers).to(self.device)
    self.discriminator_logits = torch.nn.Linear(hidden_dim, 1)
    self.train()

  def _setup_mask(self):
    mask = torch.ones(self.observation_dim, dtype=torch.float32, device=self.device)
    has_mask = False

    if self.mask_dims is not None and len(self.mask_dims) > 0:
      has_mask = True
      for start_idx, end_idx in self.mask_dims:
        if start_idx >= 0 and end_idx <= self.observation_dim:
          mask[start_idx:end_idx] = 0.0
        else:
          print(
            f"Warning: Invalid mask range ({start_idx}, {end_idx}) for "
            f"observation_dim {self.observation_dim}"
          )

    if self.mask_joint_names is not None and len(self.mask_joint_names) > 0:
      if self.joint_names is None:
        print(
          "Warning: mask_joint_names provided but joint_names is None. "
          "Skipping joint-based masking."
        )
      else:
        has_mask = True
        masked_joints = []
        joint_indices = []
        for mask_joint in self.mask_joint_names:
          try:
            idx = self.joint_names.index(mask_joint)
            joint_indices.append(idx)
            masked_joints.append(mask_joint)
          except ValueError:
            print(f"Warning: Joint '{mask_joint}' not found. Skipping.")

        for joint_idx in joint_indices:
          if joint_idx < self.joint_pos_size:
            mask[joint_idx] = 0.0
          vel_idx = self.joint_vel_start_relative + joint_idx
          if vel_idx < self.observation_dim:
            mask[vel_idx] = 0.0
        print(f"Discriminator: Masking joints {masked_joints} (pos & vel)")

    if not has_mask:
      self.mask = None
      return

    self.mask = mask.repeat(self.observation_horizon)
    masked_count = (self.mask == 0).sum().item()
    total_dims = self.observation_dim * self.observation_horizon
    print(f"Discriminator: Masking {masked_count}/{total_dims} dimensions total")

  def _apply_mask(self, x):
    if self.mask is None:
      return x
    return x * self.mask

  def forward(self, x):
    x = self._apply_mask(x)
    return self.discriminator_logits(self.architecture(x))

  def get_disc_weights(self):
    weights = []
    for m in self.architecture.modules():
      if isinstance(m, nn.Linear):
        weights.append(torch.flatten(m.weight))
    return weights

  def get_disc_logit_weights(self):
    return torch.flatten(self.discriminator_logits.weight)

  def eval_disc(self, x):
    x = self._apply_mask(x)
    return self.discriminator_logits(self.architecture(x))

  def compute_grad_pen(self, expert_data, lambda_=10):
    expert_data.requires_grad_(True)
    disc = self.eval_disc(expert_data)
    grad = autograd.grad(
      outputs=disc,
      inputs=expert_data,
      grad_outputs=torch.ones(disc.size(), device=disc.device),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return lambda_ * (grad.pow(2).sum(dim=1).mean())

  def compute_wgan_grad_pen(self, expert_data, policy_data, k=2, p=6):
    expert_data.requires_grad_(True)
    policy_data.requires_grad_(True)

    expert_d = self.eval_disc(expert_data)
    expert_grad = autograd.grad(
      outputs=expert_d,
      inputs=expert_data,
      grad_outputs=torch.ones(expert_d.size(), device=expert_d.device),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    expert_grad_norm = expert_grad.pow(2).sum(1).pow(p / 2)

    policy_d = self.eval_disc(policy_data)
    policy_grad = autograd.grad(
      outputs=policy_d,
      inputs=policy_data,
      grad_outputs=torch.ones(policy_d.size(), device=policy_d.device),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    policy_grad_norm = policy_grad.pow(2).sum(1).pow(p / 2)

    return (expert_grad_norm.mean() + policy_grad_norm.mean()) * k * 0.5

  def compute_weight_decay(self, lambda_=0.0001):
    disc_weights = torch.cat(self.get_disc_weights(), dim=-1)
    return lambda_ * torch.sum(torch.square(disc_weights))

  def compute_logit_reg(self, lambda_=0.05):
    logit_weights = self.get_disc_logit_weights()
    return lambda_ * torch.sum(torch.square(logit_weights))

  def predict_amp_reward(
    self,
    state_buf,
    task_reward,
    state_normalizer=None,
    style_reward_normalizer=None,
    *,
    update_style_normalizer: bool = True,
  ):
    with torch.no_grad():
      was_training = self.training
      self.eval()
      if state_normalizer is not None:
        batch_size = state_buf.shape[0]
        state_flat = state_buf.view(-1, state_buf.shape[-1])
        state_flat_norm = state_normalizer.normalize_torch(state_flat, self.device)
        state_buf_norm = state_flat_norm.view(batch_size, -1)
        d = self.eval_disc(state_buf_norm)
      else:
        d = self.eval_disc(state_buf.flatten(1, 2))

      if self.style_reward_function == "quad_mapping":
        style_reward = torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
      elif self.style_reward_function == "log_mapping":
        style_reward = -torch.log(
          torch.maximum(
            1 - 1 / (1 + torch.exp(-d)),
            torch.tensor(0.0001, device=self.device),
          )
        )
      elif self.style_reward_function == "wasserstein_mapping":
        if style_reward_normalizer is not None:
          d_clone = d.clone()
          style_reward = style_reward_normalizer.normalize_torch(d_clone, self.device)
          if update_style_normalizer:
            style_reward_normalizer.update(d.cpu().numpy())
        else:
          style_reward = torch.exp(torch.tanh(0.3 * d)) - torch.exp(
            -1 * torch.ones_like(d)
          )
      else:
        raise ValueError("Unexpected style reward mapping specified")
      style_reward *= (1.0 - self.reward_lerp) * self.reward_coef
      task_reward = task_reward.unsqueeze(-1) * self.reward_lerp
      style_reward = torch.nan_to_num(style_reward, nan=0.0, posinf=0.0, neginf=0.0)
      task_reward = torch.nan_to_num(task_reward, nan=0.0, posinf=0.0, neginf=0.0)
      reward = style_reward + task_reward
      if was_training:
        self.train()
    return reward.squeeze(), style_reward.squeeze(), task_reward.squeeze()
