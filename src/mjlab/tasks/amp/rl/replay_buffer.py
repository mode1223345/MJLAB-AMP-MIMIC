# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-size AMP policy replay buffer."""

from __future__ import annotations

import torch


class ReplayBuffer:
  """Fixed-size buffer to store AMP observation trajectories."""

  def __init__(self, obs_dim, obs_horizon, buffer_size, device):
    self.state_buf = torch.zeros(buffer_size, obs_horizon, obs_dim, device=device)
    self.buffer_size = buffer_size
    self.device = device
    self.step = 0
    self.num_samples = 0

  def get_buffer_size(self):
    return self.buffer_size

  def insert(self, state_buf):
    num_states = state_buf.shape[0]
    end_idx = self.step + num_states
    if end_idx > self.buffer_size:
      first = self.buffer_size - self.step
      self.state_buf[self.step : self.buffer_size] = state_buf[:first]
      self.state_buf[: end_idx - self.buffer_size] = state_buf[first:]
    else:
      self.state_buf[self.step : end_idx] = state_buf

    self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
    self.step = (self.step + num_states) % self.buffer_size

  def feed_forward_generator(self, num_mini_batch, mini_batch_size):
    for _ in range(num_mini_batch):
      sample_idxs = torch.randint(
        0, self.num_samples, (mini_batch_size,), device=self.device
      )
      yield self.state_buf[sample_idxs, :]
