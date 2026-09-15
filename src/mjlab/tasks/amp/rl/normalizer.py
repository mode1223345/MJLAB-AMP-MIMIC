# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# SPDX-License-Identifier: BSD-3-Clause

"""Running mean/std normalizer used by the AMP discriminator."""

from __future__ import annotations

import numpy as np
import torch


class RunningMeanStd:
  def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] | int = ()):
    if isinstance(shape, int):
      shape = (shape,)
    self.mean = np.zeros(shape, np.float64)
    self.var = np.ones(shape, np.float64)
    self.count = epsilon

  def update(self, arr: np.ndarray) -> None:
    batch_mean = np.mean(arr, axis=0)
    batch_var = np.var(arr, axis=0)
    batch_count = arr.shape[0]
    self.update_from_moments(batch_mean, batch_var, batch_count)

  def update_from_moments(
    self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
  ) -> None:
    delta = batch_mean - self.mean
    tot_count = self.count + batch_count

    new_mean = self.mean + delta * batch_count / tot_count
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m_2 = (
      m_a
      + m_b
      + np.square(delta) * self.count * batch_count / (self.count + batch_count)
    )
    new_var = m_2 / (self.count + batch_count)

    self.mean = new_mean
    self.var = new_var
    self.count = batch_count + self.count

    if hasattr(self, "_cache_dirty"):
      self._cache_dirty = True


class Normalizer(RunningMeanStd):
  def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0):
    super().__init__(shape=input_dim)
    self.epsilon = epsilon
    self.clip_obs = clip_obs
    self._cached_mean_torch = None
    self._cached_std_torch = None
    self._cached_device = None
    self._cache_dirty = True

  def normalize(self, input):
    return np.clip(
      (input - self.mean) / np.sqrt(self.var + self.epsilon),
      -self.clip_obs,
      self.clip_obs,
    )

  def normalize_torch(self, input, device):
    if self._cache_dirty or self._cached_device != device:
      self._cached_mean_torch = torch.tensor(
        self.mean, device=device, dtype=torch.float32
      )
      self._cached_std_torch = torch.sqrt(
        torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32)
      )
      self._cached_device = device
      self._cache_dirty = False

    return torch.clamp(
      (input - self._cached_mean_torch) / self._cached_std_torch,
      -self.clip_obs,
      self.clip_obs,
    )
