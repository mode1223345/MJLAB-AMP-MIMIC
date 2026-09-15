"""Sagittal reflection for the N3 29-DoF AMP walking task.

Thin binding of the generic :mod:`symmetry` engine; kept as its own module
because the rl cfg references
``mjlab.tasks.amp.mdp.symmetry_n3:data_augmentation_func`` by string.
The critic tail uses ``grid_flip_dynamic``: the height-scan grid shape is
read from the env's ``terrain_scan`` sensor cfg at call time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from mjlab.asset_zoo.robots.N3.constants import N3_JOINT_NAMES
from mjlab.tasks.amp.mdp.symmetry import Mirror, SymmetrySpec

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_MIRROR = Mirror(
  SymmetrySpec(
    joint_names=N3_JOINT_NAMES, orientation="gravity", critic_tail="grid_flip_dynamic"
  )
)


def flip_dof(dof: torch.Tensor) -> torch.Tensor:
  """Swap limbs; yaw/roll axes reverse and pitch axes retain their sign."""
  return _MIRROR.flip_dof(dof)


def flip_actor_obs(obs: torch.Tensor) -> torch.Tensor:
  """Mirror commands, angular velocity, gravity, joints, and actions."""
  return _MIRROR.flip_actor_obs(obs)


def flip_critic_obs(obs: torch.Tensor, env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mirror the actor prefix, linear velocity, contacts, and terrain rays."""
  return _MIRROR.flip_critic_obs(obs, env)


@torch.no_grad()
def data_augmentation_func(
  env: ManagerBasedRlEnv,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Append mirrored PPO/HIM observations and actions to the batch."""
  return _MIRROR.data_augmentation_func(env, obs, actions)
