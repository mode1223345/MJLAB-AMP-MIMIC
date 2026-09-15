# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

from tensordict import TensorDict


def store_code_state(logdir, repositories) -> list:
  """Stub: git diff capture is optional for mjlab MHA-HIM training."""
  return []


def resolve_obs_groups(
  obs: TensorDict, obs_groups: dict[str, list[str]], default_sets: list[str]
) -> dict[str, list[str]]:
  """Validate observation configuration with ``actor`` as the primary set name.

  Example::

      {
          "actor": ["actor"],
          "critic": ["critic"],
      }
  """
  # check if actor observation set exists
  if "actor" not in obs_groups.keys():
    if "actor" in obs:
      obs_groups["actor"] = ["actor"]
      warnings.warn(
        "The observation configuration dictionary 'obs_groups' must contain "
        "the 'actor' key. As an observation group with the name 'actor' was "
        "found, this is assumed to be the observation set.",
        stacklevel=2,
      )
    else:
      raise ValueError(
        "The observation configuration dictionary 'obs_groups' must contain "
        f"the 'actor' key. Found keys: {list(obs_groups.keys())}"
      )

  # check all observation sets for valid observation groups
  for set_name, groups in list(obs_groups.items()):
    groups = list(groups)
    obs_groups[set_name] = groups
    if len(groups) == 0:
      msg = (
        f"The '{set_name}' key in the 'obs_groups' dictionary can not be an empty list."
      )
      if set_name in default_sets:
        if set_name not in obs:
          msg += (
            " Consider removing the key to default to the observations "
            "used for the 'actor' set."
          )
        else:
          msg += (
            f" Consider removing the key to default to the observation "
            f"'{set_name}' from the environment."
          )
      raise ValueError(msg)
    for group in groups:
      if group not in obs:
        raise ValueError(
          f"Observation '{group}' in observation set '{set_name}' not found "
          f"in the observations from the environment. Available "
          f"observations: {list(obs.keys())}"
        )

  # fill missing observation sets
  for default_set_name in default_sets:
    if default_set_name not in obs_groups.keys():
      if default_set_name in obs:
        obs_groups[default_set_name] = [default_set_name]
        warnings.warn(
          "The observation configuration dictionary 'obs_groups' must "
          f"contain the '{default_set_name}' key. As an observation group "
          f"with the name '{default_set_name}' was found, this is assumed "
          "to be the observation set.",
          stacklevel=2,
        )
      else:
        obs_groups[default_set_name] = list(obs_groups["actor"])
        warnings.warn(
          "The observation configuration dictionary 'obs_groups' must "
          f"contain the '{default_set_name}' key. As the configuration for "
          f"'{default_set_name}' is missing, the observations from the "
          "'actor' set are used.",
          stacklevel=2,
        )

  print("-" * 80)
  print("Resolved observation sets: ")
  for set_name, groups in obs_groups.items():
    print("\t", set_name, ": ", groups)
  print("-" * 80)

  return obs_groups
