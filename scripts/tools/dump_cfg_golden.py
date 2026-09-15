#!/usr/bin/env python3
"""Dump equivalence golden files for the structural refactor.

Serializes, for every registered task: the env cfg (train and play) and the
rl cfg as normalized dicts, plus symmetry-module outputs on seeded random
tensors and policy network parameter manifests. ``tests/test_refactor_golden.py``
compares a fresh dump against ``tests/golden/refactor_golden.json``.

Regenerate intentionally (e.g. a deliberate config change) with:

  uv run python scripts/tools/dump_cfg_golden.py --regen

and review the JSON diff as part of the change.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from tensordict import TensorDict

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "refactor_golden.json"

SEED = 0


def normalize(obj: Any) -> Any:
  """Recursively convert cfg-like objects into JSON-friendly primitives.

  Dict insertion order is preserved everywhere (term ordering is behaviorally
  significant: managers iterate cfg dicts), so the golden JSON pins key order.
  """
  if isinstance(obj, enum.Enum):  # before str/int: str- and int-enums unwrap
    return normalize(obj.value)
  if obj is None or isinstance(obj, (bool, str, int)):
    return obj
  if isinstance(obj, float):
    return obj
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    return {
      f.name: normalize(getattr(obj, f.name)) for f in dataclasses.fields(type(obj))
    }
  if isinstance(obj, dict):
    return {str(k): normalize(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple, set)):
    return [normalize(v) for v in obj]
  if isinstance(obj, Path):
    return str(obj)
  if isinstance(obj, slice):
    return f"slice({obj.start!r},{obj.stop!r},{obj.step!r})"
  if callable(obj):
    return f"{obj.__module__}:{obj.__qualname__}"
  if hasattr(obj, "item"):  # numpy scalars
    return obj.item()
  raise TypeError(f"Unhandled type in golden dump: {type(obj)!r} ({obj!r})")


def dump_tasks() -> dict[str, Any]:
  import mjlab.tasks  # noqa: F401 -- populate the registry
  from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

  tasks: dict[str, Any] = {}
  for task_id in list_tasks():
    entry: dict[str, Any] = {"task_id": task_id}
    for mode, play in (("train", False), ("play", True)):
      cfg = load_env_cfg(task_id, play=play)
      entry[f"env_cfg_{mode}"] = normalize(cfg)
    entry["rl_cfg"] = normalize(load_rl_cfg(task_id))
    tasks[task_id] = entry
  return tasks


# ---------------------------------------------------------------------------
# Symmetry golden data
# ---------------------------------------------------------------------------


def _n3_style_fake_env(scene_cfg: Any) -> SimpleNamespace:
  """Fake env for the n3-style driver (reads terrain_scan sensor cfg)."""
  unwrapped = SimpleNamespace(cfg=SimpleNamespace(scene=scene_cfg))
  return SimpleNamespace(unwrapped=unwrapped)


def _tensor_recap(t: Any) -> Any:
  assert isinstance(t, torch.Tensor)
  return normalize(t.detach().cpu().tolist())


def dump_symmetry() -> dict[str, Any]:
  from mjlab.sensor import GridPatternCfg, RayCastSensorCfg
  from mjlab.tasks.amp.mdp import symmetry_n3
  from mjlab.tasks.registry import load_env_cfg

  torch.manual_seed(SEED)

  def randn(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float32)

  out: dict[str, Any] = {}

  # N3: no env needed for flips; augmentation reads the real terrain_scan cfg.
  n3_env_cfg = load_env_cfg("Mjlab-Amp-Flat-N3-Walk")
  n = 29
  actor_dim = 9 + 3 * n
  scan_cfg = next(s for s in n3_env_cfg.scene.sensors if s.name == "terrain_scan")
  assert isinstance(scan_cfg, RayCastSensorCfg)
  pattern = scan_cfg.pattern
  assert isinstance(pattern, GridPatternCfg)
  num_x = round(pattern.size[0] / pattern.resolution) + 1
  num_y = round(pattern.size[1] / pattern.resolution) + 1
  critic_dim = actor_dim + 5 + num_x * num_y
  env = cast(Any, _n3_style_fake_env(n3_env_cfg.scene))
  out["n3"] = {
    "num_joints": n,
    "actor_dim": actor_dim,
    "critic_dim": critic_dim,
    "scan_num_x": num_x,
    "scan_num_y": num_y,
    "flip_dof": _tensor_recap(symmetry_n3.flip_dof(randn(4, n))),
    "flip_actor_obs_single": _tensor_recap(
      symmetry_n3.flip_actor_obs(randn(4, actor_dim))
    ),
    "flip_actor_obs_flatten": _tensor_recap(
      symmetry_n3.flip_actor_obs(randn(4, 5 * actor_dim))
    ),
    "flip_critic_obs_single": _tensor_recap(
      symmetry_n3.flip_critic_obs(randn(4, critic_dim), env)
    ),
    "flip_critic_obs_history": _tensor_recap(
      symmetry_n3.flip_critic_obs(randn(4, 3, critic_dim), env)
    ),
  }
  obs = TensorDict(
    {
      "actor": randn(4, 5 * actor_dim),
      "critic": randn(4, 5 * critic_dim),
      "next_obs": randn(4, actor_dim),
    },
    batch_size=[4],
  )
  actions = randn(4, n)
  obs_aug, actions_aug = symmetry_n3.data_augmentation_func(env, obs, actions)
  assert obs_aug is not None and actions_aug is not None
  out["n3"]["augment_actor"] = _tensor_recap(obs_aug["actor"])
  out["n3"]["augment_critic"] = _tensor_recap(obs_aug["critic"])
  out["n3"]["augment_next_obs"] = _tensor_recap(obs_aug["next_obs"])
  out["n3"]["augment_actions"] = _tensor_recap(actions_aug)

  return out


# ---------------------------------------------------------------------------
# Policy parameter manifests
# ---------------------------------------------------------------------------


def _manifest(module: torch.nn.Module) -> list[list[Any]]:
  return [
    [name, list(param.shape), str(param.dtype)]
    for name, param in module.named_parameters()
  ]


def dump_policy_manifests() -> dict[str, Any]:
  from mjlab.tasks.amp.rl.discriminator import Discriminator
  from mjlab.tasks.amp.rl.him_actor_critic import HimActorCritic

  torch.manual_seed(SEED)

  def randn(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float32)

  # Canonical synthetic dims (same for old and new code; not task dims).
  n = 9
  steps = 5
  one_step = 8 + 3 * n  # 35
  actor_obs = one_step * steps
  critic_one_step = one_step + 5
  critic_obs = critic_one_step * steps
  obs = {
    "actor": randn(2, actor_obs),
    "critic": randn(2, critic_obs),
  }
  obs_groups = {"actor": ("actor",), "critic": ("critic",)}

  him = HimActorCritic(
    obs,
    obs_groups,
    num_actions=n,
    num_one_step_obs=one_step,
    actor_obs_normalization=True,
    critic_obs_normalization=True,
    actor_hidden_dims=(64, 32),
    critic_hidden_dims=(64, 32),
    encoder_hidden_dims=[32, 16],
    projector_hidden_dims=[32, 32],
    projector_output_dim=32,
    command_dim=3,
    estimate_dim=3,
    activation="elu",
  )

  disc = Discriminator(
    observation_dim=critic_one_step,
    observation_horizon=4,
    device="cpu",
    reward_coef=0.5,
    reward_lerp=0.3,
    shape=(32, 16),
    style_reward_function="quad_mapping",
  )

  return {
    "him_actor_critic": _manifest(him),
    "discriminator": _manifest(disc),
  }


def build_dump() -> dict[str, Any]:
  tasks = dump_tasks()
  return {
    "seed": SEED,
    "task_ids": sorted(tasks),
    "tasks": tasks,
    "symmetry": dump_symmetry(),
    "policy_manifests": dump_policy_manifests(),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output",
    type=Path,
    default=GOLDEN_PATH,
    help="Output JSON path (default: tests/golden/refactor_golden.json).",
  )
  parser.add_argument(
    "--regen",
    action="store_true",
    help="Overwrite the golden file instead of failing if it exists.",
  )
  args = parser.parse_args()

  if args.output.exists() and not args.regen:
    raise SystemExit(
      f"{args.output} already exists; pass --regen to overwrite deliberately."
    )

  dump = build_dump()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(dump, indent=1) + "\n")
  print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
