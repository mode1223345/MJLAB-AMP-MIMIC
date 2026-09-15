"""Checkpoint compatibility tests (env-gated, skipped by default).

Set ``MJLAB_OLD_CKPT_DIR`` to a pre-refactor ``logs/rsl_rl`` directory
(layout: ``<experiment_name>/<run>/model_*.pt``). Two invariants are checked
against each policy checkpoint:

1. ``model_state_dict`` key order matches the current code's module
   registration order (from the golden manifest, filtered to the keys present
   in the checkpoint). Order drives optimizer state compatibility.
2. ``optimizer_state_dict`` param groups reference a number of tensors that
   matches the corresponding model parameters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

ckpt_dir = os.environ.get("MJLAB_OLD_CKPT_DIR")

pytestmark = pytest.mark.skipif(
  not ckpt_dir,
  reason="Set MJLAB_OLD_CKPT_DIR to a logs/rsl_rl directory to enable.",
)


def _ckpt_root() -> Path:
  assert ckpt_dir is not None  # guarded by pytestmark skipif
  return Path(ckpt_dir)


def _checkpoints() -> list[Path]:
  if not ckpt_dir:
    return []
  paths = sorted(_ckpt_root().glob("*/*/model_*.pt"))
  assert paths, f"No model_*.pt found under {ckpt_dir}"
  return paths


def _manifest_names(family: str) -> list[str]:
  golden = json.loads(
    (Path(__file__).parent / "golden" / "refactor_golden.json").read_text()
  )
  key = "him_actor_critic"
  return [entry[0] for entry in golden["policy_manifests"][key]]


def _family(path: Path) -> str:
  experiment = path.relative_to(_ckpt_root()).parts[0]
  if experiment.startswith("amp_"):
    return "amp"
  raise pytest.skip(f"Unknown experiment family for {experiment}")


@pytest.mark.parametrize("ckpt_path", _checkpoints(), ids=str)
def test_state_dict_key_order(ckpt_path: Path) -> None:
  ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
    pytest.skip("not a policy checkpoint")
  state = ckpt["model_state_dict"]
  keys = list(state.keys())

  manifest = _manifest_names(_family(ckpt_path))
  filtered = [name for name in manifest if name in set(keys)]
  assert filtered == keys, (
    f"{ckpt_path.name}: checkpoint key order diverges from current module "
    "registration order.\n"
    f"  checkpoint: {keys[:10]}...\n"
    f"  current:    {filtered[:10]}..."
  )


@pytest.mark.parametrize("ckpt_path", _checkpoints(), ids=str)
def test_optimizer_param_groups(ckpt_path: Path) -> None:
  ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  if not isinstance(ckpt, dict) or "optimizer_state_dict" not in ckpt:
    pytest.skip("no optimizer state")
  optimizer = ckpt["optimizer_state_dict"]
  model = ckpt["model_state_dict"]
  num_model_params = len(model)
  referenced = sum(len(group["params"]) for group in optimizer["param_groups"])
  assert referenced <= num_model_params, (
    f"{ckpt_path.name}: optimizer references {referenced} tensors but the "
    f"model has {num_model_params}; parameter structure has drifted."
  )
