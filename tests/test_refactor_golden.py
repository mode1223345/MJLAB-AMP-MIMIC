"""Golden regression tests for the structural refactor (and beyond).

Compares the current code against ``tests/golden/refactor_golden.json``:
task registry contents, env/rl cfg fields (train and play, including dict
key order), symmetry flip/augmentation outputs on seeded tensors, and policy
network parameter manifests (names, order, shapes).

If a test fails after an *intentional* config/structure change, regenerate
with ``uv run python scripts/tools/dump_cfg_golden.py --regen`` and include
the JSON diff in your change for review.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_DUMP_MODULE_PATH = (
  Path(__file__).parents[1] / "scripts" / "tools" / "dump_cfg_golden.py"
)
_spec = importlib.util.spec_from_file_location("dump_cfg_golden", _DUMP_MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_dump_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dump_module)
build_dump = _dump_module.build_dump

GOLDEN_PATH = Path(__file__).parent / "golden" / "refactor_golden.json"


def _first_difference(
  expected: Any, actual: Any, path: str = ""
) -> tuple[str, Any, Any]:
  """Return (path, expected, actual) of the first structural difference."""
  if isinstance(expected, dict) and isinstance(actual, dict):
    for key in expected:
      if key not in actual:
        return f"{path}.{key}", expected[key], "<missing>"
      diff = _first_difference(expected[key], actual[key], f"{path}.{key}")
      if diff[0] != "<none>":
        return diff
    for key in actual:
      if key not in expected:
        return f"{path}.{key}", "<unexpected>", actual[key]
    return "<none>", None, None
  if isinstance(expected, list) and isinstance(actual, list):
    if len(expected) != len(actual):
      return f"{path} (len)", len(expected), len(actual)
    for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
      diff = _first_difference(e, a, f"{path}[{i}]")
      if diff[0] != "<none>":
        return diff
    return "<none>", None, None
  same_kind = type(expected) is type(actual) or (
    isinstance(expected, (int, float))
    and isinstance(actual, (int, float))
    and not isinstance(expected, bool)
    and not isinstance(actual, bool)
  )
  if not same_kind or expected != actual:
    return path or "<root>", expected, actual
  return "<none>", None, None


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
  return json.loads(GOLDEN_PATH.read_text())


@pytest.fixture(scope="module")
def current() -> dict[str, Any]:
  return build_dump()


def test_task_registry_list(golden: dict, current: dict) -> None:
  assert current["task_ids"] == golden["task_ids"]


def test_env_and_rl_cfgs(golden: dict, current: dict) -> None:
  expected_tasks = golden["tasks"]
  actual_tasks = current["tasks"]
  assert sorted(actual_tasks) == sorted(expected_tasks)
  for task_id, expected in expected_tasks.items():
    actual = actual_tasks[task_id]
    for section in ("env_cfg_train", "env_cfg_play", "rl_cfg"):
      diff_path, exp, act = _first_difference(
        expected[section], actual[section], f"{task_id}.{section}"
      )
      assert diff_path == "<none>", (
        f"Golden mismatch at {diff_path}:\n"
        f"  expected: {json.dumps(exp)[:400]}\n"
        f"  actual:   {json.dumps(act)[:400]}\n"
        "If intentional, regenerate with "
        "`uv run python scripts/tools/dump_cfg_golden.py --regen`."
      )


def test_symmetry(golden: dict, current: dict) -> None:
  expected = golden["symmetry"]
  actual = current["symmetry"]
  assert sorted(actual) == sorted(expected)
  for key, expected_entry in expected.items():
    for tensor_name, expected_tensor in expected_entry.items():
      diff_path, exp, act = _first_difference(
        expected_tensor,
        actual[key][tensor_name],
        f"symmetry.{key}.{tensor_name}",
      )
      assert diff_path == "<none>", (
        f"Symmetry golden mismatch at {diff_path}:\n"
        f"  expected: {json.dumps(exp)[:200]}\n"
        f"  actual:   {json.dumps(act)[:200]}"
      )


def test_policy_manifests(golden: dict, current: dict) -> None:
  expected = golden["policy_manifests"]
  actual = current["policy_manifests"]
  assert sorted(actual) == sorted(expected)
  for net, expected_manifest in expected.items():
    actual_manifest = actual[net]
    if expected_manifest != actual_manifest:
      expected_names = [p[0] for p in expected_manifest]
      actual_names = [p[0] for p in actual_manifest]
      if expected_names == actual_names:
        first_bad = next(
          i
          for i, (e, a) in enumerate(
            zip(expected_manifest, actual_manifest, strict=True)
          )
          if e != a
        )
        detail = f"shape/dtype drift at index {first_bad}: {expected_manifest[first_bad]} vs {actual_manifest[first_bad]}"
      else:
        detail = (
          f"name/order drift:\n  expected: {expected_names}\n  actual:   {actual_names}"
        )
      pytest.fail(f"Parameter manifest mismatch for {net}: {detail}")
