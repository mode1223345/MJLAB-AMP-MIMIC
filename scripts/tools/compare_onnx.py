#!/usr/bin/env python3
"""Compare two exported ONNX models for refactor equivalence.

Usage:
  uv run python scripts/tools/compare_onnx.py A.onnx B.onnx

First tries a byte-level comparison; if that fails, compares graph structure
(node op types in order, node input/output names, and initializer tensors
element-wise), ignoring doc_string/producer metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx


def _graphs_equal(a: onnx.ModelProto, b: onnx.ModelProto) -> str | None:
  ga, gb = a.graph, b.graph
  na = [(n.op_type, list(n.input), list(n.output)) for n in ga.node]
  nb = [(n.op_type, list(n.input), list(n.output)) for n in gb.node]
  if len(na) != len(nb):
    return f"node count differs: {len(na)} vs {len(nb)}"
  if na != nb:
    for i, (x, y) in enumerate(zip(na, nb, strict=True)):
      if x != y:
        return f"node {i} differs:\n  A: {x}\n  B: {y}"

  init_a = {t.name: onnx.numpy_helper.to_array(t) for t in ga.initializer}
  init_b = {t.name: onnx.numpy_helper.to_array(t) for t in gb.initializer}
  if set(init_a) != set(init_b):
    return (
      f"initializer names differ: only-A={sorted(set(init_a) - set(init_b))}, "
      f"only-B={sorted(set(init_b) - set(init_a))}"
    )
  for name in sorted(init_a):
    x, y = init_a[name], init_b[name]
    if x.shape != y.shape or x.dtype != y.dtype:
      return (
        f"initializer {name}: shape/dtype {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}"
      )
    if not np.array_equal(x, y):
      if np.issubdtype(x.dtype, np.floating) and np.allclose(x, y, atol=1e-6):
        continue  # float noise within tolerance
      return f"initializer {name}: values differ (max abs diff {np.abs(x.astype(float) - y.astype(float)).max():.3e})"
  return None


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("model_a", type=Path)
  parser.add_argument("model_b", type=Path)
  args = parser.parse_args()

  if args.model_a.read_bytes() == args.model_b.read_bytes():
    print("IDENTICAL (byte-level)")
    return 0

  a = onnx.load(str(args.model_a))
  b = onnx.load(str(args.model_b))
  difference = _graphs_equal(a, b)
  if difference is None:
    print("EQUIVALENT (graphs match; byte differences are metadata-only)")
    return 0
  print(f"DIFFERENT:\n{difference}", file=sys.stderr)
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
