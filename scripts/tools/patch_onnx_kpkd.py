#!/usr/bin/env python3
"""View or patch ``joint_stiffness`` / ``joint_damping`` in deploy ONNX metadata.

kp/kd live in ONNX ``metadata_props`` (comma-separated floats aligned with
``joint_names``). They are not part of the neural network weights.

Examples::

  # Show current metadata.
  uv run python scripts/tools/patch_onnx_kpkd.py --onnx-file model_deploy.onnx --show

  # Reload kp/kd from the N3 robot cfg (by joint name, any order).
  uv run python scripts/tools/patch_onnx_kpkd.py --onnx-file model.onnx \\
    --from-n3-constants --rescale-action-scale --in-place

  # Scale all joints.
  uv run python scripts/tools/patch_onnx_kpkd.py --onnx-file model.onnx \\
    --kp-scale 1.2 --kd-scale 1.0 --output-file model_kp120.onnx

  # Per-joint overrides (JSON keys are joint names or regex patterns).
  uv run python scripts/tools/patch_onnx_kpkd.py --onnx-file model.onnx \\
    --overrides-json overrides.json --in-place
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import onnx
import tyro
from tyro.conf import Positional

import mjlab
from mjlab.rl.exporter_utils import list_to_csv_str


def _parse_csv_floats(value: str) -> list[float]:
  return [float(x.strip()) for x in value.split(",") if x.strip()]


def _metadata_dict(model: onnx.ModelProto) -> dict[str, str]:
  return {prop.key: prop.value for prop in model.metadata_props}


def _set_metadata(model: onnx.ModelProto, key: str, value: str) -> None:
  for prop in model.metadata_props:
    if prop.key == key:
      prop.value = value
      return
  entry = onnx.StringStringEntryProto()
  entry.key = key
  entry.value = value
  model.metadata_props.append(entry)


def _require_metadata(meta: dict[str, str], key: str) -> str:
  if key not in meta:
    raise KeyError(f"ONNX metadata missing '{key}'")
  return meta[key]


def _match_joint(pattern: str, joint_name: str) -> bool:
  if pattern == joint_name:
    return True
  if any(ch in pattern for ch in ".*+?[]()|^$"):
    return re.fullmatch(pattern, joint_name) is not None
  return False


def _apply_overrides(
  joint_names: list[str],
  values: list[float],
  overrides: dict[str, float],
  *,
  label: str,
) -> list[float]:
  if not overrides:
    return values
  out = values.copy()
  for pattern, target in overrides.items():
    matched = [i for i, name in enumerate(joint_names) if _match_joint(pattern, name)]
    if not matched:
      raise ValueError(f"{label}: pattern {pattern!r} matched no joints")
    for i in matched:
      out[i] = float(target)
  return out


def _kpkd_by_joint_from_n3() -> tuple[dict[str, float], dict[str, float]]:
  from mjlab.asset_zoo.robots.N3.constants import get_n3_robot_cfg
  from mjlab.entity.entity import Entity

  robot = Entity(get_n3_robot_cfg())
  mj_model = robot.spec.compile()

  joint_name_to_ctrl_id: dict[str, int] = {}
  for actuator in robot.spec.actuators:
    joint_name = actuator.target.split("/")[-1]
    joint_name_to_ctrl_id[joint_name] = actuator.id

  kp_by_joint: dict[str, float] = {}
  kd_by_joint: dict[str, float] = {}
  for jname, ctrl_id in joint_name_to_ctrl_id.items():
    kp_by_joint[jname] = float(mj_model.actuator_gainprm[ctrl_id, 0])
    kd_by_joint[jname] = float(-mj_model.actuator_biasprm[ctrl_id, 2])
  return kp_by_joint, kd_by_joint


def _lookup_by_names(
  joint_names: list[str],
  table: dict[str, float],
  *,
  label: str,
) -> list[float]:
  missing = [n for n in joint_names if n not in table]
  if missing:
    raise KeyError(f"{label}: joints missing from lookup table: {missing}")
  return [table[n] for n in joint_names]


def _print_table(joint_names: list[str], kp: list[float], kd: list[float]) -> None:
  print(f"{'joint':32s} {'kp':>8s} {'kd':>8s}")
  print("-" * 52)
  for name, kpi, kdi in zip(joint_names, kp, kd, strict=True):
    print(f"{name:32s} {kpi:8.3f} {kdi:8.3f}")


@dataclass
class PatchConfig:
  onnx_file: Annotated[Path, Positional]
  """Path to deploy ONNX."""

  show: bool = False
  """Print metadata and exit (default when no patch flags are set)."""

  output_file: str | None = None
  """Output path. Defaults to ``<stem>_kpkd.onnx`` when patching."""

  in_place: bool = False
  """Overwrite ``onnx_file`` in place."""

  kp_scale: float = 1.0
  """Multiply all kp values."""

  kd_scale: float = 1.0
  """Multiply all kd values."""

  from_n3_constants: bool = False
  """Replace kp/kd from the N3 robot cfg (matched by joint name)."""

  overrides_json: str | None = None
  """JSON with optional ``joint_stiffness`` / ``joint_damping`` override maps."""

  rescale_action_scale: bool = False
  """If kp changes, scale ``action_scale`` by old_kp/new_kp per joint."""


def _load_overrides(path: str | None) -> tuple[dict[str, float], dict[str, float]]:
  if path is None:
    return {}, {}
  data = json.loads(Path(path).read_text(encoding="utf-8"))
  kp = {str(k): float(v) for k, v in data.get("joint_stiffness", {}).items()}
  kd = {str(k): float(v) for k, v in data.get("joint_damping", {}).items()}
  return kp, kd


def _will_patch(
  cfg: PatchConfig, kp_overrides: dict[str, float], kd_overrides: dict[str, float]
) -> bool:
  return (
    cfg.from_n3_constants
    or cfg.kp_scale != 1.0
    or cfg.kd_scale != 1.0
    or bool(kp_overrides)
    or bool(kd_overrides)
  )


def main(cfg: PatchConfig) -> None:
  onnx_path = cfg.onnx_file.resolve()
  if not onnx_path.is_file():
    raise FileNotFoundError(onnx_path)

  model = onnx.load(str(onnx_path))
  meta = _metadata_dict(model)

  joint_names = _require_metadata(meta, "joint_names").split(",")
  old_kp = _parse_csv_floats(_require_metadata(meta, "joint_stiffness"))
  old_kd = _parse_csv_floats(_require_metadata(meta, "joint_damping"))
  if len(joint_names) != len(old_kp) or len(joint_names) != len(old_kd):
    raise ValueError(
      "joint_names / joint_stiffness / joint_damping length mismatch: "
      f"{len(joint_names)} / {len(old_kp)} / {len(old_kd)}"
    )

  kp_overrides, kd_overrides = _load_overrides(cfg.overrides_json)
  do_patch = _will_patch(cfg, kp_overrides, kd_overrides)
  if cfg.show or not do_patch:
    print(f"[INFO] {onnx_path}")
    if "joint_order" in meta:
      print(f"  joint_order: {meta['joint_order']}")
    _print_table(joint_names, old_kp, old_kd)
    if not do_patch:
      return

  new_kp = old_kp.copy()
  new_kd = old_kd.copy()

  if cfg.from_n3_constants:
    kp_table, kd_table = _kpkd_by_joint_from_n3()
    new_kp = _lookup_by_names(joint_names, kp_table, label="joint_stiffness")
    new_kd = _lookup_by_names(joint_names, kd_table, label="joint_damping")

  new_kp = _apply_overrides(joint_names, new_kp, kp_overrides, label="joint_stiffness")
  new_kd = _apply_overrides(joint_names, new_kd, kd_overrides, label="joint_damping")

  if cfg.kp_scale != 1.0:
    new_kp = [v * cfg.kp_scale for v in new_kp]
  if cfg.kd_scale != 1.0:
    new_kd = [v * cfg.kd_scale for v in new_kd]

  _set_metadata(model, "joint_stiffness", list_to_csv_str(new_kp))
  _set_metadata(model, "joint_damping", list_to_csv_str(new_kd))

  if cfg.rescale_action_scale and "action_scale" in meta:
    old_scale = _parse_csv_floats(meta["action_scale"])
    if len(old_scale) != len(new_kp):
      raise ValueError("action_scale length does not match joint count")
    new_scale = [
      old_scale[i] * old_kp[i] / new_kp[i] if new_kp[i] != 0.0 else old_scale[i]
      for i in range(len(new_kp))
    ]
    _set_metadata(model, "action_scale", list_to_csv_str(new_scale))

  if cfg.in_place:
    out_path = onnx_path
  elif cfg.output_file:
    out_path = Path(cfg.output_file).resolve()
  else:
    out_path = onnx_path.with_name(f"{onnx_path.stem}_kpkd{onnx_path.suffix}")

  out_path.parent.mkdir(parents=True, exist_ok=True)
  if out_path == onnx_path:
    tmp = onnx_path.with_suffix(".onnx.tmp")
    onnx.save(model, str(tmp))
    tmp.replace(out_path)
  else:
    onnx.save(model, str(out_path))

  print(f"[OK] Wrote {out_path}")
  print("[NEW]")
  _print_table(joint_names, new_kp, new_kd)


if __name__ == "__main__":
  main(tyro.cli(PatchConfig, config=mjlab.TYRO_FLAGS))
