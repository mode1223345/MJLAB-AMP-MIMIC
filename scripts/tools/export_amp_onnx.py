#!/usr/bin/env python3
"""Export AMP HIM checkpoint (N3) to Gazebo-compatible ONNX.

Labubu: ``obs [1, 360] -> actions [1, 21]`` (5 × 72 history, HIM folded in).
F5 9-DoF: ``obs [1, 175] -> actions [1, 9]`` (5 × 35 history).
F5 11-DoF: ``obs [1, 205] -> actions [1, 11]`` (5 × 41 history).

By default remaps joints to Isaac Lab / Gazebo URDF order used by existing
Noetix deploy artifacts.

Example::

  uv run python scripts/tools/export_amp_labubu_onnx.py \\
    --checkpoint-file logs/rsl_rl/amp_labubu_walk/2026-08-10_22-00-50/model_19999.pt

  uv run python scripts/tools/export_amp_labubu_onnx.py \\
    --checkpoint-file logs/rsl_rl/amp_f5_11dof_walk/.../model_15000.pt \\
    --task Mjlab-Amp-Flat-F5-11DoF-Walk
"""

from __future__ import annotations

import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.amp.rl.onnx_export import (
  export_amp_him_policy_as_onnx,
  resolve_deploy_joint_names,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

_RUN_STAMP = re.compile(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})$")


def default_deploy_onnx_path(checkpoint: Path, joint_order: str) -> Path:
  """``…/amp_labubu_walk/2026-08-12_21-15-20/model_80000.pt``
  → ``amp_labubu_walk_260812-211520_model_80000_deploy.onnx``.
  """
  ckpt = checkpoint.resolve()
  m = _RUN_STAMP.fullmatch(ckpt.parent.name)
  if m:
    y, mo, d, h, mi, s = m.groups()
    stamp = f"{y[2:]}{mo}{d}-{h}{mi}{s}"
  else:
    stamp = ckpt.parent.name.replace("_", "-")
  order = "deploy" if joint_order == "deploy" else "mjlab"
  return ckpt.with_name(f"{ckpt.parent.parent.name}_{stamp}_{ckpt.stem}_{order}.onnx")


def copy_to_experiment_exported(checkpoint: Path, onnx_path: Path) -> Path | None:
  """Copy ONNX into ``logs/rsl_rl/<exp>/exported/`` (no-op if already there)."""
  exp_dir = checkpoint.resolve().parent.parent
  exported_dir = exp_dir / "exported"
  dest = exported_dir / onnx_path.name
  if dest.resolve() == onnx_path.resolve():
    return None
  exported_dir.mkdir(parents=True, exist_ok=True)
  shutil.copy2(onnx_path, dest)
  return dest


@dataclass
class ExportConfig:
  checkpoint_file: str
  """Path to ``model_*.pt`` from an AMP walk experiment."""
  output_file: str | None = None
  """Defaults to ``{exp}_{yymmdd}-{hhmmss}_{stem}_deploy.onnx`` beside the ckpt."""
  task: str = "Mjlab-Amp-Flat-N3-Walk"
  device: str = "cpu"
  verbose: bool = False
  joint_order: Literal["deploy", "mjlab"] = "deploy"
  """``deploy``: Isaac/Gazebo URDF order (default). ``mjlab``: MuJoCo order."""


def main(cfg: ExportConfig) -> None:
  ckpt = Path(cfg.checkpoint_file).resolve()
  if not ckpt.is_file():
    raise FileNotFoundError(ckpt)

  if cfg.output_file:
    out = Path(cfg.output_file).resolve()
  else:
    out = default_deploy_onnx_path(ckpt, cfg.joint_order)
  out.parent.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(cfg.task, play=True)
  env_cfg.scene.num_envs = 1
  # Speed up MotionLoader construction for export-only runs.
  env_cfg.amp_num_preload_transitions = 8
  # Disable startup DR so ONNX metadata uses clean default joint poses.
  for key in (
    "add_joint_default_pos",
    "foot_friction",
    "base_com",
    "add_base_mass",
  ):
    env_cfg.events.pop(key, None)
  for key in (
    "randomize_actuator_gains",
    "randomize_ankle_armature",
    "randomize_ankle_friction",
  ):
    env_cfg.events.pop(key, None)
  agent_cfg = asdict(load_rl_cfg(cfg.task))

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))
  runner_cls = load_runner_cls(cfg.task)
  assert runner_cls is not None
  runner = runner_cls(env, agent_cfg, log_dir=None, device=cfg.device)
  runner.load(str(ckpt), load_cfg={"load_optimizer": False}, map_location=cfg.device)

  train_joint_names = tuple(env.unwrapped.scene["robot"].joint_names)
  deploy_joint_names = resolve_deploy_joint_names(train_joint_names)

  onnx_path = Path(
    export_amp_him_policy_as_onnx(
      runner.alg.policy,
      str(out.parent),
      filename=out.name,
      verbose=cfg.verbose,
      env=env.unwrapped,
      run_path=str(ckpt.parent.name),
      train_joint_names=train_joint_names,
      deploy_joint_names=deploy_joint_names,
      joint_order=cfg.joint_order,
    )
  )
  one_step = int(runner.alg.policy.num_one_step_obs)
  hist = int(getattr(runner.alg.policy.him_estimator, "temporal_steps", 1))
  n_act = len(train_joint_names)
  print(f"[OK] Wrote {onnx_path}")
  copied = copy_to_experiment_exported(ckpt, onnx_path)
  if copied is not None:
    print(f"[OK] Copied {copied}")
  print(f"  I/O: obs[1, {one_step * hist}] -> actions[1, {n_act}]")
  print(f"  joint_order: {cfg.joint_order}")
  if cfg.joint_order == "deploy":
    print("  metadata joint_names: Isaac/Gazebo URDF order")
    print(
      "   ", ",".join(deploy_joint_names[: min(5, len(deploy_joint_names))]), ",..."
    )
  else:
    print("  metadata joint_names: mjlab MuJoCo order")
  env.close()


if __name__ == "__main__":
  main(tyro.cli(ExportConfig, config=mjlab.TYRO_FLAGS))
