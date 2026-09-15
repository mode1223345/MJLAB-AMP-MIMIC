"""Velocity command with optional per-axis zeroing (Labubu AMP)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityWithZeroCommand(CommandTerm):
  """SE(2) velocity command with standing and per-axis zero probabilities."""

  cfg: UniformVelocityWithZeroCommandCfg

  def __init__(self, cfg: UniformVelocityWithZeroCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_zero_vel_x_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_zero_vel_y_env = torch.zeros_like(self.is_zero_vel_x_env)
    self.is_zero_vel_yaw_env = torch.zeros_like(self.is_zero_vel_x_env)
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the Viser viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_all_envs: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

    # Native viewer keyboard teleop (see on_key).
    self._keyboard_enabled = False
    self._keyboard_cmd = torch.zeros(3, device=self.device)
    self._keyboard_scale = 0.5  # fraction of each axis max range
    self._keyboard_apply_all = True

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2],
        dim=-1,
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    self.is_zero_vel_x_env[env_ids] = (
      r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[0]
    )
    self.is_zero_vel_y_env[env_ids] = (
      r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[1]
    )
    self.is_zero_vel_yaw_env[env_ids] = (
      r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[2]
    )
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    # Pure function of the masks set at resample time; refreshing all envs is
    # safe (idempotent zeroing), matching upstream's velocity command.
    del env_ids
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0
    self.vel_command_b[self.is_zero_vel_x_env, 0] = 0.0
    self.vel_command_b[self.is_zero_vel_y_env, 1] = 0.0
    self.vel_command_b[self.is_zero_vel_yaw_env, 2] = 0.0

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    del on_change, request_action  # Unused; kept for CommandTerm API.
    from viser import Icon

    ranges = self.cfg.ranges
    axes = [
      ("lin_vel_x", max(abs(ranges.lin_vel_x[0]), abs(ranges.lin_vel_x[1]))),
      ("lin_vel_y", max(abs(ranges.lin_vel_y[0]), abs(ranges.lin_vel_y[1]))),
      ("ang_vel_z", max(abs(ranges.ang_vel_z[0]), abs(ranges.ang_vel_z[1]))),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      # Default on: with many envs it is easy to edit the wrong selected robot.
      all_envs = server.gui.add_checkbox("Apply to all envs", initial_value=True)

      for label, limit in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=limit,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-limit,
          max=limit,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    self._joystick_enabled = enabled
    self._joystick_all_envs = all_envs
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def on_key(self, key: int) -> bool:
    """Native-viewer keyboard teleop for twist commands.

    Keys (avoid conflicting with native viewer builtins)::

      T     toggle keyboard teleop on/off
      I / K +/− lin_vel_x (forward / back)
      J / L +/− lin_vel_y (left / right strafe)
      U / O +/− ang_vel_z (yaw)
      G     zero command
      [ / ] decrease / increase speed scale (0.1–1.0)

    Returns True if the key was handled.
    """
    from mjlab.viewer.native.keys import (
      KEY_G,
      KEY_I,
      KEY_J,
      KEY_K,
      KEY_L,
      KEY_LEFT_BRACKET,
      KEY_O,
      KEY_RIGHT_BRACKET,
      KEY_T,
      KEY_U,
    )

    ranges = self.cfg.ranges
    max_x = max(abs(ranges.lin_vel_x[0]), abs(ranges.lin_vel_x[1]))
    max_y = max(abs(ranges.lin_vel_y[0]), abs(ranges.lin_vel_y[1]))
    max_z = max(abs(ranges.ang_vel_z[0]), abs(ranges.ang_vel_z[1]))

    if key == KEY_T:
      self._keyboard_enabled = not self._keyboard_enabled
      if self._keyboard_enabled:
        self._keyboard_cmd.zero_()
        print(
          "[twist teleop] ON  | I/K:vx  J/L:vy  U/O:yaw  G:zero  [/]:scale  "
          f"(scale={self._keyboard_scale:.1f})"
        )
      else:
        print("[twist teleop] OFF (commands resample again)")
      return True

    if not self._keyboard_enabled:
      return False

    if key == KEY_G:
      self._keyboard_cmd.zero_()
    elif key == KEY_I:
      self._keyboard_cmd[0] = max_x * self._keyboard_scale
    elif key == KEY_K:
      self._keyboard_cmd[0] = -max_x * self._keyboard_scale
    elif key == KEY_J:
      self._keyboard_cmd[1] = max_y * self._keyboard_scale
    elif key == KEY_L:
      self._keyboard_cmd[1] = -max_y * self._keyboard_scale
    elif key == KEY_U:
      self._keyboard_cmd[2] = max_z * self._keyboard_scale
    elif key == KEY_O:
      self._keyboard_cmd[2] = -max_z * self._keyboard_scale
    elif key == KEY_LEFT_BRACKET:
      self._keyboard_scale = max(0.1, round(self._keyboard_scale - 0.1, 1))
      print(f"[twist teleop] scale={self._keyboard_scale:.1f}")
      self._keyboard_cmd.zero_()
    elif key == KEY_RIGHT_BRACKET:
      self._keyboard_scale = min(1.0, round(self._keyboard_scale + 0.1, 1))
      print(f"[twist teleop] scale={self._keyboard_scale:.1f}")
      self._keyboard_cmd.zero_()
    else:
      return False

    vx, vy, wz = (float(x) for x in self._keyboard_cmd.tolist())
    print(f"[twist teleop] cmd=({vx:+.2f}, {vy:+.2f}, {wz:+.2f})")
    return True

  def _apply_manual_command(self, cmd: torch.Tensor, env_ids: torch.Tensor) -> None:
    self.vel_command_b[env_ids] = cmd
    # Prevent standing / per-axis zero masks from fighting teleop next frame.
    self.is_standing_env[env_ids] = False
    self.is_zero_vel_x_env[env_ids] = False
    self.is_zero_vel_y_env[env_ids] = False
    self.is_zero_vel_yaw_env[env_ids] = False
    # Keep teleop sticky until the next resample window would fire.
    self.time_left[env_ids] = max(self.cfg.resampling_time_range)

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    super().compute(dt, env_ids)
    # Viser joystick takes precedence when enabled.
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      cmd = torch.tensor(
        [s.value for s in self._joystick_sliders],
        device=self.device,
        dtype=self.vel_command_b.dtype,
      )
      apply_all = self._joystick_all_envs is not None and self._joystick_all_envs.value
      if apply_all:
        env_ids = torch.arange(self.num_envs, device=self.device)
      else:
        env_ids = torch.tensor(
          [self._joystick_get_env_idx()], device=self.device, dtype=torch.long
        )
      self._apply_manual_command(cmd, env_ids)
      return

    if self._keyboard_enabled:
      if self._keyboard_apply_all:
        env_ids = torch.arange(self.num_envs, device=self.device)
      else:
        env_ids = torch.tensor([0], device=self.device, dtype=torch.long)
      self._apply_manual_command(self._keyboard_cmd, env_ids)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw commanded (blue) vs actual (cyan) body-frame linear velocity."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_mat_ws = matrix_from_quat(self.robot.data.root_link_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]

      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      origin = local_to_world(np.array([0.0, 0.0, z_offset]) * scale)
      cmd_to = local_to_world(
        (np.array([0.0, 0.0, z_offset]) + np.array([cmd[0], cmd[1], 0.0])) * scale
      )
      act_to = local_to_world(
        (np.array([0.0, 0.0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0.0]))
        * scale
      )
      visualizer.add_arrow(origin, cmd_to, color=(0.2, 0.2, 0.8, 0.7), width=0.015)
      visualizer.add_arrow(origin, act_to, color=(0.0, 0.7, 1.0, 0.7), width=0.015)


@dataclass(kw_only=True)
class UniformVelocityWithZeroCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  resampling_time_range: tuple[float, float] = (5.0, 10.0)
  rel_standing_envs: float = 0.1
  debug_vis: bool = False

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    zero_prob: tuple[float, float, float] = (0.2, 0.2, 0.2)

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityWithZeroCommand:
    return UniformVelocityWithZeroCommand(self, env)
