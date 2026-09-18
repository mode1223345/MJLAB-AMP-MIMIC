"""Overlay managers for Viser viewer orchestration.

These managers intentionally coordinate *when* higher-level updates happen
(env switches, paused/running updates, etc.) while leaving low-level render
handle lifecycle ownership inside :mod:`scene.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import mujoco
import numpy as np
import viser

from mjlab.sensor import CameraSensor
from mjlab.viewer.viser.camera_viewer import ViserCameraViewer
from mjlab.viewer.viser.force_panel import ForcePanel
from mjlab.viewer.viser.joint_panel import (
  JointStatePanel,
  action_series_name,
  pos_series_name,
  vel_series_name,
)
from mjlab.viewer.viser.reward_bar_panel import RewardBarPanel
from mjlab.viewer.viser.term_plotter import ViserTermPlotter
from mjlab.viewer.viser.termination_panel import TerminationPanel


class _EnvProtocol(Protocol):
  @property
  def unwrapped(self) -> Any: ...


class _SceneProtocol(Protocol):
  env_idx: int
  debug_visualization_enabled: bool
  needs_update: bool

  @property
  def show_contact_points(self) -> bool: ...
  @property
  def show_contact_forces(self) -> bool: ...

  def clear_debug_all(self) -> None: ...
  def clear(self) -> None: ...


@dataclass
class ViserTermOverlays:
  """Manage reward/metrics term plot tabs for Viser viewer."""

  server: viser.ViserServer
  env: _EnvProtocol
  scene: _SceneProtocol
  frame_time: float
  reward_bar_max_terms: int = 20
  reward_plotter: ViserTermPlotter | None = None
  reward_bar_panel: RewardBarPanel | None = None
  metrics_plotter: ViserTermPlotter | None = None

  @staticmethod
  def _amp_reward_term_names(env: _EnvProtocol) -> list[str]:
    probe = getattr(env.unwrapped, "amp_reward_probe", None)
    if probe is None or not hasattr(probe, "get_amp_reward_panel_terms"):
      return []
    return ["style_reward", "task_reward"]

  @staticmethod
  def _amp_reward_terms(
    env: _EnvProtocol, env_idx: int
  ) -> list[tuple[str, np.ndarray]]:
    probe = getattr(env.unwrapped, "amp_reward_probe", None)
    getter = getattr(probe, "get_amp_reward_panel_terms", None) if probe else None
    if getter is None:
      return []
    try:
      raw = getter(env_idx)
    except Exception:
      return []
    out: list[tuple[str, np.ndarray]] = []
    for name, values in raw:
      arr = np.asarray(values, dtype=np.float64).reshape(-1)
      if arr.size == 0 or not np.isfinite(arr[0]):
        arr = np.asarray([0.0], dtype=np.float64)
      out.append((name, arr))
    return out

  def setup_tabs(self, tabs: Any) -> None:
    """Create rewards/metrics tabs based on available managers."""
    if hasattr(self.env.unwrapped, "reward_manager"):
      with tabs.add_tab("Rewards", icon=viser.Icon.CHART_LINE):
        base_names = [
          name
          for name, _ in self.env.unwrapped.reward_manager.get_active_iterable_terms(
            self.scene.env_idx
          )
        ]
        term_names = self._amp_reward_term_names(self.env) + base_names
        # ``reward_bar_max_terms`` is a floor, not a cap: every active term is
        # shown, and the panel only truncates if the config asks for fewer.
        max_terms = max(self.reward_bar_max_terms, len(term_names))
        # Live bar panel (running-mean comparison).
        self.reward_bar_panel = RewardBarPanel(
          self.server,
          term_names,
          update_dt=self.frame_time,
          max_terms=max_terms,
        )
        self.reward_plotter = ViserTermPlotter(
          self.server, term_names, name="Reward", env_idx=self.scene.env_idx
        )

    if hasattr(self.env.unwrapped, "metrics_manager"):
      term_names = [
        name
        for name, _ in self.env.unwrapped.metrics_manager.get_active_iterable_terms(
          self.scene.env_idx
        )
      ]
      if term_names:
        with tabs.add_tab("Metrics", icon=viser.Icon.CHART_BAR):
          self.metrics_plotter = ViserTermPlotter(
            self.server, term_names, name="Metric", env_idx=self.scene.env_idx
          )

  def on_env_switch(self) -> None:
    """Clear histories when active environment changes."""
    env_idx = self.scene.env_idx
    if self.reward_plotter:
      self.reward_plotter.clear_histories()
      self.reward_plotter.update_env_idx(env_idx)
    if self.reward_bar_panel:
      self.reward_bar_panel.clear_histories()
    if self.metrics_plotter:
      self.metrics_plotter.clear_histories()
      self.metrics_plotter.update_env_idx(env_idx)

  def update(self, paused: bool) -> None:
    """Update term plots from the selected environment."""
    if (
      self.reward_plotter is not None or self.reward_bar_panel is not None
    ) and not paused:
      terms = self._amp_reward_terms(self.env, self.scene.env_idx) + list(
        self.env.unwrapped.reward_manager.get_active_iterable_terms(self.scene.env_idx)
      )
      if self.reward_plotter is not None:
        self.reward_plotter.update(terms)
      if self.reward_bar_panel is not None:
        self.reward_bar_panel.update(terms)

    if self.metrics_plotter is not None and not paused:
      terms = list(
        self.env.unwrapped.metrics_manager.get_active_iterable_terms(self.scene.env_idx)
      )
      self.metrics_plotter.update(terms)

  def clear_histories(self) -> None:
    """Clear all overlay histories."""
    self.on_env_switch()

  def cleanup(self) -> None:
    """Cleanup plotter resources."""
    if self.reward_plotter:
      self.reward_plotter.cleanup()
    if self.reward_bar_panel:
      self.reward_bar_panel.cleanup()
    if self.metrics_plotter:
      self.metrics_plotter.cleanup()


@dataclass
class ViserCameraOverlays:
  """Manage camera feed widgets and updates for Viser viewer."""

  server: viser.ViserServer
  env: _EnvProtocol
  mj_model: mujoco.MjModel
  camera_viewers: list[ViserCameraViewer] | None = None

  @property
  def has_cameras(self) -> bool:
    """Whether the environment has any camera sensors."""
    return any(
      isinstance(s, CameraSensor) for s in self.env.unwrapped.scene.sensors.values()
    )

  def setup_controls(self) -> None:
    """Create camera feed controls under the active GUI folder."""
    camera_sensors = [
      sensor
      for sensor in self.env.unwrapped.scene.sensors.values()
      if isinstance(sensor, CameraSensor)
    ]
    if not camera_sensors:
      self.camera_viewers = []
      return

    self.camera_viewers = [
      ViserCameraViewer(self.server, sensor, self.mj_model) for sensor in camera_sensors
    ]

  def update(self, sim_data: Any, env_idx: int, scene_offset: Any) -> None:
    """Push latest camera images/frustums to GUI."""
    if not self.camera_viewers:
      return
    for camera_viewer in self.camera_viewers:
      camera_viewer.update(sim_data, env_idx, scene_offset)

  def cleanup(self) -> None:
    """Cleanup all camera feed widgets."""
    if not self.camera_viewers:
      return
    for camera_viewer in self.camera_viewers:
      camera_viewer.cleanup()


@dataclass
class ViserDebugOverlays:
  """Manage debug visualization queueing and env-switch behavior."""

  env: _EnvProtocol
  scene: _SceneProtocol

  def on_env_switch(self) -> None:
    """Reset debug visuals when switching selected environment."""
    if self.scene.debug_visualization_enabled:
      self.scene.clear_debug_all()

  def queue(self) -> None:
    """Queue environment debug visualizers for the current frame."""
    if self.scene.debug_visualization_enabled and hasattr(
      self.env.unwrapped, "update_visualizers"
    ):
      self.scene.clear()  # Clear queued arrows from previous frame.
      self.env.unwrapped.update_visualizers(self.scene)


@dataclass
class ViserContactOverlays:
  """Manage contact-visualization orchestration from the viewer layer.

  Note: contact mesh creation/update/removal stays in ``ViserMujocoScene``.
  This manager only requests scene refreshes at the right times.
  """

  scene: _SceneProtocol

  def is_enabled(self) -> bool:
    """Whether any contact visualization is currently enabled."""
    return self.scene.show_contact_points or self.scene.show_contact_forces

  def on_env_switch(self) -> None:
    """Request a scene refresh when switching environments with contacts enabled."""
    if self.is_enabled():
      self.scene.needs_update = True


@dataclass
class ViserForceOverlays:
  """Numeric joint-torque / contact-force panel + time-series plots."""

  server: viser.ViserServer
  env: _EnvProtocol
  scene: _SceneProtocol
  force_panel: ForcePanel | None = None
  force_plotter: ViserTermPlotter | None = None

  def setup_tab(self, tabs: Any) -> None:
    """Create the Forces tab with values panel and curve plots."""
    with tabs.add_tab("Forces", icon=viser.Icon.ACTIVITY):
      self.force_panel = ForcePanel(
        self.server,
        self.env,
        get_env_idx=lambda: self.scene.env_idx,
      )
      term_names = self.force_panel.term_names
      if term_names:
        # Default: plot both foot↔terrain contact magnitudes.
        self.force_plotter = ViserTermPlotter(
          self.server,
          term_names,
          name="Force",
          env_idx=self.scene.env_idx,
          initially_enabled=list(self.force_panel.contact_term_names),
        )

  def on_env_switch(self) -> None:
    if self.force_plotter is not None:
      self.force_plotter.clear_histories()
      self.force_plotter.update_env_idx(self.scene.env_idx)

  def update(self) -> None:
    if self.force_panel is not None:
      self.force_panel.update()
      if self.force_plotter is not None and self.force_panel.last_terms:
        self.force_plotter.update(self.force_panel.last_terms)

  def cleanup(self) -> None:
    if self.force_plotter is not None:
      self.force_plotter.cleanup()
      self.force_plotter = None
    if self.force_panel is not None:
      self.force_panel.cleanup()
      self.force_panel = None


@dataclass
class ViserJointOverlays:
  """Per-joint action / position / velocity panel + time-series plots."""

  server: viser.ViserServer
  env: _EnvProtocol
  scene: _SceneProtocol
  joint_panel: JointStatePanel | None = None
  joint_plotter: ViserTermPlotter | None = None

  def setup_tab(self, tabs: Any) -> None:
    """Create the Joints tab with values table and selectable curves."""
    with tabs.add_tab("Joints", icon=viser.Icon.ADJUSTMENTS):
      self.joint_panel = JointStatePanel(
        self.server,
        self.env,
        get_env_idx=lambda: self.scene.env_idx,
      )
      term_names = self.joint_panel.term_names
      if term_names:
        # Default: first joint's action/pos/vel so the tab is immediately useful.
        first = self.joint_panel.joint_names[:1]
        initially = []
        if first:
          jn = first[0]
          if self.joint_panel.has_matching_action:
            initially.append(action_series_name(jn))
          initially.extend([pos_series_name(jn), vel_series_name(jn)])
        self.joint_plotter = ViserTermPlotter(
          self.server,
          term_names,
          name="Joint",
          env_idx=self.scene.env_idx,
          initially_enabled=initially,
        )

  def on_env_switch(self) -> None:
    if self.joint_plotter is not None:
      self.joint_plotter.clear_histories()
      self.joint_plotter.update_env_idx(self.scene.env_idx)

  def update(self) -> None:
    if self.joint_panel is not None:
      self.joint_panel.update()
      if self.joint_plotter is not None and self.joint_panel.last_terms:
        self.joint_plotter.update(self.joint_panel.last_terms)

  def cleanup(self) -> None:
    if self.joint_plotter is not None:
      self.joint_plotter.cleanup()
      self.joint_plotter = None
    if self.joint_panel is not None:
      self.joint_panel.cleanup()
      self.joint_panel = None


@dataclass
class ViserTerminationOverlays:
  """Termination status panel + 0/1 time-series plots."""

  server: viser.ViserServer
  env: _EnvProtocol
  scene: _SceneProtocol
  termination_panel: TerminationPanel | None = None
  termination_plotter: ViserTermPlotter | None = None

  def setup_tab(self, tabs: Any) -> None:
    """Create the Terminations tab when the env exposes a termination manager."""
    if not hasattr(self.env.unwrapped, "termination_manager"):
      return
    with tabs.add_tab("Terminations", icon=viser.Icon.OCTAGON):
      tm = self.env.unwrapped.termination_manager
      term_names = list(tm.active_terms)
      self.termination_panel = TerminationPanel(
        self.server,
        self.env,
        get_env_idx=lambda: self.scene.env_idx,
      )
      if term_names:
        # Timeout dominates the plot scale during normal running, so start
        # with only the failure terms visible.
        initially_enabled = [
          name for name in term_names if not tm.get_term_cfg(name).time_out
        ]
        self.termination_plotter = ViserTermPlotter(
          self.server,
          term_names,
          name="Termination",
          env_idx=self.scene.env_idx,
          initially_enabled=initially_enabled or term_names[:1],
        )

  def on_env_switch(self) -> None:
    if self.termination_plotter is not None:
      self.termination_plotter.clear_histories()
      self.termination_plotter.update_env_idx(self.scene.env_idx)

  def update(self, paused: bool) -> None:
    if self.termination_panel is None:
      return
    self.termination_panel.update()
    if self.termination_plotter is not None and not paused:
      terms = [
        (name, np.array([val], dtype=np.float64))
        for name, val in self.termination_panel.last_terms
      ]
      self.termination_plotter.update(terms)

  def cleanup(self) -> None:
    if self.termination_plotter is not None:
      self.termination_plotter.cleanup()
      self.termination_plotter = None
    if self.termination_panel is not None:
      self.termination_panel.cleanup()
      self.termination_panel = None
