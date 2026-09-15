"""Check AMP playlists without launching a graphical viewer."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import tyro


@pytest.fixture(scope="module")
def replay_module():
  path = Path(__file__).parents[1] / "scripts/tools/replay_amp_motion_json.py"
  spec = importlib.util.spec_from_file_location("amp_motion_replay_test", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  yield module
  sys.modules.pop(spec.name)


@pytest.fixture
def motion_files(tmp_path: Path):
  xml = tmp_path / "robot.xml"
  xml.write_text("""<mujoco>
  <worldbody>
    <body name="base">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
      <body name="left">
        <joint name="left_joint" axis="0 1 0"/>
        <geom type="sphere" size="0.1"/>
      </body>
      <body name="right">
        <joint name="right_joint" axis="0 1 0"/>
        <geom type="sphere" size="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""")
  # Root x marks playback order; second file reverses the joint name order.
  first = tmp_path / "a_first.json"
  second = tmp_path / "b_second.json"
  for path, roots, names, positions, dt in (
    (first, [10, 20], ["left_joint", "right_joint"], [0.1, 0.2], 0.02),
    (second, [30, 40], ["right_joint", "left_joint"], [0.4, 0.3], 0.04),
  ):
    frames = [[x, 0, 1, 0, 0, 0, 1] + positions + [0] * 8 + [1, 1] for x in roots]
    path.write_text(
      json.dumps(
        {
          "Frames": frames,
          "JointNames": names,
          "FrameDuration": dt,
        }
      )
    )
  return xml, first, second


@pytest.mark.parametrize(
  "loop,limit,start,expected",
  [
    (False, 10, 0, [10, 20, 30, 40]),
    (True, 8, 0, [10, 20, 30, 40, 10, 20, 30, 40]),
    (False, 10, 1, [20, 40]),
    (True, 1, 0, [10]),  # Closing the window stops the whole playlist.
  ],
)
def test_playlist_uses_one_window(
  replay_module, motion_files, monkeypatch, loop, limit, start, expected
):
  xml, first, second = motion_files
  poses = []
  sleeps = []
  launch = MagicMock()
  viewer = launch.return_value.__enter__.return_value
  viewer.is_running.side_effect = [True] * limit + [False]

  def on_launch(model, data):
    viewer.sync.side_effect = lambda: poses.append(data.qpos.copy())
    return launch.return_value

  launch.side_effect = on_launch
  monkeypatch.setattr(replay_module.mujoco.viewer, "launch_passive", launch)
  monkeypatch.setattr(replay_module.time, "sleep", sleeps.append)
  monkeypatch.setattr(replay_module.time, "perf_counter", lambda: 0.0)
  replay_module.replay(
    replay_module.ReplayConfig(
      motion=[str(first), str(second)],
      xml=str(xml),
      zero_xy=False,
      loop=loop,
      start_frame=start,
      speed=2.0,
    )
  )

  launch.assert_called_once()
  launch.return_value.__exit__.assert_called_once()
  np.testing.assert_allclose(np.array(poses)[:, 0], expected)
  for pose, root_x, delay in zip(poses, expected, sleeps, strict=True):
    np.testing.assert_allclose(pose[7:], [0.1, 0.2] if root_x < 30 else [0.3, 0.4])
    assert delay == (0.01 if root_x < 30 else 0.02)


def test_motion_selection_and_cli(replay_module, motion_files):
  _, first, second = motion_files
  cfg = tyro.cli(replay_module.ReplayConfig, args=["--motion", str(first)])
  assert replay_module._resolve_motion_paths(cfg.motion) == [first]
  cfg = tyro.cli(
    replay_module.ReplayConfig,
    args=[
      "--motion",
      str(first.parent),
      "--no-loop",
      "--motion",
      str(second),
      str(first),
      "--loop",
    ],
  )
  assert cfg.motion == [str(second), str(first)]
  assert cfg.loop is True
  assert replay_module._resolve_motion_paths(cfg.motion) == [second, first]
  assert replay_module._resolve_motion_paths([str(first.parent)]) == [first, second]


def test_empty_motion_inputs_fail_before_viewer(replay_module, tmp_path: Path):
  with pytest.raises(ValueError, match="No motion JSON"):
    replay_module._resolve_motion_paths([str(tmp_path)])
  motion = tmp_path / "empty.json"
  motion.write_text(json.dumps({"Frames": []}))
  with pytest.raises(ValueError, match="Invalid Frames shape"):
    replay_module._load_motion(motion)
