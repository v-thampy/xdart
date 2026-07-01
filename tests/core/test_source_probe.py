from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


class _FrameSource:
    def __init__(self, frame_indices, frames=None, *, fail_indices=False, fail_load=False):
        self._frame_indices = frame_indices
        self._frames = dict(frames or {})
        self._fail_indices = fail_indices
        self._fail_load = fail_load

    @property
    def frame_indices(self):
        if self._fail_indices:
            raise RuntimeError("indices unavailable")
        return self._frame_indices

    def load_frame(self, index):
        if self._fail_load:
            raise RuntimeError("frame unavailable")
        return self._frames[index]


def test_probe_first_frame_returns_first_2d_image():
    from xrd_tools.sources.probe import probe_first_frame, raw_is_reachable

    image = np.arange(6).reshape(2, 3)
    source = _FrameSource([5], {5: image})

    reachable, first_image = probe_first_frame(source)

    assert reachable is True
    assert np.array_equal(first_image, image)
    assert raw_is_reachable(source) is True


@pytest.mark.parametrize(
    "source",
    [
        None,
        _FrameSource([]),
        _FrameSource([0], fail_indices=True),
        _FrameSource([0], fail_load=True),
        _FrameSource([0], {0: np.arange(3)}),
        _FrameSource([0], {0: np.empty((0, 3))}),
    ],
)
def test_probe_first_frame_reports_unreachable(source):
    from xrd_tools.sources.probe import probe_first_frame, raw_is_reachable

    reachable, first_image = probe_first_frame(source)

    assert reachable is False
    assert first_image is None
    assert raw_is_reachable(source) is False


def test_probe_import_does_not_load_gui_or_heavy_readers():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(root / "src")
        if not env.get("PYTHONPATH")
        else str(root / "src") + os.pathsep + env["PYTHONPATH"]
    )
    code = """
import importlib
import sys

before = set(sys.modules)
importlib.import_module("xrd_tools.sources.probe")
loaded = set(sys.modules) - before
forbidden_roots = (
    "PySide",
    "PyQt",
    "qtpy",
    "pyqtgraph",
    "pyFAI",
    "h5py",
    "fabio",
    "xdart",
)
forbidden = sorted(
    name
    for name in loaded
    if any(name == root or name.startswith(root + ".") for root in forbidden_roots)
)
if forbidden:
    print("\\n".join(forbidden))
    raise SystemExit(1)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
