"""Import-purity sentinels for the E4-S shared leaves."""

from __future__ import annotations

import subprocess
import sys


def _probe(module: str, forbidden: tuple[str, ...]):
    script = (
        "import importlib,sys;"
        f"importlib.import_module({module!r});"
        f"roots={forbidden!r};"
        "bad=sorted(n for n in sys.modules if any("
        "n==root or n.startswith(root+'.') for root in roots));"
        "assert not bad,bad"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )


def test_hydration_leaf_is_headless_and_dependency_light():
    result = _probe(
        "xrd_tools.session.hydration",
        ("xdart", "PySide6", "PyQt5", "PyQt6", "pyqtgraph", "h5py", "numpy"),
    )
    assert result.returncode == 0, result.stderr


def test_frame_preview_leaf_is_qt_free_and_does_not_import_xdart():
    result = _probe(
        "xrd_tools.io.frame_preview",
        ("xdart", "PySide6", "PyQt5", "PyQt6", "pyqtgraph"),
    )
    assert result.returncode == 0, result.stderr
