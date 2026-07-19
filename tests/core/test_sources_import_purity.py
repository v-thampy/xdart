# -*- coding: utf-8 -*-
"""R1 — GUI-import purity of the three new modules, checked DIRECTLY.

xrd_tools.sources's package-level `import xrd_tools.sources` stays light via
lazy `__getattr__` (see test_readiness_purity.py /
test_source_probe.py::test_probe_import_does_not_load_gui_or_heavy_readers),
but that lazy seam means importing the PACKAGE never actually loads
adapters.py/directory_index.py/discover.py at all -- so it cannot catch a Qt/
GUI import newly introduced inside one of THOSE modules specifically.  This
imports each of the three directly, in a subprocess (so an already-loaded
h5py/fabio/Qt from an earlier test in the same process can never mask a
regression), and asserts none of the forbidden GUI/heavy roots load.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

_FORBIDDEN = (
    "xdart",
    "PySide6",
    "PySide2",
    "PyQt5",
    "PyQt6",
    "pyqtgraph",
    "matplotlib",
)


def _assert_module_imports_without_gui(module_name: str) -> None:
    code = textwrap.dedent(
        f"""
        import sys

        import {module_name}  # noqa: F401

        bad = sorted(
            root
            for root in {_FORBIDDEN!r}
            if root in sys.modules
            or any(name == root or name.startswith(root + ".") for name in sys.modules)
        )
        if bad:
            print(",".join(bad))
            sys.exit(1)
        """
    )
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + existing if existing else "")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, (
        f"{module_name} import pulled in a forbidden GUI module: "
        f"{proc.stdout.strip()}\n{proc.stderr.strip()}"
    )


def test_adapters_module_imports_without_gui_stack() -> None:
    _assert_module_imports_without_gui("xrd_tools.sources.adapters")


def test_directory_index_module_imports_without_gui_stack() -> None:
    _assert_module_imports_without_gui("xrd_tools.sources.directory_index")


def test_discover_module_imports_without_gui_stack() -> None:
    _assert_module_imports_without_gui("xrd_tools.sources.discover")


def test_run_plan_module_imports_without_gui_stack() -> None:
    # H19 §4: the Source-card -> Run handoff seam stays Qt-free.
    _assert_module_imports_without_gui("xrd_tools.sources.run_plan")
