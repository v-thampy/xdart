from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _assert_imports_without_forbidden_modules(import_lines: str, label: str) -> None:
    forbidden = (
        "xdart",
        "PySide6",
        "PySide2",
        "PyQt5",
        "PyQt6",
        "pyqtgraph",
        "pyFAI",
        "h5py",
        "fabio",
    )
    code = textwrap.dedent(
        f"""
        import sys

        {import_lines}

        bad = sorted(
            root
            for root in {forbidden!r}
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
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, (
        f"{label} import pulled in forbidden modules: "
        f"{proc.stdout.strip()}\n{proc.stderr.strip()}"
    )


def test_readiness_and_masks_import_without_gui_or_heavy_io_stack() -> None:
    _assert_imports_without_forbidden_modules(
        """
        import xrd_tools.session.readiness  # noqa: F401
        import xrd_tools.reduction.masks  # noqa: F401
        """,
        "readiness/masks",
    )


def test_xrd_tools_package_imports_without_gui_stack() -> None:
    _assert_imports_without_forbidden_modules(
        "import xrd_tools  # noqa: F401",
        "xrd_tools",
    )


def test_display_logic_imports_without_gui_or_heavy_io_stack() -> None:
    _assert_imports_without_forbidden_modules(
        "import xrd_tools.session.display_logic  # noqa: F401",
        "display_logic",
    )
