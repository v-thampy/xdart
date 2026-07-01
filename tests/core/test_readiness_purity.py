from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def test_readiness_and_masks_import_without_gui_or_heavy_io_stack() -> None:
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

        import xrd_tools.session.readiness  # noqa: F401
        import xrd_tools.reduction.masks  # noqa: F401

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
        "readiness/masks import pulled in forbidden modules: "
        f"{proc.stdout.strip()}\n{proc.stderr.strip()}"
    )

