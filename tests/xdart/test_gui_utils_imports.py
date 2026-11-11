"""``xdart.gui.gui_utils`` is imported on every GUI start (ScientificView's styled axes); it must not pull pandas
in for a table model nothing uses (1.2.10 review A10: about 0.26 s of a warm start)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def test_importing_gui_utils_loads_no_pandas():
    env = {**os.environ, "PYTHONPATH": str(SRC), "QT_QPA_PLATFORM": "offscreen"}
    probe = "import sys, xdart.gui.gui_utils as g; print(g.__file__); print('pandas' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    path, loaded = result.stdout.split()
    assert Path(path).resolve().is_relative_to(SRC.resolve()), path
    assert loaded == "False"
