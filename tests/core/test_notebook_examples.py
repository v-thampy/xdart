"""Public notebook generation and deterministic smoke-execution gate."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_public_notebooks_are_clean_and_smoke_executable():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "examples/notebooks/_smoke_notebooks.py", "--timeout", "120"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=720,
    )
    assert result.returncode == 0, result.stdout + result.stderr
