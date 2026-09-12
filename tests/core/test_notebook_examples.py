"""Public notebook generation and deterministic smoke-execution gate."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

# The sequential nine-notebook smoke is budgeted at 720 s below; the marker
# makes that budget the one pytest-timeout applies too, instead of whichever
# job-wide ``--timeout`` the workflow passes (pr.yml core caps at 300 s, which
# a slow runner has exceeded while the smoke itself was still healthy).
_SMOKE_BUDGET_S = 720


@pytest.mark.timeout(_SMOKE_BUDGET_S)
def test_public_notebooks_are_clean_and_smoke_executable():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "examples/notebooks/_smoke_notebooks.py", "--timeout", "120"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=_SMOKE_BUDGET_S,
    )
    assert result.returncode == 0, result.stdout + result.stderr
