from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_g18_s4_validation_script_has_cli_help():
    proc = subprocess.run(
        [sys.executable, "scripts/g18_s4_chi_validation.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0
    assert "--reference-chi" in proc.stdout
    assert "S-4" in proc.stdout
