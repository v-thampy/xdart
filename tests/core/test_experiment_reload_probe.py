from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/core/fixtures"
MANIFEST = FIXTURES / "experiment_reload_probe_manifest.json"
RESULTS = FIXTURES / "experiment_reload_probe_results.json"


def test_historical_probe_manifest_and_rows_are_complete_and_portable() -> None:
    manifest = json.loads(MANIFEST.read_text())
    results = json.loads(RESULTS.read_text())
    assert manifest["schema_version"] == results["schema_version"] == 1
    assert manifest["selection"]["count"] == len(manifest["records"]) == 600
    assert len(results["rows"]) == 600
    expected_paths = [row["relative_path"] for row in manifest["records"]]
    assert [row["relative_path"] for row in results["rows"]] == expected_paths
    assert len(set(expected_paths)) == 600
    for path in expected_paths:
        assert not Path(path).is_absolute() and ".." not in Path(path).parts
        assert Path(path).parts[0] in {"test_data", "tmp"}
    for row in results["rows"]:
        assert row["readable"] is not row["open_failure"]
        assert len(row["content_sha256"]) == 64
    counted = Counter(row["classification"] for row in results["rows"])
    assert results["aggregate"] == dict(sorted(counted.items()))
    assert results["aggregate"] == {
        "absent": 326,
        "headless_only": 260,
        "legacy_only": 2,
        "open_failure": 12,
    }
    assert sum(results["aggregate"].values()) == 600
    assert results["exit_status"] == 0 and results["failures"] == []


@pytest.mark.skipif(
    os.environ.get("XDART_RERUN_HISTORICAL_PROBE") != "1",
    reason="set XDART_RERUN_HISTORICAL_PROBE=1 for the 600-record census",
)
def test_committed_historical_probe_rows_reproduce() -> None:
    data_root = Path(os.environ.get("XDART_PROBE_ROOT", ROOT.parents[1]))
    command = [sys.executable, str(ROOT / "scripts/probe_experiment_provenance.py"),
               "--data-root", str(data_root), "--manifest", str(MANIFEST)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(command, env=env, text=True, capture_output=True,
                               check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == json.loads(RESULTS.read_text())
