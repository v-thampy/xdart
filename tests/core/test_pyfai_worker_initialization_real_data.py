from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


_ROOT = Path(__file__).resolve().parents[2]


def _real_data_root() -> Path:
    configured = os.environ.get("XDART_TEST_DATA")
    if not configured:
        pytest.skip("XDART_TEST_DATA is not configured")
    root = Path(configured)
    if not root.is_dir():
        pytest.skip("XDART_TEST_DATA is not a directory")
    return root


def test_four_workers_initialize_pyfai_with_one_numexpr_thread(
    tmp_path: Path,
) -> None:
    """Cold worker setup must not enter numexpr concurrently and abort."""
    data_root = _real_data_root()
    eiger_root = data_root / "eiger"
    master = eiger_root / "Eiger_NbN_1_thin_test__200mdeg_scan001_master.h5"
    payload = eiger_root / "Eiger_NbN_1_thin_test__200mdeg_scan001_data_000001.h5"
    poni = eiger_root / "LaB6_detxn26_detyn6p5_eta4p5.poni"
    missing = [path for path in (master, payload, poni) if not path.is_file()]
    if missing:
        pytest.skip(f"XDART_TEST_DATA lacks {missing[0].relative_to(data_root)}")

    source = tmp_path / "source"
    source.mkdir()
    for path in (master, payload):
        (source / path.name).symlink_to(path)

    environment = dict(os.environ)
    environment.update(
        {
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(_ROOT / "src"),
            "XDART_SESSION_FILE": str(tmp_path / "session.json"),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-X",
            "faulthandler",
            str(_ROOT / "scripts" / "nxs_directory_benchmark.py"),
            "--source-dir",
            str(source),
            "--ext",
            "h5",
            "--poni",
            str(poni),
            "--output-dir",
            str(tmp_path / "output"),
            "--mode",
            "2d",
            "--cores",
            "4",
            "--frame-limit",
            "4",
            "--repeat",
            "1",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    diagnostic = completed.stdout + completed.stderr
    assert completed.returncode == 0, diagnostic
    assert "red=4 wr=4 durable=4" in diagnostic
