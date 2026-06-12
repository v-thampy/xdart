"""Tests for ssrl_xrd_tools.integrate.batch."""

from __future__ import annotations

import time
from pathlib import Path

import fabio
import h5py
import numpy as np
import pytest
from pyFAI.detectors import Detector
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from ssrl_xrd_tools.integrate.batch import DirectoryWatcher, process_scan, process_series


def _make_small_ai(poni_fixture) -> AzimuthalIntegrator:
    """Create a generic 100x100 AI compatible with small synthetic images."""
    det = Detector(pixel1=75e-6, pixel2=75e-6, max_shape=(100, 100))
    return AzimuthalIntegrator(
        dist=float(poni_fixture.dist),
        poni1=0.00375,
        poni2=0.00375,
        wavelength=float(poni_fixture.wavelength),
        detector=det,
    )


def _write_edf_scan(scan_dir: Path, n_frames: int = 3, seed: int = 0) -> list[Path]:
    """Write a directory of small EDF frames and return their paths."""
    scan_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    for i in range(n_frames):
        img = rng.poisson(100, (100, 100)).astype(np.float32)
        out = scan_dir / f"frame_{i:04d}.edf"
        edf = fabio.edfimage.EdfImage(data=img)
        edf.write(str(out))
        paths.append(out)
    return paths


def test_process_scan_directory(ai_fixture, poni_fixture, tmp_path):
    """Process a directory with 3 EDF files and verify HDF5 output layout."""
    # ai_fixture is intentionally requested/used to ensure conftest fixture wiring.
    assert ai_fixture is not None

    scan_dir = tmp_path / "images"
    _write_edf_scan(scan_dir, n_frames=3, seed=1)

    ai = _make_small_ai(poni_fixture)
    out_h5 = tmp_path / "out.h5"
    out_path = process_scan(
        scan_dir,
        ai,
        out_h5,
        npt=100,
        npt_rad=50,
        npt_azim=50,
        unit="q_A^-1",
        method="csr",
        correctSolidAngle=False,
    )

    assert out_path == out_h5.resolve()
    assert out_h5.exists()

    with h5py.File(out_h5, "r") as h5:
        assert {"0", "1", "2"}.issubset(set(h5.keys()))
        for grp_name in ("0", "1", "2"):
            grp = h5[grp_name]
            assert {"q", "I", "IQChi", "Q", "Chi"}.issubset(set(grp.keys()))
            assert grp["q"].shape == (100,)
            assert grp["I"].shape == (100,)
            assert grp["IQChi"].shape == (50, 50)


def test_process_scan_skip_existing(ai_fixture, poni_fixture, tmp_path):
    """Second pass with reprocess=False should skip existing frames."""
    assert ai_fixture is not None

    scan_dir = tmp_path / "images"
    _write_edf_scan(scan_dir, n_frames=3, seed=2)

    ai = _make_small_ai(poni_fixture)
    out_h5 = tmp_path / "out_skip.h5"

    process_scan(
        scan_dir,
        ai,
        out_h5,
        npt=100,
        npt_rad=50,
        npt_azim=50,
        correctSolidAngle=False,
    )

    # Run again with default reprocess=False (should skip, but remain valid).
    process_scan(
        scan_dir,
        ai,
        out_h5,
        npt=100,
        npt_rad=50,
        npt_azim=50,
        reprocess=False,
        correctSolidAngle=False,
    )

    with h5py.File(out_h5, "r") as h5:
        assert set(h5.keys()) == {"0", "1", "2"}

    # Reprocess should also complete cleanly.
    process_scan(
        scan_dir,
        ai,
        out_h5,
        npt=100,
        npt_rad=50,
        npt_azim=50,
        reprocess=True,
        correctSolidAngle=False,
    )

    with h5py.File(out_h5, "r") as h5:
        assert set(h5.keys()) == {"0", "1", "2"}


def test_process_series(poni_fixture, tmp_path):
    """Process two scan directories and verify two outputs are produced."""
    scan1 = tmp_path / "scan1"
    scan2 = tmp_path / "scan2"
    _write_edf_scan(scan1, n_frames=2, seed=3)
    _write_edf_scan(scan2, n_frames=2, seed=4)

    ai = _make_small_ai(poni_fixture)
    output_dir = tmp_path / "processed"
    results = process_series(
        [scan1, scan2],
        ai,
        output_dir,
        npt=120,
        npt_rad=40,
        npt_azim=30,
        correctSolidAngle=False,
    )

    assert len(results) == 2
    for p in results:
        assert p.exists()
        with h5py.File(p, "r") as h5:
            assert len(h5.keys()) == 2


def test_directory_watcher_lifecycle(poni_fixture, tmp_path):
    """Watcher starts in background and stops cleanly."""
    watch_dir = tmp_path / "watch"
    output_dir = tmp_path / "watch_out"
    watch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ai = _make_small_ai(poni_fixture)
    watcher = DirectoryWatcher(
        watch_dir=watch_dir,
        ai=ai,
        output_dir=output_dir,
        patterns=("*.edf",),
        recursive=False,
        poll_interval=0.5,
        npt=100,
        npt_rad=50,
        npt_azim=50,
        correctSolidAngle=False,
    )

    thread = watcher.start_background()
    assert isinstance(watcher.processed_files, set)
    assert thread.is_alive()

    watcher.stop()
    thread.join(timeout=5.0)
    assert not thread.is_alive()


@pytest.mark.slow
def test_directory_watcher_detects_new_file(poni_fixture, tmp_path):
    """Watcher should detect and mark a newly created EDF file."""
    watch_dir = tmp_path / "watch_detect"
    output_dir = tmp_path / "watch_detect_out"
    watch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ai = _make_small_ai(poni_fixture)
    watcher = DirectoryWatcher(
        watch_dir=watch_dir,
        ai=ai,
        output_dir=output_dir,
        patterns=("*.edf",),
        recursive=False,
        poll_interval=0.5,
        npt=100,
        npt_rad=50,
        npt_azim=50,
        correctSolidAngle=False,
    )
    thread = watcher.start_background()

    try:
        # Give watcher one poll interval to start.
        time.sleep(0.2)

        new_file = watch_dir / "new_frame_0000.edf"
        img = np.random.default_rng(5).poisson(100, (100, 100)).astype(np.float32)
        fabio.edfimage.EdfImage(data=img).write(str(new_file))

        deadline = time.time() + 5.0
        seen = False
        while time.time() < deadline:
            if new_file in watcher.processed_files:
                seen = True
                break
            time.sleep(0.1)

        assert seen, "Watcher did not detect/process new EDF file within timeout"
    finally:
        watcher.stop()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
