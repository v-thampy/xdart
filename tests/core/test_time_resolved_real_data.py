"""Opt-in local acceptance gate for the documented Pt processed fixtures."""

from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np
import pytest

from xrd_tools.analysis import (
    LinearThermalExpansion,
    PeakFitPlan,
    add_lattice_results,
    add_temperature_results,
    bin_time_resolved,
    fit_peak_series,
    flag_fit_quality,
    load_time_resolved_series,
    normalize_reference_band,
)


def _data_root() -> Path:
    value = os.environ.get("XDART_TEST_DATA")
    if not value:
        pytest.skip("set XDART_TEST_DATA to run local Pt real-data acceptance")
    root = Path(value).expanduser()
    if not root.is_dir():
        pytest.skip(f"XDART_TEST_DATA is not a directory: {root}")
    return root


def _pilot_plan(dataset) -> tuple[PeakFitPlan, tuple[float, float]]:
    q = np.asarray(dataset.coords["q"].values, dtype=float)
    mean = np.nanmean(np.asarray(dataset["intensity"].values, dtype=float), axis=0)
    index = int(np.nanargmax(mean[20:-20])) + 20
    center = float(q[index])
    return (
        PeakFitPlan(
            positions=(center,),
            model="gaussian",
            background="linear",
            sigma_init=0.02,
            sigma_bounds=(0.003, 0.10),
            center_bounds_delta=0.08,
        ),
        (center - 0.10, center + 0.10),
    )


@pytest.mark.slow
def test_pt_processed_and_burst_real_data_gate():
    root = _data_root()
    processed = root / "xdart_processed_data"
    pt31 = processed / "Pt_10nm_00013.nxs"
    burst = processed / "Pt_test_burst_00007.nxs"
    neighbors = [processed / f"Pt_test_burst_{index:05d}.nxs" for index in (6, 7, 8)]
    assert pt31.is_file() and burst.is_file() and all(path.is_file() for path in neighbors)

    started = time.perf_counter()
    series31 = load_time_resolved_series(pt31, frame_period_s=0.002, source_root=root)
    load31_s = time.perf_counter() - started
    assert series31.dataset["intensity"].shape == (31, 1000)
    compact_bytes = series31.dataset["intensity"].values.nbytes
    assert series31.get_cake(0).intensity.shape == (500, 500)
    thumbnail = series31.get_thumbnail(0)
    raw = series31.get_raw(0)
    assert thumbnail.shape == (256, 245)
    assert raw.shape == (2167, 2070)
    assert series31.dataset["intensity"].values.nbytes == compact_bytes

    plan, q_range = _pilot_plan(series31.dataset)
    fits = fit_peak_series(series31.dataset, plan, q_range=q_range, pattern_indices=np.arange(8))
    fits = flag_fit_quality(fits, max_center_error=0.02)
    lattice = add_lattice_results(fits, hkls=((1, 1, 1),))
    calibration = LinearThermalExpansion(float(lattice["lattice_mean_A"].values[0]), 300.0, 9e-6)
    thermal = add_temperature_results(lattice, calibration)
    assert fits.sizes["fit_pattern"] == 8
    assert int(fits["fit_valid"].sum()) > 0
    assert np.isfinite(thermal["temperature_K"].values).any()

    started = time.perf_counter()
    burst_series = load_time_resolved_series(burst, frame_period_s=0.002, source_root=root)
    burst_load_s = time.perf_counter() - started
    assert burst_series.dataset["intensity"].shape == (1000, 1000)
    q = np.asarray(burst_series.dataset.coords["q"].values, dtype=float)
    normalized = normalize_reference_band(
        burst_series.dataset,
        q_range=(float(q[-100]), float(q[-10])),
    )
    binned = bin_time_resolved(normalized, bin_size=5)
    plan, q_range = _pilot_plan(binned)
    started = time.perf_counter()
    pilot = fit_peak_series(binned, plan, q_range=q_range, pattern_indices=np.arange(27))
    pilot = flag_fit_quality(pilot, max_center_error=0.02)
    pilot_lattice = add_lattice_results(pilot, hkls=((1, 1, 1),))
    pilot_thermal = add_temperature_results(
        pilot_lattice,
        LinearThermalExpansion(float(pilot_lattice["lattice_mean_A"].values[0]), 300.0, 9e-6),
    )
    pilot_fit_s = time.perf_counter() - started
    assert binned["intensity"].shape == (200, 1000)
    assert int(pilot["fit_valid"].sum()) > 0
    assert np.isfinite(pilot_thermal["temperature_rate_K_per_s"].values).any()

    started = time.perf_counter()
    multi = load_time_resolved_series(neighbors, frame_period_s=0.002, source_root=root)
    multi_binned = bin_time_resolved(multi.dataset, bin_size=5)
    multi_s = time.perf_counter() - started
    assert multi.dataset["intensity"].shape == (3000, 1000)
    assert multi_binned.sizes["pattern"] == 600
    assert np.array_equal(multi_binned["frame_count"].values, np.full(600, 5))
    assert multi_binned.coords["scan_name"].values[[0, 200, 400]].tolist() == [
        "Pt_test_burst_00006",
        "Pt_test_burst_00007",
        "Pt_test_burst_00008",
    ]
    assert np.all(np.diff(multi.dataset.coords["sequence_time"].values) > 0)

    print(
        "Pt real-data gate: "
        f"31-frame load={load31_s:.3f}s compact={compact_bytes}B; "
        f"burst load={burst_load_s:.3f}s pilot27={pilot_fit_s:.3f}s; "
        f"three-scan load/bin={multi_s:.3f}s"
    )
