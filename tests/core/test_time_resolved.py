from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.analysis.plans import PeakFitPlan
from xrd_tools.analysis.time_resolved import (
    LinearThermalExpansion,
    TabulatedThermalExpansion,
    add_lattice_results,
    add_temperature_results,
    bin_time_resolved,
    discover_processed_scans,
    fit_peak_series,
    flag_fit_quality,
    flag_normalization_outliers,
    load_time_resolved_series,
    normalize_reference_band,
    select_time_zero,
)
from xrd_tools.viz.plotly import (
    plot_peak_fit_frame,
    plot_thermal_history,
    plot_time_resolved_waterfall,
)


def _write_scan(
    path: Path,
    *,
    q: np.ndarray,
    intensity: np.ndarray,
    frame_start: int = 1,
    include_2d: bool = False,
) -> Path:
    frames = np.arange(frame_start, frame_start + len(intensity), dtype=np.int64)
    with h5py.File(path, "w") as h5:
        entry = h5.create_group("entry")
        g1 = entry.create_group("integrated_1d")
        g1.create_dataset("frame_index", data=frames)
        q_ds = g1.create_dataset("q", data=np.asarray(q, dtype=np.float32))
        q_ds.attrs["units"] = "q_A^-1"
        g1.create_dataset("intensity", data=np.asarray(intensity, dtype=np.float32))
        g1.create_dataset(
            "sigma", data=np.sqrt(np.maximum(intensity, 0)).astype(np.float32))

        if include_2d:
            chi = np.linspace(-20, 20, 5, dtype=np.float32)
            q2 = np.linspace(q.min(), q.max(), 6, dtype=np.float32)
            g2 = entry.create_group("integrated_2d")
            g2.create_dataset("frame_index", data=frames)
            q2_ds = g2.create_dataset("q", data=q2)
            q2_ds.attrs["units"] = "q_A^-1"
            chi_ds = g2.create_dataset("chi", data=chi)
            chi_ds.attrs["units"] = "chi_deg"
            cake = np.arange(len(frames) * len(chi) * len(q2), dtype=np.float32)
            g2.create_dataset("intensity", data=cake.reshape(len(frames), len(chi), len(q2)))

        frame_group = entry.create_group("frames")
        for frame in frames:
            fg = frame_group.create_group(f"frame_{frame:04d}")
            thumb = fg.create_dataset(
                "thumbnail", data=np.arange(12, dtype=np.uint8).reshape(3, 4))
            thumb.attrs["vmin"] = 10.0
            thumb.attrs["vmax"] = 265.0
            thumb.attrs["dtype"] = "uint8"
    return path


@pytest.fixture
def scan_pair(tmp_path):
    q = np.linspace(1.0, 5.0, 81)
    y1 = np.vstack([np.full_like(q, value) for value in (2.0, 3.0, 4.0)])
    y2 = np.vstack([np.full_like(q, value) for value in (5.0, 6.0)])
    a = _write_scan(tmp_path / "scan_2.nxs", q=q, intensity=y1, include_2d=True)
    b = _write_scan(tmp_path / "scan_10.nxs", q=q, intensity=y2)
    return a, b


def test_discover_and_load_time_resolved_series(scan_pair, tmp_path):
    paths = discover_processed_scans(tmp_path)
    assert [p.name for p in paths] == ["scan_2.nxs", "scan_10.nxs"]

    series = load_time_resolved_series(paths, frame_period_s=0.002)
    ds = series.dataset
    assert ds["intensity"].shape == (5, 81)
    assert ds.coords["scan_name"].values.tolist() == [
        "scan_2", "scan_2", "scan_2", "scan_10", "scan_10"]
    np.testing.assert_allclose(ds.coords["time"].values, [0, .002, .004, 0, .002])
    assert ds.coords["time"].attrs["units"] == "s"

    cake = series.get_cake(0)
    assert cake.intensity.shape == (5, 6)
    raw = series.get_raw(0)
    assert raw.shape == (3, 4)
    np.testing.assert_allclose(raw.ravel()[:2], [10.0, 11.0])


def test_load_q_grid_policy_interpolates_or_fails(tmp_path):
    q1 = np.linspace(1.0, 5.0, 81)
    q2 = np.linspace(1.0, 5.0, 61)
    a = _write_scan(tmp_path / "a.nxs", q=q1, intensity=np.ones((1, len(q1))))
    b = _write_scan(tmp_path / "b.nxs", q=q2, intensity=np.ones((1, len(q2))) * 2)

    with pytest.raises(ValueError, match="q grid"):
        load_time_resolved_series([a, b], q_policy="strict")

    ds = load_time_resolved_series([a, b], q_policy="interpolate").dataset
    assert ds["intensity"].shape == (2, len(q1))
    np.testing.assert_allclose(ds["intensity"].values[1], 2.0)


def test_normalize_flag_bin_and_time_zero(scan_pair):
    ds = load_time_resolved_series(scan_pair, frame_period_s=0.002).dataset
    ds = normalize_reference_band(ds, q_range=(3.6, 3.8), statistic="median")
    np.testing.assert_allclose(
        np.nanmedian(ds["intensity_normalized"].values, axis=1), 4.0)
    assert ds["normalization_factor"].values.tolist() == [2, 3, 4, 5, 6]

    ds = flag_normalization_outliers(ds, zmax=2.0)
    assert ds["pattern_valid"].dtype == np.dtype(bool)

    zeroed = select_time_zero(ds.isel(pattern=slice(0, 3)), zero_pattern=1)
    np.testing.assert_allclose(zeroed.coords["time_zeroed"], [-.002, 0, .002])

    binned = bin_time_resolved(ds, bin_size=2, reducer="mean")
    assert binned.sizes["pattern"] == 3
    assert binned.coords["scan_name"].values.tolist() == [
        "scan_2", "scan_2", "scan_10"]
    np.testing.assert_allclose(
        binned["intensity_normalized"].values[:, 0], [4.0, 4.0, 4.0])
    assert binned.coords["frame_count"].values.tolist() == [2, 1, 2]


def test_peak_series_lattice_temperature_and_rate(tmp_path):
    q = np.linspace(2.45, 3.35, 500)
    centers = np.array([2.76, 2.755, 2.75, 2.745, 2.74])
    intensity = []
    for center in centers:
        peak_111 = 100 * np.exp(-0.5 * ((q - center) / 0.012) ** 2)
        peak_200 = 60 * np.exp(-0.5 * ((q - center * np.sqrt(4 / 3)) / 0.014) ** 2)
        intensity.append(peak_111 + peak_200 + 2 + 0.2 * q)
    path = _write_scan(tmp_path / "peaks.nxs", q=q, intensity=np.asarray(intensity))
    ds = load_time_resolved_series(path, frame_period_s=0.001).dataset

    plan = PeakFitPlan(
        positions=(2.76, 2.76 * np.sqrt(4 / 3)),
        model="gaussian",
        background="linear",
        sigma_init=0.012,
        sigma_bounds=(0.003, 0.05),
        center_bounds_delta=0.08,
    )
    fits = fit_peak_series(ds, plan, q_range=(2.5, 3.3))
    assert fits.sizes["fit_pattern"] == 5
    assert fits["fit_success"].values.all()
    assert np.all(np.isfinite(fits["redchi"].values))
    np.testing.assert_allclose(fits["center_0"], centers, atol=2e-3)
    assert fits["fit"].shape == (5, fits.sizes["q_fit"])

    fits = flag_fit_quality(fits, max_center_error=0.01)
    assert fits["fit_valid"].values.all()

    lattice = add_lattice_results(fits, hkls=((1, 1, 1), (2, 0, 0)))
    assert lattice["lattice_A"].shape == (5, 2)
    assert np.all(np.diff(lattice["lattice_mean_A"].values) > 0)

    calibration = LinearThermalExpansion(
        reference_lattice_A=float(lattice["lattice_mean_A"].values[0]),
        reference_temperature_K=300.0,
        alpha_per_K=9e-6,
    )
    thermal = add_temperature_results(
        lattice, calibration, time_coord="time", smooth_window=None)
    assert thermal["temperature_K"].values[0] == pytest.approx(300.0)
    assert np.nanmax(thermal["temperature_rate_K_per_s"].values) > 0

    table = TabulatedThermalExpansion(
        temperature_K=(300.0, 600.0, 900.0),
        lattice_A=(3.92, 3.93, 3.95),
    )
    np.testing.assert_allclose(table.temperature([3.92, 3.93]), [300, 600])

    waterfall = plot_time_resolved_waterfall(ds)
    fit_figure = plot_peak_fit_frame(thermal, 0)
    thermal_figure = plot_thermal_history(thermal)
    assert len(waterfall.data) == 1
    assert len(fit_figure.data) >= 3
    assert len(thermal_figure.data) >= 4

    with pytest.raises(ValueError, match="no patterns"):
        fit_peak_series(ds, plan, pattern_indices=[])
    with pytest.raises(IndexError, match="out-of-range"):
        fit_peak_series(ds, plan, pattern_indices=[len(ds.pattern)])
