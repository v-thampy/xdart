from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

from xrd_tools.analysis.plans import PeakFitPlan
from xrd_tools.core import IntegrationResult1D, IntegrationResult2D
from xrd_tools.analysis.time_resolved import (
    LinearThermalExpansion,
    TabulatedThermalExpansion,
    add_lattice_results,
    add_temperature_results,
    bin_time_resolved,
    discover_processed_scans,
    export_time_resolved_results,
    fit_peak_series,
    flag_fit_quality,
    flag_normalization_outliers,
    lattice_from_q,
    load_time_resolved_series,
    normalize_monitor,
    normalize_reference_band,
    select_time_zero,
)
from xrd_tools.io.nexus import write_nexus
from xrd_tools.viz.plotly import (
    plot_peak_fit_frame,
    plot_thermal_history,
    plot_waterfall,
)


def _write_scan(
    path: Path,
    *,
    q: np.ndarray,
    intensity: np.ndarray,
    frame_start: int = 1,
    include_2d: bool = False,
    scan_data: dict[str, np.ndarray] | None = None,
    q_unit: str = "q_A^-1",
) -> Path:
    frames = np.arange(frame_start, frame_start + len(intensity), dtype=np.int64)
    q = np.asarray(q, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)
    results_1d = {
        int(frame): IntegrationResult1D(
            radial=q,
            intensity=row,
            sigma=np.sqrt(np.maximum(row, 0)).astype(np.float32),
            unit=q_unit,
        )
        for frame, row in zip(frames, intensity, strict=True)
    }
    results_2d = None
    if include_2d:
        chi = np.linspace(-20, 20, 5, dtype=np.float32)
        q2 = np.linspace(q.min(), q.max(), 6, dtype=np.float32)
        cake = np.arange(len(frames) * len(chi) * len(q2), dtype=np.float32)
        cakes = cake.reshape(len(frames), len(chi), len(q2))
        results_2d = {
            int(frame): IntegrationResult2D(
                radial=q2,
                azimuthal=chi,
                intensity=row.T,
                unit="q_A^-1",
                azimuthal_unit="chi_deg",
            )
            for frame, row in zip(frames, cakes, strict=True)
        }
    write_nexus(
        path,
        results_1d=results_1d,
        results_2d=results_2d,
        overwrite=True,
        compression=None,
    )
    with h5py.File(path, "r+") as h5:
        entry = h5["entry"]

        if scan_data:
            scan_group = entry.create_group("scan_data")
            scan_group.create_dataset("frame_index", data=frames)
            for name, values in scan_data.items():
                values = np.asarray(values)
                if values.shape != frames.shape:
                    raise ValueError(f"scan_data {name!r} does not match frame count")
                scan_group.create_dataset(name, data=values)

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
    a = _write_scan(tmp_path / "scan_2.nexus", q=q, intensity=y1, include_2d=True)
    b = _write_scan(tmp_path / "scan_10.nexus", q=q, intensity=y2)
    return a, b


def test_discover_and_load_time_resolved_series(scan_pair, tmp_path):
    paths = discover_processed_scans(tmp_path)
    assert [p.name for p in paths] == ["scan_2.nexus", "scan_10.nexus"]

    series = load_time_resolved_series(paths, frame_period_s=0.002)
    ds = series.dataset
    assert ds["intensity"].shape == (5, 81)
    assert ds.coords["scan_name"].values.tolist() == [
        "scan_2", "scan_2", "scan_2", "scan_10", "scan_10"]
    np.testing.assert_allclose(ds.coords["time"].values, [0, .002, .004, 0, .002])
    assert ds.coords["time"].attrs["units"] == "s"

    cake = series.get_cake(0)
    assert cake.intensity.shape == (5, 6)
    thumbnail = series.get_thumbnail(0)
    assert thumbnail.shape == (3, 4)
    with pytest.raises(KeyError, match="thumbnail fallback disabled"):
        series.get_raw(0)


def test_load_q_grid_policy_interpolates_or_fails(tmp_path):
    q1 = np.linspace(1.0, 5.0, 81)
    q2 = np.linspace(2.0, 4.0, 61)
    a = _write_scan(tmp_path / "a.nexus", q=q1, intensity=np.ones((1, len(q1))))
    b = _write_scan(tmp_path / "b.nexus", q=q2, intensity=np.ones((1, len(q2))) * 2)

    with pytest.raises(ValueError, match="q grid"):
        load_time_resolved_series([a, b], q_policy="strict")

    ds = load_time_resolved_series([a, b], q_policy="interpolate").dataset
    assert ds["intensity"].shape == (2, len(q1))
    assert ds.coords["q_interpolated"].values.tolist() == [False, True]
    assert np.isnan(ds["intensity"].values[1, 0])
    assert np.isnan(ds["intensity"].values[1, -1])
    np.testing.assert_allclose(ds["intensity"].values[1, 20:-20], 2.0)
    assert ds.attrs["q_interpolation"]["outside_source_coverage"] == "nan"


def test_discovery_ignores_raw_and_monitor_metadata_is_aligned(tmp_path):
    q = np.linspace(1.0, 5.0, 9)
    processed = _write_scan(
        tmp_path / "scan_10.nexus",
        q=q,
        intensity=np.vstack([np.ones_like(q) * 2, np.ones_like(q) * 4]),
        scan_data={"i0": np.array([2.0, 4.0])},
    )
    with h5py.File(tmp_path / "raw_2.nxs", "w") as h5:
        h5.create_group("entry").create_dataset("detector", data=np.ones((2, 3)))

    assert discover_processed_scans(tmp_path) == [processed]
    series = load_time_resolved_series(tmp_path, metadata_keys=("i0",))
    np.testing.assert_allclose(series.dataset.coords["i0"], [2.0, 4.0])
    normalized = normalize_monitor(series.dataset, "i0", target=1.0)
    np.testing.assert_allclose(normalized["intensity_normalized"], 1.0)
    assert normalized["monitor_normalization_valid"].values.tolist() == [True, True]


def test_mixed_timing_is_explicit_and_rate_requires_seconds(tmp_path):
    q = np.linspace(1.0, 5.0, 9)
    timed = _write_scan(
        tmp_path / "timed.nexus",
        q=q,
        intensity=np.ones((2, len(q))),
        scan_data={"elapsed_time": np.array([4.0, 4.2])},
    )
    untimed = _write_scan(
        tmp_path / "untimed.nexus", q=q, intensity=np.ones((2, len(q))))
    ds = load_time_resolved_series(
        [timed, untimed],
        time_key={timed.name: "elapsed_time"},
        time_unit={timed.name: "s"},
    ).dataset
    assert ds.coords["time"].attrs["units"] == "frame"
    np.testing.assert_allclose(ds.coords["time"], [0, 1, 0, 1])
    np.testing.assert_allclose(ds.coords["time_seconds"].values[:2], [0.0, 0.2])
    assert np.isnan(ds.coords["time_seconds"].values[2:]).all()
    assert ds.attrs["time_has_mixed_sources"] is True

    lattice = ds.isel(pattern=slice(0, 2)).rename({"pattern": "fit_pattern"})
    lattice["lattice_mean_A"] = ("fit_pattern", np.array([3.9, 3.91]))
    with pytest.raises(ValueError, match="physical seconds"):
        add_temperature_results(
            lattice,
            LinearThermalExpansion(3.9, 300.0, 1e-5),
            time_coord="time",
        )


def test_time_columns_require_selected_units_and_one_frame_does_not_invent_cadence(tmp_path):
    q = np.linspace(1.0, 5.0, 9)
    milliseconds = _write_scan(
        tmp_path / "milliseconds.nexus",
        q=q,
        intensity=np.ones((2, len(q))),
        scan_data={"elapsed": np.array([10.0, 12.0])},
    )

    # A suggestive name is not evidence that the values are seconds.
    unknown = load_time_resolved_series(milliseconds).dataset
    assert unknown.coords["time"].attrs["units"] == "frame"
    assert np.isnan(unknown.coords["time_seconds"]).all()
    with pytest.raises(ValueError, match="explicit time_unit"):
        load_time_resolved_series(milliseconds, time_key="elapsed")

    explicit = load_time_resolved_series(
        milliseconds, time_key="elapsed", time_unit="ms").dataset
    np.testing.assert_allclose(explicit.coords["time"], [0.0, 0.002])
    assert explicit.coords["time"].attrs["units"] == "s"
    assert explicit.coords["time_source"].values.tolist() == [
        "scan_data:elapsed[ms]", "scan_data:elapsed[ms]"]

    first = _write_scan(
        tmp_path / "first.nexus", q=q, intensity=np.ones((1, len(q))),
        scan_data={"clock": np.array([50.0])},
    )
    second = _write_scan(
        tmp_path / "second.nexus", q=q, intensity=np.ones((1, len(q))),
        scan_data={"clock": np.array([100.0])},
    )
    one_frame_scans = load_time_resolved_series(
        [first, second], time_key="clock", time_unit="ms").dataset
    np.testing.assert_allclose(one_frame_scans.coords["time"], [0.0, 0.0])
    assert one_frame_scans.coords["sequence_time"].values[0] == 0.0
    assert np.isnan(one_frame_scans.coords["sequence_time"].values[1])


def test_stacked_q_units_are_complete_compatible_and_canonical(tmp_path):
    q = np.linspace(1.0, 5.0, 9)
    intensity = np.ones((2, len(q)))
    canonical = _write_scan(tmp_path / "canonical.nexus", q=q, intensity=intensity, q_unit="q_A^-1")
    alias = _write_scan(tmp_path / "alias.nexus", q=q, intensity=intensity, q_unit="angstrom^-1")
    missing = _write_scan(tmp_path / "missing.nexus", q=q, intensity=intensity, q_unit="")
    inverse_nm = _write_scan(tmp_path / "inverse_nm.nexus", q=q, intensity=intensity, q_unit="q_nm^-1")

    compatible = load_time_resolved_series([canonical, alias]).dataset
    assert compatible.coords["q"].attrs["units"] == "q_A^-1"
    assert compatible.attrs["q_unit"] == "q_A^-1"

    with pytest.raises(ValueError, match="every stacked scan"):
        load_time_resolved_series([canonical, missing])
    with pytest.raises(ValueError, match="every stacked scan"):
        load_time_resolved_series([missing, canonical])
    with pytest.raises(ValueError, match="differs from reference"):
        load_time_resolved_series([canonical, inverse_nm])

    unknown = load_time_resolved_series([missing, missing]).dataset
    assert unknown.coords["q"].attrs["units"] == ""
    with pytest.raises(ValueError, match="inverse-angstrom"):
        lattice_from_q([2.0], (1, 1, 1), q_unit=unknown.coords["q"].attrs["units"])


def test_time_resolved_raw_accessor_is_strict_and_qualified(scan_pair, monkeypatch):
    from xrd_tools.analysis import time_resolved

    series = load_time_resolved_series(scan_pair, frame_period_s=0.002)
    observed = {}

    def _fake_raw(path, frame, **kwargs):
        observed.update(path=Path(path), frame=frame, **kwargs)
        return np.ones((20, 30))

    monkeypatch.setattr(time_resolved, "get_raw_frame", _fake_raw)
    raw = series.get_raw(3)
    assert raw.shape == (20, 30)
    assert observed == {
        "path": scan_pair[1],
        "frame": 1,
        "allow_thumbnail": False,
        "source_root": None,
    }


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
    path = _write_scan(tmp_path / "peaks.nexus", q=q, intensity=np.asarray(intensity))
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
    assert "temperature_error_K" in thermal
    assert thermal.attrs["temperature_uncertainty"] == "propagated from lattice uncertainty"

    table = TabulatedThermalExpansion(
        temperature_K=(300.0, 600.0, 900.0),
        lattice_A=(3.92, 3.93, 3.95),
    )
    np.testing.assert_allclose(table.temperature([3.92, 3.93]), [300, 600])
    assert np.isnan(table.temperature([3.90])).all()
    np.testing.assert_allclose(table.temperature_derivative_per_A([3.92, 3.93]), [30000, 15000])

    original = ds.copy(deep=True)
    waterfall = plot_waterfall(ds, y_coord="time")
    fit_figure = plot_peak_fit_frame(thermal, 0)
    thermal_figure = plot_thermal_history(thermal)
    assert len(waterfall.data) == 1
    assert len(fit_figure.data) >= 3
    assert len(thermal_figure.data) >= 4
    assert ds.identical(original)

    with pytest.raises(ValueError, match="no patterns"):
        fit_peak_series(ds, plan, pattern_indices=[])
    with pytest.raises(IndexError, match="out-of-range"):
        fit_peak_series(ds, plan, pattern_indices=[len(ds.pattern)])

    paths = export_time_resolved_results(
        thermal,
        netcdf_path=tmp_path / "results.nc",
        csv_path=tmp_path / "results.csv",
    )
    assert paths["netcdf"].is_file() and paths["csv"].is_file()
    assert "center_0" in (tmp_path / "results.csv").read_text()


@pytest.mark.parametrize("unit", ["2th_deg", "q_nm^-1", "", None])
def test_lattice_conversion_rejects_non_inverse_angstrom_axes(unit):
    fits = xr.Dataset(
        {"center_0": ("fit_pattern", np.array([2.76]))},
        coords={"fit_pattern": [0], "q_fit": np.linspace(2.6, 2.9, 9)},
    )
    if unit is not None:
        fits.coords["q_fit"].attrs["units"] = unit
    with pytest.raises(ValueError, match="inverse-angstrom"):
        add_lattice_results(fits, hkls=((1, 1, 1),))


def test_lattice_conversion_accepts_documented_inverse_angstrom_aliases():
    fits = xr.Dataset(
        {"center_0": ("fit_pattern", np.array([2.76]))},
        coords={"fit_pattern": [0], "q_fit": np.linspace(2.6, 2.9, 9)},
    )
    fits.coords["q_fit"].attrs["units"] = "1/angstrom"
    lattice = add_lattice_results(fits, hkls=((1, 1, 1),))
    assert lattice["lattice_A"].values[0, 0] == pytest.approx(3.943, rel=1e-3)
