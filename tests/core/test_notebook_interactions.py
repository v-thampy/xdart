"""Interaction contracts for the generated public notebook examples."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
import nbformat
import numpy as np
import pytest

from xrd_tools.core import IntegrationResult1D
from xrd_tools.io.nexus import write_nexus


matplotlib.use("Agg", force=True)


NOTEBOOK_ROOT = Path(__file__).parents[2] / "examples" / "notebooks"


def _execute_notebook(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    smoke: bool = True,
) -> dict[str, object]:
    """Execute clean cells in-process so visible widget callbacks can be exercised."""
    monkeypatch.setenv("XDART_NOTEBOOK_SMOKE", "1" if smoke else "0")
    monkeypatch.delenv("XDART_TEST_DATA", raising=False)
    namespace: dict[str, object] = {"__name__": "__notebook_test__"}
    notebook = nbformat.read(NOTEBOOK_ROOT / name, as_version=4)
    for number, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{name}:cell-{number}", "exec"), namespace)
    return namespace


def _write_folder_scan(path: Path, *, center: float) -> None:
    q = np.linspace(1.0, 3.3, 320)
    frames = np.arange(3, dtype=np.int64)
    intensity = np.array([
        8 + 140 * np.exp(-0.5 * ((q - (center - 0.0005 * frame)) / 0.02) ** 2)
        for frame in frames
    ], dtype=np.float32)
    write_nexus(
        path,
        results_1d={
            int(frame): IntegrationResult1D(q, row, sigma=np.sqrt(row), unit="q_A^-1")
            for frame, row in zip(frames, intensity, strict=True)
        },
        compression=None,
    )


def test_integration_stitch_and_peak_callbacks_change_results(monkeypatch: pytest.MonkeyPatch) -> None:
    integration = _execute_notebook("01_batch_integration.ipynb", monkeypatch)
    integration["npt_1d"].value = 96
    previous_runs = integration["NOTEBOOK_STATE"]["runs"]
    integration["run_button"].click()
    assert integration["NOTEBOOK_STATE"]["runs"] == previous_runs + 1
    assert integration["NOTEBOOK_STATE"]["result_1d"].radial.size == 96
    assert integration["NOTEBOOK_STATE"]["result_2d"].intensity.shape[1] == integration["npt_azim"].value

    stitch = _execute_notebook("02_multigeometry_stitching.ipynb", monkeypatch)
    stitch["q_range"].value = (1.2, 3.8)
    previous_runs = stitch["NOTEBOOK_STATE"]["runs"]
    stitch["compute"].click()
    assert stitch["NOTEBOOK_STATE"]["runs"] == previous_runs + 1
    assert stitch["NOTEBOOK_STATE"]["stitch_1d"].radial.size == 180
    assert stitch["NOTEBOOK_STATE"]["stitch_2d"].intensity.shape == (120, 48)

    peak = _execute_notebook("03_phase_and_peak_fitting.ipynb", monkeypatch)
    peak["controls"].sigma_init.value = 0.025
    previous_runs = peak["NOTEBOOK_STATE"]["runs"]
    peak["controls"]._fit_button.click()
    assert peak["NOTEBOOK_STATE"]["runs"] == previous_runs + 1
    assert peak["NOTEBOOK_STATE"]["outcome"].ok


def test_batch_sin2psi_and_reduction_callbacks_change_sinks(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _execute_notebook("04_batch_phase_fitting.ipynb", monkeypatch)
    batch["quality"].value = 0.03
    previous_runs = batch["NOTEBOOK_STATE"]["runs"]
    batch["run_button"].click()
    assert batch["NOTEBOOK_STATE"]["runs"] == previous_runs + 1
    assert len(batch["NOTEBOOK_STATE"]["outcomes"]) == 3

    sin2psi = _execute_notebook("05_sin2psi_analysis.ipynb", monkeypatch)
    sin2psi["q_range"].value = (2.8, 3.2)
    previous_runs = sin2psi["NOTEBOOK_STATE"]["runs"]
    sin2psi["fit_button"].click()
    assert sin2psi["NOTEBOOK_STATE"]["runs"] == previous_runs + 1
    assert sin2psi["NOTEBOOK_STATE"]["result"].r_squared > 0.9

    reduction = _execute_notebook("06_headless_reduction_pipeline.ipynb", monkeypatch)
    reduction["npt_1d"].value = 64
    reduction["include_2d"].value = False
    previous_runs = reduction["NOTEBOOK_STATE"]["runs"]
    reduction["run_button"].click()
    assert reduction["NOTEBOOK_STATE"]["runs"] == previous_runs + 1
    assert len(reduction["NOTEBOOK_STATE"]["sink"].frames) == 2
    token = reduction["CancelToken"]()
    reduction["NOTEBOOK_STATE"]["cancel_token"] = token
    reduction["cancel_button"].click()
    assert token.cancelled


def test_processed_time_resolved_and_rsm_callbacks_use_cached_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    processed = _execute_notebook("07_reading_processed_nxs.ipynb", monkeypatch)
    processed["frame_selector"].value = 7
    previous_runs = processed["NOTEBOOK_STATE"]["runs"]
    processed["inspect_button"].click()
    assert processed["NOTEBOOK_STATE"]["runs"] >= previous_runs + 1
    assert processed["NOTEBOOK_STATE"]["selected"] == 7
    assert processed["NOTEBOOK_STATE"]["cake"].intensity.shape == (12, 180)

    time_resolved = _execute_notebook("08_time_resolved_xrd_analysis.ipynb", monkeypatch)
    time_resolved["export_directory"] = tmp_path
    time_resolved["monitor_method"].value = "monitor"
    time_resolved["monitor_key"].value = "missing_monitor"
    time_resolved["load_button"].click()
    time_resolved["preprocess_button"].click()
    time_resolved["inspect_button"].click()
    time_resolved["fit_controls"]._fit_button.click()
    time_resolved["batch_button"].click()
    time_resolved["export_button"].click()
    assert time_resolved["NOTEBOOK_STATE"]["loads"] >= 2
    assert time_resolved["NOTEBOOK_STATE"]["prepared"] is not None
    assert time_resolved["NOTEBOOK_STATE"]["fits"] is not None
    assert time_resolved["NOTEBOOK_STATE"]["thermal"] is not None
    assert time_resolved["NOTEBOOK_STATE"]["raw_reads"] >= 1
    assert "fallback" in time_resolved["status"].value.lower() or "exported" in time_resolved["status"].value.lower()
    assert (tmp_path / "time_resolved_results.nc").is_file()
    assert (tmp_path / "time_resolved_scalars.csv").is_file()

    rsm = _execute_notebook("09_reciprocal_space_mapping.ipynb", monkeypatch)
    computes = rsm["NOTEBOOK_STATE"]["computes"]
    draws = rsm["NOTEBOOK_STATE"]["slice_draws"]
    rsm["slice_axis"].value = "h"
    assert rsm["NOTEBOOK_STATE"]["computes"] == computes
    assert rsm["NOTEBOOK_STATE"]["slice_draws"] == draws + 1
    rsm["compute"].click()
    assert rsm["NOTEBOOK_STATE"]["computes"] == computes + 1


def test_time_resolved_folder_selection_uses_only_selected_scans_and_sequence_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected_first = tmp_path / "scan_2.nexus"
    selected_second = tmp_path / "scan_10.nexus"
    _write_folder_scan(tmp_path / "scan_1.nexus", center=1.56)
    _write_folder_scan(selected_first, center=1.56)
    _write_folder_scan(selected_second, center=1.56)
    with h5py.File(tmp_path / "unrelated_99.nxs", "w") as h5:
        h5.create_group("entry").create_dataset("raw", data=np.ones((2, 2)))

    notebook = _execute_notebook("08_time_resolved_xrd_analysis.ipynb", monkeypatch, smoke=False)
    notebook["export_directory"] = tmp_path / "exports"
    notebook["selection_mode"].value = "folder"
    notebook["processed_folder"].value = str(tmp_path)
    notebook["scan_filter"].value = "scan_*.nexus"
    notebook["discover_button"].click()
    assert [label for label, _ in notebook["scan_selection"].options] == [
        "scan_1.nexus", "scan_2.nexus", "scan_10.nexus"
    ]

    notebook["scan_selection"].value = (str(selected_first), str(selected_second))
    notebook["bin_size"].value = 1
    notebook["load_button"].click()
    notebook["preprocess_button"].click()
    notebook["batch_button"].click()
    notebook["export_button"].click()

    assert notebook["NOTEBOOK_STATE"]["selected_scans"] == ("scan_2.nexus", "scan_10.nexus")
    assert notebook["NOTEBOOK_STATE"]["series"].dataset.sizes["pattern"] == 6
    assert notebook["NOTEBOOK_STATE"]["thermal_time_coord"] == "sequence_time"
    assert np.all(np.diff(notebook["NOTEBOOK_STATE"]["thermal"].coords["sequence_time"].values) > 0)
    assert (notebook["export_directory"] / "time_resolved_results.nc").is_file()
    assert (notebook["export_directory"] / "time_resolved_scalars.csv").is_file()
