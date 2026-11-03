"""NEXT-128-WINDOW-POLISH item 11: an Int NeXus result's stored 1-D rows as fit inputs (headless).

The maintainer, 2026-11-03: "I think it will be good to have an option to choose a separate source than the nexus file
selected in the main window in the fit windows ... This also fits well with the export jupyter idea." His answers:
"Yes, in 1.2.8 with Batch (Recommended)" and "Stored rows; fit the substrate (Recommended)". Fit fits a source frame's
stored 1-D curve, Batch every frame of its file, and a notebook names the source. Frozen before the product change
(DESIGN-FIT-SOURCE.md rows S2, S3, S5 and S6, headless halves); new APIs are reached inside the test bodies, so each
missing one fails its own row.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.core import FrameRecord, FrameView, axis_from_unit
from xrd_tools.io import write_frame_records
from tests.core.v2_fixture_factory import current_entry

Q = np.linspace(1.0, 4.0, 301)
WAVELENGTH = 1.0  # Å, the file's monochromator


def _sources():
    return import_module("xrd_tools.analysis.fit_sources")


def _peak(center, height=400.0):
    return 10.0 + height * np.exp(-0.5 * ((Q - center) / 0.04) ** 2)


def _result(path: Path, labels=(0, 1, 2, 3, 4), *, shift=0.0, gi=False,
            wavelength: float | None = WAVELENGTH) -> Path:
    """An Int result: one Gaussian per frame, its centre moving with the label; ``gi`` adds a q_ip mode (in qip_A^-1)
    beside the primary q_total; the monochromator records ``wavelength`` (None: no wavelength)."""
    records = []
    for label in labels:
        primary = FrameView(label=label, axis_1d=axis_from_unit("q_A^-1", Q),
                            intensity_1d=_peak(2.0 + 0.1 * label + shift), sigma_1d=np.full(Q.size, 2.0))
        if gi:
            ip = FrameView(label=label, axis_1d=axis_from_unit("qip_A^-1", Q), intensity_1d=_peak(3.0 - 0.1 * label))
            records.append(FrameRecord.from_view(primary, mode_1d="q_total").with_result_1d("q_ip", ip,
                                                                                         make_active=False))
        else:
            records.append(FrameRecord.from_view(primary))
    with h5py.File(path, "w") as handle:
        entry = current_entry(handle)
        write_frame_records(entry, records)
        if wavelength is not None:
            entry.require_group("instrument").require_group("monochromator").create_dataset(
                "wavelength", data=float(wavelength))
    return path


def _mode(path):
    from xrd_tools.io.frame_view import FrameViewReader
    with FrameViewReader(path, include_thumbnail=False, resolve_source=False) as reader:
        return reader.modes_1d()[0]


# ---- the reader's per-mode inventory --------------------------------------------------------------------------------

def test_the_reader_lists_the_frames_with_a_row_in_a_mode(tmp_path) -> None:
    from xrd_tools.io.frame_view import FrameViewReader
    path = _result(tmp_path / "gi_int2d.nexus", labels=(0, 2, 5), gi=True)
    with FrameViewReader(path, include_thumbnail=False, resolve_source=False) as reader:
        assert reader.labels_1d("q_total") == (0, 2, 5) and reader.labels_1d("q_ip") == (0, 2, 5)
        with pytest.raises(ValueError, match="no 1-D mode 'q_oop'"):
            reader.labels_1d("q_oop")


# ---- S2: one frame's stored row --------------------------------------------------------------------------------

def test_a_frame_is_its_stored_row_on_its_stored_axis_at_the_files_wavelength(tmp_path) -> None:
    path = _result(tmp_path / "sample_int1d.nexus")
    mode = _mode(path)
    snapshot = _sources().int_frame_snapshot(path, 3, mode=mode)
    axis, = snapshot.axes
    assert (axis.name, axis.unit) == ("Q", "q_A^-1") and np.array_equal(axis.values, Q)
    assert np.array_equal(snapshot.values, _peak(2.3)) and np.array_equal(snapshot.sigma, np.full(Q.size, 2.0))
    assert snapshot.wavelength_angstrom == pytest.approx(WAVELENGTH, rel=1e-12, abs=0.0)
    key = snapshot.key
    assert key.path == str(path) and key.generation == 0 and key.background == "none"
    assert dict(key.selection) == {"meaning": "radial", "frame": 3, "mode": mode}
    assert "sample_int1d" in key.result_identity and "3" in key.result_identity
    assert len(key.fingerprint) == 64
    again = _sources().int_frame_snapshot(path, 3, mode=mode)
    assert again.key == key  # the same stored row is the same input
    other = _sources().int_frame_snapshot(_result(tmp_path / "shifted_int1d.nexus", shift=0.05), 3, mode=mode)
    assert other.key.fingerprint != key.fingerprint  # another curve is another input


def test_a_missing_frame_or_mode_is_refused_by_name(tmp_path) -> None:
    path = _result(tmp_path / "gi_int2d.nexus", labels=(0, 1), gi=True)
    with pytest.raises(ValueError, match="FIT_SOURCE_FRAME_MISSING: gi_int2d.nexus has no frame 7"):
        _sources().int_frame_snapshot(path, 7, mode="q_total")
    with pytest.raises(ValueError, match=r"FIT_SOURCE_MODE_MISSING: gi_int2d.nexus has no 1-D mode 'q_oop'; "
                                         r"its modes are q_total, q_ip"):
        _sources().int_frame_snapshot(path, 0, mode="q_oop")
    with pytest.raises(ValueError, match="FIT_SOURCE_UNREADABLE"):
        _sources().int_frame_snapshot(tmp_path / "absent.nexus", 0, mode="default")


def test_a_frame_in_a_secondary_mode_is_that_modes_row(tmp_path) -> None:
    path = _result(tmp_path / "gi_int2d.nexus", gi=True)
    snapshot = _sources().int_frame_snapshot(path, 1, mode="q_ip")
    assert snapshot.axes[0].unit == "qip_A^-1" and np.array_equal(snapshot.values, _peak(2.9))


def test_a_displayed_unit_converts_q_and_two_theta_with_ints_function_and_refuses_others(tmp_path) -> None:
    from xrd_tools.core.containers import convert_radial_axis
    path = _result(tmp_path / "sample_int1d.nexus")
    mode = _mode(path)
    snapshot = _sources().int_frame_snapshot(path, 2, mode=mode, unit="2th_deg")
    axis, = snapshot.axes
    assert axis.unit == "2th_deg" and axis.name == axis_from_unit("2th_deg", axis.values).label
    assert np.array_equal(axis.values, convert_radial_axis(Q, "q_A^-1", "2th_deg", WAVELENGTH))
    assert np.array_equal(snapshot.values, _peak(2.2)) and dict(snapshot.key.selection)["unit"] == "2th_deg"
    with pytest.raises(ValueError, match="FIT_SOURCE_AXIS_MISMATCH"):
        _sources().int_frame_snapshot(path, 2, mode=mode, unit="qip_A^-1")
    no_wavelength = _result(tmp_path / "unknown_int1d.nexus", wavelength=None)
    assert _sources().int_frame_snapshot(no_wavelength, 2, mode=mode).wavelength_angstrom is None
    with pytest.raises(ValueError, match="FIT_SOURCE_WAVELENGTH_REQUIRED"):
        _sources().int_frame_snapshot(no_wavelength, 2, mode=mode, unit="2th_deg")


# ---- S3: a batch's rows ------------------------------------------------------------------------------------------

def test_a_batch_reads_every_frame_in_chunks_and_each_equals_the_single_frame(tmp_path) -> None:
    path = _result(tmp_path / "sample_int1d.nexus", labels=(0, 1, 2, 3, 4))
    mode = _mode(path)
    rows = _sources().IntResultRows(path, mode=mode, chunk=2)
    assert rows.labels() == (0, 1, 2, 3, 4)
    for label in rows.labels():
        single = _sources().int_frame_snapshot(path, label, mode=mode)
        chunked = rows.snapshot(label)
        assert chunked.key == single.key and np.array_equal(chunked.values, single.values)
        assert chunked.wavelength_angstrom == single.wavelength_angstrom
    with pytest.raises(ValueError, match="FIT_SOURCE_FRAME_MISSING"):
        rows.snapshot(9)


def test_the_displayed_fit_analyzer_batch_over_the_rows_equals_each_frames_single_fit(tmp_path) -> None:
    from xrd_tools.analysis.display_fit_operations import DisplayedPeakFitPlan, run_displayed_peak_fit
    from xrd_tools.analysis.runner import DisplayedPeakFitAnalyzer, run_batch
    path = _result(tmp_path / "sample_int1d.nexus", labels=(0, 1, 2))
    rows = _sources().IntResultRows(path, mode=_mode(path))
    outcomes = run_batch(DisplayedPeakFitAnalyzer(settings={}), tuple(
        (lambda label=label: rows.snapshot(label)) for label in rows.labels()), compact=True)
    assert [outcome.label for outcome in outcomes] == [rows.snapshot(label).key.result_identity
                                                      for label in rows.labels()]
    for label, outcome in zip(rows.labels(), outcomes):
        single = run_displayed_peak_fit(DisplayedPeakFitPlan(rows.snapshot(label).trace_input()))
        assert outcome.ok and outcome.parameter_values == dict(zip(single.parameter_names, single.parameter_values))
        assert outcome.params["center_0"] == pytest.approx(2.0 + 0.1 * label, abs=1e-3)


# ---- S5: an .xye source ------------------------------------------------------------------------------------------

def test_an_xye_source_reads_its_columns_in_its_chosen_unit_without_a_wavelength(tmp_path) -> None:
    path = tmp_path / "pattern.xye"
    path.write_text("".join(f"{q:.6f} {y:.6f} 1.5\n" for q, y in zip(Q, _peak(2.5))), encoding="utf-8")
    snapshot = _sources().xye_fit_snapshot(path, unit="q_A^-1")
    axis, = snapshot.axes
    assert axis.unit == "q_A^-1" and np.allclose(axis.values, Q, atol=1e-6) and snapshot.wavelength_angstrom is None
    assert np.allclose(snapshot.values, _peak(2.5), atol=1e-6) and np.allclose(snapshot.sigma, 1.5)
    assert snapshot.key.path == str(path) and dict(snapshot.key.selection)["unit"] == "q_A^-1"
    from xrd_tools.analysis.fit_sources import phase_fit_snapshot
    with pytest.raises(ValueError, match="PHASE_WAVELENGTH_REQUIRED"):
        phase_fit_snapshot(snapshot)  # Phase Analysis refuses an .xye source at choose time for this reason
    with pytest.raises(ValueError, match="FIT_SOURCE_UNREADABLE"):
        _sources().xye_fit_snapshot(tmp_path / "absent.xye", unit="q_A^-1")


# ---- S6: a notebook names the source -----------------------------------------------------------------------------

def test_a_notebook_names_an_int_frame_and_replays_it_or_refuses_a_changed_row(tmp_path) -> None:
    from xrd_tools.notebooks import FitNotebookRecipe, load_snapshot, recipe_from_json, recipe_to_json
    from xrd_tools.notebooks.inputs import NotebookInputError, SourceReference, int_frame_reference
    data = tmp_path / "data"
    data.mkdir()
    path = _result(data / "sample_int1d.nexus")
    mode = _mode(path)
    snapshot = _sources().int_frame_snapshot(path, 4, mode=mode, unit="2th_deg")
    reference = int_frame_reference(snapshot, data_root=data)
    assert type(reference) is SourceReference and reference.reader == "int_1d"
    assert reference.path == "sample_int1d.nexus" and reference.original_path == str(path)
    assert reference.result_fingerprint == snapshot.key.fingerprint and reference.label == snapshot.key.result_identity
    assert dict(reference.selection) == {"entry": "entry", "frame": 4, "mode": mode, "unit": "2th_deg"}
    recipe = FitNotebookRecipe("peak", reference, batch=(reference,))
    assert recipe_from_json(recipe_to_json(recipe)) == recipe
    replayed = load_snapshot(reference, bundle_root=tmp_path, data_root=data)
    assert np.array_equal(replayed.values, snapshot.values) and replayed.axes[0].unit == "2th_deg"
    assert np.array_equal(replayed.axes[0].values, snapshot.axes[0].values)
    assert replayed.key.fingerprint == snapshot.key.fingerprint
    _result(path, shift=0.02)  # the file is rewritten with other curves
    with pytest.raises(NotebookInputError, match="INPUT_CHANGED"):
        load_snapshot(reference, bundle_root=tmp_path, data_root=data)


def test_an_exported_int_frame_notebook_replays_the_guis_fit(tmp_path) -> None:
    from xrd_tools.analysis.display_fit_operations import DisplayedPeakFitPlan, run_displayed_peak_fit
    from xrd_tools.notebooks import FitNotebookRecipe, run_recipe, run_recipe_batch
    from xrd_tools.notebooks.inputs import int_frame_reference
    path = _result(tmp_path / "sample_int1d.nexus", labels=(0, 1, 2))
    rows = _sources().IntResultRows(path, mode=_mode(path))
    references = tuple(int_frame_reference(rows.snapshot(label), data_root=tmp_path) for label in rows.labels())
    recipe = FitNotebookRecipe("peak", references[-1], batch=references)
    outcome = run_recipe(recipe, bundle_root=tmp_path, data_root=tmp_path)
    single = run_displayed_peak_fit(DisplayedPeakFitPlan(rows.snapshot(2).trace_input()))
    assert outcome.ok and outcome.parameter_values == dict(zip(single.parameter_names, single.parameter_values))
    batch = run_recipe_batch(recipe, bundle_root=tmp_path, data_root=tmp_path)
    assert [round(item.params["center_0"], 3) for item in batch] == [2.0, 2.1, 2.2]
