"""NEXT-128-PEAK-FIT: file-sourced background templates, shared by the Peak Fitter and Phase Analysis (headless).

Frozen before the product change (DESIGN-PEAK-FIT-128.md, its controlling amendment Codex REVIEW-1, and the PLAN's
contract revisions; oracle rows T1-T6 and the headless halves of T4 and T8). His words: "The function itself should
be agnostic to where the BG comes from". New APIs are reached inside test bodies, so a missing one fails its own row;
the phase-model preservation row is green on the parent.
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.core import FrameRecord, FrameView, axis_from_unit
from xrd_tools.io import write_frame_records
from tests.core.v2_fixture_factory import current_entry

Q = np.linspace(1.0, 2.0, 6)


def _owner():
    return import_module("xrd_tools.analysis.background_template")


def _refused(code):
    return pytest.raises(_owner().BackgroundTemplateRefused, match=code)


def _two_mode_file(path: Path, *, entry="entry_b") -> Path:
    """Frames 1 and 2 under a nondefault entry: the primary q_ip curve (qip_A^-1) and a different secondary q_total
    curve stored in q_nm^-1."""
    records = []
    for label in (1, 2):
        primary = FrameView(label=label, axis_1d=axis_from_unit("qip_A^-1", Q), intensity_1d=np.arange(6.0) + label)
        secondary = FrameView(label=label, axis_1d=axis_from_unit("q_nm^-1", Q * 10.0),
                              intensity_1d=np.arange(6.0) * 7.0 + label)
        records.append(FrameRecord.from_view(primary, mode_1d="q_ip")
                       .with_result_1d("q_total", secondary, make_active=False))
    with h5py.File(path, "w") as handle:
        write_frame_records(current_entry(handle, entry=entry), records)
    return path


def _single_mode_file(path: Path) -> Path:
    records = [FrameRecord.from_view(FrameView(label=label, axis_1d=axis_from_unit("q_A^-1", Q),
                                               intensity_1d=np.arange(6.0) * 3.0 + label)) for label in (1, 2)]
    with h5py.File(path, "w") as handle:
        write_frame_records(current_entry(handle), records)
    return path


def _xye(path: Path, rows: str) -> Path:
    path.write_text(rows, encoding="utf-8")
    return path


def _nexus(path, frame, mode, entry="entry"):
    return _owner().BackgroundTemplateSource("nexus", str(path), frame=frame, mode=mode, entry=entry)


def _xye_source(path, unit):
    return _owner().BackgroundTemplateSource("xye", str(path), unit=unit)


# ---- T1 the source codec ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ("nexus", "xye"))
def test_t1_a_source_round_trips_through_pairs_recipe_json_and_settings_json(tmp_path, kind) -> None:
    owner = _owner()
    from xrd_tools.notebooks import FitNotebookRecipe, recipe_from_json, recipe_to_json
    from xrd_tools.notebooks.inputs import SnapshotReference
    source = (_nexus(tmp_path / "a.nexus", 2, "q_total", entry="entry_b") if kind == "nexus"
              else _xye_source(tmp_path / "sub 'α'.xye", "q_nm^-1"))
    pairs = source.to_pairs()
    assert type(pairs) is tuple and pairs == tuple(sorted(pairs)) and ("version", 1) in pairs
    assert all(type(key) is str and type(value) in (str, int) for key, value in pairs)
    assert owner.BackgroundTemplateSource.from_pairs(pairs) == source
    recipe = FitNotebookRecipe("peak", SnapshotReference("selected.npz", "0" * 64, "selected"),
                               settings={"background_template_source": pairs, "background_template_identity": "0" * 64})
    settings = dict(recipe_from_json(recipe_to_json(recipe)).settings)
    assert owner.BackgroundTemplateSource.from_pairs(settings["background_template_source"]) == source
    section = json.loads(json.dumps(dict(pairs)))  # a settings document's section is an object
    assert owner.BackgroundTemplateSource.from_pairs(section) == source


@pytest.mark.parametrize("pairs, code", (
    ((("kind", "csv"), ("path", "a"), ("version", 1)), "BACKGROUND_TEMPLATE_INVALID_SOURCE"),
    ((("kind", "xye"), ("path", "a.xye"), ("unit", "q_A^-1"), ("version", 2)), "BACKGROUND_TEMPLATE_INVALID_SOURCE"),
    ((("kind", "xye"), ("path", "a.xye"), ("unit", "q_A^-1")), "BACKGROUND_TEMPLATE_INVALID_SOURCE"),
    ((("entry", "entry"), ("kind", "nexus"), ("mode", "default"), ("path", "a.nexus"), ("version", 1)),
     "BACKGROUND_TEMPLATE_INVALID_SOURCE"),
    ((("entry", "entry"), ("frame", True), ("kind", "nexus"), ("mode", "default"), ("path", "a.nexus"),
      ("version", 1)), "BACKGROUND_TEMPLATE_INVALID_SOURCE"),
    ((("kind", "xye"), ("path", ""), ("unit", "q_A^-1"), ("version", 1)), "BACKGROUND_TEMPLATE_INVALID_SOURCE"),
    ((("kind", "xye"), ("path", "a.xye"), ("unit", "q_A^-1"), ("version", 1), ("extra", 1)),
     "BACKGROUND_TEMPLATE_INVALID_SOURCE"),
    ((("kind", "xye"), ("path", "a.xye"), ("version", 1)), "BACKGROUND_TEMPLATE_UNIT_REQUIRED"),
))
def test_t1_malformed_sources_refuse_by_name(pairs, code) -> None:
    with _refused(code):
        _owner().BackgroundTemplateSource.from_pairs(pairs)


def test_t1_an_xye_source_needs_an_explicit_unit() -> None:
    with _refused("BACKGROUND_TEMPLATE_UNIT_REQUIRED"):
        _owner().BackgroundTemplateSource("xye", "substrate.xye")


# ---- T2 the exact NeXus curve -----------------------------------------------------------------------------------

def test_t2_a_nexus_source_reads_its_exact_entry_frame_and_mode(tmp_path) -> None:
    owner = _owner()
    path = _two_mode_file(tmp_path / "two_mode.nexus")
    selected = owner.prepare_background_template(_nexus(path, 2, "q_total", entry="entry_b"))
    assert selected.unit == "q_nm^-1"
    np.testing.assert_allclose(selected.x, Q * 10.0)
    np.testing.assert_array_equal(selected.y, np.arange(6.0) * 7.0 + 2)
    assert not selected.x.flags.writeable and not selected.y.flags.writeable and len(selected.identity) == 64
    x, y = selected.on_axis("q_A^-1")
    np.testing.assert_allclose(x, Q)
    np.testing.assert_array_equal(y, selected.y)
    primary = owner.prepare_background_template(_nexus(path, 2, "q_ip", entry="entry_b"))
    assert primary.unit == "qip_A^-1" and not np.array_equal(primary.y, selected.y)
    assert primary.identity != selected.identity


def test_t2_a_single_result_file_names_its_mode_default(tmp_path) -> None:
    prepared = _owner().prepare_background_template(_nexus(_single_mode_file(tmp_path / "one.nexus"), 1, "default"))
    assert prepared.unit == "q_A^-1"
    np.testing.assert_array_equal(prepared.y, np.arange(6.0) * 3.0 + 1)


@pytest.mark.parametrize("frame, mode, entry, code", (
    (2, "q_total", "entry", "BACKGROUND_TEMPLATE_ENTRY_MISSING"),
    (9, "q_total", "entry_b", "BACKGROUND_TEMPLATE_FRAME_MISSING"),
    (2, "q_oop", "entry_b", "BACKGROUND_TEMPLATE_MODE_MISSING.*q_ip"),
    (2, "default", "entry_b", "BACKGROUND_TEMPLATE_MODE_MISSING"),
))
def test_t2_a_missing_selection_refuses_and_never_falls_back_to_the_primary(tmp_path, frame, mode, entry, code):
    path = _two_mode_file(tmp_path / "two_mode.nexus")
    with _refused(code):
        _owner().prepare_background_template(_nexus(path, frame, mode, entry=entry))


def test_t2_a_q_component_curve_is_refused_for_total_q_naming_both_units(tmp_path) -> None:
    prepared = _owner().prepare_background_template(
        _nexus(_two_mode_file(tmp_path / "two_mode.nexus"), 1, "q_ip", entry="entry_b"))
    with _refused("BACKGROUND_TEMPLATE_UNIT_UNSUPPORTED.*qip_A\\^-1.*q_A\\^-1"):
        prepared.on_axis("q_A^-1")


# ---- T3 .xye files and units ------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, hint", (
    ("iq_substrate.xye", "q_A^-1"), ("itth_substrate.xye", "2th_deg"), ("iqoop_frame.xye", "qoop_A^-1"),
    ("iqip_frame.xye", "qip_A^-1"), ("iexit_frame.xye", "exit_angle_deg"), ("substrate.xye", None),
    ("quartz.xye", None), ("/data/run/IQ_upper.xye", "q_A^-1"),
))
def test_t3_unit_hints_come_only_from_xdart_prefixes(name, hint) -> None:
    assert _owner().xye_unit_hint(name) == hint


@pytest.mark.parametrize("rows", ("10.0 5.0\n20.0 6.0\n30.0 7.0\n", "10.0\t5.0\t0.1\n20.0\t6.0\t0.1\n30.0\t7.0\t0.1\n"))
def test_t3_two_and_three_column_files_read_and_scale_from_inverse_nm(tmp_path, rows) -> None:
    owner = _owner()
    prepared = owner.prepare_background_template(_xye_source(_xye(tmp_path / "sub.xye", rows), "q_nm^-1"))
    assert prepared.unit == "q_nm^-1"
    x, y = prepared.on_axis("q_A^-1")
    np.testing.assert_allclose(x, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(y, [5.0, 6.0, 7.0])


def test_t3_a_q_component_xye_is_refused_for_total_q(tmp_path) -> None:
    prepared = _owner().prepare_background_template(
        _xye_source(_xye(tmp_path / "iqoop_sub.xye", "1 5\n2 6\n3 7\n"), "qoop_A^-1"))
    with _refused("BACKGROUND_TEMPLATE_UNIT_UNSUPPORTED.*qoop_A\\^-1.*q_A\\^-1"):
        prepared.on_axis("q_A^-1")


def test_t3_two_theta_converts_only_to_two_theta(tmp_path) -> None:
    prepared = _owner().prepare_background_template(
        _xye_source(_xye(tmp_path / "itth_sub.xye", "20 5\n30 6\n40 7\n"), "2th_deg"))
    with _refused("BACKGROUND_TEMPLATE_UNIT_UNSUPPORTED.*2th_deg.*q_A\\^-1"):
        prepared.on_axis("q_A^-1")
    np.testing.assert_array_equal(prepared.on_axis("2th_deg")[0], [20.0, 30.0, 40.0])
    np.testing.assert_allclose(prepared.on_axis("2th_rad")[0], np.radians([20.0, 30.0, 40.0]))


@pytest.mark.parametrize("rows", (
    "", "1.0 5.0\n", "1.0 nan\n2.0 6.0\n3.0 7.0\n", "1.0 5.0\n1.0 6.0\n2.0 7.0\n", "not numbers\n",
    "1 2 3 4\n2 3 4 5\n3 4 5 6\n",
))
def test_t3_malformed_files_refuse_by_name(tmp_path, rows) -> None:
    with _refused("BACKGROUND_TEMPLATE_(UNREADABLE|INVALID_CURVE)"):
        _owner().prepare_background_template(_xye_source(_xye(tmp_path / "sub.xye", rows), "q_A^-1"))


def test_t3_an_oversized_file_refuses_before_it_is_parsed(tmp_path) -> None:
    path = tmp_path / "huge.xye"
    with path.open("wb") as handle:
        handle.truncate(64 * 1024 * 1024 + 1)  # sparse: nothing is written or read
    with _refused("BACKGROUND_TEMPLATE_INVALID_CURVE"):
        _owner().prepare_background_template(_xye_source(path, "q_A^-1"))


def test_t3_unsorted_rows_are_sorted_and_the_identity_ignores_the_folder(tmp_path) -> None:
    owner = _owner()
    first = owner.prepare_background_template(_xye_source(_xye(tmp_path / "sub.xye", "3 7\n1 5\n2 6\n"), "q_A^-1"))
    np.testing.assert_array_equal(first.x, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(first.y, [5.0, 6.0, 7.0])
    moved = tmp_path / "moved"
    moved.mkdir()
    relative = owner.BackgroundTemplateSource("xye", "sub.xye", unit="q_A^-1")
    _xye(moved / "sub.xye", "3 7\n1 5\n2 6\n")
    again = owner.prepare_background_template(relative, data_root=moved, expected_identity=first.identity)
    assert again.identity == first.identity


def test_t3_t8_a_changed_file_refuses_its_recorded_identity(tmp_path) -> None:
    owner = _owner()
    path = _xye(tmp_path / "sub.xye", "1 5\n2 6\n3 7\n")
    source = _xye_source(path, "q_A^-1")
    first = owner.prepare_background_template(source)
    path.write_text("1 5\n2 9\n3 7\n", encoding="utf-8")
    with _refused("BACKGROUND_TEMPLATE_CHANGED"):
        owner.prepare_background_template(source, expected_identity=first.identity)
    reselected = owner.prepare_background_template(source)
    assert reselected.identity != first.identity
    path.unlink()
    with _refused("BACKGROUND_TEMPLATE_UNREADABLE"):
        owner.prepare_background_template(source, expected_identity=reselected.identity)


# ---- T5 the phase template's numerics, kept; the evaluator shared ------------------------------------------------

X_REF, Y_REF = np.array([3.0, 1.0, 2.0]), np.array([30.0, 10.0, 20.0])  # an unsorted reference
X_EVAL = np.array([0.0, 1.5, 2.5, 4.0])  # inside and outside the reference's support


def _scaled(model, amplitude):
    params = model.make_params()
    assert set(params) == {"bg_A"} and params["bg_A"].min == 0.0 and params["bg_A"].value == 1.0
    params["bg_A"].set(value=amplitude)
    return model.eval(params, x=X_EVAL)


def test_t5_keep_the_phase_template_interpolates_clamps_and_bounds_its_amplitude() -> None:
    from xrd_tools.analysis.fitting.phase_fitting import _make_fit_background_template_model
    np.testing.assert_allclose(_scaled(_make_fit_background_template_model(X_REF, Y_REF), 2.0), [20, 30, 50, 60])


def test_t5_the_shared_template_model_is_the_phase_model() -> None:
    from xrd_tools.analysis.fitting.models import template_background_model
    from xrd_tools.analysis.fitting.phase_fitting import _make_fit_background_template_model
    np.testing.assert_array_equal(_scaled(template_background_model(X_REF, Y_REF), 2.0),
                                  _scaled(_make_fit_background_template_model(X_REF, Y_REF), 2.0))


# ---- T6 the Peak Fitter's new behaviour ------------------------------------------------------------------------

def _peak_data(scale=2.0):
    x = np.linspace(1.0, 3.0, 401)
    template = 50.0 + 10.0 * np.sin(3.0 * x)
    peak = 120.0 * np.exp(-0.5 * ((x - 2.0) / 0.03) ** 2)
    return x, scale * template + peak, (x, template)


@pytest.mark.parametrize("background, extra", (
    ("template", set()), ("template+linear", {"bgx_slope", "bgx_intercept"}),
    ("template+chebyshev3", {"bgx_c0", "bgx_c1", "bgx_c2", "bgx_c3"}),
))
def test_t6_fit_peaks_scales_a_template_background(background, extra) -> None:
    from xrd_tools.analysis.fitting.fit import fit_peaks
    x, y, template = _peak_data()
    result = fit_peaks(x, y, positions=[2.0], model="gaussian", background=background, background_template=template)
    assert {"bg_A"} | extra <= set(result.params) and not any(name.startswith("bg_c") for name in result.params)
    assert abs(result.params["bg_A"].value - 2.0) < 0.05 and abs(result.peak_centers[0] - 2.0) < 1e-3
    seed = float(np.clip(np.nanmedian(y) / np.nanmedian(template[1]), 1e-3, 10.0))
    assert result.fit_result.init_values["bg_A"] == pytest.approx(seed, rel=1e-12)
    assert all(result.fit_result.init_values[name] == 0.0 for name in extra)


def test_t6_a_template_and_its_background_come_together() -> None:
    from xrd_tools.analysis.fitting.fit import fit_peaks
    x, y, template = _peak_data()
    with pytest.raises(ValueError, match="template"):
        fit_peaks(x, y, positions=[2.0], model="gaussian", background="template")
    with pytest.raises(ValueError, match="template"):
        fit_peaks(x, y, positions=[2.0], model="gaussian", background="linear", background_template=template)


def _trace(x, y, unit="q_A^-1"):
    from tests.core.test_p37_display_fit_operations import _trace as p37_trace
    return p37_trace(axis=x, intensity=y, axis_unit=unit)


def _peak_plan(trace, **changes):
    from tests.core.test_p37_display_fit_operations import _peak_plan as p37_plan
    return p37_plan(trace, **{"selection_mode": "count", "manual_centers": (2.0,), "n_peaks": 1, **changes})


def _run_peak(plan):
    from xrd_tools.analysis.display_fit_operations import run_displayed_peak_fit
    return run_displayed_peak_fit(plan)


def _values(result):
    return dict(zip(result.parameter_names, result.parameter_values))


@pytest.mark.parametrize("model", ("pearson7", "split_pseudovoigt"))
def test_t6_the_displayed_plan_admits_the_strain_models(model) -> None:
    x, y, _template = _peak_data()
    result = _run_peak(_peak_plan(_trace(x, y), model=model, background="linear"))
    assert result.code == "OK", (result.code, result.diagnostics)
    assert abs(_values(result)["p0_center"] - 2.0) < 2e-3


@pytest.mark.parametrize("background", ("template", "template+linear", "template+chebyshev3"))
@pytest.mark.parametrize("kind", ("xye", "nexus"))
def test_t6_a_file_template_fits_as_its_prepared_arrays_and_records_its_identity(tmp_path, background, kind) -> None:
    owner = _owner()
    from xrd_tools.analysis.fitting.fit import fit_peaks
    x, y, (tx, ty) = _peak_data()
    if kind == "xye":
        path = tmp_path / "iq_substrate.xye"
        np.savetxt(path, np.column_stack([tx * 10.0, ty]))  # stored in nm^-1, fitted in A^-1
        source = _xye_source(path, "q_nm^-1")
    else:
        path = tmp_path / "substrate.nexus"
        view = FrameView(label=3, axis_1d=axis_from_unit("q_A^-1", tx), intensity_1d=ty)
        with h5py.File(path, "w") as handle:
            write_frame_records(current_entry(handle), [FrameRecord.from_view(view)])
        source = _nexus(path, 3, "default")
    prepared = owner.prepare_background_template(source)
    result = _run_peak(_peak_plan(_trace(x, y), model="gaussian", background=background,
                                  background_template_source=source.to_pairs()))
    assert result.code == "OK", (result.code, result.diagnostics)
    assert result.background_template_identity == prepared.identity
    direct = fit_peaks(x, y, positions=[2.0], model="gaussian", background=background,
                       background_template=prepared.on_axis("q_A^-1"))
    assert _values(result)["bg_A"] == pytest.approx(direct.params["bg_A"].value, rel=1e-9)
    bound = _run_peak(_peak_plan(_trace(x, y), model="gaussian", background=background,
                                 background_template_source=source.to_pairs(),
                                 background_template_identity=prepared.identity))
    assert bound.parameter_values == result.parameter_values and bound.plan_fingerprint != result.plan_fingerprint


def test_t6_a_template_plan_keeps_the_validator_limits(tmp_path) -> None:
    x, y, (tx, ty) = _peak_data()
    path = tmp_path / "iq_substrate.xye"
    np.savetxt(path, np.column_stack([tx, ty]))
    pairs = _xye_source(path, "q_A^-1").to_pairs()
    for changes in ({"background": "template"}, {"background": "linear", "background_template_source": pairs},
                    {"background": "template", "background_template_source": (("kind", "csv"),)},
                    {"model": "moffat"}):
        result = _run_peak(_peak_plan(_trace(x, y), **changes))
        assert result.code != "OK", changes
    unit = _run_peak(_peak_plan(_trace(x, y, unit="2th_deg"), background="template",
                                background_template_source=pairs))
    assert unit.code == "BACKGROUND_TEMPLATE_UNIT_UNSUPPORTED", (unit.code, unit.diagnostics)


def test_t8_a_changed_file_refuses_a_bound_plan_and_reselecting_changes_the_request(tmp_path) -> None:
    owner = _owner()
    x, y, (tx, ty) = _peak_data()
    path = tmp_path / "iq_substrate.xye"
    np.savetxt(path, np.column_stack([tx, ty]))
    source = _xye_source(path, "q_A^-1")
    first = owner.prepare_background_template(source)
    np.savetxt(path, np.column_stack([tx, 1.5 * ty]))
    plan = _peak_plan(_trace(x, y), background="template", background_template_source=source.to_pairs(),
                      background_template_identity=first.identity)
    refused = _run_peak(plan)
    assert refused.code == "BACKGROUND_TEMPLATE_CHANGED", (refused.code, refused.diagnostics)
    reselected = _run_peak(_peak_plan(_trace(x, y), background="template",
                                      background_template_source=source.to_pairs()))
    assert reselected.code == "OK" and reselected.background_template_identity not in ("", first.identity)


def test_t4_a_displayed_capture_binds_only_when_its_file_reproduces_it(tmp_path) -> None:
    x, y, (tx, ty) = _peak_data()
    path = tmp_path / "substrate.nexus"
    view = FrameView(label=3, axis_1d=axis_from_unit("q_A^-1", tx), intensity_1d=ty)
    with h5py.File(path, "w") as handle:
        write_frame_records(current_entry(handle), [FrameRecord.from_view(view)])
    candidate = _nexus(path, 3, "default").to_pairs()
    stored = _owner().prepare_background_template(_nexus(path, 3, "default"))  # the file's float32 axis
    same = _run_peak(_peak_plan(_trace(x, y), background="template",
                                background_template=_trace(*stored.on_axis("q_A^-1")),
                                background_template_source=candidate))
    assert same.code == "OK" and same.background_template_identity == stored.identity
    # Int's displayed axis differs from the stored float32 axis by about one float32 step (PLAN decision 8)
    near = _run_peak(_peak_plan(_trace(x, y), background="template",
                                background_template=_trace(stored.x * (1 + 2e-7), stored.y),
                                background_template_source=candidate))
    assert near.code == "OK" and near.background_template_identity == stored.identity
    assert near.parameter_values == same.parameter_values  # a passing check fits the file's own curve
    differs = _run_peak(_peak_plan(_trace(x, y), background="template",
                                   background_template=_trace(stored.x, 0.9 * stored.y),
                                   background_template_source=candidate))
    assert differs.code == "OK" and differs.background_template_identity == ""
    assert any("BACKGROUND_TEMPLATE_UNREPRODUCIBLE" in line for line in differs.diagnostics)
    unnamed = _run_peak(_peak_plan(_trace(x, y), background="template",
                                   background_template=_trace(*stored.on_axis("q_A^-1"))))
    assert unnamed.code == "OK" and unnamed.background_template_identity == ""
    assert _values(unnamed)["bg_A"] == pytest.approx(_values(same)["bg_A"], rel=1e-12)
    assert _values(differs)["bg_A"] == pytest.approx(_values(same)["bg_A"] / 0.9, rel=1e-3)


# ---- T6/T7 Phase Analysis takes the same source ----------------------------------------------------------------

def test_t6_phase_analysis_fits_a_file_template_as_its_captured_arrays(tmp_path, monkeypatch) -> None:
    from xrd_tools.analysis.phase_references import PhaseLibrary
    from tests.core.test_phase_analysis_options import _cscl, _flat, _ok, _plan, _run, _trace as phase_trace
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "library"))
    entry = _cscl(PhaseLibrary(tmp_path / "library"), tmp_path)
    q, y = _flat()
    substrate = 1.0 + 0.2 * np.sin(q)
    path = tmp_path / "substrate.xye"
    np.savetxt(path, np.column_stack([q * 10.0, substrate]))
    source = _xye_source(path, "q_nm^-1")
    named = _ok(_run(_plan(phase_trace(q, y), (entry,), fit_background="template",
                           background_template_source=source.to_pairs())))
    prepared = _owner().prepare_background_template(source)
    captured = _ok(_run(_plan(phase_trace(q, y), (entry,), fit_background="template",
                              background_template=phase_trace(*prepared.on_axis("q_A^-1"), label="substrate"))))
    assert named.parameter_names == captured.parameter_names and "bg_A" in named.parameter_names
    assert named.parameter_values == pytest.approx(captured.parameter_values, rel=1e-12)
    assert named.background_template_identity == prepared.identity and captured.background_template_identity == ""
