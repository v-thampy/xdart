"""Native science, saved result and output slot tests for per-scan Live."""

from dataclasses import replace
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.core.test_bluesky_nexus import _write_bluesky_nxwriter
from tests.core.test_stitch_operation import _native_form
from tests.xdart.pages import test_rsm_page as rsm_gui, test_stitch_page as stitch_gui
from tests.xdart.pages.live_per_scan_support import start_live, tick_live, unimplemented_live, wait_until
from tests.xdart.pages.test_rsm_page import qapp  # noqa: F401
from xdart.gui.tabs.rsm import page as rsm_page
from xdart.gui.tools import stitch_owner
from xrd_tools.analysis import rsm_operation, rsm_values, stitch_native_values
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import StitchOperationExecution
from xrd_tools.analysis.stitch_sources import StitchSourceSelection
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, read_analysis_artifact

pytestmark = unimplemented_live


@pytest.fixture
def make_page(qapp):
    pages = []

    def make(kind):
        if kind == "rsm":
            page = rsm_gui._page(qapp)
            dispose = rsm_gui._close
        else:
            page = stitch_gui.StitchPage(stitch_gui.empty_host_services(stitch_gui._Status()))
            dispose = stitch_gui._dispose
        page.show()
        pages.append((page, dispose))
        return page

    yield make
    for page, dispose in reversed(pages):
        dispose(page, qapp)
    qapp.processEvents()


def _wait(qapp, page, predicate):
    try:
        wait_until(qapp, predicate, timeout_ms=60000)
    except AssertionError as error:
        raise AssertionError(page.run_bar.readinessLabel.full_text()) from error


def _click_run(qapp, page):
    _wait(qapp, page, page.run_bar.startButton.isEnabled)
    page.run_bar.startButton.click()
    _wait(qapp, page, lambda: not page.active())


def _stop_live(qapp, page):
    assert page.run_bar.stopButton.isEnabled()
    page.run_bar.stopButton.click()
    _wait(qapp, page, lambda: not page.active() and not page.run_bar.stopButton.isEnabled())
    if page.run_bar.is_live():
        page.run_bar.liveButton.click()


def _state(page):
    label = page.run_bar.readinessLabel
    return (label.full_text() + "\n" + label.toolTip() + "\n" + page.preview_text.toPlainText()).casefold()


def _ticks(page, qapp, count=2):
    for _ in range(count):
        tick_live(page, qapp)


def _write_spec(path, text):
    path.write_text(text, encoding="utf-8", newline="\n")


def _append_spec(path, text):
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _shots(folder):
    return {path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
            for path in folder.glob("*.nexus")}


def _shown(page, kind):
    if kind == "rsm":
        return None if page.shown is None else page._results.get(page._shown_path)
    return page._current_payload()


def _shown_name(page, kind):
    payload = _shown(page, kind)
    return None if payload is None else Path(payload.inspection.path).name


def _same_science(actual, expected):
    assert actual.kind is expected.kind
    assert actual.intensity.shape == expected.intensity.shape
    for (name, axis), (expected_name, expected_axis) in zip(actual.axes, expected.axes, strict=True):
        assert name == expected_name
        np.testing.assert_array_equal(axis, expected_axis)
    for field in ("intensity", "sigma", "coverage", "normalization"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    if actual.profile is not None:
        _same_science(actual.profile, expected.profile)


def _rsm_form(page, root, scans, output):
    output.mkdir(exist_ok=True)
    rsm_gui._with_source(page, root, scans=scans)
    page.set_field("output_path", str(output))
    for field, count in zip(("grid_h", "grid_k", "grid_l"), (12, 16, 20), strict=True):
        page.set_field(field, count)


def _record_runs(monkeypatch, execution):
    runs = []
    run = execution.run

    def record(self, *args, **kwargs):
        runs.append(Path(self.request.module.output.target).name)
        return run(self, *args, **kwargs)

    monkeypatch.setattr(execution, "run", record)
    return runs


def test_g1_rsm_live_seals_each_scan_and_shows_the_latest_result(qapp, make_page, tmp_path, monkeypatch):
    spec = rsm_gui._synthetic_source(tmp_path, full=True)
    second = rsm_gui._spec_scan(2, 11)
    page = make_page("rsm")
    output = tmp_path / "live"
    _rsm_form(page, tmp_path, "1,2", output)
    rsm_gui._spec_file(spec, [rsm_gui._spec_scan(1, 10)])
    page.run_bar.writeModeButton.click()  # Replace makes an unintended repeat visible.
    execution = rsm_operation.RSMOperationExecutionV2
    runs = _record_runs(monkeypatch, execution)
    emit = execution._emit
    grew_during_science = []

    def grow_later_scan_during_science(self, stage, completed):
        if stage == "science" and not grew_during_science:
            _append_spec(spec, "\n" + second[:second.rindex("\n", 0, -1) + 1])
            grew_during_science.append(True)
        return emit(self, stage, completed)

    monkeypatch.setattr(execution, "_emit", grow_later_scan_during_science)
    start_live(page)
    _ticks(page, qapp)
    first, newest = (f"source-RSM_synth-{scan}_rsm.nexus" for scan in (1, 2))
    _wait(qapp, page, lambda: _shown_name(page, "rsm") == first)
    assert grew_during_science and runs == [first]
    _ticks(page, qapp, 3)
    assert set(_shots(output)) == {first} and runs == [first]
    _append_spec(spec, second.splitlines(keepends=True)[-1])
    _ticks(page, qapp)
    _wait(qapp, page, lambda: _shown_name(page, "rsm") == newest)
    snapshot = _shots(output)
    assert set(snapshot) == {first, newest}
    _ticks(page, qapp, 3)
    assert _shots(output) == snapshot and runs == [first, newest]
    _stop_live(qapp, page)

    reference = make_page("rsm")
    expected_output = tmp_path / "finite"
    _rsm_form(reference, tmp_path, "1,2", expected_output)
    _click_run(qapp, reference)
    assert _shown_name(reference, "rsm") == newest
    for name in snapshot:
        _same_science(read_analysis_artifact(output / name), read_analysis_artifact(expected_output / name))
    assert rsm_gui._slices(page.view)[0].canvas.raw_image.size
    assert page.reopen(output / newest)
    _wait(qapp, page, lambda: not page.active())
    _same_science(_shown(page, "rsm"), _shown(reference, "rsm"))


@pytest.mark.parametrize("kind", ("rsm", "stitch"))
def test_g2_append_current_keeps_the_view_and_owns_the_skip(qapp, make_page, tmp_path, monkeypatch, kind):
    failed_path, read_attempts = None, []

    def fail_only_target(read):
        def read_artifact(path, *args, **kwargs):
            if Path(path) == failed_path:
                read_attempts.append(Path(path))
                raise OSError("saved result readback interrupted")
            return read(path, *args, **kwargs)
        return read_artifact

    if kind == "rsm":
        monkeypatch.setattr(rsm_page, "_read_rsm_record", fail_only_target(rsm_page._read_rsm_record))
    page = make_page(kind)
    output = tmp_path / "output"
    if kind == "rsm":
        rsm_gui._synthetic_source(tmp_path, full=True)
        _rsm_form(page, tmp_path, "2", output)
        _click_run(qapp, page)
        failed_path = Path(_shown(page, kind).inspection.path)
        page.set_field("scan", "1")
        execution, values, currency_name = rsm_operation.RSMOperationExecutionV2, rsm_values, "rsm_output_current"
    else:
        form = _native_form(tmp_path, "multigeometry", scans="2,3")
        for image in Path(form.image_dir).glob("*scan2_*.raw"):
            image.with_name(image.name.replace("scan2_", "scan3_")).write_bytes(image.read_bytes())
        page.load_form(form)
        _click_run(qapp, page)
        kept = output / "sample_scan2_stitch1d.nexus"
        failed_path = output / "sample_scan3_stitch1d.nexus"
        assert kept.is_file() and failed_path.is_file()
        recovered_xye, failed_xye = (path.with_name(f"iq_{path.stem}.xye") for path in (kept, failed_path))
        expected_xye = recovered_xye.read_bytes()
        recovered_xye.unlink()
        failed_xye.unlink()
        monkeypatch.setattr(stitch_owner, "read_analysis_artifact", fail_only_target(stitch_owner.read_analysis_artifact))
        page.set_field("scan", "1")
        execution, values, currency_name = StitchOperationExecution, stitch_native_values, "stitch_output_current"
    _click_run(qapp, page)
    prior = _shown(page, kind)
    assert prior is not None and Path(prior.inspection.path) != failed_path
    snapshot = _shots(output)
    page.set_field("scan", "2" if kind == "rsm" else "2,3")
    runs = _record_runs(monkeypatch, execution)
    current = getattr(values, currency_name)
    checked = []

    def record_currency(request):
        checked.append(Path(request.module.output.target))
        return current(request)

    monkeypatch.setattr(values, currency_name, record_currency)
    assert page.run_bar.write_mode() == "Append"
    start_live(page)
    _ticks(page, qapp)
    if kind == "stitch":
        _wait(qapp, page, lambda: recovered_xye.is_file())
        assert recovered_xye.read_bytes() == expected_xye
        # The first detached readback's retirement may drive the next eligible group itself.
        _ticks(page, qapp)
        _wait(qapp, page, lambda: "saved result readback interrupted" in _state(page))
        assert read_attempts == [failed_path] and not failed_xye.exists()
    else:
        _wait(qapp, page, lambda: "skip" in _state(page))
    assert failed_path in checked and not runs
    checks_at_skip = checked.copy()
    _ticks(page, qapp, 4)
    assert checked == checks_at_skip and len(read_attempts) <= 1 and not runs
    assert _shown(page, kind) is prior and _shots(output) == snapshot
    _stop_live(qapp, page)


def _stitch_nxwriter(form):
    path = _write_bluesky_nxwriter(Path(form.project_root) / "recording.nxs", n=2)
    frames = np.stack([np.fromfile(raw, dtype="int32").reshape(form.detector_shape)
                       for raw in sorted(Path(form.image_dir).glob("*scan1_*.raw"))]).astype(np.uint32)
    with h5py.File(path, "r+") as handle:
        entry = handle["entry"]
        del entry["end_time"]
        config = entry["instrument/bluesky/metadata/configuration/eiger/data"]
        config["eiger_cam_wavelength"][()] = 1.0
        config["eiger_cam_photon_energy"][()] = 12398.41984
        data = entry["data"]
        detector = entry["instrument/detectors/eiger"]
        del data["eiger_image"]
        del detector["data"]
        images = data.create_dataset("eiger_image", data=frames)
        images.attrs["signal_type"] = "detector"
        detector["data"] = images
        for name, values in (("del", (14., 15.)), ("nu", (-10., -10.))):
            positioner = entry["instrument/positioners"].create_group(name)
            positioner.attrs["NX_class"] = "NXpositioner"
            positioner.create_dataset("value", data=values)
            data.create_dataset(name, data=values)
    return path


@pytest.mark.parametrize("mode", ("1d", "2d"), ids=("g6-stitch-1d-nxwriter", "g7-stitch-2d-spec"))
def test_g6_g7_stitch_live_matches_finite_science_and_readback(qapp, make_page, tmp_path, monkeypatch, mode):
    form = _native_form(tmp_path, "multigeometry", scans="2")
    if mode == "1d":
        container = _stitch_nxwriter(form)
        form = replace(form, source_selection=StitchSourceSelection("NeXus", (str(container),)))
        name = "recording_stitch1d.nexus"
    else:
        form = replace(form, mode="2d", npt_rad_2d=32, npt_azim_2d=24)
        spec = Path(form.spec_path)
        text = spec.read_text(encoding="utf-8")
        start, end = text.index("#S 2 "), text.index("#S 3 ")
        late_scan = text[start:end]
        _write_spec(spec, text[:start])
        name = "sample_scan2_stitch2d.nexus"
    page = make_page("stitch")
    page.load_form(form)
    output = Path(form.output_path).parent
    page.run_bar.writeModeButton.click()
    page.run_bar.coresSpin.setValue(2)
    cores = page.run_bar.get_cores()
    page.set_field("record_angular_support", False)
    assert page.run_bar.current_mode() == ("Stitch 1D" if mode == "1d" else "Stitch 2D")
    calls = []
    run = StitchOperationExecution.run

    def record(self, *args, **kwargs):
        calls.append((self.request.plan.mode, kwargs["workers"], kwargs["angular_support"].enabled))
        return run(self, *args, **kwargs)

    monkeypatch.setattr(StitchOperationExecution, "run", record)
    start_live(page)
    _ticks(page, qapp, 3)
    assert page.run_bar.stopButton.isEnabled() and not _shots(output) and not calls
    if mode == "1d":
        with h5py.File(container, "r+") as handle:
            handle["entry"].create_dataset("end_time", data=b"2026-07-10T00:01:00")
    else:
        assert not page.run_bar.coresSpin.isEnabled()
        # Qt permits a programmatic change even when a control is locked to the operator.
        page.run_bar.coresSpin.setValue(1)
        page.set_field("record_angular_support", True)
        _append_spec(spec, late_scan)
    _ticks(page, qapp)
    _wait(qapp, page, lambda: _shown_name(page, "stitch") == name)
    assert calls == [(mode, cores, False)]
    snapshot = _shots(output)
    assert set(snapshot) == {name}
    _ticks(page, qapp, 3)
    assert _shots(output) == snapshot and calls == [(mode, cores, False)]
    _stop_live(qapp, page)
    reference = make_page("stitch")
    expected_output = tmp_path / "finite"
    expected_output.mkdir()
    reference.load_form(replace(form, output_path=expected_output / Path(form.output_path).name))
    reference.run_bar.coresSpin.setValue(cores)
    reference.set_field("record_angular_support", False)
    _click_run(qapp, reference)
    actual = read_analysis_artifact(output / name)
    assert actual.kind is (AnalysisArtifactKind.STITCH_1D if mode == "1d" else AnalysisArtifactKind.STITCH_2D)
    _same_science(actual, read_analysis_artifact(expected_output / name))
    assert page.frames.count() and page.curve.listDataItems()
    if mode == "2d":
        assert actual.profile is not None
        assert sorted(actual.intensity.shape) == [24, 32]
    page.open_result(output / name)
    _wait(qapp, page, lambda: not page.active())
    _same_science(_shown(page, "stitch"), actual)


@pytest.mark.parametrize("scans", ("[2-3]", "2,[2-3]"))
def test_g8_stitch_live_waits_for_all_group_members(qapp, make_page, tmp_path, monkeypatch, scans):
    form = replace(_native_form(tmp_path, "multigeometry", scans=scans), output_family=None)
    spec = Path(form.spec_path)
    text = spec.read_text(encoding="utf-8")
    third = text.index("#S 3 ")
    _write_spec(spec, text[:third])
    page = make_page("stitch")
    page.load_form(form)
    page.run_bar.writeModeButton.click()
    output = Path(form.output_path).parent
    runs = _record_runs(monkeypatch, StitchOperationExecution)
    start_live(page)
    _ticks(page, qapp)
    singleton, grouped = "sample_scan2_stitch1d.nexus", "sample_scan2-3_stitch1d.nexus"
    early = [singleton] if scans.startswith("2,") else []
    if early:
        _wait(qapp, page, lambda: _shown_name(page, "stitch") == singleton)
    assert runs == early and list(_shots(output)) == early
    _append_spec(spec, text[third:])
    _ticks(page, qapp)
    assert runs == early and list(_shots(output)) == early  # Rows alone are insufficient.
    for image in Path(form.image_dir).glob("*scan2_*.raw"):
        image.with_name(image.name.replace("scan2_", "scan3_")).write_bytes(image.read_bytes())
    _ticks(page, qapp)
    _wait(qapp, page, lambda: _shown_name(page, "stitch") == grouped)
    assert runs == [*early, grouped]
    assert len(read_analysis_artifact(output / grouped).contributing_frames) == 4
    if early:
        assert len(read_analysis_artifact(output / singleton).contributing_frames) == 2
    snapshot = _shots(output)
    _ticks(page, qapp, 3)
    assert runs == [*early, grouped] and _shots(output) == snapshot
    _stop_live(qapp, page)
    reference = make_page("stitch")
    reference.load_form(form)
    reference.set_field("output_path", str(tmp_path / "finite"))
    _click_run(qapp, reference)
    for name in snapshot:
        _same_science(read_analysis_artifact(output / name), read_analysis_artifact(tmp_path / "finite" / name))


def _spec_ub(page, spec):
    page.set_field("ub_from", "A SPEC scan's")
    page.set_field("ub_spec_path", str(spec))
    page.set_field("ub_scan", "1")


def test_g11_rsm_live_rejects_a_rewritten_selected_ub_before_science(qapp, make_page, tmp_path, monkeypatch):
    spec = rsm_gui._synthetic_source(tmp_path, full=True)
    recording = rsm_gui._spec_scan(2, 11).rsplit("\n", 2)[0] + "\n"
    rsm_gui._spec_file(spec, [rsm_gui._spec_scan(1, 10), recording])
    output = tmp_path / "live"
    page = make_page("rsm")
    _rsm_form(page, tmp_path, "1", output)
    _spec_ub(page, spec)
    terminals = []
    execution = rsm_operation.RSMOperationExecutionV2
    run = execution.run

    def drift_at_native_run(self, *args, **kwargs):
        if not terminals:
            text = spec.read_text(encoding="utf-8")
            _write_spec(spec, text.replace("#G3 1 0 0 0 1 0 0 0 1", "#G3 2 0 0 0 2 0 0 0 2", 1))
        result = run(self, *args, **kwargs)
        terminals.append(result.terminal)
        return result

    monkeypatch.setattr(execution, "run", drift_at_native_run)
    start_live(page)
    _ticks(page, qapp)
    _wait(qapp, page, lambda: bool(terminals))
    assert terminals[0].disposition is ModuleDisposition.REFUSED
    assert not _shots(output) and _shown(page, "rsm") is None
    _ticks(page, qapp, 3)
    _wait(qapp, page, lambda: _shown(page, "rsm") is not None)
    assert [terminal.disposition for terminal in terminals] == [ModuleDisposition.REFUSED, ModuleDisposition.COMMITTED]
    payload = _shown(page, "rsm")
    assert payload.axes[0][0] == "h"
    ub = json.loads(payload.provenance_json)["plan"]["ub_source"]
    assert ub["kind"] == "spec" and ub["matrix"] == [2, 0, 0, 0, 2, 0, 0, 0, 2]
    _stop_live(qapp, page)


def test_g11_rsm_live_regrids_a_legacy_spec_ub_map_once(qapp, make_page, tmp_path, monkeypatch):
    spec = rsm_gui._synthetic_source(tmp_path, full=True)
    output = tmp_path / "maps"
    page = make_page("rsm")
    _rsm_form(page, tmp_path, "1", output)
    _spec_ub(page, spec)
    bind_ub = rsm_operation._bind_rsm_ub

    def bind_legacy_ub(*args, **kwargs):
        # Construct a valid saved result with 1.2.18's whole-file UB identity.
        return replace(bind_ub(*args, **kwargs), spec_occurrence=None)

    with monkeypatch.context() as legacy:
        legacy.setattr(rsm_operation, "_bind_rsm_ub", bind_legacy_ub)
        _click_run(qapp, page)
    prior = _shown(page, "rsm")
    assert prior is not None
    snapshot = _shots(output)
    _click_run(qapp, page)
    assert "up to date" in _state(page) and _shots(output) == snapshot
    _append_spec(spec, "\n" + rsm_gui._spec_scan(3, 12).rsplit("\n", 2)[0] + "\n")
    runs = _record_runs(monkeypatch, rsm_operation.RSMOperationExecutionV2)
    start_live(page)
    _ticks(page, qapp)
    _wait(qapp, page, lambda: _shown(page, "rsm") is not None and _shots(output) != snapshot)
    target = Path(prior.inspection.path)
    assert runs == [target.name]
    regridded = _shots(output)
    _ticks(page, qapp, 3)
    assert runs == [target.name] and _shots(output) == regridded
    _stop_live(qapp, page)
    _click_run(qapp, page)
    assert "up to date" in _state(page) and _shots(output) == regridded
    _same_science(read_analysis_artifact(target), prior)


@pytest.mark.parametrize("selection_kind", ("all-scans", "directory", "multi-spec", "static-single"))
def test_g12_stitch_live_custom_names_require_one_static_group(qapp, make_page, tmp_path, selection_kind):
    form = replace(_native_form(tmp_path, "multigeometry", scans="1"), output_family="custom")
    path = Path(form.spec_path)
    selection = form.source_selection
    if selection_kind == "directory":
        selection = replace(selection, kind="Directory", paths=(str(path.parent),))
    elif selection_kind == "all-scans":
        selection = replace(selection, stitch_all=True)
    elif selection_kind == "multi-spec":
        other = path.with_name("second")
        other.write_bytes(path.read_bytes())
        selection = replace(selection, paths=(str(path), str(other)))
    page = make_page("stitch")
    page.load_form(replace(form, source_selection=selection))
    start_live(page)
    output = Path(form.output_path).parent
    if selection_kind == "static-single":
        _ticks(page, qapp)
        _wait(qapp, page, lambda: _shown_name(page, "stitch") == "custom_stitch1d.nexus")
        assert set(_shots(output)) == {"custom_stitch1d.nexus"}
        _stop_live(qapp, page)
    else:
        _wait(qapp, page, lambda: "result name" in _state(page) and not page.run_bar.stopButton.isEnabled())
        assert not page.active() and not _shots(output)


@pytest.mark.parametrize("kind", ("rsm", "stitch"))
@pytest.mark.parametrize("claim", ("append-current", "replace", "hardlink"))
def test_g12_live_keeps_staggered_native_output_claims(qapp, make_page, tmp_path, monkeypatch, kind, claim):
    page = make_page(kind)
    output = tmp_path / "output"
    if kind == "rsm":
        source = rsm_gui._synthetic_source(tmp_path, full=True)
        _rsm_form(page, tmp_path, "1", output)
        # Flattening relative path components is native RSM naming. These two distinct files can name one slot.
        first, second = tmp_path / "run-a" / "scan", tmp_path / "run" / "a-scan"
        if claim == "hardlink":
            second = second.with_name("different")
        images = source.parent / "images"
        old_stem = source.stem
        first_name, second_name = (f"{path.parent.name}-{path.name}-1_rsm.nexus" for path in (first, second))
        execution = rsm_operation.RSMOperationExecutionV2
    else:
        form = _native_form(tmp_path, "multigeometry", scans="1")
        source = Path(form.spec_path)
        page.load_form(form)
        images = Path(form.image_dir)
        old_stem = source.stem
        # Stitch names from the raw files, without the enclosing directory.
        first, second = tmp_path / "run-a" / "sample", tmp_path / "run" / "sample"
        if claim == "hardlink":
            second = second.with_name("different")
        first_name, second_name = (f"{path.name}_scan1_stitch1d.nexus" for path in (first, second))
        execution = StitchOperationExecution
    text = source.read_text(encoding="utf-8")
    header, scan = text.split("#S 1 ", 1)
    for path in (first, second):
        path.parent.mkdir(exist_ok=True)
    _write_spec(first, header + "#S 1 " + scan)
    _write_spec(second, header)  # Selected file exists, but its requested occurrence hasn't been recorded yet.
    for raw in images.glob(f"{old_stem}_scan1_*"):
        for path in (first, second):
            destination = path.parent / raw.name.replace(old_stem, path.stem, 1)
            destination.write_bytes(raw.read_bytes())
    # The native blank Images folder selects each SPEC's own directory. Sharing a directory here
    # would also match the suffix stem of a-scan and invalidate the intended output-only collision.
    page.set_field("image_dir", "")
    page.set_sources((str(first),))
    target = output / first_name
    if claim == "append-current":
        if kind == "rsm":
            page.resolve_source()
            request = rsm_values.prepare_rsm_tool_v2(page.build_form()).request
            assert Path(request.module.output.target) == target
            result = rsm_operation.run_rsm_operation_v2(request)
            assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
        else:
            _click_run(qapp, page)
        assert target.is_file(), page.run_bar.readinessLabel.full_text()
    else:
        page.run_bar.writeModeButton.click()
    page.set_sources((str(first), str(second)))
    runs = _record_runs(monkeypatch, execution)
    start_live(page)
    _ticks(page, qapp)
    if claim == "append-current":
        _wait(qapp, page, lambda: "skip" in _state(page))
        expected_runs = []
    else:
        _wait(qapp, page, lambda: _shown_name(page, kind) == first_name)
        expected_runs = [first_name]
    assert target.is_file() and runs == expected_runs
    if claim == "hardlink":
        alias = output / second_name
        try:
            os.link(target, alias)
        except (NotImplementedError, OSError) as error:
            pytest.skip(f"hardlink unavailable for native alias test: {error}")
        assert os.path.samefile(target, alias) and target != alias
    else:
        assert first_name == second_name
        replacement = target.with_name("replacement.nexus")
        replacement.write_bytes(target.read_bytes())
        replacement.replace(target)
    snapshot = _shots(output)
    _append_spec(second, "#S 1 " + scan)
    _ticks(page, qapp, 3)
    _wait(qapp, page, lambda: second_name.casefold() in _state(page)
          and any(word in _state(page) for word in ("collision", "already claimed", "same output")))
    assert runs == expected_runs and _shots(output) == snapshot
    _stop_live(qapp, page)
