"""P7–P12: mounted preview priority, result adoption and retirement."""

from dataclasses import replace
from pathlib import Path
from threading import Event, get_ident

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.core.test_container_descriptor import _vendor_eiger_master
from tests.core.test_live_per_scan import _native_nexus_form
from tests.core.test_stitch_operation import _native_form
from tests.xdart.pages import test_rsm_page as rsm_gui
from tests.xdart.pages.live_per_scan_support import start_live, tick_live, wait_until
from tests.xdart.pages.live_preview_support import (
    advance_preview, disable_preview, preview_checkbox, start_preview, unimplemented_preview,
)
from tests.xdart.pages.test_live_per_scan_native import (
    _append_spec, _click_run, _record_runs, _rsm_form, _same_science, _shots, _shown_name,
    _stop_live, _wait, _write_spec, make_page, qapp,  # noqa: F401
)
from tests.xdart.pages.test_live_preview_science import (
    _next, _prefix, _raw, _rsm_acquisition, _scan, _stitch_raw, _stitch_scan,
)
from tests.xdart.pages.test_rsm_eiger_source import _on_source
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.analysis import rsm_operation, stitch_operation
from xrd_tools.analysis.stitch_sources import list_planned_stitch_groups
from xrd_tools.core.geometry.xu_runtime import XuRuntimeSession
from xrd_tools.io import analysis_artifact, nexus
from xrd_tools.rsm.gridding import StreamingGridder

pytestmark = unimplemented_preview


def _view(page, kind):
    view = page.view if kind == "rsm" else page.scientific
    assert isinstance(view, ScientificView)
    return view


def _display_identity(view):
    keyed = view.keyed_projection
    return keyed.title if keyed is not None else view.navigation_current_key


@pytest.mark.parametrize("backend", ("xu_hist", "multigeometry"))
def test_p7_foreground_other_page_has_priority_between_preview_slices(
    qapp, make_page, tmp_path, monkeypatch, backend,
):
    rsm, stitch = make_page("rsm"), make_page("stitch")
    preview_root, finite_root = tmp_path / "preview", tmp_path / "finite"
    preview_root.mkdir()
    finite_root.mkdir()
    spec, images = _rsm_acquisition(rsm, preview_root)
    stitch.load_form(_native_form(finite_root, backend, scans="1"))
    entered, release = Event(), Event()
    events, preview_thread = [], []
    add, exit_session = StreamingGridder.add_leased, XuRuntimeSession.__exit__
    execute = stitch_operation.StitchOperationExecution._emit

    def hold_grid(gridder, *args, **kwargs):
        if not entered.is_set():
            preview_thread.append(get_ident())
            events.append("preview")
            entered.set()
            assert release.wait(20), "the preview XU slice was never released"
        else:
            events.append("preview-resumed")
        return add(gridder, *args, **kwargs)

    def exit_xu(session, *args):
        result = exit_session(session, *args)
        if preview_thread and get_ident() == preview_thread[0] and "preview-yielded" not in events:
            events.append("preview-yielded")
        return result

    def science(execution, stage, *args):
        if stage == "science" and "finite-science" not in events:
            events.append("finite-science")
        return execute(execution, stage, *args)

    monkeypatch.setattr(StreamingGridder, "add_leased", hold_grid)
    monkeypatch.setattr(XuRuntimeSession, "__exit__", exit_xu)
    monkeypatch.setattr(stitch_operation.StitchOperationExecution, "_emit", science)
    try:
        _raw(images / f"{spec.stem}_scan1_0000.raw", 51)
        owner = start_preview(rsm, qapp)
        advance_preview(rsm, owner, qapp, drain=False)
        wait_until(qapp, entered.is_set)
        assert stitch.run_bar.startButton.isEnabled()
        stitch.run_bar.startButton.click()
        assert stitch.active() and rsm.run_bar.stopButton.isEnabled()
        _append_spec(spec, "0.01 1 10 101\n")
        _raw(images / f"{spec.stem}_scan1_0001.raw", 52)
        advance_preview(rsm, owner, qapp, drain=False)
        assert events.count("preview-resumed") == 0
    finally:
        release.set()

    _wait(qapp, stitch, lambda: not stitch.active() and bool(_shots(finite_root / "output")))
    _next(rsm, owner, qapp, lambda: _prefix(owner, ("1.1", 0), ("1.1", 1)))
    assert events.index("finite-science") < events.index("preview-resumed")
    assert events.index("preview-yielded") < events.index("preview-resumed")
    if backend == "xu_hist":
        assert events.index("preview-yielded") < events.index("finite-science")


def _ranked_acquisition(page, kind, root, *, selectors="2,3", planned_last=5):
    """Save scan 1 via Run. Incomplete 2 and 3 will use the same settings and scope."""
    if kind == "rsm":
        spec = rsm_gui._synthetic_source(root, full=True)
        output = root / "live"
        _rsm_form(page, root, "1", output)
        raw_folder = spec.parent / "images"

        def scan(number, values, planned):
            return _scan(number, values, planned=planned)

        def put(member, index, value):
            file = raw_folder / f"{spec.stem}_scan{member}_{index:04d}.tif"
            file.unlink(missing_ok=True)
            return _raw(file.with_suffix(".raw"), value)

        positions = (0., 0.01, 0.02, 0.03, 0.04)
        row4 = "0.04 1 10 101\n"
    else:
        form = _native_form(root, "multigeometry", scans="1")
        spec, raw_folder = Path(form.spec_path), Path(form.image_dir)
        output = Path(form.output_path).parent
        page.load_form(form)

        def scan(number, values, planned):
            return _stitch_scan(spec, number, values, planned=planned)

        def put(member, index, value):
            return _stitch_raw(form, member, index, value)

        positions = (14., 14.01, 14.02, 14.03, 14.04)
        row4 = "14.04 -10\n"

    text = spec.read_text(encoding="utf-8")
    _write_spec(spec, text[:text.index("#S 2")])
    for item in raw_folder.glob("*scan2_*"):
        item.unlink()

    def prepare_later():
        _append_spec(spec, "\n" + scan(2, positions[:4], 4) + "\n" + scan(3, positions[:4], planned_last))
        page.set_field("scan", selectors)

    return spec, output, put, row4, prepare_later


@pytest.mark.parametrize("kind", ("rsm", "stitch"))
def test_p8_newer_preview_keeps_actual_view_during_older_adoption_and_refresh(
    qapp, make_page, tmp_path, kind,
):
    page = make_page(kind)
    spec, output, put, last_row, prepare_later = _ranked_acquisition(page, kind, tmp_path)
    _click_run(qapp, page)
    prior = _shown_name(page, kind)
    view = _view(page, kind)
    saved_key = _display_identity(view)
    prepare_later()
    owner = start_preview(page, qapp)
    advance_preview(page, owner, qapp)
    assert owner.snapshot is None and _display_identity(view) == saved_key
    for index in range(4):
        put(3, index, 23 + index)
    _next(page, owner, qapp, lambda: owner.snapshot is not None)
    newest = owner.snapshot
    preview_key = _display_identity(view)
    assert preview_key is not None and preview_key != saved_key
    assert {key for key, _ in newest.points} == {"3.1"}
    for index in range(4):
        put(2, index, 41 + index)
    for _ in range(4):
        advance_preview(page, owner, qapp)
    _wait(qapp, page, lambda: len(_shots(output)) == 2 and not page._owner.busy)
    older = next(path for path in output.glob("*.nexus") if path.name != prior)
    assert str(older) in page._listed
    assert _display_identity(view) == preview_key
    assert owner.snapshot.points == newest.points
    page._result_browser.refresh()
    page.browser.commandRequested.emit(ShellCommand(ShellCommandKind.SELECT_RESULT, str(older)))
    _next(page, owner, qapp, lambda: not page._result_browser.busy)
    assert _display_identity(view) == preview_key
    assert str(older) in page._listed
    first_new_raw = next((spec.parent / "images").glob("*scan3_0000.raw"))
    original = first_new_raw.read_bytes()
    first_new_raw.write_bytes(original[:-1])
    _next(page, owner, qapp, lambda: owner.snapshot is None)
    assert _shown_name(page, kind) == older.name
    first_new_raw.write_bytes(original)
    _next(page, owner, qapp, lambda: owner.snapshot is not None)
    _append_spec(spec, last_row)
    put(3, 4, 29)
    for _ in range(4):
        advance_preview(page, owner, qapp)
    _wait(qapp, page, lambda: len(_shots(output)) == 3 and not page._owner.busy)
    assert owner.snapshot is None and _display_identity(view) != preview_key
    assert "3" in _shown_name(page, kind)
    for _ in range(3):
        advance_preview(page, owner, qapp)
    assert owner.snapshot is None


def test_p8_repeated_scan_number_uses_physical_order_not_arrival(qapp, make_page, tmp_path):
    page = make_page("rsm")
    spec, images = _rsm_acquisition(page, tmp_path)
    rsm_gui._spec_file(spec, [_scan(1, (0.,)), _scan(1, (0.1,))])
    page.set_field("image_stem", "{spec}_scan{occurrence}_")
    page.set_field("scan", "1.1,1.2")
    late = images / f"{spec.stem}_scan1.2_0000.raw"
    _raw(late, 42)
    owner = start_preview(page, qapp)
    _next(page, owner, qapp, lambda: _prefix(owner, ("1.2", 0)))
    key = _display_identity(page.view)
    _raw(images / f"{spec.stem}_scan1.1_0000.raw", 25)
    for _ in range(4):
        advance_preview(page, owner, qapp)
    assert _prefix(owner, ("1.2", 0)) and _display_identity(page.view) == key


def test_p8_simultaneous_directory_sources_follow_one_native_group_order(qapp, make_page, tmp_path):
    page = make_page("rsm")
    spec, images = _rsm_acquisition(page, tmp_path)
    other = spec.with_name("A_second")
    rsm_gui._spec_file(other, [_scan(1, (0.1,))])
    page.set_field("source_kind", "Directory")
    page.set_sources([str(spec.parent)])
    for path, value in ((spec, 41), (other, 70)):
        _raw(images / f"{path.stem}_scan1_0000.raw", value)
    selection, _, _ = page._resolution_request()
    groups = tuple(list_planned_stitch_groups(selection))
    assert len(groups) == 2
    owner = start_preview(page, qapp)
    _next(page, owner, qapp, lambda: owner.snapshot is not None)
    assert owner.snapshot.group_identity == groups[0].identity
    chosen = _display_identity(page.view)
    for _ in range(3):
        advance_preview(page, owner, qapp)
        assert _display_identity(page.view) == chosen


@pytest.mark.parametrize("lane", ("spec-eiger", "nxwriter"))
def test_p9_container_pixels_are_left_to_completed_native_runs(qapp, make_page, tmp_path, lane, monkeypatch):
    if lane == "spec-eiger":
        page = make_page("rsm")
        spec, images = _rsm_acquisition(page, tmp_path, planned=2)
        output = tmp_path / "live"
        _on_source(page, qapp, tmp_path, "1")
        page.set_field("output_path", str(output))
        for name in ("grid_h", "grid_k", "grid_l"):
            page.set_field(name, 12)

        def finish():
            _append_spec(spec, "1 1 10 101\n")
            shape = (195, 487)
            _vendor_eiger_master(images / f"{spec.stem}_scan001_master.h5",
                                 [images / f"{spec.stem}_scan001_data_000001.h5"],
                                 per_seg=(2, *shape), chunks=(1, *shape))
    else:
        page = make_page("stitch")
        form, source = _native_nexus_form(tmp_path)
        output = Path(form.output_path).parent
        with h5py.File(source, "r+") as handle:
            del handle["sibling/end_time"]
        page.load_form(form)

        def finish():
            with h5py.File(source, "r+") as handle:
                handle["sibling"].create_dataset("end_time", data=b"2026-07-10T00:01:00")

    decode = nexus.NexusImageStack.__getitem__
    decoded = []

    def record(stack, index):
        decoded.append(index)
        return decode(stack, index)

    monkeypatch.setattr(nexus.NexusImageStack, "__getitem__", record)
    owner = start_preview(page, qapp)
    for _ in range(3):
        advance_preview(page, owner, qapp)
    assert not decoded and not _shots(output)
    assert owner.snapshot is None and owner.progress.decodes == 0
    finish()
    for _ in range(4):
        advance_preview(page, owner, qapp)
    kind = "rsm" if lane == "spec-eiger" else "stitch"
    _wait(qapp, page, lambda: _shown_name(page, kind) is not None)
    assert len(_shots(output)) == 1
    assert owner.snapshot is None and owner.progress.decodes == 0
    [saved] = output.glob("*.nexus")
    assert np.isfinite(analysis_artifact.read_analysis_artifact(saved).intensity).any()


@pytest.mark.parametrize("kind", ("rsm", "stitch"))
def test_p10_append_current_keeps_newer_preview_and_backfills_companion(
    qapp, make_page, tmp_path, monkeypatch, kind,
):
    page = make_page(kind)
    spec, output, put, _last_row, prepare_later = _ranked_acquisition(page, kind, tmp_path)
    _click_run(qapp, page)
    prepare_later()
    for index in range(4):
        put(2, index, 21 + index)
    page.set_field("scan", "2")
    _click_run(qapp, page)
    saved_path = output / _shown_name(page, kind)
    before = _shots(output)
    if kind == "stitch":
        companion = saved_path.with_name(f"iq_{saved_path.stem}.xye")
        expected_companion = companion.read_bytes()
        companion.unlink()
    page.set_field("scan", "2,3")
    for index in range(4):
        put(3, index, 31 + index)
    execution = (rsm_operation.RSMOperationExecutionV2 if kind == "rsm"
                 else stitch_operation.StitchOperationExecution)
    calls = _record_runs(monkeypatch, execution)
    assert page.run_bar.write_mode() == "Append"
    owner = start_preview(page, qapp)
    _next(page, owner, qapp, lambda: owner.snapshot is not None)
    shown = _display_identity(_view(page, kind))
    assert {key for key, _ in owner.snapshot.points} == {"3.1"}
    for _ in range(4):
        advance_preview(page, owner, qapp)
    assert any("2.1" in group.members for group in owner.retired_groups)
    assert _display_identity(_view(page, kind)) == shown and not calls
    assert _shots(output) == before
    if kind == "stitch":
        assert companion.read_bytes() == expected_companion


def test_p10_append_current_retires_own_overlay_without_opening_skipped_result(
    qapp, make_page, tmp_path, monkeypatch,
):
    page = make_page("rsm")
    _spec, output, put, _last_row, prepare_later = _ranked_acquisition(page, "rsm", tmp_path, selectors="2")
    _click_run(qapp, page)
    prior = output / _shown_name(page, "rsm")
    prepare_later()
    for index in range(4):
        put(2, index, 21 + index)
    _click_run(qapp, page)
    skipped = _shown_name(page, "rsm")
    assert prior.name != skipped and page.reopen(prior)
    _wait(qapp, page, lambda: _shown_name(page, "rsm") == prior.name and not page.active())
    runs = _record_runs(monkeypatch, rsm_operation.RSMOperationExecutionV2)
    before = _shots(output)
    owner = start_preview(page, qapp)
    # Initial observation can preview; do not advance the 2s finite stillness
    # clock until the coherent first projection has mounted.
    owner.poll_due()
    wait_until(qapp, lambda: owner.snapshot is not None)
    _next(page, owner, qapp, lambda: bool(owner.retired_groups))
    assert owner.snapshot is None and _shown_name(page, "rsm") == prior.name
    assert page.browser.current_result() != str(output / skipped)
    assert not runs and _shots(output) == before


@pytest.mark.parametrize("kind", ("rsm", "stitch"))
def test_p10_verification_pending_keeps_its_native_handoff_and_equivalent_saved_science(
    qapp, make_page, tmp_path, monkeypatch, kind,
):
    page = make_page(kind)
    spec, output, put, last_row, prepare_later = _ranked_acquisition(page, kind, tmp_path, selectors="3")
    _click_run(qapp, page)
    prior = _shown_name(page, kind)
    prepare_later()
    for index in range(4):
        put(3, index, 31 + index)
    native = rsm_operation if kind == "rsm" else stitch_operation
    read = native.read_analysis_artifact
    refusals = []

    def fail_once(*args, **kwargs):
        receipt = kwargs.get("expected_receipt")
        if receipt is not None and not refusals:
            refusals.append(receipt)
            raise OSError("the published artifact readback is temporarily unavailable")
        return read(*args, **kwargs)

    monkeypatch.setattr(native, "read_analysis_artifact", fail_once)
    owner = start_preview(page, qapp)
    _next(page, owner, qapp, lambda: owner.snapshot is not None)
    _append_spec(spec, last_row)
    put(3, 4, 39)
    for _ in range(4):
        advance_preview(page, owner, qapp)
    _wait(qapp, page, lambda: page.retry_button.isEnabled() and not page.retry_button.isHidden())
    assert len(refusals) == 1 and len(_shots(output)) == 2
    handoff = page._live_watcher.handoff
    assert handoff is not None and _shown_name(page, kind) == prior
    before = _shots(output)
    page.retry_button.click()
    _wait(qapp, page, lambda: page._live_watcher.handoff is None and not page._owner.busy)
    assert _shots(output) == before and owner.snapshot is None
    shown = _shown_name(page, kind)
    assert shown is not None and "3" in shown
    actual = analysis_artifact.read_analysis_artifact(output / shown)
    _stop_live(qapp, page)
    other = tmp_path / "ordinary"
    other.mkdir()
    page.set_field("output_path", str(other))
    _click_run(qapp, page)
    _same_science(actual, analysis_artifact.read_analysis_artifact(other / shown))
    page.set_field("output_path", str(output))
    execution = rsm_operation.RSMOperationExecutionV2 if kind == "rsm" else stitch_operation.StitchOperationExecution
    runs = _record_runs(monkeypatch, execution)
    _click_run(qapp, page)
    assert not runs and _shots(output) == before


@pytest.mark.parametrize("kind", ("rsm", "stitch"))
@pytest.mark.parametrize("command", ("stop", "close"))
def test_p11_stop_and_close_join_preview_and_revoke_queued_results(
    qapp, make_page, tmp_path, monkeypatch, kind, command,
):
    from xrd_tools.io import image

    page = make_page(kind)
    spec, output, put, last_row, prepare_later = _ranked_acquisition(
        page, kind, tmp_path, selectors="3", planned_last=6,
    )
    _click_run(qapp, page)
    saved = _shots(output)
    prepare_later()
    for index in range(4):
        put(3, index, 37 + index)
    owner = start_preview(page, qapp)
    _next(page, owner, qapp, lambda: owner.snapshot is not None)
    entered, release = Event(), Event()
    decode = image._read_raw_binary

    def hold(path, *args, **kwargs):
        if "_scan3_0004" in Path(path).stem and not entered.is_set():
            entered.set()
            assert release.wait(20), "the test never released the detector read"
        return decode(path, *args, **kwargs)

    monkeypatch.setattr(image, "_read_raw_binary", hold)
    close = page.close_page if kind == "rsm" else page.shutdown
    try:
        # Add a point while remaining below the planned end count.
        _append_spec(spec, last_row)
        put(3, 4, 44)
        advance_preview(page, owner, qapp, drain=False)
        wait_until(qapp, entered.is_set)
        if command == "close":
            assert close().status is PageCleanup.PENDING
            assert close().status is PageCleanup.PENDING
        else:
            page.run_bar.stopButton.click()
            assert page.active() and not page.preview_button.isEnabled()
        assert owner.snapshot is None
    finally:
        release.set()
    if command == "close":
        wait_until(qapp, lambda: close().status is PageCleanup.CLEAN)
    else:
        wait_until(qapp, lambda: not page.active())
    assert not owner.working and owner.progress.pending_watermark is None
    assert owner.snapshot is None and _shots(output) == saved
    if command == "stop":
        images = Path(spec).parent / "images"
        saved_inputs = {path: path.read_bytes() for path in images.glob("*scan3_*")}
        assert len(saved_inputs) == 5
        for path in saved_inputs:
            path.unlink()
        restarted = start_preview(page, qapp)
        assert restarted.snapshot is None
        _stop_live(qapp, page)
        for path, body in saved_inputs.items():
            path.write_bytes(body)
        page.begin_preview()
        _wait(qapp, page, lambda: not page.active())
        assert owner.snapshot is None and _shots(output) == saved


@pytest.mark.parametrize("kind", ("rsm", "stitch"))
def test_p11_stop_drains_a_committed_native_receipt_during_preview_live(
    qapp, make_page, tmp_path, monkeypatch, kind,
):
    page = make_page(kind)
    spec, output, put, last_row, prepare_later = _ranked_acquisition(page, kind, tmp_path, selectors="3")
    _click_run(qapp, page)
    prepare_later()
    for index in range(4):
        put(3, index, 35 + index)
    native = rsm_operation if kind == "rsm" else stitch_operation
    read = native.read_analysis_artifact
    pending = []

    def block_once(*args, **kwargs):
        receipt = kwargs.get("expected_receipt")
        if receipt is not None and not pending:
            pending.append(receipt)
            raise OSError("strict native readback temporarily refused")
        return read(*args, **kwargs)

    monkeypatch.setattr(native, "read_analysis_artifact", block_once)
    owner = start_preview(page, qapp)
    _next(page, owner, qapp, lambda: owner.snapshot is not None)
    _append_spec(spec, last_row)
    put(3, 4, 50)
    for _ in range(4):
        advance_preview(page, owner, qapp)
    _wait(qapp, page, lambda: page.retry_button.isEnabled() and not page.retry_button.isHidden())
    committed = _shots(output)
    assert len(committed) == 2 and len(pending) == 1
    page.run_bar.stopButton.click()
    assert page.active() and owner.snapshot is None
    assert page.retry_button.isEnabled()
    page.retry_button.click()
    _wait(qapp, page, lambda: not page.active())
    assert owner.snapshot is None and _shots(output) == committed
    assert page._live_watcher.handoff is None


@pytest.mark.parametrize("kind", ("rsm", "stitch"))
def test_p12_shared_advanced_and_run_bar_gate_and_persist_opt_out(qapp, make_page, tmp_path, monkeypatch, kind):
    page = make_page(kind)
    spec, output, put, last_row, prepare_later = _ranked_acquisition(page, kind, tmp_path, selectors="3")
    _click_run(qapp, page)
    prepare_later()
    for index in range(4):
        put(3, index, 45 + index)
    owner = start_preview(page, qapp)
    checkbox = preview_checkbox(page)
    assert checkbox.isChecked() and checkbox.toolTip()
    settings = page.run_bar.findChild(QtWidgets.QAbstractButton, "livePreviewSettingsButton")
    assert settings is not None and settings.toolTip()
    _next(page, owner, qapp, lambda: owner.snapshot is not None)
    assert not page.advanced_button.isEnabled() and not settings.isEnabled()
    settings.click()
    assert not page.advanced.isVisible()
    view = _view(page, kind)
    displayed = _display_identity(view)
    assert displayed is not None
    view.commandRequested.emit(ShellCommand(ShellCommandKind.SET_LOG_SCALE, True))
    assert _display_identity(view) == displayed
    view.commandRequested.emit(ShellCommand(ShellCommandKind.SET_BACKGROUND))
    assert page._background.active_key is None and _display_identity(view) == displayed
    _stop_live(qapp, page)
    settings.click()
    wait_until(qapp, lambda: page.advanced.isVisible() and checkbox.hasFocus())
    checkbox.click()
    assert not checkbox.isChecked()
    checkbox.click()
    assert checkbox.isChecked()
    page.advanced.close()
    disable_preview(page)
    document = page.settings()
    reference = make_page(kind)
    reference.apply_settings(document)
    assert not preview_checkbox(reference).isChecked()
    submitted = []
    request = owner._requests.ask

    def record(target):
        if target is not None:
            submitted.append(target)
        return request(target)

    monkeypatch.setattr(owner._requests, "ask", record)
    off = start_live(page)
    assert off.watching
    advance_preview(page, owner, qapp)
    assert owner.snapshot is None and not owner.working and owner.progress.decodes == 0
    _append_spec(spec, last_row)
    put(3, 4, 50)
    for _ in range(4):
        tick_live(page, qapp)
    _wait(qapp, page, lambda: len(_shots(output)) == 2 and not page._owner.busy)
    assert owner.snapshot is None and owner.progress.decodes == 0 and owner.progress.captures == 0
    assert not submitted
    assert _shown_name(page, kind) in _shots(output)
