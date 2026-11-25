"""Per-scan Live command, native settlement and close rows G3/G4/G5/G9/G10."""

from contextlib import contextmanager
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
from threading import Event

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.core.test_stitch_operation import _native_form
from tests.xdart.pages.live_per_scan_support import start_live, tick_live, unimplemented_live, wait_until
from tests.xdart.pages.live_preview_support import disable_preview
from tests.xdart.pages import test_rsm_page as rsm_tests
from tests.xdart.pages import test_stitch_page as stitch_tests
from xdart.gui.pages.catalog import RSM_PAGE, STITCH_PAGE
from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.stitch.page import StitchPage
from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xrd_tools.analysis.rsm_operation import RSMOperationExecutionV2
from xrd_tools.analysis.stitch_operation import StitchOperationExecution
from xrd_tools.session.readiness import ControlAction


pytestmark = unimplemented_live


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _fill(page, kind, root, qapp):
    if kind == "rsm":
        source = rsm_tests._synthetic_source(root, full=True)
        rsm_tests._with_source(page, root, scans="1,2")
        rsm_tests._wait(page, qapp, lambda: not page.active())
        return source, source.parent / "images"

    form = _native_form(root, "multigeometry", scans="1,2")
    page.load_form(form)
    stitch_tests._wait(page, qapp)
    return Path(form.spec_path), Path(form.image_dir)


@contextmanager
def _mounted(kind, root, qapp):
    page = (rsm_tests._page(qapp) if kind == "rsm"
            else StitchPage(empty_host_services(stitch_tests._Status())))
    try:
        yield page, *_fill(page, kind, root, qapp)
    finally:
        if kind == "rsm":
            rsm_tests._close(page, qapp)
        else:
            stitch_tests._dispose(page, qapp)
        qapp.processEvents()


def _take_images(images, pattern="*"):
    saved = {path: path.read_bytes() for path in images.glob(pattern) if path.is_file()}
    assert saved
    for path in saved:
        path.unlink()
    return saved


def _put_images(saved):
    for path, content in saved.items():
        path.write_bytes(content)


def _outputs(root):
    return sorted(root.rglob("*.nexus"))


def _hold_native_progress(monkeypatch, page_kind, stage):
    execution = RSMOperationExecutionV2 if page_kind == "rsm" else StitchOperationExecution
    original = execution._emit
    entered, release = Event(), Event()

    def emit(self, actual_stage, *args):
        original(self, actual_stage, *args)
        if actual_stage == stage and not entered.is_set():
            entered.set()
            assert release.wait(20), "the test did not release the native science worker"

    monkeypatch.setattr(execution, "_emit", emit)
    return entered, release


@pytest.mark.parametrize("kind", ("rsm", "stitch"))
@pytest.mark.parametrize("phase", ("watching", "queued-preflight", "science", "committed", "verification"))
def test_g3_stop_revokes_dispatch_and_drains_real_native_terminals(qapp, tmp_path, monkeypatch, kind, phase):
    with _mounted(kind, tmp_path, qapp) as (page, _source, images):
        worker = page._owner._worker
        saved = _take_images(images) if phase == "watching" else {}
        entered, release = Event(), Event()
        if phase == "queued-preflight":
            original_poll = worker.poll

            def poll(identity):
                if not release.is_set() and identity.request.action.value == "preflight":
                    if worker._terminal is not None:
                        entered.set()
                    return None  # Hold delivery of the real queued terminal.
                return original_poll(identity)

            monkeypatch.setattr(worker, "poll", poll)
        if phase in ("science", "committed"):
            stage = "science" if phase == "science" else "reload"
            entered, release = _hold_native_progress(monkeypatch, kind, stage)
        if phase == "verification":
            science = import_module(f"xrd_tools.analysis.{kind}_operation")
            original_read = science.read_analysis_artifact

            def read(*args, **kwargs):
                if kwargs.get("expected_receipt") is not None and not entered.is_set():
                    entered.set()
                    assert release.wait(20), "the test did not release strict readback"
                    raise OSError("temporarily unavailable during strict readback")
                return original_read(*args, **kwargs)

            monkeypatch.setattr(science, "read_analysis_artifact", read)

        try:
            start_live(page)
            tick_live(page, qapp)
            if phase != "watching":
                wait_until(qapp, entered.is_set)
            kept = _outputs(tmp_path)
            assert len(kept) == int(phase in ("committed", "verification"))
            assert page.run_bar.stopButton.isEnabled()
            page.run_bar.stopButton.click()
            if phase != "watching":
                assert page.active()
                assert not page.preview_button.isEnabled()
        finally:
            release.set()
        if phase == "verification":
            wait_until(qapp, lambda: page.retry_button.isEnabled() and not page.retry_button.isHidden())
            assert page.active()
            page.retry_button.click()
        wait_until(qapp, lambda: not page.active())
        _put_images(saved)
        tick_live(page, qapp, 6)
        assert not page.active()
        assert _outputs(tmp_path) == kept


@pytest.mark.parametrize("kind", ("rsm", "stitch"))
def test_g4_live_watching_fences_the_actual_host_and_page_commands(qapp, tmp_path, monkeypatch, kind):
    from xdart._gui_main import Main

    descriptor = RSM_PAGE if kind == "rsm" else STITCH_PAGE
    host = Main(page_descriptors=(descriptor,), selected_page_key=descriptor.key)
    page = host.page_handle.widget
    try:
        source, images = _fill(page, kind, tmp_path, qapp)
        _take_images(images)
        original = page.settings()
        changed = deepcopy(original)
        changed["fields"]["scan"] = "19"
        document = tmp_path / "settings.json"
        document.write_text(json.dumps(changed), encoding="utf-8")
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *a, **k: (str(document), ""))

        watcher = start_live(page)
        tick_live(page, qapp)
        bar = page.run_bar
        assert page.active() and not page._owner.busy
        assert bar.stopButton.isEnabled()
        assert "scan" in bar.liveButton.toolTip().lower()
        assert bar.readinessLabel.toolTip()
        assert not bar.liveButton.isEnabled()
        assert not bar.writeModeButton.isEnabled()
        assert not bar.modeCombo.isEnabled()

        host._shortcut_run_pause()
        bar.startButton.click()
        bar.liveButton.click()
        host._shortcut_toggle_write_mode()
        other_mode = "Stitch 1D" if kind == "rsm" else "Stitch 2D"
        host.page_handle.processing_mode.select(other_mode)
        for action in (ControlAction.PREVIEW, ControlAction.CALIBRATE, ControlAction.MAKE_MASK,
                       ControlAction.ADVANCED_PROCESSING):
            page._control_action(action)
        page.load()
        try:
            page.apply_settings(changed)
        except ValueError:
            pass  # An explicit settings refusal is allowed.
        page.set_field("scan", "21")
        page.set_sources([str(source.with_name("wrong.spec"))])
        if kind == "rsm":
            page.reopen(str(source))
            assert page._reading is None
        else:
            page.open_result(str(source))
            assert page._reloading is None
        qapp.processEvents()
        assert page._live_watcher is watcher and page.active()
        assert page.settings() == original
        assert not page._owner.busy and not page.advanced.isVisible()
        assert not any(child.isVisible() for child in page._authoring_dialogs)
        assert not _outputs(tmp_path)
        bar.stopButton.click()
        wait_until(qapp, lambda: not page.active())
    finally:
        host.close()
        wait_until(qapp, lambda: not host.isVisible())
        host.deleteLater()
        qapp.processEvents()


def test_g5_preflight_refusal_fairness_and_post_acceptance_failure(qapp, tmp_path, monkeypatch):
    from xrd_tools.analysis import stitch_operation

    with _mounted("stitch", tmp_path, qapp) as (page, source, _images):
        prepared = prepare_stitch_tool(page.build_form())
        first, second = [Path(job.module.output.target) for job in prepared.requests]
        # A real foreign output refuses the first group at native preflight.
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_bytes(b"foreign output")
        attempts, science = [], []
        original_prepare = page._owner._preflight_runner
        original_science = stitch_operation.run_stitch

        def prepare(form, *args, **kwargs):
            attempts.append(form)
            return original_prepare(form, *args, **kwargs)

        def run_science(*args, **kwargs):
            if page._owner.current_identity.request.execution.request.module.output.target == str(second):
                science.append(second)
                raise ArithmeticError("injected failure after the native acceptance barrier")
            return original_science(*args, **kwargs)

        monkeypatch.setattr(page._owner, "_preflight_runner", prepare)
        monkeypatch.setattr(stitch_operation, "run_stitch", run_science)
        start_live(page)
        tick_live(page, qapp)
        wait_until(qapp, lambda: science == [second] and not page._owner.busy)
        assert first.read_bytes() == b"foreign output" and not second.exists()
        checked = len(attempts)
        with source.open("ab") as stream:
            stream.write(b"\n#S 39 ascan del 14 15 1 1\n")
        tick_live(page, qapp, 6)
        assert len(attempts) == checked and science == [second]
        first.unlink()
        tick_live(page, qapp)
        tick_live(page, qapp)
        wait_until(qapp, lambda: first.exists() and not page._owner.busy)
        assert len(attempts) > checked and not second.exists()


@pytest.mark.parametrize("kind", ("rsm", "stitch"))
@pytest.mark.parametrize("boundary", ("observation", "native"))
def test_g9_close_joins_the_live_worker_and_latches_dispatch(qapp, tmp_path, monkeypatch, kind, boundary):
    with _mounted(kind, tmp_path, qapp) as (page, _source, _images):
        entered, release = Event(), Event()
        observed = []
        close = page.close_page if kind == "rsm" else page.shutdown
        if not page.run_bar.liveButton.isHidden() and boundary == "observation":
            observer = import_module("xrd_tools.analysis.live_scans").LiveScanObserver
            original_poll = observer.poll

            def poll(self, *args, **kwargs):
                observed.append(self)
                entered.set()
                assert release.wait(20), "the test did not release the live observation"
                return original_poll(self, *args, **kwargs)

            monkeypatch.setattr(observer, "poll", poll)
        if boundary == "native":
            entered, release = _hold_native_progress(monkeypatch, kind, "reload")

        try:
            watcher = start_live(page)
            if boundary == "native":
                tick_live(page, qapp)
            wait_until(qapp, entered.is_set)
            for _ in range(3):
                watcher.clock.advance()
                watcher.poll_due()
                qapp.processEvents()
            if boundary == "observation":
                assert len(observed) == 1
            assert close().status is PageCleanup.PENDING
            assert close().status is PageCleanup.PENDING
            watcher.poll_due()
            kept = _outputs(tmp_path)
        finally:
            release.set()
        wait_until(qapp, lambda: close().status is PageCleanup.CLEAN)
        assert _outputs(tmp_path) == kept
        assert len(kept) == int(boundary == "native")
        if boundary == "observation":
            assert len(observed) == 1
        assert not any(dialog.isVisible() for dialog in page.findChildren(QtWidgets.QDialog))


@pytest.mark.parametrize("kind", ("rsm", "stitch"))
def test_g10_stop_restart_and_finite_append_keep_the_full_selection(qapp, tmp_path, kind):
    with _mounted(kind, tmp_path, qapp) as (page, _source, images):
        later = _take_images(images, "*scan2_*")
        disable_preview(page)
        start_live(page)
        tick_live(page, qapp)
        wait_until(qapp, lambda: len(_outputs(tmp_path)) == 1 and not page._owner.busy)
        (first,) = _outputs(tmp_path)
        first_revision = first.stat().st_mtime_ns, first.read_bytes()
        page.run_bar.stopButton.click()
        wait_until(qapp, lambda: not page.active())

        start_live(page)
        tick_live(page, qapp)
        assert page.run_bar.stopButton.isEnabled()
        page.run_bar.stopButton.click()
        wait_until(qapp, lambda: not page.active())
        _put_images(later)
        if page.run_bar.is_live():
            page.run_bar.liveButton.click()
        page.begin_preview()
        wait_until(qapp, lambda: not page.active())
        if kind == "rsm":
            assert len(page.build_forms()) == 2
        else:
            assert len(page._owner.prepared.requests) == 2
        page.run_bar.startButton.click()
        wait_until(qapp, lambda: len(_outputs(tmp_path)) == 2 and not page.active())
        assert (first.stat().st_mtime_ns, first.read_bytes()) == first_revision
