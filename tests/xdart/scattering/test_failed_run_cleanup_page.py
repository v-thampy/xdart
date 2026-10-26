"""A Live Run that fails leaves the page usable (beamline 2026-10-26: a Live Append failed on an output written before
epoch_local, "dirty metadata adds columns absent on disk: ['epoch_local']"; the maintainer: "the run/stop buttons and
browser are totally unresponsive", "the run button however remains disabled and also the data browser (scans
column)"). The failed scan's output released its light-1D lease as it closed while its display owner still named it,
so the Run's cleanup stayed pending and nothing retried it. The real page, executor and Live Run over real TIFF + .txt
files: the Run has shown a covered scan and grown a new one before the older output fails it. That refusal now skips
its scan (MetadataColumnsAbsent, the maintainer's pick "Skip that scan"), so these rows turn it back into a plain
writer refusal, which still fails the Run, as any write that fails otherwise does."""
from __future__ import annotations

import types
import threading
import time

import h5py
import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph.Qt import QtCore

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _poni, _pump
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _settle, _stop, _wait
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _start
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering.adapters import run_executor
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def _intent(root):
    raw = root / "raw"
    return RunIntent(source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="txt"),
                     poni_file=str(root / "cal.poni"), project_root=str(raw), save_path=str(root / "processed"),
                     output_mode="Append", processing_mode="Int 1D", live_mode=True, max_cores=1,
                     bai_1d_args={"npt": 8, "method": "numpy"})


def _frames(events):
    return sum(1 for event in events if event.kind is StandardEventKind.FRAME_READY)


def _failed_run(qapp, page, raw, observed):  # noqa: F811
    """The page's Live Run grows a new scan, then the older output's next frame fails it: its terminal."""
    page._shell.run_controls.startButton.click()
    assert _pump(qapp, lambda: page._lifecycle.active_run_identity is not None, 30.)
    for index in (1, 2, 3):
        _frame(raw, "fresh", index, "txt", float(index))
        _pump(qapp, lambda: False, .3)
    assert _pump(qapp, lambda: _frames(observed) >= 3, 60.), "the new scan was never processed"
    _frame(raw, "old", 2, "txt", 2.0)
    assert _pump(qapp, lambda: any(event.kind in _TERMINAL for event in observed), 60.), "the Run never ended"
    _pump(qapp, lambda: False, 1.)
    return [event for event in observed if event.kind in _TERMINAL][-1]


@pytest.fixture
def older_output(tmp_path, monkeypatch):
    """A covered scan and an older output, written by a first Run; the older one's scan_data lacks epoch_local, and
    its writer's refusal fails the Run."""
    from xrd_tools.io import record_writer
    validate = record_writer.NexusRecordWriter._validate_records

    def refuse(self, *args, **kwargs):
        try:
            return validate(self, *args, **kwargs)
        except record_writer.MetadataColumnsAbsent as error:
            raise ValueError(str(error)) from None  # a write refusal of no scan's own
    monkeypatch.setattr(record_writer.NexusRecordWriter, "_validate_records", refuse)
    raw = tmp_path / "raw"
    raw.mkdir()
    _poni(tmp_path)
    for stem in ("done", "old"):
        _frame(raw, stem, 1, "txt", 1.0)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path), request_value=7731)
    events = []
    _wait(executor, events, lambda values: _frames(values) >= 2 or any(e.kind in _TERMINAL for e in values))
    _settle(executor, events, .5)
    _stop(executor, identity, events)
    with h5py.File(tmp_path / "processed" / "old_int1d.nexus", "r+") as handle:
        del handle["entry/scan_data/epoch_local"]  # as an output written before f6efad03 (2026-10-24)
    return raw


def test_a_live_run_that_fails_on_its_last_scan_ends_clean_and_leaves_run_and_the_scans_open(
        qapp, tmp_path, older_output):  # noqa: F811
    """Its cleanup settles the released lease (703753b1). A preview read still in flight when the failed output let go
    of its curves leaves that one release pending at the Run's end ("shared hydration gate remains retained": 1 of 20
    local runs, and hosted run 40303254364); the page's background retry then settles it, so Run and the scans open
    either way."""
    with _page(qapp, _intent(tmp_path)) as (page, _view, observed):
        terminal = _failed_run(qapp, page, older_output, observed)
        assert terminal.kind is StandardEventKind.FAILED and "epoch_local" in terminal.detail, terminal.detail
        run = page._run_executor._exact_run(terminal.run_identity)
        owners = None if run is None else {
            str(key).rsplit("/", 1)[-1]: (None if owner.light_lease is None else owner.light_lease.state.name,
                                          None if owner.light_slot is None else owner.light_slot.state.name,
                                          owner.light_hooks is not None, owner.light_retry_token is not None,
                                          owner.light_unsubscribe is not None, owner.publications.allocation is not None,
                                          owner.publications._light_1d is not None)
            for key, owner in run.display.artifacts.items()}
        retained = None if run is None else (
            tuple(name for name in ("session", "sink", "output", "source", "resources")
                  if getattr(run, name) is not None),
            tuple(f"{error.operation}: {error.message}" for error in run.cleanup_failures),
            run.display.light_1d_cleanup_unresolved())
        if terminal.cleanup_status is not CleanupStatus.CLEANED:  # a read was in flight: the page's retry settles it
            assert terminal.cleanup_status is CleanupStatus.CLEANUP_PENDING, (terminal.cleanup_status, owners, retained)
            assert retained is not None and not retained[0] and retained[1] == (
                "dynamic_output.finish: light-1D cleanup remains pending: shared hydration gate remains retained",
            ), (owners, retained)
            assert _pump(qapp, lambda: page._lifecycle.reset_permitted, 10.), (owners, retained)
        assert page._lifecycle.reset_permitted and page._start_permitted() == (True, "")
        assert page._context_controller.browse_lifecycle_admissible(), "the browser's scans stay locked"
        assert page._shell.browser.scans.isEnabled()


def test_a_failed_run_whose_cleanup_was_left_pending_is_settled_by_the_page_and_run_starts_again(
        qapp, tmp_path, older_output, monkeypatch):  # noqa: F811
    """The page's own recovery, for a cleanup still pending at the Run's end (the executor's settling of a released
    lease held off here, which is the beamline's state): the page closes the failed Run again on its own, Run and the
    scans open, and a new Run starts."""
    monkeypatch.setattr(run_executor, "Light1DLeaseState", types.SimpleNamespace(RELEASED=object()), raising=False)
    with _page(qapp, _intent(tmp_path)) as (page, _view, observed):
        terminal = _failed_run(qapp, page, older_output, observed)
        assert terminal.cleanup_status is CleanupStatus.CLEANUP_PENDING, terminal.cleanup_status
        assert _pump(qapp, lambda: page._start_permitted() == (True, ""), 10.), page._start_permitted()
        assert page._shell.browser.scans.isEnabled() and page._shell.run_controls.startButton.isEnabled()
        observed.clear()
        page._shell.run_controls.startButton.click()
        assert _pump(qapp, lambda: page._lifecycle.phase is RunPhase.RUNNING, 30.), page._lifecycle.phase
        page._shell.run_controls.stopButton.click()
        assert _pump(qapp, lambda: any(event.kind in _TERMINAL for event in observed), 60.)


def test_pending_failed_cleanup_keeps_the_gui_heartbeat_running(qapp, tmp_path, older_output, monkeypatch):
    # Retain the real packet's pending-cleanup setup; the changed seam below is
    # a call-through gate, never a fabricated close result.
    monkeypatch.setattr(run_executor, "Light1DLeaseState", types.SimpleNamespace(RELEASED=object()), raising=False)
    with _page(qapp, _intent(tmp_path)) as (page, _view, observed):
        entered, release = threading.Event(), threading.Event()
        actual, callers, beats = page._run_executor._close_run, [], []

        def slow_close(identity):
            callers.append(threading.current_thread() is threading.main_thread())
            entered.set()
            gate = threading.Timer(.6, release.set)
            gate.start()
            try:
                assert release.wait(5.)
                return actual(identity)
            finally:
                gate.join()

        monkeypatch.setattr(page._run_executor, "_close_run", slow_close)
        heartbeat = QtCore.QTimer(page)
        heartbeat.setInterval(10)
        heartbeat.timeout.connect(lambda: beats.append(time.monotonic())
                                  if entered.is_set() and not release.is_set() else None)
        heartbeat.start()
        try:
            terminal = _failed_run(qapp, page, older_output, observed)
            assert terminal.cleanup_status is CleanupStatus.CLEANUP_PENDING
            assert _pump(qapp, lambda: page._start_permitted() == (True, ""), 10.)
            assert callers == [False], 'failed cleanup ran on the GUI thread or overlapped itself'
            assert len(beats) >= 5, 'cleanup stopped GUI heartbeats while close was pending'
            assert page._context_controller.browse_lifecycle_admissible()
        finally:
            release.set()
            heartbeat.stop()
