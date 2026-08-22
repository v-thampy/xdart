"""Focused page/slot composition oracle for standalone Calibrate."""
from __future__ import annotations
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import subprocess
from threading import Event
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from xdart.gui.tabs.scattering import experiment_authoring as authoring
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.controls_inventory import INT_1D_POINTS
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.experiment_authoring import CalibrationRequest
from xdart.gui.tabs.scattering.operation_values import (
    OperationContextStamp, OperationIdentity, OperationTerminal,
    OperationTerminalStatus,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import FormRow
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.readiness import ControlAction, SectionId
from xrd_tools.session.run_configuration import RunIntent
from tests.xdart.scattering.test_p3_calibrate_operation import _PONI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
class _RacingStore(RunIntentStore):
    race = False
    def commit(self, candidate, *, expected_revision):
        if self.race:
            self.race = False; concurrent = self.snapshot().thaw()
            concurrent.project_root = "/concurrent"
            RunIntentStore.commit(self, concurrent, expected_revision=self.revision)
        return RunIntentStore.commit(self, candidate, expected_revision=expected_revision)
def _binary(tmp_path: Path, monkeypatch) -> Path:
    binary = tmp_path / "bin" / "pyFAI-calib2"
    binary.parent.mkdir(exist_ok=True); binary.write_text("fixture"); binary.chmod(0o700)
    monkeypatch.setenv("PATH", f"{binary.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    return binary.resolve()
def _page(tmp_path, monkeypatch, *, store=None, chooser=None):
    _binary(tmp_path, monkeypatch)
    owned = store or RunIntentStore(RunIntent(project_root=str(tmp_path)))
    page = ScatteringWorkspace(
        intents=owned, lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(), control_path_chooser=chooser)
    return page, owned
def _quick_process(monkeypatch, calls):
    class Process:
        pid = 5050
        def __init__(self, argv, **options):
            calls.append((tuple(argv), options)); Path(argv[2]).write_text(_PONI)
        def wait(self, *, timeout): return 0
        def terminate(self): calls.append("terminate")
        def kill(self): calls.append("kill")
    monkeypatch.setattr(authoring, "_popen", Process)
def _held_process(monkeypatch, calls, release, started=None):
    class Process:
        pid = 6060
        def __init__(self, argv, **options):
            calls.append((tuple(argv), options)); Path(argv[2]).write_text(_PONI)
            if started is not None: started.set()
        def wait(self, *, timeout):
            if not release.wait(timeout): raise subprocess.TimeoutExpired("fake", timeout)
            return 0
        def terminate(self): calls.append("terminate")
        def kill(self): calls.append("kill")
    monkeypatch.setattr(authoring, "_popen", Process)
    monkeypatch.setattr(authoring, "_killpg", lambda *_args: None)
def _calibrate(page):
    page._handle_shell_command(ShellCommand(
        ShellCommandKind.CONTROL_ACTION, "calibrate"))
def _join_and_drain(page):
    worker = page._operation_slot._worker
    assert worker is not None; worker.join(3); assert not worker.is_alive()
    page._drain_executor()
def _close(page, qapp):
    page.close_workspace(); page.deleteLater(); qapp.processEvents()
@dataclass(frozen=True, slots=True)
class _HeldJob: label: str = "held"
def test_projection_locks_exact_surfaces_and_preserves_dynamic_cancel_label(tmp_path, monkeypatch, qapp) -> None:
    page, store = _page(tmp_path, monkeypatch); direct = project_controls(store.snapshot(), None, RunPhase.IDLE)
    assert not direct.profile.actions_for(SectionId.EXPERIMENT)[0].enabled
    entered, release = Event(), Event()
    def body(_job, identity, _cancel, _publish): entered.set(); release.wait(2); return OperationTerminal(identity, OperationTerminalStatus.RETURNED)
    identity = page._operation_slot._begin(_HeldJob(), page._operation_context_stamp(), body)
    assert type(identity) is OperationIdentity and entered.wait(2); page._calibration_identity = identity
    try:
        state = page._project_controls(store.snapshot()); actions = state.profile.actions_for(SectionId.EXPERIMENT)
        assert actions[0].enabled and actions[0].label == "Cancel Calibration"; assert not actions[1].enabled and actions[1].action is ControlAction.MAKE_MASK
        assert all(not field.enabled for field in state.bound_controls.fields)
        assert page._start_permitted() == (False, "Experiment operation is still active")
        page._refresh_shell(); labels = {button.text() for button in page._shell.controls.findChildren(QtWidgets.QPushButton)}
        assert "Cancel Calibration" in labels; seen = []
        record = lambda name: lambda *_args: seen.append(name)
        for name in ("_run_action", "_on_field_draft", "_on_field_value", "_choose_control_path", "_edit_advanced_settings", "_edit_run_strip", "_edit_performance_diagnostics"): monkeypatch.setattr(page, name, record(name))
        monkeypatch.setattr(page, "_request_browser_catalog", record("navigation")); monkeypatch.setattr(page, "_calibrate_action", record("cancel"))
        blocked = (
            ShellCommand(ShellCommandKind.RUN_ACTION), ShellCommand(ShellCommandKind.CONTROL_DRAFT, "x", ("x",)),
            ShellCommand(ShellCommandKind.CONTROL_EDIT, "x", ("x",)), ShellCommand(ShellCommandKind.CONTROL_BROWSE, path=("x",)),
            ShellCommand(ShellCommandKind.CONTROL_ACTION, "advanced_processing"),
            *(ShellCommand(kind, 1) for kind in (ShellCommandKind.SET_PROCESSING_MODE, ShellCommandKind.SET_BATCH, ShellCommandKind.SET_CORES, ShellCommandKind.SET_LIVE, ShellCommandKind.SET_OUTPUT_POLICY)),
            ShellCommand(ShellCommandKind.MENU, "Config:Performance Diagnostics…"), ShellCommand(ShellCommandKind.MENU, "Config:Heavy residency:16"))
        revision = store.revision
        for command in blocked: page._handle_shell_command(command)
        assert seen == [] and store.revision == revision
        page._handle_shell_command(ShellCommand(ShellCommandKind.REFRESH_BROWSER)); page._handle_shell_command(ShellCommand(ShellCommandKind.CONTROL_ACTION, "calibrate"))
        assert seen == ["navigation", "cancel"]
        page._calibration_identity = None; page._handle_shell_command(ShellCommand(ShellCommandKind.CONTROL_ACTION, "calibrate"))
        assert seen == ["navigation", "cancel"]; assert not page._project_controls(store.snapshot()).profile.actions_for(SectionId.EXPERIMENT)[0].enabled
        running = project_controls(store.snapshot(), None, RunPhase.RUNNING, calibrate_available=True)
        assert not running.profile.actions_for(SectionId.EXPERIMENT)[0].enabled
        assert "calibrate_dependency_available" in inspect.signature(project_controls).parameters
        failed = project_controls(store.snapshot(), None, RunPhase.FAILED, calibrate_dependency_available=True)
        assert failed.profile.actions_for(SectionId.EXPERIMENT)[0].reason == "Calibration is unavailable in the current workspace state."
    finally:
        release.set(); page._operation_slot._worker.join(2); page._operation_slot.poll(identity); _close(page, qapp)
def test_command_settles_focused_edit_before_chooser_and_adopts_exact_cas(
    tmp_path, monkeypatch, qapp
) -> None:
    calls, chooser_revisions = [], []
    target = tmp_path / "new-calibration"
    page = None
    def choose(_path, current, _start):
        assert current == ""; chooser_revisions.append(store.revision); return str(target)
    page, store = _page(tmp_path, monkeypatch, chooser=choose)
    _quick_process(monkeypatch, calls)
    try:
        page.show(); qapp.processEvents()
        row = next(item for item in page._shell.controls.findChildren(FormRow)
                   if item.path == INT_1D_POINTS)
        row.editor.setFocus(); qapp.processEvents(); row.editor.setText("invalid")
        _calibrate(page)
        assert chooser_revisions == [] and not page._operation_slot.owned
        row.editor.setText("444"); _calibrate(page); _join_and_drain(page)
        assert chooser_revisions == [1] and len(calls) == 1
        assert store.revision == 2
        intent = store.snapshot().thaw()
        assert intent.bai_1d_args["npt"] == 444
        assert intent.poni_file == str(target.with_suffix(".poni"))
        assert "adopted" in page._notice_text.lower()
    finally: _close(page, qapp)
@pytest.mark.parametrize("outcome", ("current", "stale", "cas_race"))
def test_published_result_adopts_only_current_stamp_and_original_revision(
    tmp_path, monkeypatch, qapp, outcome
) -> None:
    release, calls = Event(), []
    store = _RacingStore(RunIntent(project_root=str(tmp_path)))
    page, store = _page(tmp_path, monkeypatch, store=store,
                        chooser=lambda *_args: str(tmp_path / f"{outcome}.poni"))
    _held_process(monkeypatch, calls, release)
    try:
        _calibrate(page); identity = page._calibration_identity
        assert type(identity) is OperationIdentity and page._operation_slot.owned
        if outcome == "stale":
            page._operation_slot.observe_stamp(OperationContextStamp(0, "other", 1))
            page._operation_slot.observe_stamp(OperationContextStamp(0))
        elif outcome == "cas_race": store.race = True
        release.set(); _join_and_drain(page)
        path = tmp_path / f"{outcome}.poni"
        assert path.exists() and len(calls) == 1
        assert (store.snapshot().thaw().poni_file == str(path)) is (outcome == "current")
        if outcome != "current": assert "not adopted" in page._notice_text.lower() or "superseded" in page._notice_text.lower()
        prior = store.revision; page._drain_executor(); assert store.revision == prior
    finally: _close(page, qapp)
def test_held_worker_keeps_gui_responsive_single_flight_and_blocks_run(
    tmp_path, monkeypatch, qapp
) -> None:
    release, calls, heartbeat = Event(), [], Event()
    page, _store = _page(tmp_path, monkeypatch,
                         chooser=lambda *_args: str(tmp_path / "held.poni"))
    _held_process(monkeypatch, calls, release)
    try:
        _calibrate(page); identity = page._calibration_identity
        QtCore.QTimer.singleShot(0, heartbeat.set); qapp.processEvents()
        assert heartbeat.is_set() and page._operation_slot._worker.is_alive()
        request = CalibrationRequest(str(tmp_path / "second.poni"), _binary(tmp_path, monkeypatch).as_posix())
        assert page._operation_slot.begin_calibrate(request, OperationContextStamp(0)) is None
        assert page._start_permitted()[0] is False and len(calls) == 1
        release.set(); _join_and_drain(page)
        assert page._operation_slot.poll(identity) is None
    finally: release.set(); _close(page, qapp)
def test_duplicate_cancel_notice_never_claims_irrevocable_publication(
    tmp_path, monkeypatch, qapp
) -> None:
    release, started, calls = Event(), Event(), []
    page, _store = _page(tmp_path, monkeypatch,
                         chooser=lambda *_args: str(tmp_path / "cancel.poni"))
    _held_process(monkeypatch, calls, release, started)
    try:
        _calibrate(page); assert started.wait(2)
        _calibrate(page)
        assert page._notice_text == "Cancelling calibration…"
        _calibrate(page)
        assert page._notice_text == "Calibration cancellation was not accepted."
        assert "irrevocable" not in page._notice_text.lower()
        release.set(); _join_and_drain(page)
    finally: release.set(); _close(page, qapp)
def test_close_reuses_same_calibration_child_and_stage_until_clean(
    tmp_path, monkeypatch, qapp
) -> None:
    release, started, calls = Event(), Event(), []
    page, store = _page(tmp_path, monkeypatch,
                        chooser=lambda *_args: str(tmp_path / "closing.poni"))
    _held_process(monkeypatch, calls, release, started)
    _calibrate(page); worker = page._operation_slot._worker
    try:
        assert started.wait(2)
        first, second = page.close_workspace(), page.close_workspace()
        assert first.cleanup_status.value == second.cleanup_status.value == "cleanup_pending"
        assert page._operation_slot._worker is worker and len(calls) == 1
        release.set(); worker.join(3)
        cleaned = page.close_workspace()
        assert cleaned.cleanup_status.value == "cleaned" and len(calls) == 1
        assert store.snapshot().thaw().poni_file == ""
    finally:
        release.set(); page.close_workspace(); page.deleteLater(); qapp.processEvents()
