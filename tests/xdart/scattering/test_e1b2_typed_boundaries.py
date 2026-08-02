from __future__ import annotations

from types import SimpleNamespace

import pytest
from pyqtgraph.Qt import QtWidgets

from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey, StandardEventKind
from xdart.gui.tabs.scattering.events import (
    CleanupStatus, DetachedDiagnostic, ExecutorAccepted, ExecutorClosed, PreflightAccepted,
    RunIdentity,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell


class _Sources:
    def capture(self, *_args):
        raise AssertionError("capture is outside these boundary probes")

    def cancel(self, *_args) -> None:
        return None

    def observe(self, *_args):
        raise AssertionError("observe is outside these boundary probes")

    def cancel_observation(self, *_args) -> None:
        return None


class _Executor:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.close_receipt_factory = None

    def start(self, _configuration, _source, run_identity):
        return ExecutorAccepted(run_identity)

    def pause(self, _run_identity) -> None:
        return None

    def resume(self, _run_identity) -> None:
        return None

    def stop(self, _run_identity) -> None:
        return None

    def close(self, run_identity):
        if self.close_receipt_factory is None:
            return ExecutorClosed(run_identity, CleanupStatus.CLEANED)
        return self.close_receipt_factory(run_identity)

    def drain_events(self):
        events, self.events = tuple(self.events), []
        return events

@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _active_page(executor: _Executor) -> tuple[ScatteringWorkspace, ScatteringCoordinator, RunIdentity]:
    lifecycle = ScatteringCoordinator()
    configuration = RunIntent().freeze()
    begun = lifecycle.begin_start()
    promoted = lifecycle.preflight_accepted(PreflightAccepted(begun.request_id, configuration))
    identity = promoted.run_identity
    assert identity is not None
    assert lifecycle.executor_accepted(ExecutorAccepted(identity)).phase is RunPhase.RUNNING
    page = ScatteringWorkspace(intents=RunIntentStore(), lifecycle=lifecycle,
                               sources=_Sources(), executor=executor)
    return page, lifecycle, identity


def _dispose(page: ScatteringWorkspace, qapp: QtWidgets.QApplication) -> None:
    page.close_workspace()
    page.deleteLater()
    qapp.processEvents()


def test_foreign_event_shape_is_inert_before_lifecycle_mutation(qapp: QtWidgets.QApplication) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    executor.events.append(SimpleNamespace(run_identity=identity, kind=StandardEventKind.FINISHED,
                                           cleanup_status=CleanupStatus.CLEANED,
                                           artifact="forged.nxs", detail=""))
    try:
        page._drain_executor()
        assert lifecycle.phase is RunPhase.RUNNING
        assert lifecycle.active_run_identity is identity
    finally:
        _dispose(page, qapp)


def test_malformed_cleaned_close_receipt_cannot_acknowledge_owners(qapp: QtWidgets.QApplication) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    executor.close_receipt_factory = lambda run_identity: ExecutorClosed(
        run_identity, CleanupStatus.CLEANED, primary=object(),
    )
    page.close_workspace()
    qapp.processEvents()
    try:
        assert lifecycle.closed is True
        assert lifecycle.phase is RunPhase.STOPPING
        assert lifecycle.event_sequence > 0
    finally:
        page.deleteLater()
        qapp.processEvents()


def test_valid_close_diagnostics_survive_the_pipeline_projection(qapp: QtWidgets.QApplication) -> None:
    executor = _Executor()
    page, _lifecycle, _identity = _active_page(executor)
    diagnostic = DetachedDiagnostic("builtins", "RuntimeError", "close failed", "sink.abort")
    executor.close_receipt_factory = lambda run_identity: ExecutorClosed(
        run_identity, CleanupStatus.CLEANUP_PENDING, diagnostic, (diagnostic,),
    )
    try:
        closed = page._pipeline.close()
        assert closed.primary is diagnostic
        assert closed.cleanup_failures == (diagnostic,)
        assert closed.cleanup_status is CleanupStatus.CLEANUP_PENDING
    finally:
        page.deleteLater()
        qapp.processEvents()


def test_page_rejects_events_and_shell_navigation_after_close(
    qapp: QtWidgets.QApplication,
) -> None:
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    page.close_workspace()
    executor.events.append(SimpleNamespace(run_identity=identity, kind=StandardEventKind.FINISHED,
                                           cleanup_status=CleanupStatus.CLEANED,
                                           artifact="late.nxs", detail=""))
    late = DisplayFrameKey(
        identity, "late", "late.nxs", 1, 1
    )
    before = (
        lifecycle.phase,
        lifecycle.event_sequence,
        shell.scientific.title.text(),
    )
    page._drain_executor()
    shell.commandRequested.emit(
        ShellCommand(
            ShellCommandKind.SELECT_FRAME,
            frame=late,
            frames=(late,),
        )
    )
    after = (
        lifecycle.phase,
        lifecycle.event_sequence,
        shell.scientific.title.text(),
    )
    try:
        assert after == before
    finally:
        page.deleteLater()
        qapp.processEvents()
