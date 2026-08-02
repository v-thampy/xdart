"""Frozen E2-LV-R4.1 composed-Close and detector fail-closed oracle."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
import time

import numpy as np
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering._admission import ImmediateAdmission
from xdart.gui.tabs.scattering.contracts import (
    SourceCapture,
    SourceObservation,
    SourceObservationRequest,
    SourceObservationStatus,
)
from tests.xdart.scattering.test_e2lv_live_display import _wait
from xdart.gui.tabs.scattering.context_controller import ContextController
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_runtime import RunDisplayState
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    ExecutorAccepted,
    ExecutorClosed,
    RunIdentity,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.start_outcomes import StartCapture
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import (
    ScatteringWorkspaceShell,
)
from xdart.modules.frame_publication import FramePublication
from xdart.modules.display_context import (
    AcquisitionContext,
    ContextKind,
    new_context_token,
)
from xrd_tools.core import Axis, FrameRecord, FrameView
from xrd_tools.io.image_source import RawFrameResult
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec


class _Sources:
    def __init__(self) -> None:
        self.epoch = 0

    def capture(self, source, request_id):
        self.epoch += 1
        return SourceCapture(request_id, self.epoch, source)

    def cancel(self, _request_id) -> None:
        return None

    def observe(
        self, request: SourceObservationRequest
    ) -> SourceObservation:
        return SourceObservation(
            request.observation_id,
            request.intent_revision,
            request.source,
            SourceObservationStatus.AVAILABLE,
            "frame.tif",
            True,
            False,
        )

    def cancel_observation(self, _observation_id: int) -> None:
        return None

    def publish_motor_knowledge(self, _observation) -> None:
        return None

    def project_motor_knowledge(self, _source, _fingerprint=None):
        return None


class _CountingSources(_Sources):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_calls = 0

    def cancel(self, request_id) -> None:
        self.cancel_calls += 1
        super().cancel(request_id)


class _FailOnceCancelSources(_Sources):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_calls: list[object] = []

    def cancel(self, request_id) -> None:
        self.cancel_calls.append(request_id)
        if len(self.cancel_calls) == 1:
            raise RuntimeError("source cancel failed once")


class _SequencedExecutor(ImmediateAdmission):
    def __init__(self, responses: list[object]) -> None:
        self.events: list[object] = []
        self.start_calls = 0
        self.responses = responses
        self.close_calls: list[RunIdentity] = []
        self.context_identity: RunIdentity | None = None
        self.context: AcquisitionContext | None = None
        self.cleaned_substeps = {
            "session": 0,
            "sink": 0,
            "source": 0,
            "admission": 0,
        }

    def _remember_context(self, configuration, identity: RunIdentity) -> None:
        display = RunDisplayState(identity, max_payload_items=2)
        context = AcquisitionContext(
            context_token=new_context_token(ContextKind.ACQUISITION),
            run_configuration=configuration,
            config_generation=identity.generation,
            config_fingerprint=identity.fingerprint,
            run_scan_key="cleanup-boundary",
            source_path="/cleanup-boundary/frame.tif",
            scan=object(),
            frame=None,
            frame_ids=display.catalog,
            frames=display.artifacts,
            viewer_rows_1d=(),
            viewer_rows_2d=(),
            publication_store=display,
            origin="scattering-standard",
        )
        context.adopt_record_store(display)
        self.context_identity = identity
        self.context = context

    def start(self, configuration, source, identity, admission):
        self.start_calls += 1
        self._remember_context(configuration, identity)
        return ExecutorAccepted(identity)

    def acquisition_context(self, identity):
        return self.context if identity is self.context_identity else None

    def stop(self, _identity) -> None:
        return None

    def pause(self, _identity) -> None:
        return None

    def resume(self, _identity) -> None:
        return None

    def drain_events(self):
        events, self.events = tuple(self.events), []
        return events

    def close(self, identity):
        self.close_calls.append(identity)
        if len(self.close_calls) == 1:
            for owner in self.cleaned_substeps:
                self.cleaned_substeps[owner] += 1
        response = self.responses.pop(0)
        if response == "pending":
            return ExecutorClosed(identity, CleanupStatus.CLEANUP_PENDING)
        if response == "cleaned":
            return ExecutorClosed(identity, CleanupStatus.CLEANED)
        if response == "foreign":
            return ExecutorClosed(
                RunIdentity(identity.generation, identity.fingerprint),
                CleanupStatus.CLEANED,
            )
        return response


class _FailOnceCloseCoordinator(ScatteringCoordinator):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("lifecycle close failed once")
        return super().close()


class _FailOnceOwnersCoordinator(ScatteringCoordinator):
    def __init__(self) -> None:
        super().__init__()
        self.owners_closed_calls: list[RunIdentity] = []

    def owners_closed(self, event):
        self.owners_closed_calls.append(event.run_identity)
        if len(self.owners_closed_calls) == 1:
            raise RuntimeError("owners_closed failed once")
        return super().owners_closed(event)


class _RaiseAfterCloseCoordinator(ScatteringCoordinator):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        result = super().close()
        if self.close_calls == 1:
            raise RuntimeError("close raised after taking effect")
        return result


class _RaiseAfterOwnersCoordinator(ScatteringCoordinator):
    def __init__(self) -> None:
        super().__init__()
        self.owners_closed_calls: list[RunIdentity] = []

    def owners_closed(self, event):
        self.owners_closed_calls.append(event.run_identity)
        result = super().owners_closed(event)
        if len(self.owners_closed_calls) == 1:
            raise RuntimeError("owners_closed raised after taking effect")
        return result


def _mounted(
    page: ScatteringWorkspace,
) -> tuple[ScatteringWorkspaceShell, ContextController]:
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    controller = page._context_controller
    assert type(controller) is ContextController
    return shell, controller


def _active_page(
    tmp_path: Path,
    executor: _SequencedExecutor,
    *,
    lifecycle: ScatteringCoordinator | None = None,
    sources: _Sources | None = None,
) -> tuple[
    QtWidgets.QApplication,
    ScatteringWorkspace,
    ScatteringCoordinator,
    RunIdentity,
]:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    selected = tmp_path / "frame.tif"
    selected.write_bytes(b"frame")
    poni = tmp_path / "tiny.poni"
    poni.write_text("poni")
    coordinator = lifecycle or ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=image_series_spec(selected),
                poni_file=str(poni),
                project_root=str(tmp_path),
                save_path=str(tmp_path / "out.nxs"),
                output_mode="Overwrite",
            )
        ),
        lifecycle=coordinator,
        sources=sources or _Sources(),
        executor=executor,
    )
    shell, controller = _mounted(page)
    shell.run_controls.startButton.click()
    _wait(qapp, lambda: coordinator.phase is RunPhase.RUNNING)
    identity = coordinator.active_run_identity
    assert identity is not None
    controller.adopt_acquisition(identity)
    assert controller.run_identity is identity
    return qapp, page, coordinator, identity


def _dispose(
    page: ScatteringWorkspace,
    qapp: QtWidgets.QApplication,
) -> None:
    page.close_workspace()
    page.deleteLater()
    qapp.processEvents()


def test_active_public_close_completes_exact_lifecycle_owner(
    tmp_path: Path,
) -> None:
    executor = _SequencedExecutor(["pending", "cleaned"])
    qapp, page, lifecycle, identity = _active_page(tmp_path, executor)
    try:
        pending = page.close_workspace()
        assert lifecycle.phase is RunPhase.STOPPING
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert pending.cleanup_identity is identity

        terminal = page.close_workspace()
        assert executor.close_calls == [identity, identity]
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is identity
        assert lifecycle.phase is RunPhase.CLOSED
    finally:
        _dispose(page, qapp)


def test_lifecycle_close_failure_retains_exact_cleanup_owner_for_retry(
    tmp_path: Path,
) -> None:
    lifecycle = _FailOnceCloseCoordinator()
    executor = _SequencedExecutor(["pending", "cleaned"])
    qapp, page, lifecycle, identity = _active_page(
        tmp_path, executor, lifecycle=lifecycle
    )
    try:
        pending = page.close_workspace()
        assert lifecycle.phase is RunPhase.RUNNING
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert pending.cleanup_identity is identity

        terminal = page.close_workspace()
        assert lifecycle.close_calls == 2
        assert executor.close_calls == [identity, identity]
        assert lifecycle.phase is RunPhase.CLOSED
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is identity
        assert [
            (
                failure.operation,
                failure.exception.message,
            )
            for failure in terminal.recovery_failures
        ] == [
            (
                "lifecycle.close",
                "lifecycle close failed once",
            )
        ]
    finally:
        _dispose(page, qapp)


def test_lifecycle_close_retry_preserves_one_clean_executor_receipt(
    tmp_path: Path,
) -> None:
    lifecycle = _FailOnceCloseCoordinator()
    executor = _SequencedExecutor(["cleaned"])
    qapp, page, lifecycle, identity = _active_page(
        tmp_path, executor, lifecycle=lifecycle
    )
    try:
        pending = page.close_workspace()
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert pending.cleanup_identity is identity
        assert executor.close_calls == [identity]

        terminal = page.close_workspace()
        assert lifecycle.close_calls == 2
        assert executor.close_calls == [identity]
        assert lifecycle.phase is RunPhase.CLOSED
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is identity
    finally:
        _dispose(page, qapp)


def test_lifecycle_retry_reuses_exact_preparing_source_owner_once(
    tmp_path: Path,
) -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    selected = tmp_path / "frame.tif"
    selected.write_bytes(b"frame")
    lifecycle = _FailOnceCloseCoordinator()
    sources = _FailOnceCancelSources()
    executor = _SequencedExecutor([])
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=image_series_spec(selected),
                project_root=str(tmp_path),
            )
        ),
        lifecycle=lifecycle,
        sources=sources,
        executor=executor,
    )
    capture = page._pipeline.begin()
    assert type(capture) is StartCapture
    assert lifecycle.phase is RunPhase.PREPARING
    try:
        first = page.close_workspace()
        assert first.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert lifecycle.phase is RunPhase.PREPARING

        terminal = page.close_workspace()
        duplicate = page.close_workspace()
        assert lifecycle.phase is RunPhase.CLOSED
        assert lifecycle.close_calls == 2
        assert executor.close_calls == []
        assert len(sources.cancel_calls) == 2
        assert sources.cancel_calls[0] is capture.request_id
        assert sources.cancel_calls[1] is capture.request_id
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert duplicate is terminal
        assert page._terminal_close is terminal
        assert page._closed is True
        assert [
            (
                failure.operation,
                failure.exception.message,
            )
            for failure in duplicate.recovery_failures
        ] == [
            ("lifecycle.close", "lifecycle close failed once"),
            ("source.cancel", "source cancel failed once"),
        ]
    finally:
        page.deleteLater()
        qapp.processEvents()


def test_lifecycle_close_after_effect_reconciles_exact_terminal_once(
    tmp_path: Path,
) -> None:
    lifecycle = _RaiseAfterCloseCoordinator()
    executor = _SequencedExecutor(["cleaned"])
    qapp, page, lifecycle, identity = _active_page(
        tmp_path, executor, lifecycle=lifecycle
    )
    try:
        terminal = page.close_workspace()
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is identity
        assert lifecycle.phase is RunPhase.CLOSED
        assert lifecycle.close_calls == 1
        assert executor.close_calls == [identity]
        assert [
            (
                failure.operation,
                failure.exception.message,
            )
            for failure in terminal.recovery_failures
        ] == [
            (
                "lifecycle.close",
                "close raised after taking effect",
            )
        ]
        assert page.close_workspace() is terminal
        assert lifecycle.close_calls == 1
        assert executor.close_calls == [identity]
    finally:
        _dispose(page, qapp)


def test_owners_closed_after_effect_reconciles_exact_terminal_once(
    tmp_path: Path,
) -> None:
    lifecycle = _RaiseAfterOwnersCoordinator()
    executor = _SequencedExecutor(["cleaned"])
    qapp, page, lifecycle, identity = _active_page(
        tmp_path, executor, lifecycle=lifecycle
    )
    try:
        terminal = page.close_workspace()
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is identity
        assert lifecycle.phase is RunPhase.CLOSED
        assert lifecycle.owners_closed_calls == [identity]
        assert executor.close_calls == [identity]
        assert page.close_workspace() is terminal
        assert lifecycle.owners_closed_calls == [identity]
        assert executor.close_calls == [identity]
    finally:
        _dispose(page, qapp)


def test_foreign_cleanup_receipt_is_inert_and_retains_exact_owner(
    tmp_path: Path,
) -> None:
    executor = _SequencedExecutor(["pending", "foreign", "cleaned"])
    qapp, page, lifecycle, identity = _active_page(tmp_path, executor)
    try:
        first = page.close_workspace()
        second = page.close_workspace()
        assert lifecycle.phase is RunPhase.STOPPING
        assert first.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert second.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert second.cleanup_identity is identity

        terminal = page.close_workspace()
        assert executor.close_calls == [identity, identity, identity]
        assert lifecycle.phase is RunPhase.CLOSED
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is identity
    finally:
        _dispose(page, qapp)


def test_rejected_owners_closed_retains_composed_pending_owner(
    tmp_path: Path,
) -> None:
    lifecycle = _FailOnceOwnersCoordinator()
    executor = _SequencedExecutor(["pending", "cleaned", "cleaned"])
    qapp, page, lifecycle, identity = _active_page(
        tmp_path, executor, lifecycle=lifecycle
    )
    try:
        first = page.close_workspace()
        second = page.close_workspace()
        assert lifecycle.phase is RunPhase.STOPPING
        assert lifecycle.owners_closed_calls == [identity]
        assert first.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert second.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert second.cleanup_identity is identity

        terminal = page.close_workspace()
        assert lifecycle.owners_closed_calls == [identity, identity]
        assert executor.close_calls == [identity, identity]
        assert lifecycle.phase is RunPhase.CLOSED
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is identity
    finally:
        _dispose(page, qapp)


def test_public_close_retry_does_not_replay_cleaned_substeps(
    tmp_path: Path,
) -> None:
    sources = _CountingSources()
    executor = _SequencedExecutor(["pending", "cleaned"])
    qapp, page, lifecycle, identity = _active_page(
        tmp_path, executor, sources=sources
    )
    try:
        assert sources.cancel_calls == 1
        pending = page.close_workspace()
        terminal = page.close_workspace()
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert pending.cleanup_identity is identity
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is identity
        assert executor.close_calls == [identity, identity]
        assert sources.cancel_calls == 1
        assert executor.cleaned_substeps == {
            "session": 1,
            "sink": 1,
            "source": 1,
            "admission": 1,
        }
        assert lifecycle.phase is RunPhase.CLOSED
    finally:
        _dispose(page, qapp)
