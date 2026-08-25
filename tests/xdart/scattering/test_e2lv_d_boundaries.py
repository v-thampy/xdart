"""Frozen E2-LV-D retirement and hydration-flight redesign oracle."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import gc
from importlib import import_module
from pathlib import Path
from threading import Event, Thread
import time

import numpy as np
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering._admission import (
    admission_for,
    install_admission,
)
from xdart.gui.tabs.scattering import display_runtime
from tests.xdart.scattering.test_e2lv_r4_1_boundaries import (
    _SequencedExecutor,
    _active_page,
    _mounted,
)
from tests.xdart.scattering.test_e2lv_r4_2_boundaries import (
    _closed_display_state,
    _finish_run,
    _integrated_record,
)
from tests.xdart.scattering.test_e3_context_contract import _browse
from xdart.gui.tabs.scattering.acquisition_runtime import AcquisitionRuntime
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import (
    StandardRunExecutor,
    _StandardRun,
)
from xdart.gui.tabs.scattering.browse_values import BrowseCleanupReceipt
from xdart.gui.tabs.scattering.contracts import (
    AdmissionFailure,
    AdmissionReceipt,
    AdmissionReleased,
    AdmissionToken,
    SourceCapture,
    StartCapture,
)
from xdart.gui.tabs.scattering.display_runtime import (
    DetectorHydrationOutcome,
)
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    ExecutorAccepted,
    ExecutorClosed,
    ExecutorStartFailed,
    RequestId,
    RunIdentity,
)
from xdart.modules.display_context import ContextKind, new_context_token
from xdart.modules.frame_publication import FramePublication
from xrd_tools.core import Axis, FrameRecord, FrameView
from xrd_tools.io.image_source import RawFrameResult
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec


def _retirement_api():
    return import_module(
        "xdart.gui.tabs.scattering.display_retirement"
    )


def _proof(identity: RunIdentity | None):
    api = _retirement_api()
    return api.DisplayRetirementReceipt(
        identity, CleanupStatus.CLEANED
    )


def _with_proof(value, identity: RunIdentity | None):
    return replace(value, display_retirement=_proof(identity))


def _wait_qt(
    qapp: QtWidgets.QApplication,
    condition,
    *,
    timeout: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.001)
    assert condition()


def _wait_thread(condition, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert condition()


def _dispose(
    page,
    qapp: QtWidgets.QApplication,
) -> None:
    receipt = page.close_workspace()
    deadline = time.monotonic() + 2.0
    while (
        receipt.cleanup_status is not CleanupStatus.CLEANED
        and time.monotonic() < deadline
    ):
        qapp.processEvents()
        receipt = page.close_workspace()
    page.deleteLater()
    qapp.processEvents()
    # These tests create the extra page lifetimes introduced by the D oracle.
    # Collect their wrapper cycles here, on the GUI thread, so a later
    # admission worker cannot become the first thread to discover them.
    gc.collect()


class _RetiringExecutor(_SequencedExecutor):
    def __init__(
        self,
        retirement_responses: list[str],
        *,
        second_start: str,
        block_retirement: bool = False,
    ) -> None:
        super().__init__([])
        self._retirement_responses = deque(retirement_responses)
        self._second_start = second_start
        self._admission_token: AdmissionToken | None = None
        self._admission_result: object | None = None
        self._admission_worker: Thread | None = None
        self._retirement_owner = None
        self.first_identity: RunIdentity | None = None
        self.second_identity: RunIdentity | None = None
        self.timeline: list[tuple[str, RunIdentity]] = []
        self.retirement_substeps = 0
        self.entered = Event()
        self.release = Event()
        if not block_retirement:
            self.release.set()

    def begin_admission(self, capture):
        token = AdmissionToken(
            capture.request_id, capture.intent_snapshot.revision
        )
        self._admission_token = token
        receipt = admission_for(capture)
        if self.first_identity is None:
            self._admission_result = receipt
            return token
        worker = Thread(
            target=self._retire_for_admission,
            args=(receipt,),
            daemon=True,
        )
        self._admission_worker = worker
        worker.start()
        return token

    def _retire_for_admission(self, receipt: AdmissionReceipt) -> None:
        try:
            owner = self._owner()
            closed = owner.attempt()
            if closed.cleanup_status is CleanupStatus.CLEANED:
                self._admission_result = _with_proof(
                    receipt, self.first_identity
                )
            else:
                self._admission_result = AdmissionFailure(
                    self._admission_token,
                    "Historical display cleanup remains pending.",
                )
        except Exception as error:
            self._admission_result = AdmissionFailure(
                self._admission_token, f"{type(error).__name__}: {error}"
            )

    def _owner(self):
        if self._retirement_owner is None:
            api = _retirement_api()
            self._retirement_owner = api.DisplayRetirementOwner(
                self.first_identity, self._close_first
            )
        return self._retirement_owner

    def _close_first(self) -> ExecutorClosed:
        identity = self.first_identity
        assert identity is not None
        self.close_calls.append(identity)
        self.timeline.append(("close", identity))
        if len(self.close_calls) == 1:
            self.retirement_substeps += 1
        self.entered.set()
        assert self.release.wait(2.0)
        response = self._retirement_responses.popleft()
        return ExecutorClosed(
            identity,
            (
                CleanupStatus.CLEANED
                if response == "cleaned"
                else CleanupStatus.CLEANUP_PENDING
            ),
        )

    def poll_admission(self, token):
        if token is not self._admission_token:
            return None
        return self._admission_result

    def release_admission(self, token):
        if token is not self._admission_token:
            return AdmissionReleased(token, CleanupStatus.CLEANED)
        worker = self._admission_worker
        status = (
            CleanupStatus.CLEANUP_PENDING
            if worker is not None and worker.is_alive()
            else CleanupStatus.CLEANED
        )
        released = AdmissionReleased(token, status)
        owner = self._retirement_owner
        if (
            status is CleanupStatus.CLEANED
            and owner is not None
            and owner.receipt.cleanup_status is CleanupStatus.CLEANED
        ):
            released = _with_proof(released, self.first_identity)
        if status is CleanupStatus.CLEANED:
            self._admission_token = None
            self._admission_result = None
        return released

    cancel_admission = release_admission

    def start(self, configuration, _source, identity, _admission):
        self.start_calls += 1
        self.timeline.append(("start", identity))
        if self.start_calls == 1:
            self.first_identity = identity
            self._remember_context(configuration, identity)
            return ExecutorAccepted(identity)
        self.second_identity = identity
        if self._second_start == "accepted":
            self._remember_context(configuration, identity)
            return ExecutorAccepted(identity)
        return ExecutorStartFailed(
            identity, CleanupStatus.CLEANUP_PENDING
        )

    def close(self, identity):
        if identity is self.first_identity:
            return self._owner().attempt()
        if identity is self.second_identity:
            self.close_calls.append(identity)
            self.timeline.append(("close", identity))
            return ExecutorClosed(identity, CleanupStatus.CLEANED)
        return ExecutorClosed(identity, CleanupStatus.CLEANUP_PENDING)


class _UnprovedRetirementExecutor(_RetiringExecutor):
    def _retire_for_admission(self, receipt: AdmissionReceipt) -> None:
        self._admission_result = receipt


def test_unproved_second_admission_cannot_clear_or_overlap_a(
    tmp_path: Path,
) -> None:
    executor = _UnprovedRetirementExecutor(
        ["cleaned"], second_start="accepted"
    )
    qapp, page, lifecycle, first = _active_page(tmp_path, executor)
    _finish_run(page, lifecycle, first)
    shell, controller = _mounted(page)
    acquisition = controller.acquisition_context
    try:
        shell.run_controls.startButton.click()
        _wait_qt(
            qapp,
            lambda: (
                lifecycle.phase.value == "idle"
                and shell.run_controls.startButton.isEnabled()
            ),
        )
        assert executor.start_calls == 1
        assert executor.second_identity is None
        assert controller.run_identity is first
        assert controller.acquisition_context is acquisition
        assert page._admission is None
    finally:
        _dispose(page, qapp)


def test_failed_b_cannot_overwrite_cleanly_retired_a(
    tmp_path: Path,
) -> None:
    executor = _RetiringExecutor(
        ["cleaned"], second_start="failed"
    )
    qapp, page, lifecycle, first = _active_page(tmp_path, executor)
    _finish_run(page, lifecycle, first)
    shell, controller = _mounted(page)
    try:
        shell.run_controls.startButton.click()
        _wait_qt(
            qapp,
            lambda: lifecycle.phase.value == "failed",
        )
        second = executor.second_identity
        assert second is not None and second is not first
        assert executor.timeline.index(("close", first)) < (
            executor.timeline.index(("start", second))
        )
        assert controller.run_identity is None
        assert controller.acquisition_context is None

        terminal = page.close_workspace()
        assert executor.close_calls == [first, second]
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is second
        assert lifecycle.phase.value == "closed"
    finally:
        _dispose(page, qapp)


def test_successful_b_starts_only_after_exact_a_retirement(
    tmp_path: Path,
) -> None:
    executor = _RetiringExecutor(
        ["cleaned"], second_start="accepted"
    )
    qapp, page, lifecycle, first = _active_page(tmp_path, executor)
    _finish_run(page, lifecycle, first)
    shell, controller = _mounted(page)
    try:
        shell.run_controls.startButton.click()
        _wait_qt(
            qapp,
            lambda: lifecycle.phase.value == "running",
        )
        second = executor.second_identity
        assert second is not None and second is not first
        assert executor.timeline[:3] == [
            ("start", first),
            ("close", first),
            ("start", second),
        ]
        assert controller.run_identity is None
        assert controller.acquisition_context is None
    finally:
        _dispose(page, qapp)


def test_pending_a_retirement_blocks_b_and_retries_same_owner(
    tmp_path: Path,
) -> None:
    executor = _RetiringExecutor(
        ["pending", "cleaned"], second_start="accepted"
    )
    qapp, page, lifecycle, first = _active_page(tmp_path, executor)
    _finish_run(page, lifecycle, first)
    shell, controller = _mounted(page)
    try:
        shell.run_controls.startButton.click()
        _wait_qt(
            qapp,
            lambda: lifecycle.phase.value == "idle"
            and shell.run_controls.startButton.isEnabled(),
        )
        assert executor.start_calls == 1
        assert controller.run_identity is first
        assert executor.close_calls == [first]

        shell.run_controls.startButton.click()
        _wait_qt(
            qapp,
            lambda: lifecycle.phase.value == "running",
        )
        assert executor.start_calls == 2
        assert executor.close_calls == [first, first]
        assert executor.retirement_substeps == 1
        assert controller.run_identity is None
        assert controller.acquisition_context is None
    finally:
        _dispose(page, qapp)


def test_close_during_retirement_joins_one_exact_owner(
    tmp_path: Path,
) -> None:
    executor = _RetiringExecutor(
        ["cleaned"],
        second_start="accepted",
        block_retirement=True,
    )
    qapp, page, lifecycle, first = _active_page(tmp_path, executor)
    _finish_run(page, lifecycle, first)
    shell, _controller = _mounted(page)
    try:
        shell.run_controls.startButton.click()
        assert executor.entered.wait(2.0)

        pending = page.close_workspace()
        assert executor.close_calls == [first]
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert pending.cleanup_identity is first
        executor.release.set()
        worker = executor._admission_worker
        assert worker is not None
        worker.join(2.0)

        terminal = page.close_workspace()
        assert executor.close_calls == [first]
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is first
        assert lifecycle.phase.value == "closed"
    finally:
        executor.release.set()
        _dispose(page, qapp)


def test_executor_cannot_replace_closed_active_without_retirement_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    selected = image_series_spec(tmp_path / "raw_0001.tif")
    configuration = RunIntent(
        source_spec=selected,
        poni_file=str(tmp_path / "calibration.poni"),
        save_path=str(tmp_path / "out.nxs"),
        output_mode="Overwrite",
    ).freeze()
    capture = SourceCapture(RequestId(1), 1, selected)
    identity = RunIdentity.from_configuration(configuration)
    executor = StandardRunExecutor()
    admission = install_admission(
        executor, configuration, capture
    )
    historical = _StandardRun(
        None,
        RunIdentity(99, "historical"),
        None,
        None,
        None,
        None,
        tmp_path / "historical.nxs",
        closed=True,
        cleanup_status=CleanupStatus.CLEANED,
    )
    executor._active = historical

    class _NoopThread:
        ident = None

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return False

        def join(self, *_args, **_kwargs) -> None:
            pass

    monkeypatch.setattr(executor_module, "Thread", _NoopThread)
    result = executor.start(
        configuration, capture, identity, admission
    )

    assert type(result) is ExecutorStartFailed
    assert executor._active is historical


def test_real_pending_retirement_keeps_a_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    selected = image_series_spec(tmp_path / "raw_0001.tif")
    intent = RunIntent(
        source_spec=selected,
        poni_file=str(tmp_path / "calibration.poni"),
        save_path=str(tmp_path / "out.nxs"),
        output_mode="Overwrite",
    )
    snapshot = RunIntentStore(intent).snapshot()
    request = RequestId(1)
    capture = StartCapture(
        request,
        1,
        snapshot,
        SourceCapture(request, 1, selected),
    )
    historical = _StandardRun(
        None,
        RunIdentity(99, "pending-historical"),
        None,
        None,
        None,
        None,
        tmp_path / "historical.nxs",
        closed=True,
    )
    executor = StandardRunExecutor()
    executor._active = historical
    monkeypatch.setattr(
        executor,
        "_close_run",
        lambda identity: ExecutorClosed(
            identity, CleanupStatus.CLEANUP_PENDING
        ),
    )

    token = executor.begin_admission(capture)
    _wait_thread(
        lambda: type(executor.poll_admission(token))
        is AdmissionFailure
    )
    assert executor._active is historical
    assert executor._retirement is not None
    assert executor._retirement.run_identity is historical.identity


def _assert_clean_precontext_failure_needs_no_foreign_retirement_proof(
    tmp_path: Path,
    monkeypatch,
    *,
    armed_live: bool,
) -> None:
    selected = image_series_spec(tmp_path / "raw_0001.tif")
    intent = RunIntent(
        source_spec=selected,
        poni_file=str(tmp_path / "calibration.poni"),
        save_path=str(tmp_path / "failed.nxs"),
        output_mode="Overwrite",
    )
    configuration = intent.freeze()
    identity = RunIdentity.from_configuration(configuration)
    capture = SourceCapture(RequestId(70), 1, selected)
    run = _StandardRun(
        configuration,
        identity,
        None,
        None,
        None,
        None,
        tmp_path / "failed.nxs",
        capture=capture,
    )
    runtime = None
    if armed_live:
        runtime = AcquisitionRuntime()
        runtime._arm_live()
        run.context_runtime = runtime
    executor = StandardRunExecutor()
    executor._active = run
    monkeypatch.setattr(
        executor,
        "_execute_admitted",
        lambda _run: (_ for _ in ()).throw(
            RuntimeError("pre-context construction failed")
        ),
    )

    executor._run(run)

    terminal, = executor.drain_events()
    assert terminal.kind is StandardEventKind.FAILED
    assert terminal.cleanup_status is CleanupStatus.CLEANED
    assert executor._active is run
    assert run.unpublished_display_retired is True
    if runtime is not None:
        assert runtime._live_terminal is True
        assert runtime._live_retired is True
    assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED

    snapshot = RunIntentStore(intent).snapshot()
    request = RequestId(71)
    start = StartCapture(
        request,
        1,
        snapshot,
        SourceCapture(request, 1, selected),
    )
    monkeypatch.setattr(
        executor_module,
        "build_admission_receipt",
        lambda capture, **_kwargs: admission_for(capture),
    )
    token = executor.begin_admission(start)
    _wait_thread(
        lambda: type(executor.poll_admission(token)) is AdmissionReceipt
    )
    admitted = executor.poll_admission(token)
    assert type(admitted) is AdmissionReceipt
    assert executor._active is None
    assert (
        admitted.display_retirement
        is _retirement_api().NO_DISPLAY_RETIREMENT
    )
    assert (
        executor.release_admission(token).cleanup_status
        is CleanupStatus.CLEANED
    )


def test_clean_precontext_failure_needs_no_foreign_retirement_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _assert_clean_precontext_failure_needs_no_foreign_retirement_proof(
        tmp_path,
        monkeypatch,
        armed_live=False,
    )


def test_clean_armed_live_precontext_failure_retires_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _assert_clean_precontext_failure_needs_no_foreign_retirement_proof(
        tmp_path,
        monkeypatch,
        armed_live=True,
    )








def test_cancelled_admission_delivers_exact_clean_retirement_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    historical_identity = RunIdentity(99, "historical")
    historical = _StandardRun(
        None,
        historical_identity,
        None,
        None,
        None,
        None,
        tmp_path / "historical.nxs",
        closed=True,
        cleanup_status=CleanupStatus.CLEANED,
    )
    executor = StandardRunExecutor()
    executor._active = historical
    entered = Event()
    release = Event()

    def blocked_close(identity: RunIdentity) -> ExecutorClosed:
        assert identity is historical_identity
        entered.set()
        assert release.wait(2.0)
        return ExecutorClosed(identity, CleanupStatus.CLEANED)

    monkeypatch.setattr(executor, "_close_run", blocked_close)
    monkeypatch.setattr(
        executor_module,
        "build_admission_receipt",
        lambda capture, **_kwargs: admission_for(capture),
    )
    source = image_series_spec(tmp_path / "raw_0001.tif")
    snapshot = RunIntentStore(
        RunIntent(
            source_spec=source,
            poni_file=str(tmp_path / "calibration.poni"),
            save_path=str(tmp_path / "out.nxs"),
            output_mode="Overwrite",
        )
    ).snapshot()
    request = RequestId(71)
    token = executor.begin_admission(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        )
    )
    assert entered.wait(2.0)

    first = executor.cancel_admission(token)
    assert first.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert first.display_retirement.run_identity is historical_identity
    release.set()
    _wait_thread(
        lambda: (
            executor._admission is not None
            and executor._admission.worker_done
            and executor._admission.retirement_receipt.cleanup_status
            is CleanupStatus.CLEANED
        )
    )

    consumed = executor.release_admission(token)
    assert consumed.cleanup_status is CleanupStatus.CLEANED
    assert consumed.display_retirement.run_identity is historical_identity
    assert (
        consumed.display_retirement.cleanup_status
        is CleanupStatus.CLEANED
    )
    assert executor._admission is None
    assert executor._retirement is None


def test_page_retries_pending_admission_release_without_starting_b(
    tmp_path: Path,
) -> None:
    executor = _RetiringExecutor(
        ["cleaned"],
        second_start="accepted",
        block_retirement=True,
    )
    qapp, page, lifecycle, first = _active_page(tmp_path, executor)
    _finish_run(page, lifecycle, first)
    shell, controller = _mounted(page)
    try:
        shell.run_controls.startButton.click()
        assert executor.entered.wait(2.0)
        token = page._admission
        assert token is not None

        released = page._release_admission(token)
        assert released.cleanup_status is CleanupStatus.CLEANUP_PENDING
        executor.release.set()
        _wait_qt(qapp, lambda: page._admission is None)

        assert controller.run_identity is None
        assert controller.acquisition_context is None
        assert executor.start_calls == 1
    finally:
        executor.release.set()
        _dispose(page, qapp)


def test_page_rejects_malformed_and_equal_foreign_admission_release(
    tmp_path: Path,
    monkeypatch,
) -> None:
    executor = _SequencedExecutor(["cleaned"])
    qapp, page, lifecycle, identity = _active_page(tmp_path, executor)
    _finish_run(page, lifecycle, identity)
    token = AdmissionToken(RequestId(991), 7)
    foreign = AdmissionToken(RequestId(991), 7)
    assert foreign == token and foreign is not token
    responses = iter(
        (
            object(),
            AdmissionReleased(foreign, CleanupStatus.CLEANED),
        )
    )
    page._admission = token
    monkeypatch.setattr(
        executor,
        "release_admission",
        lambda _token: next(responses),
    )
    try:
        for _ in range(2):
            released = page._release_admission(token)
            assert released.token is token
            assert (
                released.cleanup_status
                is CleanupStatus.CLEANUP_PENDING
            )
            assert page._admission is token
            assert page._context_controller.run_identity is identity
    finally:
        page._admission = None
        _dispose(page, qapp)


def test_page_retains_one_shot_retirement_proof_until_browse_is_clean(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.browse_hydration import (
        _BrowseHydrationOwner,
    )

    executor = _SequencedExecutor(["cleaned"])
    qapp, page, lifecycle, first = _active_page(tmp_path, executor)
    _finish_run(page, lifecycle, first)
    shell, controller = _mounted(page)
    request, browse = _browse(
        new_context_token(ContextKind.BROWSE),
        1,
        scan_key="retained-b",
    )
    controller._runtime.adopt_browse(browse, request)
    controller._browse_hydration_owner = _BrowseHydrationOwner(browse)
    browse_statuses = deque(
        (CleanupStatus.CLEANUP_PENDING, CleanupStatus.CLEANED)
    )

    def release_browse(context):
        assert context is browse
        status = browse_statuses.popleft()
        if status is CleanupStatus.CLEANED:
            context.release()
        return BrowseCleanupReceipt(request, status)

    monkeypatch.setattr(
        controller._browse_loader,
        "release_context",
        release_browse,
    )
    token = AdmissionToken(RequestId(992), 8)
    proof = _proof(first)
    release_calls: list[AdmissionToken] = []

    def release_once(candidate):
        assert candidate is token
        release_calls.append(candidate)
        return (
            AdmissionReleased(
                token,
                CleanupStatus.CLEANED,
                display_retirement=proof,
            )
            if len(release_calls) == 1
            else AdmissionReleased(token, CleanupStatus.CLEANED)
        )

    page._admission = token
    monkeypatch.setattr(executor, "release_admission", release_once)
    try:
        first_release = page._release_admission(token)
        assert first_release.cleanup_status is CleanupStatus.CLEANED
        assert page._admission is token
        assert controller.run_identity is first
        assert controller.browse_context is browse

        second_release = page._release_admission(token)
        assert second_release is first_release
        assert release_calls == [token]
        assert page._admission is None
        assert controller.run_identity is None
        assert controller.acquisition_context is None
        assert controller.browse_context is None

        page._refresh_shell()
        shell.run_controls.startButton.click()
        _wait_qt(qapp, lambda: lifecycle.phase.value == "running")
        assert executor.start_calls == 2
    finally:
        _dispose(page, qapp)
