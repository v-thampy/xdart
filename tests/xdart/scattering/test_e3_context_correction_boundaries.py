"""Frozen E3-C.R operation, cleanup, and qualification discriminators."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event
import time

import pytest

from xdart.gui.tabs.scattering.adapters import browse_loader as loader_module
from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
from xdart.gui.tabs.scattering.adapters.run_executor import (
    StandardRunExecutor,
    _StandardRun,
)
from xdart.gui.tabs.scattering.acquisition_runtime import AcquisitionRuntime
from xdart.gui.tabs.scattering.context_controller import ContextController
from xdart.gui.tabs.scattering.context_projection import (
    ContextProjection,
    ProjectionRequest,
)
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    DurablePaused,
    PauseRequested,
    ResumeRequested,
    RunIdentity,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.modules.display_context import BrowseContext
from xdart.modules.frame_publication import PublicationStore
from xrd_tools.core import FrameRecord
from xrd_tools.session.frame_record_store import FrameRecordStore

from tests.xdart.scattering.test_e3_context_contract import (
    _browse,
    _configuration,
    _current_key,
    _running_controller,
    _select_browse,
    _view,
)


def _wait_for(predicate, *, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value is not None:
            return value
        time.sleep(0.002)
    raise AssertionError("expected terminal browse result")


def _real_loader_controller(
    monkeypatch,
    *,
    read_records,
    join_timeout: float = 0.05,
):
    monkeypatch.setattr(loader_module, "read_provenance", lambda _path: {})
    _, lifecycle, executor, _, acquisition = _running_controller()
    loader = BrowseLoader(
        join_timeout=join_timeout,
        open_scan=lambda _path: object(),
        read_records=read_records,
    )
    controller = ContextController(
        lifecycle=lifecycle,
        executor=executor,
        browse_loader=loader,
        projection=ContextProjection(),
    )
    controller.adopt_acquisition(executor.identity)
    controller.pause()
    return controller, lifecycle, executor, loader, acquisition


def _one_record(_path):
    yield FrameRecord.from_view(_view(1, 5.0))


@pytest.mark.parametrize("equal_distinct", (False, True))
def test_browse_projection_requires_exact_accepted_run_identity(
    equal_distinct: bool,
):
    controller, _, _, loader, _ = _running_controller()
    controller.pause()
    _select_browse(controller, loader)
    accepted = controller.project_request(
        _current_key(controller))
    identity = (
        replace(accepted.run_identity)
        if equal_distinct
        else RunIdentity(
            accepted.run_identity.generation + 1,
            accepted.run_identity.fingerprint + "-foreign",
        )
    )
    assert identity is not accepted.run_identity
    request = ProjectionRequest(
        identity,
        accepted.selection,
        accepted.frame,
    )
    assert controller.resolve_projection(request) is None


def test_controller_rejects_foreign_identity_before_projection_port():
    controller, _, _, loader, _ = _running_controller()
    controller.pause()
    _select_browse(controller, loader)
    accepted = controller.project_request(
        _current_key(controller))
    foreign = replace(accepted.run_identity)
    calls = []

    class PermissiveProjection:
        def project(self, *args):
            calls.append(args)
            return object()

    controller._projection = PermissiveProjection()
    request = ProjectionRequest(
        foreign,
        accepted.selection,
        accepted.frame,
    )
    assert controller.resolve_projection(request) is None
    assert calls == []


def test_projection_port_independently_rejects_foreign_browse_identity():
    controller, _, _, loader, _ = _running_controller()
    controller.pause()
    _select_browse(controller, loader)
    accepted = controller.project_request(
        _current_key(controller))
    foreign = replace(accepted.run_identity)
    request = ProjectionRequest(
        foreign,
        accepted.selection,
        accepted.frame,
    )
    assert controller._projection.project(
        controller.browse_context,
        request,
        controller.selection,
        accepted.run_identity,
        controller.frame_keys,
    ) is None


def test_pause_exception_is_typed_failure_atomic_and_retryable():
    controller, lifecycle, executor, _, _ = _running_controller()
    accepted = executor.pause
    attempts = 0

    def fail_once(identity):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("pause exploded")
        return accepted(identity)

    executor.pause = fail_once
    failed = controller.pause()
    assert type(failed).__name__ == "PauseFailed"
    assert failed.run_identity is executor.identity
    assert failed.diagnostic.operation == "context.pause"
    assert failed.diagnostic.message == "pause exploded"
    assert lifecycle.phase is RunPhase.RUNNING
    paused = controller.pause()
    assert paused.run_identity is executor.identity
    assert lifecycle.phase is RunPhase.PAUSED


def test_acquisition_runtime_restores_gate_and_session_on_pause_exception():
    identity = RunIdentity(1, "runtime")
    runtime = AcquisitionRuntime()

    class Session:
        resumes = 0
        submits = 0

        def pause(self, *, timeout):
            raise RuntimeError(f"pause failed at {timeout}")

        def resume(self):
            self.resumes += 1

        def submit(self, _frame):
            self.submits += 1
            return True

    session = Session()
    with pytest.raises(RuntimeError, match="pause failed"):
        runtime.pause(session, identity, 0.01)
    assert session.resumes == 1
    assert runtime.submit(session, object()) is True
    assert session.submits == 1


@pytest.mark.parametrize("phase", ("pausing", "resuming"))
def test_stop_remains_contained_during_command_transition(phase: str):
    controller, lifecycle, executor, _, _ = _running_controller()
    identity = executor.identity
    assert lifecycle.pause_requested(
        PauseRequested(identity)
    ).phase is RunPhase.PAUSING
    if phase == "resuming":
        assert lifecycle.durable_paused(
            DurablePaused(identity, 1)
        ).phase is RunPhase.PAUSED
        assert lifecycle.resume_requested(
            ResumeRequested(identity)
        ).phase is RunPhase.RESUMING
    result = controller.stop()
    assert result.phase is RunPhase.STOPPING
    assert executor.stops == [identity]


def test_resume_exception_preserves_exact_b_and_is_retryable():
    controller, lifecycle, executor, loader, acquisition = (
        _running_controller()
    )
    controller.pause()
    _, browse = _select_browse(controller, loader)
    before = controller.selection
    accepted = executor.resume
    attempts = 0

    def fail_once(identity):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("resume exploded")
        return accepted(identity)

    executor.resume = fail_once
    failed = controller.resume()
    assert type(failed).__name__ == "ResumeFailed"
    assert failed.run_identity is executor.identity
    assert failed.diagnostic.operation == "context.resume"
    assert failed.diagnostic.message == "resume exploded"
    assert lifecycle.phase is RunPhase.PAUSED
    assert controller.selection is before
    assert controller.selection.names(browse)
    assert browse.invalidated is False
    selection = controller.resume()
    assert selection.names(acquisition)
    assert lifecycle.phase is RunPhase.RUNNING


def test_pending_b_is_cancelled_then_exact_c_starts_automatically(
    monkeypatch,
    tmp_path: Path,
):
    entered_b = Event()
    release_b = Event()
    path_b = tmp_path / "browse-b.nxs"
    path_c = tmp_path / "browse-c.nxs"
    path_b.write_bytes(b"b")
    path_c.write_bytes(b"c")

    def records(path):
        if Path(path) == path_b:
            entered_b.set()
            release_b.wait(timeout=2.0)
            return
        yield FrameRecord.from_view(_view(1, 7.0))

    controller, _, _, loader, _ = _real_loader_controller(
        monkeypatch, read_records=records
    )
    request_b = controller.begin_browse(str(path_b))
    assert entered_b.wait(timeout=1.0)
    request_c = controller.begin_browse(str(path_c))
    assert request_c is not request_b
    assert loader.poll(request_c) is None
    release_b.set()
    outcome = _wait_for(controller.poll_browse)
    assert outcome.request is request_c
    assert controller.browse_context is not None
    assert controller.browse_context.load_request is request_c
    assert controller.browse_context.requested_path == str(path_c)


def test_close_fail_once_release_is_truthful_exact_and_retryable(
    monkeypatch,
    tmp_path: Path,
):
    path = tmp_path / "browse.nxs"
    path.write_bytes(b"b")
    controller, _, _, loader, _ = _real_loader_controller(
        monkeypatch, read_records=_one_record
    )
    request = controller.begin_browse(str(path))
    _wait_for(controller.poll_browse)
    browse = controller.browse_context
    assert browse is not None
    original = BrowseContext.release
    attempts = 0

    def fail_once(self):
        nonlocal attempts
        if self is browse:
            attempts += 1
            if attempts == 1:
                raise RuntimeError("release exploded")
        return original(self)

    monkeypatch.setattr(BrowseContext, "release", fail_once)
    pending = controller.close()
    assert pending.request is request
    assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert controller.browse_context is browse
    cleaned = controller.close()
    assert cleaned.request is request
    assert cleaned.cleanup_status is CleanupStatus.CLEANED
    assert browse.released is True
    assert controller.close() is cleaned


def test_persistent_release_failure_remains_visible_and_owned(
    monkeypatch,
    tmp_path: Path,
):
    path = tmp_path / "browse.nxs"
    path.write_bytes(b"b")
    controller, _, _, _, _ = _real_loader_controller(
        monkeypatch, read_records=_one_record
    )
    request = controller.begin_browse(str(path))
    _wait_for(controller.poll_browse)
    browse = controller.browse_context
    assert browse is not None
    original = BrowseContext.release

    def always_fail(self):
        if self is browse:
            raise RuntimeError("still retained")
        return original(self)

    monkeypatch.setattr(BrowseContext, "release", always_fail)
    first = controller.close()
    second = controller.close()
    assert first.request is second.request is request
    assert first.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert second.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert first.cleanup_failures[-1].message == "still retained"
    assert second.cleanup_failures[-1].message == "still retained"
    assert controller.browse_context is browse
    assert browse.released is False


def test_close_live_cancelled_worker_retains_retry_identity(
    monkeypatch,
    tmp_path: Path,
):
    entered = Event()
    release = Event()
    path = tmp_path / "pending.nxs"
    path.write_bytes(b"pending")

    def records(_path):
        entered.set()
        release.wait(timeout=2.0)
        return iter(())

    controller, _, _, loader, acquisition = _real_loader_controller(
        monkeypatch,
        read_records=records,
        join_timeout=0.001,
    )
    request = controller.begin_browse(str(path))
    assert entered.wait(timeout=1.0)
    pending = controller.close()
    assert pending.request is request
    assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert acquisition.record_store is not None
    assert acquisition.publication_store is not None
    release.set()
    worker = loader._worker
    assert worker is not None
    worker.join(timeout=1.0)
    cleaned = controller.close()
    assert cleaned.request is request
    assert cleaned.cleanup_status is CleanupStatus.CLEANED
    assert controller.close() is cleaned


@pytest.mark.parametrize("seam", ("open", "read"))
def test_baseexception_at_browse_worker_boundary_has_one_terminal_outcome(
    monkeypatch,
    tmp_path: Path,
    seam: str,
):
    class WorkerPoison(BaseException):
        pass

    path = tmp_path / "poison.nxs"
    path.write_bytes(b"poison")
    monkeypatch.setattr(loader_module, "read_provenance", lambda _path: {})

    def fail_open(_path):
        raise WorkerPoison("open poison")

    def fail_read(_path):
        raise WorkerPoison("read poison")
        yield

    loader = BrowseLoader(
        open_scan=fail_open if seam == "open" else lambda _path: object(),
        read_records=fail_read if seam == "read" else _one_record,
    )
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest
    from xdart.modules.display_context import ContextKind, new_context_token

    request = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE), 1, str(path)
    )
    loader.begin(request)
    outcome = _wait_for(lambda: loader.poll(request))
    assert outcome.request is request
    assert outcome.status.value == "failed"
    assert outcome.detail == f"{seam} poison"
    assert loader.poll(request) is outcome
    assert loader.consume(outcome) is None
    assert loader.consume(outcome) is None


def test_two_artifact_adoption_moves_exact_current_display_scan():
    configuration = _configuration()
    identity = RunIdentity.from_configuration(configuration)
    executor = StandardRunExecutor()
    scan_a = object()
    scan_b = object()
    run = _StandardRun(
        configuration,
        identity,
        scan_a,
        None,
        None,
        None,
        Path(configuration.save_path),
    )
    run.display.set_factories(FrameRecordStore, PublicationStore)
    run.display.configure(partition_count=2, npt=2, frame_bytes=48)
    owner_a = run.display.add_artifact(
        Path("/out/a.nxs"),
        "run.a",
        mask=None,
        mask_saturation=True,
        measurement_mode="Standard",
    )
    context = executor._adopt_acquisition_context(
        run, owner_a, source_path="/data/a_0001.tif"
    )
    epoch_a = context.commit_epoch
    owner_b = run.display.add_artifact(
        Path("/out/b.nxs"),
        "run.b",
        mask=None,
        mask_saturation=True,
        measurement_mode="Standard",
    )
    run.scan = scan_b
    assert executor._adopt_acquisition_context(
        run, owner_b, source_path="/data/b_0001.tif"
    ) is context
    bindings = context.display_bindings()
    assert context.scan is scan_a
    assert bindings.scan is scan_b
    assert context.scan_key == "run.b"
    assert context.source == "/data/b_0001.tif"
    assert context.commit_epoch == epoch_a + 1
    assert bindings.record_store is run.display
    assert bindings.publication_store is run.display


def test_foreign_cleanup_receipt_cannot_complete_exact_b(
    monkeypatch,
):
    from xdart.gui.tabs.scattering import browse_values

    receipt_type = browse_values.BrowseCleanupReceipt
    controller, _, _, loader, _ = _running_controller()
    controller.pause()
    request, browse = _select_browse(controller, loader)
    foreign = replace(request)
    assert foreign == request and foreign is not request
    loader.release_context = lambda _context: receipt_type(
        foreign, CleanupStatus.CLEANED
    )
    result = controller.close()
    assert result.request is request
    assert result.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert controller.browse_context is browse
