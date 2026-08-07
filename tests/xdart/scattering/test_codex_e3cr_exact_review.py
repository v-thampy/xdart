"""Independent exact-object adversaries for E3-C.R at 93fae07f."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import pytest

from tests.xdart.scattering.test_e3_context_contract import _view
from tests.xdart.scattering.test_e3_context_correction_boundaries import (
    _real_loader_controller,
    _wait_for,
)
from xdart.gui.tabs.scattering import adapters
from xdart.gui.tabs.scattering.acquisition_runtime import (
    AcquisitionRuntime,
    CommandCompensationFailure,
)
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.modules.display_context import (
    AcquisitionContext,
    BrowseContext,
    CommitGate,
    ContextKind,
    new_context_token,
)
from xrd_tools.core import FrameRecord


def _context(scan: object) -> AcquisitionContext:
    return AcquisitionContext(
        context_token=new_context_token(ContextKind.ACQUISITION),
        run_configuration=object(),
        config_generation=1,
        config_fingerprint="accepted",
        run_scan_key="scan.a",
        source_path="/data/a.nxs",
        scan=scan,
        frame=None,
        frame_ids=[],
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store={},
    )


def _one_record(_path):
    yield FrameRecord.from_view(_view(1, 5.0))


def test_rescope_selection_and_epoch_are_not_observable_partially(
    monkeypatch,
):
    admitted = object()
    current = object()
    context = _context(admitted)
    old = (
        context.scan_key,
        context.source,
        context.current_display_scan,
        context.commit_epoch,
    )
    entered = Event()
    release = Event()
    original_advance = CommitGate.advance

    def blocked_advance(self):
        entered.set()
        assert release.wait(timeout=2.0)
        return original_advance(self)

    monkeypatch.setattr(CommitGate, "advance", blocked_advance)
    worker = Thread(
        target=context.rescope_to,
        args=("scan.b", "/data/b.nxs", current),
    )
    worker.start()
    assert entered.wait(timeout=1.0)
    observed = (
        context.scan_key,
        context.source,
        context.current_display_scan,
        context.commit_epoch,
    )
    release.set()
    worker.join(timeout=1.0)
    new = (
        context.scan_key,
        context.source,
        context.current_display_scan,
        context.commit_epoch,
    )

    assert observed in {old, new}


def test_old_epoch_can_commit_after_new_scope_fields_are_visible(monkeypatch):
    context = _context(object())
    old_epoch = context.commit_epoch
    entered = Event()
    release = Event()
    original_advance = CommitGate.advance

    def blocked_advance(self):
        entered.set()
        assert release.wait(timeout=2.0)
        return original_advance(self)

    monkeypatch.setattr(CommitGate, "advance", blocked_advance)
    worker = Thread(
        target=context.rescope_to,
        args=("scan.b", "/data/b.nxs", object()),
    )
    worker.start()
    assert entered.wait(timeout=1.0)
    assert context.scan_key == "scan.b"
    assert context.source == "/data/b.nxs"

    # A request minted for the old scope can still take the commit window
    # after the new scope's mutable fields are already public.
    admitted = context.commit_gate.enter(old_epoch)
    if admitted:
        context.commit_gate.leave()
    release.set()
    worker.join(timeout=1.0)
    assert admitted is False


def test_pause_compensation_failure_does_not_reopen_submission_gate():
    runtime = AcquisitionRuntime()

    class Session:
        def pause(self, *, timeout):
            raise RuntimeError("pause failed after stopping")

        def resume(self):
            raise RuntimeError("resume compensation failed")

    with pytest.raises(RuntimeError, match="pause failed after stopping"):
        runtime.pause(Session(), object(), 0.01)

    assert runtime._gate.is_set() is False


def test_projection_pause_and_failed_compensation_retain_both_exact_causes():
    runtime = AcquisitionRuntime()
    primary = RuntimeError("queued projection failed")
    recovery = RuntimeError("resume after projection failed")

    class Session:
        def pause(self, *, timeout):
            return True

        def resume(self):
            raise recovery

    def fail_projection(_timeout: float) -> bool:
        raise primary

    with pytest.raises(CommandCompensationFailure) as captured:
        runtime.pause(
            Session(),
            object(),
            0.01,
            drain_projection=fail_projection,
        )

    assert tuple(item.message for item in captured.value.diagnostics) == (
        "queued projection failed",
        "resume after projection failed",
    )
    assert tuple(item.operation for item in captured.value.diagnostics) == (
        "context.pause",
        "context.pause.compensation",
    )
    assert runtime._gate.is_set() is False


def test_rejected_ready_browse_retains_failed_cleanup_owner(
    monkeypatch,
    tmp_path: Path,
):
    path = tmp_path / "browse.nxs"
    path.write_bytes(b"browse")
    controller, _, _, loader, _ = _real_loader_controller(
        monkeypatch, read_records=_one_record
    )
    request = controller.begin_browse(str(path))
    outcome = _wait_for(lambda: loader.poll(request))
    assert outcome.request is request
    context = loader._context
    assert type(context) is BrowseContext

    # Make the otherwise-ready outcome inadmissible before the controller
    # consumes it, as happens when Stop wins the race.  Install the failure
    # before Stop because Stop now cancels and retires the ready operation
    # synchronously through the loader.
    original = BrowseContext.release

    def fail_exact(self):
        if self is context:
            raise RuntimeError("release remains pending")
        return original(self)

    monkeypatch.setattr(BrowseContext, "release", fail_exact)
    controller.stop()
    assert controller.poll_browse() is None
    assert context.released is False
    assert controller.browse_context is context or loader._context is context


def test_close_keeps_one_identity_across_active_and_queued_requests(
    monkeypatch,
    tmp_path: Path,
):
    from xdart.gui.tabs.scattering.adapters import browse_loader as loader_module

    entered = Event()
    release = Event()
    path_b = tmp_path / "b.nxs"
    path_c = tmp_path / "c.nxs"
    path_b.write_bytes(b"b")
    path_c.write_bytes(b"c")

    def records(path):
        if Path(path) == path_b:
            entered.set()
            release.wait(timeout=2.0)
            return
        yield FrameRecord.from_view(_view(1, 7.0))

    controller, _, _, loader, _ = _real_loader_controller(
        monkeypatch,
        read_records=records,
        join_timeout=0.001,
    )
    request_b = controller.begin_browse(str(path_b))
    assert entered.wait(timeout=1.0)
    request_c = controller.begin_browse(str(path_c))
    assert request_c is not request_b

    pending = controller.close()
    assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
    release.set()
    worker = loader._worker
    assert worker is not None
    worker.join(timeout=1.0)
    cleaned = controller.close()
    assert cleaned.cleanup_status is CleanupStatus.CLEANED
    assert cleaned.request is pending.request


def test_thread_construction_failure_does_not_leave_released_b_selected(
    monkeypatch,
    tmp_path: Path,
):
    from xdart.gui.tabs.scattering.adapters import browse_loader as loader_module

    first = tmp_path / "first.nxs"
    second = tmp_path / "second.nxs"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    controller, _, _, _, acquisition = _real_loader_controller(
        monkeypatch, read_records=_one_record
    )
    controller.begin_browse(str(first))
    _wait_for(controller.poll_browse)
    browse = controller.browse_context
    assert type(browse) is BrowseContext
    assert controller.selection.names(browse)

    class ThreadPoison(BaseException):
        pass

    def fail_thread(*_args, **_kwargs):
        raise ThreadPoison("construction failed")

    monkeypatch.setattr(loader_module, "Thread", fail_thread)
    with pytest.raises(ThreadPoison, match="construction failed"):
        controller.begin_browse(str(second))

    assert not browse.released or controller.selection.names(acquisition)
