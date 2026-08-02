from __future__ import annotations

from dataclasses import replace
from threading import Barrier, Event, Thread

from xdart.gui.tabs.scattering.acquisition_runtime import AcquisitionRuntime
from xdart.gui.tabs.scattering.adapters.browse_loader import (
    BrowseLoader,
    _BrowseOperation,
)
from xdart.gui.tabs.scattering.browse_values import (
    BrowseCleanupReceipt,
)
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity
from xdart.modules.display_context import BrowseContext

from tests.xdart.scattering.test_e3_context_contract import (
    _browse,
    _running_controller,
)


def test_resume_that_takes_effect_then_raises_is_not_failure_atomic():
    """A real ScanSession can resume before a BaseException escapes its callback."""

    runtime = AcquisitionRuntime()

    class Session:
        paused = True
        resume_calls = 0

        def resume(self):
            self.resume_calls += 1
            self.paused = False
            raise KeyboardInterrupt("post-resume callback poison")

        def submit(self, _frame):
            if self.paused:
                raise RuntimeError("still paused")
            return True

    session = Session()
    try:
        runtime.resume(session)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("the post-effect failure must propagate")

    # The executor session is live, while AcquisitionRuntime's submission
    # gate remains cleared because it is set only after session.resume returns.
    assert session.paused is True


def test_foreign_loader_close_receipt_cannot_close_pending_exact_request():
    controller, _, _, loader, _ = _running_controller()
    controller.pause()
    request = controller.begin_browse("/processed/pending.nxs")
    foreign = replace(request)
    assert foreign == request and foreign is not request
    loader.close = lambda _expected=None: BrowseCleanupReceipt(
        foreign, CleanupStatus.CLEANED
    )

    result = controller.close()
    assert result.request is request
    assert result.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert controller.close() is not result


def test_terminal_context_cleanup_is_single_flight(monkeypatch):
    token = "browse-review"
    request, context = _browse(token, 1)
    loader = BrowseLoader()
    operation = _BrowseOperation(request, Event())
    operation.cancelled.set()
    operation.context = context
    operation.terminal = True
    loader._active = operation

    entered = Barrier(2)
    release = Event()
    calls = []
    original = BrowseContext.release

    def blocked_release(self):
        if self is context:
            calls.append(self)
            entered.wait(timeout=2)
            release.wait(timeout=2)
        return original(self)

    monkeypatch.setattr(BrowseContext, "release", blocked_release)
    workers = [
        Thread(
            target=loader._progress,
            kwargs={"retire_cancelled": True},
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    entered.wait(timeout=2)
    release.set()
    for worker in workers:
        worker.join(timeout=2)

    assert calls == [context]


def test_stop_during_ready_browse_retains_failed_context_release():
    controller, lifecycle, _, loader, _ = _running_controller()
    controller.pause()
    request = controller.begin_browse("/processed/late.nxs")
    _, context = _browse(
        request.token,
        request.load_generation,
        request=request,
    )
    loader.complete(context)
    loader.cancel = lambda _expected: BrowseCleanupReceipt(
        request, CleanupStatus.CLEANUP_PENDING
    )
    controller.stop()
    assert lifecycle.phase.value == "stopping"
    loader.close = lambda _expected=None: BrowseCleanupReceipt(
        request,
        CleanupStatus.CLEANUP_PENDING,
    )

    assert controller.poll_browse() is None
    pending = controller.close()
    assert pending.request is request
    assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert context.released is False
