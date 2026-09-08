"""Frozen oracle for atomic Browse B-to-C public presentation."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from shutil import copy2
from threading import Event, Lock, Thread
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore

from xdart.gui.tabs.scattering.adapters import browse_loader as loader_module
from xdart.gui.tabs.scattering.adapters.browse_loader import (
    BrowseLoader,
    _BrowseOperation,
)
from xdart.gui.tabs.scattering.browse_values import (
    BrowseCleanupReceipt,
    BrowseLoadOutcome,
    BrowseLoadRequest,
    BrowseLoadStatus,
)
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.shell_values import (
    ShellCommand,
    ShellCommandKind,
    ShellProjection,
)
from xdart.modules.display_context import BrowseContext, ContextKind
from xdart.modules.display_context import new_context_token
from xrd_tools.io import FrameScalarCatalog

from tests.xdart.scattering.test_e3_context_contract import (
    _api,
    _browse,
    _current_key,
    _running_controller,
    _select_browse,
)
from tests.xdart.scattering.test_e3_join_oracle import (
    _mount,
    _pause,
    _produce_browse_artifact,
    _run,
    _wait,
)


class _EmptyScalarReader:
    def __init__(self, source, *, resolve_source, callback=None):
        assert resolve_source is False
        self._path = str(Path(source).resolve())
        self._callback = callback

    def __enter__(self):
        return self

    def read_scalar_catalog(self, *, cancelled):
        if self._callback is not None:
            self._callback(self._path, cancelled)
        if cancelled():
            raise InterruptedError("Browse scalar catalog read cancelled")
        return FrameScalarCatalog(self._path, "entry", ())

    def __exit__(self, _exc_type, _exc, _tb):
        return None


def _empty_reader_factory(callback=None):
    def open_reader(source, *, resolve_source):
        return _EmptyScalarReader(
            source,
            resolve_source=resolve_source,
            callback=callback,
        )

    return open_reader


def _admit_fake_browse(monkeypatch):
    monkeypatch.setattr(
        loader_module,
        "canonical_browse_scan_key",
        lambda source: Path(source).stem,
    )


def _begin_replacement():
    controller, _, executor, loader, acquisition = _running_controller()
    controller.pause()
    _, browse_b = _select_browse(
        controller, loader, scan_key="browse.b"
    )
    selection_b = controller.selection
    navigation_b = controller.navigation
    frame_b = _current_key(controller)
    records_b = browse_b.record_store
    publications_b = browse_b.publication_store
    request_c = controller.begin_browse("/processed/browse.c.nxs")
    return (
        controller,
        executor,
        loader,
        acquisition,
        browse_b,
        records_b,
        publications_b,
        selection_b,
        navigation_b,
        frame_b,
        request_c,
    )


def _assert_values_only_hold(
    controller,
    *,
    acquisition,
    browse_b,
    records_b,
    publications_b,
    request_c,
    selection_b,
    navigation_b,
    frame_b,
) -> object:
    assert browse_b.released is True
    assert browse_b.invalidated is True
    assert browse_b.commit_gate.cancelled is True
    assert len(records_b) == 0
    assert len(publications_b) == 0
    assert len(browse_b.frame_ids) == 0
    assert len(browse_b.frames) == 0
    assert len(browse_b.viewer_rows_1d) == 0
    assert len(browse_b.viewer_rows_2d) == 0
    assert controller.browse_context is None
    assert controller.retained_contexts == (acquisition,)
    assert controller.projectable_contexts == (acquisition,)
    assert controller.selection is selection_b
    assert controller.navigation is navigation_b
    assert controller.navigation.current is frame_b
    assert controller.frame_keys is navigation_b.frames
    assert controller.resident_frame_keys == frozenset()
    assert controller.project_navigation() == ()
    assert controller.owns_frame(frame_b) is False
    assert controller.select_navigation(frame_b, (frame_b,)) is False
    with pytest.raises(RuntimeError, match="no display selection"):
        controller.project_request(frame_b)

    candidates = [
        value
        for value in vars(controller._runtime).values()
        if is_dataclass(value)
        and any(
            getattr(value, field.name) is request_c
            for field in fields(value)
        )
    ]
    assert len(candidates) == 1
    pending = candidates[0]
    assert type(pending).__dataclass_params__.frozen is True
    assert not hasattr(pending, "__dict__")
    assert tuple(field.name for field in fields(pending)) == (
        "request",
        "selection",
        "navigation",
    )
    assert pending.request is request_c
    assert pending.selection is selection_b
    assert pending.navigation is navigation_b
    assert not any(
        isinstance(value, BrowseContext) or callable(value)
        for value in (
            pending.request,
            pending.selection,
            pending.navigation,
        )
    )
    return pending


@pytest.mark.parametrize(
    "status",
    (
        BrowseLoadStatus.REFUSED,
        BrowseLoadStatus.FAILED,
        BrowseLoadStatus.CANCELLED,
    ),
)
def test_exact_c_terminal_removes_hold_and_selects_a_once(status) -> None:
    (
        controller,
        _executor,
        loader,
        acquisition,
        browse_b,
        records_b,
        publications_b,
        selection_b,
        navigation_b,
        frame_b,
        request_c,
    ) = _begin_replacement()
    _assert_values_only_hold(
        controller,
        acquisition=acquisition,
        browse_b=browse_b,
        records_b=records_b,
        publications_b=publications_b,
        request_c=request_c,
        selection_b=selection_b,
        navigation_b=navigation_b,
        frame_b=frame_b,
    )
    outcome = BrowseLoadOutcome(request_c, status, "terminal C")
    loader.outcome = outcome

    assert controller.poll_browse() is outcome
    selection_a = controller.selection
    assert selection_a.names(acquisition)
    assert selection_a.display_generation == (
        selection_b.display_generation + 1
    )
    assert controller.poll_browse() is None
    assert controller.selection is selection_a


@pytest.mark.parametrize("clone_outcome", (False, True))
def test_foreign_or_cloned_c_terminal_outcome_is_inert(
    clone_outcome: bool,
) -> None:
    (
        controller,
        _executor,
        loader,
        acquisition,
        browse_b,
        records_b,
        publications_b,
        selection_b,
        navigation_b,
        frame_b,
        request_c,
    ) = _begin_replacement()
    pending = _assert_values_only_hold(
        controller,
        acquisition=acquisition,
        browse_b=browse_b,
        records_b=records_b,
        publications_b=publications_b,
        request_c=request_c,
        selection_b=selection_b,
        navigation_b=navigation_b,
        frame_b=frame_b,
    )
    canonical = BrowseLoadOutcome(
        request_c, BrowseLoadStatus.FAILED, "canonical"
    )
    loader.outcome = canonical
    request = request_c if clone_outcome else replace(request_c)
    foreign = BrowseLoadOutcome(request, BrowseLoadStatus.FAILED, "foreign")
    loader.poll = (
        lambda request: foreign if request is request_c else None
    )

    assert controller.poll_browse() is None
    assert controller._runtime._pending_replacement is pending
    assert controller._browse_request is request_c
    assert controller.selection is selection_b
    assert controller.navigation is navigation_b


@pytest.mark.parametrize("command", ("resume", "stop"))
def test_resume_or_stop_removes_hold_and_cancels_exact_c(command: str) -> None:
    (
        controller,
        executor,
        loader,
        acquisition,
        browse_b,
        records_b,
        publications_b,
        selection_b,
        navigation_b,
        frame_b,
        request_c,
    ) = _begin_replacement()
    _assert_values_only_hold(
        controller,
        acquisition=acquisition,
        browse_b=browse_b,
        records_b=records_b,
        publications_b=publications_b,
        request_c=request_c,
        selection_b=selection_b,
        navigation_b=navigation_b,
        frame_b=frame_b,
    )

    getattr(controller, command)()

    assert loader.cancelled == [request_c]
    assert controller.browse_pending is False
    assert controller.selection.names(acquisition)
    assert controller.selection.display_generation == (
        selection_b.display_generation + 1
    )
    if command == "resume":
        assert executor.resumes == [executor.identity]
    else:
        assert executor.stops == [executor.identity]


def test_close_waits_for_exact_cleanup_then_caches_duplicate_receipt() -> None:
    (
        controller,
        _executor,
        loader,
        acquisition,
        browse_b,
        records_b,
        publications_b,
        selection_b,
        navigation_b,
        frame_b,
        request_c,
    ) = _begin_replacement()
    pending = _assert_values_only_hold(
        controller,
        acquisition=acquisition,
        browse_b=browse_b,
        records_b=records_b,
        publications_b=publications_b,
        request_c=request_c,
        selection_b=selection_b,
        navigation_b=navigation_b,
        frame_b=frame_b,
    )
    exact_close = loader.close
    attempts = 0

    def close_pending_once(expected=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return BrowseCleanupReceipt(
                request_c, CleanupStatus.CLEANUP_PENDING
            )
        return exact_close(expected)

    loader.close = close_pending_once
    first = controller.close()
    assert first.request is request_c
    assert first.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert controller._runtime._pending_replacement is pending
    assert controller.selection is selection_b
    loader.outcome = BrowseLoadOutcome(
        request_c, BrowseLoadStatus.FAILED, "late after Close"
    )
    assert controller.poll_browse() is None
    assert controller._runtime._pending_replacement is pending

    cleaned = controller.close()
    assert cleaned.request is request_c
    assert cleaned.cleanup_status is CleanupStatus.CLEANED
    assert controller.selection is None
    assert controller.close() is cleaned
    assert loader.released.count(browse_b) == 1


def test_foreign_previous_generation_close_receipt_is_inert() -> None:
    (
        controller,
        _executor,
        loader,
        acquisition,
        browse_b,
        records_b,
        publications_b,
        selection_b,
        navigation_b,
        frame_b,
        request_c,
    ) = _begin_replacement()
    pending = _assert_values_only_hold(
        controller,
        acquisition=acquisition,
        browse_b=browse_b,
        records_b=records_b,
        publications_b=publications_b,
        request_c=request_c,
        selection_b=selection_b,
        navigation_b=navigation_b,
        frame_b=frame_b,
    )
    _, _, browse_values = _api()
    foreign = browse_values.BrowseLoadRequest(
        "browse-foreign",
        request_c.load_generation - 1,
        "/processed/foreign.nxs",
    )
    exact_close = loader.close
    loader.close = lambda expected=None: BrowseCleanupReceipt(
        foreign, CleanupStatus.CLEANED
    )

    refused = controller.close()
    assert refused.request is request_c
    assert refused.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert controller._runtime._pending_replacement is pending
    assert controller.selection is selection_b

    loader.close = exact_close
    cleaned = controller.close()
    assert cleaned.request is request_c
    assert cleaned.cleanup_status is CleanupStatus.CLEANED


def test_exact_ready_c_replaces_hold_without_intermediate_public_state(
    monkeypatch,
) -> None:
    (
        controller,
        _executor,
        loader,
        acquisition,
        browse_b,
        records_b,
        publications_b,
        selection_b,
        navigation_b,
        frame_b,
        request_c,
    ) = _begin_replacement()
    _assert_values_only_hold(
        controller,
        acquisition=acquisition,
        browse_b=browse_b,
        records_b=records_b,
        publications_b=publications_b,
        request_c=request_c,
        selection_b=selection_b,
        navigation_b=navigation_b,
        frame_b=frame_b,
    )
    _, browse_c = _browse(
        request_c.token,
        request_c.load_generation,
        scan_key="browse.c",
        request=request_c,
    )
    display_context = __import__(
        "xdart.modules.display_context",
        fromlist=["DisplaySelection"],
    )
    original = display_context.DisplaySelection.for_context.__func__
    observed = []

    def observe_construction(cls, context, generation):
        observed.append(
            (controller.selection, controller.browse_context)
        )
        return original(cls, context, generation)

    monkeypatch.setattr(
        display_context.DisplaySelection,
        "for_context",
        classmethod(observe_construction),
    )
    loader.complete(browse_c)
    outcome = controller.poll_browse()

    assert outcome.request is request_c
    assert observed == [(selection_b, None)]
    assert controller.browse_context is browse_c
    assert controller.selection.names(browse_c)
    assert controller.selection.display_generation == (
        selection_b.display_generation + 1
    )
    assert controller.navigation.current is not frame_b
    assert controller._runtime._pending_replacement is None


def test_selecting_held_b_is_inert_and_cannot_reopen_released_owner() -> None:
    (
        controller,
        _executor,
        loader,
        acquisition,
        browse_b,
        records_b,
        publications_b,
        selection_b,
        navigation_b,
        frame_b,
        request_c,
    ) = _begin_replacement()
    pending = _assert_values_only_hold(
        controller,
        acquisition=acquisition,
        browse_b=browse_b,
        records_b=records_b,
        publications_b=publications_b,
        request_c=request_c,
        selection_b=selection_b,
        navigation_b=navigation_b,
        frame_b=frame_b,
    )

    assert controller.select_browser_target(frame_b.artifact) is True
    assert loader.request is request_c
    assert controller._browse_request is request_c
    assert controller._runtime._pending_replacement is pending
    assert controller.selection is selection_b
    assert controller.navigation is navigation_b


def test_cancelled_active_and_queued_replacement_retire_all_loader_owners(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _admit_fake_browse(monkeypatch)
    entered_c = Event()
    release_c = Event()
    c = tmp_path / "c.nexus"
    d = tmp_path / "d.nexus"
    c.write_bytes(b"c")
    d.write_bytes(b"d")

    def blocked_catalog(source, _cancelled):
        if Path(source) == c:
            entered_c.set()
            release_c.wait(timeout=30.0)

    loader = BrowseLoader(
        join_timeout=0.01,
        open_scan=lambda _source: object(),
        open_reader=_empty_reader_factory(blocked_catalog),
    )
    request_c = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE), 1, str(c)
    )
    request_d = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE), 2, str(d)
    )
    try:
        loader.begin(request_c)
        assert entered_c.wait(timeout=5.0)
        loader.begin(request_d)
        pending = loader.cancel(request_d)
        assert pending.request is request_d
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert loader._queued is not None
        assert loader._queued.request is request_d
        release_c.set()
        worker = loader._worker
        assert worker is not None
        worker.join(timeout=5.0)
        cleaned = loader.cancel(request_d)
        assert cleaned.request is request_d
        assert cleaned.cleanup_status is CleanupStatus.CLEANED
        assert loader._active is None
        assert loader._queued is None
        assert loader._worker is None
    finally:
        release_c.set()
        loader.close()


def test_close_never_waits_for_blocked_active_browse_and_keeps_latest_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _admit_fake_browse(monkeypatch)
    entered = Event()
    release = Event()
    active_path = tmp_path / "active.nexus"
    queued_path = tmp_path / "latest.nexus"
    active_path.write_bytes(b"active")
    queued_path.write_bytes(b"latest")

    def blocked_catalog(_source, _cancelled):
        entered.set()
        assert release.wait(timeout=5.0)

    loader = BrowseLoader(
        open_scan=lambda _source: object(),
        open_reader=_empty_reader_factory(blocked_catalog),
    )
    active = BrowseLoadRequest("active", 1, str(active_path))
    latest = BrowseLoadRequest("latest", 2, str(queued_path))
    worker = None
    try:
        loader.begin(active)
        assert entered.wait(timeout=5.0)
        loader.begin(latest)
        worker = loader._worker
        started = time.monotonic()
        pending = loader.close(latest)
        assert time.monotonic() - started < 0.2
        assert pending.request is latest
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert worker is not None and worker.is_alive()

        release.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        cleaned = loader.close(latest)
        assert cleaned.request is latest
        assert cleaned.cleanup_status is CleanupStatus.CLEANED
        assert loader._active is None and loader._queued is None
    finally:
        release.set()
        if worker is not None:
            worker.join(timeout=5.0)
        loader.close(latest if loader.owns_request(latest) else None)


def test_cancel_d_during_promotion_cannot_launch_unowned_d(
    tmp_path: Path,
) -> None:
    c = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE),
        1,
        str(tmp_path / "c.nxs"),
    )
    d_path = tmp_path / "d.nxs"
    d_path.write_bytes(b"d")
    d = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE), 2, str(d_path)
    )
    io_calls: list[tuple[str, str]] = []

    def open_scan(source):
        io_calls.append(("open", str(source)))
        return object()

    def read_catalog(source, _cancelled):
        io_calls.append(("read", str(source)))

    loader = BrowseLoader(
        open_scan=open_scan,
        open_reader=_empty_reader_factory(read_catalog),
    )
    active = _BrowseOperation(c, Event())
    active.cancelled.set()
    active.outcome = BrowseLoadOutcome(
        c, BrowseLoadStatus.CANCELLED
    )
    active.terminal = True
    loader._active = active
    loader._queued = _BrowseOperation(d, Event())
    promotion_exited = Event()
    continue_promotion = Event()
    original_lock = Lock()

    class PromotionGate:
        paused = False

        def __enter__(self):
            original_lock.acquire()
            return self

        def __exit__(self, *_exc):
            original_lock.release()
            promoted = loader._active
            if (
                not self.paused
                and loader._queued is None
                and (
                    promoted is None
                    or promoted.request is d
                )
            ):
                self.paused = True
                promotion_exited.set()
                if not continue_promotion.wait(timeout=10.0):
                    raise TimeoutError("promotion gate was not released")

    loader._lock = PromotionGate()
    promotion = Thread(target=loader._progress)
    promotion.start()
    try:
        assert promotion_exited.wait(timeout=5.0)
        pending = loader.cancel(d)
        assert pending.request is d
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        continue_promotion.set()
        promotion.join(timeout=5.0)
        assert not promotion.is_alive()
        worker = loader._worker
        if worker is not None:
            worker.join(timeout=5.0)
        cleaned = loader.cancel(d)
        assert cleaned.request is d
        assert cleaned.cleanup_status is CleanupStatus.CLEANED
        assert io_calls == []
        assert loader._active is None
        assert loader._queued is None
        assert loader._worker is None
    finally:
        continue_promotion.set()
        promotion.join(timeout=5.0)
        loader.close(d)


def test_malformed_exact_ready_cleanup_retry_clears_hold_once(
    monkeypatch,
) -> None:
    (
        controller,
        _executor,
        _base_loader,
        acquisition,
        _browse_b,
        _records_b,
        _publications_b,
        selection_b,
        _navigation_b,
        _frame_b,
        request_c,
    ) = _begin_replacement()
    _, candidate = _browse(
        request_c.token,
        request_c.load_generation,
        scan_key="browse.c",
        request=request_c,
    )
    candidate.invalidate()
    outcome = BrowseLoadOutcome(
        request_c, BrowseLoadStatus.READY
    )
    operation = _BrowseOperation(request_c, Event())
    operation.context = candidate
    operation.outcome = outcome
    operation.terminal = True
    loader = BrowseLoader()
    loader._active = operation
    controller._browse_loader = loader
    pending = controller._runtime._pending_replacement
    release = BrowseContext.release
    attempts = 0

    def fail_once(context):
        nonlocal attempts
        if context is candidate:
            attempts += 1
            if attempts == 1:
                raise RuntimeError("candidate cleanup retry")
        return release(context)

    monkeypatch.setattr(BrowseContext, "release", fail_once)
    try:
        assert controller.poll_browse() is None
        assert attempts == 1
        assert controller._runtime._pending_replacement is pending
        assert loader._active is operation

        cleaned = controller.poll_browse()
        assert type(cleaned) is BrowseCleanupReceipt
        assert cleaned.request is request_c
        assert cleaned.cleanup_status is CleanupStatus.CLEANED
        assert attempts == 2
        assert loader._active is None
        assert loader._queued is None
        assert loader._worker is None
        assert controller.browse_pending is False
        assert controller._runtime._pending_replacement is None
        assert controller.selection.names(acquisition)
        assert controller.selection.display_generation == (
            selection_b.display_generation + 1
        )
    finally:
        loader.close(request_c)


@pytest.mark.parametrize("cancel_queued", (False, True))
def test_close_correlates_active_c_and_queued_d_to_exact_d(
    tmp_path: Path,
    cancel_queued: bool,
    monkeypatch,
) -> None:
    _admit_fake_browse(monkeypatch)
    entered_c = Event()
    release_c = Event()
    c = tmp_path / "c.nexus"
    d = tmp_path / "d.nexus"
    c.write_bytes(b"c")
    d.write_bytes(b"d")

    def blocked_catalog(source, _cancelled):
        if Path(source) == c:
            entered_c.set()
            release_c.wait(timeout=30.0)

    loader = BrowseLoader(
        join_timeout=0.001,
        open_scan=lambda _source: object(),
        open_reader=_empty_reader_factory(blocked_catalog),
    )
    request_c = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE), 1, str(c)
    )
    request_d = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE), 2, str(d)
    )
    try:
        loader.begin(request_c)
        assert entered_c.wait(timeout=5.0)
        loader.begin(request_d)
        if cancel_queued:
            loader.cancel(request_d)
        pending = loader.close(request_d)
        assert pending.request is request_d
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        release_c.set()
        worker = loader._worker
        assert worker is not None
        worker.join(timeout=5.0)
        cleaned = loader.close(request_d)
        assert cleaned.request is request_d
        assert cleaned.cleanup_status is CleanupStatus.CLEANED
        assert loader.close(request_d) is cleaned
        assert loader._active is None
        assert loader._queued is None
        assert loader._worker is None
    finally:
        release_c.set()
        try:
            loader.close(request_d)
        except TypeError:
            loader.close()


def test_queued_thread_start_failure_is_one_exact_terminal_d(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class StartFailure:
        ident = None

        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("D thread refused")

        def is_alive(self):
            return False

    c = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE),
        1,
        str(tmp_path / "c.nxs"),
    )
    d = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE),
        2,
        str(tmp_path / "d.nxs"),
    )
    loader = BrowseLoader()
    active = _BrowseOperation(c, Event())
    active.outcome = BrowseLoadOutcome(c, BrowseLoadStatus.CANCELLED)
    active.terminal = True
    loader._active = active
    monkeypatch.setattr(loader_module, "Thread", StartFailure)

    assert loader.begin(d) is d
    outcome = loader.poll(d)
    assert outcome is not None
    assert outcome.request is d
    assert outcome.status is BrowseLoadStatus.FAILED
    assert outcome.detail == "D thread refused"
    assert loader.owns_outcome(outcome) is True
    assert loader.consume(outcome) is None
    assert loader._active is None
    assert loader._queued is None
    assert loader._worker is None


def _image_copy(item):
    image = item.image
    return None if image is None else np.array(image, copy=True)


def _trace_copy(shell):
    return tuple(
        (
            np.array(item.xData, copy=True),
            np.array(item.yData, copy=True),
        )
        for item in shell.scientific.curve.listDataItems()
    )


def test_every_public_projection_holds_released_b_until_atomic_c(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_b = _produce_browse_artifact(monkeypatch, tmp_path / "b")
    source_c = _produce_browse_artifact(monkeypatch, tmp_path / "c")
    active_catalog = tmp_path / "a" / "project" / "processed"
    b = active_catalog / "browse-b.nexus"
    c = active_catalog / "browse-c.nexus"
    entered_c = Event()
    release_c = Event()
    rig = _mount(
        monkeypatch,
        tmp_path / "a",
        labels=tuple(range(1, 100)),
        reduction_delay=0.001,
        browse_entered=entered_c,
        browse_release=release_c,
        browse_gate_path=c,
    )
    try:
        copy2(source_b, b)
        copy2(source_c, c)
        rig.command(ShellCommand(ShellCommandKind.REFRESH_BROWSER))

        def browser_artifacts() -> set[str]:
            return {
                str(
                    rig.shell.browser.scans.item(index).data(
                        QtCore.Qt.ItemDataRole.UserRole
                    )
                )
                for index in range(rig.shell.browser.scans.count())
            }

        _wait(
            rig.app,
            lambda: {str(b), str(c)} <= browser_artifacts(),
            diagnostic=lambda: sorted(browser_artifacts()),
        )
        _run(rig)
        _pause(rig)
        rig.command(ShellCommand(ShellCommandKind.SELECT_SCAN, str(b)))
        _wait(
            rig.app,
            lambda: (
                rig.controller.selection is not None
                and rig.controller.selection.kind is ContextKind.BROWSE
                and rig.shell.scientific.raw.image.image is not None
                and rig.shell.scientific.cake.image.image is not None
                and bool(rig.shell.scientific.curve.listDataItems())
            ),
        )
        context_b = rig.controller.browse_context
        assert context_b is not None
        records_b = context_b.record_store
        publications_b = context_b.publication_store
        selection_b = rig.controller.selection
        navigation_b = rig.controller.navigation
        frame_b = navigation_b.current
        assert frame_b is not None
        title_b = rig.shell.scientific.title.text()
        raw_b = _image_copy(rig.shell.scientific.raw.image)
        cake_b = _image_copy(rig.shell.scientific.cake.image)
        traces_b = _trace_copy(rig.shell)
        observations: list[
            tuple[ShellCommandKind, ShellProjection, tuple]
        ] = []
        active_command = [ShellCommandKind.SELECT_SCAN]
        real_apply = rig.shell.apply_state

        def observe(
            state: ShellProjection,
            *,
            preserve_display: bool = False,
            preserve_scientific: bool = False,
            replace_scientific_on_failure: bool = False,
        ) -> None:
            real_apply(
                state,
                preserve_display=preserve_display,
                preserve_scientific=preserve_scientific,
                replace_scientific_on_failure=(
                    replace_scientific_on_failure
                ),
            )
            current_index = rig.shell.browser.frames.currentIndex()
            selected_scans = tuple(
                item.data(QtCore.Qt.ItemDataRole.UserRole)
                for item in rig.shell.browser.scans.selectedItems()
            )
            observations.append(
                (
                    active_command[0],
                    state,
                    (
                        rig.controller.selection,
                        rig.controller.navigation,
                        rig.shell.scientific.frame_selector.currentData(),
                        current_index.data(
                            QtCore.Qt.ItemDataRole.UserRole
                        ),
                        selected_scans,
                        rig.shell.scientific.title.text(),
                        _image_copy(rig.shell.scientific.raw.image),
                        _image_copy(rig.shell.scientific.cake.image),
                        _trace_copy(rig.shell),
                        len(records_b),
                        len(publications_b),
                    ),
                )
            )

        monkeypatch.setattr(rig.shell, "apply_state", observe)

        def dispatch(command: ShellCommand) -> int:
            active_command[0] = command.kind
            before = len(observations)
            rig.command(command)
            return before

        dispatch(ShellCommand(ShellCommandKind.SELECT_SCAN, str(c)))
        _wait(rig.app, entered_c.is_set)
        request_c = rig.controller._browse_request
        assert request_c is not None
        _assert_values_only_hold(
            rig.controller,
            acquisition=rig.controller.acquisition_context,
            browse_b=context_b,
            records_b=records_b,
            publications_b=publications_b,
            request_c=request_c,
            selection_b=selection_b,
            navigation_b=navigation_b,
            frame_b=frame_b,
        )

        for command in (
            ShellCommand(ShellCommandKind.SET_COLOR_MAP, "viridis"),
            ShellCommand(ShellCommandKind.SET_LOG_SCALE, True),
            ShellCommand(ShellCommandKind.SET_PLOT_MODE, "Overlay"),
            ShellCommand(ShellCommandKind.SELECT_SCAN, frame_b.artifact),
        ):
            before = dispatch(command)
            assert len(observations) > before
            assert all(
                triggering_kind is command.kind
                for triggering_kind, _state, _rendered
                in observations[before:]
            )

        before = len(observations)
        browse_facts = tuple(rig.browse_facts)
        for kind in (
            ShellCommandKind.SELECT_FRAME,
            ShellCommandKind.HYDRATE_FRAME,
            ShellCommandKind.SELECT_BROWSER_FRAMES,
        ):
            dispatch(
                ShellCommand(
                    kind,
                    frame=frame_b,
                    frames=(frame_b,),
                )
            )
        assert len(observations) == before
        assert tuple(rig.browse_facts) == browse_facts

        interim = list(observations)
        assert interim
        for _triggering_kind, state, rendered in interim:
            (
                selection,
                navigation,
                footer_key,
                browser_key,
                selected_scans,
                title,
                raw,
                cake,
                traces,
                record_count,
                publication_count,
            ) = rendered
            assert selection is selection_b
            assert navigation is navigation_b
            assert state.navigation is navigation_b
            assert state.navigation.current is frame_b
            assert state.navigation.selected == (frame_b,)
            assert state.browser.selected_scan == frame_b.artifact
            assert state.scientific.heavy is None
            assert state.scientific.traces == ()
            assert state.scientific.retain_display is True
            assert footer_key is frame_b
            assert browser_key is frame_b
            assert selected_scans == (frame_b.artifact,)
            assert title == title_b
            np.testing.assert_array_equal(raw, raw_b)
            np.testing.assert_array_equal(cake, cake_b)
            assert len(traces) == len(traces_b)
            for actual, expected in zip(traces, traces_b):
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
            assert record_count == publication_count == 0

        release_c.set()
        _wait(
            rig.app,
            lambda: (
                rig.controller.browse_context is not None
                and rig.controller.browse_context.requested_path == str(c)
                and rig.controller.navigation.current is not frame_b
            ),
        )
        frame_c = rig.controller.navigation.current
        assert frame_c is not None
        transitions = [
            state.navigation.current
            for _triggering_kind, state, _rendered in observations
            if state.navigation.current is not frame_b
        ]
        assert transitions
        assert transitions[0] is frame_c
        assert all(key is frame_c for key in transitions)
        assert rig.controller.selection.display_generation == (
            selection_b.display_generation + 1
        )
        assert rig.controller._runtime._pending_replacement is None
    finally:
        release_c.set()
        receipt = rig.close()
        assert receipt.cleanup_status is CleanupStatus.CLEANED
        rig.page.deleteLater()
        rig.app.processEvents()
