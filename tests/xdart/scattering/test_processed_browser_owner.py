"""Qt-free ownership tests for the processed Browser lifecycle."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event
import time

import pytest

from xdart.gui.tabs.scattering.browser_catalog import BrowserCatalogEntry
from xdart.gui.tabs.scattering.browse_values import (
    BrowseLoadOutcome,
    BrowseLoadRequest,
    BrowseLoadStatus,
    LoadedBrowseCapture,
)
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.processed_browser import (
    AverageReloadDirective,
    BrowserCatalogWake,
    BrowserRefreshEffect,
    ProcessedBrowserOwner,
    ReintegrateReloadDirective,
    TerminalBrowsePaintReceipt,
    TerminalPaintMode,
    TerminalRebindAuthorization,
)
from xdart.modules.display_context import (
    BrowseContext,
    ContextKind,
    DisplaySelection,
    HydrationOwner,
)
from xrd_tools.io.output_transaction import StreamTerminal, TargetSnapshot


def _wait(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("processed Browser owner did not settle")


def _entry(directory: str, label: str) -> BrowserCatalogEntry:
    return BrowserCatalogEntry(
        str(Path(directory) / f"{label}.nexus"),
        label,
        1,
    )


def _loaded_capture(
    request: BrowseLoadRequest,
    *,
    size: int = 17,
    digest: str = "d" * 64,
) -> LoadedBrowseCapture:
    context = object.__new__(BrowseContext)
    selection = DisplaySelection(
        ContextKind.BROWSE,
        HydrationOwner(request.token, "scan", request.source_path, 1),
        2,
    )
    return LoadedBrowseCapture(
        context,
        request,
        selection,
        request.source_path,
        "entry",
        TargetSnapshot(True, size, 4, 2, 3, digest),
        (0, 1),
    )


def test_catalog_request_freezes_date_policy_across_latest_only_queue() -> None:
    gates = [Event(), Event()]
    calls: list[tuple[str, bool]] = []
    wakes: list[BrowserCatalogWake] = []

    def reader(directory: str, **kwargs):
        index = len(calls)
        calls.append((directory, kwargs["inspect_directory_contents"]))
        assert gates[index].wait(3.0)
        return (_entry(directory, f"result-{index}"),)

    owner = ProcessedBrowserOwner(
        save_path="/processed",
        processing_mode="Int 2D",
        deliver=wakes.append,
        catalog_reader=reader,
    )
    try:
        first = owner.request_catalog()
        assert first is not None
        _wait(lambda: len(calls) == 1)
        assert owner.set_date_sorted(True)
        queued = owner.queued_request
        assert queued is not None
        assert not first.inspect_directory_contents
        assert queued.inspect_directory_contents

        gates[0].set()
        _wait(lambda: bool(wakes))
        stale = owner.consume_catalog(wakes.pop(0))
        assert stale.refresh is BrowserRefreshEffect.NONE
        _wait(lambda: len(calls) == 2)
        assert calls == [
            ("/processed", False),
            ("/processed", True),
        ]

        gates[1].set()
        _wait(lambda: bool(wakes))
        accepted = owner.consume_catalog(wakes.pop(0))
        assert accepted.refresh is BrowserRefreshEffect.CATALOG
        assert tuple(entry.label for entry in owner.catalog) == ("result-1",)
    finally:
        for gate in gates:
            gate.set()
        owner.begin_close()
        _wait(owner.retry_close)


def test_active_catalog_keeps_only_the_latest_directory_replacement() -> None:
    gates = [Event(), Event()]
    calls: list[str] = []
    wakes: list[BrowserCatalogWake] = []

    def reader(directory: str, **_kwargs):
        index = len(calls)
        calls.append(directory)
        assert gates[index].wait(3.0)
        return (_entry(directory, Path(directory).name),)

    owner = ProcessedBrowserOwner(
        save_path="/root",
        processing_mode="Int 2D",
        deliver=wakes.append,
        catalog_reader=reader,
    )
    try:
        owner.set_directory("/root/a", explicit=False)
        _wait(lambda: calls == ["/root/a"])
        owner.set_directory("/root/b", explicit=False)
        owner.set_directory("/root/c", explicit=False)
        assert owner.queued_request is not None
        assert owner.queued_request.directory == "/root/c"

        gates[0].set()
        _wait(lambda: bool(wakes))
        assert (
            owner.consume_catalog(wakes.pop(0)).refresh
            is BrowserRefreshEffect.NONE
        )
        _wait(lambda: len(calls) == 2)
        assert calls == ["/root/a", "/root/c"]

        gates[1].set()
        _wait(lambda: bool(wakes))
        owner.consume_catalog(wakes.pop(0))
        assert tuple(entry.label for entry in owner.catalog) == ("c",)
    finally:
        for gate in gates:
            gate.set()
        owner.begin_close()
        _wait(owner.retry_close)


def test_foreign_equal_wake_cannot_consume_the_owned_catalog() -> None:
    wakes: list[BrowserCatalogWake] = []
    owner = ProcessedBrowserOwner(
        save_path="/processed",
        processing_mode="Int 2D",
        deliver=wakes.append,
        catalog_reader=lambda directory, **_kwargs: (_entry(directory, "one"),),
    )
    try:
        request = owner.request_catalog()
        assert request is not None
        _wait(lambda: bool(wakes))
        owned = wakes[-1]
        foreign = BrowserCatalogWake(owned.token)

        assert (
            owner.consume_catalog(foreign).refresh
            is BrowserRefreshEffect.NONE
        )
        assert owner.active_wake is owned
        assert (
            owner.consume_catalog(owned).refresh
            is BrowserRefreshEffect.CATALOG
        )
    finally:
        owner.begin_close()
        _wait(owner.retry_close)


@pytest.mark.parametrize(
    ("reader", "notice"),
    (
        (lambda _directory, **_kwargs: ("not-an-entry",), "invalid data"),
        (
            lambda _directory, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("catalog exploded")
            ),
            "catalog exploded",
        ),
    ),
)
def test_current_catalog_invalid_result_and_reader_failure_are_explicit(
    reader, notice: str,
) -> None:
    wakes: list[BrowserCatalogWake] = []
    owner = ProcessedBrowserOwner(
        save_path="/processed",
        processing_mode="Int 2D",
        deliver=wakes.append,
        catalog_reader=reader,
    )
    try:
        assert owner.request_catalog() is not None
        _wait(lambda: bool(wakes))
        transition = owner.consume_catalog(wakes[-1])
        assert transition.refresh is BrowserRefreshEffect.FULL
        assert transition.notice is not None
        assert notice in transition.notice
        assert owner.catalog == ()
    finally:
        owner.begin_close()
        _wait(owner.retry_close)


def test_explicit_directory_veto_survives_frames_until_next_run_follow() -> None:
    owner = ProcessedBrowserOwner(
        save_path="/configured",
        processing_mode="Int 2D",
        deliver=lambda _wake: None,
        catalog_reader=lambda _directory, **_kwargs: (),
    )
    first = RunIdentity(1, "first")
    second = RunIdentity(2, "second")
    try:
        owner.begin_follow(first)
        moved = owner.follow_processed_artifact(
            DisplayFrameKey(first, "one", "/run/first.nxs", 0, 1)
        )
        assert moved.refresh is BrowserRefreshEffect.FULL
        assert owner.directory == "/run"

        owner.set_directory("/manual", explicit=True)
        owner.follow_processed_artifact(
            DisplayFrameKey(second, "two", "/other/second.nxs", 0, 1)
        )
        assert owner.directory == "/manual"

        owner.begin_follow(second)
        owner.follow_processed_artifact(
            DisplayFrameKey(second, "two", "/other/second.nxs", 0, 1)
        )
        assert owner.directory == "/other"
        assert not owner.explicit_directory
    finally:
        owner.begin_close()
        _wait(owner.retry_close)


def test_transient_frame_retires_only_at_its_catalog_barrier() -> None:
    gates = [Event(), Event()]
    wakes: list[BrowserCatalogWake] = []
    calls = 0

    def reader(_directory: str, **_kwargs):
        nonlocal calls
        index = calls
        calls += 1
        assert gates[index].wait(3.0)
        return ()

    owner = ProcessedBrowserOwner(
        save_path="/processed",
        processing_mode="Int 2D",
        deliver=wakes.append,
        catalog_reader=reader,
    )
    identity = RunIdentity(1, "terminal")
    frame = DisplayFrameKey(identity, "terminal", "/processed/final.nxs", 0, 1)
    try:
        first = owner.request_catalog()
        assert first is not None
        owner.set_transient_frame(frame)
        second = owner.request_catalog()
        owner.mark_transient_catalog_barrier(second)
        assert owner.transient_frame is frame

        gates[0].set()
        _wait(lambda: bool(wakes))
        owner.consume_catalog(wakes.pop(0))
        assert owner.transient_frame is frame

        _wait(lambda: calls == 2)
        gates[1].set()
        _wait(lambda: bool(wakes))
        transition = owner.consume_catalog(wakes.pop(0))
        assert transition.refresh is BrowserRefreshEffect.CATALOG
        assert owner.transient_frame is None
    finally:
        for gate in gates:
            gate.set()
        owner.begin_close()
        _wait(owner.retry_close)


def test_close_is_nonblocking_and_late_wake_is_inert() -> None:
    entered = Event()
    release = Event()
    wakes: list[BrowserCatalogWake] = []

    def reader(_directory: str, **_kwargs):
        entered.set()
        assert release.wait(3.0)
        return ()

    owner = ProcessedBrowserOwner(
        save_path="/processed",
        processing_mode="Int 2D",
        deliver=wakes.append,
        catalog_reader=reader,
    )
    owner.request_catalog()
    assert entered.wait(1.0)

    started = time.monotonic()
    assert not owner.begin_close()
    assert time.monotonic() - started < 0.2
    release.set()
    _wait(lambda: bool(wakes))
    assert owner.consume_catalog(wakes[-1]).refresh is BrowserRefreshEffect.NONE
    _wait(owner.retry_close)
    assert owner.closed
    assert not owner.pool_open


def test_reload_custody_is_one_exact_union_slot() -> None:
    target = "/processed/result.nexus"
    request = BrowseLoadRequest(
        "reintegrate-reload", 1, target, source_root="/project"
    )
    reintegrate = ReintegrateReloadDirective(request, target)
    average_target = "/processed/average.nexus"
    seal = StreamTerminal(
        average_target, 17, "d" * 64, 1, 2, 3, 4, 5
    )
    average = AverageReloadDirective(
        average_target, "entry", seal, "/project"
    )
    owner = ProcessedBrowserOwner(
        save_path="/processed",
        processing_mode="Int 2D",
        deliver=lambda _wake: None,
        catalog_reader=lambda _directory, **_kwargs: (),
    )
    try:
        assert owner.adopt_reload(reintegrate) is reintegrate
        assert owner.pending_reintegrate_reload is reintegrate
        assert owner.pending_average_reload is None
        assert owner.busy and owner.polling_needed and owner.preserve_science
        assert not owner.retire_reload(
            ReintegrateReloadDirective(request, target)
        )
        with pytest.raises(RuntimeError, match="already owns"):
            owner.adopt_reload(average)
        assert owner.retire_reload(reintegrate)
        assert owner.adopt_reload(average) is average
        assert owner.pending_average_reload is average
        assert average.source_root == "/project"
    finally:
        owner.begin_close()
        _wait(owner.retry_close)
    assert owner.pending_average_reload is None


def test_terminal_paint_receipts_are_exact_and_rebind_falls_back_once() -> None:
    ticks = iter((1.0, 2.0, 3.0, 4.0, 5.0))
    target = "/processed/final.nexus"
    seal = StreamTerminal(target, 17, "d" * 64, 1, 2, 3, 4, 5)
    request = BrowseLoadRequest(
        "terminal", 1, target, seal, source_root="/project"
    )
    identity = RunIdentity(7, "terminal")
    owner = ProcessedBrowserOwner(
        save_path="/processed",
        processing_mode="Int 2D",
        deliver=lambda _wake: None,
        catalog_reader=lambda _directory, **_kwargs: (),
        clock=lambda: next(ticks),
    )
    capture = _loaded_capture(request)
    try:
        handoff = owner.begin_terminal_handoff(
            request,
            identity,
            "/staging/final.nexus",
            0,
            (0, 1),
            seal,
            timing_start=owner.begin_terminal_timing(enabled=True),
        )
        assert handoff is not None
        settlement = owner.settle_terminal(
            BrowseLoadOutcome(request, BrowseLoadStatus.READY), capture
        )
        assert settlement is not None
        assert settlement.reuse_seal_authorized
        authorization = TerminalRebindAuthorization(
            identity, handoff.source_artifact, target
        )
        assert owner.authorize_terminal_rebind(
            settlement.presentation, authorization
        )

        paint = owner.begin_terminal_paint(
            settlement.presentation,
            capture,
            owns_request=True,
            reuse_science=True,
        )
        assert paint is not None
        assert paint.mode is TerminalPaintMode.REBIND
        assert paint.authorization is authorization
        foreign = replace(paint)
        stale = owner.complete_terminal_paint(
            TerminalBrowsePaintReceipt(foreign, True, False)
        )
        assert not stale.accepted
        fallback = owner.complete_terminal_paint(
            TerminalBrowsePaintReceipt(paint, True, True)
        )
        assert fallback.accepted and not fallback.retired

        repaint = owner.begin_terminal_paint(
            settlement.presentation,
            capture,
            owns_request=True,
            reuse_science=False,
        )
        assert repaint is not None
        assert repaint.mode is TerminalPaintMode.REPAINT_FALLBACK
        done = owner.complete_terminal_paint(
            TerminalBrowsePaintReceipt(repaint, True, False)
        )
        assert done.accepted and done.retired
        assert owner.terminal_presentation is None
        assert not owner.terminal_perf_active
    finally:
        owner.begin_close()
        _wait(owner.retry_close)


def test_terminal_timing_start_is_same_owner_and_single_use() -> None:
    target = "/processed/final.nexus"
    request = BrowseLoadRequest("terminal-timing", 1, target)
    identity = RunIdentity(10, "terminal-timing")
    owner = ProcessedBrowserOwner(
        save_path="/processed",
        processing_mode="Int 2D",
        deliver=lambda _wake: None,
        catalog_reader=lambda _directory, **_kwargs: (),
        clock=lambda: 11.0,
    )
    foreign_owner = ProcessedBrowserOwner(
        save_path="/processed",
        processing_mode="Int 2D",
        deliver=lambda _wake: None,
        catalog_reader=lambda _directory, **_kwargs: (),
        clock=lambda: 11.0,
    )
    try:
        timing_start = owner.begin_terminal_timing(enabled=True)
        foreign_start = foreign_owner.begin_terminal_timing(enabled=True)
        assert timing_start is not None and foreign_start is not None
        assert owner.begin_terminal_handoff(
            request,
            identity,
            target,
            0,
            (0,),
            None,
            timing_start=foreign_start,
        ) is None
        handoff = owner.begin_terminal_handoff(
            request,
            identity,
            target,
            0,
            (0,),
            None,
            timing_start=timing_start,
        )
        assert handoff is not None
        assert owner.begin_terminal_handoff(
            request,
            identity,
            target,
            0,
            (0,),
            None,
            timing_start=timing_start,
        ) is None
        assert not owner.retire_terminal_timing(timing_start)
        assert foreign_owner.retire_terminal_timing(foreign_start)
        assert not foreign_owner.retire_terminal_timing(foreign_start)
    finally:
        owner.begin_close()
        foreign_owner.begin_close()
        _wait(owner.retry_close)
        _wait(foreign_owner.retry_close)


def test_failed_terminal_paint_retains_exact_presentation_for_retry() -> None:
    target = "/processed/final.nexus"
    request = BrowseLoadRequest("terminal-retry", 1, target)
    identity = RunIdentity(8, "terminal-retry")
    owner = ProcessedBrowserOwner(
        save_path="/processed",
        processing_mode="Int 2D",
        deliver=lambda _wake: None,
        catalog_reader=lambda _directory, **_kwargs: (),
    )
    capture = _loaded_capture(request)
    try:
        owner.begin_terminal_handoff(
            request,
            identity,
            target,
            0,
            (0,),
            None,
            timing_start=None,
        )
        settlement = owner.settle_terminal(
            BrowseLoadOutcome(request, BrowseLoadStatus.READY), capture
        )
        assert settlement is not None
        paint = owner.begin_terminal_paint(
            settlement.presentation,
            capture,
            owns_request=True,
            reuse_science=False,
        )
        assert paint is not None
        pending = owner.complete_terminal_paint(
            TerminalBrowsePaintReceipt(paint, False, False)
        )
        assert pending.accepted and pending.schedule_repaint
        assert owner.terminal_presentation is settlement.presentation
    finally:
        owner.begin_close()
        _wait(owner.retry_close)


def test_clock_failure_drops_only_terminal_telemetry() -> None:
    target = "/processed/final.nexus"
    request = BrowseLoadRequest("terminal-no-clock", 1, target)
    identity = RunIdentity(9, "terminal-no-clock")

    def broken_clock() -> float:
        raise RuntimeError("clock unavailable")

    owner = ProcessedBrowserOwner(
        save_path="/processed",
        processing_mode="Int 2D",
        deliver=lambda _wake: None,
        catalog_reader=lambda _directory, **_kwargs: (),
        clock=broken_clock,
    )
    capture = _loaded_capture(request)
    try:
        handoff = owner.begin_terminal_handoff(
            request,
            identity,
            target,
            0,
            (0,),
            None,
            timing_start=owner.begin_terminal_timing(enabled=True),
        )
        assert handoff is not None
        assert not owner.terminal_perf_active
        settlement = owner.settle_terminal(
            BrowseLoadOutcome(request, BrowseLoadStatus.READY), capture
        )
        assert settlement is not None
        paint = owner.begin_terminal_paint(
            settlement.presentation,
            capture,
            owns_request=True,
            reuse_science=False,
        )
        assert paint is not None
        completion = owner.complete_terminal_paint(
            TerminalBrowsePaintReceipt(paint, True, False)
        )
        assert completion.accepted and completion.retired
        assert owner.terminal_presentation is None
    finally:
        owner.begin_close()
        _wait(owner.retry_close)
