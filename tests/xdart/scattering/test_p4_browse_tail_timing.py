"""Focused terminal Browse-tail timing and status oracles."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.core.test_vnext_p34_existing_replacement import _seed_existing


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _StepClock:
    def __init__(self, values):
        self._values = iter(values)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return float(next(self._values))


class _FailingClock:
    def __init__(self, fail_at: int):
        self.fail_at = fail_at
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self.calls == self.fail_at:
            raise KeyboardInterrupt("timing clock failed")
        return float(self.calls)


def _loaded_capture(request, *, context=None, labels=(0, 1, 2)):
    from xdart.gui.tabs.scattering.browse_values import LoadedBrowseCapture
    from xdart.modules.display_context import (
        BrowseContext,
        ContextKind,
        DisplaySelection,
        HydrationOwner,
    )
    from xrd_tools.io.output_transaction import TargetSnapshot
    from xrd_tools.reduction import prepare_reintegrate_bundle

    if context is not None:
        return LoadedBrowseCapture(
            context,
            request,
            DisplaySelection.for_context(context, 2),
            context.requested_path,
            context.target_entry,
            context.target_snapshot,
            context.loaded_labels,
            context.prepared_reintegrate_offer,
        )
    detached = object.__new__(BrowseContext)
    offer = prepare_reintegrate_bundle(
        None, entry="entry", labels=tuple(labels),
    )
    object.__setattr__(detached, "prepared_reintegrate_offer", offer)
    return LoadedBrowseCapture(
        detached,
        request,
        DisplaySelection(
            ContextKind.BROWSE,
            HydrationOwner(request.token, "scan", request.source_path, 1),
            2,
        ),
        request.source_path,
        "entry",
        TargetSnapshot(True, 17, 4, 2, 3, "d" * 64),
        tuple(labels),
        offer,
    )


def _processed_owner(clock):
    from xdart.gui.tabs.scattering.processed_browser import ProcessedBrowserOwner

    return ProcessedBrowserOwner(
        save_path="/processed",
        processing_mode="Int 2D",
        deliver=lambda _wake: None,
        catalog_reader=lambda _directory, **_kwargs: (),
        clock=clock,
    )


@pytest.mark.parametrize("sealed", (False, True), ids=("snapshot", "terminal"))
def test_browse_worker_reports_one_frozen_stage_aggregate(
    tmp_path, sealed: bool,
) -> None:
    from xdart.gui.tabs.scattering.adapters import browse_loader as module
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadStatus,
        BrowseLoadTiming,
    )

    seeded = _seed_existing(tmp_path)
    gate = [True]
    gate_calls = []
    catalog_passes = []
    real_reader = module.FrameViewReader

    class Reader:
        def __init__(self, path, *, resolve_source):
            assert resolve_source is False
            self._reader = real_reader(path, resolve_source=resolve_source)

        def __enter__(self):
            self._reader.__enter__()
            return self

        def read_scalar_catalog(self, *, cancelled):
            catalog_passes.append(str(self._reader.path))
            return self._reader.read_scalar_catalog(cancelled=cancelled)

        def __exit__(self, exc_type, exc, tb):
            return self._reader.__exit__(exc_type, exc, tb)

    clock = _StepClock(range(16))
    loader = module.BrowseLoader(
        clock=clock,
        perf_enabled=lambda: gate_calls.append(True) or gate[0],
        open_reader=Reader,
    )
    request = BrowseLoadRequest(
        "timed-terminal" if sealed else "timed-snapshot",
        1,
        str(seeded.target.resolve()),
        seeded.terminal.commit_identity if sealed else None,
    )
    loader.begin(request)
    # The performance decision belongs to this admitted operation and cannot
    # be changed while its worker is active.
    gate[0] = False
    worker = loader._worker
    assert worker is not None
    worker.join(20)
    assert not worker.is_alive()

    outcome = loader.poll(request)
    assert outcome is not None
    assert outcome.status is BrowseLoadStatus.READY
    assert type(outcome.timing) is BrowseLoadTiming
    assert outcome.timing == BrowseLoadTiming(
        str(seeded.target.resolve()),
        "terminal" if sealed else "snapshot",
        1.0,
        1.0,
        1.0,
        len(seeded.labels),
        1.0,
        1.0,
        1.0,
        15.0,
        prepared_capsule_s=1.0,
        prepared_bundle_bytes=0,
        prepared_1d_status="MISS",
        prepared_2d_status="MISS",
    )
    assert gate_calls == [True]
    assert clock.calls == 16
    assert catalog_passes == [str(seeded.target.resolve())]
    with pytest.raises(FrozenInstanceError):
        outcome.timing.record_count = 0

    context = loader.consume(outcome)
    assert context is not None
    assert context.loaded_labels == seeded.labels
    assert loader.release_context(context).cleanup_status.value == "cleaned"


def test_browse_worker_disabled_gate_never_reads_clock(tmp_path) -> None:
    from xdart.gui.tabs.scattering.adapters import browse_loader as module
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadStatus,
    )

    seeded = _seed_existing(tmp_path)

    def forbidden_clock() -> float:
        raise AssertionError("disabled Browse timing read its clock")

    loader = module.BrowseLoader(
        clock=forbidden_clock,
        perf_enabled=lambda: False,
    )
    request = BrowseLoadRequest(
        "untimed", 1, str(seeded.target.resolve()),
    )
    loader.begin(request)
    worker = loader._worker
    assert worker is not None
    worker.join(20)
    assert not worker.is_alive()
    outcome = loader.poll(request)
    assert outcome is not None
    assert outcome.status is BrowseLoadStatus.READY
    assert outcome.timing is None
    context = loader.consume(outcome)
    assert context is not None
    assert loader.release_context(context).cleanup_status.value == "cleaned"


def test_terminal_browse_log_uses_worker_canonical_path_for_symlink_alias(
    tmp_path, caplog, monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.adapters import browse_loader as module
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadStatus,
        BrowseLoadTiming,
    )
    from xdart.gui.tabs.scattering.events import RunIdentity
    from xdart.gui.tabs.scattering.processed_browser import (
        TerminalBrowsePaintReceipt,
        TerminalPaintMode,
    )

    seeded = _seed_existing(tmp_path)
    alias = tmp_path / f"terminal-alias{seeded.target.suffix}"
    alias.symlink_to(seeded.target)
    canonical = str(seeded.target.resolve())
    request = BrowseLoadRequest("alias-terminal", 7, str(alias))
    assert request.source_path != canonical
    loader = module.BrowseLoader(
        clock=_StepClock(range(16)),
        perf_enabled=lambda: True,
    )
    loader.begin(request)
    worker_thread = loader._worker
    assert worker_thread is not None
    worker_thread.join(20)
    assert not worker_thread.is_alive()
    outcome = loader.poll(request)
    assert outcome is not None
    assert outcome.status is BrowseLoadStatus.READY
    assert type(outcome.timing) is BrowseLoadTiming
    assert outcome.timing.canonical_path == canonical
    context = loader.consume(outcome)
    assert context is not None
    capture = _loaded_capture(request, context=context)
    owner = _processed_owner(_StepClock((30.0, 30.25, 30.5, 30.5, 31.0)))
    identity = RunIdentity(7, "alias-terminal")
    handoff = owner.begin_terminal_handoff(
        request,
        identity,
        request.source_path,
        None,
        (),
        None,
        timing_start=owner.begin_terminal_timing(enabled=True),
    )
    assert handoff is not None
    settle_started = owner.begin_settle_timing(request)
    settlement = owner.settle_terminal(outcome, capture)
    assert settlement is not None
    assert owner.finish_settle_timing(request, settle_started, outcome)
    caplog.set_level(
        "INFO", logger="xdart.gui.tabs.scattering.processed_browser"
    )
    gui_resolves: list[str] = []

    def forbidden_gui_resolve(path, *args, **kwargs):
        gui_resolves.append(str(path))
        raise AssertionError("GUI terminal presentation resolved a path")

    try:
        with monkeypatch.context() as gui_guard:
            gui_guard.setattr(Path, "resolve", forbidden_gui_resolve)
            paint = owner.begin_terminal_paint(
                settlement.presentation,
                capture,
                owns_request=True,
                reuse_science=False,
            )
            assert paint is not None
            assert paint.mode is TerminalPaintMode.REPAINT
            completion = owner.complete_terminal_paint(
                TerminalBrowsePaintReceipt(paint, True, False)
            )
            assert completion.accepted and completion.retired
        assert gui_resolves == []
        records = [
            record for record in caplog.records
            if "[PERF-BROWSE]" in record.getMessage()
        ]
        assert len(records) == 1
        message = records[0].getMessage()
        assert (
            f"source={canonical} token=alias-terminal generation=7"
            in message
        )
        assert f"source={request.source_path} " not in message
    finally:
        assert loader.release_context(context).cleanup_status.value == "cleaned"
        owner.begin_close()
        assert owner.retry_close()


@pytest.mark.parametrize("fail_at", (1, 16), ids=("start", "final"))
def test_browse_worker_clock_failure_drops_timing_not_ready_context(
    tmp_path, fail_at: int,
) -> None:
    from xdart.gui.tabs.scattering.adapters import browse_loader as module
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadStatus,
    )

    seeded = _seed_existing(tmp_path)
    clock = _FailingClock(fail_at)
    loader = module.BrowseLoader(
        clock=clock,
        perf_enabled=lambda: True,
    )
    request = BrowseLoadRequest(
        f"clock-failure-{fail_at}",
        1,
        str(seeded.target.resolve()),
    )
    loader.begin(request)
    worker = loader._worker
    assert worker is not None
    worker.join(20)
    assert not worker.is_alive()

    outcome = loader.poll(request)
    assert outcome is not None
    assert outcome.status is BrowseLoadStatus.READY
    assert outcome.timing is None
    context = loader.consume(outcome)
    assert context is not None
    assert context.loaded_labels == seeded.labels
    assert loader.release_context(context).cleanup_status.value == "cleaned"


@pytest.mark.parametrize("mode", ("rebind", "repaint-fallback"))
def test_terminal_browse_logs_once_at_actual_ready_paint(
    caplog, mode: str,
) -> None:
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadOutcome,
        BrowseLoadRequest,
        BrowseLoadStatus,
        BrowseLoadTiming,
    )
    from xdart.gui.tabs.scattering.events import RunIdentity
    from xdart.gui.tabs.scattering.processed_browser import (
        TerminalBrowsePaintReceipt,
        TerminalPaintMode,
        TerminalRebindAuthorization,
    )

    request = BrowseLoadRequest("gui-timing", 1, "/out/a.nxs")
    worker = BrowseLoadTiming(
        "/out/a.nxs",
        "terminal", 0.1, 0.2, 0.3, 651, 0.4, 0.5, 0.6, 2.1,
    )
    clock = _StepClock((
        10.0,
        10.25, 10.5,
        11.0, 11.25,
        12.0, 12.25,
        12.5, 13.0,
        13.0, 14.0,
        16.0, 18.0,
    ))
    owner = _processed_owner(clock)
    capture = _loaded_capture(request)
    identity = RunIdentity(1, "gui-timing")
    handoff = owner.begin_terminal_handoff(
        request,
        identity,
        request.source_path,
        None,
        (),
        None,
        timing_start=owner.begin_terminal_timing(enabled=True),
    )
    assert handoff is not None
    for _ in range(3):
        poll_started = owner.begin_poll_timing(request)
        assert owner.finish_poll_timing(request, poll_started)
    outcome = BrowseLoadOutcome(
        request, BrowseLoadStatus.READY, timing=worker,
    )
    settle_started = owner.begin_settle_timing(request)
    settlement = owner.settle_terminal(outcome, capture)
    assert settlement is not None
    assert owner.finish_settle_timing(request, settle_started, outcome)
    authorization = TerminalRebindAuthorization(
        identity, handoff.source_artifact, request.source_path,
    )
    assert owner.authorize_terminal_rebind(
        settlement.presentation, authorization,
    )
    caplog.set_level(
        "INFO", logger="xdart.gui.tabs.scattering.processed_browser"
    )
    try:
        paint = owner.begin_terminal_paint(
            settlement.presentation,
            capture,
            owns_request=True,
            reuse_science=True,
        )
        assert paint is not None
        assert paint.mode is TerminalPaintMode.REBIND
        completion = owner.complete_terminal_paint(
            TerminalBrowsePaintReceipt(
                paint, True, mode == "repaint-fallback",
            )
        )
        assert completion.accepted
        if mode == "repaint-fallback":
            assert not completion.retired
            assert not [
                record for record in caplog.records
                if "[PERF-BROWSE]" in record.getMessage()
            ]
            paint = owner.begin_terminal_paint(
                settlement.presentation,
                capture,
                owns_request=True,
                reuse_science=False,
            )
            assert paint is not None
            assert paint.mode is TerminalPaintMode.REPAINT_FALLBACK
            completion = owner.complete_terminal_paint(
                TerminalBrowsePaintReceipt(paint, True, False)
            )
            assert completion.accepted and completion.retired
        else:
            assert completion.retired

        records = [
            record for record in caplog.records
            if "[PERF-BROWSE]" in record.getMessage()
        ]
        assert len(records) == 1
        message = records[0].getMessage()
        assert f"mode={mode}" in message
        assert "source=/out/a.nxs token=gui-timing generation=1" in message
        assert "seal=terminal records=651" in message
        assert "poll/adopt=0.750s(n=3)" in message
        assert "settle=0.500s" in message
        assert (
            "presentation=1.000s" if mode == "rebind"
            else "presentation=3.000s"
        ) in message
        assert not owner.terminal_perf_active
        assert owner.terminal_presentation is None
    finally:
        owner.begin_close()
        assert owner.retry_close()


def test_terminal_browse_stale_fallback_cannot_complete_new_request(
    caplog, tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadOutcome,
        BrowseLoadRequest,
        BrowseLoadStatus,
        BrowseLoadTiming,
    )
    from xdart.gui.tabs.scattering.events import RunIdentity
    from xdart.gui.tabs.scattering.processed_browser import (
        TerminalBrowsePaintReceipt,
        TerminalPaintMode,
        TerminalRebindAuthorization,
    )

    source = str((tmp_path / "same.nxs").resolve())
    request_a = BrowseLoadRequest("run-a", 1, source)
    request_b = BrowseLoadRequest("run-b", 2, source)
    capture_a = _loaded_capture(request_a)
    capture_b = _loaded_capture(request_b)
    assert capture_a.context is not capture_b.context
    worker = BrowseLoadTiming(
        source,
        "terminal", 0.1, 0.2, 0.3, 3, 0.4, 0.5, 0.6, 2.1,
    )
    owner = _processed_owner(_StepClock((
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0,
        2.0, 2.1, 2.2, 2.5, 3.0,
    )))
    identity_a = RunIdentity(1, "run-a")
    identity_b = RunIdentity(2, "run-b")
    caplog.set_level(
        "INFO", logger="xdart.gui.tabs.scattering.processed_browser"
    )
    try:
        handoff_a = owner.begin_terminal_handoff(
            request_a,
            identity_a,
            source,
            None,
            (),
            None,
            timing_start=owner.begin_terminal_timing(enabled=True),
        )
        assert handoff_a is not None
        outcome_a = BrowseLoadOutcome(
            request_a, BrowseLoadStatus.READY, timing=worker,
        )
        settle_started = owner.begin_settle_timing(request_a)
        settlement_a = owner.settle_terminal(outcome_a, capture_a)
        assert settlement_a is not None
        assert owner.finish_settle_timing(
            request_a, settle_started, outcome_a,
        )
        authorization = TerminalRebindAuthorization(
            identity_a, handoff_a.source_artifact, request_a.source_path,
        )
        assert owner.authorize_terminal_rebind(
            settlement_a.presentation, authorization,
        )
        rebind = owner.begin_terminal_paint(
            settlement_a.presentation,
            capture_a,
            owns_request=True,
            reuse_science=True,
        )
        assert rebind is not None
        assert rebind.mode is TerminalPaintMode.REBIND
        fallback = owner.complete_terminal_paint(
            TerminalBrowsePaintReceipt(rebind, True, True)
        )
        assert fallback.accepted and not fallback.retired
        repaint = owner.begin_terminal_paint(
            settlement_a.presentation,
            capture_a,
            owns_request=True,
            reuse_science=False,
        )
        assert repaint is not None
        assert repaint.mode is TerminalPaintMode.REPAINT_FALLBACK
        stale_receipt = TerminalBrowsePaintReceipt(repaint, True, False)
        done_a = owner.complete_terminal_paint(stale_receipt)
        assert done_a.accepted and done_a.retired
        first = [
            record for record in caplog.records
            if "[PERF-BROWSE]" in record.getMessage()
        ]
        assert len(first) == 1
        assert "token=run-a generation=1" in first[0].getMessage()

        handoff_b = owner.begin_terminal_handoff(
            request_b,
            identity_b,
            source,
            None,
            (),
            None,
            timing_start=owner.begin_terminal_timing(enabled=True),
        )
        assert handoff_b is not None
        outcome_b = BrowseLoadOutcome(
            request_b, BrowseLoadStatus.READY, timing=worker,
        )
        settle_started = owner.begin_settle_timing(request_b)
        settlement_b = owner.settle_terminal(outcome_b, capture_b)
        assert settlement_b is not None
        assert owner.finish_settle_timing(
            request_b, settle_started, outcome_b,
        )
        paint_b = owner.begin_terminal_paint(
            settlement_b.presentation,
            capture_b,
            owns_request=True,
            reuse_science=False,
        )
        assert paint_b is not None
        assert paint_b.mode is TerminalPaintMode.REPAINT
        stale = owner.complete_terminal_paint(stale_receipt)
        assert not stale.accepted
        assert owner.terminal_presentation is settlement_b.presentation
        assert owner.terminal_request is request_b
        assert owner.terminal_perf_active
        assert len([
            record for record in caplog.records
            if "[PERF-BROWSE]" in record.getMessage()
        ]) == 1

        done_b = owner.complete_terminal_paint(
            TerminalBrowsePaintReceipt(paint_b, True, False)
        )
        assert done_b.accepted and done_b.retired
        records = [
            record for record in caplog.records
            if "[PERF-BROWSE]" in record.getMessage()
        ]
        assert len(records) == 2
        message = records[1].getMessage()
        assert f"source={Path(source).resolve()}" in message
        assert "token=run-b generation=2" in message
        assert "token=run-a" not in message
        assert records[0].getMessage() != records[1].getMessage()
        assert owner.terminal_presentation is None
        assert not owner.terminal_perf_active
    finally:
        owner.begin_close()
        assert owner.retry_close()


def test_terminal_browse_loading_status_restores_complete_after_repaint(
    qapp, monkeypatch, caplog,
) -> None:
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadOutcome,
        BrowseLoadRequest,
        BrowseLoadStatus,
        BrowseLoadTiming,
    )
    from xdart.gui.tabs.scattering.display_values import (
        StandardEventKind,
        StandardRunEvent,
    )
    from xdart.gui.tabs.scattering.processed_browser import (
        TerminalBrowsePaintReceipt,
        TerminalPaintMode,
    )
    from tests.xdart.scattering.test_e1b2_page_command_boundaries import (
        _Executor,
        _active_page,
        _dispose,
        _shell,
    )

    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    executor.events.append(StandardRunEvent(
        identity,
        StandardEventKind.FINISHED,
        completed=3,
        total=3,
        detail="Complete · 3 Frames",
    ))
    page._drain_executor()
    assert lifecycle.phase.value == "idle"
    assert _shell(page).run_controls.readinessLabel.full_text() == (
        "Complete · 3 Frames"
    )

    request = BrowseLoadRequest("terminal-status", 1, "/out/a.nxs")
    capture = _loaded_capture(request)
    pending = [True]
    controller = page._context_controller
    monkeypatch.setattr(
        type(controller), "browse_pending",
        property(lambda _self: pending[0]),
    )
    monkeypatch.setattr(
        controller, "owns_browse_request", lambda candidate: candidate is request,
    )
    monkeypatch.setattr(
        controller, "capture_loaded_browse",
        lambda candidate: capture if candidate is request else None,
    )
    assert page._start_permitted()[1] == "Browse cleanup remains pending"
    monkeypatch.setattr(
        page._processed_browser,
        "_clock",
        _StepClock((0, 10, 11, 12, 14, 15, 18)),
    )
    handoff = page._processed_browser.begin_terminal_handoff(
        request,
        identity,
        request.source_path,
        None,
        (),
        None,
        timing_start=page._processed_browser.begin_terminal_timing(
            enabled=True
        ),
    )
    assert handoff is not None
    timing = BrowseLoadTiming(
        "/out/a.nxs",
        "terminal", 0.1, 0.2, 0.3, 3, 0.4, 0.5, 0.6, 2.1,
    )
    outcome = BrowseLoadOutcome(
        request, BrowseLoadStatus.READY, timing=timing,
    )

    def poll_browse(*, reintegrate_successor_owner=None):
        assert reintegrate_successor_owner is None
        pending[0] = False
        return outcome

    monkeypatch.setattr(controller, "poll_browse", poll_browse)
    readiness_during_paint: list[str] = []
    receipts: list[TerminalBrowsePaintReceipt] = []
    real_complete = page._processed_browser.complete_terminal_paint

    def complete(receipt):
        assert type(receipt) is TerminalBrowsePaintReceipt
        receipts.append(receipt)
        readiness_during_paint.append(
            _shell(page).run_controls.readinessLabel.full_text()
        )
        return real_complete(receipt)

    monkeypatch.setattr(
        page._processed_browser, "complete_terminal_paint", complete,
    )
    caplog.set_level(
        "INFO", logger="xdart.gui.tabs.scattering.processed_browser"
    )
    try:
        page._refresh_shell(
            preserve_scientific=True,
            skip_scientific_projection=True,
        )
        assert _shell(page).run_controls.readinessLabel.full_text() == (
            "Loading finalized Browse context…"
        )

        page._drain_executor()

        assert readiness_during_paint == [
            "Loading finalized Browse context…"
        ]
        assert len(receipts) == 1
        assert receipts[0].request.mode is TerminalPaintMode.REPAINT
        assert receipts[0].applied
        assert not receipts[0].repaint_pending
        assert _shell(page).run_controls.readinessLabel.full_text() == (
            "Complete · 3 Frames"
        )
        records = [
            record for record in caplog.records
            if "[PERF-BROWSE]" in record.getMessage()
        ]
        assert len(records) == 1
        message = records[0].getMessage()
        assert "mode=repaint" in message
        assert "source=/out/a.nxs token=terminal-status generation=1" in message
        assert "gui-total=18.000s" in message
        assert "poll/adopt=1.000s(n=1)" in message
        assert "settle=2.000s" in message
        assert "presentation=3.000s" in message
        assert page._processed_browser.terminal_presentation is None
        assert not page._processed_browser.terminal_perf_active
    finally:
        _dispose(page, qapp)


@pytest.mark.parametrize(
    "admission",
    ("pending", "cache-debt", "refused"),
)
def test_scan_activation_reconciles_to_committed_browser_until_browse_adopts(
    qapp, monkeypatch, admission: str,
) -> None:
    """A clicked row is intent, not a committed Browse selection."""

    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest
    from tests.xdart.scattering.test_e1b2_page_command_boundaries import (
        _Executor,
        _active_page,
        _dispose,
    )

    page, _, _ = _active_page(_Executor())
    target = "/out/new-selection.nxs"
    request = BrowseLoadRequest("pending-selection", 1, target)
    refreshes: list[dict[str, object]] = []
    timers: list[bool] = []
    controller = page._context_controller
    monkeypatch.setattr(
        controller, "select_browser_target", lambda _value: False,
    )
    monkeypatch.setattr(
        page, "_refresh_shell",
        lambda **kwargs: refreshes.append(dict(kwargs)),
    )
    monkeypatch.setattr(page, "_ensure_timer", lambda: timers.append(True))
    if admission == "cache-debt":
        monkeypatch.setattr(page, "_release_browse_1d_debt", lambda: False)
        monkeypatch.setattr(
            controller,
            "begin_browse",
            lambda _value, *, source_root=None: pytest.fail(
                "cache debt reached Browse admission"
            ),
        )
    elif admission == "refused":
        monkeypatch.setattr(page, "_release_browse_1d_debt", lambda: True)

        def refuse(_value, *, source_root=None):
            raise RuntimeError("previous Browse cleanup is pending")

        monkeypatch.setattr(controller, "begin_browse", refuse)
    else:
        monkeypatch.setattr(page, "_release_browse_1d_debt", lambda: True)
        monkeypatch.setattr(
            controller,
            "begin_browse",
            lambda _value, *, source_root=None: request,
        )
    try:
        page._select_scan(target)

        # BrowserView paints the clicked QListWidget row before dispatching
        # SELECT_SCAN.  Every not-yet-adopted path must immediately replay the
        # authoritative projection so old science cannot wear the new row.
        assert refreshes == (
            [{}]
            if admission == "refused"
            else [{"preserve_scientific": True}]
        )
        assert timers == ([] if admission == "refused" else [True])
    finally:
        _dispose(page, qapp)
