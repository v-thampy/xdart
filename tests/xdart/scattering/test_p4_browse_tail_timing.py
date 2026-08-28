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

    clock = _StepClock(range(14))
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
        13.0,
    )
    assert gate_calls == [True]
    assert clock.calls == 14
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
    qapp, tmp_path, caplog, monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.adapters import browse_loader as module
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadStatus,
        BrowseLoadTiming,
    )
    from xdart.gui.tabs.scattering.page import (
        _TerminalBrowsePerf,
        _TerminalBrowsePresentation,
    )
    from tests.xdart.scattering.test_e1b2_page_command_boundaries import (
        _Executor,
        _active_page,
        _dispose,
    )

    seeded = _seed_existing(tmp_path)
    alias = tmp_path / "terminal-alias.nxs"
    alias.symlink_to(seeded.target)
    canonical = str(seeded.target.resolve())
    request = BrowseLoadRequest("alias-terminal", 7, str(alias))
    assert request.source_path != canonical
    loader = module.BrowseLoader(
        clock=_StepClock(range(14)),
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

    page, _, _ = _active_page(_Executor())
    owner = _TerminalBrowsePresentation(request, context)
    perf = _TerminalBrowsePerf(request, 30.0, worker=outcome.timing)
    page._terminal_browse_presentation = owner
    page._terminal_browse_perf = perf
    page._browse_clock = _StepClock((31.0,))
    controller = page._context_controller
    monkeypatch.setattr(
        controller,
        "owns_browse_request",
        lambda candidate: candidate is request,
    )
    monkeypatch.setattr(
        controller,
        "capture_reintegrate_browse",
        lambda: (context, request),
    )
    monkeypatch.setattr(page, "_refresh_event_shell", lambda **_kw: None)
    caplog.set_level("INFO", logger="xdart.gui.tabs.scattering.page")
    gui_resolves: list[str] = []

    def forbidden_gui_resolve(path, *args, **kwargs):
        gui_resolves.append(str(path))
        raise AssertionError("GUI terminal presentation resolved a path")

    try:
        with monkeypatch.context() as gui_guard:
            gui_guard.setattr(Path, "resolve", forbidden_gui_resolve)
            page._finish_terminal_browse_presentation(
                owner,
                perf,
                started=30.5,
                mode="repaint",
                applied=True,
            )
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
        _dispose(page, qapp)


@pytest.mark.parametrize("fail_at", (1, 14), ids=("start", "final"))
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
    qapp, caplog, monkeypatch, mode: str,
) -> None:
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadTiming,
    )
    from xdart.gui.tabs.scattering.page import (
        _TerminalBrowsePerf,
        _TerminalBrowsePresentation,
    )
    from tests.xdart.scattering.test_e1b2_page_command_boundaries import (
        _Executor,
        _active_page,
        _dispose,
    )

    executor = _Executor()
    page, _, _ = _active_page(executor)
    request = BrowseLoadRequest("gui-timing", 1, "/out/a.nxs")
    worker = BrowseLoadTiming(
        "/out/a.nxs",
        "terminal", 0.1, 0.2, 0.3, 651, 0.4, 0.5, 0.6, 2.1,
    )
    perf = _TerminalBrowsePerf(
        request,
        10.0,
        poll_adopt_count=3,
        poll_adopt_s=0.75,
        settle_s=0.5,
        worker=worker,
    )
    context = object()
    owner = _TerminalBrowsePresentation(request, context)
    page._terminal_browse_presentation = owner
    page._terminal_browse_perf = perf
    controller = page._context_controller
    monkeypatch.setattr(
        controller, "owns_browse_request",
        lambda candidate: candidate is request,
    )
    monkeypatch.setattr(
        controller, "capture_reintegrate_browse",
        lambda: (context, request),
    )
    monkeypatch.setattr(page, "_refresh_event_shell", lambda **_kw: None)
    caplog.set_level("INFO", logger="xdart.gui.tabs.scattering.page")
    try:
        if mode == "rebind":
            page._scientific_repaint_pending = False
            page._browse_clock = _StepClock((14.0,))
            page._finish_terminal_browse_presentation(
                owner, perf, started=13.0, mode="rebind", applied=True,
            )
            assert perf.presentation_s == 1.0
        else:
            page._scientific_repaint_pending = True
            page._browse_clock = _StepClock((14.0, 18.0))
            page._finish_terminal_browse_presentation(
                owner, perf, started=13.0, mode="rebind", applied=True,
            )
            assert perf.fallback_pending
            assert not [
                record for record in caplog.records
                if "[PERF-BROWSE]" in record.getMessage()
            ]
            page._scientific_repaint_pending = False
            page._finish_terminal_browse_presentation(
                owner,
                perf,
                started=16.0,
                mode="repaint-fallback",
                applied=True,
            )
            assert perf.presentation_s == 3.0

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
        assert page._terminal_browse_perf is None
        assert page._terminal_browse_presentation is None
    finally:
        _dispose(page, qapp)


def test_terminal_browse_stale_fallback_cannot_complete_new_request(
    qapp, caplog, monkeypatch, tmp_path,
) -> None:
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest,
        BrowseLoadTiming,
    )
    from xdart.gui.tabs.scattering.page import (
        _TerminalBrowsePerf,
        _TerminalBrowsePresentation,
    )
    from tests.xdart.scattering.test_e1b2_page_command_boundaries import (
        _Executor,
        _active_page,
        _dispose,
    )

    page, _, _ = _active_page(_Executor())
    source = str((tmp_path / "same.nxs").resolve())
    request_a = BrowseLoadRequest("run-a", 1, source)
    request_b = BrowseLoadRequest("run-b", 2, source)
    context_a, context_b = object(), object()
    owner_a = _TerminalBrowsePresentation(request_a, context_a)
    owner_b = _TerminalBrowsePresentation(request_b, context_b)
    worker = BrowseLoadTiming(
        source,
        "terminal", 0.1, 0.2, 0.3, 3, 0.4, 0.5, 0.6, 2.1,
    )
    perf_a = _TerminalBrowsePerf(request_a, 1.0, worker=worker)
    perf_b = _TerminalBrowsePerf(request_b, 2.0, worker=worker)
    controller = page._context_controller
    active = {"request": request_a, "context": context_a}
    monkeypatch.setattr(
        controller, "owns_browse_request",
        lambda candidate: candidate is active["request"],
    )
    monkeypatch.setattr(
        controller, "capture_reintegrate_browse",
        lambda: (active["context"], active["request"]),
    )
    monkeypatch.setattr(page, "_refresh_event_shell", lambda **_kw: None)
    page._terminal_browse_presentation = owner_a
    page._terminal_browse_perf = perf_a
    page._browse_clock = _StepClock((2.0, 3.0))
    caplog.set_level("INFO", logger="xdart.gui.tabs.scattering.page")
    try:
        page._finish_terminal_browse_presentation(
            owner_a, perf_a, started=1.5, mode="repaint", applied=True,
        )
        first = [
            record for record in caplog.records
            if "[PERF-BROWSE]" in record.getMessage()
        ]
        assert len(first) == 1
        assert "token=run-a generation=1" in first[0].getMessage()

        active.update(request=request_b, context=context_b)
        page._terminal_browse_presentation = owner_b
        page._terminal_browse_perf = perf_b
        page._finish_terminal_browse_presentation(
            owner_a, perf_a, started=2.5, mode="repaint-fallback",
            applied=True,
        )
        assert page._terminal_browse_presentation is owner_b
        assert page._terminal_browse_perf is perf_b
        assert len([
            record for record in caplog.records
            if "[PERF-BROWSE]" in record.getMessage()
        ]) == 1

        page._finish_terminal_browse_presentation(
            owner_b, perf_b, started=2.5, mode="repaint", applied=True,
        )
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
        assert page._terminal_browse_presentation is None
        assert page._terminal_browse_perf is None
    finally:
        _dispose(page, qapp)


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
    from xdart.gui.tabs.scattering.page import (
        _TerminalBrowseHandoff,
        _TerminalBrowsePerf,
        _TerminalBrowsePresentation,
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
    context = object()
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
        controller, "capture_reintegrate_browse",
        lambda: (context, request),
    )
    assert page._start_permitted()[1] == "Browse cleanup remains pending"
    page._terminal_browse_handoff = _TerminalBrowseHandoff(
        request, identity, request.source_path, None, (),
    )
    perf = _TerminalBrowsePerf(request, 0.0)
    page._terminal_browse_perf = perf
    timing = BrowseLoadTiming(
        "/out/a.nxs",
        "terminal", 0.1, 0.2, 0.3, 3, 0.4, 0.5, 0.6, 2.1,
    )
    outcome = BrowseLoadOutcome(
        request, BrowseLoadStatus.READY, timing=timing,
    )

    def poll_browse():
        pending[0] = False
        return outcome

    def settle(candidate, *, preserve_perf=False):
        assert candidate is outcome
        page._clear_terminal_browse(preserve_perf=preserve_perf)
        page._terminal_browse_presentation = _TerminalBrowsePresentation(
            request, context,
        )
        return False

    monkeypatch.setattr(controller, "poll_browse", poll_browse)
    monkeypatch.setattr(page, "_settle_terminal_browse", settle)
    readiness_during_paint = []
    real_finish = page._finish_terminal_browse_presentation

    def finish(*args, **kwargs):
        readiness_during_paint.append(
            _shell(page).run_controls.readinessLabel.full_text()
        )
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(page, "_finish_terminal_browse_presentation", finish)
    page._browse_clock = _StepClock((10, 11, 12, 14, 15, 18))
    caplog.set_level("INFO", logger="xdart.gui.tabs.scattering.page")
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
            lambda _value: pytest.fail("cache debt reached Browse admission"),
        )
    elif admission == "refused":
        monkeypatch.setattr(page, "_release_browse_1d_debt", lambda: True)

        def refuse(_value):
            raise RuntimeError("previous Browse cleanup is pending")

        monkeypatch.setattr(controller, "begin_browse", refuse)
    else:
        monkeypatch.setattr(page, "_release_browse_1d_debt", lambda: True)
        monkeypatch.setattr(controller, "begin_browse", lambda _value: request)
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
