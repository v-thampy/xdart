# -*- coding: utf-8 -*-
"""X1 O-3 c3R — dependency-aware run-end release, and hydration commit ownership.

FROZEN BEFORE the c3R production edits (§9.4).  Every case here is red at
`8af41539` and states the contract the two corrections must satisfy.

Two owner-graph defects are pinned:

**§9.1 — the run-end projection released owners after a failed prerequisite.**
The finalization claim was consumed BEFORE the fallible finalizer, so a failure
left the context un-finalized and unable to retry, while the release seam —
which checked only that the claim had been taken — dropped it anyway.  One seam
earlier, a failed select-A did not stop the independent release-B, so B could be
cleared while the display still named B.  Dependent seams stay separately named,
but each must FAIL CLOSED until its prerequisite is complete.

**§9.2 — hydration ownership was checked after a mutable store had been chosen
and already changed.**  The worker resolved its stores from a live provider at
EXECUTION time, so a request built under B could land in A; admission ran only
when the completion returned to the GUI, after the payload was already inserted;
and the dedupe key omitted the context, so the same label in two sub-scans
collapsed to one request.

The four reviewer discriminators preserved at
``/Users/vthampy/repos/tmp/test_codex_o3_c3_exact_review.py`` are promoted here
verbatim in intent (§9.4 requires them as focused gates).

RULE 10: every invocation of this module must set ``XDART_SESSION_FILE`` to a
unique path under ``/Users/vthampy/repos/tmp``.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import Qt
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
from xdart.gui.tabs.static_scan.frame_hydration_worker import (
    FrameHydrationWorker,
)
from xdart.gui.tabs.static_scan.static_scan_widget import (
    RUN_ORIGIN_REINTEGRATE,
    staticWidget,
)
from xdart.modules.display_context import (
    AcquisitionContext,
    BrowseContext,
    ContextKind,
    DisplayContextError,
    DisplaySelection,
    new_context_token,
)

_DIRECT = Qt.QtCore.Qt.ConnectionType.DirectConnection


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

class _Store:
    """A store that records what was cleared and what was inserted."""

    def __init__(self, name="store"):
        self.name = name
        self.cleared = 0
        self.inserted = []

    def clear(self):
        self.cleared += 1


def _acquisition(scan=None, **overrides):
    fields = dict(
        context_token=new_context_token(ContextKind.ACQUISITION),
        run_configuration=None,
        config_generation=None,
        config_fingerprint="",
        run_scan_key="run-a",
        source_path="/data/run_a.nxs",
        scan=scan if scan is not None else object(),
        frame=None,
        frame_ids=[],
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store=_Store("A"),
    )
    fields.update(overrides)
    return AcquisitionContext(**fields)


def _browse(**overrides):
    token = new_context_token(ContextKind.BROWSE)
    fields = dict(
        context_token=token,
        load_generation=1,
        operation=None,
        requested_path="/data/browse_b.nxs",
        scan_key="browse-b",
        scan=object(),
        frame=None,
        frame_ids=["1"],
        frames={},
        viewer_rows_1d={},
        viewer_rows_2d={},
        publication_store=_Store("B"),
    )
    fields.update(overrides)
    return BrowseContext(**fields)


# --------------------------------------------------------------------------- #
# §9.1 — the finalization state machine
# --------------------------------------------------------------------------- #

def test_a_failed_finalization_returns_the_context_to_a_retryable_state():
    """A failed attempt must NOT consume the context's one chance.

    The parent took the one-shot claim before the fallible finalizer, so a
    failure left the context permanently unable to finalize while the release
    seam dropped it anyway — the scientific work was silently never done.
    """
    context = _acquisition()
    assert context.finalized is False

    first = context.begin_finalization()
    assert first is context.scan, "the first attempt was not granted"
    # A second attempt while one is IN PROGRESS must be refused.
    assert context.begin_finalization() is None
    context.fail_finalization()

    # Back to retryable — and the failure is remembered as a DIAGNOSTIC fact,
    # never as release authority.
    assert context.finalized is False
    assert context.finalization_attempts == 1
    second = context.begin_finalization()
    assert second is context.scan, "a failed attempt was not retryable"
    context.complete_finalization()
    assert context.finalized is True
    assert context.finalization_attempts == 2

    # A finalized context is never finalized twice.
    assert context.begin_finalization() is None
    assert context.finalization_attempts == 2


# --------------------------------------------------------------------------- #
# §9.4 — the four preserved reviewer discriminators
# --------------------------------------------------------------------------- #

def test_failed_finalization_retains_context_and_retries_only_that_seam():
    """Reviewer discriminator 1, promoted verbatim in intent."""
    calls = []

    def fail_once(*_args):
        calls.append("finish")
        if len(calls) == 1:
            raise RuntimeError("one-shot finalization failure")

    host = SimpleNamespace(
        _acquisition_context=_acquisition(),
        displayframe=SimpleNamespace(finish_processing=fail_once),
    )

    with pytest.raises(RuntimeError):
        staticWidget._finalize_acquisition_context_scan(host)

    with pytest.raises(DisplayContextError):
        staticWidget._release_acquisition_context(host)
    assert host._acquisition_context is not None

    staticWidget._finalize_acquisition_context_scan(host)
    assert calls == ["finish", "finish"]
    assert host._acquisition_context.finalized is True
    staticWidget._release_acquisition_context(host)
    assert host._acquisition_context is None


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widget(qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget as sw

    value = sw()
    try:
        yield value
    finally:
        try:
            value._acquisition_context = None
            value._browse_context = None
            value._display_selection = None
            value._run_active = False
        except Exception:
            pass
        try:
            value._controls_v2_refresh_timer.cancel()
        except Exception:
            pass
        value.close()
        value.deleteLater()
        qapp.processEvents()


def _browsing_run(widget, monkeypatch, tmp_path):
    """A real paused run with a real admitted browse selected."""
    widget._enter_run_state(origin=RUN_ORIGIN_REINTEGRATE)
    widget.h5viewer.paused_browse_active = True
    queued = []
    monkeypatch.setattr(widget.h5viewer, "_ensure_file_thread_running",
                        lambda: None)
    monkeypatch.setattr(widget.h5viewer.file_thread.queue, "put",
                        queued.append)
    context = widget._begin_paused_browse(str(tmp_path / "browsed.nxs"))
    widget._on_browse_loaded(queued[0])
    assert widget._display_selection.names(context)
    return context


def _seam_errors(receipt):
    """Every seam that reported a failure, with both error objects."""
    out = {}
    for seam, attempt in receipt.attempts.items():
        if attempt.initial_error is not None:
            out[seam] = (attempt.initial_error, attempt.recovery_error,
                         attempt.completed)
    return out


def test_a_failed_select_a_retains_b_and_releases_it_only_on_retry(
        widget, monkeypatch, tmp_path):
    """§9.1.7 case 1 — fail-once select A.

    Release-B is a DEPENDENT seam.  Clearing B while the display still names B
    is the mixed-context state §8.2 closed at the request seam; the run-end
    projection must not reopen it from the other end.
    """
    browse = _browsing_run(widget, monkeypatch, tmp_path)
    cleared = []
    real_clear = browse.publication_store.clear
    monkeypatch.setattr(browse.publication_store, "clear",
                        lambda: (cleared.append(1), real_clear())[1])
    calls = []
    real_select = staticWidget._select_acquisition_context

    def fail_once(self, *, origin=""):
        calls.append(origin)
        if len(calls) == 1:
            raise RuntimeError("one-shot select failure")
        return real_select(self, origin=origin)

    monkeypatch.setattr(staticWidget, "_select_acquisition_context", fail_once)

    receipt = widget._new_projection_receipt()
    with pytest.raises(RuntimeError, match="one-shot select failure"):
        widget._exit_run_state(receipt)

    # First pass: A was not selected, so B is RETAINED, not cleared.
    assert browse.released is False, "B was released after a failed select-A"
    assert cleared == []
    assert widget._browse_context is browse
    failures = _seam_errors(receipt)
    assert "select_acquisition_context" in failures
    assert "release_browse_context" in failures

    # Retry: A is selected, and only then is B released — exactly once.  The
    # whole dependent chain then completes in order, so both context owners
    # are gone and every seam that failed is recorded as recovered.
    staticWidget._run_idle_lifecycle_projection(widget, receipt)
    assert len(calls) == 2
    assert browse.released is True
    assert len(cleared) == 1, "B's store was cleared more than once"
    assert widget._browse_context is None
    assert widget._acquisition_context is None
    for seam in ("select_acquisition_context", "release_browse_context"):
        attempt = receipt.attempts[seam]
        assert attempt.initial_error is not None
        assert attempt.completed is True, f"{seam} never recovered"


def test_a_persistent_select_failure_retains_b_and_both_contexts(
        widget, monkeypatch, tmp_path):
    """§9.1.7 case 2 — both exception objects survive, nothing is released."""
    browse = _browsing_run(widget, monkeypatch, tmp_path)
    acquisition = widget._acquisition_context
    errors = [RuntimeError("initial select failure"),
              ValueError("recovery select failure")]

    attempts = []

    def failing(self, *, origin=""):
        index = min(len(attempts), len(errors) - 1)
        attempts.append(origin)
        raise errors[index]

    monkeypatch.setattr(staticWidget, "_select_acquisition_context", failing)

    receipt = widget._new_projection_receipt()
    with pytest.raises(RuntimeError, match="initial select failure"):
        widget._exit_run_state(receipt)
    staticWidget._run_idle_lifecycle_projection(widget, receipt)

    attempt = receipt.attempts["select_acquisition_context"]
    assert attempt.initial_error is errors[0]
    assert attempt.recovery_error is errors[1]
    assert attempt.completed is False
    assert browse.released is False
    assert widget._browse_context is browse
    assert widget._acquisition_context is acquisition


def test_a_fail_once_finalizer_finalizes_once_then_releases(
        widget, monkeypatch, tmp_path):
    """§9.1.7 case 3 — two attempts, ONE successful finalization."""
    widget._enter_run_state(origin=RUN_ORIGIN_REINTEGRATE)
    acquisition = widget._acquisition_context
    calls = []

    def fail_once(run_scan=None, run_scan_key=None):
        calls.append((run_scan, run_scan_key))
        if len(calls) == 1:
            raise RuntimeError("one-shot finalization failure")

    monkeypatch.setattr(widget.displayframe, "finish_processing", fail_once)

    receipt = widget._new_projection_receipt()
    with pytest.raises(RuntimeError, match="one-shot finalization failure"):
        widget._exit_run_state(receipt)
    assert widget._acquisition_context is acquisition, (
        "an un-finalized context was released")
    assert acquisition.finalized is False

    staticWidget._run_idle_lifecycle_projection(widget, receipt)
    assert len(calls) == 2
    assert acquisition.finalized is True
    assert acquisition.finalization_attempts == 2
    assert widget._acquisition_context is None


def test_a_persistent_finalizer_failure_retains_the_owner_and_refuses_a_new_run(
        widget, monkeypatch, tmp_path):
    """§9.1.7 case 4 — a cleanup-pending owner blocks the next Run."""
    widget._enter_run_state(origin=RUN_ORIGIN_REINTEGRATE)
    acquisition = widget._acquisition_context
    errors = [RuntimeError("initial finalize failure"),
              ValueError("recovery finalize failure")]

    def failing(run_scan=None, run_scan_key=None):
        index = min(failing.count, len(errors) - 1)
        failing.count += 1
        raise errors[index]

    failing.count = 0
    monkeypatch.setattr(widget.displayframe, "finish_processing", failing)

    receipt = widget._new_projection_receipt()
    with pytest.raises(RuntimeError, match="initial finalize failure"):
        widget._exit_run_state(receipt)
    staticWidget._run_idle_lifecycle_projection(widget, receipt)

    attempt = receipt.attempts["finish_processing"]
    assert attempt.initial_error is errors[0]
    assert attempt.recovery_error is errors[1]
    assert widget._acquisition_context is acquisition
    assert acquisition.finalized is False

    # A new Run must not overwrite a retained cleanup owner.
    with pytest.raises(DisplayContextError):
        widget._enter_run_state(origin=RUN_ORIGIN_REINTEGRATE)
    assert widget._acquisition_context is acquisition
    assert widget._run_active is False


def test_public_close_retries_the_retained_cleanup_owner(
        widget, monkeypatch, tmp_path):
    """§10.4.1/10.4.4 — the REAL Close entry, not the private helper.

    The residual path existed but only ``_close_run_lifecycle()`` reached it,
    and public ``close()`` never called that.  A run whose finalization failed
    twice was therefore abandoned at Close with its stamp never written — and
    the in-tree case that called itself a Close test proved nothing about it,
    because it invoked the private helper directly.
    """
    widget._enter_run_state(origin=RUN_ORIGIN_REINTEGRATE)
    acquisition = widget._acquisition_context
    finish_calls = []
    writing_calls = []

    def fail_twice(run_scan=None, run_scan_key=None):
        finish_calls.append(1)
        if len(finish_calls) <= 2:
            raise RuntimeError("finalization failure")

    monkeypatch.setattr(widget.displayframe, "finish_processing", fail_twice)
    monkeypatch.setattr(widget.h5viewer, "set_run_writing",
                        lambda active: writing_calls.append(active))

    receipt = widget._new_projection_receipt()
    with pytest.raises(RuntimeError):
        widget._exit_run_state(receipt)
    staticWidget._run_idle_lifecycle_projection(widget, receipt)
    assert widget._run_active is False
    assert widget._acquisition_context is acquisition
    assert len(finish_calls) == 2
    writes_before_close = len(writing_calls)

    widget.close()

    assert len(finish_calls) == 3, "public Close made no third attempt"
    assert acquisition.finalized is True
    assert widget._acquisition_context is None
    assert len(writing_calls) == writes_before_close, (
        "Close replayed an already-successful unrelated seam")
    # The retired owner's commit authority is withdrawn with it.
    assert acquisition.commit_gate.cancelled is True


def test_every_public_run_action_refuses_a_retained_cleanup_owner(
        widget, monkeypatch, tmp_path):
    """§11.3.4 — the FIVE public actions, driven for real.

    The landed matrix was false-green: it replaced calibration loading with a
    pure lambda, called the private Stitch predicate instead of
    ``start_stitch()``, and never drove the image or NeXus Start entries.  Here
    every action is the real public entry; only the DOWNSTREAM worker boundary
    is observed.  Nothing may load calibration, mutate a carrier or start a
    thread while a cleanup owner is retained.
    """
    widget._enter_run_state(origin=RUN_ORIGIN_REINTEGRATE)
    retained = widget._acquisition_context
    widget._run_active = False
    assert widget._controls_v2_active_run_owner() == "context-cleanup"

    tree = widget.integratorTree
    events = []
    # Downstream observation only — the public entry and the admission ORDER
    # are real.  Calibration loading is watched, not replaced by a lambda that
    # cannot fail: it is the first thing the old check ran after.
    monkeypatch.setattr(tree, "_block_if_no_frames", lambda _dim: False)
    monkeypatch.setattr(tree, "_block_if_reload_only_frames",
                        lambda _dim: False)
    monkeypatch.setattr(tree, "_ensure_reintegration_calibration",
                        lambda _dim: events.append("calibration") or True)
    monkeypatch.setattr(tree, "_apply_gi_config_to_scan",
                        lambda: events.append("gi"))
    monkeypatch.setattr(tree, "_apply_threshold_config_to_thread",
                        lambda: events.append("thresholds"))
    monkeypatch.setattr(tree.integrator_thread, "start",
                        lambda: events.append("reintegrate-start"),
                        raising=False)
    monkeypatch.setattr(widget, "_build_stitch_params",
                        lambda mode: events.append("stitch-params") or {})
    monkeypatch.setattr(widget.stitch_thread, "start",
                        lambda: events.append("stitch-start"), raising=False)
    monkeypatch.setattr(widget.stitch_thread, "isRunning", lambda: False)
    monkeypatch.setattr(widget.wrangler.thread, "start",
                        lambda: events.append("wrangler-start"), raising=False)

    stitch_mode = getattr(widget.stitch_thread, "mode", None)
    stitch_params = getattr(widget.stitch_thread, "params", None)
    scan_name = widget.scan.name
    args_before = dict(widget.scan.bai_1d_args)

    # 1 + 2: both reintegration entries.
    tree.bai_1d(None)
    tree.bai_2d(None)
    # 3: Stitch, through its real public entry.
    widget.start_stitch("1d")
    # 4 + 5: image Start and NeXus Start, through the real wrangler entries.
    for wrangler in _available_wranglers(widget):
        wrangler.start()

    assert events == [], f"a public action acted on a retained owner: {events}"
    assert getattr(widget.stitch_thread, "mode", None) == stitch_mode
    assert getattr(widget.stitch_thread, "params", None) == stitch_params
    assert widget.scan.name == scan_name
    assert dict(widget.scan.bai_1d_args) == args_before
    assert widget._acquisition_context is retained
    assert widget._run_active is False

    # The defensive second check still refuses at the run-state owner.
    with pytest.raises(DisplayContextError):
        widget._enter_run_state(origin=RUN_ORIGIN_REINTEGRATE)
    assert widget._run_active is False


def test_a_raising_owner_probe_refuses_every_reintegration_action(
        widget, monkeypatch, tmp_path):
    """§11.3.2 — an INSTALLED but broken predicate is never permission."""
    tree = widget.integratorTree
    events = []
    monkeypatch.setattr(tree, "_block_if_no_frames", lambda _dim: False)
    monkeypatch.setattr(tree, "_block_if_reload_only_frames",
                        lambda _dim: False)
    monkeypatch.setattr(tree, "_ensure_reintegration_calibration",
                        lambda _dim: events.append("calibration") or True)
    monkeypatch.setattr(tree, "_apply_gi_config_to_scan",
                        lambda: events.append("gi"))
    monkeypatch.setattr(tree, "_apply_threshold_config_to_thread",
                        lambda: events.append("thresholds"))
    monkeypatch.setattr(tree.integrator_thread, "start",
                        lambda: events.append("start"), raising=False)

    def broken(_action):
        raise RuntimeError("owner probe failed")

    monkeypatch.setattr(tree, "_refuse_run_action", broken, raising=False)

    tree.bai_1d(None)
    tree.bai_2d(None)
    assert events == [], (
        f"a failed owner probe was treated as permission: {events}")


def _available_wranglers(widget):
    """Every real wrangler entry this widget exposes (image and NeXus)."""
    found, seen = [], set()
    for name in ("wrangler",):
        one = getattr(widget, name, None)
        if one is not None and id(one) not in seen:
            seen.add(id(one))
            found.append(one)
    for one in getattr(widget, "_wranglers", ()) or ():
        if one is not None and id(one) not in seen:
            seen.add(id(one))
            found.append(one)
    return found
