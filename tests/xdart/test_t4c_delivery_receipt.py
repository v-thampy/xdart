"""O-1a-T4C (§36) — the finish-delivery receipt is MANDATORY and its evidence typed.

FROZEN CHECKPOINT ORACLE. §36.6 requires these tests to exist and be red at
`2109c008` before any production edit is authorized. They are the acceptance
contract for the single T-4C implementation commit, not for another symptom round.

The two defects §36 reproduced:

* §36.1 `wrangler_finished()` allocates one receipt and passes it to
  `_wrangler_finished_body`, but the body's later `_finalize_processing_run` call
  (ssw:14192) omits it.  In the all-idle shape the early projection has already
  run, so the omission looks inert.  It becomes ACTIVE in a real owner
  transition — Stitch active at the body's first observation, idle by the later
  finalizer — where that finalizer becomes the delivery's FIRST projection
  attempt, silently allocates a second receipt, and the outer `finally` then
  replays the whole projection against its own still-unattempted receipt: both
  successful and failed seams execute twice and the first-pass report is lost.
* §36.2 the retry loop stores the real recovery exception in
  ``receipt["retried"]``, but the reporter reduces retries to seam MEMBERSHIP and
  emits the INITIAL exception under a "recovery failed" label, so the actual
  recovery exception and its type never reach the operator.

Root cause (§36.4) is shape, not a forgotten keyword: receipt arguments default to
``None`` at multiple helpers, any omission silently allocates a second receipt,
and initial/retry exceptions live in parallel untyped collections that the
reporter then collapses.

RULE 10: every invocation must set ``XDART_SESSION_FILE`` to a unique path under
``/Users/vthampy/repos/tmp``.
"""

from __future__ import annotations

import ast
import gc
import inspect
import weakref

import pytest
from pyqtgraph.Qt import QtWidgets


_EXPECTED_DEFAULT_SEAMS = (
    # X1 O-3 (c3): the run-end context seams are four SEPARATELY NAMED,
    # independently retryable substeps — select the acquisition context,
    # finalize its scan exactly once, release the browse owner exactly once,
    # and release the acquisition context only once finalization succeeded.
    "select_acquisition_context",
    "finish_processing",
    "release_browse_context",
    "release_acquisition_context",
    "transient_reads",
    "set_processing_active",
    "set_run_writing",
    "set_open_enabled",
    "enable_integration",
    "advanced_widget_1d",
    "advanced_widget_2d",
    "invalidate_render_cache",
    "integration_control_state",
    "set_mode_row_enabled",
    "set_stop_enabled",
    "run_bookkeeping",
    "readiness_projection",
)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widget(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    value = staticWidget()
    try:
        yield value
    finally:
        try:
            value._exit_run_state(value._new_projection_receipt())
        except Exception:
            pass
        value.close()
        value.deleteLater()
        qapp.processEvents()


def _all_owners_idle(widget, monkeypatch):
    for thread in (widget.wrangler.thread,
                   widget.integratorTree.integrator_thread,
                   widget.stitch_thread):
        monkeypatch.setattr(thread, "isRunning", lambda: False)


def _capture_reports(monkeypatch):
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    reports = []
    monkeypatch.setattr(
        staticWidget, "_report_run_lifecycle_closure_failures",
        lambda _w, origin, failures, primary: reports.append(
            (origin, list(failures), primary)))
    return reports


def _trace_receipt_identity(monkeypatch):
    """Record the receipt object seen at EVERY projection-capable production hop.

    §36.6 item 3: entry -> rich body -> finalizer -> projection -> outer collector
    must all receive the SAME object allocated at the entry point."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    seen = []
    allocations = []
    real_new = staticWidget._new_projection_receipt

    def _new_receipt():
        receipt = real_new()
        allocations.append(receipt)
        return receipt

    monkeypatch.setattr(
        staticWidget, "_new_projection_receipt", staticmethod(_new_receipt))

    def _wrap(name, positional_index=None, keyword="receipt"):
        real = getattr(staticWidget, name)

        def _spy(self, *args, **kwargs):
            got = kwargs.get(keyword)
            if got is None and positional_index is not None:
                if len(args) > positional_index:
                    got = args[positional_index]
            seen.append((name, got))
            return real(self, *args, **kwargs)

        monkeypatch.setattr(staticWidget, name, _spy)

    _wrap("_wrangler_finished_body", positional_index=0)
    _wrap("_finalize_processing_run")
    _wrap("_exit_run_state", positional_index=0)
    _wrap("_run_idle_lifecycle_projection", positional_index=0)
    _wrap("_close_run_lifecycle")
    _wrap("_drive_exit_run_state", positional_index=0)
    return seen, allocations


def _explicit_seam_census(widget, monkeypatch):
    """Count the frozen default seam roster without deriving expectations from it."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    census = {seam: 0 for seam in _EXPECTED_DEFAULT_SEAMS}
    real_substeps = staticWidget._idle_lifecycle_substeps

    def _counting_substeps(self):
        out = []
        for seam, thunk in real_substeps(self):
            def _wrapped(seam=seam, thunk=thunk):
                census[seam] = census.get(seam, 0) + 1
                return thunk()

            out.append((seam, _wrapped))
        return tuple(out)

    monkeypatch.setattr(
        staticWidget, "_idle_lifecycle_substeps", _counting_substeps)
    return census


# --------------------------------------------------------------------------- #
# §36.6 items 1-2 — the two preserved reproducers, promoted in-tree.
# --------------------------------------------------------------------------- #

def test_wrangler_rich_body_threads_the_delivery_receipt(widget, monkeypatch):
    """§36.6 item 1a (promoted). The entry-point receipt must reach the rich
    body's later shared finalizer — ssw:14192 omits it today."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    _all_owners_idle(widget, monkeypatch)
    widget._enter_run_state()
    observed = []
    real = staticWidget._finalize_processing_run
    monkeypatch.setattr(staticWidget, "_owns_finished_run", lambda _s: True)

    def _observe(self, *, reset_overlay, origin, receipt=None):
        observed.append(receipt)
        return real(self, reset_overlay=reset_overlay, origin=origin,
                    receipt=receipt)

    monkeypatch.setattr(staticWidget, "_finalize_processing_run", _observe)
    widget.wrangler_finished()

    assert len(observed) == 1
    assert observed[0] is not None, (
        "the wrangler rich body lost the entry-point receipt")


def test_wrangler_owner_transition_neither_replays_nor_erases(
        widget, monkeypatch):
    """§36.6 items 1b + 4 (promoted). The PRODUCTION owner transition: Stitch
    active at the body's first observation, idle by the later finalizer.

    Each successful seam runs exactly once, the failed seam runs once plus only
    its explicit recovery, and the ordered report survives."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    widget._enter_run_state()
    monkeypatch.setattr(widget.wrangler.thread, "isRunning", lambda: False)
    monkeypatch.setattr(
        widget.integratorTree.integrator_thread, "isRunning", lambda: False)
    observations = {"count": 0}

    def _stitch_active_then_idle():
        observations["count"] += 1
        return observations["count"] == 1

    monkeypatch.setattr(
        widget.stitch_thread, "isRunning", _stitch_active_then_idle)
    monkeypatch.setattr(staticWidget, "_owns_finished_run", lambda _s: True)
    census = _explicit_seam_census(widget, monkeypatch)
    processing, writing = [], []
    initial = RuntimeError("one-shot wrangler writing failure")

    def _processing(value):
        processing.append(value)

    def _writing(value):
        writing.append(value)
        if len(writing) == 1:
            raise initial

    monkeypatch.setattr(
        widget.displayframe, "set_processing_active", _processing)
    monkeypatch.setattr(widget.h5viewer, "set_run_writing", _writing)
    reports = _capture_reports(monkeypatch)
    released = []
    real_release = staticWidget._clear_controls_v2_run_source_authority

    def _release(self):
        released.append(1)
        return real_release(self)

    monkeypatch.setattr(
        staticWidget, "_clear_controls_v2_run_source_authority", _release)
    allocations = []
    real_new = staticWidget._new_projection_receipt

    def _new_receipt():
        receipt = real_new()
        allocations.append(receipt)
        return receipt

    monkeypatch.setattr(
        staticWidget, "_new_projection_receipt", staticmethod(_new_receipt))

    with pytest.raises(RuntimeError, match="one-shot wrangler writing failure") as caught:
        widget.wrangler_finished()

    assert caught.value is initial
    assert len(allocations) == 1, "one wrangler delivery allocated multiple receipts"
    assert released == [1], "wrangler source authority was not released exactly once"
    assert processing == [False], (
        f"a successful seam was replayed: set_processing_active{processing}")
    assert writing == [False, False], (
        f"the failed seam did not get exactly one explicit recovery: {writing}")
    expected_census = {seam: 1 for seam in _EXPECTED_DEFAULT_SEAMS}
    expected_census["set_run_writing"] = 2
    assert census == expected_census
    assert reports == [
        ("wrangler", [("set_run_writing (recovered on retry)", initial)], True)
    ], f"the ordered first-pass report was changed or erased: {reports}"


def test_outstanding_recovery_failure_preserves_both_attempt_reasons(
        widget, monkeypatch):
    """§36.6 items 2 + 5 (promoted). An initial ``RuntimeError`` and a DIFFERENT
    retry ``ValueError`` must BOTH survive, distinctly typed and labelled."""
    _all_owners_idle(widget, monkeypatch)
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)
    widget._enter_run_state()
    attempts = []
    initial = RuntimeError("initial write release failure")
    recovery = ValueError("retry write release failure")

    def _fail_differently(*_a, **_k):
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            raise initial
        raise recovery

    monkeypatch.setattr(widget.h5viewer, "set_run_writing", _fail_differently)
    reports = _capture_reports(monkeypatch)

    with pytest.raises(RuntimeError, match="initial write release failure") as caught:
        widget.integrator_thread_finished()

    assert caught.value is initial
    assert attempts == [1, 2]
    assert len(reports) == 1
    evidence = [(label, type(exc).__name__, str(exc))
                for label, exc in reports[0][1]]
    assert evidence == [
        ("set_run_writing (initial failure)", "RuntimeError",
         "initial write release failure"),
        ("set_run_writing (recovery failed)", "ValueError",
         "retry write release failure"),
    ], f"recovery evidence was collapsed: {evidence}"
    assert reports[0][1][0][1] is initial
    assert reports[0][1][1][1] is recovery


# --------------------------------------------------------------------------- #
# §36.6 item 3 — exact receipt identity across all three owners.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("origin", ("wrangler", "integrator", "stitch"))
def test_one_receipt_object_reaches_every_projection_capable_hop(
        widget, monkeypatch, origin):
    """§36.6 item 3. Every production call to a projection-capable helper in one
    delivery must receive the SAME object allocated at the entry point."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    _all_owners_idle(widget, monkeypatch)
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)
    monkeypatch.setattr(staticWidget, "_owns_finished_run", lambda _s: True)
    widget._enter_run_state()
    seen, allocations = _trace_receipt_identity(monkeypatch)

    slot = {"wrangler": widget.wrangler_finished,
            "integrator": widget.integrator_thread_finished,
            "stitch": widget.stitch_thread_finished}[origin]
    slot()

    expected_hops = {
        "wrangler": [
            "_wrangler_finished_body",
            "_drive_exit_run_state",
            "_exit_run_state",
            "_run_idle_lifecycle_projection",
            "_finalize_processing_run",
            "_drive_exit_run_state",
            "_exit_run_state",
            "_close_run_lifecycle",
        ],
        "integrator": [
            "_finalize_processing_run",
            "_drive_exit_run_state",
            "_exit_run_state",
            "_run_idle_lifecycle_projection",
            "_close_run_lifecycle",
        ],
        "stitch": [
            "_drive_exit_run_state",
            "_exit_run_state",
            "_run_idle_lifecycle_projection",
            "_close_run_lifecycle",
        ],
    }
    assert [name for name, _ in seen] == expected_hops[origin]
    assert len(allocations) == 1, (
        f"{origin} delivery allocated {len(allocations)} receipts")
    missing = [name for name, got in seen if got is None]
    assert not missing, (
        f"{origin} delivery: these hops received no receipt: {missing}")
    assert all(got is allocations[0] for _name, got in seen), (
        f"{origin} delivery replaced its entry receipt")


# --------------------------------------------------------------------------- #
# §36.6 item 6 — a one-shot recovery reports "recovered", not "outstanding".
# --------------------------------------------------------------------------- #

def test_one_shot_recovery_reports_recovered_without_outstanding_entry(
        widget, monkeypatch):
    """§36.6 item 6. When recovery SUCCEEDS the initial reason stays visible as
    recovered, and no outstanding-recovery entry is emitted."""
    _all_owners_idle(widget, monkeypatch)
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)
    widget._enter_run_state()
    calls = []
    initial = RuntimeError("one-shot writing failure")

    def _one_shot(*_a, **_k):
        calls.append(1)
        if len(calls) == 1:
            raise initial

    monkeypatch.setattr(widget.h5viewer, "set_run_writing", _one_shot)
    reports = _capture_reports(monkeypatch)

    with pytest.raises(RuntimeError, match="one-shot writing failure") as caught:
        widget.integrator_thread_finished()

    assert caught.value is initial
    assert calls == [1, 1], "the failed seam did not receive exactly one retry"
    assert len(reports) == 1
    assert reports[0][1] == [
        ("set_run_writing (recovered on retry)", initial)
    ], f"one-shot recovery was misreported: {reports[0][1]}"


# --------------------------------------------------------------------------- #
# §36.6 items 7-8 — complete applicable-seam census, no permissive assertion.
# --------------------------------------------------------------------------- #

def test_no_projection_rich_failure_runs_every_applicable_seam_once(
        widget, monkeypatch):
    """§36.6 items 7 + 8. A rich failure BEFORE any projection attempt must still
    run source release and EVERY applicable seam exactly once — asserted as a
    complete census, replacing T-4.2 case 7's permissive one-or-zero check."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    monkeypatch.setattr(widget.wrangler.thread, "isRunning", lambda: False)
    monkeypatch.setattr(
        widget.integratorTree.integrator_thread, "isRunning", lambda: False)
    observations = {"count": 0}

    def _stitch_active_then_idle():
        observations["count"] += 1
        return observations["count"] == 1

    monkeypatch.setattr(
        widget.stitch_thread, "isRunning", _stitch_active_then_idle)
    widget.wrangler.source_run_plan = "PLAN"
    widget._enter_run_state()

    census = _explicit_seam_census(widget, monkeypatch)
    released = []
    real_release = staticWidget._clear_controls_v2_run_source_authority
    monkeypatch.setattr(
        staticWidget, "_clear_controls_v2_run_source_authority",
        lambda w: (released.append(1), real_release(w))[1])
    # Fail the wrangler rich body after the early hold and before its later
    # projection. The outer closure observes Stitch idle, releases wrangler-owned
    # source authority, and must drive the explicit default seam roster once.
    monkeypatch.setattr(
        widget, "_flush_pending_update",
        lambda: (_ for _ in ()).throw(
            RuntimeError("rich failure before projection")))

    with pytest.raises(RuntimeError, match="rich failure before projection"):
        widget.wrangler_finished()

    assert released == [1], (
        "wrangler source authority was not released exactly once")
    assert census == {seam: 1 for seam in _EXPECTED_DEFAULT_SEAMS}, (
        f"applicable-seam census is not exactly-once: {census}")


def test_receipt_contract_is_typed_mandatory_and_allocated_only_at_entries():
    """§36.5/§36.6: make the structural design executable, not documentary."""
    from xdart.gui.tabs.static_scan import static_scan_widget as module

    receipt_type = getattr(module, "_ProjectionReceipt", None)
    attempt_type = getattr(module, "_SeamAttempt", None)
    assert isinstance(receipt_type, type)
    assert isinstance(attempt_type, type)

    helper_names = (
        "_exit_run_state",
        "_run_idle_lifecycle_projection",
        "_drive_exit_run_state",
        "_close_run_lifecycle",
        "_finalize_processing_run",
        "_wrangler_finished_body",
    )
    for name in helper_names:
        parameter = inspect.signature(
            getattr(module.staticWidget, name)).parameters["receipt"]
        assert parameter.default is inspect.Parameter.empty, (
            f"{name} still permits a receipt-less production call")
        assert parameter.annotation is receipt_type, (
            f"{name} does not require the typed delivery receipt")

    receipt = module.staticWidget._new_projection_receipt()
    assert type(receipt) is receipt_type
    assert type(receipt.attempts) is dict
    attempt = attempt_type("probe")
    assert (attempt.seam, attempt.initial_error, attempt.recovery_error,
            attempt.recovery_attempted, attempt.completed) == (
                "probe", None, None, False, False)

    source = inspect.getsource(module.staticWidget)
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "staticWidget")
    allocation_sites = []
    missing_receipt_calls = []
    projection_calls = set(helper_names)
    for method in class_node.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in (node for node in ast.walk(method)
                     if isinstance(node, ast.Call)):
            target = getattr(call.func, "attr", None)
            if target == "_new_projection_receipt":
                allocation_sites.append(method.name)
            if target in projection_calls:
                carries_receipt = any(
                    isinstance(node, ast.Name) and node.id == "receipt"
                    for node in ast.walk(call)
                )
                if not carries_receipt:
                    missing_receipt_calls.append((method.name, target))

    # X1 O-3 (c3R.1 §10.4.1): public Close is a fourth legitimate LOCAL
    # allocation site.  A run whose finalization failed twice is retained as a
    # cleanup owner, and Close is the qualified delivery that retries it — with
    # its own per-delivery receipt, dropped on return like every other one.
    # Nothing is stored on the widget; the pin below still forbids that.
    assert sorted(allocation_sites) == [
        "_drive_residual_cleanup_on_close",
        "integrator_thread_finished",
        "stitch_thread_finished",
        "wrangler_finished",
    ]
    assert missing_receipt_calls == []
    assert "_run_lifecycle_restored" not in source

    drive_source = inspect.getsource(module.staticWidget._drive_exit_run_state)
    assert "__func__" not in drive_source
    drive_tree = ast.parse(inspect.cleandoc(drive_source))
    exit_calls = [
        node for node in ast.walk(drive_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "exit_fn"
    ]
    assert exit_calls and all(len(call.args) == 1 for call in exit_calls)


# --------------------------------------------------------------------------- #
# §36.6 item 9 — the receipt is delivery-local and leaves no trace.
# --------------------------------------------------------------------------- #

def test_receipt_is_not_retained_anywhere_after_the_delivery(
        widget, monkeypatch):
    """§36.6 item 9. No widget attribute, timer, or callback may hold the receipt
    after the delivery returns, and a later duplicate delivery must not see it."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    _all_owners_idle(widget, monkeypatch)
    monkeypatch.setattr(widget, "thread_state_changed", lambda: None)
    monkeypatch.setattr(staticWidget, "_owns_finished_run", lambda _s: True)
    widget._enter_run_state()
    seen, allocations = _trace_receipt_identity(monkeypatch)
    widget.integrator_thread_finished()

    assert seen, "no hop observed"
    assert len(allocations) == 1
    delivered = allocations[0]
    delivered_ref = weakref.ref(delivered)
    delivered_id = id(delivered)

    held = [name for name, value in vars(widget).items() if value is delivered]
    assert held == [], f"the receipt is retained on the widget as {held}"

    # A later duplicate delivery must allocate its own, not inherit this one.
    seen.clear()
    allocations.clear()
    widget.integrator_thread_finished()
    assert len(allocations) == 1
    later = allocations[0]
    assert later is not delivered, (
        "a later delivery inherited the previous delivery's receipt")

    # Clear every test-owned holder. A weakref surviving collection would prove
    # retention through a widget child, closure, timer, callback, or other owner.
    seen.clear()
    allocations.clear()
    del later
    del delivered
    gc.collect()
    assert delivered_ref() is None, (
        f"delivery receipt {delivered_id} remained reachable after the delivery")
