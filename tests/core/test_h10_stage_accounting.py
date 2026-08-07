# -*- coding: utf-8 -*-
"""H10-C0 — the frozen finite oracle for session stage accounting (§6 of
``handoffs/h10_session_policy_handoff_2026-08-02.md``, docs authority
``1f98ac1e``).

The complete H10 packet is accepted only if the production-wired oracle pins
these TWELVE semantic rows.  All twelve meanings are frozen HERE AND NOW;
staging their production-wired tests later does not authorize changing them.

1.  accepted then cancelled before compute;
2.  compute succeeds but sink write raises;
3.  buffered write stays non-persisted/non-durable until receipt;
4.  an actual ``CompositeSink`` whose first child makes one target recoverable
    and whose later child raises leaves aggregate ``written`` absent, emits no
    receipt for the failed target/mode pairs, and retains the earlier exact
    target receipt;
5.  a 1D receipt does not make a fresh 2D mode durable;
6.  publication-dropped is terminal but never durable;
7.  replace/re-feed cannot inflate counts;
8.  a heavy mode evicts only after its current ``result_revision`` is durable;
    persisted-but-not-durable remains resident, including at final sweep;
9.  coordinated category allocations stay within the session envelope;
10. normal queue-full worker exit delivers every successfully enqueued,
    non-skipped source item exactly once before EOS;
11. repeated bulk-read failure/rewind records discovery and skip once; and
12. a real HDF5 read/write isolation probe preserves
    lock→pause→write→resume, including exceptions.

Executable staging is binding: rows 1–7 are H10-C1-owned and implemented in
THIS module; rows 8–9 receive production-wired red tests immediately before
H10-C2, row 12 before H10-C3, rows 10–11 before H10-C4; H10-C5 reruns all
twelve together at the terminal exact object.

The ``correction*`` tests below are the §12.4 supplemental C1 discriminators
for the primary review's five production defect clusters.  They do not
renumber or release frozen rows 8–12.

The ``correction_r2_*`` tests are the §13.4 round-2 discriminators (docs
authority ``4fa5187f``) for the primary review of the round-1 rework: the
acceptance-admission authority must fail LOUD — an admission failure becomes a
run failure that reaps the in-flight permit and the dispatched future exactly
once and publishes nothing, never a queued item without an exact accepted
attempt identity; outcome attempt/result identity is validated and
replay-idempotent — exactly one terminal outcome identity exists per
``(label, attempt_revision)``, forged/unaccepted/contradictory outcomes raise
atomically, exact replay is a no-op that cannot invalidate a durable receipt;
and cancellation diagnostics never subtract a distinct-label total from an
attempt total.  The owner census is NAME-based (§13.3): any imported
``StageLedger`` symbol and every production ``StageLedger(...)`` construction
count, regardless of the intermediary module, including inside the definition
module itself.

Frozen vocabulary (§4.1): stages are identity SETS, never increment-only
counters — *accepted* (the session accepted the item for processing),
*completed* (reduction reached a typed result or typed terminal disposition;
the per-item disposition distinguishes a result from failed /
cancelled-before-completion), *written* (the TOP-LEVEL run sink hook returned
successfully, including for a buffering sink; deliberately NOT
target-qualified, and never widened by one child of a partially failed
``CompositeSink``), *persisted* (exact label/result-mode pairs passed through
the persistence layer into the logical on-disk representation of one declared
target) and *durable* (the sink's successful flush/close/publish boundary
reports those exact pairs recoverable by the application — application-level
recoverability, NOT an fsync/power-loss guarantee).  Persistence and
durability receipts are target-qualified as well as mode-qualified, so target
persistence is NOT required to be a subset of aggregate ``written`` before H23
supplies cross-target transaction semantics.  A mode receipt may never widen
to a whole frame; ``FrameEvent.generation`` is a caller-owned render-staleness
stamp and is excluded from scientific persistence identity.

Every successful submit advances a monotonic per-label ``attempt_revision``
and makes that attempt's PENDING state ledger-visible before any
compute-outcome, sink-write or public completion callback can publish.
``accepted`` is the distinct-label projection; ``pending``, ``completed``,
``dispositions`` and ``errors`` are LATEST-attempt label projections; an older
overlapping attempt may still mint its own ``result_revision`` but can never
replace the latest attempt's disposition or supply its error.  Every recorded
outcome must name its exact accepted ``attempt_revision`` — an outcome can
never originate from an unaccepted label or attempt — and replay of the same
attempt is identity-checked: exact replay is a no-op, contradiction raises
atomically, and each produced mode mints at most one revision per outcome.
The public
compatibility projections are derived identity facts, not counters:
``frames_submitted`` is the number of distinct accepted labels and
``frames_completed`` / ``ReductionResult.n_processed`` the number of distinct
labels whose top-level sink hook returned successfully at least once.

Source observation (discovered/enqueued/skipped) is a separate identity axis
in the same public snapshot and never masquerades as processing acceptance.

Streaming submission is ONE publication transaction owned by a single private
per-item ticket (§18): it owns the acquired in-flight permit, the item's one
``PENDING -> ACCEPTED(attempt)``/``REJECTED`` decision, the Future the executor
returned, and the exact reversible scan-inventory staging.  Acceptance stays
where §4.1/§12.2.1 froze it — the authority runs on the caller thread only
after a Future exists, the writer can identify the same item and the inventory
is staged — and worker and writer effects stay gated until that one decision is
published.  A rejected item therefore needs NO Future method: whenever its
callable runs it observes the decision and exits before reduction,
``worker_process``, sink access, outcome or completion, and no scan-inventory,
queue, sink, outcome, completion, progress or ledger fact publishes for it.
The public executor contract is unchanged and unnarrowed — ``submit()`` plus a
Future with a blocking, no-argument ``result()`` — with one explicit
precondition: a streaming executor must be asynchronous (``submit()`` returns
before the submitted callable needs its decision).
"""
from __future__ import annotations

import ast
import dataclasses
import dis
import logging
import pathlib
import queue
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from typing import Any

import numpy as np
import pytest

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.reduction import (
    CancelToken,
    CompositeSink,
    Frame,
    Integration2DPlan,
    MemorySink,
    ReductionPlan,
    ReductionSession,
    Scan,
)
import xrd_tools.reduction.core as reduction_core
from xrd_tools.session import ScanSession

# Bounds every gate/probe wait in this module so no row can wedge a worker,
# the writer thread or pytest itself (§12.4.2).
GATE_TIMEOUT = 10.0


def _sa():
    """The stage-accounting contract module (imported inside tests so each
    oracle row reports its own red at a parent that lacks the contract —
    a per-row contract-absence failure, not a collection-time setup error)."""
    from xrd_tools.session import stage_accounting

    return stage_accounting


def _r1d(value: float) -> IntegrationResult1D:
    return IntegrationResult1D(radial=np.array([0.0, 1.0]),
                               intensity=np.array([value, value + 1.0]),
                               sigma=None, unit="q_A^-1")


def _r2d(value: float) -> IntegrationResult2D:
    return IntegrationResult2D(radial=np.array([0.0, 1.0]),
                               azimuthal=np.array([0.0, 1.0]),
                               intensity=np.full((2, 2), value),
                               sigma=None)


def _frames(n: int) -> list[Frame]:
    return [Frame(i, image=np.full((2, 2), i, dtype=float)) for i in range(n)]


@pytest.fixture(autouse=True)
def _fake_integrate(monkeypatch):
    # Integration is a trusted kernel (H10 §3); the seam under test is the
    # accounting pipeline around it — same fixture policy as test_session_api.
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    monkeypatch.setattr(reduction_core, "integrate_2d",
                        lambda image, ai, **kw: _r2d(float(np.sum(image))))


def _session(n=2, *, sink=None, obligations=("nexus",), plan=None,
             executor: object = 2, frames=None, **kw) -> ScanSession:
    return ScanSession(
        plan if plan is not None else ReductionPlan(integration_2d=None),
        Scan("h10", _frames(n) if frames is None else list(frames),
             integrator=object()),
        sink=MemorySink() if sink is None else sink,
        executor=executor,
        obligations=obligations,
        **kw,
    )


class _ExplodingWriteSink:
    """Real sink whose write hook raises for one frame index — the declared
    fallible boundary (sink write) failing, not a fake of the seam under test."""

    def __init__(self, fail_index: int) -> None:
        self.fail_index = fail_index
        self.frames: dict[int, object] = {}

    def begin(self, scan, plan) -> None:
        self.frames.clear()

    def write(self, frame, reduction) -> None:
        if int(frame.index) == self.fail_index:
            raise OSError("disk full: simulated sink write failure")
        self.frames[int(frame.index)] = reduction

    def finish(self, result) -> None:
        return None


class _ArmedFailSink:
    """Real sink that writes/replaces normally until ``fail_next`` is armed,
    then raises once from the top-level hook (declared fallible boundary)."""

    def __init__(self) -> None:
        self.frames: dict[int, object] = {}
        self.fail_next = False

    def begin(self, scan, plan) -> None:
        self.frames.clear()

    def write(self, frame, reduction) -> None:
        self._store(frame, reduction)

    def replace(self, frame, reduction) -> None:
        self._store(frame, reduction)

    def finish(self, result) -> None:
        return None

    def _store(self, frame, reduction) -> None:
        if self.fail_next:
            self.fail_next = False
            raise OSError("disk full: simulated sink write failure")
        self.frames[int(frame.index)] = reduction


class _TargetReceiptSink:
    """Real child sink that makes ONE declared target recoverable: its write
    emits exactly that target-qualified receipt through the public ledger API
    (what a real flush/close boundary owner does)."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.ledger: Any = None
        self.mode: Any = None
        self.written: list[int] = []

    def bind(self, ledger, mode) -> None:
        self.ledger, self.mode = ledger, mode

    def begin(self, scan, plan) -> None:
        self.written.clear()

    def write(self, frame, reduction) -> None:
        label = int(frame.index)
        self.ledger.record_durable(
            [self.ledger.receipt(label, self.mode, self.target)])
        self.written.append(label)

    def finish(self, result) -> None:
        return None


class _RaisingChildSink:
    """Real child sink whose write raises — the later child of the partially
    failing CompositeSink in row 4.  It emits NO receipt for its target."""

    def __init__(self, target: str) -> None:
        self.target = target

    def begin(self, scan, plan) -> None:
        return None

    def write(self, frame, reduction) -> None:
        raise OSError(f"{self.target}: simulated child write failure")

    def finish(self, result) -> None:
        return None


class _GatedPrepSink:
    """Real sink whose pool-side ``worker_process`` prep hook blocks on the
    n-th call — the production seam that lets a test order two overlapping
    attempts on one label deterministically."""

    def __init__(self, gates: dict[int, threading.Event]) -> None:
        self.gates = gates
        self.calls = 0
        self.frames: dict[int, object] = {}

    def begin(self, scan, plan) -> None:
        self.frames.clear()

    def worker_process(self, frame, reduction) -> None:
        self.calls += 1                       # one pool worker → serialized
        gate = self.gates.get(self.calls)
        if gate is not None:
            gate.wait(timeout=GATE_TIMEOUT)

    def write(self, frame, reduction) -> None:
        self.frames[int(frame.index)] = reduction

    def replace(self, frame, reduction) -> None:
        self.frames[int(frame.index)] = reduction

    def finish(self, result) -> None:
        return None


class _PrepFailSink:
    """Real sink whose pool-side ``worker_process`` prep raises for one index
    AFTER reduction produced a typed result."""

    def __init__(self, fail_index: int) -> None:
        self.fail_index = fail_index
        self.frames: dict[int, object] = {}

    def begin(self, scan, plan) -> None:
        self.frames.clear()

    def worker_process(self, frame, reduction) -> None:
        if int(frame.index) == self.fail_index:
            raise RuntimeError("thumbnail prep failed")

    def write(self, frame, reduction) -> None:
        self.frames[int(frame.index)] = reduction

    def finish(self, result) -> None:
        return None


# ── row 1 — accepted then cancelled before compute ──────────────────────────

def test_row1_accepted_then_cancelled_before_compute():
    """An ACCEPTED frame whose compute never ran ends as the typed terminal
    disposition CANCELLED_BEFORE_COMPLETION — distinguishable, never written,
    never persisted or durable, and never silently 'still pending'."""
    sa = _sa()
    gate = threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        pool.submit(gate.wait)                 # occupy the ONE worker
        sess = _session(1, executor=pool)
        assert sess.submit(_frames(1)[0]) is True   # accepted, queued behind gate
        sess.stop()                                 # cancel BEFORE compute starts
        gate.set()                                  # now let the worker reach it
        result = sess.finish()
        assert result.cancelled is True
    finally:
        gate.set()
        pool.shutdown(wait=True)

    snap = sess.accounting_snapshot()
    assert snap.accepted == frozenset({0})
    assert snap.dispositions[0] is sa.ItemDisposition.CANCELLED_BEFORE_COMPLETION
    # §4.1: "completed" = a typed result OR a typed terminal disposition — the
    # item is out of the pipeline; the disposition says HOW.  Nothing pends.
    assert 0 in snap.completed
    assert snap.pending == frozenset()
    assert snap.written == frozenset()
    assert snap.persisted == frozenset()
    assert snap.durable == frozenset()
    assert snap.mode_complete == frozenset()
    snap.verify_conservation()


# ── row 2 — compute succeeds but sink write raises ──────────────────────────

def test_row2_compute_success_with_failing_sink_write():
    """Compute success is observed from the per-item outcome receipt BEFORE the
    sink write, so a failed write yields completed-but-not-written — never
    inferred from accepted-minus-written counts, and never conflated with a
    compute failure."""
    sa = _sa()
    sink = _ExplodingWriteSink(fail_index=1)
    sess = _session(2, sink=sink)
    for fr in _frames(2):
        assert sess.submit(fr) is True
    result = sess.finish(raise_on_failure=False)
    assert result.failed is True                     # fail-loud preserved

    m1 = sa.ResultMode.one_d()
    snap = sess.accounting_snapshot()
    assert snap.accepted == frozenset({0, 1})
    assert snap.dispositions[0] is sa.ItemDisposition.COMPLETED
    assert snap.dispositions[1] is sa.ItemDisposition.COMPLETED   # compute DID succeed
    assert (0, m1) in snap.written
    assert (1, m1) not in snap.written               # the write is what failed
    assert snap.persisted == frozenset()
    assert snap.durable == frozenset()
    # The public completion projection counts distinct successfully written
    # labels, so it cannot represent frame 1 as completed work.
    assert sess.frames_completed == 1
    assert result.n_processed == 1
    assert snap.completed == frozenset({0, 1})
    snap.verify_conservation()


# ── row 3 — buffered write is not persisted/durable until receipt ────────────

def test_row3_buffered_write_stays_non_persisted_non_durable_until_receipt():
    """A buffering sink's successful write advances WRITTEN only.  Persisted /
    durable advance exclusively on explicit target-qualified receipts at the
    flush boundary — never implied by write success."""
    sa = _sa()
    sess = _session(2)                       # MemorySink buffers in memory
    for fr in _frames(2):
        sess.submit(fr)
    sess.finish()

    m1 = sa.ResultMode.one_d()
    snap = sess.accounting_snapshot()
    assert snap.written == frozenset({(0, m1), (1, m1)})
    assert snap.persisted == frozenset()
    assert snap.durable == frozenset()
    assert snap.mode_complete == frozenset()

    led = sess.accounting
    led.record_durable([led.receipt(0, m1, "nexus"),
                        led.receipt(1, m1, "nexus")])
    snap2 = sess.accounting_snapshot()
    assert snap2.durable == frozenset({(0, m1, "nexus"), (1, m1, "nexus")})
    assert snap2.persisted == frozenset({(0, m1, "nexus"), (1, m1, "nexus")})
    assert snap2.mode_complete == frozenset({0, 1})
    snap2.verify_conservation()


# ── row 4 — partial CompositeSink: no child success widens aggregate written ──

def test_row4_composite_sink_partial_failure_leaves_aggregate_written_absent():
    """An ACTUAL ``CompositeSink`` driven through ``ScanSession``: the first
    child makes one target recoverable and records that exact target-qualified
    receipt, the later child raises.  The run fails loudly, the aggregate
    ``(label, mode)`` write stays ABSENT (no child success widens it), the
    failed target emits nothing, and the earlier target receipt stays truthful
    — target persistence is deliberately not a subset of aggregate written
    until H23 supplies cross-target transaction semantics."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()
    recoverable = _TargetReceiptSink("nexus")
    failing = _RaisingChildSink("xye")
    sess = _session(1, sink=CompositeSink((recoverable, failing)),
                    obligations=("nexus", "xye"))
    recoverable.bind(sess.accounting, m1)

    assert sess.submit(_frames(1)[0]) is True
    result = sess.finish(raise_on_failure=False)
    assert result.failed is True
    assert "simulated child write failure" in (result.error or "")

    snap = sess.accounting_snapshot()
    assert recoverable.written == [0]                # the first child DID succeed
    assert snap.dispositions[0] is sa.ItemDisposition.COMPLETED
    assert (0, m1) not in snap.written               # aggregate hook never returned
    assert snap.written == frozenset()
    assert (0, m1, "nexus") in snap.persisted        # earlier receipt stays truthful
    assert (0, m1, "nexus") in snap.durable
    assert (0, m1, "xye") not in snap.persisted      # failed target emits NOTHING
    assert (0, m1, "xye") not in snap.durable
    assert snap.mode_complete == frozenset()         # the xye obligation is unmet
    assert sess.frames_completed == 0                # no top-level write returned
    assert result.n_processed == 0
    snap.verify_conservation()


# ── row 5 — a 1D receipt never widens to the 2D mode or the frame ────────────

def test_row5_one_d_receipt_does_not_make_fresh_two_d_mode_durable():
    """A mode receipt may never widen to the whole frame: with required modes
    {1d, 2d}, a durable 1D receipt leaves the 2D mode undurable and the frame
    NOT mode-complete."""
    sa = _sa()
    plan = ReductionPlan(integration_2d=Integration2DPlan(npt_rad=2, npt_azim=2))
    sess = _session(1, plan=plan)
    sess.submit(_frames(1)[0])
    sess.finish()

    m1, m2 = sa.ResultMode.one_d(), sa.ResultMode.two_d()
    snap = sess.accounting_snapshot()
    assert set(snap.required_modes) == {m1, m2}
    assert snap.written == frozenset({(0, m1), (0, m2)})

    led = sess.accounting
    led.record_durable([led.receipt(0, m1, "nexus")])
    snap2 = sess.accounting_snapshot()
    assert (0, m1, "nexus") in snap2.durable
    assert (0, m2, "nexus") not in snap2.durable     # no widening
    assert snap2.mode_complete == frozenset()        # 2D mode still owed

    led.record_durable([led.receipt(0, m2, "nexus")])
    assert sess.accounting_snapshot().mode_complete == frozenset({0})


# ── row 6 — publication-dropped is terminal but never durable ─────────────────

def test_row6_publication_dropped_is_terminal_never_persisted_or_durable():
    sa = _sa()
    sess = _session(1)
    sess.submit(_frames(1)[0])
    sess.finish()

    m1 = sa.ResultMode.one_d()
    led = sess.accounting
    receipt = led.receipt(0, m1, "nexus")     # minted before the drop
    led.record_publication_dropped(0, m1, expected_revision=1)

    snap = sess.accounting_snapshot()
    assert (0, m1) in snap.publication_dropped
    assert snap.persisted == frozenset()
    assert snap.durable == frozenset()
    assert snap.mode_complete == frozenset()
    with pytest.raises(ValueError):
        led.record_durable([receipt])         # a dropped mode can NEVER certify
    with pytest.raises(ValueError):
        led.record_persisted([receipt])
    snap2 = sess.accounting_snapshot()
    assert snap2.durable == frozenset() and snap2.persisted == frozenset()
    snap2.verify_conservation()


# ── row 7 — replace/re-feed cannot inflate counts ────────────────────────────

def test_row7_replace_refeed_cannot_inflate_counts_and_dirties_exact_mode():
    """Re-feeding an index is idempotent on every identity set AND on both
    public compatibility projections, mints a new ``result_revision`` for
    exactly the re-produced mode, and invalidates durability for exactly that
    mode at the old revision — a stale receipt can never resurrect it."""
    sa = _sa()
    sess = _session(1)
    m1 = sa.ResultMode.one_d()
    sess.submit(_frames(1)[0])
    assert sess.pause(timeout=GATE_TIMEOUT) is True   # drain: completion + write landed

    led = sess.accounting
    assert led.current_revision(0, m1) == 1
    stale = led.receipt(0, m1, "nexus")       # certifies revision 1
    led.record_durable([stale])
    assert sess.accounting_snapshot().mode_complete == frozenset({0})

    sess.resume()
    sess.submit(Frame(0, image=np.full((2, 2), 9.0)))   # replace re-feed
    result = sess.finish()

    snap = sess.accounting_snapshot()
    assert snap.accepted == frozenset({0})               # identity set: no inflation
    assert len(snap.accepted) == 1
    assert sess.frames_submitted == 1                    # distinct accepted labels
    assert sess.frames_completed == 1                    # distinct written labels
    assert result.n_processed == 1
    assert snap.attempt_revisions[0] == 2                # the attempt axis DID advance
    assert snap.revisions[(0, m1)] == 2                  # exact mode dirtied
    assert (0, m1) in snap.written                       # re-written at revision 2
    assert (0, m1, "nexus") not in snap.durable          # revision-1 receipt is stale
    assert snap.mode_complete == frozenset()

    led.record_durable([stale])                          # stale replay: no effect
    assert (0, m1, "nexus") not in sess.accounting_snapshot().durable

    led.record_durable([led.receipt(0, m1, "nexus")])    # fresh revision-2 receipt
    snap3 = sess.accounting_snapshot()
    assert (0, m1, "nexus") in snap3.durable
    assert snap3.mode_complete == frozenset({0})
    snap3.verify_conservation()


# ── §12.2.1 — acceptance is visible before any public completion callback ─────

def test_correction1_acceptance_is_ledger_visible_before_public_completion(
        monkeypatch):
    """Forced legal interleaving: the writer completes and publishes a frame
    while ``submit`` is still on the caller thread.  Acceptance (and the new
    attempt's PENDING state) must already be ledger-visible, so no observer can
    ever see completed/written work that was never accepted."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()
    observed: dict[str, Any] = {}
    published = threading.Event()
    real_submit = reduction_core.ReductionSession.submit

    def _submit_then_let_the_writer_finish(self, frame, image=None):
        accepted = real_submit(self, frame, image)
        if accepted:
            published.wait(timeout=GATE_TIMEOUT)
        return accepted

    monkeypatch.setattr(reduction_core.ReductionSession, "submit",
                        _submit_then_let_the_writer_finish)
    sess = _session(1, executor=1)

    def _observe(_event) -> None:
        observed["snapshot"] = sess.accounting_snapshot()
        observed["frames_completed"] = sess.frames_completed
        published.set()

    sess.on_frame_completed(_observe)
    assert sess.submit(_frames(1)[0]) is True
    sess.finish()

    assert published.is_set(), "the writer never published the completion"
    snap = observed["snapshot"]
    assert snap.accepted == frozenset({0})
    assert snap.attempt_revisions[0] == 1
    assert snap.dispositions[0] is sa.ItemDisposition.COMPLETED
    assert (0, m1) in snap.written
    assert observed["frames_completed"] == 1
    snap.verify_conservation()          # containment held DURING the callback


# ── §12.2.2 — a worker-prep failure is not a compute failure ─────────────────

def test_correction2_worker_prep_failure_keeps_the_typed_completed_result():
    """``worker_process`` runs on the pool AFTER reduction produced a typed
    result.  A prep failure keeps the typed COMPLETED outcome and its exact
    produced modes, is reported through the existing fail-loud run path, and
    leaves the aggregate write absent — it is never reported as a failed
    compute with no result revision."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()
    sink = _PrepFailSink(fail_index=1)
    sess = _session(2, sink=sink)
    for fr in _frames(2):
        assert sess.submit(fr) is True
    result = sess.finish(raise_on_failure=False)

    assert result.failed is True                        # fail-loud preserved
    assert "thumbnail prep failed" in (result.error or "")
    snap = sess.accounting_snapshot()
    assert snap.dispositions[1] is sa.ItemDisposition.COMPLETED
    assert snap.revisions[(1, m1)] == 1                 # the typed result exists
    assert 1 not in snap.errors                         # compute did not fail
    assert (1, m1) not in snap.written                  # ... and it was never written
    assert 1 not in sink.frames
    assert snap.dispositions[0] is sa.ItemDisposition.COMPLETED
    assert (0, m1) in snap.written
    assert sess.frames_completed == 1
    snap.verify_conservation()


# ── §12.2.3 — latest-attempt state under overlapping attempts ────────────────

def test_correction3_older_attempt_cannot_replace_the_latest_pending_view():
    """Two overlapping attempts on ONE label: the older attempt's COMPLETED
    outcome mints its own result revision and its write stays truthful, but it
    may not present itself as the latest attempt's disposition.  The latest
    attempt stays PENDING, and the prior result's certifications coexist
    truthfully with it until a new typed result exists."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()
    gate_1, gate_2 = threading.Event(), threading.Event()
    published = threading.Event()
    sink = _GatedPrepSink({1: gate_1, 2: gate_2})
    pool = ThreadPoolExecutor(max_workers=1)
    sess = _session(1, sink=sink, executor=pool, inflight_max=4)
    try:
        sess.on_frame_completed(lambda _e: published.set())
        assert sess.submit(_frames(1)[0]) is True                    # attempt 1
        assert sess.submit(Frame(0, image=np.full((2, 2), 4.0))) is True   # attempt 2
        gate_1.set()                              # attempt 1 computes and writes
        assert published.wait(timeout=GATE_TIMEOUT) is True

        snap = sess.accounting_snapshot()
        assert snap.dispositions[0] is sa.ItemDisposition.PENDING
        assert 0 in snap.pending and 0 not in snap.completed
        assert 0 not in snap.errors
        assert snap.attempt_revisions[0] == 2
        assert snap.revisions[(0, m1)] == 1       # the older attempt still minted it
        assert (0, m1) in snap.written            # ... and its write is truthful

        led = sess.accounting
        led.record_durable([led.receipt(0, m1, "nexus")])
        snap2 = sess.accounting_snapshot()
        assert (0, m1, "nexus") in snap2.durable  # coexists with a PENDING latest
        assert snap2.mode_complete == frozenset({0})
        assert snap2.dispositions[0] is sa.ItemDisposition.PENDING
        snap2.verify_conservation()
    finally:
        gate_1.set()
        gate_2.set()
        sess.finish(raise_on_failure=False, join_timeout=30)
        pool.shutdown(wait=True)

    snap3 = sess.accounting_snapshot()
    assert snap3.dispositions[0] is sa.ItemDisposition.COMPLETED     # attempt 2 landed
    assert snap3.revisions[(0, m1)] == 2
    assert snap3.mode_complete == frozenset()     # the revision-1 receipt went stale
    assert sess.frames_submitted == 1 and sess.frames_completed == 1
    snap3.verify_conservation()


def test_correction3_stale_error_cannot_attach_to_a_later_attempt(monkeypatch):
    """The latest attempt's error is absent unless that same attempt supplied
    one: an older attempt's compute failure may not become a later
    cancelled-before-completion attempt's error, and the later no-error
    disposition may not inherit the earlier failure string."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()
    token = CancelToken()
    gate = threading.Event()

    def _integrate(image, ai, **kw):
        if float(np.max(image)) == 0.0:           # attempt 1 only
            token.cancel()                        # a Stop lands as this frame fails
            raise RuntimeError("integration kernel failed")
        return _r1d(float(np.sum(image)))

    monkeypatch.setattr(reduction_core, "integrate_1d", _integrate)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        pool.submit(gate.wait)                    # occupy the ONE worker
        sess = _session(1, executor=pool, inflight_max=4, cancel_token=token)
        assert sess.submit(_frames(1)[0]) is True                     # attempt 1
        assert sess.submit(Frame(0, image=np.full((2, 2), 3.0))) is True   # attempt 2
        gate.set()
        result = sess.finish(raise_on_failure=False, join_timeout=30)
    finally:
        gate.set()
        pool.shutdown(wait=True)

    assert result.failed is True
    assert "integration kernel failed" in (result.error or "")
    snap = sess.accounting_snapshot()
    assert snap.dispositions[0] is sa.ItemDisposition.CANCELLED_BEFORE_COMPLETION
    assert 0 not in snap.errors                   # attempt 1's string is not attempt 2's
    assert snap.attempt_revisions[0] == 2
    assert (0, m1) not in snap.revisions          # neither attempt produced a result
    assert snap.written == frozenset()
    assert sess.frames_completed == 0
    snap.verify_conservation()


# ── §12.2.4 — public projections are derived identity facts ──────────────────

def test_correction4_public_projections_are_derived_identity_facts():
    """``frames_submitted``/``frames_completed``/``n_processed``/``ProgressEvent``
    are projections of the ledger's identity facts, never counters: a
    successful re-feed cannot inflate them, and a later replacement whose
    top-level sink hook fails cannot decrement the historical completion."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()
    progress: list[Any] = []
    sink = _ArmedFailSink()
    sess = _session(2, sink=sink)
    sess.on_progress(progress.append)
    for fr in _frames(2):
        assert sess.submit(fr) is True
    assert sess.pause(timeout=GATE_TIMEOUT) is True
    assert (sess.frames_submitted, sess.frames_completed) == (2, 2)

    # (a) a successful re-feed cannot inflate either public total
    sess.resume()
    assert sess.submit(Frame(0, image=np.full((2, 2), 5.0))) is True
    assert sess.pause(timeout=GATE_TIMEOUT) is True
    assert sess.frames_submitted == 2
    assert sess.frames_completed == 2
    assert (progress[-1].submitted, progress[-1].completed) == (2, 2)

    # (b) a replacement that mints a new revision but whose top-level sink hook
    #     FAILS cannot decrement the historical completion or progress
    sess.resume()
    sink.fail_next = True
    assert sess.submit(Frame(0, image=np.full((2, 2), 7.0))) is True
    result = sess.finish(raise_on_failure=False)

    assert result.failed is True
    assert sess.frames_submitted == 2
    assert sess.frames_completed == 2
    assert result.n_processed == 2
    assert (progress[-1].submitted, progress[-1].completed) == (2, 2)
    snap = sess.accounting_snapshot()
    assert snap.accepted == frozenset({0, 1})
    assert dict(snap.attempt_revisions) == {0: 3, 1: 1}
    assert snap.revisions[(0, m1)] == 3           # three typed results for one label
    assert (0, m1) not in snap.written            # the current revision is unwritten
    assert (1, m1) in snap.written
    # ... yet label 0's HISTORICAL write identity survives: that history, not
    # the current-revision projection, is what the public totals derive from.
    assert snap.written_labels == frozenset({0, 1})
    snap.verify_conservation()


# ── §12.2.5 — receipt Iterables never execute caller code under the lock ─────

def test_correction5_receipt_generator_does_not_deadlock_the_ledger():
    """``record_persisted``/``record_durable``/``record_written`` declare an
    ``Iterable``; a generator that mints receipts through the public
    ``receipt()`` API must not re-enter the non-reentrant ledger lock.  Probed
    on a FINISHED session's ledger in a bounded daemon thread so a regression
    can never wedge the writer, the pool or pytest."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()
    sess = _session(1)
    sess.submit(_frames(1)[0])
    sess.finish()                     # the writer can never touch the ledger again
    led = sess.accounting

    done = threading.Event()
    errors: list[BaseException] = []

    def _certify() -> None:
        try:
            led.record_written(0, (m for m in (m1,)))
            led.record_persisted(led.receipt(0, m, "nexus") for m in (m1,))
            led.record_durable(led.receipt(0, m, "nexus") for m in (m1,))
        except BaseException as exc:          # pragma: no cover - regression path
            errors.append(exc)
        finally:
            done.set()

    probe = threading.Thread(target=_certify, name="h10-receipt-generator",
                             daemon=True)
    probe.start()
    assert done.wait(timeout=GATE_TIMEOUT) is True, (
        "record_persisted/record_durable deadlocked: caller code ran while the "
        "non-reentrant ledger lock was held")
    assert not errors, errors
    snap = sess.accounting_snapshot()
    assert (0, m1, "nexus") in snap.durable
    assert (0, m1) in snap.written


# ── §13.2.1 — the acceptance authority must not fail open ────────────────────

def test_correction_r2_admission_failure_cannot_fail_open(monkeypatch):
    """``accept_cb`` is an ADMISSION AUTHORITY, not an observer: when the
    ledger's ``record_accepted`` raises, submit must become a loud run failure
    that publishes NOTHING — no queued item, no outcome, no sink write, no
    public completion, and never a ``True`` return without an exact accepted
    attempt identity — and must reap the in-flight permit and the
    already-dispatched future exactly once (no deadlock, no leaked slot)."""
    sa = _sa()
    sink = MemorySink()
    sess = _session(1, sink=sink)
    completions: list[Any] = []
    sess.on_frame_completed(completions.append)

    def _authority_down(self, label, *, publish_acceptance=None):
        raise RuntimeError("admission authority failed")

    monkeypatch.setattr(sa.StageLedger, "record_accepted", _authority_down)

    with pytest.raises(RuntimeError, match="admission authority failed"):
        sess.submit(_frames(1)[0])

    # Queue/permit cleanup: any queued item drains with a balanced
    # ``task_done`` (§18.5 supersedes the immediate ``qsize`` assertion — a
    # correctly rejected ticket may still be held by the writer when the failed
    # submit returns), and the in-flight permit came back EXACTLY once (a lost
    # permit shrinks the window forever; a double release widens it beyond the
    # declared bound).
    eng = sess._session
    assert _drain_write_queue(eng)
    assert eng._write_queue.qsize() == 0
    assert _free_slots(eng) == eng.inflight_max

    # Loud run failure: recorded, sticky at the next submit, surfaced by finish.
    with pytest.raises(RuntimeError, match="admission authority failed"):
        sess.submit(_frames(1)[0])
    result = sess.finish(raise_on_failure=False, join_timeout=30)
    assert result.failed is True
    assert "admission authority failed" in (result.error or "")

    # No publication anywhere: sink, events, public projections, ledger.
    assert sink.frames == {}
    assert completions == []
    assert sess.frames_submitted == 0
    assert sess.frames_completed == 0
    snap = sess.accounting_snapshot()
    assert snap.accepted == frozenset()
    assert snap.written_labels == frozenset()
    assert dict(snap.dispositions) == {}
    assert dict(snap.attempt_revisions) == {}
    assert dict(snap.errors) == {}
    assert snap.refused == frozenset()   # an authority failure is not a refusal
    snap.verify_conservation()


# ── §13.2.2 — outcome attempt/result identity is validated + replay-idempotent ─

def test_correction_r2_forged_future_attempt_is_rejected_atomically():
    """An outcome naming an attempt that was never accepted (attempt 99 or 0
    while the latest accepted attempt is 1) raises and mutates NOTHING: no
    result revision, no disposition change, no error installation."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()
    led = sa.StageLedger(required_modes=[m1], obligations=("nexus",))
    assert led.record_accepted(0) == 1

    with pytest.raises(ValueError):
        led.record_outcome(0, sa.ItemDisposition.COMPLETED,
                           produced=[m1], attempt=99)
    with pytest.raises(ValueError):
        led.record_outcome(0, sa.ItemDisposition.FAILED,
                           error="forged failure", attempt=99)
    with pytest.raises(ValueError):
        led.record_outcome(0, sa.ItemDisposition.COMPLETED,
                           produced=[m1], attempt=0)

    snap = led.snapshot()
    assert snap.dispositions[0] is sa.ItemDisposition.PENDING
    assert dict(snap.revisions) == {}
    assert dict(snap.errors) == {}
    snap.verify_conservation()


def test_correction_r2_unaccepted_label_or_missing_attempt_cannot_mint():
    """No result/disposition/error identity can originate from an unaccepted
    label, and an outcome carrying no exact attempt identity is invalid — the
    ledger never guesses which attempt an outcome belongs to."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()
    led = sa.StageLedger(required_modes=[m1], obligations=("nexus",))

    with pytest.raises(ValueError):
        led.record_outcome(5, sa.ItemDisposition.COMPLETED,
                           produced=[m1], attempt=1)     # label never accepted
    led.record_accepted(7)
    with pytest.raises(ValueError):
        led.record_outcome(7, sa.ItemDisposition.COMPLETED,
                           produced=[m1])                # no attempt identity
    snap = led.snapshot()
    assert 5 not in snap.dispositions
    assert dict(snap.revisions) == {}
    assert snap.dispositions[7] is sa.ItemDisposition.PENDING
    snap.verify_conservation()


def test_correction_r2_exact_replay_is_a_no_op_and_preserves_durable_receipts():
    """Exactly one terminal outcome identity exists per ``(label, attempt)``:
    replaying the same completed attempt must not mint another
    ``result_revision`` — a replay that re-minted would silently invalidate a
    durable receipt for data that never changed."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()
    led = sa.StageLedger(required_modes=[m1], obligations=("nexus",))
    attempt = led.record_accepted(0)
    led.record_outcome(0, sa.ItemDisposition.COMPLETED,
                       produced=[m1], attempt=attempt)
    led.record_durable([led.receipt(0, m1, "nexus")])
    before = led.snapshot()
    assert (0, m1, "nexus") in before.durable
    assert before.mode_complete == frozenset({0})

    led.record_outcome(0, sa.ItemDisposition.COMPLETED,
                       produced=[m1], attempt=attempt)   # EXACT replay
    assert led.current_revision(0, m1) == 1              # no re-mint
    after = led.snapshot()
    assert (0, m1, "nexus") in after.durable             # the receipt survives
    assert after.mode_complete == frozenset({0})
    assert after == before                               # a true no-op
    after.verify_conservation()


def test_correction_r2_contradictory_replay_raises_and_is_atomic():
    """A second, DIFFERENT outcome for the same ``(label, attempt)`` is a
    contradiction, not a replacement: it raises, and the invalid batch mints
    nothing — not even for modes absent from the first outcome."""
    sa = _sa()
    m1, m2 = sa.ResultMode.one_d(), sa.ResultMode.two_d()
    led = sa.StageLedger(required_modes=[m1], obligations=("nexus",))
    attempt = led.record_accepted(0)
    led.record_outcome(0, sa.ItemDisposition.COMPLETED,
                       produced=[m1], attempt=attempt)

    with pytest.raises(ValueError):
        led.record_outcome(0, sa.ItemDisposition.COMPLETED,
                           produced=[m1, m2], attempt=attempt)   # different batch
    with pytest.raises(ValueError):
        led.record_outcome(0, sa.ItemDisposition.FAILED,
                           error="late contradiction", attempt=attempt)

    snap = led.snapshot()
    assert snap.revisions[(0, m1)] == 1
    assert (0, m2) not in snap.revisions   # atomic: the new mode minted nothing
    assert snap.dispositions[0] is sa.ItemDisposition.COMPLETED
    assert 0 not in snap.errors
    snap.verify_conservation()


def test_correction_r2_duplicate_produced_modes_mint_at_most_one_revision():
    """Each produced mode mints at most one revision per outcome: a duplicated
    mode entry in one batch is deduplicated, never double-minted."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()
    led = sa.StageLedger(required_modes=[m1], obligations=("nexus",))
    attempt = led.record_accepted(0)
    led.record_outcome(0, sa.ItemDisposition.COMPLETED,
                       produced=[m1, m1], attempt=attempt)
    assert led.current_revision(0, m1) == 1
    led.snapshot().verify_conservation()


def test_correction_r2_legal_older_attempt_mints_once_without_view_authority():
    """A legal OLDER accepted attempt still mints its result exactly once —
    without replacing the latest attempt's PENDING view or supplying its error
    — and its own replay rules hold: exact replay no-op, contradiction raises."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()
    led = sa.StageLedger(required_modes=[m1], obligations=("nexus",))
    assert led.record_accepted(0) == 1
    assert led.record_accepted(0) == 2

    led.record_outcome(0, sa.ItemDisposition.COMPLETED,
                       produced=[m1], attempt=1)         # the older attempt
    snap = led.snapshot()
    assert snap.dispositions[0] is sa.ItemDisposition.PENDING   # view untouched
    assert snap.revisions[(0, m1)] == 1                         # ... but it minted

    led.record_outcome(0, sa.ItemDisposition.COMPLETED,
                       produced=[m1], attempt=1)         # exact replay: no-op
    assert led.current_revision(0, m1) == 1
    with pytest.raises(ValueError):
        led.record_outcome(0, sa.ItemDisposition.FAILED,
                           error="stale failure", attempt=1)    # contradiction
    snap2 = led.snapshot()
    assert snap2.dispositions[0] is sa.ItemDisposition.PENDING
    assert 0 not in snap2.errors

    led.record_outcome(0, sa.ItemDisposition.COMPLETED,
                       produced=[m1], attempt=2)         # the latest lands
    snap3 = led.snapshot()
    assert snap3.dispositions[0] is sa.ItemDisposition.COMPLETED
    assert snap3.revisions[(0, m1)] == 2
    snap3.verify_conservation()


def test_correction_r2_conservation_rejects_unaccepted_identity_origins():
    """§13.2.2 conservation extension: NO result / disposition / error /
    attempt identity may originate from a label that was never accepted."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()

    def _snap(**overrides):
        base = dict(
            required_modes=(), obligations=frozenset(),
            accepted=frozenset(), refused=frozenset(),
            pending=frozenset(), completed=frozenset(),
            dispositions={}, errors={}, attempt_revisions={},
            revisions={}, written=frozenset(), written_labels=frozenset(),
            persisted=frozenset(), durable=frozenset(),
            publication_dropped=frozenset(), mode_complete=frozenset(),
            discovered=frozenset(), enqueued=frozenset(), skipped={},
        )
        base.update(overrides)
        return sa.StageSnapshot(**base)

    _snap().verify_conservation()                # the empty ledger holds
    for bad in (
        dict(revisions={(3, m1): 1}),
        dict(dispositions={3: sa.ItemDisposition.COMPLETED}),
        dict(errors={3: "orphan error"}),
        dict(attempt_revisions={3: 1}),
    ):
        with pytest.raises(ValueError):
            _snap(**bad).verify_conservation()


# ── §13.2.3 — cancellation diagnostics never subtract incompatible axes ──────

def test_correction_r2_double_write_then_cancel_logs_no_fictitious_drop(caplog):
    """Two SUCCESSFUL write attempts of one label (write + replace) followed by
    Stop: every public projection stays distinct-label-truthful and the private
    cancellation diagnostic must not fabricate a dropped in-flight item by
    subtracting a distinct-label total from an attempt total."""
    sess = _session(1)
    with caplog.at_level(logging.INFO, logger="xrd_tools.reduction.core"):
        assert sess.submit(_frames(1)[0]) is True
        assert sess.pause(timeout=GATE_TIMEOUT) is True    # write 1 landed
        sess.resume()
        assert sess.submit(Frame(0, image=np.full((2, 2), 9.0))) is True
        assert sess.pause(timeout=GATE_TIMEOUT) is True    # write 2 landed (same label)
        sess.resume()
        sess.stop()                              # cancel with nothing in flight
        result = sess.finish()

    assert result.cancelled is True
    assert result.failed is False
    assert sess.frames_submitted == 1
    assert sess.frames_completed == 1
    assert result.n_processed == 1
    snap = sess.accounting_snapshot()
    assert snap.attempt_revisions[0] == 2        # both attempts are real
    assert snap.written_labels == frozenset({0})
    cancel_claims = [rec.getMessage() for rec in caplog.records
                     if rec.name == "xrd_tools.reduction.core"
                     and rec.getMessage().startswith("cancelled:")]
    assert cancel_claims == []   # nothing was dropped; claiming so is fiction
    snap.verify_conservation()


# ── §14 — the admission-gated worker lifecycle ───────────────────────────────
#
# These rows pin what must be true when the authority rejects an item that has
# ALREADY been dispatched (the frozen dispatch-before-admission order).  They
# synchronize on entry to the SUBMITTED CALLABLE via a real bounded pool
# wrapper — never on entry to integration, because a correct gate must make
# integration entry impossible.


class _EntryProbeExecutor:
    """A REAL ``ThreadPoolExecutor`` wrapped so a row can synchronize on entry
    to the engine's submitted callable.  The pool, the callable and the futures
    are the production ones; only the entry signal and the shutdown tally are
    added (the engine's own executor seam, not a fake of the seam under test)."""

    def __init__(self, inner: Any, tail: float = 0.0) -> None:
        self._inner = inner
        self._tail = tail
        self.entered = threading.Event()
        self.futures: list[Any] = []
        self.shutdown_calls = 0

    @property
    def _max_workers(self) -> Any:
        # The engine sizes its in-flight window from this real pool value.
        return getattr(self._inner, "_max_workers", None)

    def submit(self, fn, *args, **kwargs):
        def _entered_then_run():
            self.entered.set()
            try:
                return fn(*args, **kwargs)
            finally:
                # A bounded per-task tail (real pools do bookkeeping after the
                # callable returns) turns "retired synchronously" into an
                # ordering fact instead of a scheduling race.
                if self._tail:
                    time.sleep(self._tail)

        future = self._inner.submit(_entered_then_run)
        self.futures.append(future)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_calls += 1
        self._inner.shutdown(wait=wait, cancel_futures=cancel_futures)


class _LifecycleProbeSink:
    """Real sink recording the exact worker/terminal-cleanup ordering: whether
    the pool-side ``worker_process`` hook ran at all, whether it ran AFTER
    ``abort``, and how many top-level writes happened."""

    def __init__(self) -> None:
        self.aborted = threading.Event()
        self.worker_entered = threading.Event()
        self.wrote = threading.Event()
        self.worker_after_abort = False
        self.write_thread: int | None = None
        self.writes = 0
        self.finished = 0

    def begin(self, scan, plan) -> None:
        return None

    def worker_process(self, frame, reduction) -> None:
        self.worker_entered.set()
        if self.aborted.is_set():
            self.worker_after_abort = True

    def write(self, frame, reduction) -> None:
        self.writes += 1
        self.write_thread = threading.get_ident()
        self.wrote.set()

    def finish(self, result) -> None:
        self.finished += 1

    def abort(self, result) -> None:
        self.aborted.set()


def _authority_down(self, label, *, publish_acceptance=None):
    """``StageLedger.record_accepted`` replacement: the declared accounting
    authority is down."""
    raise RuntimeError("admission authority failed")


def _authority_down_after_entry(probe: Any):
    """The same authority failure, delayed until the dispatched callable has
    really started — the exact already-running rejection.  Any probe exposing
    an ``entered`` event works, including the §15.3 no-timeout wrapper."""

    def _down(self, label, *, publish_acceptance=None):
        assert probe.entered.wait(GATE_TIMEOUT), (
            "the submitted callable never started; this row cannot pin the "
            "already-running rejection"
        )
        raise RuntimeError("admission authority failed")

    return _down


def _settled_within(future: Any, timeout: float = GATE_TIMEOUT) -> bool:
    """Whether an EXACT future reached a terminal state within the bound."""
    done, _pending = futures_wait([future], timeout=timeout)
    return bool(done)


def _drain_write_queue(eng, timeout: float = GATE_TIMEOUT) -> bool:
    """Bounded, condition-driven drain of the engine's write queue (§18.5).

    A correctly rejected item may still be queued — or already held by the
    writer — when the failed ``submit`` returns, so the queue fact is "it
    drains and ``task_done`` balances", not an immediate ``qsize``.  This waits
    on the queue's own ``all_tasks_done`` condition, never a timing sleep."""
    q = eng._write_queue
    deadline = time.monotonic() + timeout
    with q.all_tasks_done:
        while q.unfinished_tasks:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            q.all_tasks_done.wait(remaining)
    return True


def _free_slots(eng) -> int:
    """Capacity derived from the sole in-flight membership owner."""
    return eng.inflight_max - len(eng._inflight._members)


def _inventory_identity(sess):
    """Identity-only snapshot of the session's scan inventory: frames hold
    numpy arrays, so object IDENTITY and order — not ``==`` — are the exact
    comparison for a staging rollback."""
    scan = sess.scan
    return ([id(frame) for frame in scan.frames],
            {int(label): id(frame)
             for label, frame in scan._frame_by_index.items()},
            dict(sess._session._scan_frame_positions))


def _assert_nothing_published(sess, sink) -> None:
    """The exact empty/bounded facts a rejected admission must leave behind."""
    eng = sess._session
    assert _drain_write_queue(eng), (
        "a rejected item never drained from the write queue: task_done did "
        "not balance within the bound")
    assert eng._write_queue.qsize() == 0
    assert eng._submitted == 0               # the staged attempt fact rolled back
    free = _free_slots(eng)
    assert free == eng.inflight_max          # permit returned EXACTLY once
    assert sink.writes == 0
    assert sess.frames_submitted == 0
    assert sess.frames_completed == 0
    snap = sess.accounting_snapshot()
    assert snap.accepted == frozenset()
    assert snap.written_labels == frozenset()
    assert dict(snap.dispositions) == {}
    assert dict(snap.attempt_revisions) == {}
    assert snap.refused == frozenset()       # an authority failure is no refusal
    snap.verify_conservation()


def test_round3_owned_pool_rejected_callable_never_reduces_or_preps(
        monkeypatch):
    """OWNED executor, callable already running when admission fails: no
    reduction and no pool-side ``worker_process`` may run at all — and nothing
    may run after terminal sink cleanup — while ``finish`` still shuts the owned
    pool down exactly once.

    (The §14.3 owned-pool lifecycle row carried forward under §18; only its
    NAME is truthed-up — §18 rejects the item at its ticket instead of retiring
    a dispatched future, and every fact asserted here is unchanged.)"""
    sa = _sa()
    sink = _LifecycleProbeSink()
    integration_entered = threading.Event()

    def _integrate(image, ai, **kw):
        integration_entered.set()
        # Deterministic (not timed) race: an unretired orphan reaches the
        # worker hook only after terminal abort has run.
        sink.aborted.wait(GATE_TIMEOUT)
        return _r1d(1.0)

    monkeypatch.setattr(reduction_core, "integrate_1d", _integrate)
    sess = _session(1, sink=sink, executor=1)
    eng = sess._session
    probe = _EntryProbeExecutor(eng._worker)     # wrap the pool the engine OWNS
    eng._worker = probe
    monkeypatch.setattr(sa.StageLedger, "record_accepted",
                        _authority_down_after_entry(probe))

    with pytest.raises(RuntimeError, match="admission authority failed"):
        sess.submit(_frames(1)[0])
    assert probe.entered.is_set()                # the race was really pinned

    result = sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)
    assert result.failed is True
    assert "admission authority failed" in (result.error or "")
    assert sink.aborted.is_set()
    assert not sink.worker_after_abort, (
        "an unretired future ran worker_process after terminal sink abort")
    assert not sink.worker_entered.is_set(), (
        "a rejected item ran the sink's pool-side worker_process hook")
    assert not integration_entered.is_set(), (
        "a rejected item entered integration")
    assert probe.shutdown_calls == 1
    _assert_nothing_published(sess, sink)


def test_round3_finish_stays_finite_after_a_started_callable_is_rejected(
        monkeypatch):
    """A finite ``finish(join_timeout=...)`` must RETURN within a bounded probe
    after a started callable was rejected — an unretired dispatched future must
    never stall terminal cleanup — while integration is never entered."""
    sa = _sa()
    sink = _LifecycleProbeSink()
    integration_entered = threading.Event()
    release = threading.Event()

    def _integrate(image, ai, **kw):
        integration_entered.set()
        release.wait(GATE_TIMEOUT)      # the stalled worker of §14.2
        return _r1d(1.0)

    monkeypatch.setattr(reduction_core, "integrate_1d", _integrate)
    sess = _session(1, sink=sink, executor=1)
    eng = sess._session
    probe = _EntryProbeExecutor(eng._worker)
    eng._worker = probe
    monkeypatch.setattr(sa.StageLedger, "record_accepted",
                        _authority_down_after_entry(probe))

    with pytest.raises(RuntimeError, match="admission authority failed"):
        sess.submit(_frames(1)[0])
    assert probe.entered.is_set()

    finished = threading.Event()
    # §15.2: capture BOTH legs of the background call — a finish that returned
    # fast by raising must not read as a green finite finish.
    outcome: dict[str, Any] = {}

    def _finish() -> None:
        try:
            outcome["result"] = sess.finish(raise_on_failure=False,
                                            join_timeout=0.1)
        except BaseException as exc:      # noqa: BLE001 - asserted below
            outcome["error"] = exc
        finally:
            finished.set()

    prober = threading.Thread(target=_finish, name="h10-finish-probe",
                              daemon=True)
    prober.start()
    try:
        assert finished.wait(2.0), (
            "finish(join_timeout=0.1) did not return within a bounded probe; "
            "an unretired dispatched future stalled terminal cleanup"
        )
        assert "error" not in outcome, (
            "the finite finish raised in the probe thread: "
            f"{outcome.get('error')!r}")
        assert outcome["result"].failed is True
        assert not integration_entered.is_set(), (
            "the corrected gate must retire the callable BEFORE integration")
    finally:
        release.set()
        prober.join(GATE_TIMEOUT)
    assert not prober.is_alive()


def test_round3_rejected_fresh_frame_never_enters_the_scan_inventory(
        monkeypatch):
    """A fresh label whose admission fails must publish no scan-inventory fact:
    no ``scan.frames`` entry, no ``_frame_by_index`` entry, no position."""
    sa = _sa()
    sink = _LifecycleProbeSink()
    sess = _session(1, sink=sink)
    scan = sess.scan
    positions_before = dict(sess._session._scan_frame_positions)
    monkeypatch.setattr(sa.StageLedger, "record_accepted", _authority_down)

    with pytest.raises(RuntimeError, match="admission authority failed"):
        sess.submit(Frame(99, image=np.full((2, 2), 3.0)))

    assert 99 not in scan._frame_by_index
    assert [int(frame.index) for frame in scan.frames] == [0]
    assert dict(sess._session._scan_frame_positions) == positions_before
    _assert_nothing_published(sess, sink)
    sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)


def test_round3_rejected_replacement_preserves_the_prior_inventory_entry(
        monkeypatch):
    """A REPLACEMENT frame whose admission fails must leave the previous frame
    object AND its exact position untouched: a rejected attempt may not swap the
    object an accepted attempt was reduced from."""
    sa = _sa()
    sess = _session(1)
    assert sess.submit(_frames(1)[0]) is True
    assert sess.pause(timeout=GATE_TIMEOUT) is True       # the write landed
    sess.resume()
    scan = sess.scan
    original = scan._frame_by_index[0]
    positions = dict(sess._session._scan_frame_positions)
    monkeypatch.setattr(sa.StageLedger, "record_accepted", _authority_down)

    replacement = Frame(0, image=np.full((2, 2), 7.0))
    with pytest.raises(RuntimeError, match="admission authority failed"):
        sess.submit(replacement)

    assert scan._frame_by_index[0] is original
    assert scan.frames[positions[0]] is original
    assert dict(sess._session._scan_frame_positions) == positions
    assert all(frame is not replacement for frame in scan.frames)
    snap = sess.accounting_snapshot()
    assert snap.attempt_revisions[0] == 1     # the rejected attempt never minted
    snap.verify_conservation()
    sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)


def test_round3_never_started_rejected_callable_never_runs(monkeypatch):
    """The other rejection polarity: when admission fails while the dispatched
    callable has not started yet, it must never reduce, prep, write or publish
    once it does get a worker.  (§18 no longer cancels that Future — the
    rejected ticket is what stops it — so this row pins the EFFECT, not the
    ``cancel()`` call.)"""
    sa = _sa()
    sink = _LifecycleProbeSink()
    integration_entered = threading.Event()

    def _integrate(image, ai, **kw):
        integration_entered.set()
        return _r1d(1.0)

    monkeypatch.setattr(reduction_core, "integrate_1d", _integrate)
    pool = ThreadPoolExecutor(max_workers=1)
    hold = threading.Event()
    blocker = pool.submit(hold.wait, GATE_TIMEOUT)   # occupy the only worker
    probe = _EntryProbeExecutor(pool)
    try:
        sess = _session(1, sink=sink, executor=probe)
        monkeypatch.setattr(sa.StageLedger, "record_accepted", _authority_down)
        with pytest.raises(RuntimeError, match="admission authority failed"):
            sess.submit(_frames(1)[0])

        assert len(probe.futures) == 1
        assert not probe.entered.is_set()            # still queued behind the hold
        hold.set()                                   # NOW give it a worker
        blocker.result(GATE_TIMEOUT)
        assert _settled_within(probe.futures[0]), (
            "the rejected callable never finished once a worker freed up")
        result = sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)
        assert result.failed is True
        assert not integration_entered.is_set()
        assert not sink.worker_entered.is_set()
        _assert_nothing_published(sess, sink)
    finally:
        hold.set()
        pool.shutdown(wait=True)


# ── §18.5 — the single publication ticket ────────────────────────────────────
#
# The ratified root-cause design: ONE private per-item ticket owns the permit,
# the item's single ``PENDING -> ACCEPTED/REJECTED`` decision, the Future the
# executor returned and the exact reversible inventory staging.  Acceptance
# stays where §4.1/§12.2.1 froze it, so a rejected item needs no Future method
# at all, both queue-publication polarities are observationally identical, and
# an interrupted acceptance tail can never roll the ledger back.  Groups 1-9 are
# frozen against exact ``dff4d0bc``; rows marked "regression freeze" already
# hold there and are frozen so this correction cannot lose them.


class _MinimalFuture:
    """The pre-H10 minimum a returned Future must satisfy: a blocking,
    no-argument ``result()`` and NOTHING else.  Any ``cancel``/``done``/
    ``exception`` probe — or a ``timeout`` argument — explodes, so a path that
    quietly re-narrows the public executor contract cannot pass."""

    _FORBIDDEN = frozenset({"cancel", "cancelled", "done", "exception",
                            "running", "add_done_callback"})

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def result(self, *args: Any, **kwargs: Any):
        assert not args and not kwargs, (
            "the engine called result(timeout=...) on a Future whose contract "
            "is a blocking, no-argument result()")
        return self._inner.result()

    def __getattr__(self, name: str):
        if name in _MinimalFuture._FORBIDDEN:
            raise AssertionError(
                f"the engine required {name}() on a minimal duck Future")
        raise AttributeError(name)


class _MinimalDuckExecutor:
    """The public contract's minimum executor: ``submit()`` only, wrapping a
    REAL asynchronous pool.  ``shutdown``/``_max_workers`` are absent — the
    engine may never shut down a pool it does not own, and must fall back to
    its documented default in-flight window."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.tickets: list[Any] = []
        self.entered = threading.Event()

    def submit(self, fn, *args, **kwargs):
        self.tickets.append(args[0] if args else None)

        def _entered_then_run():
            self.entered.set()
            return fn(*args, **kwargs)

        return _MinimalFuture(self._inner.submit(_entered_then_run))

    def __getattr__(self, name: str):
        if name == "shutdown":
            raise AssertionError("the engine shut down a pool it does not own")
        raise AttributeError(name)


class _DeadExecutor:
    """An executor whose ``submit()`` refuses outright: a pool/interpreter
    dispatch failure with NO effect — nothing is ever scheduled."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def _max_workers(self) -> int:
        return 2

    def submit(self, fn, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("executor refused the dispatch")


class _ScheduleThenRaiseExecutor:
    """The hostile-but-schedulable dispatch failure: the engine's callable is
    really scheduled on a REAL pool and THEN ``submit()`` raises, so the caller
    sees a failed submit while a live callable already holds the item."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.entered = threading.Event()
        self.tickets: list[Any] = []
        self.inner_futures: list[Any] = []
        self.shutdown_calls = 0

    @property
    def _max_workers(self) -> Any:
        return getattr(self._inner, "_max_workers", None)

    def submit(self, fn, *args, **kwargs):
        self.tickets.append(args[0] if args else None)

        def _entered_then_run():
            self.entered.set()
            return fn(*args, **kwargs)

        self.inner_futures.append(self._inner.submit(_entered_then_run))
        raise RuntimeError("executor scheduled the work and then failed")

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shutdown_calls += 1
        self._inner.shutdown(wait=wait, cancel_futures=cancel_futures)


def _unpark(objs) -> None:
    """Parent-compatibility cleanup ONLY: publish a rejected decision on any
    captured decision object, so a red run at ``dff4d0bc`` — whose gate is
    never published on these paths — does not idle out its 60 s bound."""
    for obj in objs:
        publish = getattr(obj, "decide", None) or getattr(obj, "publish", None)
        if callable(publish):
            try:
                publish(False)
            except BaseException:            # noqa: BLE001 - cleanup only
                pass


def test_g1_executor_submit_failure_before_effect_publishes_nothing():
    """Group 1a — a dispatch failure with no effect: the caller's exception is
    exact, nothing is scheduled, nothing is published, the permit is returned
    exactly once and ``finish`` is finite/failed.  (Regression freeze.)"""
    sink = _LifecycleProbeSink()
    executor = _DeadExecutor()
    sess = _session(1, sink=sink, executor=executor)
    before = _inventory_identity(sess)

    with pytest.raises(RuntimeError, match="executor refused the dispatch"):
        sess.submit(Frame(99, image=np.full((2, 2), 3.0)))

    assert executor.calls == 1
    assert _inventory_identity(sess) == before
    _assert_nothing_published(sess, sink)
    result = sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)
    assert result.failed is True
    assert "executor refused the dispatch" in (result.error or "")


def test_g1_scheduled_then_failed_dispatch_never_enters_reduction(monkeypatch):
    """Group 1b — schedule-then-raise: the hidden callable really runs, and it
    must observe the item's REJECTED decision and exit before reduction or
    ``worker_process``.  A failed submit may not return leaving a live callable
    parked on an undecided item."""
    sink = _LifecycleProbeSink()
    integration_entered = threading.Event()

    def _integrate(image, ai, **kw):
        integration_entered.set()
        return _r1d(1.0)

    monkeypatch.setattr(reduction_core, "integrate_1d", _integrate)
    pool = ThreadPoolExecutor(max_workers=1)
    probe = _ScheduleThenRaiseExecutor(pool)
    try:
        sess = _session(1, sink=sink, executor=probe)
        before = _inventory_identity(sess)
        with pytest.raises(RuntimeError,
                           match="scheduled the work and then failed"):
            sess.submit(_frames(1)[0])

        assert probe.entered.wait(GATE_TIMEOUT), (
            "the scheduled callable never ran; this row cannot pin the "
            "hidden-callable polarity")
        assert _settled_within(probe.inner_futures[0]), (
            "a failed submit returned while its scheduled callable was still "
            "parked on an undecided item")
        assert not integration_entered.is_set()
        assert not sink.worker_entered.is_set()
        assert _inventory_identity(sess) == before
        _assert_nothing_published(sess, sink)
        result = sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)
        assert result.failed is True
        assert probe.shutdown_calls == 0     # never shut down a caller's pool
    finally:
        _unpark(probe.tickets)
        pool.shutdown(wait=True)


class _PutFailQueue(queue.Queue):
    """The REAL stdlib write queue with one armed publication failure.
    ``before=True`` raises without enqueueing; ``before=False`` completes the
    real enqueue — including ``unfinished_tasks`` — and THEN raises."""

    def __init__(self, *args, before: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.before = before
        self.armed = False
        self.raised = 0

    def put(self, item, *args, **kwargs):
        if self.armed and item is not reduction_core._STREAM_SENTINEL:
            self.armed = False
            self.raised += 1
            if self.before:
                raise RuntimeError("write queue refused the item")
            super().put(item, *args, **kwargs)
            raise RuntimeError("write queue enqueued the item and then failed")
        return super().put(item, *args, **kwargs)


def _session_with_queue(monkeypatch, probe_queue, n=1, **kw) -> ScanSession:
    """Install a real ``queue.Queue`` SUBCLASS as the engine's write queue
    before the writer thread starts — swapping it afterwards would leave the
    writer parked on the original object.  The patch is scoped to the single
    construction call."""
    with monkeypatch.context() as patched:
        patched.setattr(reduction_core.queue, "Queue",
                        lambda *a, **k: probe_queue)
        return _session(n, **kw)


@pytest.mark.parametrize("before", [True, False])
def test_g2_queue_publication_failure_is_identical_in_both_polarities(
        monkeypatch, before):
    """Group 2 — a write-queue publication failure BEFORE effect and one that
    enqueues and THEN raises must leave observationally identical public facts,
    balance ``task_done`` exactly, and restore exactly ONE permit (never two,
    never none)."""
    sink = _LifecycleProbeSink()
    integration_entered = threading.Event()

    def _integrate(image, ai, **kw):
        integration_entered.set()
        return _r1d(1.0)

    monkeypatch.setattr(reduction_core, "integrate_1d", _integrate)
    probe_queue = _PutFailQueue(before=before)
    sess = _session_with_queue(monkeypatch, probe_queue, 1, sink=sink,
                               executor=1)
    eng = sess._session
    assert eng._write_queue is probe_queue
    before_inventory = _inventory_identity(sess)
    probe_queue.armed = True

    with pytest.raises(RuntimeError, match="write queue"):
        sess.submit(_frames(1)[0])

    assert probe_queue.raised == 1
    assert not integration_entered.is_set()
    assert not sink.worker_entered.is_set()
    assert _inventory_identity(sess) == before_inventory
    _assert_nothing_published(sess, sink)
    assert probe_queue.unfinished_tasks == 0
    result = sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)
    assert result.failed is True
    assert "write queue" in (result.error or "")
    # The balance has to survive terminal teardown too: an over-called
    # ``task_done`` kills the writer thread, after which the ``finish``
    # sentinel is never consumed and this count stays stuck at one.
    assert probe_queue.unfinished_tasks == 0, (
        "the write queue did not balance across finish: the writer thread "
        "died on an unbalanced task_done")


@pytest.mark.parametrize("replacement", [False, True])
def test_g3_rejected_admission_touches_no_future_method(monkeypatch,
                                                        replacement):
    """Group 3 — the authority refuses a fresh label and a replacement: the
    ticket and the Future may exist, but NO Future method may be inspected (the
    minimal duck explodes on ``cancel``/``done``/``result(timeout=...)``), the
    inventory restores exactly, nothing publishes and ``finish`` is bounded."""
    sa = _sa()
    sink = _LifecycleProbeSink()
    integration_entered = threading.Event()

    def _integrate(image, ai, **kw):
        integration_entered.set()
        return _r1d(1.0)

    monkeypatch.setattr(reduction_core, "integrate_1d", _integrate)
    pool = ThreadPoolExecutor(max_workers=2)
    probe = _MinimalDuckExecutor(pool)
    try:
        sess = _session(1, sink=sink, executor=probe)
        eng = sess._session
        if replacement:
            assert sess.submit(_frames(1)[0]) is True
            assert sess.pause(timeout=GATE_TIMEOUT) is True   # the write landed
            sess.resume()
            # The ACCEPTED frame legitimately reduced and prepped; the facts
            # below belong to the REJECTED one that follows it.
            integration_entered.clear()
            sink.worker_entered.clear()
        before = _inventory_identity(sess)
        monkeypatch.setattr(sa.StageLedger, "record_accepted", _authority_down)

        frame = (Frame(0, image=np.full((2, 2), 7.0)) if replacement
                 else Frame(99, image=np.full((2, 2), 3.0)))
        with pytest.raises(RuntimeError, match="admission authority failed"):
            sess.submit(frame)

        assert _inventory_identity(sess) == before
        assert all(item is not frame for item in sess.scan.frames)
        assert not integration_entered.is_set()
        assert not sink.worker_entered.is_set()
        assert _drain_write_queue(eng)
        assert _free_slots(eng) == eng.inflight_max
        assert eng._submitted == (1 if replacement else 0)
        snap = sess.accounting_snapshot()
        assert dict(snap.attempt_revisions) == ({0: 1} if replacement else {})
        assert snap.refused == frozenset()
        snap.verify_conservation()
        result = sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)
        assert result.failed is True
        assert sink.writes == (1 if replacement else 0)
    finally:
        _unpark(probe.tickets)
        pool.shutdown(wait=True)


@pytest.mark.parametrize("composed", [True, False])
def test_g4_accepted_cycle_on_a_minimal_duck_executor(composed):
    """Group 4 — a full accepted submit -> reduce -> write -> finish cycle on an
    asynchronous executor whose Future exposes ONLY a blocking, no-argument
    ``result()``, parameterized over ScanSession's integer attempt and a raw
    ``ReductionSession`` whose ticket is ACCEPTED(None).  (Regression freeze:
    this correction may not narrow the public executor contract.)"""
    pool = ThreadPoolExecutor(max_workers=2)
    probe = _MinimalDuckExecutor(pool)
    try:
        if composed:
            sess = _session(1, sink=MemorySink(), executor=probe)
            assert sess.submit(_frames(1)[0]) is True
            result = sess.finish(join_timeout=GATE_TIMEOUT)
            assert sess.frames_submitted == 1
            assert sess.frames_completed == 1
            snap = sess.accounting_snapshot()
            assert snap.accepted == frozenset({0})
            assert snap.written_labels == frozenset({0})
            assert snap.attempt_revisions[0] == 1
            snap.verify_conservation()
        else:
            eng = ReductionSession(
                ReductionPlan(integration_2d=None),
                Scan("h10", _frames(1), integrator=object()),
                sink=MemorySink(), executor=probe, execution="streaming")
            assert eng.accept_cb is None       # ACCEPTED(None), never PENDING
            assert eng.submit(_frames(1)[0]) is True
            result = eng.finish(join_timeout=GATE_TIMEOUT)
        assert result.failed is False
        assert result.n_processed == 1
        assert probe.entered.is_set()
    finally:
        pool.shutdown(wait=True)


def test_g5_accepted_publication_order_gates_worker_and_writer(monkeypatch):
    """Group 5 — at the exact instant the authority runs, the Future is already
    bound to the ticket, the SAME ticket is already identifiable on the write
    queue and the inventory/submitted facts are staged, while worker and writer
    stay gated: no reduction, ``worker_process``, outcome, write or completion
    may precede the published decision.  Accepted writes stay on the writer
    thread."""
    sa = _sa()
    sink = _LifecycleProbeSink()
    integration_entered = threading.Event()

    def _integrate(image, ai, **kw):
        integration_entered.set()
        return _r1d(1.0)

    monkeypatch.setattr(reduction_core, "integrate_1d", _integrate)
    pool = ThreadPoolExecutor(max_workers=1)
    probe = _MinimalDuckExecutor(pool)
    observed: dict[str, Any] = {}
    real_accept = sa.StageLedger.record_accepted

    def _observing_accept(self, label, *, publish_acceptance=None):
        eng = sess._session
        ticket = probe.tickets[0] if probe.tickets else None
        observed["future_bound"] = getattr(ticket, "future", None) is not None
        observed["queued"] = eng._write_queue.unfinished_tasks
        observed["submitted"] = eng._submitted
        observed["inventory"] = int(label) in eng.scan._frame_by_index
        observed["reduced"] = integration_entered.is_set()
        observed["prepped"] = sink.worker_entered.is_set()
        observed["writes"] = sink.writes
        return real_accept(
            self, label, publish_acceptance=publish_acceptance)

    monkeypatch.setattr(sa.StageLedger, "record_accepted", _observing_accept)
    try:
        sess = _session(1, sink=sink, executor=probe)
        assert sess.submit(_frames(1)[0]) is True
        assert observed["future_bound"] is True, (
            "the authority ran before the returned Future was bound to the "
            "item's ticket")
        assert observed["queued"] == 1, (
            "the authority ran before that same item was identifiable on the "
            "write queue")
        assert observed["submitted"] == 1
        assert observed["inventory"] is True
        assert observed["reduced"] is False
        assert observed["prepped"] is False
        assert observed["writes"] == 0
        result = sess.finish(join_timeout=GATE_TIMEOUT)
        assert result.failed is False
        assert sink.writes == 1
        assert sink.write_thread not in (None, threading.get_ident()), (
            "an accepted write left the writer thread")
    finally:
        pool.shutdown(wait=True)


@pytest.mark.parametrize("polarity", ["before_effect", "after_effect"])
def test_g6_operator_interrupt_in_the_acceptance_tail(monkeypatch, polarity):
    """Group 6 — a genuine operator interruption AFTER the authority minted the
    attempt.  Acceptance is irreversible: the ticket may never be rolled back,
    relabelled REJECTED or stranded PENDING, the inventory/submitted facts stay
    committed, the run is sticky-failed and really cancelled, writer ownership
    plus permit/``task_done`` stay exact, ``finish`` is finite/failed, and the
    interrupted call emits NO caller-thread progress event.

    ``before_effect``: the interrupt lands before the decision has effect, so
    cleanup is recorded first and acceptance is then replayed idempotently —
    the worker observes the cancellation and produces that exact attempt's
    typed CANCELLED_BEFORE_COMPLETION with no reduction, prep or write.
    ``after_effect``: the decision already opened the gate and this row waits
    for the REAL write before the injected raise surfaces, so the accepted
    attempt's COMPLETED outcome and its write must both stand."""
    sa = _sa()
    base = reduction_core._StreamPublication   # AttributeError at the parent
    sink = _LifecycleProbeSink()
    integration_entered = threading.Event()

    def _integrate(image, ai, **kw):
        integration_entered.set()
        return _r1d(1.0)

    monkeypatch.setattr(reduction_core, "integrate_1d", _integrate)
    interrupt = KeyboardInterrupt("operator interrupt in the acceptance tail")
    seen: list[Any] = []

    class _InterruptingTicket(base):
        """The REAL ticket with ONE injected asynchronous interrupt at a named
        edge of the acceptance publication."""

        fired = False

        def complete_wake(self):
            seen.append(self)
            receipt = self.decision
            if (receipt is not None
                    and receipt[0] == reduction_core._TICKET_ACCEPTED
                    and not _InterruptingTicket.fired):
                _InterruptingTicket.fired = True
                if polarity == "before_effect":
                    raise interrupt
                super().complete_wake()
                assert sink.wrote.wait(GATE_TIMEOUT), (
                    "the after-effect polarity needs the real write to land "
                    "before the injected interrupt")
                raise interrupt
            return super().complete_wake()

    monkeypatch.setattr(reduction_core, "_StreamPublication",
                        _InterruptingTicket)
    sess = _session(1, sink=sink, executor=1)
    eng = sess._session
    caller = threading.get_ident()
    progress_threads: list[int] = []
    sess.on_progress(lambda event: progress_threads.append(
        threading.get_ident()))

    with pytest.raises(KeyboardInterrupt):
        sess.submit(_frames(1)[0])

    assert caller not in progress_threads, (
        "an interrupted submit emitted a caller-thread progress event")
    assert _InterruptingTicket.fired is True
    ticket = seen[0]
    assert ticket.decision == (reduction_core._TICKET_ACCEPTED, 1), (
        "the acceptance tail left the ticket "
        f"{ticket.decision!r} after the authority minted an attempt")
    assert eng._submitted == 1
    assert 0 in eng.scan._frame_by_index
    assert eng.cancel_token.cancelled is True
    snap = sess.accounting_snapshot()
    assert snap.accepted == frozenset({0})
    assert snap.attempt_revisions[0] == 1

    result = sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)
    assert result.failed is True
    assert _drain_write_queue(eng)
    assert eng._write_queue.unfinished_tasks == 0
    assert _free_slots(eng) == eng.inflight_max
    final = sess.accounting_snapshot()
    assert final.accepted == frozenset({0})
    assert final.attempt_revisions[0] == 1
    if polarity == "before_effect":
        assert (final.dispositions[0]
                is sa.ItemDisposition.CANCELLED_BEFORE_COMPLETION)
        assert sink.writes == 0
        assert not integration_entered.is_set()
        assert not sink.worker_entered.is_set()
        assert final.written_labels == frozenset()
    else:
        assert final.dispositions[0] is sa.ItemDisposition.COMPLETED
        assert sink.writes == 1
        assert final.written_labels == frozenset({0})
    final.verify_conservation()


class _HostileReprError(RuntimeError):
    """A legal authority failure whose diagnostic ``repr`` raises a
    BaseException.  Formatting a field may never let hostile data escape into
    the lifecycle, and may never replace the exception the caller must see."""

    def __repr__(self) -> str:
        raise KeyboardInterrupt("a field's repr escaped the trace")


@pytest.mark.parametrize("trace_on", [False, True])
def test_g7_hostile_field_representation_never_escapes(monkeypatch, tmp_path,
                                                       trace_on):
    """Group 7a — with tracing off AND on, a field whose ``__repr__`` raises
    ``KeyboardInterrupt`` may not escape, may not replace the authority
    failure, and (traced) must still leave the event recorded with its other
    fields."""
    trace = tmp_path / "admission.trace"
    if trace_on:
        monkeypatch.setenv("XDART_H10_ADMISSION_TRACE", str(trace))
    else:
        monkeypatch.delenv("XDART_H10_ADMISSION_TRACE", raising=False)
    sa = _sa()
    sink = _LifecycleProbeSink()
    hostile = _HostileReprError("admission authority failed")

    def _hostile_authority(self, label, *, publish_acceptance=None):
        raise hostile

    monkeypatch.setattr(sa.StageLedger, "record_accepted", _hostile_authority)
    sess = _session(1, sink=sink, executor=1)
    raised: BaseException | None = None
    try:
        sess.submit(_frames(1)[0])
    except BaseException as exc:          # noqa: BLE001 - identity-checked
        raised = exc

    if raised is not hostile:
        # Identity is checked WITHOUT asserting on the object: pytest's own
        # assertion rewriting would repr the operands, and this one's repr
        # raises KeyboardInterrupt — which would abort the pytest session
        # instead of failing this row.
        pytest.fail("a field's hostile representation replaced the authority "
                    f"failure with {type(raised).__name__}")
    _assert_nothing_published(sess, sink)
    result = sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)
    assert result.failed is True
    if trace_on:
        lines = [line for line in trace.read_text().splitlines()
                 if "submit_rejected" in line]
        assert lines, "the rejection event was lost with the hostile field"
        assert "label=0" in lines[0], (
            "the traced event lost its representable fields")
        assert reduction_core._TRACE_UNREPRESENTABLE in lines[0]


def test_g7_trace_io_interrupt_is_contained_by_the_optional_helper(tmp_path):
    """Group 7b (§19.5.1) — a GENUINE ``KeyboardInterrupt`` raised inside the
    optional helper on the very FIRST traced event is contained there: the
    accepted submission still completes and writes exactly once, finish
    succeeds, and permit/queue capacity matches the untraced run instead of the
    leaked-permit false success the rejected object produced."""
    baseline = _trace_scenario_facts(_accepted_trace_scenario, tmp_path, None,
                                     trace_on=False)
    arms: list[Any] = []
    facts = _trace_scenario_facts(_accepted_trace_scenario, tmp_path,
                                  "permit_acquired", arms=arms)
    arm = arms[0]

    assert arm.fired.is_set(), (
        "the first traced event never reached the interrupting I/O step")
    assert facts["submit"] == [True], (
        "an optional diagnostic changed whether the frame was accepted")
    assert facts["writes"] == 1
    assert facts["failed"] is False
    assert facts["finish_error"] is None
    assert facts["free_slots"] == facts["inflight_max"], (
        "an interrupted diagnostic leaked this item's in-flight membership")
    assert facts == baseline, (
        "an interrupted optional diagnostic changed the run's lifecycle facts")


_OLD_ADMISSION_SYMBOLS = ("_AdmissionGate", "_gated_stream_reduce",
                          "_retire_gated_future", "_await_terminal_future",
                          "_ADMISSION_GATE_TIMEOUT", "_ADMISSION_REAP_TIMEOUT",
                          "_ADMISSION_REAP_POLL")
_CORE_SITE = "xrd_tools/reduction/core.py"


def _parse_source(relative: str) -> ast.Module:
    return ast.parse((_SRC_ROOT / relative).read_text(encoding="utf-8"))


def _referenced_tokens(tree: ast.AST) -> list[str]:
    """Every identifier the module actually references — alias-resistant:
    Name/Attribute/def/class/import-as targets plus the string argument of
    ``getattr``/``setattr``/``hasattr``."""
    tokens: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.append(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            tokens.append(node.name)
        elif isinstance(node, ast.alias):
            tokens.append(node.name.split(".")[-1])
            if node.asname:
                tokens.append(node.asname)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in {"getattr", "setattr", "hasattr"}
              and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
              and isinstance(node.args[1].value, str)):
            tokens.append(node.args[1].value)
    return tokens


def test_g8_one_publication_ticket_owner_and_no_retirement_symbols():
    """Group 8/D1 — one capacity owner, ticket, queue and consumer only."""
    tree = _parse_source(_CORE_SITE)
    windows = [node for node in ast.walk(tree)
               if isinstance(node, ast.ClassDef)
               and node.name == "_InFlightWindow"]
    window_constructions = [node for node in ast.walk(tree)
                            if isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name)
                            and node.func.id == "_InFlightWindow"]
    assert len(windows) == len(window_constructions) == 1
    definitions = [node for node in ast.walk(tree)
                   if isinstance(node, ast.ClassDef)
                   and node.name == "_StreamPublication"]
    assert len(definitions) == 1, (
        "the publication ticket must have exactly one definition")
    constructions = [node for node in ast.walk(tree)
                     if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name)
                     and node.func.id == "_StreamPublication"]
    assert len(constructions) == 1, (
        "the ticket must be constructed at exactly one site")
    session = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.ClassDef)
                   and node.name == "ReductionSession")
    queues = [node for node in ast.walk(session) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)
              and node.func.attr == "Queue"]
    assert len(queues) == 1, "a second write-queue owner appeared"
    consumers = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "get"
                 and isinstance(node.func.value, ast.Attribute)
                 and node.func.value.attr == "_write_queue"]
    assert len(consumers) == 1, (
        "the write queue must have exactly one consumer")
    pools = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "ThreadPoolExecutor"]
    assert len(pools) == 3, (
        "a new executor owner appeared in the reduction engine")
    referenced = _referenced_tokens(tree)
    for retired in ("Semaphore", "_semaphore", "_released", "_held",
                    "_count", "_capacity", "_available"):
        assert retired not in referenced, (
            f"a second capacity fact survived: {retired}")
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        body = path.read_text(encoding="utf-8")
        for symbol in _OLD_ADMISSION_SYMBOLS:
            assert symbol not in body, (
                f"{path.name} still names the retired symbol {symbol}")
    for init in ("xrd_tools/reduction/__init__.py",
                 "xrd_tools/session/__init__.py"):
        exported = _referenced_tokens(_parse_source(init))
        assert "_StreamPublication" not in exported, (
            f"{init} exports the private publication ticket")


_STALE_ADMISSION_PROSE = (
    "reap", "retire", "already-dispatched",
    "before the item is published to the write queue",
    "before the writer can see the item",
    "waits for admission",
)
# The exact §18.2 publication order every acceptance-authority site must state.
_PUBLICATION_ORDER = ("future bound", "ticket queued", "inventory staged",
                      "then this authority", "accepted opens")
_AUTHORITY_FAILURE_CONTRACT = (
    "before publishing an acceptance proof",
    "rejects the ticket",
    "undoes its staged facts",
    "leaves no ledger fact",
    "once the authority publishes the attempt",
    "acceptance is irreversible",
    "remains inventoried and ledger-visible",
    "sticky-fails and cancels",
    "accepted writer path completes it",
)
_UNQUALIFIED_ROLLBACK_PROSE = (
    "the engine rejects that item's shared publication ticket",
    "rolls its staged facts back before the exception",
)


def _accept_cb_field_prose() -> str:
    """The ``accept_cb`` field's comment block — prose the AST cannot reach,
    censused from the source lines immediately above the field."""
    lines = (_SRC_ROOT / _CORE_SITE).read_text(encoding="utf-8").splitlines()
    anchors = [n for n, line in enumerate(lines)
               if line.startswith("    accept_cb: ")]
    assert len(anchors) == 1, "the accept_cb field moved or was duplicated"
    block: list[str] = []
    cursor = anchors[0] - 1
    while cursor >= 0 and lines[cursor].lstrip().startswith("#"):
        block.append(lines[cursor].lstrip("# ").strip())
        cursor -= 1
    return " ".join(reversed(block))


def test_g8_documentation_census_pins_the_executor_and_thread_contract():
    """Group 8 (§19.5.5) — the single-orchestrator boundary and the
    asynchronous-executor precondition must be stated in the
    ``ReductionSession``, ``submit``, ``run_reduction`` and ScanSession-facing
    prose; the acceptance-authority sites must state the actual §18.2
    publication order; and the stale pre-queue-admission/reap/retire prose must
    be gone from every censused site."""
    docs: dict[str, str] = {}
    core_tree = _parse_source(_CORE_SITE)
    for node in ast.walk(core_tree):
        if isinstance(node, ast.ClassDef) and node.name == "ReductionSession":
            docs["ReductionSession"] = ast.get_docstring(node) or ""
            for child in node.body:
                if (isinstance(child, ast.FunctionDef)
                        and child.name == "submit"):
                    docs["ReductionSession.submit"] = (
                        ast.get_docstring(child) or "")
        elif isinstance(node, ast.FunctionDef) and node.name == "run_reduction":
            docs["run_reduction"] = ast.get_docstring(node) or ""
    for node in ast.walk(_parse_source(_LEDGER_COMPOSITION_SITE)):
        if isinstance(node, ast.ClassDef) and node.name == "ScanSession":
            for child in node.body:
                if (isinstance(child, ast.FunctionDef)
                        and child.name in ("submit", "_on_accepted")):
                    docs[f"ScanSession.{child.name}"] = (
                        ast.get_docstring(child) or "")
    assert set(docs) == {"ReductionSession", "ReductionSession.submit",
                         "run_reduction", "ScanSession.submit",
                         "ScanSession._on_accepted"}
    docs["accept_cb"] = _accept_cb_field_prose()
    censused = {site: " ".join(doc.split()).lower()  # never hinge on wrapping
                for site, doc in docs.items()}
    for site in ("ReductionSession", "ReductionSession.submit",
                 "run_reduction", "ScanSession.submit"):
        assert "one orchestrating thread" in censused[site], (
            f"{site} does not pin the single-orchestrator boundary")
        assert "asynchronous" in censused[site], (
            f"{site} does not state the asynchronous-executor precondition")
    for site in ("accept_cb", "ScanSession._on_accepted"):
        at = [censused[site].find(phrase) for phrase in _PUBLICATION_ORDER]
        assert -1 not in at, (
            f"{site} does not state the §18.2 publication order: missing "
            f"{[p for p, i in zip(_PUBLICATION_ORDER, at) if i < 0]}")
        assert at == sorted(at), (
            f"{site} states the publication order out of sequence: {at}")
    for phrase in ("future", "ticket is queued", "inventory is staged"):
        assert phrase in censused["ReductionSession.submit"], (
            f"submit()'s prose does not state the publication order fact "
            f"{phrase!r}")
    for site, text in censused.items():
        for phrase in _STALE_ADMISSION_PROSE:
            assert phrase not in text, (
                f"stale admission-order prose survives in {site}: {phrase!r}")
    stale = (_SRC_ROOT / _LEDGER_COMPOSITION_SITE).read_text(encoding="utf-8")
    for phrase in ("already-dispatched", "retire", "reap"):
        assert phrase not in stale, (
            f"stale admission-order prose survives in scan_session.py: "
            f"{phrase!r}")
    contract_errors = [
        f"ScanSession.submit is missing {phrase!r}"
        for phrase in _AUTHORITY_FAILURE_CONTRACT
        if phrase not in censused["ScanSession.submit"]
    ]
    contract_errors.extend(
        f"ScanSession.submit still makes rollback unconditional: {phrase!r}"
        for phrase in _UNQUALIFIED_ROLLBACK_PROSE
        if phrase in censused["ScanSession.submit"]
    )
    core_prose = (_SRC_ROOT / _CORE_SITE).read_text(
        encoding="utf-8").lower()
    if "semaphore" in core_prose:
        contract_errors.append(
            "core.py still names the retired semaphore capacity owner")
    assert not contract_errors, "\n".join(contract_errors)


# ---------------------------------------------------------------------------
# §19 — globally non-interfering optional trace, and an expired bound that is
# a diagnostic wake rather than a decision.
# ---------------------------------------------------------------------------


class _TraceInterrupt:
    """Arm a GENUINE ``KeyboardInterrupt`` inside ``_admission_trace``'s OWN
    body for exactly one named lifecycle event.

    The production helper is never replaced by a raising stub — that would
    bypass the guard under test (§19.5.2).  A thin recorder publishes, per
    thread, which event is being formatted, and the interrupt is raised from
    the helper's own ``open`` step, so the only thing that can contain it is
    the helper's real outer guard."""

    def __init__(self, monkeypatch, path, event: str | None) -> None:
        self.event = event
        self.fired = threading.Event()
        self._local = threading.local()
        real_trace = reduction_core._admission_trace
        real_open = open
        monkeypatch.setenv("XDART_H10_ADMISSION_TRACE", str(path))

        def _recording_trace(event_name, **fields):
            self._local.event = event_name
            try:
                return real_trace(event_name, **fields)
            finally:
                self._local.event = None

        def _interrupting_open(*args, **kwargs):
            if (self.event is not None and not self.fired.is_set()
                    and getattr(self._local, "event", None) == self.event):
                self.fired.set()
                raise KeyboardInterrupt(
                    f"operator interrupt inside the {self.event} trace")
            return real_open(*args, **kwargs)

        monkeypatch.setattr(reduction_core, "_admission_trace",
                            _recording_trace)
        monkeypatch.setattr(reduction_core, "open", _interrupting_open,
                            raising=False)


def _drive_traced(sess, sink, n_frames: int) -> dict[str, Any]:
    """Drive one session to its end and return the exact lifecycle facts an
    optional diagnostic may never change.  Escaping interrupts are recorded as
    facts (not re-raised) so a red row reports the DIFFERENCE from the untraced
    run rather than an opaque KeyboardInterrupt."""
    facts: dict[str, Any] = {"submit": [], "finish_error": None,
                             "failed": None}
    for frame in _frames(n_frames):
        try:
            facts["submit"].append(sess.submit(frame))
        except BaseException as exc:              # noqa: BLE001 - recorded
            facts["submit"].append(type(exc).__name__)
    try:
        result = sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)
        facts["failed"] = bool(result.failed)
    except BaseException as exc:                  # noqa: BLE001 - recorded
        facts["finish_error"] = type(exc).__name__
    eng = sess._session
    facts["writes"] = sink.writes
    facts["finished_sink"] = sink.finished
    facts["aborted_sink"] = sink.aborted.is_set()
    facts["inflight_max"] = eng.inflight_max
    facts["free_slots"] = _free_slots(eng)
    facts["queue_balanced"] = eng._write_queue.unfinished_tasks == 0
    facts["writer_alive"] = bool(eng._writer_thread is not None
                                 and eng._writer_thread.is_alive())
    snap = sess.accounting_snapshot()
    facts["accepted"] = sorted(snap.accepted)
    facts["written_labels"] = sorted(snap.written_labels)
    facts["dispositions"] = {int(label): disposition.value
                             for label, disposition in snap.dispositions.items()}
    return facts


def _arm_trace(monkeypatch, tmp_path, event, trace_on, arms, name):
    if not trace_on:
        monkeypatch.delenv("XDART_H10_ADMISSION_TRACE", raising=False)
        return
    arms.append(_TraceInterrupt(monkeypatch, tmp_path / name, event))


def _accepted_trace_scenario(monkeypatch, tmp_path, event, trace_on, arms=()):
    """One ACCEPTED frame, submit → finish: permit, worker, writer dequeue,
    sentinel, writer join, sink finish and owned-pool shutdown all fire."""
    _arm_trace(monkeypatch, tmp_path, event, trace_on, arms, "accepted.trace")
    sink = _LifecycleProbeSink()
    return _drive_traced(_session(1, sink=sink, executor=1), sink, 1)


def _rejected_trace_scenario(monkeypatch, tmp_path, event, trace_on, arms=()):
    """One REJECTED frame (the declared authority is down): the writer's drop
    branch and the sink abort/terminal-cleanup events fire."""
    monkeypatch.setattr(_sa().StageLedger, "record_accepted", _authority_down)
    _arm_trace(monkeypatch, tmp_path, event, trace_on, arms, "rejected.trace")
    sink = _LifecycleProbeSink()
    return _drive_traced(_session(1, sink=sink, executor=1), sink, 1)


def _trace_scenario_facts(scenario, tmp_path, event, *, trace_on=True,
                          arms=()):
    """Run one scenario under its OWN monkeypatch context so a baseline and an
    interrupted run cannot leak patches into each other."""
    with pytest.MonkeyPatch.context() as mp:
        return scenario(mp, tmp_path, event, trace_on, arms)


_TRACED_LIFECYCLE_EVENTS = [
    (_accepted_trace_scenario, "permit_acquired"),
    (_accepted_trace_scenario, "writer_item_dequeued"),
    (_accepted_trace_scenario, "writer_sentinel_exit"),
    (_accepted_trace_scenario, "writer_join_begin"),
    (_accepted_trace_scenario, "writer_join_end"),
    (_accepted_trace_scenario, "sink_finish_begin"),
    (_accepted_trace_scenario, "sink_finish_end"),
    (_accepted_trace_scenario, "pool_shutdown_enter"),
    (_accepted_trace_scenario, "pool_shutdown_exit"),
    (_rejected_trace_scenario, "writer_dropped_unaccepted"),
    (_rejected_trace_scenario, "sink_abort_begin"),
    (_rejected_trace_scenario, "sink_abort_end"),
]


@pytest.mark.parametrize(
    "scenario,event", _TRACED_LIFECYCLE_EVENTS,
    ids=[f"{event}" for _scenario, event in _TRACED_LIFECYCLE_EVENTS])
def test_g10_optional_trace_never_interferes_with_a_lifecycle_event(
        tmp_path, scenario, event):
    """§19.5.2 — a real operator interrupt landing inside the optional helper
    at ANY lifecycle event — permit acquisition, the rejected writer drop, the
    accepted writer dequeue, sentinel exit, writer join, sink finish/abort or
    owned-pool shutdown — reproduces the untraced run exactly and leaves the
    process healthy."""
    baseline = _trace_scenario_facts(scenario, tmp_path, None, trace_on=False)
    arms: list[Any] = []
    facts = _trace_scenario_facts(scenario, tmp_path, event, arms=arms)

    assert arms[0].fired.is_set(), (
        f"the {event} trace never reached the interrupting I/O step; this row "
        "cannot pin non-interference")
    assert facts["queue_balanced"] is True
    assert facts["writer_alive"] is False, (
        f"an interrupted {event} diagnostic left the writer thread behind")
    assert facts["free_slots"] == facts["inflight_max"], (
        f"an interrupted {event} diagnostic leaked in-flight membership")
    assert facts == baseline, (
        f"an interrupted {event} diagnostic changed the run's lifecycle facts")


class _PendingOnceEvent:
    """The ticket's decision Event with ONE expired bounded wait on the
    parameterized observer: its first wait THERE returns ``False`` while the
    decision is still PENDING — exactly what a real bound expiring looks like —
    and every later wait blocks on the real decision."""

    def __init__(self, observer: str) -> None:
        self._observer = observer
        self._real = threading.Event()
        self.expired = threading.Event()

    def _is_target(self) -> bool:
        name = threading.current_thread().name
        if self._observer == "writer":
            return name.startswith("reduction-writer")
        return name.startswith("ThreadPoolExecutor")

    def set(self) -> None:
        self._real.set()

    def is_set(self) -> bool:
        return self._real.is_set()

    def wait(self, timeout=None) -> bool:
        if self._is_target() and not self.expired.is_set():
            self.expired.set()
            return False
        return self._real.wait(timeout)


def _pending_once_tickets(monkeypatch, observer: str) -> list[Any]:
    """Replace the ticket's decision Event only — the real production ticket,
    permit, decision and rollback are untouched."""
    real_cls = reduction_core._StreamPublication
    made: list[Any] = []

    class _PendingOnceTicket(real_cls):
        __slots__ = ()

        def __init__(self, frame):
            super().__init__(frame)
            self._decided = _PendingOnceEvent(observer)
            made.append(self)

    monkeypatch.setattr(reduction_core, "_StreamPublication",
                        _PendingOnceTicket)
    return made


@pytest.mark.parametrize("observer", ["worker", "writer"])
def test_g11_an_expired_bound_is_a_wake_not_a_decision(monkeypatch, observer):
    """§19.5.3 — an observer whose bounded wait expires while the ticket is
    still PENDING may not reduce, drop, release a permit or call ``task_done``
    before the REAL authority returns; after the decision the exact attempt
    completes, writes once, leaves no pending disposition and finishes."""
    sa = _sa()
    sink = _LifecycleProbeSink()
    integration_entered = threading.Event()

    def _integrate(image, ai, **kw):
        integration_entered.set()
        return _r1d(1.0)

    monkeypatch.setattr(reduction_core, "integrate_1d", _integrate)
    tickets = _pending_once_tickets(monkeypatch, observer)
    observed: dict[str, Any] = {}
    real_accept = sa.StageLedger.record_accepted

    def _held_authority(self, label, *, publish_acceptance=None):
        eng = sess._session
        assert tickets and tickets[0]._decided.expired.wait(GATE_TIMEOUT), (
            f"the {observer} never reached its bounded wait; this row cannot "
            "pin the PENDING semantics")
        observed["reduced"] = integration_entered.is_set()
        observed["prepped"] = sink.worker_entered.is_set()
        observed["writes"] = sink.writes
        observed["dropped"] = eng._dropped_attempts
        observed["unfinished"] = eng._write_queue.unfinished_tasks
        observed["free_slots"] = _free_slots(eng)
        observed["inflight_max"] = eng.inflight_max
        return real_accept(
            self, label, publish_acceptance=publish_acceptance)

    monkeypatch.setattr(sa.StageLedger, "record_accepted", _held_authority)
    sess = _session(1, sink=sink, executor=1)
    assert sess.submit(_frames(1)[0]) is True
    assert observed["reduced"] is False, (
        "an expired bound let the worker reduce an undecided item")
    assert observed["prepped"] is False
    assert observed["writes"] == 0
    assert observed["dropped"] == 0, (
        "an expired bound was treated as a rejection and dropped the item")
    assert observed["unfinished"] == 1, (
        "an expired bound let an observer retire the item from the write "
        "queue before the caller decided")
    assert observed["free_slots"] == observed["inflight_max"] - 1, (
        "an expired bound released the undecided item's in-flight membership")
    result = sess.finish(join_timeout=GATE_TIMEOUT)
    assert result.failed is False
    assert sink.writes == 1
    snap = sess.accounting_snapshot()
    assert snap.accepted == frozenset({0})
    assert dict(snap.dispositions) == {0: sa.ItemDisposition.COMPLETED}
    assert snap.written_labels == frozenset({0})
    snap.verify_conservation()


def _ticket_state_compares(tree: ast.AST) -> set[tuple[str, tuple, tuple]]:
    """Every comparison against a ticket-state constant, keyed by the enclosing
    function: ``(function, operators, state names)``."""
    found: set[tuple[str, tuple, tuple]] = set()
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Compare):
                continue
            names = tuple(sorted({inner.id for inner in ast.walk(node)
                                  if isinstance(inner, ast.Name)
                                  and inner.id.startswith("_TICKET_")}))
            if names:
                found.add((func.name, tuple(sorted(
                    {type(op).__name__ for op in node.ops})), names))
    return found


def test_g12_only_an_exact_rejected_decision_drops_an_item():
    """§19.5.4 — worker and writer drop on exactly ``_TICKET_REJECTED``; no
    branch anywhere uses ``!= ACCEPTED``; absence is represented only by a
    missing immutable receipt, and ``await_decision`` keeps waiting for one."""
    tree = _parse_source(_CORE_SITE)
    compares = _ticket_state_compares(tree)
    negations = sorted(c for c in compares
                       if "_TICKET_ACCEPTED" in c[2]
                       and {"NotEq", "IsNot"} & set(c[1]))
    assert not negations, (
        "an observer still drops on 'not ACCEPTED', so an undecided item has "
        f"an autonomous drop effect: {negations}")
    for observer in ("_ticketed_stream_reduce", "_writer_loop"):
        exact = [c for c in compares if c[0] == observer
                 and c[2] == ("_TICKET_REJECTED",) and c[1] == ("Eq",)]
        assert exact, (
            f"{observer} does not drop on the exact REJECTED decision")
    assert "_TICKET_PENDING" not in _referenced_tokens(tree), (
        "the retired split PENDING state survived the immutable receipt")
    waits = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef)
             and node.name == "await_decision"]
    assert len(waits) == 1
    assert any(isinstance(node, (ast.While, ast.For))
               for node in ast.walk(waits[0])), (
        "await_decision must keep waiting while the decision is PENDING; a "
        "single bounded wait makes an expired bound an outcome")
    decision_reads = [node for node in ast.walk(waits[0])
                      if isinstance(node, ast.Attribute)
                      and node.attr == "_decision"
                      and isinstance(node.ctx, ast.Load)]
    assert len(decision_reads) == 1, (
        "await_decision must read the immutable receipt exactly once per wake")


class _ExplodingSortFrames(list):
    """The scan's own frame list with one armed failure inside the ordering
    rebuild — an inventory-staging failure AFTER the append, position and
    index mutations already happened."""

    armed = False

    def sort(self, *args, **kwargs):
        if self.armed:
            self.armed = False
            raise RuntimeError("scan inventory sort failed")
        return super().sort(*args, **kwargs)


class _ExplodingIndexMap(dict):
    """The scan's own label -> frame map with one armed failure, so a
    REPLACEMENT stage fails after the frame-list entry was already swapped."""

    armed = False

    def __setitem__(self, key, value):
        if self.armed:
            self.armed = False
            raise RuntimeError("scan inventory index update failed")
        return super().__setitem__(key, value)


@pytest.mark.parametrize("polarity", ["fresh_resort", "replacement"])
def test_g9_inventory_stage_failure_restores_the_exact_prior_inventory(
        monkeypatch, polarity):
    """Group 9 — the inventory stage fails AFTER partial mutation: frame order,
    ``_frame_by_index``, ``_scan_frame_positions``, prior object identity and
    the staged ``_submitted`` fact must all restore exactly, leaving the same
    unaccepted public facts as groups 1-3."""
    sink = _LifecycleProbeSink()
    if polarity == "fresh_resort":
        # A fresh label BELOW the last one forces the sort/position rebuild.
        sess = _session(sink=sink, executor=1,
                        frames=[Frame(0, image=np.full((2, 2), 1.0)),
                                Frame(5, image=np.full((2, 2), 2.0))])
        sess.scan.frames = _ExplodingSortFrames(sess.scan.frames)
        armed = sess.scan.frames
        target = Frame(3, image=np.full((2, 2), 3.0))
    else:
        sess = _session(1, sink=sink, executor=1)
        sess.scan._frame_by_index = _ExplodingIndexMap(
            sess.scan._frame_by_index)
        armed = sess.scan._frame_by_index
        target = Frame(0, image=np.full((2, 2), 7.0))
    before = _inventory_identity(sess)
    armed.armed = True

    with pytest.raises(RuntimeError, match="scan inventory"):
        sess.submit(target)

    assert armed.armed is False, "the staging failure was never injected"
    assert _inventory_identity(sess) == before
    assert all(item is not target for item in sess.scan.frames)
    _assert_nothing_published(sess, sink)
    result = sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)
    assert result.failed is True
    assert "scan inventory" in (result.error or "")


# ── §12.3.7 — real compute failure and observer isolation ────────────────────

def test_real_compute_failure_is_accepted_failed_with_no_result_revision(
        monkeypatch):
    """A real integration failure propagates through the real reduce/future/
    writer path: the label stays ACCEPTED, its typed disposition is FAILED with
    the compute error, and no result revision, write or completion projection
    is ever minted for it."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()

    def _integrate(image, ai, **kw):
        if float(np.max(image)) == 1.0:                 # frame 1 only
            raise RuntimeError("integration kernel failed")
        return _r1d(float(np.sum(image)))

    monkeypatch.setattr(reduction_core, "integrate_1d", _integrate)
    sink = MemorySink()
    sess = _session(2, sink=sink)
    for fr in _frames(2):
        assert sess.submit(fr) is True
    result = sess.finish(raise_on_failure=False)

    assert result.failed is True
    snap = sess.accounting_snapshot()
    assert 1 in snap.accepted
    assert snap.dispositions[1] is sa.ItemDisposition.FAILED
    assert "integration kernel failed" in snap.errors[1]
    assert (1, m1) not in snap.revisions
    assert (1, m1) not in snap.written
    assert 1 not in sink.frames
    assert snap.dispositions[0] is sa.ItemDisposition.COMPLETED
    assert sess.frames_completed == 1
    assert result.n_processed == 1
    snap.verify_conservation()


def test_outcome_cb_exception_neither_kills_the_writer_nor_loses_the_queue():
    """A raising ``outcome_cb`` observer is caught + logged on the writer
    thread (T0-7/S1): every remaining queued frame is still written and the run
    still succeeds."""
    seen: list[int] = []

    def _raising_observer(receipt) -> None:
        seen.append(int(receipt.frame_index))
        raise RuntimeError("observer blew up")

    frames = _frames(4)
    sink = MemorySink()
    session = ReductionSession(
        ReductionPlan(integration_2d=None),
        Scan("h10", frames, integrator=object()),
        sink=sink, execution="streaming", executor=1,
        outcome_cb=_raising_observer,
    )
    for fr in frames:
        assert session.submit(fr) is True
    result = session.finish()

    assert sorted(sink.frames) == [0, 1, 2, 3]      # nothing lost behind the raise
    assert sorted(seen) == [0, 1, 2, 3]             # every outcome still delivered
    assert result.n_processed == 4
    assert result.failed is False


# ── contract-freeze guards (C1 vocabulary, not §6 rows) ──────────────────────

def test_receipt_batch_is_validated_before_it_is_applied():
    """A batch containing any invalid receipt certifies NOTHING from that batch
    (validate-then-apply), and replaying a valid receipt is idempotent."""
    sa = _sa()
    m1 = sa.ResultMode.one_d()
    sess = _session(1, obligations=("nexus", "xye"))
    sess.submit(_frames(1)[0])
    sess.finish()

    led = sess.accounting
    led.record_durable([led.receipt(0, m1, "nexus")])
    snap = sess.accounting_snapshot()
    assert (0, m1, "nexus") in snap.durable
    led.record_durable([led.receipt(0, m1, "nexus")])          # replay: idempotent
    assert sess.accounting_snapshot().durable == snap.durable

    bogus = sa.StageReceipt(label=0, mode=m1, revision=1, target="undeclared")
    with pytest.raises(ValueError):
        led.record_durable([led.receipt(0, m1, "xye"), bogus])
    assert (0, m1, "xye") not in sess.accounting_snapshot().durable

    led.record_durable([led.receipt(0, m1, "xye")])            # retry succeeds
    snap3 = sess.accounting_snapshot()
    assert snap3.mode_complete == frozenset({0})
    snap3.verify_conservation()


def test_source_observation_axis_is_identity_keyed_and_distinct():
    """§4.1: discovered/enqueued/skipped are identity sets on their own axis —
    idempotent replay, one reason per skipped identity, disjoint from both
    enqueued and processing acceptance."""
    sa = _sa()
    led = sa.StageLedger(required_modes=[sa.ResultMode.one_d()],
                         obligations=("nexus",))
    led.record_discovered("run/frame_0001")
    led.record_discovered("run/frame_0001")             # replay: idempotent
    led.record_discovered("run/frame_0002")
    led.record_enqueued("run/frame_0001")
    led.record_enqueued("run/frame_0001")
    led.record_skipped("run/frame_0002", reason="not-an-image")
    led.record_skipped("run/frame_0002", reason="not-an-image")

    snap = led.snapshot()
    assert snap.discovered == frozenset({"run/frame_0001", "run/frame_0002"})
    assert snap.enqueued == frozenset({"run/frame_0001"})
    assert dict(snap.skipped) == {"run/frame_0002": "not-an-image"}
    assert snap.accepted == frozenset()                 # enqueued ≠ accepted
    with pytest.raises(ValueError):
        led.record_skipped("run/frame_0001", reason="late-skip")   # already enqueued
    with pytest.raises(ValueError):
        led.record_enqueued("run/frame_0002")                      # already skipped
    with pytest.raises(ValueError):
        led.record_skipped("run/frame_0002", reason="different-reason")
    snap.verify_conservation()


def test_generation_is_excluded_from_persistence_identity():
    """§4.1: ``FrameEvent.generation`` is a render-staleness stamp — persistence
    identity is exactly (label, mode, revision, target), nothing more."""
    sa = _sa()
    names = {f.name for f in dataclasses.fields(sa.StageReceipt)}
    assert names == {"label", "mode", "revision", "target"}
    snap_fields = {f.name for f in dataclasses.fields(sa.StageSnapshot)}
    assert "generation" not in snap_fields


# ── owner-graph census (§12.3.8) ─────────────────────────────────────────────

_SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src"

# The ONE declared StageLedger owner/composition site, repo-relative.
_LEDGER_DEFINITION = "xrd_tools/session/stage_accounting.py"
_LEDGER_COMPOSITION_SITE = "xrd_tools/session/scan_session.py"
_OWNER_MODULE = "xrd_tools.session.stage_accounting"
_OWNER_PACKAGE = "xrd_tools.session"
_CADENCE_COMPAT_SHIM = "xrd_tools/reduction/cadence.py"
_OWNER_NAME = "StageLedger"


def _module_package(path: pathlib.Path) -> str:
    """Dotted package a relative import inside *path* resolves against."""
    return ".".join(path.parent.relative_to(_SRC_ROOT).parts)


def _absolute_import_module(package: str, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    if node.level > 1:
        parts = parts[:-(node.level - 1)]
    return ".".join([p for p in parts if p] + ([node.module] if node.module else []))


def _census_file(path: pathlib.Path) -> tuple[bool, int, set[str]]:
    """(imports the owner symbol, constructs the owner N times, modules imported).

    §13.3: the census is NAME-based for the owner symbol.  ``StageLedger`` is a
    reserved production name, so ANY ``from <anywhere> import StageLedger
    [as alias]`` counts as importing the owner — a re-export chain through an
    intermediary module cannot launder it — and every ``StageLedger(...)``
    construction counts wherever it appears (including inside the definition
    module itself), whether reached through the class name, an import alias, a
    module attribute, or a simple in-file assignment alias.  Deliberately a
    bounded static census, not a runtime import walker."""
    package = _module_package(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    owner_aliases: set[str] = {_OWNER_NAME}
    imported_modules: set[str] = set()
    imports_owner = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
                if alias.name == _OWNER_MODULE:          # `import a.b.c [as x]`
                    imports_owner = True
        elif isinstance(node, ast.ImportFrom):
            module = _absolute_import_module(package, node)   # resolves `from .x`
            imported_modules.add(module)
            for alias in node.names:
                if alias.name == _OWNER_NAME:            # the symbol, from ANYWHERE
                    owner_aliases.add(alias.asname or alias.name)
                    imports_owner = True
                elif (module in (_OWNER_MODULE, _OWNER_PACKAGE)
                        and alias.name == _OWNER_MODULE.rsplit(".", 1)[1]):
                    imports_owner = True                 # the owner module object

    # Simple in-file assignment aliases (`_L = StageLedger`,
    # `_L = mod.StageLedger`): a bounded fixpoint over plain-Name targets so a
    # rebound name cannot hide a construction.  Not a dataflow framework.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if not ((isinstance(value, ast.Name) and value.id in owner_aliases)
                    or (isinstance(value, ast.Attribute)
                        and value.attr == _OWNER_NAME)):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in owner_aliases:
                    owner_aliases.add(target.id)
                    changed = True

    constructions = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # The reserved name or any alias of it (`from x import StageLedger as
        # _L; _L(...)`, `_L2 = _L; _L2(...)`), or any attribute construction
        # (`sa.StageLedger(...)`, `pkg.mod.StageLedger(...)`) — an aliased,
        # relative or intermediary-module import cannot hide a second owner.
        if isinstance(func, ast.Name) and func.id in owner_aliases:
            constructions += 1
        elif isinstance(func, ast.Attribute) and func.attr == _OWNER_NAME:
            constructions += 1
    return imports_owner, constructions, imported_modules


def test_owner_graph_census_allows_exactly_one_stage_ledger_owner():
    """§6/§12.3.8/§13.3 owner-graph guard: an AST census of the PRODUCTION tree
    proves exactly one declared StageLedger owner/composition site — catching a
    second owner introduced through a relative, aliased or INTERMEDIARY-module
    import (``from .scan_session import StageLedger as _X``), an assignment
    alias, or a construction inside the definition module itself — and that the
    dependency direction stays session→reduction."""
    files = sorted(_SRC_ROOT.rglob("*.py"))
    relative = {path: path.relative_to(_SRC_ROOT).as_posix() for path in files}
    assert _LEDGER_COMPOSITION_SITE in relative.values()
    assert "xrd_tools/reduction/core.py" in relative.values()
    assert len(files) > 50, "the census scanned an implausibly small tree"

    importers: set[str] = set()
    constructors: dict[str, int] = {}
    reduction_imports_session: set[str] = set()
    for path, name in relative.items():
        imports_owner, constructions, modules = _census_file(path)
        if imports_owner and name != _LEDGER_DEFINITION:
            importers.add(name)
        # §13.3: a construction counts EVERYWHERE — the definition module is
        # not exempt from the constructor census.
        if constructions:
            constructors[name] = constructions
        if name.startswith("xrd_tools/reduction/") and any(
                module == _OWNER_PACKAGE or module.startswith(_OWNER_PACKAGE + ".")
                for module in modules):
            reduction_imports_session.add(name)

    assert importers == {_LEDGER_COMPOSITION_SITE}, (
        f"StageLedger is imported outside its declared composition site: "
        f"{sorted(importers - {_LEDGER_COMPOSITION_SITE})}")
    assert set(constructors) == {_LEDGER_COMPOSITION_SITE}, (
        f"a second StageLedger owner exists: "
        f"{sorted(set(constructors) - {_LEDGER_COMPOSITION_SITE})}")
    assert constructors[_LEDGER_COMPOSITION_SITE] == 1, (
        "the composition site must build exactly one ledger per session")
    # H10-C2-A moved the cadence DEFINITION behind the session policy owner,
    # so exactly ONE reduction module may import the session layer: the
    # compatibility re-export shim.  It must define nothing of its own.
    assert reduction_imports_session == {_CADENCE_COMPAT_SHIM}, (
        f"the engine must not import the session layer beyond the one cadence "
        f"compatibility shim: {sorted(reduction_imports_session)}")
    shim = ast.parse((_SRC_ROOT / _CADENCE_COMPAT_SHIM).read_text(encoding="utf-8"))
    assert [node for node in shim.body if isinstance(node, ast.ClassDef)] == [], (
        "the shim RE-EXPORTS the one policy definition; it defines nothing")
    assert any(isinstance(node, ast.ImportFrom)
               and node.module == "xrd_tools.session.policy"
               and any(alias.name == "FlushPolicy" for alias in node.names)
               for node in shim.body), (
        "the shim must re-export FlushPolicy from its new session owner")


def test_stage_accounting_is_qt_free_in_fresh_interpreter():
    """The accounting owner is headless (§4.4 boundary): importing it and the
    composed ScanSession in a clean interpreter must load no Qt/pyqtgraph."""
    probe = (
        "import sys; "
        "import xrd_tools.session.stage_accounting; "
        "import xrd_tools.session.scan_session; "
        "leaked=[m for m in sys.modules if m.split('.')[0] in "
        "('PySide6','PyQt5','PyQt6','qtpy','pyqtgraph','xdart')]; "
        "assert not leaked, leaked; print('qt-free ok')"
    )
    proc = subprocess.run([sys.executable, "-c", probe],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "qt-free ok" in proc.stdout


# ── R8 released composite-owner oracle (§§4–6 + review amendments A/C) ──────


class _RealSigintAtOpcode:
    """Deliver one real SIGINT immediately after a named bytecode effect."""

    def __init__(self, code, opname: str) -> None:
        instructions = list(dis.get_instructions(code))
        positions = [i for i, instruction in enumerate(instructions)
                     if instruction.opname == opname]
        assert len(positions) == 1, (opname, positions)
        self.code = code
        self.offset = instructions[positions[0] + 1].offset
        self.fired = False

    def __call__(self, frame, event, arg):
        if frame.f_code is self.code:
            frame.f_trace_opcodes = True
            if (event == "opcode" and frame.f_lasti == self.offset
                    and not self.fired):
                self.fired = True
                sys.settrace(None)
                signal.raise_signal(signal.SIGINT)
        return self


def _terminal_memberships(eng) -> int:
    """Final owner count, with the exact-parent semaphore only for red proof."""
    if hasattr(eng, "_inflight"):
        return len(eng._inflight._members)
    free = 0
    while eng._semaphore.acquire(blocking=False):
        free += 1
    for _ in range(free):
        eng._semaphore.release()
    return eng.inflight_max - free


def test_r8_c1_real_sigint_after_capacity_effect_is_exception_complete():
    """A real locked-runtime SIGINT after the capacity LP cannot leak it."""
    sess = _session(1, executor=1)
    eng = sess._session
    if hasattr(reduction_core, "_InFlightWindow"):
        code = reduction_core._InFlightWindow.try_acquire.__code__
        arm = _RealSigintAtOpcode(code, "STORE_SUBSCR")
    else:  # exact e0a906e6 parent: post-decrement Condition.__exit__ gap
        code = threading.Semaphore.acquire.__code__
        arm = _RealSigintAtOpcode(code, "STORE_ATTR")
    raised = None
    try:
        sys.settrace(arm)
        sess.submit(_frames(1)[0])
    except BaseException as exc:  # the real operator interrupt is a fact
        raised = exc
    finally:
        sys.settrace(None)
    result = sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)
    assert arm.fired is True
    assert isinstance(raised, KeyboardInterrupt)
    assert _terminal_memberships(eng) == 0
    assert result.failed is True
    assert isinstance(eng._failure, KeyboardInterrupt)


class _FaultingAttemptDict(dict):
    def __init__(self, values, *, operation: str, when: str) -> None:
        super().__init__(values)
        self.operation = operation
        self.when = when
        self.fired = False

    def _raise(self, operation: str, when: str) -> None:
        if (not self.fired and self.operation == operation
                and self.when == when):
            self.fired = True
            raise KeyboardInterrupt(f"{when} {operation}")

    def __setitem__(self, key, value):
        self._raise("set", "before")
        super().__setitem__(key, value)
        self._raise("set", "after")

    def pop(self, key, default=None):
        self._raise("pop", "before")
        value = super().pop(key, default)
        self._raise("pop", "after")
        return value


class _FaultingAttemptSet(set):
    def __init__(self, values, *, when: str) -> None:
        super().__init__(values)
        self.when = when
        self.fired = False

    def add(self, value):
        if self.when == "before" and not self.fired:
            self.fired = True
            raise KeyboardInterrupt("before accepted add")
        super().add(value)
        if self.when == "after" and not self.fired:
            self.fired = True
            raise KeyboardInterrupt("after accepted add")


@pytest.mark.parametrize("edge", ["attempt", "accepted", "disposition", "error"])
@pytest.mark.parametrize("when", ["before", "after"])
def test_r8_a1_acceptance_is_one_complete_attempt_state(edge, when):
    """Every old split-store edge exposes only the old or new whole state."""
    sa = _sa()
    led = sa.StageLedger(required_modes=())
    assert led.record_accepted(7) == 1
    led.record_outcome(7, sa.ItemDisposition.FAILED, error="old",
                       attempt=1)
    target = edge
    if not hasattr(led, "_accepted"):
        target = "attempt"  # R8 has exactly one acceptance assignment.
    if target == "attempt":
        fault = _FaultingAttemptDict(led._attempts, operation="set", when=when)
        led._attempts = fault
    elif target == "accepted":
        fault = _FaultingAttemptSet(led._accepted, when=when)
        led._accepted = fault
    elif target == "disposition":
        fault = _FaultingAttemptDict(
            led._dispositions, operation="set", when=when)
        led._dispositions = fault
    else:
        fault = _FaultingAttemptDict(led._errors, operation="pop", when=when)
        led._errors = fault
    with pytest.raises(KeyboardInterrupt):
        led.record_accepted(7)
    assert fault.fired is True
    snap = led.snapshot()
    old = (snap.attempt_revisions.get(7) == 1
           and snap.dispositions.get(7) is sa.ItemDisposition.FAILED
           and snap.errors.get(7) == "old")
    new = (snap.attempt_revisions.get(7) == 2
           and snap.dispositions.get(7) is sa.ItemDisposition.PENDING
           and 7 not in snap.errors)
    assert old or new, (dict(snap.attempt_revisions),
                        dict(snap.dispositions), dict(snap.errors))
    assert snap.accepted == frozenset({7})
    snap.verify_conservation()


def test_r8_a2_post_lp_publisher_fault_is_completed_before_propagation():
    led = _sa().StageLedger(required_modes=())
    published = []
    first = True

    def _publisher(attempt):
        nonlocal first
        if first:
            first = False
            raise KeyboardInterrupt("before publisher effect")
        published.append(attempt)

    with pytest.raises(KeyboardInterrupt):
        led.record_accepted(4, publish_acceptance=_publisher)
    assert published == [1]
    snap = led.snapshot()
    assert snap.attempt_revisions[4] == 1
    assert snap.dispositions[4] is _sa().ItemDisposition.PENDING


def test_r8_accepted_ticket_never_exposes_rejection_undo():
    ticket = reduction_core._StreamPublication(_frames(1)[0])
    undo = lambda: None
    ticket.unstage = undo
    ticket.store_accepted(1)
    assert ticket.take_rejection_undo() is None
    assert ticket.unstage is undo


def test_r8_f2_publish_then_raise_is_accepted_terminal_failure(monkeypatch):
    """A compliant authority that committed/published may never be rejected."""
    sa = _sa()
    real = sa.StageLedger.record_accepted

    def _publish_then_raise(self, label, *, publish_acceptance=None):
        if publish_acceptance is None:  # exact-parent semantic-red path
            real(self, label)
        else:
            real(self, label, publish_acceptance=publish_acceptance)
        raise KeyboardInterrupt("after authoritative acceptance")

    monkeypatch.setattr(sa.StageLedger, "record_accepted", _publish_then_raise)
    sess = _session(1, executor=1)
    eng = sess._session
    with pytest.raises(KeyboardInterrupt):
        sess.submit(_frames(1)[0])
    result = sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)
    snap = sess.accounting_snapshot()
    assert result.failed is True
    assert eng._submitted == 1
    assert 0 in eng.scan._frame_by_index
    assert snap.accepted == frozenset({0})
    assert snap.attempt_revisions[0] == 1
    assert snap.dispositions[0] is sa.ItemDisposition.CANCELLED_BEFORE_COMPLETION
    assert _terminal_memberships(eng) == 0
    assert eng._write_queue.unfinished_tasks == 0


def test_r8_r1_interrupt_cannot_publish_accepted_without_attempt(monkeypatch):
    """A post-state interrupt exposes one whole receipt, never split fields."""
    base = reduction_core._StreamPublication
    made = []

    class _ReceiptEdgeTicket(base):
        __slots__ = ()
        fired = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

        def decide(self, accepted, attempt=None):  # exact-parent split edge
            if accepted and not self.fired:
                _ReceiptEdgeTicket.fired = True
                self.state = reduction_core._TICKET_ACCEPTED
                raise KeyboardInterrupt("between state and attempt stores")
            return super().decide(accepted, attempt)

        def store_accepted(self, attempt):  # R8 complete-receipt edge
            if not self.fired:
                _ReceiptEdgeTicket.fired = True
                self._decision = (reduction_core._TICKET_ACCEPTED, attempt)
                raise KeyboardInterrupt("after complete receipt store")
            return super().store_accepted(attempt)

    monkeypatch.setattr(reduction_core, "_StreamPublication", _ReceiptEdgeTicket)
    sess = _session(1, executor=1)
    with pytest.raises(KeyboardInterrupt):
        sess.submit(_frames(1)[0])
    result = sess.finish(raise_on_failure=False, join_timeout=GATE_TIMEOUT)
    snap = sess.accounting_snapshot()
    assert _ReceiptEdgeTicket.fired is True
    assert result.failed is True
    assert snap.attempt_revisions[0] == 1
    assert snap.dispositions[0] is _sa().ItemDisposition.CANCELLED_BEFORE_COMPLETION
    assert _terminal_memberships(sess._session) == 0
    assert sess._session._write_queue.unfinished_tasks == 0


@pytest.mark.parametrize(
    "case", ["missing", "mismatch", "publish_bool", "publish_float",
             "return_bool"])
def test_r8_acceptance_proof_is_present_matching_positive_exact_int(case):
    """The advanced authority seam never coerces, guesses or calls twice."""
    calls = 0

    def _authority(frame, publish_acceptance=None):
        nonlocal calls
        calls += 1
        if publish_acceptance is None:  # exact parent has the old one-arg API
            return 1
        if case == "mismatch":
            publish_acceptance(1)
            return 2
        if case == "publish_bool":
            publish_acceptance(True)
            return True
        if case == "publish_float":
            publish_acceptance(1.0)
            return 1.0
        if case == "return_bool":
            return True
        return 1

    frame = _frames(1)[0]
    session = ReductionSession(
        ReductionPlan(integration_2d=None),
        Scan("proof", [frame], integrator=object()), MemorySink(),
        execution="streaming", executor=1, inflight_max=1,
        accept_cb=_authority,
    )
    raised = None
    try:
        session.submit(frame)
    except BaseException as exc:
        raised = exc
    result = session.finish(raise_on_failure=False,
                            join_timeout=GATE_TIMEOUT)
    assert calls == 1
    assert isinstance(raised, (TypeError, ValueError, RuntimeError))
    assert result.failed is True
    assert _terminal_memberships(session) == 0
    assert session._write_queue.unfinished_tasks == 0


class _FaultingCapacityDict(dict):
    def __init__(self, values=(), *, operation: str, when: str,
                 writer_only: bool = False) -> None:
        super().__init__(values)
        self.operation = operation
        self.when = when
        self.writer_only = writer_only
        self.fired = False

    def _target(self) -> bool:
        return (not self.writer_only
                or threading.current_thread().name.startswith("reduction-writer"))

    def _raise(self, operation: str, when: str) -> None:
        if (self._target() and not self.fired and self.operation == operation
                and self.when == when):
            self.fired = True
            raise RuntimeError(f"capacity {when} {operation}")

    def __setitem__(self, key, value):
        self._raise("insert", "before")
        super().__setitem__(key, value)
        self._raise("insert", "after")

    def pop(self, key, default=None):
        self._raise("pop", "before")
        value = super().pop(key, default)
        self._raise("pop", "after")
        return value


class _FaultingChanged:
    def __init__(self, inner, when: str) -> None:
        self.inner = inner
        self.when = when
        self.fired = False

    def set(self):
        target = threading.current_thread().name.startswith("reduction-writer")
        if target and self.when == "before" and not self.fired:
            self.fired = True
            raise RuntimeError("capacity before set")
        result = self.inner.set()
        if target and self.when == "after" and not self.fired:
            self.fired = True
            raise RuntimeError("capacity after set")
        return result

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _FaultAfterRelease:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.fired = False

    def release(self, ticket):
        result = self.inner.release(ticket)
        if (threading.current_thread().name.startswith("reduction-writer")
                and not self.fired):
            self.fired = True
            raise RuntimeError("between release and task_done")
        return result

    def __getattr__(self, name):
        return getattr(self.inner, name)


@pytest.mark.parametrize("when", ["before", "after"])
def test_r8_c2_acquire_fault_recovers_exact_membership(when):
    session = ReductionSession(
        ReductionPlan(integration_2d=None),
        Scan("acquire", _frames(1), integrator=object()), MemorySink(),
        execution="streaming", executor=1, inflight_max=1)
    owner = session._inflight
    fault = _FaultingCapacityDict(owner._members, operation="insert", when=when)
    owner._members = fault
    with pytest.raises(RuntimeError, match="capacity"):
        session.submit(_frames(1)[0])
    result = session.finish(raise_on_failure=False,
                            join_timeout=GATE_TIMEOUT)
    assert fault.fired is True
    assert len(owner._members) == 0
    assert session._write_queue.unfinished_tasks == 0
    assert result.failed is True


def test_r8_c2_capacity_never_widens_past_limit():
    owner = reduction_core._InFlightWindow(2)
    tickets = [object(), object(), object()]
    assert owner.try_acquire(tickets[0], 0) is True
    assert owner.try_acquire(tickets[1], 0) is True
    assert owner.try_acquire(tickets[2], 0) is False
    assert len(owner._members) == owner.limit == 2
    assert owner.release(tickets[0]) is True
    assert owner.release(tickets[0]) is False
    assert owner.release(tickets[1]) is True
    assert len(owner._members) == 0


def test_r8_d1_atomic_owner_and_acceptance_api_census():
    core = _parse_source(_CORE_SITE)
    ledger = _parse_source(_LEDGER_DEFINITION)
    scan = _parse_source(_LEDGER_COMPOSITION_SITE)
    attempt_states = [node for node in ast.walk(ledger)
                      if isinstance(node, ast.ClassDef)
                      and node.name == "_AttemptState"]
    assert len(attempt_states) == 1
    assert [node.target.id for node in attempt_states[0].body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)] == [
                "attempt", "disposition", "error"]
    ledger_tokens = _referenced_tokens(ledger)
    for retired in ("_accepted", "_dispositions", "_errors"):
        assert retired not in ledger_tokens

    session_class = next(node for node in core.body
                         if isinstance(node, ast.ClassDef)
                         and node.name == "ReductionSession")
    accept_field = next(node for node in session_class.body
                        if isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Name)
                        and node.target.id == "accept_cb")
    assert ast.unparse(accept_field.annotation) == (
        "Callable[[Frame, Callable[[int], None]], int] | None")
    emit = next(node for node in session_class.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_emit_accepted")
    assert [arg.arg for arg in emit.args.args] == [
        "self", "frame", "publish_acceptance"]
    callback_calls = [node for node in ast.walk(emit)
                      if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name)
                      and node.func.id == "cb"]
    assert len(callback_calls) == 1 and len(callback_calls[0].args) == 2
    assert not [node for node in ast.walk(emit)
                if isinstance(node, ast.ExceptHandler)
                and isinstance(node.type, ast.Name)
                and node.type.id == "TypeError"]

    ticket_class = next(node for node in core.body
                        if isinstance(node, ast.ClassDef)
                        and node.name == "_StreamPublication")
    store = next(node for node in ticket_class.body
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "store_accepted")
    stores = [node for node in ast.walk(store)
              if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Attribute)
                      and target.attr == "_decision" for target in node.targets)]
    assert len(stores) == 1
    assert not [node for node in ast.walk(store)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "set"]
    submit = next(node for node in session_class.body
                  if isinstance(node, ast.FunctionDef) and node.name == "submit")
    wake_calls = [node for node in ast.walk(submit)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "complete_wake"]
    release_calls = [node for node in ast.walk(submit)
                     if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "release"]
    assert len(wake_calls) == 2
    assert len(release_calls) == 1
    writer = next(node for node in session_class.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_writer_loop")
    completion = next(node for node in session_class.body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_complete_stream_publication")
    outer_tries = [node for node in completion.body if isinstance(node, ast.Try)]
    assert len(outer_tries) == 1 and outer_tries[0].finalbody
    assert any(isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute)
               and node.func.attr == "task_done"
               for node in ast.walk(outer_tries[0].finalbody[0]))
    rejected = next(node for node in ast.walk(writer)
                    if isinstance(node, ast.If)
                    and any(isinstance(inner, ast.Name)
                            and inner.id == "_TICKET_REJECTED"
                            for inner in ast.walk(node.test)))
    assert not [node for node in ast.walk(ast.Module(body=rejected.body))
                if isinstance(node, ast.Attribute) and node.attr == "future"]

    on_accepted = next(node for node in ast.walk(scan)
                       if isinstance(node, ast.FunctionDef)
                       and node.name == "_on_accepted")
    assert [arg.arg for arg in on_accepted.args.args] == [
        "self", "frame", "publish_acceptance"]
    calls = [node for node in ast.walk(on_accepted)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "record_accepted"]
    assert len(calls) == 1
    assert [kw.arg for kw in calls[0].keywords] == ["publish_acceptance"]

    docs = [ast.get_docstring(session_class) or "", ast.get_docstring(emit) or ""]
    docs.extend(ast.get_docstring(node) or "" for node in core.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "run_reduction")
    docs.append(ast.get_docstring(on_accepted) or "")
    normalized = " ".join(" ".join(docs).split()).lower()
    assert "accept_cb(frame, publish_acceptance)" in normalized
    assert "positive exact" in normalized


@pytest.mark.parametrize("path", ["accepted", "rejected"])
@pytest.mark.parametrize("edge", ["before_pop", "after_pop", "before_set",
                                   "after_set", "after_release"])
def test_r8_terminal_release_fault_keeps_writer_and_queue_exact(path, edge):
    frame = _frames(1)[0]
    sink = MemorySink()
    session = ReductionSession(
        ReductionPlan(integration_2d=None),
        Scan("release", [frame], integrator=object()), sink,
        execution="streaming", executor=1, inflight_max=1)
    owner = session._inflight
    if path == "rejected":
        ticket = reduction_core._StreamPublication(frame)
        assert owner.try_acquire(ticket, 0) is True
    if edge.endswith("pop"):
        fault = _FaultingCapacityDict(
            owner._members, operation="pop", when=edge.split("_")[0],
            writer_only=True)
        owner._members = fault
    elif edge.endswith("set"):
        fault = _FaultingChanged(owner._changed, edge.split("_")[0])
        owner._changed = fault
    else:
        fault = _FaultAfterRelease(owner)
        session._inflight = fault
    if path == "accepted":
        assert session.submit(frame) is True
    else:
        ticket.reject_and_wake()
        session._write_queue.put(ticket)
    assert session.drain(timeout=GATE_TIMEOUT) is True
    active_owner = getattr(session._inflight, "inner", session._inflight)
    assert fault.fired is True
    assert len(active_owner._members) == 0
    assert active_owner._changed.is_set()
    assert session._write_queue.unfinished_tasks == 0
    assert session._writer_thread.is_alive()
    assert isinstance(session._failure, RuntimeError)
    writer = session._writer_thread
    result = session.finish(raise_on_failure=False,
                            join_timeout=GATE_TIMEOUT)
    assert result.failed is True
    assert not writer.is_alive()
