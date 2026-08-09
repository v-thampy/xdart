# -*- coding: utf-8 -*-
"""H10-C2-A frozen oracle — durability, cadence and eviction (headless groups).

Ratification 2026-08-04 (SHA-256 62a1a67f…) splits the fifteen groups by
owner: headless ledger/session/store groups **G1-G7, G13-G15** live here; the
real Qt/writer groups **G8-G12** live in ``tests/xdart/test_qt_nexus_sink.py``
under the ``test_h10_c2a_`` prefix.

Discipline: the ledger/session/store/hydrator seam is always the real
production object (only the pyFAI kernel is substituted, as in the accepted C1
oracle); ``_requires_c2a`` names an absent C2-A contract as a SEMANTIC
assertion and is only ever a node's FIRST red — **G7/G13 never use it**, they
fail behaviourally; concurrency rows use ``threading.Event`` gates and bounded
joins, never a sleep; source inspection appears only in the G15 census.
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import threading
from contextlib import contextmanager

import numpy as np
import pytest

from xrd_tools.core import DEFAULT_MODE_KEY, FrameRecord, FrameView
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.reduction import (
    Frame, Integration2DPlan, MemorySink, ReductionPlan, Scan)
import xrd_tools.reduction.core as reduction_core
from xrd_tools.session import FrameRecordStore, ScanSession
from xrd_tools.session.stage_accounting import (
    ItemDisposition, ResultMode, StageLedger, StageReceipt)

NEXUS = "nexus:/scratch/c2a/run.nxs"
XYE = "xye:/scratch/c2a/run"
OTHER = "tiled:/scratch/c2a/other"
M1 = ResultMode.one_d(DEFAULT_MODE_KEY)
M2 = ResultMode.two_d(DEFAULT_MODE_KEY)
K1 = ("1d", DEFAULT_MODE_KEY)
K2 = ("2d", DEFAULT_MODE_KEY)
WAIT = 20.0          # bounded Event/join budget — never a semantic fact
NX1 = dict(targets_by_mode={M1: (NEXUS,)},      # the NeXus-only adapter
           store_targets_by_mode={M1: (NEXUS,)})


# ── absent-contract explainers (never used by G7/G13) ──────────────────────
def _accepts(fn, name: str) -> bool:
    return name in inspect.signature(fn).parameters


def _requires_c2a(condition: bool, why: str) -> None:
    assert condition, f"H10-C2-A: {why}"      # a semantic red, not a crash


def _requires_ledger_maps() -> None:
    _requires_c2a(
        _accepts(StageLedger.__init__, "targets_by_mode"),
        "StageLedger must take an immutable per-ResultMode targets_by_mode "
        "map; a global set cannot express NeXus 1D+2D + an XYE 1-D sidecar")


def _requires_session_maps() -> None:
    for name in ("targets_by_mode", "store_targets_by_mode"):
        _requires_c2a(
            _accepts(ScanSession.__init__, name),
            f"ScanSession must take {name} — ledger output persistence and "
            "store-hydratable recovery are different projections")


def _requires_session_projection() -> None:
    """The session maps, its receipt entry points, the store's atomic swap."""
    _requires_session_maps()
    for name in ("record_persisted", "record_durable",
                 "record_publication_dropped"):
        _requires_c2a(hasattr(ScanSession, name),
                      f"ScanSession must own {name}(): receipt application "
                      "commits the ledger, then reconciles the store")
    for name in ("replace_projection", "durable_modes", "dropped_modes"):
        _requires_c2a(hasattr(FrameRecordStore, name),
                      f"FrameRecordStore must expose ONE atomic projection "
                      f"replacement plus its read side ({name} missing)")


# ── real production fixtures (only the pyFAI kernel is substituted) ────────
def _r1d(value: float) -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([value, value + 1.0]),
        sigma=None, unit="q_A^-1")


def _r2d(value: float) -> IntegrationResult2D:
    return IntegrationResult2D(
        radial=np.array([0.0, 1.0]), azimuthal=np.array([0.0, 90.0]),
        intensity=np.full((2, 2), float(value)), sigma=None,
        unit="q_A^-1", azimuthal_unit="chi_deg")


@pytest.fixture(autouse=True)
def _fake_integrate(monkeypatch):
    """Substitute ONLY the pyFAI kernel, exactly as the accepted C1 oracle."""
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))
    monkeypatch.setattr(reduction_core, "integrate_2d",
                        lambda image, ai, **kw: _r2d(float(np.sum(image))))


def _plan(two_d: bool = False) -> ReductionPlan:
    return ReductionPlan(integration_2d=Integration2DPlan() if two_d else None)


def _frames(n: int) -> list[Frame]:
    return [Frame(i, image=np.full((4, 4), float(i + 1))) for i in range(n)]


def _session(frames, *, store=None, two_d=False, sink=None, **kw) -> ScanSession:
    return ScanSession(
        _plan(two_d), Scan("c2a", list(frames), integrator=object()),
        sink=MemorySink() if sink is None else sink,
        executor=2, record_store=store, **kw)


def _drain(session: ScanSession) -> None:
    """Quiesce the real writer thread at a frame boundary (bounded)."""
    assert session.pause(timeout=WAIT), "writer did not drain within the bound"


@contextmanager
def _run(frames, **kw):
    """A real ScanSession that is always resumed and finalised."""
    session = _session(frames, **kw)
    try:
        yield session
    finally:
        session.resume()
        session.finish(raise_on_failure=False)


def _record_of(store, label):
    record = store.get(label)
    assert record is not None, f"label {label} vanished from the store"
    return record


def _intensity(store, label) -> float:
    view = _record_of(store, label).results_1d[DEFAULT_MODE_KEY]
    assert view.intensity_1d is not None, "record was thinned unexpectedly"
    return float(np.asarray(view.intensity_1d)[0])


def _heavy_keys(record) -> set:
    """The record's resident heavy ``(dim, mode)`` payloads."""
    return ({("1d", m) for m, v in record.results_1d.items()
             if v.intensity_1d is not None}
            | {("2d", m) for m, v in record.results_2d.items()
               if v.intensity_2d is not None})


def _outcome(session, label, outcome, *, attempt, produced_1d=True) -> None:
    """One typed compute outcome through ScanSession's OWN projection owner
    (the engine's ``outcome_cb``) — never a direct ledger-only call."""
    session._on_outcome(reduction_core.FrameOutcomeReceipt(
        frame_index=int(label), outcome=outcome, replacing=False,
        produced_1d=produced_1d, produced_2d=False, attempt=attempt))


class _ProbeSink(MemorySink):
    """Real sink recording its boundary calls + a caller probe there, with one
    injectable failure at the real sink-write boundary (the run failure that
    drives the session's abort terminal path)."""

    def __init__(self, probe=None):
        super().__init__()
        self._probe = probe
        self.calls: list[str] = []
        self.terminal: list[tuple] = []
        self.fail_write: set = set()

    def _sample(self, name):
        self.calls.append(name)
        self.terminal.append(
            (name, None if self._probe is None else self._probe()))

    def begin(self, scan, plan):
        super().begin(scan, plan)
        self.calls.append("begin")

    def write(self, frame, reduction):
        if int(frame.index) in self.fail_write:
            raise OSError(f"injected sink write failure for {frame.index}")
        super().write(frame, reduction)

    def finish(self, result):
        super().finish(result)
        self._sample("finish")

    def abort(self, result):
        self._sample("abort")


class _InstrumentedStore(FrameRecordStore):
    """The REAL store kernel plus a call ledger and injectable failures: no
    locking/merging/thinning/bounds logic is re-implemented — every override
    records the interaction and delegates to the production method."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.calls: list[tuple] = []
        self.fail_projection: set = set()
        self.fail_upsert: set = set()
        self.gate_label = None
        self.gate_entered = threading.Event()
        self.gate_release = threading.Event()
        self.gate_release.set()
        self._gate_used = False

    def upsert(self, record, **kw):
        self.calls.append(("upsert", record.label))
        if record.label in self.fail_upsert:
            raise RuntimeError(f"injected upsert failure for {record.label}")
        return super().upsert(record, **kw)

    def mark_persisted(self, labels, **kw):
        self.calls.append(("mark_persisted", labels))
        return super().mark_persisted(labels, **kw)

    def mark_dropped(self, labels, **kw):
        self.calls.append(("mark_dropped", labels))
        return super().mark_dropped(labels, **kw)

    def replace_projection(self, label, **kw):
        self.calls.append(("replace_projection", label))
        if label in self.fail_projection:
            raise RuntimeError(f"injected projection failure for {label}")
        out = super().replace_projection(label, **kw)
        # The gate opens AFTER the real atomic clear committed and BEFORE the
        # ledger mint — exactly the window the transaction must make safe.
        if label == self.gate_label and not self._gate_used:
            self._gate_used = True
            self.gate_entered.set()
            assert self.gate_release.wait(WAIT), "gate never released"
        return out

    def names(self) -> list[str]:
        return [name for name, _label in self.calls]


# ══ G1 — mixed 1-D/2-D target applicability ═════════════════════════════════

def _g1_ledger() -> StageLedger:
    """A real completed 1-D+2-D frame under the exact NeXus+XYE-sidecar map."""
    _requires_ledger_maps()
    ledger = StageLedger(required_modes=(M1, M2),
                         targets_by_mode={M1: (NEXUS, XYE), M2: (NEXUS,)})
    ledger.record_accepted(0, publish_acceptance=None)
    ledger.record_outcome(0, ItemDisposition.COMPLETED,
                          produced=[M1, M2], attempt=1)
    return ledger


def test_g1_mixed_mode_run_completes_without_an_impossible_xye_2d_receipt():
    """NeXus 1D+2D with an XYE 1-D sidecar completes on APPLICABLE targets."""
    ledger = _g1_ledger()
    ledger.record_durable([ledger.receipt(0, M1, NEXUS),
                           ledger.receipt(0, M1, XYE),
                           ledger.receipt(0, M2, NEXUS)])
    snapshot = ledger.snapshot()
    assert 0 in snapshot.mode_complete, (
        "mode-completion consumes every APPLICABLE target PER MODE; the "
        "Cartesian product would demand an impossible XYE-2D receipt")
    snapshot.verify_conservation()


def test_g1_target_is_rejected_for_a_mode_it_does_not_apply_to():
    ledger = _g1_ledger()
    with pytest.raises(ValueError):
        ledger.receipt(0, M2, XYE)          # XYE never applies to 2-D
    with pytest.raises(ValueError):
        ledger.receipt(0, M1, OTHER)        # undeclared target
    assert ledger.receipt(0, M1, XYE).target == XYE, (
        "validating a target against the UNION instead of its own mode is "
        "exactly ratification mutation 2")
    # A FORGED receipt bypassing ledger.receipt() must still be rejected at
    # APPLICATION time — the union is never the validation domain.
    for forged in (StageReceipt(0, M2, 1, XYE), StageReceipt(0, M1, 1, OTHER)):
        for apply in (ledger.record_persisted, ledger.record_durable):
            with pytest.raises(ValueError):
                apply([forged])
    assert ledger.snapshot().persisted == frozenset(), (
        "a rejected forged receipt mutates nothing")


# ══ G2 — exact target-map validation ════════════════════════════════════════

_G2_REJECTIONS = {
    "missing-mode-key": dict(required_modes=(M1, M2),
                             targets_by_mode={M1: (NEXUS,)}),
    "extra-mode-key": dict(required_modes=(M1,),
                           targets_by_mode={M1: (NEXUS,), M2: (NEXUS,)}),
    "empty-value": dict(required_modes=(M1,), targets_by_mode={M1: ()}),
    "blank-target": dict(required_modes=(M1,), targets_by_mode={M1: ("",)}),
    "non-string-target": dict(required_modes=(M1,), targets_by_mode={M1: (7,)}),
    "global-plus-map": dict(required_modes=(M1,), obligations=(NEXUS,),
                            targets_by_mode={M1: (NEXUS,)}),
}


@pytest.mark.parametrize("case", sorted(_G2_REJECTIONS))
def test_g2_target_map_validation_is_exact(case):
    """Keyset equality, non-empty string values and the global/map conflict
    are all rejected at construction."""
    _requires_ledger_maps()
    with pytest.raises((ValueError, TypeError)):
        StageLedger(**_G2_REJECTIONS[case])


def test_g2_legacy_empty_obligations_accepted_but_never_completes():
    _requires_ledger_maps()
    ledger = StageLedger(required_modes=(M1,), obligations=())
    ledger.record_accepted(0, publish_acceptance=None)
    ledger.record_outcome(0, ItemDisposition.COMPLETED, produced=[M1],
                          attempt=1)
    assert ledger.targets_by_mode[M1] == frozenset(), (
        "legacy empty obligations produce EMPTY per-mode sets")
    assert ledger.snapshot().mode_complete == frozenset(), (
        "a dormant caller with no declared target can never complete "
        "vacuously")


def test_g2_the_frozen_target_mapping_expands_and_cannot_be_mutated():
    """Global obligations expand to every required mode, and the caller's own
    map, the ledger map and the snapshot map are three views the run's frozen
    applicability must survive."""
    _requires_ledger_maps()
    expanded = StageLedger(required_modes=(M1, M2), obligations=(NEXUS,))
    assert expanded.targets_by_mode[M1] == frozenset({NEXUS})
    assert expanded.targets_by_mode[M2] == frozenset({NEXUS})
    assert expanded.obligations == frozenset({NEXUS}), (
        "obligations stays the read-only compatibility UNION")
    caller = {M1: [NEXUS], M2: [NEXUS]}
    ledger = StageLedger(required_modes=(M1, M2), targets_by_mode=caller)
    caller[M1].append(XYE)              # mutate the CALLER's own containers
    caller[M2] = [OTHER]
    caller.pop(M1)
    assert dict(ledger.snapshot().targets_by_mode) == \
        dict(ledger.targets_by_mode) == dict(expanded.targets_by_mode), (
        "targets_by_mode is copied+frozen at construction; a caller can never "
        "widen an applicability decision the run already published")
    for mapping in (ledger.targets_by_mode, ledger.snapshot().targets_by_mode):
        with pytest.raises(TypeError):
            mapping[M1] = frozenset({XYE})              # immutable view
        with pytest.raises((TypeError, AttributeError)):
            mapping.pop(M2)


# ══ G3 — adapter map shapes and the top-level write boundary ════════════════

_G3_SHAPES = {
    "nexus-only": (True, {M1: (NEXUS,), M2: (NEXUS,)}, {M1: (NEXUS,), M2: (NEXUS,)}),
    "nexus-plus-xye": (True, {M1: (NEXUS, XYE), M2: (NEXUS,)},
                       {M1: (NEXUS,), M2: (NEXUS,)}),
    "xye-only": (False, {M1: (XYE,)}, {M1: ()}),
}


@pytest.mark.parametrize("shape", sorted(_G3_SHAPES))
def test_g3_adapter_target_maps_are_accepted_and_exposed(shape):
    """The three exact GUI adapter configurations start a real session and
    are exposed read-only; never inferred from a sink type."""
    _requires_session_maps()
    two_d, targets, store_targets = _G3_SHAPES[shape]
    store = FrameRecordStore(max_heavy_items=None)
    frames = _frames(1)
    with _run(frames, store=store, two_d=two_d, targets_by_mode=targets,
              store_targets_by_mode=store_targets) as session:
        session.submit(frames[0])
        _drain(session)
        snapshot = session.accounting_snapshot()
        exposed = snapshot.targets_by_mode
        assert {mode: set(exposed[mode]) for mode in exposed} == \
            {mode: set(value) for mode, value in targets.items()}
        assert snapshot.persisted == frozenset(), "no declaration, no mint"
        assert store.get(0) is not None, "the record still reaches the store"


def test_g3_write_boundary_mints_exactly_the_declared_single_target_receipt():
    """``record_store_persisted_on_write=True`` mints ONLY the one declared
    applicable target per mode — never a synthesized ``sink:write`` token."""
    _requires_session_maps()
    _requires_c2a(_accepts(ScanSession.__init__, "write_targets_by_mode"),
                  "record_store_persisted_on_write=True must declare exactly "
                  "one applicable target per mode via write_targets_by_mode")
    store = FrameRecordStore(max_heavy_items=None)
    frames = _frames(1)
    with _run(frames, store=store, targets_by_mode={M1: (NEXUS, XYE)},
              store_targets_by_mode={M1: (NEXUS,)},
              write_targets_by_mode={M1: (NEXUS,)},
              record_store_persisted_on_write=True) as session:
        session.submit(frames[0])
        _drain(session)
        snapshot = session.accounting_snapshot()
        assert snapshot.persisted == frozenset({(0, M1, NEXUS)}), (
            "the successful top-level write is durable for exactly the ONE "
            "declared target; widening it is ratification mutation 3")
        assert snapshot.durable == frozenset({(0, M1, NEXUS)})
        assert 0 not in snapshot.mode_complete, (
            "XYE is applicable but not certified, so the mode is not complete")


_G3_BASE = dict(targets_by_mode={M1: (NEXUS, XYE)},
                store_targets_by_mode={M1: (NEXUS,)})
_ONE_D = dict(targets_by_mode={M1: (NEXUS,)})


def _wtm(**kw):
    return dict(_G3_BASE, record_store_persisted_on_write=True, **kw)


_G3_REJECTIONS = {
    "write-flag-false-with-map": dict(
        _G3_BASE, record_store_persisted_on_write=False,
        write_targets_by_mode={M1: (NEXUS,)}),
    "write-flag-true-without-map": _wtm(),
    "write-missing-mode": _wtm(write_targets_by_mode={}),
    "write-extra-mode": _wtm(
        write_targets_by_mode={M1: (NEXUS,), M2: (NEXUS,)}),
    "write-empty-value": _wtm(write_targets_by_mode={M1: ()}),
    "write-multi-target": _wtm(write_targets_by_mode={M1: (NEXUS, XYE)}),
    "write-undeclared-target": _wtm(write_targets_by_mode={M1: (OTHER,)}),
    # store_targets_by_mode is a SUBSET map over the same required-mode keyset.
    "store-missing-mode": dict(_ONE_D, store_targets_by_mode={}),
    "store-extra-mode": dict(
        _ONE_D, store_targets_by_mode={M1: (NEXUS,), M2: (NEXUS,)}),
    "store-non-subset": dict(
        _ONE_D, store_targets_by_mode={M1: (NEXUS, XYE)}),
}


@pytest.mark.parametrize("case", sorted(_G3_REJECTIONS))
def test_g3_invalid_target_maps_reject_before_any_sink_or_writer_start(case):
    """Every ambiguous/incoherent target/store/write map is rejected BEFORE
    the sink's ``begin`` and the reduction writer start."""
    _requires_session_maps()
    sink = _ProbeSink()
    with pytest.raises((ValueError, TypeError)):
        _session(_frames(1), store=FrameRecordStore(max_heavy_items=None),
                 sink=sink, **_G3_REJECTIONS[case])
    assert sink.calls == [], (
        f"validation precedes ANY start effect; sink saw {sink.calls}")


# ══ G4/G5 — persistence, hydratability, durability and deletion ════════════

@contextmanager
def _hydratable_run(store, targets, store_targets):
    """A real run whose store carries a registered NeXus-shaped hydrator.
    Yields ``(session, hydrations)``: every hydrator invocation is recorded."""
    frames = _frames(2)
    disk: dict = {}
    hydrations: list = []

    def _hydrator(label):
        hydrations.append(label)
        return disk.get(label)

    store.set_hydrator(_hydrator)           # the real registered-hydrator seam
    with _run(frames, store=store, targets_by_mode=targets,
              store_targets_by_mode=store_targets) as session:
        for frame in frames:
            session.submit(frame)
        _drain(session)
        for label in (int(f.index) for f in frames):
            disk[label] = store.get(label)  # the recoverable on-"disk" row
        yield session, hydrations


_G4_BOUNDS = {                       # each bound driven INDEPENDENTLY
    "heavy-pressure": dict(max_heavy_items=1, max_items=None),
    "item-pressure": dict(max_heavy_items=None, max_items=1),
    "final-sweep": dict(max_heavy_items=None, max_items=None),
}


@pytest.mark.parametrize("bound", sorted(_G4_BOUNDS))
def test_g4_nexus_success_plus_xye_failure_survives_every_pressure(bound):
    """The binding consequence: NeXus success + XYE failure is store-
    hydratable but survives heavy pressure, item pressure AND the final
    sweep — one bound at a time, so none can mask another."""
    _requires_session_projection()
    store = _InstrumentedStore(**_G4_BOUNDS[bound])
    with _hydratable_run(store, {M1: (NEXUS, XYE)},
                         {M1: (NEXUS,)}) as (session, _hydrations):
        # Only the NeXus target succeeded for label 0.
        session.record_persisted([session.accounting.receipt(0, M1, NEXUS)])
        session.record_durable([session.accounting.receipt(0, M1, NEXUS)])
        assert K1 in store.hydratable_modes(0), (
            "an exact receipt on a store target + a registered hydrator IS "
            "the store-hydratable projection")
        assert K1 not in store.durable_modes(0), (
            "durability needs EVERY applicable target (mutation 8)")
        store.upsert(_record_of(store, 1))     # a REAL bounds enforcement pass
        assert store.has_heavy_payload(0), f"{bound} must retain the arrays"
        assert store.get(0) is not None, f"{bound} must retain the record"
    assert store.has_heavy_payload(0) and store.get(0) is not None, (
        "the post-terminal sweep releases only CURRENT durable heavy data")


def test_g4_a_thinned_durable_control_recovers_through_the_hydrator():
    """Adding the missing XYE durability permits heavy eviction; the genuinely
    thinned durable control then recovers through the REGISTERED hydrator."""
    _requires_session_projection()
    store = _InstrumentedStore(max_heavy_items=1, max_items=None)
    with _hydratable_run(store, {M1: (NEXUS, XYE)},
                         {M1: (NEXUS,)}) as (session, hydrations):
        session.record_durable([session.accounting.receipt(0, M1, NEXUS)])
        assert store.has_heavy_payload(0), "one target is not every target"
        session.record_durable([session.accounting.receipt(0, M1, XYE)])
        assert K1 in store.durable_modes(0)
        assert not store.has_heavy_payload(0), "durable: may now release"
        assert store.get(0) is not None, "the light record is retained"
        assert hydrations == [], "nothing has asked to recover it yet"
        recovered = store.get_or_hydrate(0)
        assert hydrations == [0], (
            f"a THINNED hydratable record recovers ONCE; got {hydrations}")
        assert recovered is not None and _heavy_keys(recovered) == {K1}, (
            "the recovery returns the mode's exact heavy payload")


@pytest.mark.parametrize("case", ("xye-only", "dropped-only"))
def test_g5_durable_without_a_store_target_is_never_hydratable(case):
    """XYE-only output is real ledger durability with NO store hydrator, and a
    dropped-only record is a non-publication: both may release heavy arrays,
    both drive the SAME production hydration path without ever invoking the
    NeXus hydrator, and neither authorizes whole-record deletion."""
    _requires_session_projection()
    store = _InstrumentedStore(max_heavy_items=1, max_items=1)
    xye_only = case == "xye-only"
    with _hydratable_run(store, {M1: (XYE,)} if xye_only else {M1: (NEXUS,)},
                         {M1: ()} if xye_only else {M1: (NEXUS,)}
                         ) as (session, hydrations):
        if xye_only:
            session.record_durable([session.accounting.receipt(0, M1, XYE)])
            assert K1 in store.durable_modes(0)
        else:
            session.record_publication_dropped(0, M1, expected_revision=1)
            session.record_publication_dropped(0, M1, expected_revision=1)
            assert K1 in store.dropped_modes(0), "a duplicate drop is a no-op"
            with pytest.raises(ValueError):     # above current is rejected
                session.record_publication_dropped(0, M1, expected_revision=9)
        assert K1 not in store.hydratable_modes(0), (
            "durability/non-publication must NOT imply hydratability (7)")
        assert not store.has_heavy_payload(0), "heavy arrays may be released"
        assert store.get_or_hydrate(0) is not None
        assert hydrations == [], (
            "the SAME production hydration path must NOT invoke the NeXus "
            f"hydrator for a mode with no store target; got {hydrations}")
        assert store.get(0) is not None, (
            "deletion needs every remaining non-dropped mode hydratable and "
            "that set non-empty; retained LIGHT (17)")


# ══ G6 — the conservative result/store transaction ═════════════════════════

def test_g6_pressure_between_atomic_clear_and_mint_sees_conservative_state():
    """A real pressure reader event-gated BETWEEN the atomic store clear and
    the revision mint sees dirty state, never a stale positive."""
    _requires_session_projection()
    store = _InstrumentedStore(max_heavy_items=None)
    frames = _frames(1)
    observed: dict = {}
    with _run(frames, store=store, **NX1) as session:
        try:
            session.submit(frames[0])
            _drain(session)
            session.record_durable([session.accounting.receipt(0, M1, NEXUS)])
            assert K1 in store.durable_modes(0)
            store.gate_label = 0        # gate the CLEAR of the affected pair
            store.gate_release.clear()
            session.resume()

            def _reader():
                assert store.gate_entered.wait(WAIT), "clear never happened"
                observed["durable"] = set(store.durable_modes(0))
                observed["heavy"] = store.has_heavy_payload(0)
                observed["revision"] = session.accounting.current_revision(0, M1)
                store.gate_release.set()

            reader = threading.Thread(target=_reader, name="c2a-pressure")
            reader.start()
            session.submit(Frame(0, image=np.full((4, 4), 9.0)))
            _drain(session)
            reader.join(WAIT)
            assert not reader.is_alive(), "pressure reader never finished"
            assert observed["durable"] == set(), (
                "the projection is cleared BEFORE the ledger outcome, so "
                "pressure sees a false NEGATIVE (mutation 4 clears after)")
            assert observed["heavy"] is True, "dirty data is always retained"
            assert observed["revision"] == 1, "no mint before the clear"
            assert session.accounting.current_revision(0, M1) == 2
        finally:
            store.gate_release.set()


def test_g6_exact_outcome_replay_restores_the_unchanged_certification():
    """Replay drives ScanSession's OWN projection owner transaction."""
    _requires_session_projection()
    store = _InstrumentedStore(max_heavy_items=None)
    frames = _frames(1)
    with _run(frames, store=store, **NX1) as session:
        session.submit(frames[0])
        _drain(session)
        session.record_durable([session.accounting.receipt(0, M1, NEXUS)])
        assert set(store.durable_modes(0)) == {K1}
        # An EXACT replay cannot re-mint or invalidate a certification.
        _outcome(session, 0, reduction_core.FrameOutcome.COMPLETED, attempt=1)
        assert session.accounting.current_revision(0, M1) == 1
        assert set(store.durable_modes(0)) == {K1}, (
            "an exact replay restores the unchanged certification")


@pytest.mark.parametrize(
    "leg", ("contradiction", "projection-exception", "failed-upsert"))
def test_g6_conservative_block_survives_an_unrelated_later_receipt(leg):
    """A contradiction, an owner exception or a failed upsert leaves the pair
    blocked: real heavy pressure AND the post-terminal sweep retain its data
    even when the RAW store projection could not be cleared."""
    _requires_session_projection()
    # Keep the durable control AT the bound until the block exists.  Making it
    # durable while already over-bound would legally thin it before this test's
    # discriminator begins (and contradict G4's required immediate eviction).
    store = _InstrumentedStore(max_heavy_items=2)
    frames = _frames(3)
    session = _session(frames, store=store, **NX1)
    try:
        for frame in frames[:2]:
            session.submit(frame)
        _drain(session)
        session.record_durable([session.accounting.receipt(0, M1, NEXUS)])
        assert K1 in store.durable_modes(0)
        assert store.has_heavy_payload(0), (
            "the control is durable but not over-bound before blocking")

        if leg == "contradiction":
            try:        # rejected loudly or fenced quietly: pin the STATE
                _outcome(session, 0, reduction_core.FrameOutcome.FAILED,
                         attempt=1, produced_1d=False)
            except (ValueError, RuntimeError):
                pass
            assert session.accounting_snapshot().dispositions[0] is \
                ItemDisposition.COMPLETED, (
                    "a contradictory outcome for the same attempt is rejected")
        else:
            if leg == "projection-exception":
                store.fail_projection = {0}
            else:
                store.fail_upsert = {0}
            session.resume()
            session.submit(Frame(0, image=np.full((4, 4), 9.0)))
            _drain(session)
            store.fail_projection = set()
            store.fail_upsert = set()

        # Only NOW exceed the bound.  A raw stale durable projection would
        # select label 0, while the session-local block must retain it.
        session.resume()
        session.submit(frames[2])
        _drain(session)
        # An UNRELATED later receipt must not unblock the fenced pair.
        session.record_durable([session.accounting.receipt(1, M1, NEXUS)])
        assert K1 in store.durable_modes(1), "the unrelated pair certifies"
        store.upsert(_record_of(store, 2))     # another REAL enforcement pass
        assert store.has_heavy_payload(0), (
            f"the {leg} leg stays blocked: heavy pressure retains it even "
            "though the RAW store projection could not be cleared, and an "
            "unrelated receipt cannot lift the fence")
    finally:
        session.resume()
        session.finish(raise_on_failure=False)
    assert store.has_heavy_payload(0), (
        f"the post-terminal final sweep must honour the {leg} fence too")


# ══ G7 — the in-flight hydration fence (BEHAVIOURAL ONLY) ══════════════════

def test_g7_event_gated_hydration_cannot_resurrect_the_stale_revision():
    """A real ``get_or_hydrate()`` gated OUTSIDE the store lock races a real
    ``ScanSession`` replacement outcome + post-upsert reconciliation.  No
    capability probe, no signature check, no source inspection: on the accepted
    parent the stale revision-N record merges back over revision N+1 AND
    re-publishes revision N's captured projection."""
    store = FrameRecordStore(max_heavy_items=1,
                             require_persisted_for_eviction=False)
    frames = [Frame(0, image=np.full((4, 4), 1.0), source_identity="src:N",
                    metadata={"revision": "N"}),
              Frame(1, image=np.full((4, 4), 2.0), source_identity="src:other")]
    session = _session(frames, store=store, two_d=True, obligations=(NEXUS,))
    entered = threading.Event()
    release = threading.Event()
    result: dict = {}
    try:
        session.submit(frames[0])
        _drain(session)
        # The EXACT revision-N record object, source identity and complete
        # store projection a hydrator would capture under the store lock:
        # 1-D published, 2-D consciously NOT published.
        stale_record = _record_of(store, 0)
        stale_intensity = _intensity(store, 0)
        stale_source = store.source_identity(0)
        store.mark_persisted(0, modes=[K1])
        store.mark_dropped(0, modes=[K2])
        stale_persisted = set(store.persisted_modes(0))
        stale_heavy = _heavy_keys(_record_of(store, 0))
        session.resume()
        session.submit(frames[1])           # heavy bound thins label 0
        _drain(session)
        assert not store.has_heavy_payload(0), "label 0 must be hydratable"
        assert stale_persisted == {K1} and stale_heavy == {K1}, (
            f"revision N's captured projection is exact: persisted "
            f"{stale_persisted}, dropped 2-D, resident heavy {stale_heavy}")

        from xrd_tools.session import FrameHydrationResult

        def _hydrator(request):
            entered.set()
            assert release.wait(WAIT), "hydrator gate never released"
            return FrameHydrationResult(
                request, stale_record,
            )                            # what revision N had on disk

        store.set_hydrator(_hydrator, revision_qualified=True)
        stale_hydratable = set(store.hydratable_modes(0))
        assert stale_hydratable == {K1}, "revision N is store-hydratable"
        worker = threading.Thread(
            target=lambda: result.__setitem__("record", store.get_or_hydrate(0)),
            name="c2a-hydrator")
        worker.start()
        assert entered.wait(WAIT), "the hydrator never ran outside the lock"

        # While it waits, drive a REAL replacement outcome + post-upsert
        # reconciliation through ScanSession -> ledger revision N+1.
        session.resume()
        session.submit(Frame(0, image=np.full((4, 4), 9.0),
                             source_identity="src:N+1",
                             metadata={"revision": "N+1"}))
        _drain(session)
        assert session.accounting.current_revision(0, M1) == 2
        fresh_intensity = _intensity(store, 0)
        fresh_source = store.source_identity(0)
        assert fresh_intensity != stale_intensity
        assert fresh_source != stale_source

        release.set()
        worker.join(WAIT)
        assert not worker.is_alive(), "the hydrator never completed"

        assert result["record"] is not stale_record, (
            "get_or_hydrate returned the captured revision-N object after the "
            "replacement outcome won the race")
        # Every captured fact is re-asserted against the CURRENT record.
        assert _intensity(store, 0) == fresh_intensity, (
            "the stale hydrated object was written back, resurrecting "
            "revision N (mutation 9: remove the stale hydration fence)")
        assert store.source_identity(0) == fresh_source, (
            "the current record's source identity is the replacement's")
        assert set(store.persisted_modes(0)) != stale_persisted, (
            "the stale capture republished revision N's persisted projection")
        assert set(store.hydratable_modes(0)) != stale_hydratable, (
            "and its hydratable projection with it")
        assert _heavy_keys(_record_of(store, 0)) != stale_heavy, (
            "revision N+1 produced BOTH modes; the stale capture must not "
            "reinstate revision N's partial payload")
        assert set(store.dropped_modes(0)) == set(), (
            "the reconciliation republishes the COMPLETE projection: revision "
            "N's dropped 2-D mode cannot survive a replacement publishing it")
    finally:
        release.set()
        session.resume()
        session.finish(raise_on_failure=False)


# ══ G13 — one atomic store projection replacement (BEHAVIOURAL ONLY) ═══════

def test_g13_projection_publication_is_one_atomic_replacement():
    """A real ``ScanSession`` run over the real store kernel with EVERY
    mandatory target/store/write map, gated against ONE actual bounded
    production enforcement decision.  Per label the 1-D mode is fully durable
    (NeXus is its only applicable target) while the 2-D mode is NeXus-persisted
    but never durable (XYE never succeeded), so the correct decision releases
    1-D arrays yet retains the 2-D payload and the record."""
    midpoint = threading.Event()        # complete atomic OR partial legacy write
    proceed = threading.Event()
    exposed: list = []
    decision: dict = {}

    class _MidpointStore(_InstrumentedStore):
        """Real store; gate complete atomic publication or a partial legacy
        write while the session transaction is still live."""

        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self._atomic_gate_used = False

        def _expose(self, name):
            exposed.append(name)
            midpoint.set()
            assert proceed.wait(WAIT), "pressure reader never released"

        def replace_projection(self, label, **kw):
            out = super().replace_projection(label, **kw)
            # The first replacement AFTER the real store upsert is the
            # mandatory post-upsert reconcile.  Earlier calls are clears or a
            # receipt that beat the upsert and cannot satisfy this row.
            upserted = any(name == "upsert" and owner == label
                           for name, owner in self.calls)
            if label == 0 and upserted and not self._atomic_gate_used:
                self._atomic_gate_used = True
                self._expose("replace_projection")
            return out

        def mark_persisted(self, labels, **kw):
            out = super().mark_persisted(labels, **kw)
            self._expose("mark_persisted")
            return out

        def mark_dropped(self, labels, **kw):
            out = super().mark_dropped(labels, **kw)
            self._expose("mark_dropped")
            return out

    # One result plus one prepared trigger genuinely exceed both bounds.  The
    # pressure actor performs ONE upsert — no get/upsert pseudo-snapshot.
    store = _MidpointStore(max_heavy_items=1, max_items=1)
    frames = _frames(1)
    trigger = FrameRecord.from_view(FrameView.from_results(
        label=99, result_1d=_r1d(99.0), result_2d=_r2d(99.0)))
    session = _session(frames, store=store, two_d=True,
                       targets_by_mode={M1: (NEXUS,), M2: (NEXUS, XYE)},
                       store_targets_by_mode={M1: (NEXUS,), M2: (NEXUS,)},
                       write_targets_by_mode={M1: (NEXUS,), M2: (NEXUS,)},
                       record_store_persisted_on_write=True)

    def _pressure():
        assert midpoint.wait(WAIT), "the run never signalled the reader"
        decision["midpoint"] = exposed[-1] != "replace_projection"
        decision["at"] = len(store.calls)
        # ONE lock-protected production enforcement decision over a distinct,
        # already-prepared record — never a split read/modify/write sequence.
        store.upsert(trigger)
        current = store.get(0)
        decision["heavy"] = set() if current is None else _heavy_keys(current)
        decision["present"] = current is not None
        proceed.set()

    reader = threading.Thread(target=_pressure, name="c2a-projection-reader")
    reader.start()
    try:
        session.submit(frames[0])
        _drain(session)
    finally:
        session.resume()
        session.finish(raise_on_failure=False)
        midpoint.set()          # deterministic release: no timing sleep
        proceed.set()
        reader.join(WAIT)
    assert not reader.is_alive()

    owner = store.calls[:decision.get("at", len(store.calls))]
    names = [name for name, _label in owner]
    label_0 = [(i, name) for i, (name, label) in enumerate(store.calls)
               if label == 0]
    assert [n for n in names if n.startswith("mark_")] == [], (
        "reconciliation may NEVER publish through independent mark_* calls; "
        f"this run observed {names!r} (ratification mutation 5)")
    upsert_at = next(i for i, (_name, label) in enumerate(store.calls)
                     if _name == "upsert" and label == 0)
    before = [name for i, name in label_0 if i < upsert_at]
    after = [name for i, name in label_0 if i > upsert_at]
    assert before and all(name == "replace_projection" for name in before)
    assert after == ["replace_projection"], (
        "the exact owner graph needs ONE complete post-upsert reconcile; an "
        "earlier clear/receipt cannot mask its omission or duplication "
        f"(mutation 6).  Before={before!r}, after={after!r}")
    assert not decision.get("midpoint"), (
        "a pressure reader took its one enforcement decision from a "
        "MID-TRANSACTION projection")
    assert decision.get("present"), (
        "NOT whole-record deletable: the resident heavy 2-D mode is persisted "
        "but not durable (XYE never succeeded)")
    assert K2 in decision.get("heavy", set()), (
        "that same decision may release the fully durable 1-D arrays but must "
        f"retain the non-durable 2-D payload; it kept {decision.get('heavy')}")


# ══ G14 — one idempotent post-terminal final sweep ═════════════════════════

def test_g14_final_sweep_runs_once_after_the_terminal_boundary():
    """ONE post-terminal sweep per first boundary, none on a repeat."""
    _requires_session_projection()
    store = _InstrumentedStore(max_heavy_items=None)
    frames = _frames(1)
    sink = _ProbeSink(lambda: store.has_heavy_payload(0))
    session = _session(frames, store=store, sink=sink, **NX1)
    session.submit(frames[0])
    _drain(session)
    session.record_durable([session.accounting.receipt(0, M1, NEXUS)])
    heavy = _record_of(store, 0)            # the resident heavy record
    session.resume()
    session.finish(raise_on_failure=False)

    assert sink.terminal == [("finish", True)], (
        "the sweep runs AFTER the terminal boundary returns; the sink still "
        "saw resident heavy data")
    assert not store.has_heavy_payload(0), (
        "ONE post-terminal sweep releases current durable heavy data (18)")
    hydrations: list[int] = []
    from xrd_tools.session import FrameHydrationResult
    store.set_hydrator(
        lambda request: (
            hydrations.append(int(request.label)),
            FrameHydrationResult(request, heavy),
        )[1],
        revision_qualified=True,
    )
    store.get_or_hydrate(0)                 # same certified revision, re-armed
    assert hydrations == [0]
    assert store.has_heavy_payload(0), "the re-armed payload is resident"
    assert K1 in store.durable_modes(0), (
        "re-arming must preserve exact-current durability; otherwise a second "
        "erroneous sweep would retain it and false-green")
    session.finish(raise_on_failure=False)
    assert sink.terminal == [("finish", True)], "one terminal boundary"
    assert store.has_heavy_payload(0), (
        "a repeated finish() runs NO second sweep: the re-armed durable heavy "
        "payload is untouched")


@pytest.mark.parametrize("terminal", ("finish", "abort"))
def test_g14_final_sweep_retains_persisted_stale_blocked_and_failed_data(
        terminal):
    """The sweep releases only CURRENT durable heavy data, else retains — on
    BOTH real session terminal paths (a clean finish and a real run failure)."""
    _requires_session_projection()
    store = _InstrumentedStore(max_heavy_items=None)
    frames = _frames(5)
    sink = _ProbeSink()
    session = _session(frames, store=store, sink=sink, **NX1)
    for frame in frames[:4]:
        session.submit(frame)
    _drain(session)
    for label in (0, 2, 3):
        session.record_durable([session.accounting.receipt(label, M1, NEXUS)])
    session.record_persisted([session.accounting.receipt(1, M1, NEXUS)])
    # label 2 becomes STALE: a replacement mints revision 2 over the durable 1.
    session.resume()
    session.submit(Frame(2, image=np.full((4, 4), 9.0)))
    _drain(session)
    # label 3 was DURABLE and is now BLOCKED by a real projection-owner
    # exception, so ignoring the block would wrongly evict live data.
    assert store.has_heavy_payload(3) and K1 in store.durable_modes(3)
    store.fail_projection = {3}
    session.resume()
    session.submit(Frame(3, image=np.full((4, 4), 9.0)))
    _drain(session)
    store.fail_projection = set()
    session.resume()
    if terminal == "abort":
        sink.fail_write = {4}       # a REAL sink-write failure fails the run
        session.submit(frames[4])
    session.finish(raise_on_failure=False)

    assert [name for name, _s in sink.terminal] == [terminal], (
        f"the run must reach the real {terminal} terminal boundary")
    if terminal == "finish":
        assert not store.has_heavy_payload(0), "durable data is released"
    assert store.has_heavy_payload(1), "persisted-only data is retained"
    assert store.has_heavy_payload(2), "a stale receipt cannot authorize it"
    assert store.has_heavy_payload(3), (
        "the blocked pair's RAW projection still reads durable, so ignoring "
        "the session fence would wrongly evict it; the sweep fails toward "
        "retention")


# ══ G15 — bounded AST owner censuses ═══════════════════════════════════════

def _src_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2] / "src"


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _chain(node) -> str | None:
    """The dotted receiver text of an attribute expression, or None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return None
    return ".".join(reversed(parts))


def _scopes(tree: ast.Module) -> dict[int, str]:
    scope: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                scope.setdefault(id(child), node.name)
    return scope


def _method_calls(path: pathlib.Path, names: set[str]):
    """Bounded fact: (attr, receiver_text, enclosing_function, line) sites."""
    tree = _tree(path)
    scope = _scopes(tree)
    return [(node.func.attr, _chain(node.func.value),
             scope.get(id(node), "<module>"), node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr in names]


def _constructor_calls(path: pathlib.Path, names: set[str]):
    tree = _tree(path)
    scope = _scopes(tree)
    return [(node.func.id, scope.get(id(node), "<module>"), node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in names]


def _py_files(root: pathlib.Path):
    """``(path_relative_to_src, path)`` for every production module."""
    return [(str(p.relative_to(root)), p) for p in sorted(root.rglob("*.py"))
            if ".egg-info" not in str(p)]


def _tokens(node) -> set[str]:
    """Every attribute/name token appearing inside one expression."""
    return {child.attr for child in ast.walk(node)
            if isinstance(child, ast.Attribute)} | {
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def test_g15_one_ledger_construction_site_and_one_cadence_definition():
    src = _src_root()
    sites = [(rel, where, line) for rel, path in _py_files(src)
             for _name, where, line in _constructor_calls(path, {"StageLedger"})]
    assert [(rel, where) for rel, where, _line in sites] == [
        ("xrd_tools/session/scan_session.py", "__init__")], (
        f"exactly ONE ledger CONSTRUCTION SITE (not one file); got {sites}")

    definitions = [(rel, node.lineno) for rel, path in _py_files(src)
                   for node in ast.walk(_tree(path))
                   if isinstance(node, ast.ClassDef)
                   and node.name == "FlushPolicy"]
    assert len(definitions) == 1 and definitions[0][0] == \
        "xrd_tools/session/policy.py", (
        "the cadence DEFINITION moves behind the session policy owner; a "
        f"second one is mutation 19.  Got {definitions}")
    cadence = src / "xrd_tools/reduction/cadence.py"
    reexports = [node for node in _tree(cadence).body
                 if isinstance(node, ast.ImportFrom)
                 and any(alias.name == "FlushPolicy" for alias in node.names)]
    assert reexports, (
        "xrd_tools.reduction.cadence becomes a COMPATIBILITY RE-EXPORT")


_CADENCE_CONTROLLERS = (
    "xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py",
    "xdart/gui/tabs/static_scan/wranglers/nexus_wrangler_thread.py",
)
_CADENCE_ADAPTER = "xdart/gui/tabs/static_scan/wranglers/scan_session.py"
_CADENCE_OBSERVER = "xdart/gui/tabs/static_scan/wranglers/qt_nexus_sink.py"
_POLICY_NAMES = {"FlushPolicy", "SessionPolicy"}
_CADENCE_COUNTERS = {"_since_save", "_frames_since_save", "frames_since_save"}
_CADENCE_THRESHOLDS = {"LIVE_SAVE_INTERVAL", "_LIVE_SAVE_INTERVAL",
                       "live_save_interval", "_flush_interval",
                       "_in_memory_cap", "hard_threshold"}


def test_g15_three_cadence_consumers_share_one_policy_definition():
    """Controllers delegate cadence; the display observer owns none of it."""
    src = _src_root()
    authority: list[tuple] = []
    for rel in (*_CADENCE_CONTROLLERS, _CADENCE_OBSERVER):
        path = src / rel
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(
                    alias.name in _POLICY_NAMES for alias in node.names):
                authority.append((rel, "import", node.lineno))
            if isinstance(node, ast.Call):
                tail = (node.func.id if isinstance(node.func, ast.Name)
                        else node.func.attr if isinstance(node.func, ast.Attribute)
                        else None)
                if tail in _POLICY_NAMES:
                    authority.append((rel, "construct", node.lineno))
            if (isinstance(node, ast.Compare)
                    and _tokens(node) & _CADENCE_COUNTERS
                    and _tokens(node) & _CADENCE_THRESHOLDS):
                authority.append((rel, "comparison", node.lineno))
    assert authority == [], (
        "controllers and QtFrameObserver may not import, construct or recreate "
        f"the session cadence policy; found {authority}")

    for rel in _CADENCE_CONTROLLERS:
        calls = _method_calls(src / rel, {"should_flush", "commit_epoch"})
        assert {attr for attr, _recv, _where, _line in calls} == {
            "should_flush", "commit_epoch",
        }, f"{rel} must delegate the complete cadence decision; got {calls}"

    adapter_calls = _method_calls(
        src / _CADENCE_ADAPTER, {"should_flush", "commit_epoch"})
    assert [row[:3] for row in adapter_calls if row[0] == "should_flush"] == [
        ("should_flush", "self._session.policy", "should_flush")
    ], f"the adapter must delegate to the mounted session policy: {adapter_calls}"
    assert [row[:3] for row in adapter_calls if row[0] == "commit_epoch"] == [
        ("commit_epoch", "self._session", "commit_epoch")
    ], f"the adapter must delegate the matching epoch commit: {adapter_calls}"

    observer_calls = _method_calls(
        src / _CADENCE_OBSERVER, {"should_flush", "commit_epoch"})
    observer_tokens = _tokens(_tree(src / _CADENCE_OBSERVER))
    assert observer_calls == [] and not (_CADENCE_COUNTERS & observer_tokens), (
        "QtFrameObserver is display-only and may own no cadence state; "
        f"calls={observer_calls}, counters={_CADENCE_COUNTERS & observer_tokens}")


_PROJECTION_WRITES = {"mark_persisted", "mark_durable", "mark_dropped",
                      "replace_projection"}
_STORE_RECEIVERS = ("record_store", "_record_store", "_streaming_record_store",
                    "store", "_store")


def test_g15_scan_session_is_the_only_production_store_projection_writer():
    """Owner-aware census: ``LiveFrameSeries.mark_persisted`` (a ``.frames``
    receiver) is a DIFFERENT owner and stays legal everywhere."""
    offenders: list[tuple] = []
    live_series: list[tuple] = []
    for rel, path in _py_files(_src_root()):
        for attr, recv, where, line in _method_calls(path, _PROJECTION_WRITES):
            tail = "" if recv is None else recv.rsplit(".", 1)[-1]
            if tail == "frames":
                live_series.append((rel, attr, recv))
            elif rel != "xrd_tools/session/scan_session.py":
                offenders.append((rel, attr, recv, where, line))
    assert live_series, (
        "the LiveFrameSeries owner stays distinguishable from the store API")
    assert offenders == [], (
        "ScanSession is the ONLY production writer of the store projection; "
        f"found {offenders} (mutation 19: a direct GUI writer)")


def test_g15_c2_headless_store_projection_owner_split():
    """C2 census: headless/core only; the nine canonical consumers stay C3."""
    root = _src_root()
    canonical_gui = {
        "xdart/gui/tabs/static_scan/h5viewer.py",
        "xdart/gui/tabs/static_scan/scan_threads.py",
        "xdart/gui/tabs/static_scan/static_scan_widget.py",
        "xdart/gui/tabs/static_scan/wranglers/image_wrangler.py",
        "xdart/gui/tabs/static_scan/wranglers/image_wrangler_thread.py",
        "xdart/gui/tabs/static_scan/wranglers/nexus_wrangler.py",
        "xdart/gui/tabs/static_scan/wranglers/nexus_wrangler_thread.py",
        "xdart/gui/tabs/static_scan/wranglers/qt_nexus_sink.py",
        "xdart/gui/tabs/static_scan/wranglers/wrangler_widget.py",
    }
    scoped = [
        (rel, path) for rel, path in _py_files(root)
        if rel.startswith("xrd_tools/") or rel.startswith("xdart/modules/")
    ]
    assert canonical_gui.isdisjoint(rel for rel, _path in scoped)
    offenders: list[tuple] = []
    live_series: list[tuple] = []
    for rel, path in scoped:
        for attr, recv, where, line in _method_calls(path, _PROJECTION_WRITES):
            tail = "" if recv is None else recv.rsplit(".", 1)[-1]
            if tail == "frames":
                live_series.append((rel, attr, recv))
            elif rel != "xrd_tools/session/scan_session.py":
                offenders.append((rel, attr, recv, where, line))
    series = _tree(root / "xdart/modules/ewald/frame_series.py")
    live_class = next(node for node in ast.walk(series)
                      if isinstance(node, ast.ClassDef)
                      and node.name == "LiveFrameSeries")
    assert "mark_persisted" in {
        node.name for node in live_class.body if isinstance(node, ast.FunctionDef)
    }, "the distinct LiveFrameSeries.frames owner disappeared"
    assert offenders == [], (
        "only xrd_tools/session/scan_session.py may write the C2 headless "
        f"store projection; found {offenders}"
    )


def _assert_scan_session_bind_contract(src: pathlib.Path) -> None:
    cls = next(node for node in _tree(src / "xrd_tools/session/scan_session.py").body
               if isinstance(node, ast.ClassDef) and node.name == "ScanSession")
    init = next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    binds = [node for node in ast.walk(init) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "bind_session"]
    forward = [node for node in binds
               if _chain(node.func.value) in {"sink", "self._user_sink"}]
    restores = [node for node in binds if _chain(node.func.value) == "nexus"]
    writers = [node.lineno for node in ast.walk(init) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name)
               and node.func.id == "ReductionSession"]
    assert len(forward) == 1 and writers and forward[0].lineno < min(writers), (
        "ScanSession must bind one selected sink before constructing the writer; "
        f"forward={[node.lineno for node in forward]}, writers={writers}")
    bind = forward[0]
    assert len(bind.args) == 1 and not bind.keywords
    assert ast.unparse(bind.args[0]) == (
        "self._dynamic_boundary if dynamic_accounting is not None else "
        "_StageBoundaryFacade(self)"
    ), "the selected sink must receive only the exact writer boundary/facade"

    boundary = [node for node in ast.walk(init) if isinstance(node, ast.Assign)
                and any(_chain(target) == "self._dynamic_boundary"
                        for target in node.targets)]
    assert len(boundary) == 1 and ast.unparse(boundary[0].value) == (
        "None if dynamic_accounting is None else dynamic_accounting.writer_boundary"
    ), "the dynamic facade must be the accounting writer boundary"
    assert len(restores) == 2 and all(
        len(node.args) == 1 and not node.keywords
        and ast.unparse(node.args[0]) == "prior_facade"
        for node in restores
    ), "both constructor-failure paths must restore the borrowed prior facade"


def test_g15_sink_uses_the_private_facade_and_xrd_tools_stays_qt_free():
    src = _src_root()
    _assert_scan_session_bind_contract(src)

    tree = _tree(src / "xdart/gui/tabs/static_scan/wranglers/qt_nexus_sink.py")
    assert "bind_session" not in {node.name for node in ast.walk(tree)
                                  if isinstance(node, ast.FunctionDef)}, (
        "QtFrameObserver is display-only and exposes no session-binding hook")
    reads = sorted({chain for chain in (_chain(node) for node in ast.walk(tree)
                                        if isinstance(node, ast.Attribute))
                    if chain and "session" in chain
                    and chain.endswith((".accounting", ".record_store"))})
    assert reads == [], (
        "the sink and XYE helper never read session.accounting / "
        f"session.record_store; found {reads}")

    qt_offenders: list[tuple] = []
    for rel, path in _py_files(src):
        if not rel.startswith("xrd_tools/"):
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import):
                module = ",".join(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
            else:
                continue
            if any(token in module for token in
                   ("PySide6", "PyQt", "pyqtgraph", "xdart")):
                qt_offenders.append((rel, module))
    assert qt_offenders == [], (
        f"xrd_tools must never import Qt or xdart; found {qt_offenders}")


def test_g15_c2_private_facade_and_headless_import_purity_split():
    """The headless facade owner and import-purity split survive C3."""
    src = _src_root()
    _assert_scan_session_bind_contract(src)

    offenders: list[tuple[str, str]] = []
    for rel, path in _py_files(src):
        if not rel.startswith("xrd_tools/"):
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            offenders.extend(
                (rel, module) for module in modules
                if module.startswith(("PySide", "PyQt", "pyqtgraph", "qtpy", "xdart"))
            )
    assert offenders == [], f"xrd_tools imports Qt or xdart: {offenders}"
