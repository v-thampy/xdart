# -*- coding: utf-8 -*-
"""H10-C2-B — the ONE descriptor-backed session resource envelope.

Every row here is a *semantic* red on the accepted C2-A parent: the envelope
owner does not exist there, so ``_policy()`` fails inside the test body rather
than at collection time.

The closed contract under test:

* one public owner (``xrd_tools.session.policy``) resolves one immutable
  ``SessionResourceAllocation`` from descriptor-backed
  ``SessionResourceRequirements`` plus one envelope;
* the byte equations are exact and their five categories are disjoint
  partitions (staging / records / publication are never a union);
* ``envelope < F`` is a typed pre-effect ``SessionEnvelopeError``;
  ``F <= envelope < M`` is the ONE detector-driven oversize exception;
* an explicit allocation is input evidence, recomputed with the same code;
* ``XDART_PREFETCH_QUEUE_SIZE`` is one request, clamped once.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


# -- the owner under test, imported inside bodies (semantic red, not a
# collection error, on the C2-A parent where it does not yet exist) ----------
def _policy():
    import xrd_tools.session.policy as policy
    for name in ("SessionResourceRequirements", "SessionResourceAllocation",
                 "SessionEnvelopeError", "resolve_session_policy"):
        assert hasattr(policy, name), (
            f"xrd_tools.session.policy must own {name}: it is the single public "
            "resource-envelope owner")
    return policy


def _requirements(**kw):
    p = _policy()
    base = dict(height=64, width=32, native_itemsize=2,
                modes_1d=1, modes_2d=1, npt_1d=1000,
                npt_rad=500, npt_azim=360, sigma_1d=1, sigma_2d=0)
    base.update(kw)
    return p.SessionResourceRequirements(**base)


def _terms(req):
    """P, G, T, A1, A2 recomputed here from the ratified equations."""
    pixels = req.height * req.width
    P = pixels * req.native_itemsize
    G = req.background_bytes
    T = 4 * min(pixels, 256 * 256)
    A1 = req.modes_1d * 8 * req.npt_1d * (2 + req.sigma_1d)
    A2 = req.modes_2d * 8 * ((1 + req.sigma_2d) * req.npt_rad * req.npt_azim
                             + req.npt_rad + req.npt_azim)
    return P, G, T, A1, A2


# ── group 6/12 owner shape ───────────────────────────────────────────────────

def test_g1_requirements_expose_the_exact_ratified_byte_terms():
    req = _requirements()
    P, G, T, A1, A2 = _terms(req)
    assert req.native_frame_bytes == P
    assert req.thumbnail_bytes == T
    assert req.result_1d_bytes == A1
    assert req.result_2d_bytes == A2
    # the complete immutable value IS the fingerprint - never an opaque hash
    assert isinstance(req.fingerprint, tuple)
    assert req.fingerprint == _requirements().fingerprint
    assert req.fingerprint != _requirements(npt_1d=1001).fingerprint


def test_g1_thumbnail_term_is_capped_at_the_256x256_preview():
    small = _requirements(height=8, width=8)
    assert small.thumbnail_bytes == 4 * 64
    big = _requirements(height=4096, width=4096)
    assert big.thumbnail_bytes == 4 * 256 * 256


def test_g1_runtime_mode_multiplicity_is_bounded_by_the_finite_schema():
    p = _policy()
    with pytest.raises(ValueError):
        p.SessionResourceRequirements(height=8, width=8, native_itemsize=2,
                                      modes_1d=6, npt_1d=10)
    with pytest.raises(ValueError):
        p.SessionResourceRequirements(height=8, width=8, native_itemsize=2,
                                      modes_2d=4, npt_rad=10, npt_azim=10)
    # the exact schema maxima are accepted
    p.SessionResourceRequirements(height=8, width=8, native_itemsize=2,
                                  modes_1d=5, npt_1d=10,
                                  modes_2d=3, npt_rad=10, npt_azim=10)


# ── group 3: the exact peak, no double-charged aliases ───────────────────────

def test_g3_source_native_charges_queue_producer_held_and_inflight_once():
    p = _policy()
    req = _requirements()
    P, _G, _T, _A1, _A2 = _terms(req)
    pol = p.resolve_session_policy(
        req, envelope_bytes=64 * 1024 ** 3,
        requests={"queue_depth": 4, "reduction_inflight": 3, "workers": 3,
                  "owner_block_bytes": 8 * P},
        env={})
    alloc = pol.allocation
    # owner block + (queue_depth + producer-held + inflight) * P, exactly once.
    assert alloc.categories["source_native"] == (
        alloc.owner_block_bytes
        + (alloc.queue_depth + 1 + alloc.reduction_inflight) * P)
    # the in-flight native views ALIAS their queue-origin arrays: the charge
    # lives in source_native, never a second time in the worker category.
    assert alloc.categories["worker"] == (
        alloc.workers * (p.INTEGRATOR_RESERVE_BYTES
                         + 8 * req.height * req.width
                         + 4 * req.height * req.width
                         + req.worker_background_bytes)
        + alloc.reduction_inflight * (req.result_1d_bytes + req.result_2d_bytes))


def test_g3_dropping_the_producer_held_frame_changes_the_exact_total():
    """The ``+1`` producer-held native frame is load-bearing arithmetic."""
    p = _policy()
    req = _requirements()
    P, _G, _T, _A1, _A2 = _terms(req)
    pol = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                   requests={"queue_depth": 4}, env={})
    alloc = pol.allocation
    without_producer_held = (alloc.owner_block_bytes
                             + (alloc.queue_depth + alloc.reduction_inflight) * P)
    assert alloc.categories["source_native"] - without_producer_held == P


# ── group 10: three distinct charged partitions ──────────────────────────────

def test_g10_staging_records_and_publication_are_separate_partitions():
    p = _policy()
    req = _requirements()
    P, G, T, A1, A2 = _terms(req)
    pol = p.resolve_session_policy(
        req, envelope_bytes=64 * 1024 ** 3,
        requests={"staging_items": 4, "record_items": 7, "record_heavy_items": 3,
                  "publication_items": 5, "publication_heavy_items": 2,
                  "thumbnail_items": 6},
        env={})
    a = pol.allocation
    assert a.categories["staging"] == a.staging_items * (P + G + A1 + A2 + T)
    assert a.categories["records"] == (a.record_items * A1
                                       + a.record_heavy_items * A2)
    assert a.categories["publication"] == (a.publication_items * A1
                                           + a.publication_heavy_items * (P + G + A2)
                                           + a.thumbnail_items * T)
    # hydration and reintegration can make all three simultaneous, so the
    # minimum is their SUM - a union (max) would understate the real peak.
    union = max(a.categories["staging"], a.categories["records"],
                a.categories["publication"])
    assert (a.categories["staging"] + a.categories["records"]
            + a.categories["publication"]) > union
    assert a.assigned_bytes == sum(a.categories[name] for name in p.CATEGORIES)


def test_g10_simultaneous_hydration_and_staging_hold_distinct_arrays():
    """A hydrated record does not alias the staging array it was thinned from,
    so records must be charged as their own partition beside staging."""
    import numpy as np
    from xrd_tools.core import Axis, FrameRecord, FrameView, TwoDKind
    from xrd_tools.session import FrameHydrationResult, FrameRecordStore

    def _heavy(label):
        return FrameRecord.from_view(
            FrameView(
                label=label,
                axis_2d_x=Axis("Q", "q_A^-1", values=np.linspace(1.0, 2.0, 4)),
                axis_2d_y=Axis("Chi", "chi_deg", values=np.linspace(0.0, 1.0, 3)),
                intensity_2d=np.arange(12, dtype=float).reshape(3, 4) + label,
                two_d_kind=TwoDKind.Q_CHI,
                source_path="/data/master.h5",
                source_frame_index=int(label),
            ),
            mode_2d="q_chi",
        )

    store = FrameRecordStore(max_items=8, max_heavy_items=1,
                             require_persisted_for_eviction=False)
    store.set_hydrator(
        lambda request: FrameHydrationResult(
            request, _heavy(request.label),
        ),
        revision_qualified=True,
    )
    staged = _heavy(0)
    store.upsert(staged)
    store.upsert(_heavy(1))          # heavy pressure thins label 0
    hydrated = store.get_or_hydrate(0)
    assert hydrated is not None
    staged_view = staged.results_2d["q_chi"]
    rehydrated_view = hydrated.results_2d["q_chi"]
    assert rehydrated_view.intensity_2d is not staged_view.intensity_2d
    np_assert = np.testing.assert_allclose
    np_assert(rehydrated_view.intensity_2d, staged_view.intensity_2d)


# ── group 7/8/9: F, M and the ONE detector-driven oversize exception ─────────

def test_g7_envelope_below_the_fixed_floor_raises_a_typed_pre_effect_error():
    p = _policy()
    req = _requirements()
    with pytest.raises(p.SessionEnvelopeError) as exc:
        p.resolve_session_policy(req, envelope_bytes=1024, env={})
    err = exc.value
    assert err.available_bytes == 1024
    assert err.required_bytes > err.available_bytes
    assert err.floor_bytes > err.available_bytes
    assert set(err.categories) == set(p.CATEGORIES)
    assert err.requirements.fingerprint == req.fingerprint


def test_g8_between_floor_and_minimum_is_all_minima_plus_exact_excess():
    p = _policy()
    req = _requirements()
    floor = p.floor_bytes(req)
    minimum = p.minimum_bytes(req)
    assert floor < minimum
    envelope = (floor + minimum) // 2
    pol = p.resolve_session_policy(
        req, envelope_bytes=envelope,
        requests={"queue_depth": 32, "staging_items": 64, "workers": 8},
        env={})
    a = pol.allocation
    assert a.oversize_excess_bytes == minimum - envelope
    assert a.assigned_bytes == minimum
    # every tunable count is pinned to its minimum - no request is honored
    assert (a.queue_depth, a.staging_items, a.workers) == (1, 1, 1)
    assert (a.record_items, a.record_heavy_items) == (1, 1)
    assert (a.publication_items, a.publication_heavy_items,
            a.thumbnail_items) == (1, 1, 1)
    assert a.reduction_inflight == 1
    assert a.owner_block_bytes == req.native_frame_bytes


def test_g9_large_output_grids_cannot_pass_through_the_oversize_exception():
    """F keeps every A1/A2 output-grid term, so a huge grid is NOT detector
    oversize - it is a hard undersize error."""
    p = _policy()
    req = _requirements(npt_1d=4_000_000, npt_rad=4000, npt_azim=3600,
                        height=8, width=8, native_itemsize=2)
    floor = p.floor_bytes(req)
    with pytest.raises(p.SessionEnvelopeError):
        p.resolve_session_policy(req, envelope_bytes=floor - 1, env={})


def test_g9_worker_reserve_and_projection_minima_stay_inside_the_floor():
    p = _policy()
    req = _requirements()
    floor = p.floor_bytes(req)
    A1, A2 = req.result_1d_bytes, req.result_2d_bytes
    # integrator reserve + every A1/A2 minimum survive the detector strip
    assert floor == (p.INTEGRATOR_RESERVE_BYTES        # worker reserve
                     + (A1 + A2)                       # staging
                     + (A1 + A2)                       # records
                     + (A1 + A2)                       # publication
                     + (A1 + A2))                      # reduction inflight
    assert floor > p.INTEGRATOR_RESERVE_BYTES
    # ... and the detector-sized terms are exactly what M - F removes
    P, G, T = req.native_frame_bytes, req.background_bytes, req.thumbnail_bytes
    HW = req.height * req.width
    assert p.minimum_bytes(req) - floor == (
        4 * P                                  # owner block + queue + held + inflight
        + (P + G + T)                          # staging
        + (P + G + T)                          # publication
        + (8 * HW + 4 * HW + req.worker_background_bytes))


def test_g8_at_or_above_the_minimum_there_is_no_oversize_excess():
    p = _policy()
    req = _requirements()
    pol = p.resolve_session_policy(req, envelope_bytes=p.minimum_bytes(req),
                                   env={})
    assert pol.allocation.oversize_excess_bytes == 0
    assert pol.allocation.assigned_bytes <= pol.allocation.envelope_bytes


# ── group 6: an explicit allocation is evidence, never authority ─────────────

def test_g6_explicit_allocation_is_recomputed_and_accepted_when_exact():
    p = _policy()
    req = _requirements()
    built = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                     requests={"queue_depth": 3}, env={})
    again = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                     allocation=built.allocation, env={})
    # after exact revalidation the caller's EXACT object survives - never an
    # equal reconstruction.  ``origin`` records where it was first resolved.
    assert again.allocation is built.allocation
    assert again.allocation.origin == built.allocation.origin


def test_g6_explicit_allocation_with_a_forged_category_is_rejected():
    import dataclasses
    p = _policy()
    req = _requirements()
    good = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                    env={}).allocation
    forged = dataclasses.replace(
        good, categories=dict(good.categories, worker=0))
    with pytest.raises(ValueError):
        p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                 allocation=forged, env={})


def test_g6_explicit_allocation_with_a_forged_total_is_rejected():
    import dataclasses
    p = _policy()
    req = _requirements()
    good = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                    env={}).allocation
    with pytest.raises(ValueError):
        p.resolve_session_policy(
            req, envelope_bytes=64 * 1024 ** 3, env={},
            allocation=dataclasses.replace(good, assigned_bytes=1))


def test_g6_explicit_allocation_for_other_requirements_is_rejected():
    p = _policy()
    req = _requirements()
    other = _requirements(npt_1d=999)
    good = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                    env={}).allocation
    with pytest.raises(ValueError):
        p.resolve_session_policy(other, envelope_bytes=64 * 1024 ** 3,
                                 allocation=good, env={})


def test_g6_explicit_allocation_below_a_count_minimum_is_rejected():
    import dataclasses
    p = _policy()
    req = _requirements()
    good = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                    env={}).allocation
    with pytest.raises(ValueError):
        p.resolve_session_policy(
            req, envelope_bytes=64 * 1024 ** 3, env={},
            allocation=dataclasses.replace(good, counts={**good.counts, "workers": 0}))


def test_g6_workers_may_never_exceed_reduction_inflight():
    import dataclasses
    p = _policy()
    req = _requirements()
    good = p.resolve_session_policy(
        req, envelope_bytes=64 * 1024 ** 3,
        requests={"workers": 2, "reduction_inflight": 4}, env={}).allocation
    assert 1 <= good.workers <= good.reduction_inflight
    with pytest.raises(ValueError):
        p.resolve_session_policy(
            req, envelope_bytes=64 * 1024 ** 3, env={},
            allocation=dataclasses.replace(good, counts={
                **good.counts, "reduction_inflight": 1, "workers": 2}))


# ── group 4: the prefetch request is parsed once and clamped once ────────────

def test_g4_prefetch_env_is_one_request_clamped_by_the_allocation():
    p = _policy()
    req = _requirements()
    generous = p.resolve_session_policy(
        req, envelope_bytes=64 * 1024 ** 3,
        env={p.PREFETCH_QUEUE_ENV: "9"})
    assert generous.allocation.queue_depth == 9

    # the same request under a tight envelope is CLAMPED, not honored
    tight = p.resolve_session_policy(
        req, envelope_bytes=p.minimum_bytes(req) + req.native_frame_bytes,
        env={p.PREFETCH_QUEUE_ENV: "9"})
    assert tight.allocation.queue_depth < 9
    assert tight.allocation.assigned_bytes <= tight.allocation.envelope_bytes


def test_g4_malformed_or_absent_prefetch_request_falls_back_to_the_default():
    p = _policy()
    req = _requirements()
    for raw in ("", "   ", "not-an-int", None):
        env = {} if raw is None else {p.PREFETCH_QUEUE_ENV: raw}
        pol = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                       env=env)
        assert pol.allocation.queue_depth == p.DEFAULT_PREFETCH_QUEUE_DEPTH


def test_g4_resolution_never_reads_the_real_process_environment():
    """The owner snapshots the injected mapping - one parse, no os.environ."""
    p = _policy()
    req = _requirements()
    os.environ[p.PREFETCH_QUEUE_ENV] = "13"
    try:
        pol = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                       env={})
        assert pol.allocation.queue_depth == p.DEFAULT_PREFETCH_QUEUE_DEPTH
    finally:
        os.environ.pop(p.PREFETCH_QUEUE_ENV, None)


# ── grant discipline ─────────────────────────────────────────────────────────

def test_automatic_inflight_is_twice_the_actual_worker_grant():
    p = _policy()
    from xrd_tools.core import staging
    req = p.SessionResourceRequirements(
        height=2167, width=2070, native_itemsize=4,
        modes_1d=1, modes_2d=1, npt_1d=1000,
        npt_rad=500, npt_azim=500)
    envelope = 32 * 1024 ** 3

    def automatic(workers):
        return p.resolve_session_policy(
            req, envelope_bytes=envelope,
            env={staging.REDUCTION_WORKERS_ENV: str(workers)}).allocation

    assert {workers: automatic(workers).reduction_inflight
            for workers in (1, 2, 3, 4, 5, 16)} == {
                1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 16: 32}
    constrained = p.resolve_session_policy(
        req, envelope_bytes=4 * 1024 ** 3,
        env={staging.REDUCTION_WORKERS_ENV: "4"}).allocation
    assert (constrained.workers, constrained.reduction_inflight,
            constrained.staging_items) == (3, 6, 30)


def test_requested_workers_are_normalized_unless_the_mapping_is_explicit(
    monkeypatch,
):
    p = _policy()
    from xrd_tools.core import staging
    monkeypatch.setattr(
        staging, "total_physical_ram_bytes", lambda: 8 * 1024 ** 3,
    )
    req = _requirements()
    normalized = p.resolve_session_policy(
        req, envelope_bytes=64 * 1024 ** 3, requested_workers=12, env={},
    ).allocation
    explicit = p.resolve_session_policy(
        req, envelope_bytes=64 * 1024 ** 3, requested_workers=4,
        requests={"workers": 4}, env={},
    ).allocation
    assert (normalized.workers, normalized.reduction_inflight) == (2, 4)
    assert (explicit.workers, explicit.reduction_inflight) == (4, 8)
    same = p.resolve_session_policy(
        req, allocation=explicit, requests=dict(explicit.counts), env={},
    )
    assert same.allocation is explicit


def test_explicit_workers_only_mapping_gets_automatic_two_per_actual():
    p = _policy()
    req = _requirements()
    automatic = {
        workers: p.resolve_session_policy(
            req, envelope_bytes=64 * 1024 ** 3,
            requests={"workers": workers}, env={},
        ).allocation
        for workers in (12, 16)
    }
    assert {workers: (a.workers, a.reduction_inflight)
            for workers, a in automatic.items()} == {12: (12, 24), 16: (16, 32)}
    bounded = p.resolve_session_policy(
        req, envelope_bytes=64 * 1024 ** 3,
        requests={"workers": 16, "reduction_inflight": 3}, env={},
    ).allocation
    assert (bounded.workers, bounded.reduction_inflight) == (3, 3)


def test_grants_stay_between_each_minimum_and_its_request():
    p = _policy()
    req = _requirements()
    requests = {"queue_depth": 6, "staging_items": 12, "record_items": 4096,
                "record_heavy_items": 12, "publication_items": 4096,
                "publication_heavy_items": 12, "thumbnail_items": 512,
                "workers": 4, "reduction_inflight": 8}
    pol = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                   requests=requests, env={})
    a = pol.allocation
    for name, request in requests.items():
        granted = getattr(a, name)
        assert 1 <= granted <= request, name
    assert a.assigned_bytes <= a.envelope_bytes


def test_no_single_category_may_overrun_the_envelope():
    p = _policy()
    req = _requirements()
    envelope = p.minimum_bytes(req) * 3
    pol = p.resolve_session_policy(
        req, envelope_bytes=envelope,
        requests={"queue_depth": 10_000, "staging_items": 10_000,
                  "record_items": 10_000_000, "workers": 64,
                  "reduction_inflight": 128, "publication_items": 10_000_000},
        env={})
    a = pol.allocation
    assert a.assigned_bytes <= envelope
    for name in p.CATEGORIES:
        assert a.categories[name] <= envelope


def test_the_grant_order_is_one_documented_deterministic_sequence():
    p = _policy()
    assert isinstance(p.GRANT_ORDER, tuple)
    assert set(p.GRANT_ORDER) == {
        "workers", "reduction_inflight", "queue_depth", "owner_block_bytes",
        "staging_items", "record_heavy_items", "publication_heavy_items",
        "thumbnail_items", "record_items", "publication_items"}
    req = _requirements()
    first = p.resolve_session_policy(req, envelope_bytes=7 * 1024 ** 3, env={})
    second = p.resolve_session_policy(req, envelope_bytes=7 * 1024 ** 3, env={})
    assert first.allocation == second.allocation


# ── policy owner integration ─────────────────────────────────────────────────

def test_session_policy_keeps_its_accepted_flush_owner_and_adds_one_allocation():
    p = _policy()
    assert p.SessionPolicy().allocation is None            # cadence-only C1 path
    assert p.SessionPolicy().flush == p.FlushPolicy()
    pol = p.resolve_session_policy(_requirements(),
                                   envelope_bytes=64 * 1024 ** 3,
                                   flush=p.FlushPolicy(interval=3), env={})
    assert pol.flush.interval == 3
    assert pol.allocation is not None
    assert pol.should_flush(frames_since_flush=3) is True


def test_allocation_and_requirements_are_immutable():
    p = _policy()
    pol = p.resolve_session_policy(_requirements(),
                                   envelope_bytes=64 * 1024 ** 3, env={})
    with pytest.raises(Exception):
        pol.allocation.queue_depth = 99
    with pytest.raises(Exception):
        pol.allocation.requirements.height = 99


def test_session_package_exports_the_envelope_names_lazily():
    import xrd_tools.session as session
    for name in ("SessionResourceRequirements", "SessionResourceAllocation",
                 "SessionEnvelopeError", "resolve_session_policy"):
        assert name in session.__all__
        assert getattr(session, name) is not None


def test_g8_explicit_oversize_above_a_minimum_count_is_rejected():
    """In ``F <= envelope < M`` an explicit allocation is legal ONLY with every
    tunable at its exact minimum - an above-minimum count cannot ride the
    narrow detector-driven exception."""
    import dataclasses
    p = _policy()
    req = _requirements()
    envelope = (p.floor_bytes(req) + p.minimum_bytes(req)) // 2
    minima = p.resolve_session_policy(req, envelope_bytes=envelope,
                                      env={}).allocation
    assert minima.oversize_excess_bytes > 0
    p.resolve_session_policy(req, envelope_bytes=envelope, allocation=minima,
                             env={})                       # exact minima: legal
    # A SELF-CONSISTENT above-minimum allocation: every category and total
    # recomputes exactly, so ONLY the minimum-count rule can reject it.  (A
    # naive ``replace`` of one count is caught earlier by the category check,
    # which makes it an inert discriminator.)
    forged = p._build(req, {**minima.counts, "queue_depth": 2}, envelope,
                      p.floor_bytes(req), p.minimum_bytes(req), minima.origin,
                      oversize=minima.oversize_excess_bytes)
    assert dict(forged.categories) != dict(minima.categories)
    with pytest.raises(ValueError):
        p.resolve_session_policy(req, envelope_bytes=envelope, env={},
                                 allocation=forged)


def test_g6_explicit_envelope_provenance_is_the_allocations_declaration():
    """Revalidation uses the allocation's OWN declared envelope; a supplied
    override must equal it, and the machine default is never substituted."""
    p = _policy()
    req = _requirements()
    built = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                     env={}).allocation
    same = p.resolve_session_policy(req, allocation=built, env={})
    assert same.allocation is built
    assert same.allocation.envelope_bytes == 64 * 1024 ** 3
    with pytest.raises(ValueError):
        p.resolve_session_policy(req, envelope_bytes=32 * 1024 ** 3,
                                 allocation=built, env={})


def test_g4_the_environment_is_snapshotted_once_not_read_live():
    """The resolver freezes one mapping and passes it to every pure staging
    helper, so a mutation mid-resolution cannot be observed."""
    p = _policy()
    req = _requirements()

    class _CountingEnv(dict):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.reads = 0

        def get(self, key, default=None):
            self.reads += 1
            return super().get(key, default)

    env = _CountingEnv({p.PREFETCH_QUEUE_ENV: "5"})
    first = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3, env=env)
    assert first.allocation.queue_depth == 5
    reads_after_snapshot = env.reads
    env[p.PREFETCH_QUEUE_ENV] = "9"       # mutating the ORIGINAL mapping
    assert first.allocation.queue_depth == 5
    assert env.reads == reads_after_snapshot


# ── requirements derivation: standard vs GI mode multiplicity ────────────────

class _Plan:
    def __init__(self, one_d=None, two_d=None, gi=None, extra=None):
        self.integration_1d = one_d
        self.integration_2d = two_d
        self.gi = gi
        self.extra = extra or {}


class _Desc:
    def __init__(self, shape=(64, 32), itemsize=2):
        import numpy as np
        self.frame_shape = shape
        self.dtype = np.dtype(f"uint{itemsize * 8}")


class _Int1D:
    npt = 1000
    error_model = None


class _Int2D:
    npt_rad = 500
    npt_azim = 360
    error_model = None


def test_g1_standard_plan_charges_one_mode_per_enabled_dimension():
    p = _policy()
    req = p.requirements_from(_Desc(), _Plan(one_d=_Int1D(), two_d=_Int2D()))
    assert (req.modes_1d, req.modes_2d) == (1, 1)
    assert (req.npt_1d, req.npt_rad, req.npt_azim) == (1000, 500, 360)
    one_only = p.requirements_from(_Desc(), _Plan(one_d=_Int1D()))
    assert (one_only.modes_1d, one_only.modes_2d) == (1, 0)
    assert one_only.result_2d_bytes == 0


def test_g1_gi_plan_charges_its_frozen_mode_sets_when_declared():
    p = _policy()
    plan = _Plan(one_d=_Int1D(), two_d=_Int2D(), gi=object(),
                 extra={"enabled_modes_1d": ("q_total", "q_ip", "q_oop"),
                        "enabled_modes_2d": ("qip_qoop", "q_chi")})
    req = p.requirements_from(_Desc(), plan)
    assert (req.modes_1d, req.modes_2d) == (3, 2)


def test_g1_gi_plan_without_declared_modes_charges_the_schema_maxima():
    """A multi-mode GI run must never be charged as one 1-D / one 2-D result
    merely because the plan carries one integration configuration object."""
    p = _policy()
    req = p.requirements_from(_Desc(),
                              _Plan(one_d=_Int1D(), two_d=_Int2D(), gi=object()))
    assert (req.modes_1d, req.modes_2d) == (p.MAX_MODES_1D, p.MAX_MODES_2D)


def test_g1_requirements_refuse_a_descriptor_without_exact_shape_or_dtype():
    p = _policy()
    blind = _Desc()
    blind.frame_shape = None
    with pytest.raises(ValueError):
        p.requirements_from(blind, _Plan(one_d=_Int1D()))
    no_dtype = _Desc()
    no_dtype.dtype = None
    with pytest.raises(ValueError):
        p.requirements_from(no_dtype, _Plan(one_d=_Int1D()))


# ── group 12: censuses ───────────────────────────────────────────────────────

def _src_root():
    import xrd_tools
    return Path(str(xrd_tools.__file__)).resolve().parent.parent


def _py_files(*packages):
    root = _src_root()
    for package in packages:
        yield from sorted((root / package).rglob("*.py"))


def test_g12_exactly_one_resolver_and_no_envelope_exceeded_flag():
    resolvers, flags = [], []
    for path in _py_files("xrd_tools", "xdart"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "resolve_session_policy"):
                resolvers.append(str(path))
            if isinstance(node, ast.Name) and node.id == "envelope_exceeded":
                flags.append(str(path))
            if isinstance(node, ast.Attribute) and node.attr == "envelope_exceeded":
                flags.append(str(path))
    assert len(resolvers) == 1, resolvers
    assert resolvers[0].endswith("xrd_tools/session/policy.py")
    assert flags == []


def test_g12_the_prefetch_request_is_parsed_in_exactly_one_place():
    """B1 scope: ``xrd_tools`` owns exactly one parse.  The cross-package
    census (the wrangler's retired module constant) is a B2 gate."""
    hits = []
    for path in _py_files("xrd_tools"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant)
                    and node.value == "XDART_PREFETCH_QUEUE_SIZE"):
                hits.append(str(path))
    assert len(hits) == 1, hits
    assert hits[0].endswith("xrd_tools/session/policy.py")


def test_g12_policy_imports_no_qt_and_stays_stdlib_at_import_time():
    path = _src_root() / "xrd_tools" / "session" / "policy.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            top_level.append(node.module or "")
    assert all(mod.split(".")[0] in {"__future__", "dataclasses", "types",
                                     "typing", "collections", "os", "math"}
               for mod in top_level), top_level


# ── correction 1: the public request / requirement / allocation boundary ─────
#
# Requests are positive upper bounds, requirements are exact integers, and an
# explicit allocation is schema-checked before any byte arithmetic runs.

def test_c1_requirements_reject_negative_background_and_worker_background():
    p = _policy()
    with pytest.raises(ValueError):
        _requirements(background_bytes=-1)
    with pytest.raises(ValueError):
        _requirements(worker_background_bytes=-1)


def test_c1_sigma_flags_are_exactly_zero_or_one():
    for bad in (-1, 2, 3):
        with pytest.raises(ValueError):
            _requirements(sigma_1d=bad)
        with pytest.raises(ValueError):
            _requirements(sigma_2d=bad)


def test_c1_requirement_fields_reject_booleans_and_fractions():
    for field, bad in (("height", 64.5), ("width", True), ("native_itemsize", 2.0),
                       ("npt_1d", 1000.5), ("modes_1d", True),
                       ("background_bytes", 1.5), ("sigma_1d", True)):
        with pytest.raises(ValueError):
            _requirements(**{field: bad})


def test_c1_explicit_allocation_rejects_a_fractional_or_malformed_count():
    p = _policy()
    req = _requirements()
    good = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                    env={}).allocation
    import dataclasses
    for counts in (
            {**good.counts, "queue_depth": 1.5},          # fractional
            {**good.counts, "workers": True},             # boolean
            {k: v for k, v in good.counts.items() if k != "workers"},  # missing
            {**good.counts, "extra_knob": 1},             # extra key
    ):
        with pytest.raises(ValueError):
            p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3, env={},
                                     allocation=dataclasses.replace(
                                         good, counts=counts))


def test_c1_explicit_allocation_rejects_a_non_unit_owner_block():
    import dataclasses
    p = _policy()
    req = _requirements()
    good = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                    env={}).allocation
    # owner_block_bytes lives in WHOLE native-frame units
    with pytest.raises(ValueError):
        p.resolve_session_policy(
            req, envelope_bytes=64 * 1024 ** 3, env={},
            allocation=dataclasses.replace(good, counts={
                **good.counts,
                "owner_block_bytes": req.native_frame_bytes + 1}))


def test_c1_explicit_allocation_rejects_a_malformed_category_map():
    import dataclasses
    p = _policy()
    req = _requirements()
    good = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                    env={}).allocation
    for cats in ({k: v for k, v in good.categories.items() if k != "worker"},
                 {**good.categories, "bogus": 1},
                 {**good.categories, "worker": float(good.categories["worker"])}):
        with pytest.raises(ValueError):
            p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3, env={},
                                     allocation=dataclasses.replace(
                                         good, categories=cats))


def test_c1_unknown_zero_negative_or_fractional_requests_are_rejected():
    p = _policy()
    req = _requirements()
    for bad in ({"not_a_knob": 4}, {"queue_depth": 0}, {"workers": -1},
                {"staging_items": 2.5}, {"thumbnail_items": True}):
        with pytest.raises(ValueError):
            p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                     requests=bad, env={})


def test_c1_every_grant_stays_at_or_below_its_own_request():
    """The inflight grant may never exceed its request just because more
    workers were asked for; workers clamp DOWN to the inflight request."""
    p = _policy()
    req = _requirements()
    a = p.resolve_session_policy(
        req, envelope_bytes=64 * 1024 ** 3,
        requests={"workers": 4, "reduction_inflight": 1}, env={}).allocation
    assert a.reduction_inflight == 1
    assert a.workers <= a.reduction_inflight == 1
    for name, request in (("workers", 4), ("reduction_inflight", 1)):
        assert getattr(a, name) <= request


def test_c1_requested_workers_and_the_workers_request_must_agree():
    p = _policy()
    req = _requirements()
    p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                             requested_workers=2, requests={"workers": 2},
                             env={})                       # one request, agreed
    with pytest.raises(ValueError):
        p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                 requested_workers=2, requests={"workers": 3},
                                 env={})


def test_c1_declared_gi_sets_above_the_schema_maximum_are_rejected():
    p = _policy()
    with pytest.raises(ValueError):
        p.requirements_from(_Desc(), _Plan(
            one_d=_Int1D(), two_d=_Int2D(), gi=object(),
            extra={"enabled_modes_1d": tuple(f"m{i}" for i in range(6))}))
    with pytest.raises(ValueError):
        p.requirements_from(_Desc(), _Plan(
            one_d=_Int1D(), two_d=_Int2D(), gi=object(),
            extra={"enabled_modes_2d": tuple(f"m{i}" for i in range(4))}))


def test_c1_empty_or_malformed_gi_declarations_are_rejected():
    p = _policy()
    for extra in ({"enabled_modes_1d": ()},          # empty
                  {"enabled_modes_1d": "q_total"},   # a string is malformed
                  {"enabled_modes_1d": 3}):          # not a sequence
        with pytest.raises(ValueError):
            p.requirements_from(_Desc(), _Plan(one_d=_Int1D(), gi=object(),
                                               extra=extra))


def test_c1_a_declaration_on_a_disabled_dimension_is_rejected():
    p = _policy()
    with pytest.raises(ValueError):
        p.requirements_from(_Desc(), _Plan(
            one_d=_Int1D(), gi=object(),
            extra={"enabled_modes_2d": ("qip_qoop",)}))   # 2-D is disabled
    # ... while absence still charges the conservative maximum for 1-D only
    req = p.requirements_from(_Desc(), _Plan(one_d=_Int1D(), gi=object()))
    assert (req.modes_1d, req.modes_2d) == (p.MAX_MODES_1D, 0)


# ── continuation: the request mapping is validated on EVERY resolution route ─

def test_c1_the_oversize_route_still_validates_the_public_requests():
    """``F <= envelope < M`` returns the minima, but an unknown, zero, negative
    or fractional request must still reject rather than be silently ignored."""
    p = _policy()
    req = _requirements()
    envelope = (p.floor_bytes(req) + p.minimum_bytes(req)) // 2
    assert p.floor_bytes(req) <= envelope < p.minimum_bytes(req)
    for bad in ({"not_a_knob": 1}, {"queue_depth": 0}, {"workers": -2},
                {"staging_items": 1.5}, {"thumbnail_items": True}):
        with pytest.raises(ValueError):
            p.resolve_session_policy(req, envelope_bytes=envelope,
                                     requests=bad, env={})
    # a well-formed bound above the minima still yields the exact minima
    a = p.resolve_session_policy(req, envelope_bytes=envelope,
                                 requests={"queue_depth": 4}, env={}).allocation
    assert a.queue_depth == 1 and a.oversize_excess_bytes > 0


def test_c1_an_explicit_allocation_must_fit_every_supplied_request_bound():
    """With an explicit allocation the supplied requests remain upper bounds:
    a grant above any of them rejects, and the mapping is still schema-checked."""
    p = _policy()
    req = _requirements()
    good = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                    requests={"queue_depth": 6},
                                    env={}).allocation
    assert good.queue_depth == 6
    with pytest.raises(ValueError):
        p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3, env={},
                                 allocation=good,
                                 requests={"queue_depth": good.queue_depth - 1})
    for bad in ({"nope": 1}, {"workers": 0}, {"queue_depth": 2.5}):
        with pytest.raises(ValueError):
            p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                     allocation=good, requests=bad, env={})
    same = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                    allocation=good, env={},
                                    requests={"queue_depth": good.queue_depth})
    assert same.allocation is good


# ── correction 2: each worker spelling is validated BEFORE the twins merge ───

def test_c2_a_malformed_worker_twin_is_rejected_on_every_route():
    """``True == 1`` and ``1.0 == 1``, so comparing the two worker spellings
    before validating them let the merge erase a malformed public request.
    Each spelling must reject on its own, on every resolution route."""
    p = _policy()
    req = _requirements()
    ordinary = 64 * 1024 ** 3
    oversize = (p.floor_bytes(req) + p.minimum_bytes(req)) // 2
    assert p.floor_bytes(req) <= oversize < p.minimum_bytes(req)
    explicit = p.resolve_session_policy(
        req, envelope_bytes=ordinary, env={},
        requests={"workers": 1, "reduction_inflight": 1}).allocation
    routes = ({"envelope_bytes": ordinary},
              {"envelope_bytes": oversize},
              {"envelope_bytes": ordinary, "allocation": explicit})
    for twin in (True, 1.0):
        for route in routes:
            with pytest.raises(ValueError):
                p.resolve_session_policy(req, requested_workers=1, env={},
                                         requests={"workers": twin}, **route)


def test_c2_a_malformed_requested_workers_is_rejected_on_its_own():
    p = _policy()
    req = _requirements()
    for bad in (True, 2.0, 0, -1):
        with pytest.raises(ValueError):
            p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                     requested_workers=bad, env={})
    # two independently valid, equal spellings still canonicalize to one bound
    a = p.resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                 requested_workers=2, requests={"workers": 2},
                                 env={}).allocation
    assert a.workers <= 2
