"""Exact-object discriminators for GI reuse ownership and its byte allowance."""

import threading

import numpy as np
import pytest

from tests.core.test_warm_integrators_gi import SHAPE, _plan, _reduce, _scan
import xrd_tools.reduction.core as core
from xrd_tools.reduction import MemorySink, ReductionSession
from xrd_tools.reduction.core import GIFreezeError, WarmIntegrators, _ReductionIntegratorProvider


def test_an_oversized_real_integrator_is_not_retained_over_a_positive_allowance():
    reuse = WarmIntegrators(1, allowance=1)
    _reduce(_scan("oversized", (0.2,)), reuse=reuse)
    retained = sum(size for entries in reuse._idle.values() for _ai, size in entries)
    assert retained <= 1, f"idle table bytes {retained} exceed allowance 1"


def test_a_failed_two_angle_scout_returns_neither_angles_tables():
    reuse = WarmIntegrators(1, allowance=1 << 30)
    scan = _scan("blank-two-angles", (0.1, 0.2))
    for frame in scan.frames:
        frame.image = np.zeros(SHAPE, dtype=np.float32)
    plan = _plan(scan)
    plan.mask[:] = True
    with pytest.raises(GIFreezeError):
        ReductionSession(
            plan, scan, sink=MemorySink(), execution="streaming", executor=1,
            integrator_reuse=reuse, gi_freeze_mode="scout_union",
        )
    assert len(reuse) == 0, f"failed scout returned idle keys {list(reuse._idle)}"
    assert not reuse._lent


def test_an_early_return_cannot_repopulate_a_discarded_and_cleared_run(monkeypatch):
    reuse = WarmIntegrators(1, allowance=1 << 30)
    scan = _scan("late-handback", (0.1, 0.2))
    plan = _plan(scan)
    provider = _ReductionIntegratorProvider(
        scan=scan, plan=plan, ai=scan.integrator, fi=None,
        initial_incident_angle=0.1, reuse=reuse,
    )
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    real_return = reuse.give_back
    errors = []

    def delayed_return(key, integrator):
        entered.set()
        assert release.wait(5)
        real_return(key, integrator)

    monkeypatch.setattr(reuse, "give_back", delayed_return)

    def work():
        try:
            masks = {}
            for frame in scan.frames:
                core._reduce_frame(frame, None, plan, provider, masks)
        except BaseException as exc:
            errors.append(exc)

    def close():
        provider.discard()
        reuse.clear()
        closed.set()

    worker = threading.Thread(target=work)
    worker.start()
    assert entered.wait(5)
    closer = threading.Thread(target=close)
    closer.start()
    # Correct atomic transfer may keep discard waiting; release it either way.
    closed.wait(0.1)
    release.set()
    worker.join(5)
    closer.join(5)
    assert not worker.is_alive() and not closer.is_alive()
    assert not errors
    assert len(reuse) == 0, "a late early-return repopulated the cleared Run store"
    assert not reuse._lent


def test_a_failed_fiber_is_not_reused_and_then_returned_by_a_later_frame(monkeypatch):
    reuse = WarmIntegrators(1, allowance=1 << 30)
    scan = _scan("failed-then-same-angle", (0.1, 0.1, 0.2))
    plan = _plan(scan)
    provider = _ReductionIntegratorProvider(
        scan=scan, plan=plan, ai=scan.integrator, fi=None,
        initial_incident_angle=0.1, reuse=reuse,
    )
    real_2d = core._run_gi_2d
    failed = []

    def fail_after_first_real_integration(image, integrator, *args, **kwargs):
        result = real_2d(image, integrator, *args, **kwargs)
        if not failed:
            failed.append(integrator)
            raise RuntimeError("first GI frame failed")
        return result

    monkeypatch.setattr(core, "_run_gi_2d", fail_after_first_real_integration)
    masks = {}
    try:
        with pytest.raises(RuntimeError, match="first GI frame failed"):
            core._reduce_frame(scan.frames[0], None, plan, provider, masks)
        for frame in scan.frames[1:]:
            core._reduce_frame(frame, None, plan, provider, masks)
        assert all(ai is not failed[0] for entries in reuse._idle.values() for ai, _size in entries), (
            "the failed GI integrator was reused and returned after a later frame"
        )
    finally:
        provider.discard()
        reuse.clear()


def test_a_dynamic_mask_declines_this_providers_cache_before_building(monkeypatch):
    reuse = WarmIntegrators(1, allowance=1 << 30)
    _reduce(_scan("warm", (0.1,)), reuse=reuse)
    assert len(reuse) == 1
    scan = _scan("dynamic-mask", (0.1,))
    scan.frames[0].mask = np.zeros(SHAPE, dtype=bool)
    scan.frames[0].mask[0, 0] = True
    plan = _plan(scan)
    provider = _ReductionIntegratorProvider(
        scan=scan, plan=plan, ai=scan.integrator, fi=None,
        initial_incident_angle=0.1, reuse=reuse,
    )
    real_build = provider._new_fiber
    idle_at_build = []

    def build(gi):
        idle_at_build.append((len(reuse), dict(reuse._lent)))
        return real_build(gi)

    monkeypatch.setattr(provider, "_new_fiber", build)
    core._reduce_frame(scan.frames[0], None, plan, provider, {})
    provider.give_back()
    assert idle_at_build == [(1, {next(iter(reuse._sizes)): 1})], (
        "the active unpriceable worker must be reserved while the known warm table remains idle"
    )
    assert len(reuse) == 1 and not reuse._lent


def test_unmeasurable_tables_are_not_retained_with_a_positive_allowance():
    reuse = WarmIntegrators(1, allowance=1 << 30)
    scan = _scan("unmeasured", (0.1,))
    key = core._warm_integrator_key("fiber", scan.integrator, None, SHAPE, _plan(scan), angle=0.1)
    assert core._table_bytes(scan.integrator) == 0
    assert reuse.borrow(key) is None
    reuse.give_back(key, scan.integrator)
    assert len(reuse) == 0 and not reuse._lent


def test_unpriceable_provider_reserves_known_family_and_keeps_cross_run_tables():
    from tests.core.test_warm_integrators import _plan as standard_plan, _reduce as reduce_standard
    from tests.core.test_warm_integrators import _scan as standard_scan

    reuse = WarmIntegrators(2, allowance=1 << 30)
    reduce_standard(standard_scan("priced", frames=1), standard_plan(), reuse=reuse)
    assert len(reuse) == 1
    key = next(iter(reuse._idle))
    family = key[:2]
    price = reuse._sizes[family]
    reuse.fit(2, allowance=3 * price)

    unpriceable = standard_scan("flatfield", flat=True)
    plan = standard_plan()
    provider = _ReductionIntegratorProvider(
        scan=unpriceable, plan=plan, ai=unpriceable.integrator, fi=None,
        initial_incident_angle=None, reuse=reuse,
    )
    assert provider._reuse_key(None, SHAPE) is None

    lent_price = sum(reuse._sizes[item] * count for item, count in reuse._lent.items())
    idle_price = sum(size for entries in reuse._idle.values() for _ai, size in entries)
    assert reuse._lent.get(family) == reuse._limit
    assert idle_price == price, "known cross-Run standard tables should remain inside the residual allowance"
    assert idle_price + lent_price <= reuse._allowance
    provider.give_back()
    assert len(reuse) == 1 and not reuse._lent


def test_unpriceable_provider_without_a_known_family_keeps_conservative_clear():
    from tests.core.test_warm_integrators import _plan as standard_plan, _reduce as reduce_standard
    from tests.core.test_warm_integrators import _scan as standard_scan

    reuse = WarmIntegrators(1, allowance=1 << 30)
    reduce_standard(standard_scan("priced", frames=1), standard_plan(), reuse=reuse)
    assert len(reuse) == 1
    unknown = standard_scan("unpriced-flatfield", flat=True, shape=(60, 80))
    provider = _ReductionIntegratorProvider(
        scan=unknown, plan=standard_plan(shape=(60, 80)), ai=unknown.integrator, fi=None,
        initial_incident_angle=None, reuse=reuse,
    )
    assert provider._reuse_key(None, (60, 80)) is None
    assert len(reuse) == 0 and not reuse._lent


def test_discard_serializes_with_a_real_provider_family_reservation(monkeypatch):
    from tests.core.test_warm_integrators import _plan as standard_plan, _reduce as reduce_standard
    from tests.core.test_warm_integrators import _scan as standard_scan

    reuse = WarmIntegrators(2, allowance=1 << 30)
    reduce_standard(standard_scan("priced-for-race", frames=1), standard_plan(), reuse=reuse)
    key = next(iter(reuse._idle))
    family = key[:2]
    price = reuse._sizes[family]
    reuse.fit(2, allowance=3 * price)
    unpriceable = standard_scan("flatfield-reserve-race", flat=True)
    provider = _ReductionIntegratorProvider(
        scan=unpriceable, plan=standard_plan(), ai=unpriceable.integrator, fi=None,
        initial_incident_angle=None, reuse=reuse,
    )

    reserve_entered, allow_reserve = threading.Event(), threading.Event()
    reserve_released = threading.Event()
    real_reserve = reuse.reserve_unpriceable
    real_release = provider._release_unpriceable_reserves

    def delayed_reserve(reserve_family, workers):
        reserve_entered.set()
        assert allow_reserve.wait(5), "test did not release the real reserve operation"
        return real_reserve(reserve_family, workers)

    def observed_release():
        reserve_released.set()
        return real_release()

    monkeypatch.setattr(reuse, "reserve_unpriceable", delayed_reserve)
    monkeypatch.setattr(provider, "_release_unpriceable_reserves", observed_release)
    errors = []

    def reserve_family():
        try:
            provider._preserve_known_family_or_clear("standard", SHAPE)
        except BaseException as error:
            errors.append(error)

    def discard_provider():
        try:
            provider.discard()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=reserve_family)
    worker.start()
    assert reserve_entered.wait(5), "real provider did not reach its known-family reserve"
    discard = threading.Thread(target=discard_provider)
    discard.start()
    try:
        # A correct provider holds _bound_lock from the closed check through
        # reserve installation, so discard cannot release an empty map.
        assert not reserve_released.wait(0.05), "discard passed an in-flight family reservation"
    finally:
        allow_reserve.set()
        worker.join(5)
        discard.join(5)

    assert not worker.is_alive() and not discard.is_alive()
    assert not errors, errors
    assert not provider._unpriceable_reserves
    assert not reuse._lent


def test_companion_scout_recursion_and_reused_maps_match_fresh_results():
    from tests.core.test_gi_companion_modes import _plan as companion_plan, _scan as companion_scan
    from tests.core.test_warm_integrators_gi import _arrays, _same

    def reduce(reuse):
        scan = companion_scan()
        session = ReductionSession(
            companion_plan("qip_qoop", "q_chi"), scan, sink=MemorySink(),
            execution="streaming", executor=1, integrator_reuse=reuse,
            gi_freeze_mode="scout_union",
        )
        for frame in scan.frames:
            session.submit(frame)
        return session.finish()

    reuse = WarmIntegrators(1, allowance=1 << 30)
    reduce(reuse)
    reused, fresh = reduce(reuse), reduce(None)
    assert _same(_arrays(reused), _arrays(fresh))
    for index, frame in reused.frames.items():
        ours, theirs = frame.extra_results_2d["q_chi"], fresh.frames[index].extra_results_2d["q_chi"]
        for name in ("radial", "azimuthal", "intensity", "sigma"):
            a, b = getattr(ours, name), getattr(theirs, name)
            assert (a is None and b is None) or np.array_equal(a, b, equal_nan=True)
