"""GI integrators reused per incident angle, parallax reused exactly, and the store's byte budget (real pyFAI
integrators, real sessions). Reuse never changes a result: every row compares against a fresh build, bitwise.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("pyFAI")

from pyFAI.detectors import Detector, detector_factory  # noqa: E402
from pyFAI.geometry.core import Geometry  # noqa: E402
from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: E402

import xrd_tools.integrate.gid as gid  # noqa: E402
import xrd_tools.reduction.core as core  # noqa: E402
from xrd_tools.core.containers import PONI  # noqa: E402
from xrd_tools.reduction import Frame, MemorySink, ReductionPlan, ReductionSession, Scan  # noqa: E402
from xrd_tools.reduction.core import (  # noqa: E402
    GI1DMode, GI2DMode, GIFreezeError, GIMode, Integration1DPlan, Integration2DPlan, WarmIntegrators,
    _ReductionIntegratorProvider, _warm_integrator_key,
)

SHAPE = (64, 80)
CYCLE = (0.1, 0.2, 0.3, 0.1, 0.2, 0.3)


def _ai(*, wavelength=1e-10, shape=SHAPE, sensor=None):
    if sensor is None:
        detector = Detector(pixel1=1e-4, pixel2=1e-4, max_shape=shape)
    else:
        detector = detector_factory("Pilatus100k", {"orientation": 3, "sensor": sensor})
    ai = AzimuthalIntegrator(dist=0.1, poni1=0.002, poni2=0.004, rot1=0.01, detector=detector, wavelength=wavelength)
    if sensor is not None:
        ai.enable_parallax(True)
    return ai


def _scan(name, angles, *, seed=0, **calibration):
    ai = _ai(**calibration)
    shape = tuple(ai.detector.shape)
    poni = PONI(dist=0.1, poni1=0.002, poni2=0.004, rot1=0.01, rot2=0.0, rot3=0.0, wavelength=ai.wavelength,
                detector="" if calibration.get("sensor") is None else "Pilatus100k")
    rng = np.random.default_rng(seed)
    return Scan(name, [Frame(index, image=rng.poisson(40.0, shape).astype(np.float32), metadata={"th": angle})
                       for index, angle in enumerate(angles)], poni=poni, integrator=ai)


def _plan(scan, *, gi=True, mask_row=10):
    shape = tuple(scan.integrator.detector.shape)
    mask = np.zeros(shape, dtype=bool)
    mask[mask_row:mask_row + 3, :] = True
    return ReductionPlan(
        integration_1d=Integration1DPlan(npt=40), integration_2d=Integration2DPlan(npt_rad=30, npt_azim=20),
        gi=GIMode(incidence_motor="th", mode_1d=GI1DMode.Q_TOTAL, mode_2d=GI2DMode.QIP_QOOP, sample_orientation=1)
        if gi else None, mask=mask)


def _reduce(scan, *, reuse=None, cores=1, freeze=None, gi=True):
    session = ReductionSession(_plan(scan, gi=gi), scan, sink=MemorySink(), execution="streaming", executor=cores,
                               integrator_reuse=reuse, gi_freeze_mode=freeze)
    for frame in scan.frames:
        session.submit(frame)
    return session.finish()


def _arrays(result):
    out = []
    for index in sorted(result.frames):
        frame = result.frames[index]
        out.append([frame.result_1d.radial, frame.result_1d.intensity, frame.result_1d.sigma, frame.result_2d.radial,
                    frame.result_2d.azimuthal, frame.result_2d.intensity, frame.result_2d.sigma])
    return out


def _same(left, right):
    return len(left) == len(right) and all(
        (a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b, equal_nan=True))
        for one, two in zip(left, right) for a, b in zip(one, two))


@pytest.fixture
def builds(monkeypatch):
    """Real geometry-table builds: pyFAI caches each computed array in ``_cached_array``; count each growth."""
    built = []
    for name in ("center_array", "corner_array", "delta_array"):
        real = getattr(Geometry, name)

        def counting(self, *args, _real=real, **kwargs):
            before = len(self._cached_array)
            out = _real(self, *args, **kwargs)
            if len(self._cached_array) > before:
                built.append(id(self))
            return out

        monkeypatch.setattr(Geometry, name, counting)
    return built


@pytest.fixture
def promotions(monkeypatch):
    """GI integrators the sessions promote (each starts with no tables)."""
    made = []
    real = gid._fiber_from_integrator

    def counting(*args, **kwargs):
        made.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(gid, "_fiber_from_integrator", counting)
    return made


def test_a_gi_series_at_the_same_angle_reuses_the_workers_integrator_and_its_tables(builds, promotions):
    reuse = WarmIntegrators(1)
    _reduce(_scan("first", (0.2, 0.2)), reuse=reuse)
    first = len(builds)
    second = _reduce(_scan("second", (0.2, 0.2), seed=1), reuse=reuse)
    assert len(promotions) == 1 and len(builds) == first  # no new integrator, no new table
    assert _same(_arrays(second), _arrays(_reduce(_scan("second", (0.2, 0.2), seed=1))))


def test_each_incident_angle_keeps_its_own_tables_within_and_across_series(builds, promotions):
    reuse = WarmIntegrators(1, allowance=1 << 30)
    first = _reduce(_scan("first", CYCLE), reuse=reuse)
    per_angle = len(builds) / 3  # the first visit of each of the three angles built its tables
    assert len(promotions) == 3 and len(builds) == 3 * per_angle
    second = _reduce(_scan("second", CYCLE, seed=1), reuse=reuse)
    assert len(promotions) == 3 and len(builds) == 3 * per_angle  # every revisit borrowed its angle's tables
    assert _same(_arrays(first), _arrays(_reduce(_scan("first", CYCLE))))
    assert _same(_arrays(second), _arrays(_reduce(_scan("second", CYCLE, seed=1))))


def test_several_workers_cycling_angles_stay_exact_and_build_less_than_without_the_store(builds):
    reuse = WarmIntegrators(3, allowance=1 << 30)
    _reduce(_scan("first", CYCLE * 2), reuse=reuse, cores=3)
    _reduce(_scan("second", CYCLE * 2, seed=1), reuse=reuse, cores=3)
    with_store = len(builds)
    builds.clear()
    fresh_first = _reduce(_scan("first", CYCLE * 2), cores=3)
    fresh = _reduce(_scan("second", CYCLE * 2, seed=1), cores=3)
    assert with_store < len(builds)
    assert _same(_arrays(_reduce(_scan("second", CYCLE * 2, seed=1), reuse=reuse, cores=3)), _arrays(fresh))
    assert fresh_first.n_processed == 12


def test_a_tight_allowance_keeps_fewer_angles_and_stays_exact(builds):
    probe = WarmIntegrators(1, allowance=1 << 30)
    _reduce(_scan("probe", (0.1,)), reuse=probe)
    size = core._table_bytes(probe.borrow(next(iter(probe._idle))))
    reuse = WarmIntegrators(1, allowance=size + size // 2)  # room for about one angle's tables
    first = _reduce(_scan("first", CYCLE), reuse=reuse)
    assert len(reuse) <= 2 and sum(entry[1] for entries in reuse._idle.values() for entry in entries) <= 2 * size
    assert _same(_arrays(first), _arrays(_reduce(_scan("first", CYCLE))))


def test_the_gi_scout_keeps_its_own_integrator_apart_from_the_workers(promotions):
    reuse = WarmIntegrators(1, allowance=1 << 30)
    first = _reduce(_scan("first", (0.2, 0.2)), reuse=reuse, freeze="scout_union")
    assert len(promotions) == 2  # the scout's and the worker's
    # the workers' first build had no known size, so it dropped the idle scout integrator: built once more
    _reduce(_scan("second", (0.2, 0.2), seed=1), reuse=reuse, freeze="scout_union")
    assert len(promotions) == 3
    third = _reduce(_scan("third", (0.2, 0.2), seed=2), reuse=reuse, freeze="scout_union")
    assert len(promotions) == 3  # both sizes known: both borrowed from now on
    assert {key[0] for key in reuse._idle} == {"scout", "fiber"}
    assert _same(_arrays(first), _arrays(_reduce(_scan("first", (0.2, 0.2)), freeze="scout_union")))
    assert _same(_arrays(third), _arrays(_reduce(_scan("third", (0.2, 0.2), seed=2), freeze="scout_union")))


def test_a_failed_freeze_gives_its_scout_integrator_nothing_back():
    reuse = WarmIntegrators(1, allowance=1 << 30)
    scan = _scan("blank", (0.2,))
    scan.frames[0].image = np.zeros(SHAPE, dtype=np.float32)
    plan = _plan(scan)
    plan.mask[:] = True  # nothing left to scout: the freeze refuses after its loop
    with pytest.raises(GIFreezeError):
        ReductionSession(plan, scan, sink=MemorySink(), execution="streaming", executor=1, integrator_reuse=reuse,
                         gi_freeze_mode="scout_union")
    assert len(reuse) == 0 and not reuse._lent


def test_a_callers_accepted_gi_integrator_is_never_lent_to_the_store():
    reuse = WarmIntegrators(1, allowance=1 << 30)
    scan = _scan("accepted", (0.2,))
    accepted = gid._fiber_from_integrator(scan.integrator, incident_angle=0.2, tilt_angle=0.0, sample_orientation=1)
    provider = _ReductionIntegratorProvider(scan=scan, plan=_plan(scan), ai=scan.integrator, fi=accepted,
                                            initial_incident_angle=0.2, reuse=reuse)
    assert provider.fiber(0.2, None, SHAPE) is accepted  # the owner's own object, not a loan
    worker = []
    thread = threading.Thread(target=lambda: worker.append(provider.fiber(0.2, None, SHAPE)))
    thread.start()
    thread.join()
    provider.fiber_done()
    provider.give_back()
    assert worker[0] is not accepted and len(reuse) == 0 and not reuse._lent


def test_parallax_on_is_reused_exactly_and_a_changed_sensor_or_energy_builds_its_own(monkeypatch):
    made = []
    real = core._ReductionIntegratorProvider._new_standard

    def counting(self):
        made.append(1)
        return real(self)

    monkeypatch.setattr(core._ReductionIntegratorProvider, "_new_standard", counting)
    si = {"material": "Si", "thickness": 0.00045}
    reuse = WarmIntegrators(1)
    _reduce(_scan("first", (0.0,), sensor=si), reuse=reuse, gi=False)
    second = _reduce(_scan("second", (0.0,), seed=1, sensor=si), reuse=reuse, gi=False)
    assert len(made) == 1
    assert _same(_arrays(second), _arrays(_reduce(_scan("second", (0.0,), seed=1, sensor=si), gi=False)))
    keys = {_warm_integrator_key("standard", _ai(sensor=sensor, wavelength=wavelength), None, (195, 487),
                                 _plan(_scan("k", (0.0,), sensor=si)))
            for sensor, wavelength in ((si, 1e-10), ({**si, "thickness": 0.001}, 1e-10), (si, 1.1e-10), (None, 1e-10))}
    assert len(keys) == 4 and None not in keys  # parallax on, thicker sensor, other energy, parallax off: all distinct
    before = len(made)
    third = _reduce(_scan("third", (0.0,), seed=2, sensor={**si, "thickness": 0.001}), reuse=reuse, gi=False)
    assert len(made) == before + 1  # a thicker sensor: its own build
    assert _same(_arrays(third), _arrays(_reduce(_scan("third", (0.0,), seed=2, sensor={**si, "thickness": 0.001}),
                                                 gi=False)))


class _Tables:
    """A stand-in integrator holding ``size`` bytes of tables (the store measures what an integrator holds)."""

    def __init__(self, size):
        self.engines = {}
        self._cached_array = {"table": np.zeros(size, dtype=np.uint8)}


def _key(angle, role="fiber"):
    return (role, "family", angle)


def test_the_store_budget_counts_idle_and_lent_tables_and_never_drops_a_lent_one():
    store = WarmIntegrators(2, allowance=2500)
    held = _Tables(1000)
    assert store.borrow(_key(0.1)) is None  # first build of the family: size unknown
    store.give_back(_key(0.1), held)
    assert store.borrow(_key(0.1)) is held  # lent: no longer idle
    assert store.borrow(_key(0.2)) is None  # a build of a known family reserves its 1000 bytes
    store.give_back(_key(0.2), _Tables(1000))
    store.give_back(_key(0.1), held)
    assert len(store) == 2  # 2000 idle bytes fit 2500
    assert store.borrow(_key(0.3)) is None  # reserving 1000 more: the least recent (0.2) goes
    assert set(store._idle) == {_key(0.1)}
    store.give_back(_key(0.3), _Tables(1000))
    assert len(store) == 2
    for angle in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
        store.fit(2, allowance=1 << 40)
        store.borrow(_key(angle))
        store.give_back(_key(angle), _Tables(1))
    assert len([key for key in store._idle if key[0] == "fiber"]) == WarmIntegrators.KEYS_PER_ROLE
    assert store.borrow(_key(9.9, role="standard")) is None  # a new family of unknown size: every idle one goes
    assert len(store) == 0
    store.release(_key(9.9, role="standard"))
    assert not store._lent


def test_with_no_allowance_only_each_roles_most_recent_key_stays():
    store = WarmIntegrators(4)
    store.give_back(_key(0.1), _Tables(10))
    store.give_back(_key(0.1, role="scout"), _Tables(10))
    store.give_back(_key(0.2), _Tables(10))
    assert set(store._idle) == {_key(0.1, role="scout"), _key(0.2)}


def test_the_allowance_comes_from_the_runs_memory_plan():
    from types import SimpleNamespace

    from xrd_tools.session.policy import INTEGRATOR_RESERVE_BYTES as GIB, warm_integrator_allowance
    plan = SimpleNamespace(envelope_bytes=20 * GIB, assigned_bytes=6 * GIB)  # 14 GiB of the envelope unspent
    assert warm_integrator_allowance(plan, 4, None) == 4 * GIB  # RAM unknown (Windows today): the funded reserve
    assert warm_integrator_allowance(None, 4, 64 * GIB) == 4 * GIB  # no memory plan: the same
    assert warm_integrator_allowance(plan, 4, 64 * GIB) == 16 * GIB  # reserve plus headroom, up to a quarter of RAM
    assert warm_integrator_allowance(plan, 4, 8 * GIB) == 4 * GIB  # a quarter of 8 GiB is below the reserve


def test_gi_with_parallax_on_is_reused_exactly(promotions):
    si = {"material": "Si", "thickness": 0.00045}
    reuse = WarmIntegrators(1)
    _reduce(_scan("first", (0.2,), sensor=si), reuse=reuse)
    second = _reduce(_scan("second", (0.2,), seed=1, sensor=si), reuse=reuse)
    assert len(promotions) == 1
    assert _same(_arrays(second), _arrays(_reduce(_scan("second", (0.2,), seed=1, sensor=si))))


def test_a_build_that_fails_after_borrowing_returns_its_reservation(monkeypatch):
    reuse = WarmIntegrators(1, allowance=1 << 30)
    scan = _scan("fails", (0.2,))
    provider = _ReductionIntegratorProvider(scan=scan, plan=_plan(scan), ai=scan.integrator, fi=None,
                                            initial_incident_angle=0.2, reuse=reuse)

    def broken(*args, **kwargs):
        raise RuntimeError("promotion failed")

    monkeypatch.setattr(gid, "_fiber_from_integrator", broken)
    with pytest.raises(RuntimeError, match="promotion failed"):
        provider.fiber(0.2, None, SHAPE)
    assert not reuse._lent
    standard = _scan("standard", (0.0,))
    plan = _plan(standard, gi=False)
    provider = _ReductionIntegratorProvider(scan=standard, plan=plan, ai=standard.integrator, fi=None,
                                            initial_incident_angle=None, reuse=reuse)
    verdicts = iter((True, False))  # the accepted integrator passes, its clone falls back to the unbound path
    monkeypatch.setattr(core, "_stock_pyfai_detector_mask_semantics", lambda *_args: next(verdicts))
    provider.standard_with_run_mask(np.asarray(plan.mask, dtype=bool), SHAPE)
    assert not reuse._lent
