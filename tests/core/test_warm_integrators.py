"""A Run's series reuse the worker integrators whose pyFAI tables are built, with results identical to a fresh build.

Each series builds a new integrator and each worker deep-copies it; pyFAI's deepcopy drops the tables, so without the
Run's ``WarmIntegrators`` every worker of every series rebuilds them. Real pyFAI integrators and real sessions here.
"""

from __future__ import annotations

import threading
import time
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytest.importorskip("pyFAI")

from pyFAI.detectors import Detector, detector_factory  # noqa: E402
from pyFAI.integrator.azimuthal import AzimuthalIntegrator  # noqa: E402

import xrd_tools.reduction.core as core  # noqa: E402
from xrd_tools.core.containers import PONI  # noqa: E402
from xrd_tools.reduction import Frame, MemorySink, ReductionPlan, ReductionSession, Scan  # noqa: E402
from xrd_tools.reduction.core import (  # noqa: E402
    Integration1DPlan, Integration2DPlan, WarmIntegrators, _warm_integrator_key,
)

SHAPE = (64, 80)


def _scan(name, *, seed=0, frames=1, wavelength=1e-10, shape=SHAPE, rot1=0.01, orientation=0, chi_at_zero=False,
          flat=False):
    detector = Detector(pixel1=1e-4, pixel2=1e-4, max_shape=shape, orientation=orientation)
    if flat:
        detector.flatfield = np.full(shape, 0.9, dtype=np.float32)  # the detector's own flat field
    ai = AzimuthalIntegrator(dist=0.1, poni1=0.003, poni2=0.004, rot1=rot1, detector=detector, wavelength=wavelength)
    if chi_at_zero:
        ai.setChiDiscAtZero()
    poni = PONI(dist=0.1, poni1=0.003, poni2=0.004, rot1=rot1, rot2=0.0, rot3=0.0, wavelength=wavelength, detector="")
    rng = np.random.default_rng(seed)
    return Scan(name, [Frame(index, image=rng.poisson(50.0, shape).astype(np.float32)) for index in range(frames)],
                poni=poni, integrator=ai)


def _plan(*, shape=SHAPE, mask_row=10, npt=100, radial_range=None, extra=None):
    mask = np.zeros(shape, dtype=bool)
    mask[mask_row:mask_row + 4, :] = True
    return ReductionPlan(integration_1d=Integration1DPlan(npt=npt, error_model="poisson", radial_range=radial_range),
                         integration_2d=Integration2DPlan(npt_rad=50, npt_azim=36, error_model="poisson"), mask=mask,
                         extra=extra or {})


def _reduce(scan, plan, *, reuse=None, cores=1):
    session = ReductionSession(plan, scan, sink=MemorySink(), execution="streaming", executor=cores,
                               integrator_reuse=reuse)
    for frame in scan.frames:
        session.submit(frame)
    return session.finish()


def _arrays(result):
    frames = [result.frames[index] for index in sorted(result.frames)]
    return [(frame.result_1d.radial, frame.result_1d.intensity, frame.result_1d.sigma, frame.result_2d.radial,
             frame.result_2d.azimuthal, frame.result_2d.intensity, frame.result_2d.sigma) for frame in frames]


def _same(left, right):
    return len(left) == len(right) and all(
        (a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b, equal_nan=True))
        for one, two in zip(left, right) for a, b in zip(one, two))


@pytest.fixture
def builds(monkeypatch):
    """Count the worker integrators the sessions build (each starts without pyFAI's tables)."""
    built = []
    real = core._ReductionIntegratorProvider._new_standard

    def counting(self):
        built.append(threading.get_ident())
        return real(self)

    monkeypatch.setattr(core._ReductionIntegratorProvider, "_new_standard", counting)
    return built


def test_the_next_series_borrows_the_built_integrator_and_matches_a_fresh_reduction(builds):
    reuse = WarmIntegrators(1)
    _reduce(_scan("first", seed=0), _plan(), reuse=reuse)
    assert len(builds) == 1 and len(reuse) == 1
    second = _reduce(_scan("second", seed=1), _plan(), reuse=reuse)
    assert len(builds) == 1  # borrowed, tables already built
    assert len(reuse) == 1  # given back again
    fresh = _reduce(_scan("second", seed=1), _plan())
    assert all(item is not None for item in _arrays(fresh)[0])  # axes, intensities and sigmas all compared
    assert _same(_arrays(second), _arrays(fresh))


_CHANGES = {
    "wavelength": (dict(wavelength=1.1e-10), {}), "geometry": (dict(rot1=0.02), {}),
    "orientation": (dict(orientation=2), {}), "chi discontinuity": (dict(chi_at_zero=True), {}),
    "mask": ({}, dict(mask_row=30)), "shape": (dict(shape=(60, 80)), dict(shape=(60, 80))),
    "settings": ({}, dict(npt=120)), "range": ({}, dict(radial_range=(0.05, 0.25))),
}


@pytest.mark.parametrize("change", sorted(_CHANGES))
def test_a_series_whose_tables_would_differ_builds_its_own(builds, change):
    scan_change, plan_change = _CHANGES[change]
    reuse = WarmIntegrators(1)
    _reduce(_scan("first", seed=0), _plan(), reuse=reuse)
    second = _reduce(_scan("second", seed=1, **scan_change), _plan(**plan_change), reuse=reuse)
    assert len(builds) == 2 and len(reuse) == 1  # the first series' integrator was dropped, not kept beside it
    assert _same(_arrays(second), _arrays(_reduce(_scan("second", seed=1, **scan_change), _plan(**plan_change))))


def test_state_the_key_cannot_hold_is_never_reused(builds):
    reuse = WarmIntegrators(1)
    _reduce(_scan("first", seed=0, flat=True), _plan(), reuse=reuse)
    second = _reduce(_scan("second", seed=1, flat=True), _plan(), reuse=reuse)
    assert len(builds) == 2 and len(reuse) == 0  # the detector's own flat field: rebuilt, nothing kept
    assert _same(_arrays(second), _arrays(_reduce(_scan("second", seed=1, flat=True), _plan())))
    sensor = detector_factory("Pilatus100k", {"orientation": 3, "sensor": {"material": "Si", "thickness": 0.00045}})
    parallax = AzimuthalIntegrator(dist=0.1, poni1=0.010013, poni2=0.040007, detector=sensor, wavelength=1e-10)
    parallax.enable_parallax(True)
    # the sensor parallax correction is keyed exactly (its sensor and beam configuration), so it is reused now
    assert _warm_integrator_key("standard", parallax, None, tuple(sensor.shape),
                                _plan(shape=tuple(sensor.shape))) is not None
    plain = _scan("plain").integrator
    odd = ReductionPlan(integration_1d=Integration1DPlan(npt=100, extra={"weights": np.ones(3)}))
    assert _warm_integrator_key("standard", plain, None, SHAPE, _plan()) is not None
    assert _warm_integrator_key("standard", plain, None, SHAPE, odd) is None


def test_a_session_that_does_not_own_its_pool_never_borrows_or_gives_back(builds):
    reuse = WarmIntegrators(1)
    _reduce(_scan("first", seed=0), _plan(), reuse=reuse)
    with ThreadPoolExecutor(max_workers=2) as pool:
        second = _reduce(_scan("second", seed=1), _plan(), reuse=reuse, cores=pool)
    assert len(builds) == 2 and len(reuse) == 1  # built its own; the Run's integrator stayed in the store
    assert _same(_arrays(second), _arrays(_reduce(_scan("second", seed=1), _plan())))


def test_concurrent_workers_each_hold_their_own_integrator(monkeypatch):
    holders: dict[int, set[int]] = {}
    real = core._ReductionIntegratorProvider.standard_with_run_mask

    def recording(self, run_mask, image_shape):
        ai, bound = real(self, run_mask, image_shape)
        holders.setdefault(id(ai), set()).add(threading.get_ident())
        return ai, bound

    monkeypatch.setattr(core._ReductionIntegratorProvider, "standard_with_run_mask", recording)
    reuse = WarmIntegrators(4)
    _reduce(_scan("first", seed=0, frames=8), _plan(), reuse=reuse, cores=4)
    first = dict(holders)
    holders.clear()
    second = _reduce(_scan("second", seed=1, frames=8), _plan(), reuse=reuse, cores=4)
    # within a series no integrator serves two threads; the store keeps at most the granted workers
    assert all(len(threads) == 1 for threads in first.values())
    assert all(len(threads) == 1 for threads in holders.values())
    assert 1 <= len(reuse) <= 4
    assert set(holders) & set(first)  # the second series reused at least one built integrator
    assert _same(_arrays(second), _arrays(_reduce(_scan("second", seed=1, frames=8), _plan(), cores=4)))


class _FailingWrite(MemorySink):
    def write(self, frame, reduction):
        raise OSError("disk full")


class _FailingFinish(MemorySink):
    def finish(self, result):
        raise OSError("output vanished")


class _SlowWrite(MemorySink):
    def write(self, frame, reduction):
        time.sleep(0.5)
        super().write(frame, reduction)


def _session(scan, sink, reuse):
    session = ReductionSession(_plan(), scan, sink=sink, execution="streaming", executor=1, integrator_reuse=reuse)
    session.submit(scan.frames[0])
    return session


def test_a_failed_series_gives_nothing_back_and_a_clean_one_gives_back_once():
    reuse = WarmIntegrators(4)
    assert _session(_scan("write"), _FailingWrite(), reuse).finish(raise_on_failure=False).failed
    assert len(reuse) == 0
    with pytest.raises(OSError, match="output vanished"):
        _session(_scan("finish"), _FailingFinish(), reuse).finish()
    assert len(reuse) == 0
    slow = _session(_scan("timeout"), _SlowWrite(), reuse)
    assert slow.finish(raise_on_failure=False, join_timeout=0.05).failed  # writer timed out: never joined
    slow._writer_thread.join(5.0)
    assert len(reuse) == 0
    clean = _session(_scan("clean"), MemorySink(), reuse)
    clean.finish()
    clean.finish()
    assert len(reuse) == 1


def test_the_store_keeps_one_key_within_the_grant():
    store = WarmIntegrators(4)  # no byte allowance: one key per role
    a, b = ("standard", "a", None), ("standard", "b", None)
    first, second, third = object(), object(), object()
    for integrator in (first, second, third):
        store.give_back(a, integrator)
    store.fit(2)  # a session granted two workers
    assert len(store) == 2
    assert store.borrow(b) is None and len(store) == 0  # another key drops the idle set before anything is built
    store.give_back(b, third)
    store.give_back(a, first)  # and giving back under another key of the role drops the other key's
    assert len(store) == 1 and store.borrow(a) is first and len(store) == 0
    store.give_back(a, second)
    store.clear()
    assert len(store) == 0


@pytest.mark.parametrize("sink", [_FailingWrite, _FailingFinish, _SlowWrite])
def test_failed_finish_does_not_retain_worker_integrators(monkeypatch, sink):
    held = []
    real = core._ReductionIntegratorProvider._new_standard

    def observe(self):
        ai = real(self)
        held.append(weakref.ref(ai))
        return ai

    monkeypatch.setattr(core._ReductionIntegratorProvider, "_new_standard", observe)
    session = _session(_scan("failed"), sink(), WarmIntegrators(1))
    try:
        session.finish(raise_on_failure=False, join_timeout=0.05 if sink is _SlowWrite else 5.0)
    except OSError:
        pass
    if session._writer_thread is not None:
        session._writer_thread.join(5.0)
    gc.collect()
    assert held and all(reference() is None for reference in held)


def test_unsupported_series_releases_idle_tables_before_building(monkeypatch):
    reuse = WarmIntegrators(1)
    _reduce(_scan("first"), _plan(), reuse=reuse)
    assert len(reuse) == 1
    real = core._ReductionIntegratorProvider._new_standard

    def observe(self):
        assert len(reuse) == 0, "unsupported series retained the previous warm set"
        return real(self)

    monkeypatch.setattr(core._ReductionIntegratorProvider, "_new_standard", observe)
    result = _reduce(_scan("flat", flat=True), _plan(), reuse=reuse)
    assert not result.failed and len(reuse) == 0


def test_a_run_end_keeps_the_integrators_under_each_roles_latest_key():
    """The executor's store outlives its Runs (beamline 2026-10-23, each Run's first frame rebuilt pyFAI's tables):
    at a Run's end it keeps the idle integrators under each role's most recently returned key."""
    reuse = WarmIntegrators(4)
    older, latest, fiber = ("standard", "family", 1, None), ("standard", "family", 2, None), ("fiber", "gi", 1, 0.1)
    kept = [object() for _ in range(5)]
    for key, integrator in zip((older, older, latest, latest, fiber), kept):
        reuse.give_back(key, integrator)
    reuse.keep_for_next_run()
    assert len(reuse) == 3
    assert reuse.borrow(fiber) is kept[4] and reuse.borrow(latest) is kept[3] and reuse.borrow(latest) is kept[2]
    assert reuse.borrow(older) is None


@pytest.fixture
def started(monkeypatch):
    """The workers each session starts, which it chooses from its own exact key (Codex N-770941b2 F4)."""
    counts = []
    real = core._coerce_executor

    def recording(executor):
        counts.append(executor)
        return real(executor)

    monkeypatch.setattr(core, "_coerce_executor", recording)
    return counts


def _two_built(monkeypatch, scan, plan, reuse):
    """Reduce ``scan`` on two workers that each build an integrator (each waits until both hold one), so the store
    then holds two under the series' key."""
    barrier, waited = threading.Barrier(2, timeout=30), set()
    real = core._ReductionIntegratorProvider.standard_with_run_mask

    def both(self, run_mask, image_shape):
        bound = real(self, run_mask, image_shape)
        if threading.get_ident() not in waited:
            waited.add(threading.get_ident())
            barrier.wait()
        return bound

    with monkeypatch.context() as patch:
        patch.setattr(core._ReductionIntegratorProvider, "standard_with_run_mask", both)
        _reduce(scan, plan, reuse=reuse, cores=2)
    assert len(reuse) == 2


def test_a_short_series_of_large_tables_starts_the_workers_built_under_its_key(monkeypatch, builds, started):
    reuse = WarmIntegrators(4)
    _two_built(monkeypatch, _scan("first", seed=0, frames=2), _plan(), reuse)
    monkeypatch.setattr(core, "_LARGE_TABLE_BYTES", 1)  # these measured tables count as large
    second = _reduce(_scan("second", seed=1, frames=3), _plan(), reuse=reuse, cores=4)
    assert started == [2, 2] and len(builds) == 2 and len(reuse) == 2  # three frames on the two built, none built
    assert _same(_arrays(second), _arrays(_reduce(_scan("second", seed=1, frames=3), _plan())))


def test_a_larger_detector_after_a_cheap_one_is_priced_by_its_own_size(monkeypatch, started):
    monkeypatch.setattr(core, "_LARGE_DETECTOR_PIXELS", 10_000)  # between this detector (5120) and the next (20480)
    reuse = WarmIntegrators(4)
    _reduce(_scan("small", seed=0, frames=2), _plan(), reuse=reuse, cores=4)  # its tables measured, and cheap
    large = (128, 160)
    _reduce(_scan("large", seed=1, frames=3, shape=large), _plan(shape=large), reuse=reuse, cores=4)
    assert started == [4, 1]  # its own family is not yet measured, so its pixels decide


@pytest.mark.parametrize(("change", "expected"), (
    ("chi discontinuity", 1), ("geometry", 1), ("mask", 1), ("orientation", 1), ("range", 1), ("wavelength", 1),
    ("settings", 4), ("shape", 4),  # another family, not yet measured: this small detector's size decides
))
def test_a_changed_series_counts_no_tables_built_under_another_key(monkeypatch, started, change, expected):
    scan_change, plan_change = _CHANGES[change]
    reuse = WarmIntegrators(4)
    _two_built(monkeypatch, _scan("first", seed=0, frames=2), _plan(), reuse)
    monkeypatch.setattr(core, "_LARGE_TABLE_BYTES", 1)  # the family's measured tables count as large
    _reduce(_scan("second", seed=1, frames=3, **scan_change), _plan(**plan_change), reuse=reuse, cores=4)
    assert started == [2, expected]


def test_a_series_no_integrator_is_reused_for_starts_by_its_detector_size(monkeypatch, started):
    reuse = WarmIntegrators(4)
    _two_built(monkeypatch, _scan("first", seed=0, frames=2), _plan(), reuse)
    monkeypatch.setattr(core, "_LARGE_TABLE_BYTES", 1)
    monkeypatch.setattr(core, "_LARGE_DETECTOR_PIXELS", 1)  # this detector counts as large
    masked = _scan("masked", seed=1, frames=3)
    for frame in masked.frames:
        frame.mask = np.zeros(SHAPE, dtype=bool)  # a frame's own mask is never bound into a kept integrator
    _reduce(masked, _plan(), reuse=reuse, cores=4)
    energies = {"energy_selection": {"frame_labels": [0, 1, 2], "frame_energies_eV": [12000.0, 12100.0, 12200.0]}}
    _reduce(_scan("energy", seed=2, frames=3), _plan(extra=energies), reuse=reuse, cores=4)
    flat = _scan("flat", seed=3, frames=3, flat=True)  # the detector's own flat field: no key holds it
    _reduce(flat, _plan(), reuse=reuse, cores=4)
    assert started == [2, 1, 1, 1]


def test_a_mask_the_frames_reject_fails_them_as_before(started):
    wrong = ReductionPlan(integration_1d=Integration1DPlan(npt=100), mask=np.zeros((10, 10), dtype=bool))
    with pytest.raises(ValueError, match="does not match image shape"):
        _reduce(_scan("wrong", frames=2), wrong, reuse=WarmIntegrators(4), cores=4)
    assert started == [4]  # no key: this small detector's size decides, and the frames report the mask
