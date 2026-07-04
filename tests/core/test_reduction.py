"""Tests for the scan/frame headless reduction API."""

from __future__ import annotations

import threading
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.reduction import (
    CancelToken,
    Frame,
    GIMode,
    Integration1DPlan,
    Integration2DPlan,
    MaskSpec,
    MemorySink,
    NexusSink,
    ReductionPlan,
    Scan,
    ReductionSession,
    run_reduction,
)
import xrd_tools.reduction.core as reduction_core


def _r1d(value: float, *, unit: str = "q_A^-1") -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([value, value + 1.0]),
        sigma=None,
        unit=unit,
    )


def _r2d(value: float, *, unit: str = "q_A^-1") -> IntegrationResult2D:
    return IntegrationResult2D(
        radial=np.array([0.0, 1.0]),
        azimuthal=np.array([-1.0, 1.0]),
        intensity=np.full((2, 2), value),
        sigma=None,
        unit=unit,
    )


def test_scan_sorts_frames_and_synthesizes_metadata() -> None:
    scan = Scan(
        name="sample_scan1",
        frames=[
            Frame(index=2, image=np.zeros((2, 2)), metadata={"I0": 20}),
            Frame(index=1, image=np.zeros((2, 2)), metadata={"I0": 10}),
        ],
        energy=12.398,
        wavelength=1.0,
        motors={"tth": [0.0, 1.0]},
        sample_name="sample",
    )

    assert [f.index for f in scan.frames] == [1, 2]
    meta = scan.to_metadata()
    assert meta is not None
    assert meta.scan_id == "sample_scan1"
    np.testing.assert_allclose(meta.counters["i0"], [10, 20])
    np.testing.assert_allclose(meta.angles["tth"], [0.0, 1.0])


def test_scan_frame_source_iterates_bounded_chunks() -> None:
    from xrd_tools.reduction import FrameSource

    scan = Scan(
        "source",
        [Frame(2, image=np.full((2, 2), 2.0)),
         Frame(0, image=np.zeros((2, 2))),
         Frame(1, image=np.ones((2, 2)))],
    )
    assert isinstance(scan, FrameSource)
    chunks = list(scan.iter_chunks(2))
    assert [indices for _, indices in chunks] == [[0, 1], [2]]
    np.testing.assert_array_equal(chunks[0][0][:, 0, 0], [0.0, 1.0])


def test_scan_frame_source_clears_images_loaded_by_chunks() -> None:
    calls: list[int] = []

    def _loader(frame: Frame) -> np.ndarray:
        calls.append(frame.index)
        return np.full((2, 2), frame.index)

    preloaded = Frame(0, image=np.zeros((2, 2)))
    lazy = [Frame(1, loader=_loader), Frame(2, loader=_loader)]
    scan = Scan("bounded", [preloaded, *lazy])

    chunks = list(scan.iter_chunks(1))
    assert [indices for _, indices in chunks] == [[0], [1], [2]]
    assert calls == [1, 2]
    assert preloaded.image is not None
    assert lazy[0].image is None
    assert lazy[1].image is None


def test_frame_loads_with_custom_loader_once() -> None:
    calls: list[int] = []

    def _loader(frame: Frame) -> np.ndarray:
        calls.append(frame.index)
        return np.full((2, 2), frame.index)

    frame = Frame(index=3, loader=_loader)

    np.testing.assert_array_equal(frame.load_image(), np.full((2, 2), 3))
    np.testing.assert_array_equal(frame.load_image(), np.full((2, 2), 3))
    assert calls == [3]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"integration_1d": None, "integration_2d": None},
            "integration_1d",
        ),
    ],
)
def test_reduction_plan_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ReductionPlan(**kwargs)


def test_run_reduction_chunk_size_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    # chunk_size is a run_reduction kwarg now, not a plan field.
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kwargs: _r1d(1.0))
    scan = Scan("scan", [Frame(0, image=np.ones((2, 2)))], integrator=object())
    with pytest.raises(ValueError, match="chunk_size"):
        run_reduction(ReductionPlan(), scan, chunk_size=0)


def test_run_reduction_uses_frame_source_iter_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reduction_core,
        "integrate_1d",
        lambda image, ai, **kwargs: _r1d(float(image[0, 0])),
    )

    class ChunkSource:
        name = "chunked"
        frame_indices = [10, 11, 12]
        integrator = object()
        output_path = None

        def __init__(self):
            self.loaded: list[int] = []
            self.chunks: list[int] = []

        def load_frame(self, index):
            self.loaded.append(int(index))
            raise AssertionError("run_reduction should use iter_chunks pixels")

        def frame_for(self, index):
            return Frame(
                int(index),
                loader=lambda frame: self.load_frame(frame.index),
            )

        def iter_chunks(self, chunk_size):
            self.chunks.append(int(chunk_size))
            yield np.stack([np.full((2, 2), 10.0), np.full((2, 2), 11.0)]), [10, 11]
            yield np.stack([np.full((2, 2), 12.0)]), [12]

        def to_scan(self, **kwargs):
            return Scan(
                self.name,
                [self.frame_for(i) for i in self.frame_indices],
                integrator=self.integrator,
            )

    source = ChunkSource()
    result = run_reduction(ReductionPlan(), source, chunk_size=2)

    assert source.chunks == [2]
    assert source.loaded == []
    assert result.n_processed == 3
    assert [result.frames[i].result_1d.intensity[0] for i in (10, 11, 12)] == [
        10.0,
        11.0,
        12.0,
    ]


def test_run_reduction_executor_matches_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reduction_core,
        "integrate_1d",
        lambda image, ai, **kwargs: _r1d(float(np.sum(image))),
    )
    frames_serial = [Frame(i, image=np.full((2, 2), i, dtype=float)) for i in range(5)]
    frames_parallel = [Frame(i, image=np.full((2, 2), i, dtype=float)) for i in range(5)]
    plan = ReductionPlan(integration_2d=None)

    serial = run_reduction(plan, Scan("serial", frames_serial, integrator=object()))

    lock = threading.Lock()
    barrier = threading.Barrier(2)
    worker_thread_ids: set[int] = set()
    call_count = 0

    def fake_parallel_1d(image, ai, **kwargs):
        nonlocal call_count
        with lock:
            worker_thread_ids.add(threading.get_ident())
            call_number = call_count
            call_count += 1
        if call_number < 2:
            barrier.wait(timeout=5)
        return _r1d(float(np.sum(image)))

    monkeypatch.setattr(reduction_core, "integrate_1d", fake_parallel_1d)
    parallel = run_reduction(
        plan,
        Scan("parallel", frames_parallel, integrator=object()),
        executor=2,
        chunk_size=5,
    )

    assert list(serial.frames) == list(parallel.frames)
    for idx in serial.frames:
        np.testing.assert_allclose(
            parallel.frames[idx].result_1d.intensity,
            serial.frames[idx].result_1d.intensity,
        )
    assert len(worker_thread_ids) >= 2


def test_integration_plan_validation() -> None:
    with pytest.raises(ValueError, match="npt"):
        Integration1DPlan(npt=0)
    with pytest.raises(ValueError, match="npt_rad"):
        Integration2DPlan(npt_rad=0)


def test_gi_mode_is_a_sum_type() -> None:
    # GIMode bundles the GI-only parameters into one optional field on
    # ReductionPlan; the type system rules out "gi=False, incident_angle=2.5".
    gi = GIMode(incident_angle=2.5)
    assert gi.incident_angle == 2.5
    assert gi.tilt_angle == 0.0
    plan = ReductionPlan(gi=gi)
    assert plan.gi is gi
    # GIMode is frozen — can't mutate after construction
    with pytest.raises(Exception):
        gi.incident_angle = 5.0  # type: ignore[misc]


def test_run_reduction_1d_2d_mask_threshold_norm_and_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_1d(image, ai, **kwargs):
        calls.append({"kind": "1d", "image": image.copy(), **kwargs})
        return _r1d(float(np.nansum(image)), unit=kwargs["unit"])

    def fake_2d(image, ai, **kwargs):
        calls.append({"kind": "2d", "image": image.copy(), **kwargs})
        return _r2d(float(np.nanmean(image)), unit=kwargs["unit"])

    monkeypatch.setattr(reduction_core, "integrate_1d", fake_1d)
    monkeypatch.setattr(reduction_core, "integrate_2d", fake_2d)

    events = []
    scan = Scan(
        name="scan",
        frames=[
            Frame(
                index=0,
                image=np.array([[1.0, 2.0], [100.0, -5.0]]),
                background=np.ones((2, 2)),
                mask=np.array([[False, True], [False, False]]),
            )
        ],
        integrator=object(),
    )
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(
            npt=5,
            unit="2th_deg",
            method="BBox",
        ),
        integration_2d=Integration2DPlan(
            npt_rad=6,
            npt_azim=7,
            unit="q_A^-1",
            method="csr",
        ),
        mask=np.array([[False, False], [True, False]]),
        threshold_min=0.0,
        threshold_max=50.0,
    )
    scan.frames[0].normalization_factor = 9.0

    result = run_reduction(plan, scan, chunk_size=1, progress_cb=events.append)

    assert result.n_processed == 1
    assert result.frames[0].result_1d is not None
    assert result.frames[0].result_2d is not None
    assert [e.stage for e in events] == [
        "start", "chunk", "load", "integrate", "write", "finish"
    ]
    assert calls[0]["npt"] == 5
    assert calls[0]["method"] == "BBox"
    assert calls[0]["normalization_factor"] == 9.0
    np.testing.assert_array_equal(
        calls[0]["mask"],
        np.array([[False, True], [True, False]]),
    )
    assert not np.isnan(calls[0]["image"][0, 1])  # mask passed separately
    assert calls[0]["image"][0, 0] == 0.0  # background subtracted
    assert np.isnan(calls[0]["image"][1, 0])  # threshold_max
    assert np.isnan(calls[0]["image"][1, 1])  # threshold_min
    assert calls[1]["npt_rad"] == 6
    assert calls[1]["npt_azim"] == 7


def test_run_reduction_resolves_flat_mask_spec_after_image_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_1d(image, ai, **kwargs):
        calls.append(kwargs)
        return _r1d(1.0)

    monkeypatch.setattr(reduction_core, "integrate_1d", fake_1d)

    result = run_reduction(
        ReductionPlan(mask=MaskSpec(np.array([0, 3]))),
        Scan("scan", [Frame(0, image=np.ones((2, 2)))], integrator=object()),
    )

    assert result.n_processed == 1
    np.testing.assert_array_equal(
        calls[0]["mask"],
        np.array([[True, False], [False, True]]),
    )


def test_run_reduction_uses_independent_1d_2d_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_1d(image, ai, **kwargs):
        calls.append({"kind": "1d", **kwargs})
        return _r1d(1.0)

    def fake_2d(image, ai, **kwargs):
        calls.append({"kind": "2d", **kwargs})
        return IntegrationResult2D(
            radial=np.array([0.0, 1.0]),
            azimuthal=np.array([-5.0, 5.0]),
            intensity=np.ones((2, 2)),
            unit=kwargs["unit"],
        )

    monkeypatch.setattr(reduction_core, "integrate_1d", fake_1d)
    monkeypatch.setattr(reduction_core, "integrate_2d", fake_2d)

    result = run_reduction(
        ReductionPlan(
            integration_1d=Integration1DPlan(
                npt=11,
                unit="q_A^-1",
                method="csr",
                radial_range=(1.0, 2.0),
                azimuth_range=(-30.0, 30.0),
                monitor_key="i0",
            ),
            integration_2d=Integration2DPlan(
                npt_rad=12,
                npt_azim=13,
                unit="2th_deg",
                method="BBox",
                radial_range=(10.0, 20.0),
                azimuth_range=(80.0, 100.0),
                azimuth_offset=90.0,
                monitor_key="i1",
            ),
        ),
        Scan(
            "scan",
            [Frame(0, image=np.ones((2, 2)), metadata={"I0": 25.0, "i1": 50.0})],
            integrator=object(),
        ),
    )

    assert calls[0]["normalization_factor"] == 25.0
    assert calls[0]["npt"] == 11
    assert calls[0]["unit"] == "q_A^-1"
    assert calls[0]["radial_range"] == (1.0, 2.0)
    assert calls[0]["azimuth_range"] == (-30.0, 30.0)
    assert calls[1]["normalization_factor"] == 50.0
    assert calls[1]["npt_rad"] == 12
    assert calls[1]["npt_azim"] == 13
    assert calls[1]["unit"] == "2th_deg"
    assert calls[1]["radial_range"] == (10.0, 20.0)
    assert calls[1]["azimuth_range"] == (-10.0, 10.0)
    np.testing.assert_allclose(result.frames[0].result_2d.azimuthal, [85.0, 95.0])


def test_run_reduction_dispatches_non_gi_chi_to_integrate_radial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fail_integrate_1d(*args, **kwargs):
        raise AssertionError("chi_deg must use integrate_radial, not integrate_1d")

    def fake_integrate_radial(image, ai, **kwargs):
        calls.append(dict(kwargs))
        npt = int(kwargs["npt"])
        return IntegrationResult1D(
            radial=np.linspace(-180.0, 180.0, npt),
            intensity=np.arange(npt, dtype=float),
            sigma=None,
            unit="chi_deg",
        )

    monkeypatch.setattr(reduction_core, "integrate_1d", fail_integrate_1d)
    monkeypatch.setattr(reduction_core, "integrate_radial", fake_integrate_radial)

    plan = ReductionPlan(
        integration_1d=Integration1DPlan(
            npt=9,
            unit="chi_deg",
            method="csr",
            radial_range=(0.5, 4.0),
            azimuth_range=(-90.0, 90.0),
            monitor_key="i0",
            error_model="poisson",
            polarization_factor=0.99,
            extra={"variance": "drop-me", "correctSolidAngle": False},
        ),
        integration_2d=None,
    )
    scan = Scan(
        "chi",
        [Frame(0, image=np.ones((4, 4)), metadata={"i0": 2.0})],
        integrator=object(),
    )

    result = run_reduction(plan, scan)

    assert result.frames[0].result_1d.unit == "chi_deg"
    assert len(calls) == 1
    call = calls[0]
    assert call["npt"] == 9
    assert call["radial_unit"] == "q_A^-1"
    assert call["radial_range"] == (0.5, 4.0)
    assert call["method"] == "csr"
    assert call["azimuth_range"] == (-90.0, 90.0)
    assert call["normalization_factor"] == 2.0
    assert call["polarization_factor"] == 0.99
    assert call["correctSolidAngle"] is False
    assert "error_model" not in call
    assert "variance" not in call


@pytest.mark.slow
def test_run_reduction_non_gi_chi_matches_integrate_radial(
    ai_fixture,
    synthetic_image,
) -> None:
    from xrd_tools.integrate.single import integrate_radial

    plan = ReductionPlan(
        integration_1d=Integration1DPlan(
            npt=72,
            unit="chi_deg",
            method="csr",
            radial_range=(0.5, 5.0),
            extra={"correctSolidAngle": False},
        ),
        integration_2d=None,
    )
    scan = Scan(
        "chi",
        [Frame(0, image=synthetic_image)],
        integrator=ai_fixture,
    )

    result = run_reduction(plan, scan)
    expected = integrate_radial(
        synthetic_image,
        ai_fixture,
        npt=72,
        radial_unit="q_A^-1",
        method="csr",
        radial_range=(0.5, 5.0),
        correctSolidAngle=False,
    )

    actual = result.frames[0].result_1d
    assert actual.unit == "chi_deg"
    np.testing.assert_allclose(actual.radial, expected.radial, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        actual.intensity,
        expected.intensity,
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True,
    )


def test_non_gi_chi_reduction_reapplies_chi_offset_s4(ai_fixture, synthetic_image):
    """S-4: a Mode-A (chi_deg) 1D reduction re-adds chi_offset to the OUTPUT chi
    axis (mirroring Integration2DPlan.azimuth_offset), so the written 1D chi axis
    matches the 2D cake chi instead of staying in the raw pyFAI frame.  The
    INTENSITIES are unchanged (only the axis is relabeled).  1D<->2D CONSISTENCY
    is verified here; the ABSOLUTE frame is the maintainer's real-data validation
    against the team chi reference (the ship gate)."""
    from xrd_tools.integrate.single import integrate_radial

    OFFSET = 90.0
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(
            npt=72, unit="chi_deg", method="csr", radial_range=(0.5, 5.0),
            azimuth_offset=OFFSET, extra={"correctSolidAngle": False}),
        integration_2d=None,
    )
    scan = Scan("chi", [Frame(0, image=synthetic_image)], integrator=ai_fixture)
    result = run_reduction(plan, scan)

    raw = integrate_radial(  # the raw pyFAI chi frame (no offset)
        synthetic_image, ai_fixture, npt=72, radial_unit="q_A^-1", method="csr",
        radial_range=(0.5, 5.0), correctSolidAngle=False)

    actual = result.frames[0].result_1d
    assert actual.unit == "chi_deg"
    np.testing.assert_allclose(actual.radial, raw.radial + OFFSET,
                               rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(actual.intensity, raw.intensity,
                               rtol=1e-6, atol=1e-8, equal_nan=True)


def test_non_gi_chi_explicit_partial_range_shifts_input_s4(monkeypatch):
    # S-4 (INPUT half, the mirror of the 2D test above): an explicit PARTIAL chi
    # range is shifted by -azimuth_offset into the raw pyFAI frame BEFORE
    # integrating -- exactly like the 2D (80,100)->(-10,10) -- and re-added at
    # output.  Auto/full-domain masked this (no shift); passing the raw range
    # straight through integrated 90deg off the 2D for the default offset.
    captured: dict = {}

    def fake_radial(image, ai, **kwargs):
        captured.update(kwargs)
        return IntegrationResult1D(
            radial=np.array([-10.0, 10.0]),      # raw pyFAI chi frame
            intensity=np.array([1.0, 2.0]), sigma=None, unit="chi_deg")

    monkeypatch.setattr(reduction_core, "integrate_radial", fake_radial)
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(
            npt=13, unit="chi_deg", method="csr", radial_range=(0.5, 5.0),
            azimuth_range=(80.0, 100.0), azimuth_offset=90.0),
        integration_2d=None)
    result = run_reduction(
        plan, Scan("chi", [Frame(0, image=np.ones((4, 4)))], integrator=object()))

    # INPUT shifted by -90 -> integrates the SAME raw bins as the 2D
    assert captured["azimuth_range"] == (-10.0, 10.0)
    # OUTPUT re-labeled +90 (back to the panel/display frame the user asked for)
    np.testing.assert_allclose(result.frames[0].result_1d.radial, [80.0, 100.0])


def test_chunked_error_clear_waits_for_running_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D6: a sink.write failure must not leave a racing tail worker's raw
    pinned.  cancel() cannot stop a future that is already RUNNING, and that
    worker assigns ``frame.image`` (via ``load_image``) — with a naive
    immediate clear in the error handler, the assignment lands AFTER the
    clear and the raw stays pinned until session close.  The clear must be
    ordered after the running tail has resolved."""
    import contextlib
    import time as _time

    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _r1d(float(np.sum(image))))

    tail_started = threading.Event()
    write_failed = threading.Event()

    def _blocking_loader(frame: Frame) -> np.ndarray:
        # Runs in the WORKER inside load_image, immediately before the
        # ``frame.image = ...`` assignment: park until the error handler is
        # running so the re-pin lands after a naive clear.
        tail_started.set()
        write_failed.wait(5)
        _time.sleep(0.1)
        return np.ones((1, 1))

    class _FailingSink:
        def begin(self, scan, plan):
            pass

        def write(self, frame, reduction):
            tail_started.wait(5)           # the tail worker is mid-load
            write_failed.set()
            raise RuntimeError("boom")

        def finish(self, result):
            pass

    frames = [Frame(index=0, image=np.zeros((1, 1))),
              Frame(index=1, loader=_blocking_loader)]
    session = ReductionSession(
        ReductionPlan(integration_2d=None),
        Scan("d6", frames, integrator=object()),
        sink=_FailingSink(), executor=2, chunk_size=2,
        clear_frame_images=True,
    )
    try:
        with pytest.raises(RuntimeError, match="boom"):
            session.process(frames)
    finally:
        with contextlib.suppress(BaseException):
            session.finish()
    # finish() shut the owned pool down, so the tail assignment (if any)
    # has happened by now.  Ordered clear => nothing stays pinned.
    for fr in frames:
        assert fr.image is None
        assert fr.background is None


def test_run_reduction_cancellation_and_clear_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = CancelToken()
    seen: list[int] = []

    def fake_1d(image, ai, **kwargs):
        seen.append(int(image[0, 0]))
        token.cancel()
        return _r1d(float(image[0, 0]))

    monkeypatch.setattr(reduction_core, "integrate_1d", fake_1d)
    scan = Scan(
        name="scan",
        frames=[
            Frame(index=0, image=np.zeros((1, 1))),
            Frame(index=1, image=np.ones((1, 1))),
        ],
        integrator=object(),
    )

    result = run_reduction(
        ReductionPlan(),
        scan,
        chunk_size=2,
        clear_frame_images=True,
        cancel_token=token,
    )

    assert result.cancelled
    assert result.n_processed == 1
    assert seen == [0]
    assert scan.frames[0].image is None
    assert scan.frames[1].image is not None


def test_reduction_session_reuses_thread_integrators_across_process_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_thread_ids: list[int] = []
    integrate_lock = threading.Lock()
    integrate_barrier = threading.Barrier(2)
    integrate_calls = 0

    def fake_poni_to_integrator(poni):
        build_thread_ids.append(threading.get_ident())
        return object()

    def fake_1d(image, ai, **kwargs):
        nonlocal integrate_calls
        with integrate_lock:
            call_number = integrate_calls
            integrate_calls += 1
        if call_number < 2:
            integrate_barrier.wait(timeout=5)
        return _r1d(float(np.sum(image)))

    monkeypatch.setattr(reduction_core, "poni_to_integrator", fake_poni_to_integrator)
    monkeypatch.setattr(reduction_core, "integrate_1d", fake_1d)
    plan = ReductionPlan(integration_2d=None)
    poni = object()
    frames = [Frame(i, image=np.full((2, 2), i, dtype=float)) for i in range(4)]
    # Match the GUI/live path: the session opens on the first chunk, then later
    # chunks arrive as explicit Frame objects without rebuilding resources.
    scan = Scan(
        "scan",
        [Frame(0, image=np.zeros((2, 2))), Frame(1, image=np.ones((2, 2)))],
        poni=poni,
    )

    with ReductionSession(plan, scan, executor=2, chunk_size=2) as session:
        session.process(frames[:2])
        session.process(frames[2:])
        result = session.finish()

    assert result.n_processed == 4
    assert session.scan.frame_indices == [0, 1, 2, 3]
    assert session.integrator_provider_builds == 1
    assert len(build_thread_ids) == 2
    assert len(set(build_thread_ids)) == 2


def test_reduction_session_parallel_shares_plan_mask_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parallel batch reuses the persistent plan-mask cache (PERF-1).

    Before the fix the worker submit passed a fresh ``{}`` per frame, so the
    plan's bool mask was re-expanded once per frame; now the session's shared
    cache is passed, so the expansion happens ~once per detector shape (at most
    once per concurrent worker), not once per frame.
    """

    expansions: list[str] = []
    real_as_bool = reduction_core._as_bool_mask
    lock = threading.Lock()

    def counting_as_bool(mask, name, *, image_shape=None):
        if name == "ReductionPlan.mask":
            with lock:
                expansions.append(name)
        return real_as_bool(mask, name, image_shape=image_shape)

    monkeypatch.setattr(reduction_core, "_as_bool_mask", counting_as_bool)
    monkeypatch.setattr(
        reduction_core, "integrate_1d", lambda image, ai, **kwargs: _r1d(1.0)
    )

    n_frames, n_workers = 12, 2
    plan = ReductionPlan(
        integration_2d=None,
        mask=MaskSpec(np.zeros(4, dtype=int)),  # flat mask for a 2x2 image
    )
    frames = [Frame(i, image=np.ones((2, 2))) for i in range(n_frames)]
    scan = Scan("scan", [Frame(0, image=np.ones((2, 2)))], integrator=object())

    with ReductionSession(
        plan, scan, executor=n_workers, chunk_size=n_frames
    ) as session:
        session.process(frames)
        result = session.finish()

    assert result.n_processed == n_frames
    # Expanded at most once per concurrent worker -- crucially not once per
    # frame (12), which was the pre-fix cost.
    assert 0 < len(expansions) <= n_workers


def test_reduction_session_parallel_shares_frame_mask_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expansions: list[str] = []
    real_as_bool = reduction_core._as_bool_mask
    lock = threading.Lock()

    def counting_as_bool(mask, name, *, image_shape=None):
        if name == "Frame.mask":
            with lock:
                expansions.append(name)
        return real_as_bool(mask, name, image_shape=image_shape)

    monkeypatch.setattr(reduction_core, "_as_bool_mask", counting_as_bool)
    monkeypatch.setattr(
        reduction_core, "integrate_1d", lambda image, ai, **kwargs: _r1d(1.0)
    )

    n_frames, n_workers = 12, 2
    shared_mask = np.array([0, 3], dtype=np.int64)
    frames = [
        Frame(i, image=np.ones((2, 2)), mask=MaskSpec(shared_mask))
        for i in range(n_frames)
    ]
    scan = Scan("scan", [Frame(0, image=np.ones((2, 2)))], integrator=object())

    with ReductionSession(
        ReductionPlan(integration_2d=None),
        scan,
        executor=n_workers,
        chunk_size=n_frames,
    ) as session:
        session.process(frames)
        result = session.finish()

    assert result.n_processed == n_frames
    assert 0 < len(expansions) <= n_workers


def test_reduction_session_replace_refed_index_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-feeding an already-processed index is a replace, not a new frame.

    Reintegrate / replace re-feeds must not double-count progress
    (``n_processed`` stays at the number of distinct indices) and must not
    double-``write`` the sink -- the re-feed routes to the sink's ``replace``
    hook when it has one.
    """

    monkeypatch.setattr(
        reduction_core,
        "integrate_1d",
        lambda image, ai, **kwargs: _r1d(float(np.sum(image))),
    )

    class RecordingSink:
        def __init__(self) -> None:
            self.writes: list[int] = []
            self.replaces: list[int] = []
            self.frames: dict[int, object] = {}

        def begin(self, scan, plan) -> None:
            self.writes.clear()
            self.replaces.clear()
            self.frames.clear()

        def write(self, frame, reduction) -> None:
            self.writes.append(int(frame.index))
            self.frames[int(frame.index)] = reduction

        def replace(self, frame, reduction) -> None:
            self.replaces.append(int(frame.index))
            self.frames[int(frame.index)] = reduction

        def finish(self, result) -> None:
            return None

    plan = ReductionPlan(integration_2d=None)
    sink = RecordingSink()
    scan = Scan("scan", [Frame(0, image=np.zeros((2, 2)))], integrator=object())

    with ReductionSession(plan, scan, sink=sink) as session:
        session.process([Frame(0, image=np.zeros((2, 2)))])
        session.process([Frame(1, image=np.ones((2, 2)))])
        # Re-feed index 0 (reintegrate / replace) with a different image.
        session.process([Frame(0, image=np.full((2, 2), 5.0))])
        result = session.finish()

    # Progress counts distinct indices {0, 1}, not the three feeds.
    assert result.n_processed == 2
    assert session.scan.frame_indices == [0, 1]
    # One logical first-write per index; the re-feed went to replace.
    assert sorted(sink.writes) == [0, 1]
    assert sink.replaces == [0]
    # The replaced product reflects the latest re-fed image (sum 4*5 = 20).
    np.testing.assert_allclose(session.frames[0].result_1d.intensity[0], 20.0)


def test_memory_sink_can_be_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kwargs: _r1d(1.0))
    sink = MemorySink()
    result = run_reduction(
        ReductionPlan(),
        Scan("scan", [Frame(0, image=np.ones((2, 2)))], integrator=object()),
        sink,
    )

    assert result.frames[0] is sink.frames[0]


def test_nexus_sink_writes_frame_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kwargs: _r1d(3.0))
    out = tmp_path / "scan.nxs"
    scan = Scan(
        "scan",
        [Frame(0, image=np.ones((2, 2)))],
        integrator=object(),
        energy=12.398,
        wavelength=1.0,
    )

    result = run_reduction(ReductionPlan(), scan, NexusSink(out, overwrite=True))

    assert result.output_path == out
    with h5py.File(out, "r") as h5:
        # Stacked read_scan-compatible layout: (n_frames, n_q).
        assert "entry/integrated_1d/intensity" in h5
        assert list(h5["entry/integrated_1d/frame_index"][()]) == [0]
        np.testing.assert_allclose(
            h5["entry/integrated_1d/intensity"][0],
            [3.0, 4.0],
        )


def test_nexus_sink_persists_non_gi_chi_1d_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_integrate_radial(image, ai, **kwargs):
        npt = int(kwargs["npt"])
        return IntegrationResult1D(
            radial=np.linspace(-180.0, 180.0, npt),
            intensity=np.arange(npt, dtype=float),
            sigma=None,
            unit="chi_deg",
        )

    monkeypatch.setattr(reduction_core, "integrate_radial", fake_integrate_radial)
    out = tmp_path / "chi_scan.nxs"
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(npt=5, unit="chi_deg", radial_range=(1.0, 2.0)),
        integration_2d=None,
    )
    scan = Scan(
        "chi",
        [Frame(0, image=np.ones((4, 4)))],
        integrator=object(),
    )

    run_reduction(plan, scan, NexusSink(out, overwrite=True))

    with h5py.File(out, "r") as h5:
        g = h5["entry/integrated_1d"]
        assert g.attrs["axis_kind"] == "azimuthal"
        assert g["q"].attrs["units"] == "chi_deg"
        np.testing.assert_allclose(g["q"][()], np.linspace(-180.0, 180.0, 5))


def test_nexus_sink_atomic_overwrite_preserves_target_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reduction_core,
        "integrate_1d",
        lambda image, ai, **kwargs: _r1d(3.0),
    )

    def fail_write(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(reduction_core, "write_nexus_frame", fail_write)
    out = tmp_path / "scan.nxs"
    original = b"old complete file"
    out.write_bytes(original)

    with pytest.raises(RuntimeError, match="simulated write failure"):
        run_reduction(
            ReductionPlan(),
            Scan("scan", [Frame(0, image=np.ones((2, 2)))], integrator=object()),
            NexusSink(out, overwrite=True),
        )

    assert out.read_bytes() == original
    assert not list(tmp_path.glob(".scan.nxs.*.tmp"))


def test_reduction_validation_for_shapes_and_duplicate_frames() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Scan(
            "scan",
            [Frame(0, image=np.ones((2, 2))), Frame(0, image=np.ones((2, 2)))],
        )

    scan = Scan("scan", [Frame(0, image=np.ones((2, 2)))], integrator=object())
    with pytest.raises(ValueError, match="mask shape"):
        run_reduction(ReductionPlan(mask=np.ones((3, 3), dtype=bool)), scan)

    scan = Scan(
        "scan",
        [Frame(0, image=np.ones((2, 2)), background=np.ones((3, 3)))],
        integrator=object(),
    )
    with pytest.raises(ValueError, match="background shape"):
        run_reduction(ReductionPlan(), scan)

    scan = Scan("scan", [Frame(0)], integrator=object())
    with pytest.raises(ValueError, match="no image"):
        run_reduction(ReductionPlan(), scan)


def test_zero_scalar_background_is_noop_without_allocation() -> None:
    image = np.ones((3, 3), dtype=np.uint16)

    assert reduction_core._subtract_background(image, 0.0) is image


def test_nexus_sink_flush_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kwargs: _r1d(float(image[0, 0])))
    calls = []

    class FakeH5:
        def __init__(self):
            self._h5 = h5py.File(tmp_path / "flush_policy_fake.h5", "w")

        def require_group(self, *args, **kwargs):
            return self._h5.require_group(*args, **kwargs)

        def flush(self):
            calls.append("flush")
            self._h5.flush()

        def close(self):
            calls.append("close")
            self._h5.close()

    monkeypatch.setattr(reduction_core, "open_nexus_writer", lambda *a, **k: FakeH5())
    monkeypatch.setattr(reduction_core, "write_nexus_frame", lambda *a, **k: None)

    result = run_reduction(
        ReductionPlan(),
        Scan(
            "scan",
            [
                Frame(0, image=np.zeros((2, 2))),
                Frame(1, image=np.ones((2, 2))),
                Frame(2, image=np.full((2, 2), 2.0)),
            ],
            integrator=object(),
            energy=12.398,
            wavelength=1.0,
        ),
        # complete_record=False: this test is about flush cadence, not the
        # per-frame record.
        NexusSink("scan.nxs", overwrite=True, flush_every=2, atomic=False,
                  complete_record=False),
    )

    assert result.n_processed == 3
    assert calls == ["flush", "flush", "close"]  # frame 2, finish, close

    with pytest.raises(ValueError, match="flush_every"):
        NexusSink("scan.nxs", flush_every=0).begin(
            Scan("scan", [Frame(0, image=np.ones((2, 2)))], integrator=object()),
            ReductionPlan(),
        )


def test_existing_notebook_import_surface_still_imports() -> None:
    from xrd_tools.analysis.fitting import FitConfig, PhaseFitter, fit_peaks
    from xrd_tools.analysis.strain import sin2psi_analysis
    from xrd_tools.integrate import (
        create_multigeometry_integrators,
        integrate_1d,
        integrate_2d,
        stitch_1d,
        stitch_2d,
    )

    assert integrate_1d
    assert integrate_2d
    assert create_multigeometry_integrators
    assert stitch_1d
    assert stitch_2d
    assert PhaseFitter
    assert fit_peaks
    assert FitConfig
    assert sin2psi_analysis


# ---------------------------------------------------------------------------
# scan_data persistence (#nexussink_scan_data — core provenance)
# ---------------------------------------------------------------------------

def test_scan_to_scan_data_assembles_per_frame_table() -> None:
    import math

    import pandas as pd

    scan = Scan(
        name="conditions",
        frames=[
            # frame 0 has tag (string) but no stress; frame 1 has stress but no tag
            Frame(index=0, image=np.zeros((2, 2)),
                  metadata={"temperature": 300.0, "tag": "a"}),
            Frame(index=1, image=np.zeros((2, 2)),
                  metadata={"temperature": 310.0, "stress": 5.0}),
        ],
        motors={"th": np.array([0.1, 0.2])},
    )
    df = scan.to_scan_data()

    assert list(df.index) == [0, 1]
    assert {"temperature", "tag", "stress", "th"} <= set(df.columns)
    # union of frame metadata, aligned per frame
    assert df.loc[0, "temperature"] == 300.0
    assert df.loc[1, "temperature"] == 310.0
    # per-frame motor array folded in
    assert df.loc[0, "th"] == 0.1 and df.loc[1, "th"] == 0.2
    # keys missing on a frame → NaN/None
    assert pd.isna(df.loc[0, "stress"])
    assert pd.isna(df.loc[1, "tag"])
    # column order is metadata-first-seen then motors
    assert list(df.columns)[:3] == ["temperature", "tag", "stress"]


def test_scan_to_scan_data_empty_scan() -> None:
    df = Scan(name="empty", frames=[]).to_scan_data()
    assert len(df) == 0
    assert list(df.columns) == []


def test_nexussink_persists_scan_data_roundtrip(tmp_path: Path) -> None:
    from xrd_tools.io.read import get_metadata

    path = tmp_path / "scan_conditions.nxs"
    scan = Scan(
        name="ramp",
        frames=[
            Frame(index=1, image=np.zeros((2, 2)),
                  metadata={"temperature": 300.0, "stress": 2.0, "th": 0.15}),
            Frame(index=2, image=np.zeros((2, 2)),
                  metadata={"temperature": 305.0, "stress": 4.0, "th": 0.25}),
            Frame(index=3, image=np.zeros((2, 2)),
                  metadata={"temperature": 310.0, "stress": 6.0, "th": 0.35}),
        ],
    )
    sink = NexusSink(path)
    sink.begin(scan, ReductionPlan())
    for frame in scan.frames:
        sink.write(
            frame,
            reduction_core.FrameReduction(
                frame_index=frame.index, result_1d=_r1d(float(frame.index)),
            ),
        )
    sink.finish(reduction_core.ReductionResult("ramp", {}, len(scan)))

    meta = get_metadata(path)
    sd = meta["scan_data"]
    assert {"temperature", "stress", "th"} <= set(sd)
    # per-frame condition columns round-trip, aligned to frame indices 1,2,3
    np.testing.assert_allclose(sd["temperature"], [300.0, 305.0, 310.0])
    np.testing.assert_allclose(sd["stress"], [2.0, 4.0, 6.0])
    np.testing.assert_allclose(sd["th"], [0.15, 0.25, 0.35])


def test_reduction_saturation_mask_gated_by_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3-C: plan.mask_saturation=True masks a saturated module (fraction-guarded);
    default (False) leaves the headless mask unchanged.  A frame with ONLY sparse
    ceiling pixels (genuine Bragg saturation) is NOT masked."""
    calls: list[dict] = []
    monkeypatch.setattr(
        reduction_core, "integrate_1d",
        lambda image, ai, **kw: (calls.append(kw),
                                 _r1d(1.0, unit=kw.get("unit", "q_A^-1")))[1])

    def _run(img, mask_saturation):
        calls.clear()
        run_reduction(
            ReductionPlan(
                integration_1d=Integration1DPlan(npt=5, unit="q_A^-1", method="BBox"),
                mask_saturation=mask_saturation),
            Scan(name="sat", frames=[Frame(index=0, image=img.copy())],
                 integrator=object()),
            chunk_size=1,
        )
        return calls[0]["mask"]

    img = np.zeros((100, 100), dtype=np.uint16)
    img[:5, :] = 65535          # dead/overflowed module: 500/10000 = 5% > 1e-4

    assert _run(img, False) is None                      # OFF (default): no-op
    mask_on = _run(img, True)
    assert mask_on is not None
    assert mask_on[:5, :].all() and mask_on.sum() == 500  # module excluded

    # A single genuinely-saturated Bragg pixel (<= 1e-4) is preserved.
    sparse = np.zeros((100, 100), dtype=np.uint16)
    sparse[50, 50] = 65535
    assert _run(sparse, True) is None


def test_reduction_saturation_ceiling_dtype_derived_not_hardcoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ceiling follows the integer dtype (uint8 -> 255), never assumes 65535;
    a float frame (ceiling None) is a no-op even with the flag on."""
    calls: list[dict] = []
    monkeypatch.setattr(
        reduction_core, "integrate_1d",
        lambda image, ai, **kw: (calls.append(kw),
                                 _r1d(1.0, unit=kw.get("unit", "q_A^-1")))[1])

    def _run(img):
        calls.clear()
        run_reduction(
            ReductionPlan(
                integration_1d=Integration1DPlan(npt=5, unit="q_A^-1", method="BBox"),
                mask_saturation=True),
            Scan(name="x", frames=[Frame(index=0, image=img.copy())],
                 integrator=object()),
            chunk_size=1,
        )
        return calls[0]["mask"]

    u8 = np.zeros((100, 100), dtype=np.uint8)
    u8[:5, :] = 255             # uint8 ceiling is 255, not 65535
    m = _run(u8)
    assert m is not None and m[:5, :].all() and m.sum() == 500

    f = np.zeros((100, 100), dtype=float)
    f[:5, :] = 65535.0          # float -> integer dtype lost -> ceiling None -> no-op
    assert _run(f) is None
