"""Tests for the scan/frame headless reduction API."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.reduction import (
    CancelToken,
    Frame,
    MemorySink,
    NexusSink,
    ReductionPlan,
    Scan,
    run_reduction,
)
import ssrl_xrd_tools.reduction.core as reduction_core


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
            Frame(index=2, image=np.zeros((2, 2)), metadata={"i0": 20}),
            Frame(index=1, image=np.zeros((2, 2)), metadata={"i0": 10}),
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
        ({"integrate_1d": False, "integrate_2d": False}, "integrate_1d"),
        ({"npt_1d": 0}, "npt_1d"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"gi": True}, "gi_incident_angle"),
    ],
)
def test_reduction_plan_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ReductionPlan(**kwargs)


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
                mask=np.array([[False, True], [False, False]]),
            )
        ],
        integrator=object(),
    )
    plan = ReductionPlan(
        integrate_1d=True,
        integrate_2d=True,
        npt_1d=5,
        npt_rad_2d=6,
        npt_azim_2d=7,
        unit="2th_deg",
        method_1d="BBox",
        method_2d="csr",
        mask=np.array([[False, False], [True, False]]),
        threshold_min=0.0,
        threshold_max=50.0,
        normalization_factors={0: 9.0},
        chunk_size=1,
    )

    result = run_reduction(plan, scan, progress_cb=events.append)

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
    assert np.isnan(calls[0]["image"][1, 0])  # threshold_max
    assert np.isnan(calls[0]["image"][1, 1])  # threshold_min
    assert calls[1]["npt_rad"] == 6
    assert calls[1]["npt_azim"] == 7


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
        ReductionPlan(integrate_2d=False, clear_frame_images=True, chunk_size=2),
        scan,
        cancel_token=token,
    )

    assert result.cancelled
    assert result.n_processed == 1
    assert seen == [0]
    assert scan.frames[0].image is None
    assert scan.frames[1].image is not None


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
        assert "entry/reduction/0/int_1d/intensity" in h5
        np.testing.assert_allclose(
            h5["entry/reduction/0/int_1d/intensity"][()],
            [3.0, 4.0],
        )


def test_existing_notebook_import_surface_still_imports() -> None:
    from ssrl_xrd_tools.analysis.fitting import FitConfig, PhaseFitter, fit_peaks
    from ssrl_xrd_tools.analysis.strain import sin2psi_analysis
    from ssrl_xrd_tools.integrate import (
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
