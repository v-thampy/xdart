from __future__ import annotations

import json

import numpy as np
import pytest

from ssrl_xrd_tools.analysis import (
    AnalysisResult,
    PeakFitPlan,
    RSMPlan,
    Sin2PsiPlan,
    StitchPlan,
    run_peak_fit,
    run_rsm,
    run_sin2psi,
    run_stitch,
)
import ssrl_xrd_tools.analysis.plans as plan_mod
from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.core.scan import Scan, ScanFrame
from ssrl_xrd_tools.sources import MemoryFrameSource


def _r1d(value: float = 1.0) -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([value, value + 1.0]),
        sigma=None,
        unit="q_A^-1",
    )


def test_run_stitch_collects_images_and_metadata(monkeypatch):
    calls = {}

    def fake_stitch_images(images, base_poni, **kwargs):
        calls["images"] = images
        calls["base_poni"] = base_poni
        calls["kwargs"] = kwargs
        return _r1d(5.0)

    monkeypatch.setattr(plan_mod, "stitch_images", fake_stitch_images)
    source = MemoryFrameSource(
        [
            ScanFrame(2, image=np.ones((2, 2)), metadata={"del": 1.0, "nu": 0.5, "I0": 10.0}),
            ScanFrame(5, image=np.full((2, 2), 2.0), metadata={"DEL": 2.0, "NU": 0.7, "i0": 20.0}),
        ],
        name="angles",
    )
    poni = object()
    result = run_stitch(
        StitchPlan(
            base_poni=poni,
            rot1_key="del",
            rot2_key="nu",
            monitor_key="i0",
            mode="1d",
            npt_1d=7,
        ),
        source,
    )

    assert isinstance(result, AnalysisResult)
    assert result.payload.intensity.tolist() == [5.0, 6.0]
    assert calls["base_poni"] is poni
    assert [im.sum() for im in calls["images"]] == [4.0, 8.0]
    assert calls["kwargs"]["rot1_angles"].tolist() == [1.0, 2.0]
    assert calls["kwargs"]["rot2_angles"].tolist() == [0.5, 0.7]
    assert calls["kwargs"]["normalization"].tolist() == [10.0, 20.0]
    assert calls["kwargs"]["npt_1d"] == 7


def test_run_stitch_runs_real_synthetic_multigeometry():
    """#8c: drive run_stitch END-TO-END through the REAL pyFAI MultiGeometry
    engine (not monkeypatched) on a small synthetic detector-angle stack —
    validating the notebook-stable StitchPlan API on real inputs."""
    from ssrl_xrd_tools.core.containers import PONI

    shape = (195, 487)                         # Pilatus 100k (small + real -> fast)
    base_poni = PONI(
        dist=0.2,
        poni1=shape[0] * 172e-6 / 2.0,
        poni2=shape[1] * 172e-6 / 2.0,
        rot1=0.0, rot2=0.0, rot3=0.0,
        wavelength=1.0e-10, detector="Pilatus100k",
    )

    def _ring(seed):
        rng = np.random.default_rng(seed)
        ny, nx = shape
        y, x = np.mgrid[:ny, :nx]
        r = np.sqrt((y - ny / 2.0) ** 2 + (x - nx / 2.0) ** 2)
        return (500.0 * np.exp(-((r - 60.0) / 10.0) ** 2)
                + rng.poisson(3, size=shape)).astype(float)

    # Three frames at increasing detector rot1 (deg) + a monitor counter.
    frames = [ScanFrame(i, image=_ring(i),
                        metadata={"rot1": float(5 * i), "I0": float(10 + i)})
              for i in range(3)]
    source = MemoryFrameSource(frames, name="stitch")

    result = run_stitch(
        StitchPlan(base_poni=base_poni, rot1_key="rot1", monitor_key="I0",
                   mode="1d", npt_1d=200),
        source,
    )

    assert result.kind == "stitch"
    payload = result.payload
    assert payload.radial.shape == (200,) and payload.intensity.shape == (200,)
    assert np.isfinite(payload.radial).all()
    assert np.nanmax(payload.intensity) > 0          # the ring integrates to signal
    assert result.provenance["frame_indices"] == [0, 1, 2]


def test_canonical_scan_exposes_scan_data_for_rsm_consumers():
    scan = Scan(
        "scan",
        [ScanFrame(10, image=np.ones((1, 1)), metadata={"th": 1.5})],
        motors={"del": np.array([2.5])},
    )

    assert scan.scan_data.loc[10, "th"] == 1.5
    assert scan.scan_data.loc[10, "del"] == 2.5


def test_run_rsm_delegates_single_and_multi_source(monkeypatch):
    calls = []
    source = MemoryFrameSource([np.ones((2, 2))], name="rsm")
    plan = RSMPlan(mapper=object(), diff_motors=("th",), bins=(3, 4, 5), energy=12000.0)

    def fake_process(scan, mapper, diff_motors, bins, **kwargs):
        calls.append(("single", scan, diff_motors, bins, kwargs))
        return "volume-single"

    def fake_grid(mapper, scan_inputs, diff_motors, bins, **kwargs):
        calls.append(("multi", list(scan_inputs), diff_motors, bins, kwargs))
        return "volume-multi"

    monkeypatch.setattr(plan_mod, "process_scan_from_nexus", fake_process)
    monkeypatch.setattr(plan_mod, "grid_scans_streaming", fake_grid)

    single = run_rsm(plan, source)
    multi = run_rsm(plan, [source])

    assert single.payload == "volume-single"
    assert multi.payload == "volume-multi"
    assert calls[0][0] == "single"
    assert calls[0][2] == ("th",)
    assert calls[0][3] == (3, 4, 5)
    assert calls[0][4]["energy"] == 12000.0
    assert calls[1][0] == "multi"
    assert calls[1][1][0].scan is source


def test_peak_fit_plan_and_result_envelope_are_json_safe(monkeypatch):
    monkeypatch.setattr(plan_mod, "fit_peaks", lambda *args, **kwargs: {"ok": True})
    result = run_peak_fit(
        PeakFitPlan(positions=(1.0,), model="gaussian", fit_kwargs={"method": "leastsq"}),
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 3.0, 1.0]),
    )

    data = json.loads(result.to_json())
    assert data["kind"] == "peak_fit"
    assert data["payload_type"] == "dict"
    assert data["provenance"]["plan"]["positions"] == [1.0]
    assert data["provenance"]["plan"]["fit_kwargs"]["method"] == "leastsq"


def test_peak_fit_plan_runs_real_synthetic_peak():
    x = np.linspace(0.0, 3.0, 301)
    y = 2.0 + 0.2 * x + 50.0 * np.exp(-0.5 * ((x - 1.25) / 0.08) ** 2)

    result = run_peak_fit(
        PeakFitPlan(
            positions=(1.25,),
            model="gaussian",
            background="linear",
            sigma_init=0.08,
            sigma_bounds=(0.02, 0.2),
            center_bounds_delta=0.2,
        ),
        x,
        y,
    )

    assert result.kind == "peak_fit"
    assert result.payload.success
    assert result.payload.n_peaks == 1
    assert result.payload.peak_centers[0] == pytest.approx(1.25, abs=0.01)


def test_sin2psi_plan_runs_real_synthetic_map():
    q = np.linspace(1.75, 2.25, 241)
    chi = np.linspace(-60.0, 60.0, 25)
    intensity = np.empty((q.size, chi.size), dtype=float)
    q0 = 2.0
    for j, chi_val in enumerate(chi):
        center = q0 + 0.025 * np.sin(np.deg2rad(abs(chi_val))) ** 2
        intensity[:, j] = 5.0 + 100.0 * np.exp(-0.5 * ((q - center) / 0.025) ** 2)

    result2d = IntegrationResult2D(
        radial=q,
        azimuthal=chi,
        intensity=intensity,
        unit="q_A^-1",
        azimuthal_unit="chi_deg",
    )

    result = run_sin2psi(
        Sin2PsiPlan(
            q_range=(1.92, 2.12),
            chi_centers=(-50.0, -25.0, 0.0, 25.0, 50.0),
            chi_width=8.0,
            model="gaussian",
            background="constant",
            sigma_init=0.025,
            sigma_bounds=(0.005, 0.08),
            center_bounds_delta=0.08,
        ),
        result2d,
    )

    assert result.kind == "sin2psi"
    assert len(result.payload.peak_fits) == 5
    assert np.all(np.isfinite(result.payload.d_values))
    assert result.payload.r_squared > 0.95
