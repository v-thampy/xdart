from __future__ import annotations

import json

import numpy as np

from ssrl_xrd_tools.analysis import (
    AnalysisResult,
    PeakFitPlan,
    RSMPlan,
    StitchPlan,
    run_peak_fit,
    run_rsm,
    run_stitch,
)
import ssrl_xrd_tools.analysis.plans as plan_mod
from ssrl_xrd_tools.core.containers import IntegrationResult1D
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
