from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from xrd_tools.analysis import (
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
import xrd_tools.analysis.plans as plan_mod
from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.core.scan import Scan, ScanFrame
from xrd_tools.sources import MemoryFrameSource


_HAS_LMFIT = importlib.util.find_spec("lmfit") is not None


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
    from xrd_tools.core.containers import PONI

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


def _pilatus_poni():
    from xrd_tools.core.containers import PONI
    shape = (195, 487)
    return shape, PONI(
        dist=0.2, poni1=shape[0] * 172e-6 / 2.0, poni2=shape[1] * 172e-6 / 2.0,
        rot1=0.0, rot2=0.0, rot3=0.0, wavelength=1.0e-10, detector="Pilatus100k")


def _ring_source(name, *, n, rot_base, source_path):
    shape, _ = _pilatus_poni()

    def _ring(seed):
        rng = np.random.default_rng(seed)
        ny, nx = shape
        y, x = np.mgrid[:ny, :nx]
        r = np.sqrt((y - ny / 2.0) ** 2 + (x - nx / 2.0) ** 2)
        return (500.0 * np.exp(-((r - 60.0) / 10.0) ** 2)
                + rng.poisson(3, size=shape)).astype(float)

    frames = [ScanFrame(i, image=_ring(rot_base * 10 + i),
                        metadata={"rot1": float(rot_base + 5 * i), "I0": float(10 + i)},
                        source_path=source_path, source_frame_index=i)
              for i in range(1, n + 1)]
    return MemoryFrameSource(frames, name=name)


def test_run_stitch_single_in_a_list_is_equivalent_to_bare_source():
    """A group of ONE source must reduce to exactly the single-source result (no
    spurious CompositeFrameSource re-indexing)."""
    _, base_poni = _pilatus_poni()
    src = _ring_source("s", n=3, rot_base=0, source_path="/data/s.h5")
    plan = StitchPlan(base_poni=base_poni, rot1_key="rot1", mode="1d", npt_1d=200)

    bare = run_stitch(plan, src)
    grouped = run_stitch(plan, [src])

    np.testing.assert_array_equal(bare.payload.radial, grouped.payload.radial)
    np.testing.assert_array_equal(bare.payload.intensity, grouped.payload.intensity)
    # both flat (single scan, no scan_labels) → scan_label None
    assert {r["scan_label"] for r in bare.frame_records} == {None}
    assert [r["frame_index"] for r in bare.frame_records] == [1, 2, 3]


def test_run_stitch_groups_multiple_sources_and_tags_frame_records():
    """Two scans grouped → one merged payload + scan-tagged contributing-frame
    records (per-scan labels, real scan numbers via scan_labels)."""
    _, base_poni = _pilatus_poni()
    s5 = _ring_source("s5", n=2, rot_base=0, source_path="/data/scan5.h5")
    s7 = _ring_source("s7", n=2, rot_base=30, source_path="/data/scan7.h5")
    plan = StitchPlan(base_poni=base_poni, rot1_key="rot1", mode="1d", npt_1d=200)

    result = run_stitch(plan, [s5, s7], scan_labels=[5, 7])

    assert result.kind == "stitch"
    assert result.payload.radial.shape == (200,)
    assert np.nanmax(result.payload.intensity) > 0
    # 4 contributing frames, scan-tagged with per-scan labels 5-1,5-2,7-1,7-2
    tagged = [(r["scan_label"], r["frame_index"], r["source_path"])
              for r in result.frame_records]
    assert tagged == [
        (5, 1, "/data/scan5.h5"), (5, 2, "/data/scan5.h5"),
        (7, 1, "/data/scan7.h5"), (7, 2, "/data/scan7.h5"),
    ]


def test_run_stitch_empty_group_raises():
    _, base_poni = _pilatus_poni()
    plan = StitchPlan(base_poni=base_poni, rot1_key="rot1", mode="1d", npt_1d=8)
    with pytest.raises(ValueError, match="empty source group"):
        run_stitch(plan, [])


def test_run_stitch_frame_subset_records_only_contributing_frames():
    """A frame_indices SUBSET must record only the frames that actually merged —
    not every frame in the source (else the raw-popup lists non-contributing
    frames, and frame_records disagree with provenance)."""
    _, base_poni = _pilatus_poni()
    src = _ring_source("s", n=5, rot_base=0, source_path="/data/s.h5")  # labels 1..5
    plan = StitchPlan(base_poni=base_poni, rot1_key="rot1", mode="1d", npt_1d=64)

    result = run_stitch(plan, src, frame_indices=[1, 3])

    assert result.provenance["frame_indices"] == [1, 3]
    assert {r["frame_index"] for r in result.frame_records} == {1, 3}


def test_run_stitch_grouped_frame_subset_maps_global_to_member():
    """A subset over a GROUP is in the composite's GLOBAL index; the records must
    map it back to the right scan+local frame (not the global index)."""
    _, base_poni = _pilatus_poni()
    s5 = _ring_source("s5", n=2, rot_base=0, source_path="/data/scan5.h5")   # local 1,2
    s7 = _ring_source("s7", n=2, rot_base=30, source_path="/data/scan7.h5")  # local 1,2
    plan = StitchPlan(base_poni=base_poni, rot1_key="rot1", mode="1d", npt_1d=64)

    # global 0,1 = scan5 local 1,2 ; global 2,3 = scan7 local 1,2. Pick global 0 and 3.
    result = run_stitch(plan, [s5, s7], scan_labels=[5, 7], frame_indices=[0, 3])

    tagged = sorted((r["scan_label"], r["frame_index"]) for r in result.frame_records)
    assert tagged == [(5, 1), (7, 2)]


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


def test_run_rsm_attaches_scan_tagged_frame_records(monkeypatch):
    """run_rsm carries the raw-popup records too — a grouped RSM tags each
    contributing frame by its scan (RSM has the raw popup as well)."""
    monkeypatch.setattr(plan_mod, "grid_scans_streaming",
                        lambda *a, **k: "volume-multi")
    s5 = MemoryFrameSource(
        [ScanFrame(1, image=np.ones((2, 2)), source_path="/d/scan5.h5", source_frame_index=1)],
        name="s5")
    s7 = MemoryFrameSource(
        [ScanFrame(1, image=np.ones((2, 2)), source_path="/d/scan7.h5", source_frame_index=1)],
        name="s7")
    plan = RSMPlan(mapper=object(), diff_motors=("th",), bins=(3, 4, 5), energy=12000.0)

    result = run_rsm(plan, [s5, s7], scan_labels=[5, 7])
    assert [(r["scan_label"], r["frame_index"], r["source_path"]) for r in result.frame_records] == [
        (5, 1, "/d/scan5.h5"), (7, 1, "/d/scan7.h5")]


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


@pytest.mark.skipif(not _HAS_LMFIT, reason="requires lmfit")
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


@pytest.mark.skipif(not _HAS_LMFIT, reason="requires lmfit")
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
