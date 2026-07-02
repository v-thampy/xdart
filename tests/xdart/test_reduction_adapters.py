"""Tests for xdart's boundary into xrd_tools.reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from threading import RLock
from types import SimpleNamespace

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.core.containers import PONI
from xrd_tools.reduction import (
    FrameReduction,
    Integration1DPlan,
    Integration2DPlan,
    MaskSpec,
    ReductionPlan,
    ReductionResult,
)

from xdart.modules.live import LiveFrame, LiveFrameSeries, LiveScan
from xdart.modules.live_compat import normalize_live_class_names
import xdart.modules.reduction as reduction_adapters
from xdart.modules.reduction import (
    StandardPlanCache,
    frame_from_live_frame,
    plan_from_live_scan,
    reduce_live_frame,
    reduce_live_frames,
    scan_from_live_scan,
)




@pytest.fixture(autouse=True)
def _run_in_tmp(tmp_path, monkeypatch):
    """LiveScan defaults ``data_file`` to CWD-relative ``<name>.nxs`` and the
    frame-series persistence then writes it — several tests here construct
    ``LiveScan("scan", ...)`` without a data_file, littering the repo root
    with a real scan.nxs on every run.  Run each test from tmp instead."""
    monkeypatch.chdir(tmp_path)


def _poni() -> PONI:
    return PONI(
        dist=0.1,
        poni1=0.01,
        poni2=0.02,
        wavelength=1e-10,
        detector="Detector",
    )


def _r1d() -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([10.0, 11.0]),
        sigma=None,
        unit="q_A^-1",
    )


def _r2d() -> IntegrationResult2D:
    return IntegrationResult2D(
        radial=np.array([0.0, 1.0]),
        azimuthal=np.array([-90.0, 90.0]),
        intensity=np.ones((2, 2)),
        sigma=None,
        unit="q_A^-1",
    )


class _FakeIntegrator:
    pass


def test_legacy_ewald_aliases_are_dropped() -> None:
    """The Live rename's transitional Ewald* class + function aliases were
    removed at the end of the rename release window.  Reader-side string
    normalisation in :mod:`xdart.modules.live_compat` still keeps old
    ``.nxs`` files loading (covered by a separate test below).
    """
    import xdart.modules.ewald as ewald_pkg
    import xdart.modules.reduction as reduction_pkg

    for legacy_name in ("EwaldArch", "EwaldSphere", "ArchSeries"):
        assert not hasattr(ewald_pkg, legacy_name), (
            f"{legacy_name} should have been dropped; still on "
            f"xdart.modules.ewald"
        )

    for legacy_fn in (
        "frame_from_ewald_arch",
        "scan_from_ewald_sphere",
        "plan_from_ewald_sphere",
        "reduce_ewald_arch",
        "dispatch_arch_reduction",
    ):
        assert not hasattr(reduction_pkg, legacy_fn), (
            f"{legacy_fn} should have been dropped; still on "
            f"xdart.modules.reduction"
        )


def test_legacy_live_class_names_are_normalized_from_reader_data() -> None:
    provenance = {
        "config": {
            "class": "EwaldSphere",
            "frames": ["EwaldArch", "xdart.modules.ewald.arch.EwaldArch"],
            "series": {"type": "ArchSeries"},
        },
        "unchanged": "not_a_class_name",
    }

    normalized = normalize_live_class_names(provenance)

    assert normalized["config"]["class"] == "LiveScan"
    assert normalized["config"]["frames"] == [
        "LiveFrame",
        "xdart.modules.ewald.frame.LiveFrame",
    ]
    assert normalized["config"]["series"]["type"] == "LiveFrameSeries"
    assert normalized["unchanged"] == "not_a_class_name"


def test_live_scan_loader_normalizes_legacy_reduction_provenance(
    monkeypatch,
    tmp_path,
) -> None:
    import xrd_tools.io.nexus as nexus_io

    fake_ds = SimpleNamespace(
        attrs={
            "reduction": {
                "config": {
                    "bai_1d_args": {"owner": "EwaldArch"},
                    "bai_2d_args": {"owner": "EwaldSphere"},
                },
            },
        },
        sizes={},
        data_vars={},
        dims={},
        coords={},
    )
    monkeypatch.setattr(nexus_io, "read_scan_metadata", lambda _path: fake_ds)

    live_scan = LiveScan("old", data_file=str(tmp_path / "old_037.nxs"))
    live_scan._load_from_nexus_v2(None)

    assert live_scan.bai_1d_args["owner"] == "LiveFrame"
    assert live_scan.bai_2d_args["owner"] == "LiveScan"


def test_live_scan_loader_uses_union_of_1d_and_2d_labels(monkeypatch, tmp_path) -> None:
    import xrd_tools.io.nexus as nexus_io

    class _FakeDataset:
        attrs = {"reduction": {}}
        sizes = {"frame": 2, "frame_2d": 2}
        data_vars = {}
        dims = {"frame": 2, "frame_2d": 2}
        coords = {
            "frame": SimpleNamespace(values=np.array([0, 2])),
            "frame_2d": SimpleNamespace(values=np.array([1, 2])),
        }

        def __getitem__(self, key):
            return self.coords[key]

    fake_ds = _FakeDataset()
    monkeypatch.setattr(nexus_io, "read_scan_metadata", lambda _path: fake_ds)

    live_scan = LiveScan("union", data_file=str(tmp_path / "union.nxs"))
    live_scan._load_from_nexus_v2(None)
    assert list(live_scan.frames.index) == [0, 1, 2]


def test_live_scan_loader_reindexes_metadata_to_union_labels(monkeypatch, tmp_path) -> None:
    import xrd_tools.io.nexus as nexus_io

    class _Var:
        def __init__(self, values, dims):
            self.values = np.asarray(values)
            self.dims = dims

    class _FakeDataset:
        attrs = {"reduction": {}}
        sizes = {"frame": 2, "frame_2d": 2}
        data_vars = {"i0": _Var([10.0, 20.0], ("frame",))}
        dims = {"frame": 2, "frame_2d": 2}
        coords = {
            "frame": _Var([0, 2], ("frame",)),
            "frame_2d": _Var([1, 2], ("frame_2d",)),
        }

        def __getitem__(self, key):
            if key in self.coords:
                return self.coords[key]
            return self.data_vars[key]

    monkeypatch.setattr(nexus_io, "read_scan_metadata", lambda _path: _FakeDataset())

    live_scan = LiveScan("union_meta", data_file=str(tmp_path / "union_meta.nxs"))
    live_scan._load_from_nexus_v2(None)

    assert list(live_scan.frames.index) == [0, 1, 2]
    assert list(live_scan.scan_data.index) == [0, 1, 2]
    assert live_scan.scan_data.loc[0, "i0"] == 10.0
    assert np.isnan(live_scan.scan_data.loc[1, "i0"])
    assert live_scan.scan_data.loc[2, "i0"] == 20.0


def test_frame_from_live_frame_maps_simple_fields(tmp_path) -> None:
    frame = LiveFrame(
        idx=4,
        map_raw=np.arange(4).reshape(2, 2),
        poni=_poni(),
        bg_raw=np.ones((2, 2)),
        mask=np.array([1, 3]),
        scan_info={"th": 1.2, "i0": 99.0},
    )
    frame.source_file = "frame_0004.tif"
    frame.source_frame_idx = 2
    frame._source_root = str(tmp_path)
    frame.map_norm = 5.0

    frame = frame_from_live_frame(frame)

    assert frame.index == 4
    np.testing.assert_array_equal(frame.image, np.arange(4).reshape(2, 2))
    assert isinstance(frame.mask, MaskSpec)
    np.testing.assert_array_equal(
        frame.mask.to_bool((2, 2)),
        np.array([[False, True], [False, True]]),
    )
    assert frame.metadata["th"] == 1.2
    assert frame.source_path == tmp_path / "frame_0004.tif"
    assert frame.source_frame_index == 2
    np.testing.assert_array_equal(frame.background, np.ones((2, 2)))
    assert frame.normalization_factor is None


def test_frame_from_live_frame_can_skip_large_background() -> None:
    frame = LiveFrame(
        idx=4,
        map_raw=np.arange(4).reshape(2, 2),
        poni=_poni(),
        bg_raw=np.ones((2, 2)),
    )

    frame = frame_from_live_frame(
        frame,
        include_image=False,
        include_background=False,
    )

    assert frame.image is None
    assert frame.background is None


def test_scan_from_live_scan_uses_scan_frame_names() -> None:
    a2 = LiveFrame(idx=2, map_raw=np.ones((2, 2)), poni=_poni(),
                   scan_info={"th": 2.0})
    a1 = LiveFrame(idx=1, map_raw=np.zeros((2, 2)), poni=_poni(),
                   scan_info={"th": 1.0})
    scan = LiveScan(
        "scan42",
        frames=[a2, a1],
        scan_data=pd.DataFrame({"th": [1.0, 2.0]}, index=[1, 2]),
        mg_args={"wavelength": 0.7293e-10},
        data_file="scan42.nxs",
    )

    scan = scan_from_live_scan(scan)

    assert scan.name == "scan42"
    assert [f.index for f in scan.frames] == [1, 2]
    assert scan.poni == _poni()
    assert scan.wavelength == pytest.approx(0.7293)
    np.testing.assert_allclose(scan.motors["th"], [1.0, 2.0])
    assert scan.output_path.name == "scan42.nxs"


def test_scan_from_live_scan_uses_authoritative_reloaded_wavelength() -> None:
    live = LiveScan(
        "scan42",
        frames=[LiveFrame(idx=1, map_raw=np.ones((2, 2)), poni=_poni())],
        mg_args={"wavelength": 1e-10},
    )
    live._persisted_wavelength_m = 1e-10

    scan = scan_from_live_scan(live)

    assert scan.wavelength == pytest.approx(1.0)


def test_scan_from_live_scan_fills_frame_metadata_from_scan_data(tmp_path) -> None:
    frame = LiveFrame(idx=5, map_raw=np.ones((2, 2)), poni=_poni())
    scan = LiveScan(
        "scan",
        frames=[frame],
        # data_file pinned to tmp: LiveScan defaults to CWD-relative
        # "<name>.nxs", and the frame-series persistence then litters the
        # repo root with a real scan.nxs on every test run.
        data_file=str(tmp_path / "scan.nxs"),
        scan_data=pd.DataFrame({"i0": [42.0]}, index=[5]),
    )

    headless = scan_from_live_scan(scan)

    assert headless.frames[0].metadata["i0"] == 42.0


def test_plan_from_live_scan_maps_integration_settings() -> None:
    frame = LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    scan = LiveScan(
        "scan",
        frames=[frame],
        bai_1d_args={
            "numpoints": 123,
            "unit": "2th_deg",
            "method": "BBox",
            "radial_range": (1.0, 2.0),
            "azimuth_range": (70.0, 110.0),
            "monitor": "i0",
            "chi_offset": 90.0,
            "gi_mode_1d": "q_ip",
        },
        bai_2d_args={
            "npt_rad": 20,
            "npt_azim": 30,
            "unit": "q_A^-1",
            "method": "csr",
            "azimuth_range": (-10.0, 10.0),
            "monitor": "i1",
            "gi_mode_2d": "qip_qoop",
        },
        global_mask=np.array([0, 3]),
    )

    plan = plan_from_live_scan(scan)

    assert plan.integration_1d is not None
    assert plan.integration_2d is not None
    assert plan.integration_1d.npt == 123
    assert plan.integration_2d.npt_rad == 20
    assert plan.integration_2d.npt_azim == 30
    assert plan.integration_1d.unit == "2th_deg"
    assert plan.integration_2d.unit == "q_A^-1"
    assert plan.integration_1d.method == "BBox"
    assert plan.integration_2d.method == "csr"
    assert plan.integration_1d.radial_range == (1.0, 2.0)
    assert plan.integration_2d.radial_range is None
    assert plan.integration_1d.azimuth_range == (-20.0, 20.0)
    assert plan.integration_2d.azimuth_range == (-10.0, 10.0)
    assert plan.integration_1d.monitor_key == "i0"
    assert plan.integration_2d.monitor_key == "i1"
    assert plan.integration_2d.azimuth_offset == 0.0
    assert "gi_mode_1d" not in plan.integration_1d.extra
    assert "gi_mode_2d" not in plan.integration_2d.extra
    # plan.gi is None on a non-GI scan (was bool flag pre-S2)
    assert plan.gi is None
    np.testing.assert_array_equal(
        plan.mask,
        np.array([[True, False], [False, True]]),
    )


def test_plan_from_live_scan_preserves_non_gi_chi_1d_mode() -> None:
    frame = LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    scan = LiveScan(
        "scan",
        frames=[frame],
        skip_2d=True,
        bai_1d_args={
            "numpoints": 181,
            "unit": "chi_deg",
            "method": "csr",
            "radial_range": (0.5, 4.0),
            "azimuth_range": (-90.0, 90.0),
        },
    )

    plan = plan_from_live_scan(scan, integrate_2d=False)

    assert plan.integration_1d is not None
    assert plan.integration_2d is None
    assert plan.integration_1d.unit == "chi_deg"
    assert plan.integration_1d.npt == 181
    assert plan.integration_1d.method == "csr"
    assert plan.integration_1d.radial_range == (0.5, 4.0)
    assert plan.integration_1d.azimuth_range == (-90.0, 90.0)
    assert plan.gi is None


def test_plan_from_gi_scan_requires_incident_angle_or_motor() -> None:
    scan = LiveScan(
        "scan",
        frames=[LiveFrame(idx=0, map_raw=np.ones((2, 2)))],
        gi=True,
        incidence_motor="",
    )

    try:
        plan_from_live_scan(scan)
    except ValueError as exc:
        assert "gi_incident_angle" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing GI angle to fail")


def test_plan_from_gi_scan_maps_manual_numeric_incidence() -> None:
    scan = LiveScan(
        "scan",
        frames=[LiveFrame(idx=0, map_raw=np.ones((2, 2)))],
        gi=True,
        incidence_motor="3.0",
    )

    plan = plan_from_live_scan(scan)

    assert plan.gi is not None
    assert plan.gi.incident_angle == 3.0
    assert plan.gi.incidence_motor is None
    assert plan.integration_1d.unit == "q_A^-1"
    assert plan.integration_2d.unit == "qip_A^-1"


def test_plan_from_reloaded_gi_scan_recovers_orientation_from_gi_config() -> None:
    """Reintegrate-on-reload regression: a ``.nxs``-reloaded scan carries its GI
    geometry only in ``scan.gi_config`` (the live run sets direct
    ``sample_orientation`` / ``tilt_angle`` attrs via sync_live_scan_gi_settings;
    a reload restores only the dict).  ``plan_from_live_scan`` must recover them
    from ``gi_config`` — otherwise sample_orientation silently defaults to 1 and
    the GI out-of-plane (Q_oop) axis flips sign vs the live run.
    """
    class _Frames:
        index = [0]

    scan = SimpleNamespace(
        gi=True,
        skip_2d=False,
        bai_1d_args={"gi_mode_1d": "q_total"},
        bai_2d_args={"gi_mode_2d": "qip_qoop"},
        # No direct sample_orientation / tilt_angle attrs — only gi_config,
        # exactly as a reloaded scan presents them.
        gi_config={"sample_orientation": 4, "tilt_angle": 1.5},
        incidence_motor="3.0",
        global_mask=None,
        frames=_Frames(),
    )
    assert not hasattr(scan, "sample_orientation")

    plan = plan_from_live_scan(scan)

    assert plan.gi is not None
    assert plan.gi.sample_orientation == 4
    assert plan.gi.tilt_angle == 1.5


def test_plan_uses_detector_shape_for_flat_mask_on_reload() -> None:
    """Reintegrate-on-reload regression: the flat ``global_mask`` indexes the
    FULL-RES detector, so the plan must build the mask against
    ``scan.detector_shape`` — NOT ``frames[0].map_raw.shape``, which on a
    reloaded scan can be a THUMBNAIL (the full-res flat indices then fall out of
    bounds and the mask is silently dropped, so reintegrate runs UNMASKED).
    """
    class _Frames:
        index = [0]

        def __getitem__(self, i):
            return SimpleNamespace(map_raw=np.ones((4, 4)))  # reloaded thumbnail

    scan = SimpleNamespace(
        gi=False,
        skip_2d=False,
        bai_1d_args={},
        bai_2d_args={},
        # Masks pixel 63 — valid in the 8x8 detector, out of bounds in the 4x4
        # thumbnail (which is what the buggy frame-shape path used).
        global_mask=np.array([63], dtype=np.int64),
        detector_shape=(8, 8),
        frames=_Frames(),
    )

    mask = plan_from_live_scan(scan).mask

    assert mask is not None                       # not dropped
    assert np.asarray(mask).shape == (8, 8)       # baked at detector shape
    assert bool(np.asarray(mask)[7, 7])           # the masked pixel survives


def test_standard_plan_cache_returns_headless_plan_for_gi_scan() -> None:
    scan = LiveScan(
        "scan",
        frames=[LiveFrame(idx=0, map_raw=np.ones((2, 2)), scan_info={"th": 0.2})],
        gi=True,
        incidence_motor="th",
    )

    plan = StandardPlanCache().get(scan)

    assert plan is not None
    assert plan.gi is not None
    assert plan.gi.incidence_motor == "th"


def test_standard_plan_cache_can_use_native_builder_override() -> None:
    scan = LiveScan(
        "scan",
        frames=[LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())],
        bai_1d_args={"numpoints": 10},
        bai_2d_args={"npt_rad": 11, "npt_azim": 12},
    )
    native = ReductionPlan(
        integration_1d=Integration1DPlan(npt=321),
        integration_2d=Integration2DPlan(npt_rad=222, npt_azim=111),
    )
    calls = []

    def builder(live_scan, *, integrate_1d=True, integrate_2d=True):
        calls.append((live_scan, integrate_1d, integrate_2d))
        return native

    cache = StandardPlanCache(plan_builder=builder)

    assert cache.get(scan) is native
    assert calls == [(scan, True, True)]
    assert cache.get(scan) is native
    assert calls == [(scan, True, True)]
    cache.plan_builder = None
    assert cache.get(scan) is not native


def test_standard_plan_cache_prepares_builder_scan_before_fingerprint() -> None:
    scan = LiveScan(
        "scan",
        frames=[LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())],
        bai_1d_args={"numpoints": 10},
        bai_2d_args={},
    )
    calls = []

    def builder(live_scan, *, integrate_1d=True, integrate_2d=True):
        calls.append(("build", live_scan.bai_1d_args["numpoints"]))
        return ReductionPlan(
            integration_1d=Integration1DPlan(
                npt=live_scan.bai_1d_args["numpoints"]
            )
        )

    def prepare_scan(live_scan):
        calls.append(("prepare", live_scan.bai_1d_args["numpoints"]))
        live_scan.bai_1d_args["numpoints"] = 44

    builder.prepare_scan = prepare_scan
    builder.plan_cache_key = ("snapshot", 44)
    cache = StandardPlanCache(plan_builder=builder)

    plan = cache.get(scan, integrate_1d=True, integrate_2d=False)

    assert plan.integration_1d.npt == 44
    assert calls == [("prepare", 10), ("build", 44)]

    scan.bai_1d_args["numpoints"] = 10
    assert cache.get(scan, integrate_1d=True, integrate_2d=False) is plan
    assert calls == [("prepare", 10), ("build", 44), ("prepare", 10)]


def test_reduce_live_frame_populates_existing_frame(monkeypatch) -> None:
    frame = LiveFrame(
        idx=7,
        map_raw=np.arange(4).reshape(2, 2),
        poni=_poni(),
        bg_raw=np.ones((2, 2)),
        scan_info={"I0": 33.0},
        mask=np.array([0]),
    )
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(monitor_key="i0"),
        integration_2d=Integration2DPlan(),
    )

    def fake_run_reduction(plan_arg, scan_arg, **kwargs):
        assert plan_arg.integration_1d.monitor_key == "i0"
        assert scan_arg.name == "scan"
        assert scan_arg.integrator == "ai"
        np.testing.assert_array_equal(scan_arg.frames[0].background, np.ones((2, 2)))
        assert isinstance(scan_arg.frames[0].mask, MaskSpec)
        np.testing.assert_array_equal(
            scan_arg.frames[0].mask.to_bool((2, 2)),
            np.array([[True, False], [False, False]]),
        )
        np.testing.assert_array_equal(
            plan_arg.mask,
            np.array([[False, False], [False, True]]),
        )
        return ReductionResult(
            scan_name="scan",
            frames={7: FrameReduction(7, result_1d=_r1d(), result_2d=_r2d())},
            n_processed=1,
        )

    monkeypatch.setattr(reduction_adapters, "run_reduction", fake_run_reduction)

    returned = reduce_live_frame(
        frame,
        plan,
        scan_name="scan",
        global_mask=np.array([3]),
        integrator="ai",
    )

    assert returned is frame
    assert frame.int_1d is not None
    assert frame.int_2d is not None
    assert frame.map_norm == 33.0


def test_reduce_live_frame_preserves_int_1d_when_plan_skips_1d(monkeypatch) -> None:
    """A 2D-only plan (a GI 2D reintegrate sets integrate_1d=False) must PRESERVE
    the frame's existing int_1d instead of clobbering it with the run's absent/
    wrong 1D — otherwise the live display shows a corrupted (spike) 1D."""
    frame = LiveFrame(idx=7, map_raw=np.arange(4).reshape(2, 2), poni=_poni(),
                      scan_info={"I0": 1.0}, mask=np.array([0]))
    sentinel_1d = object()
    frame.int_1d = sentinel_1d                       # pre-existing clean 1D
    plan = ReductionPlan(integration_1d=None,        # 2D-only
                         integration_2d=Integration2DPlan())

    monkeypatch.setattr(
        reduction_adapters, "run_reduction",
        lambda plan_arg, scan_arg, **kwargs: ReductionResult(
            scan_name="scan",
            frames={7: FrameReduction(7, result_1d=_r1d(), result_2d=_r2d())},
            n_processed=1))

    reduce_live_frame(frame, plan, scan_name="scan", integrator="ai")
    assert frame.int_1d is sentinel_1d               # preserved, NOT overwritten
    assert frame.int_2d is not None                  # 2D refreshed


def test_reduce_live_frames_uses_one_headless_batch(monkeypatch) -> None:
    frames = [
        LiveFrame(idx=1, map_raw=np.ones((2, 2)), poni=_poni(), scan_info={"i0": 2.0}),
        LiveFrame(idx=2, map_raw=np.full((2, 2), 2.0), poni=_poni(), scan_info={"i0": 4.0}),
    ]
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(monitor_key="i0"),
        integration_2d=Integration2DPlan(),
    )
    calls = []

    def fake_run_reduction(plan_arg, scan_arg, **kwargs):
        calls.append((plan_arg, scan_arg, kwargs))
        return ReductionResult(
            scan_name=scan_arg.name,
            frames={
                frame.index: FrameReduction(
                    frame.index,
                    result_1d=_r1d(),
                    result_2d=_r2d(),
                )
                for frame in scan_arg.frames
            },
            n_processed=len(scan_arg.frames),
        )

    monkeypatch.setattr(reduction_adapters, "run_reduction", fake_run_reduction)

    out = reduce_live_frames(
        frames,
        plan,
        scan_name="scan",
        global_mask=np.array([0]),
        integrator="ai",
        executor=2,
    )

    assert out == frames
    assert len(calls) == 1
    assert calls[0][1].name == "scan"
    assert [frame.index for frame in calls[0][1].frames] == [1, 2]
    assert calls[0][2]["executor"] == 2
    assert calls[0][2]["chunk_size"] == 2
    assert frames[0].int_1d is not None
    assert frames[1].int_2d is not None
    assert frames[0].map_norm == 2.0
    assert frames[1].map_norm == 4.0


def test_reduce_live_frames_skips_missing_results(monkeypatch) -> None:
    frames = [
        LiveFrame(idx=1, map_raw=np.ones((2, 2)), poni=_poni()),
        LiveFrame(idx=2, map_raw=np.full((2, 2), 2.0), poni=_poni()),
    ]
    plan = ReductionPlan(integration_1d=Integration1DPlan(), integration_2d=None)

    def fake_run_reduction(plan_arg, scan_arg, **kwargs):
        return ReductionResult(
            scan_name=scan_arg.name,
            frames={
                1: FrameReduction(
                    1,
                    result_1d=_r1d(),
                    result_2d=None,
                )
            },
            n_processed=1,
            cancelled=True,
        )

    monkeypatch.setattr(reduction_adapters, "run_reduction", fake_run_reduction)

    out = reduce_live_frames(frames, plan, scan_name="scan")

    assert out == [frames[0]]
    assert frames[0].int_1d is not None
    assert frames[1].int_1d is None


def test_mask_conversion_ignores_incompatible_mask() -> None:
    """A matching mask is applied; a shape-incompatible mask is ignored
    (plan.mask is None) with a warning rather than raising — reducing
    unmasked beats aborting the scan."""
    frame = LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    scan = LiveScan(
        "scan",
        frames=[frame],
        global_mask=np.array([[1, 0], [0, 1]]),
    )
    plan = plan_from_live_scan(scan)
    np.testing.assert_array_equal(
        plan.mask,
        np.array([[True, False], [False, True]]),
    )

    bad = LiveScan(
        "scan",
        frames=[frame],
        global_mask=np.ones((3, 3)),  # wrong shape for a 2x2 image
    )
    bad_plan = plan_from_live_scan(bad)
    assert bad_plan.mask is None


def test_flat_global_mask_is_preserved_until_shape_is_known() -> None:
    scan = LiveScan(
        "scan",
        frames=[LiveFrame(idx=0, map_raw=None, poni=_poni())],
        global_mask=np.array([0, 3]),
    )

    plan = plan_from_live_scan(scan)

    assert isinstance(plan.mask, MaskSpec)
    np.testing.assert_array_equal(
        plan.mask.to_bool((2, 2)),
        np.array([[True, False], [False, True]]),
    )


def test_nexus_worker_builds_headless_frame_shell() -> None:
    from xdart.gui.tabs.static_scan.wranglers import nexus_wrangler_thread

    scan = SimpleNamespace(
        name="scan",
        data_file="scan_out.nxs",
        skip_2d=False,
        _cached_integrator=_FakeIntegrator(),
        bai_1d_args={},
        bai_2d_args={},
    )
    worker = SimpleNamespace(
        command="",
        gi=False,
        poni=_poni(),
        incidence_motor="th",
        sample_orientation=4,
        tilt_angle=0,
        mask=np.array([3]),
        _xye_lock=RLock(),
        _xye_buffer=[],
        nexus_file="scan.nxs",
    )
    worker._resolve_frame_mask = lambda _scan, _img: np.array([0])

    frame = nexus_wrangler_thread.nexusThread._build_frame(
        worker,
        scan,
        5,
        np.ones((2, 2)),
        {"i0": 1.0},
    )

    assert frame.idx == 5
    assert frame.int_1d is None
    assert frame.int_2d is None
    assert frame.source_file.endswith("scan.nxs")
    assert frame.source_frame_idx == 5
    assert frame.skip_map_raw is True
    np.testing.assert_array_equal(frame.mask, np.array([0]))
    assert worker._xye_buffer == []


def test_spec_sequential_standard_path_calls_headless_reduction(monkeypatch) -> None:
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler_thread

    calls = []

    def fake_reduce_frames(frames, plan, **kwargs):
        calls.append((frames[0].idx, plan, kwargs))
        for frame in frames:
            frame.int_1d = _r1d()
            frame.int_2d = _r2d()
        return list(frames)

    monkeypatch.setattr(image_wrangler_thread, "reduce_live_frames", fake_reduce_frames)

    class _Signal:
        def emit(self, *args):
            return None

    scan = SimpleNamespace(
        name="scan",
        data_file="scan_out.nxs",
        skip_2d=False,
        _cached_integrator=_FakeIntegrator(),
        bai_1d_args={},
        bai_2d_args={},
    )
    from xdart.modules.reduction import StandardPlanCache
    worker = SimpleNamespace(
        showLabel=_Signal(),
        _middle_truncate=lambda text: text,
        _apply_threshold_inline=lambda img: img,
        _resolve_frame_mask=lambda _scan, _img: np.array([0]),
        gi=False,
        poni=_poni(),
        incidence_motor="th",
        sample_orientation=4,
        tilt_angle=0,
        series_average=False,
        mask=np.array([3]),
        batch_mode=True,
        xye_only=True,
        sub_label="",
        _xye_lock=RLock(),
        _xye_buffer=[],
        _plan_cache=StandardPlanCache(),
        _reduction_session_key_for=lambda *_args: ("scan", 1),
        _get_reduction_session=lambda _key, _factory: object(),
        _cancel_token=lambda: None,
    )

    image_wrangler_thread.imageThread._process_one(
        worker,
        scan,
        "frame_0001.tif",
        1,
        np.ones((2, 2)),
        {"i0": 1.0},
        0,
    )

    assert calls and calls[0][0] == 1
    assert calls[0][2]["scan_name"] == "scan"
    assert worker._xye_buffer[0][0] == 1


def test_single_frame_reintegration_uses_headless_reduction(monkeypatch) -> None:
    from xdart.gui.tabs.static_scan import scan_threads
    from xdart.gui.tabs.static_scan.scan_threads import integratorThread
    from xdart.modules.frame_publication import PublicationStore

    calls = []

    def fake_reduce_frames(frames, plan, **kwargs):
        calls.append((plan, list(frames), kwargs))
        for frame in frames:
            frame.int_1d = _r1d()
            frame.int_2d = None
        return list(frames)

    monkeypatch.setattr(scan_threads, "reduce_live_frames", fake_reduce_frames)

    frame = LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    scan = LiveScan(
        "scan",
        frames=[frame],
        bai_1d_args={"monitor": "i0"},
    )
    scan._cached_integrator = _FakeIntegrator()
    scan.global_mask = np.array([3])

    store = PublicationStore()
    thread = integratorThread(
        scan,
        None,
        None,
        None,
        [0],
        publication_store=store,
    )
    thread.bai_1d_SI()

    assert calls
    assert calls[0][0].integration_1d.monitor_key == "i0"
    assert calls[0][2]["scan_name"] == "scan"
    assert frame.int_1d is not None
    pub = store.get(0)
    assert pub is not None
    np.testing.assert_allclose(pub.view.intensity_1d, frame.int_1d.intensity)


def test_single_frame_2d_reintegration_is_2d_only(monkeypatch) -> None:
    # A 2D reintegrate (even the single-frame path) must compute ONLY the 2D and
    # leave the existing 1D untouched.  Recomputing the 1D here ran an
    # independent full-range/unmasked 1D pass that overwrote the good int_1d
    # (the flat-line + high-Q spike) and forced a 1D-stack rewrite that crashed a
    # partial save on Stop.  The fake reduce respects the plan so the assertion
    # actually exercises the preservation, not the fake.
    from xdart.gui.tabs.static_scan import scan_threads
    from xdart.gui.tabs.static_scan.scan_threads import integratorThread
    from xdart.modules.frame_publication import PublicationStore

    calls = []

    def fake_reduce_frames(frames, plan, **kwargs):
        calls.append(plan)
        for frame in frames:
            if plan.integration_1d is not None:
                frame.int_1d = _r1d()
            if plan.integration_2d is not None:
                frame.int_2d = _r2d()
        return list(frames)

    monkeypatch.setattr(scan_threads, "reduce_live_frames", fake_reduce_frames)

    # Seed the frame with the "good" 1D from a prior fresh integrate.
    good_1d = _r1d()
    frame = LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    frame.int_1d = good_1d
    scan = LiveScan("scan", frames=[frame])
    scan._cached_integrator = _FakeIntegrator()
    store = PublicationStore()
    thread = integratorThread(
        scan,
        None,
        None,
        None,
        [0],
        publication_store=store,
    )

    thread.bai_2d_SI()

    assert calls
    # 2D-only plan: the 1D pass must NOT be requested.
    assert calls[0].integration_1d is None
    assert calls[0].integration_2d is not None
    # The pre-existing good 1D survives untouched; the 2D is fresh.
    assert frame.int_1d is good_1d
    assert frame.int_2d is not None
    pub = store.get(0)
    assert pub is not None
    np.testing.assert_allclose(pub.view.intensity_2d, frame.int_2d.intensity.T)


def test_compute_bad_pixel_mask_toggle_is_authoritative() -> None:
    """'Auto Mask Saturated' is the AUTHORITATIVE on/off: OFF masks NOTHING
    (returns None) -- the raw frame incl. the uint32 sentinel + negatives is kept
    (strong Bragg peaks that saturate are the user's to keep); ON masks the
    uint32 dead/hot sentinel (4294967295) + negatives.  (Was wrong: the sentinel
    used to be masked ALWAYS, making the toggle pointless.)"""
    from xdart.modules.reduction import compute_bad_pixel_mask

    raw = np.ones((4, 4), dtype=np.uint32)
    raw[0, 0] = 4294967295
    raw[1, 1] = 4294967295
    # OFF -> nothing masked, even the uint32 sentinel
    assert compute_bad_pixel_mask(raw, mask_saturation=False) is None
    # ON -> the sentinel pixels are masked
    on = compute_bad_pixel_mask(raw, mask_saturation=True)
    assert on is not None and set(on.tolist()) == {0, 5}   # (0,0),(1,1) row-major
    # negatives are cut on ON, kept on OFF
    sraw = np.zeros((2, 2), dtype=np.int32)
    sraw[0, 1] = -5
    assert compute_bad_pixel_mask(sraw, mask_saturation=False) is None
    assert set(compute_bad_pixel_mask(sraw, mask_saturation=True).tolist()) == {1}
    # all-good frame -> None either way
    assert compute_bad_pixel_mask(
        np.ones((4, 4), dtype=np.uint32), mask_saturation=True) is None


def test_compute_bad_pixel_mask_saturation_ceiling_when_on() -> None:
    """When ON, the fraction-guarded uint16 65535 ceiling is cut (a whole module
    at the ceiling, >1e-4 fraction).  When OFF, nothing is cut."""
    from xdart.modules.reduction import compute_bad_pixel_mask

    raw = np.zeros((100, 100), dtype=np.uint16)
    raw[:, :50] = 65535                      # half the frame at the ceiling
    assert compute_bad_pixel_mask(raw, mask_saturation=False) is None
    on = compute_bad_pixel_mask(raw, mask_saturation=True)
    assert on is not None and on.size == 5000


def test_compute_bad_pixel_mask_float_ceiling_matches_display_policy() -> None:
    """Equivalence-spine guard (the reviews' blocker): on a FLOAT raw with a
    saturated block, live's _resolve_frame_mask uses display_logic's 65535 float
    fallback and masks it.  With core's 'auto' ceiling (None for float) the
    reintegrate path would NOT -> live≠reintegrate.  Both paths now pass the
    display ceiling, so they agree on float frames too."""
    from xdart.modules.reduction import compute_bad_pixel_mask
    from xdart.gui.tabs.static_scan.display_logic import (
        integer_saturation_ceiling)

    raw = np.zeros((100, 100), dtype=np.float32)
    raw[:, :50] = 65535.0                     # saturated block, but FLOAT dtype
    # core 'auto' ceiling -> None for float -> saturation NOT masked
    assert compute_bad_pixel_mask(raw, mask_saturation=True) is None
    # display ceiling (65535 float fallback) -> masked, matching the live path
    ceil = integer_saturation_ceiling(raw)
    on = compute_bad_pixel_mask(
        raw, mask_saturation=True, saturation_ceiling=ceil)
    assert on is not None and on.size == 5000


def test_bad_pixel_counts_reports_dummy_not_saturation_for_uint32() -> None:
    from xdart.modules.reduction import bad_pixel_counts

    raw = np.ones((10, 10), dtype=np.uint32)
    raw[0, 0] = 4294967295
    c = bad_pixel_counts(raw)
    assert c["size"] == 100
    assert c["uint32_dummy"] == 1
    assert c["negative"] == 0
    assert c["saturation"] == 0     # uint32 ceiling is excluded from saturation


def test_reintegrate_prep_stamps_uint32_bad_pixel_mask() -> None:
    """A frame lazy-loaded from the .nxs for reintegration carries mask=None
    (the per-frame bad-pixel mask is not persisted).
    _prepare_frame_for_headless_reduction must stamp it back on so the reduce
    excludes the uint32 dummies -- exactly what the live wrangler's
    _resolve_frame_mask does on the (clean) fresh path.  Regression: without it
    the dummies spiked the reintegrated 1D."""
    from xdart.gui.tabs.static_scan.scan_threads import integratorThread

    raw = np.ones((4, 4), dtype=np.uint32)
    raw[0, 0] = 4294967295
    frame = LiveFrame(idx=0, map_raw=raw, poni=_poni())
    frame.mask = None
    scan = LiveScan("scan", frames=[frame])
    scan.static = False
    scan.gi = False
    thread = integratorThread(scan, None, None, None, [0])
    # Default: Mask Saturated ON -> the dummy is stamped
    out = thread._prepare_frame_for_headless_reduction(frame)
    assert out.mask is not None
    assert set(np.asarray(out.mask).tolist()) == {0}    # the dummy at (0,0)

    # Mask Saturated OFF -> NOTHING masked (the toggle is authoritative): the
    # frame integrates raw, saturated pixels kept.
    from xdart.modules.reduction import ThresholdSaturationConfig
    frame.mask = None
    thread.threshold_config = ThresholdSaturationConfig(mask_saturation=False)
    out = thread._prepare_frame_for_headless_reduction(frame)
    assert out.mask is None


def test_plan_changes_output_shape_reads_intensity_array() -> None:
    """The partial-savable pre-check reads the stored IntegrationResult's
    .intensity shape.  Regression: np.shape(IntegrationResult1D) is () so a
    [-1] index raised IndexError, the check was swallowed, partial_savable
    stayed True, and the Stop "discard?" popup never showed on a shape-changing
    1D reintegrate (and an unchanged-npt 2D was mis-flagged unsavable)."""
    from xdart.gui.tabs.static_scan.scan_threads import integratorThread

    changed = integratorThread._plan_changes_output_shape
    f0 = SimpleNamespace(int_1d=_r1d(), int_2d=_r2d())   # 1D npt=2, 2D (2,2)
    # 1D: unchanged npt -> no shape change; changed npt -> shape change
    assert changed(SimpleNamespace(npt=2), None, f0) is False
    assert changed(SimpleNamespace(npt=2000), None, f0) is True
    # 2D: unchanged dims -> no change (old set(()) bug wrongly said True);
    #     changed dims -> change
    assert changed(None, SimpleNamespace(npt_rad=2, npt_azim=2), f0) is False
    assert changed(None, SimpleNamespace(npt_rad=500, npt_azim=500), f0) is True


def test_maybe_flag_unsavable_detects_axis_change_at_same_npt() -> None:
    """A reintegrate that keeps npt but changes the radial axis (e.g. a
    different range / GI mode) still can't partial-save -- the writer rejects a
    changed axis at the same npt, which the npt-only pre-check can't predict.
    _maybe_flag_unsavable catches it after the first frame so the Stop "discard?"
    popup shows.  (The user's npt=2000 1D run that failed to persist yet showed
    no popup.)"""
    from xdart.gui.tabs.static_scan.scan_threads import integratorThread

    scan = LiveScan(
        "scan", frames=[LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())])
    thread = integratorThread(scan, None, None, None, [0])

    def _frame_with_axis(lo, hi):
        f = LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
        f.int_1d = IntegrationResult1D(
            radial=np.linspace(lo, hi, 100), intensity=np.ones(100),
            sigma=None, unit="q_A^-1")
        return f

    thread._reint_stored_sig = integratorThread._frame_output_signature(
        _frame_with_axis(0.0, 5.0))

    # same npt (100) but a DIFFERENT extent -> not savable
    thread.reintegrate_partial_savable = True
    thread._maybe_flag_unsavable(_frame_with_axis(0.0, 8.0), False, "1D")
    assert thread.reintegrate_partial_savable is False

    # identical axis -> stays savable (a same-settings re-run can persist)
    thread.reintegrate_partial_savable = True
    thread._maybe_flag_unsavable(_frame_with_axis(0.0, 5.0), False, "1D")
    assert thread.reintegrate_partial_savable is True


def test_reintegrate_all_refreshes_publication_store(monkeypatch) -> None:
    # A1: reintegrating ("Integrate 2D") must republish into the
    # PublicationStore so the cake (payload path is preferred) shows the NEW
    # pixels, not the pre-reintegrate ones.  fake_reduce makes the 2D shape
    # depend on bai_2d_args['npt_rad'] so a changed arg is observable.
    from xdart.gui.tabs.static_scan import scan_threads
    from xdart.gui.tabs.static_scan.scan_threads import integratorThread
    from xdart.modules.frame_publication import PublicationStore

    def fake_reduce_frames(frames, plan, **kwargs):
        npt = int(scan.bai_2d_args.get("npt_rad", 10))
        for frame in frames:
            frame.int_1d = _r1d()
            frame.int_2d = IntegrationResult2D(
                radial=np.linspace(0.0, 1.0, npt),
                azimuthal=np.linspace(-90.0, 90.0, 3),
                intensity=np.ones((npt, 3)),
                unit="q_A^-1", azimuthal_unit="chi_deg",
            )
        return list(frames)

    monkeypatch.setattr(scan_threads, "reduce_live_frames", fake_reduce_frames)

    frame = LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    scan = LiveScan("scan", frames=[frame])
    scan._cached_integrator = _FakeIntegrator()
    scan.skip_2d = False
    scan.bai_2d_args = {"npt_rad": 10, "gi_mode_2d": "qip_qoop"}
    store = PublicationStore()

    thread = integratorThread(
        scan, None, None, None, [0],
        publication_store=store,
    )
    thread._prepare_reintegrate_shadow = lambda: None
    thread._write_reintegrate_shadow_batch = lambda *a, **k: None
    thread._finalize_reintegrate_shadow = lambda *a, **k: None
    thread._drop_reintegrate_shadow = lambda: None

    thread.bai_2d_all()
    pub = store.get(0)
    assert pub is not None
    # view stores (y=azimuthal, x=radial) → (3, npt)
    assert pub.view.intensity_2d.shape == (3, 10)

    # Reintegrate with npt_rad halved → the store must reflect the NEW args,
    # not the stale (3, 10) cake.
    scan.bai_2d_args = {"npt_rad": 5, "gi_mode_2d": "qip_qoop"}
    thread.bai_2d_all()
    pub2 = store.get(0)
    assert pub2 is not None
    assert pub2.view.intensity_2d.shape == (3, 5)


def test_reintegrate_close_surfaces_write_failure() -> None:
    from types import SimpleNamespace

    from xdart.gui.tabs.static_scan.scan_threads import integratorThread
    from xdart.modules.ewald import LiveScan

    def boom_finish():
        raise RuntimeError("disk full")

    thread = integratorThread(
        LiveScan("scan"),
        None,
        None,
        None,
        [],
    )
    messages: list[str] = []
    thread.writeError.connect(messages.append)
    thread._reduction_session = SimpleNamespace(finish=boom_finish)
    thread._reduction_session_key = "scan"

    thread._close_reduction_session()

    assert isinstance(thread._reduction_write_error, RuntimeError)
    assert messages
    assert "Reintegration save FAILED" in messages[0]
    assert "disk full" in messages[0]
    assert thread._reduction_session is None
    assert thread._reduction_session_key is None


def test_show_reintegration_write_error_forwards_to_wrangler_status() -> None:
    # C3 end-to-end (audit gap): the integratorThread.writeError signal is wired
    # to staticWidget._show_reintegration_write_error, which must forward the
    # message to the wrangler's showLabel status channel.  Exercised via a duck
    # self so no real QWidget is needed.
    from types import SimpleNamespace

    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    shown: list[str] = []
    duck = SimpleNamespace(
        wrangler=SimpleNamespace(showLabel=SimpleNamespace(emit=shown.append)))
    staticWidget._show_reintegration_write_error(
        duck, "Reintegration save FAILED — output .nxs may be incomplete: disk full")
    assert shown == [
        "Reintegration save FAILED — output .nxs may be incomplete: disk full"]

    # A missing/odd wrangler must be swallowed (status is best-effort), not raise.
    staticWidget._show_reintegration_write_error(
        SimpleNamespace(wrangler=None), "ignored")


# ---------------------------------------------------------------------------
# Mask-incompatibility handling: a mask that doesn't fit the image is ignored
# (returns None) with a warning, not a hard error that aborts the scan.
# ---------------------------------------------------------------------------

def test_flat_mask_out_of_bounds_is_ignored(caplog):
    """Flat-index mask with indices past the image size → ignored (None)."""
    shape = (4, 5)  # 20 pixels
    bad = np.array([0, 1, 999], dtype=np.int64)  # 999 out of range
    with caplog.at_level("WARNING"):
        out = reduction_adapters._flat_mask_as_bool(bad, shape)
    assert out is None
    assert any("out of bounds" in r.message for r in caplog.records)


def test_maskspec_out_of_bounds_is_ignored(caplog):
    """BUG-2: a MaskSpec whose flat indices don't fit the image makes
    MaskSpec.to_bool raise; _flat_mask_as_bool must ignore it (None) rather
    than let the ValueError tear down the run thread."""
    shape = (4, 5)  # 20 pixels
    bad = MaskSpec(np.array([0, 1, 999], dtype=np.int64))  # 999 out of range
    with caplog.at_level("WARNING"):
        out = reduction_adapters._flat_mask_as_bool(bad, shape)
    assert out is None
    assert any("Ignoring mask" in r.message for r in caplog.records)


def test_2d_mask_shape_mismatch_is_ignored():
    shape = (4, 5)
    bad = np.ones((4, 6), dtype=bool)  # wrong width
    assert reduction_adapters._flat_mask_as_bool(bad, shape) is None


def test_bool_mask_length_mismatch_is_ignored():
    shape = (4, 5)
    bad = np.ones(19, dtype=bool)  # 19 != 20
    assert reduction_adapters._flat_mask_as_bool(bad, shape) is None


def test_valid_flat_mask_still_applied():
    shape = (2, 3)  # 6 pixels
    good = np.array([0, 5], dtype=np.int64)
    out = reduction_adapters._flat_mask_as_bool(good, shape)
    assert out is not None
    assert out.shape == shape
    assert out.ravel()[0] and out.ravel()[5]
    assert out.sum() == 2


def test_open_live_reduction_session_retention_policy() -> None:
    """S2: streaming sink-driven sessions disable product retention (the sink
    owns the per-frame data; retaining every FrameReduction was ~14 GB on a
    10k-frame 2D batch).  Chunked sessions KEEP retention — their callers
    read results back via reduce_live_frames(session=...) -> session.frames
    (serial live, reintegration, GI scouts)."""
    from xrd_tools.reduction import MemorySink
    from xdart.modules.reduction import open_live_reduction_session

    frame = LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    plan = ReductionPlan(integration_2d=None)

    streaming = open_live_reduction_session(
        [frame], plan, scan_name="s", sink=MemorySink(), execution="streaming")
    assert streaming.retain_products is False

    chunked = open_live_reduction_session([frame], plan, scan_name="s")
    assert chunked.retain_products is True

    # Streaming WITHOUT a sink has no other way to hand results back: retain.
    streaming_no_sink = open_live_reduction_session(
        [frame], plan, scan_name="s", execution="streaming")
    assert streaming_no_sink.retain_products is True


def test_open_live_scan_session_is_graceful() -> None:
    """B-1 regression: the GUI streaming live/batch WRITE path
    (open_live_scan_session -> ScanSession) must run GRACEFUL, not the headless
    loud default — otherwise a single degraded frame (dead monitor /
    MissingNormalizationError, or all-dummy 2D / GIAllDummyError) raises mid-stream,
    re-raises at finish(), and the GUI reports "Save FAILED" + halts, aborting the
    whole-scan save even though the good frames are persisted.  ScanSession forwards
    its strict policy to the internal ReductionSession; the adapter must opt into
    graceful() like its three siblings (reduce_live_frame / reduce_live_frames /
    open_live_reduction_session)."""
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import StrictPolicy

    frame = LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    plan = ReductionPlan(integration_2d=None)
    sess = open_live_scan_session([frame], plan, scan_name="s")
    assert sess._session.strict == StrictPolicy.graceful()
    assert sess._session.strict != StrictPolicy.loud()
    assert sess._session.strict.missing_normalization is False
    assert sess._session.strict.gi_all_dummy is False


def test_persistent_session_does_not_accumulate_products(monkeypatch) -> None:
    """S2 (serial flavor): the true-live per-frame path reuses ONE chunked
    session for the whole watch run; reduce_live_frames must release each
    call's harvested reductions so the session stays O(chunk), not O(scan)
    (full 2D arrays otherwise accumulate for hours)."""
    from xdart.modules.reduction import open_live_reduction_session

    monkeypatch.setattr(
        reduction_adapters, "run_reduction", None)  # must not be used here

    def _mk(idx):
        return LiveFrame(idx=idx, map_raw=np.full((2, 2), float(idx + 1)),
                         poni=_poni())

    plan = ReductionPlan(integration_2d=None)
    session = open_live_reduction_session([_mk(0)], plan, scan_name="s")

    import xrd_tools.reduction.core as ssrl_core

    def _fake_1d(image, ai, **kw):
        v = float(np.sum(image))
        return IntegrationResult1D(
            radial=np.array([0.0, 1.0]),
            intensity=np.array([v, v + 1.0]), sigma=None, unit="q_A^-1")

    monkeypatch.setattr(ssrl_core, "integrate_1d", _fake_1d)

    for idx in range(3):                       # three per-frame calls
        out = reduce_live_frames([_mk(idx)], plan, scan_name="s",
                                 session=session)
        assert len(out) == 1 and out[0].int_1d is not None
        assert session.frames == {}            # released after each harvest


# ---------------------------------------------------------------------------
# ThresholdSaturationConfig -> plan (reintegrate pixel-rejection policy)
# ---------------------------------------------------------------------------

def test_apply_threshold_saturation_to_plan():
    """The wrangler's Intensity-Threshold / Mask-Saturated policy maps onto the
    headless ReductionPlan fields the reducer already applies after load_image."""
    from xdart.modules.reduction import (
        ThresholdSaturationConfig, apply_threshold_saturation_to_plan)

    base = ReductionPlan()

    # None cfg → identity preserved (so a cached session can be reused).
    assert apply_threshold_saturation_to_plan(base, None) is base

    # Threshold ON + Mask Saturated → all three fields set; fresh object.
    cfg = ThresholdSaturationConfig(apply_threshold=True, threshold_min=5,
                                    threshold_max=100, mask_saturation=True)
    p = apply_threshold_saturation_to_plan(base, cfg)
    assert p is not base
    assert p.threshold_min == 5 and p.threshold_max == 100
    assert p.mask_saturation is True

    # Threshold OFF → band collapses to None (parity with _apply_threshold_inline
    # no-op) even if min/max are set; saturation still applies independently.
    cfg2 = ThresholdSaturationConfig(apply_threshold=False, threshold_min=5,
                                     threshold_max=100, mask_saturation=True)
    p2 = apply_threshold_saturation_to_plan(base, cfg2)
    assert p2.threshold_min is None and p2.threshold_max is None
    assert p2.mask_saturation is True

    # All-off cfg matches the plan defaults → identity preserved (session reuse).
    assert apply_threshold_saturation_to_plan(
        base, ThresholdSaturationConfig()) is base
