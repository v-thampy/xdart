"""Tests for xdart's boundary into ssrl_xrd_tools.reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from threading import RLock
from types import SimpleNamespace

from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.core.containers import PONI
from ssrl_xrd_tools.reduction import (
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
    dispatch_live_frame_reduction,
    frame_from_live_frame,
    plan_from_live_scan,
    reduce_live_frame,
    reduce_live_frames,
    scan_from_live_scan,
)




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


class _BorrowPool:
    class _Borrowed:
        def __init__(self, ai):
            self.ai = ai

        def __enter__(self):
            return self.ai

        def __exit__(self, *args):
            return False

    def __init__(self):
        self.ai = _FakeIntegrator()

    def borrow(self):
        return self._Borrowed(self.ai)


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
    import ssrl_xrd_tools.io.nexus as nexus_io

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
    import ssrl_xrd_tools.io.nexus as nexus_io

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
    import ssrl_xrd_tools.io.nexus as nexus_io

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
    np.testing.assert_array_equal(
        frame.mask,
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
        mg_args={"wavelength": 1e-10},
        data_file="scan42.nxs",
    )

    scan = scan_from_live_scan(scan)

    assert scan.name == "scan42"
    assert [f.index for f in scan.frames] == [1, 2]
    assert scan.poni == _poni()
    assert scan.wavelength == 1.0
    np.testing.assert_allclose(scan.motors["th"], [1.0, 2.0])
    assert scan.output_path.name == "scan42.nxs"


def test_scan_from_live_scan_fills_frame_metadata_from_scan_data() -> None:
    frame = LiveFrame(idx=5, map_raw=np.ones((2, 2)), poni=_poni())
    scan = LiveScan(
        "scan",
        frames=[frame],
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


def test_standard_plan_cache_returns_headless_plan_for_gi_scan() -> None:
    from xdart.modules.reduction import StandardPlanCache

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


def test_dispatch_gi_frame_uses_headless_reduction_when_plan_present(monkeypatch) -> None:
    calls = []

    def fake_reduce(frame, plan, *, scan_name, global_mask, integrator):
        calls.append((frame.idx, plan, scan_name, global_mask, integrator))
        frame.int_1d = _r1d()
        frame.int_2d = _r2d()
        return frame

    monkeypatch.setattr(reduction_adapters, "reduce_live_frame", fake_reduce)
    frame = LiveFrame(idx=4, map_raw=np.ones((2, 2)), gi=True, scan_info={"th": 0.2})
    scan = LiveScan("scan", frames=[frame], gi=True, incidence_motor="th")

    dispatch_live_frame_reduction(
        frame,
        scan,
        standard_plan=ReductionPlan(gi=reduction_adapters.GIMode(incidence_motor="th")),
        integrator="ai",
        global_mask=np.array([0]),
    )

    assert calls and calls[0][0] == 4


def test_dispatch_requires_headless_plan() -> None:
    frame = LiveFrame(idx=4, map_raw=np.ones((2, 2)))
    scan = LiveScan("scan", frames=[frame])

    with pytest.raises(ValueError, match="ReductionPlan"):
        dispatch_live_frame_reduction(
            frame,
            scan,
            standard_plan=None,
            integrator="ai",
            global_mask=None,
        )


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

    def fake_run_reduction(plan_arg, scan_arg):
        assert plan_arg.integration_1d.monitor_key == "i0"
        assert scan_arg.name == "scan"
        assert scan_arg.integrator == "ai"
        np.testing.assert_array_equal(scan_arg.frames[0].background, np.ones((2, 2)))
        np.testing.assert_array_equal(
            scan_arg.frames[0].mask,
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


def test_nexus_worker_standard_path_calls_headless_reduction(monkeypatch) -> None:
    from xdart.gui.tabs.static_scan.wranglers import nexus_wrangler_thread

    calls = []

    def fake_reduce(frame, plan, *, scan_name, global_mask, integrator):
        calls.append((frame.idx, plan, scan_name, global_mask, integrator))
        frame.int_1d = _r1d()
        frame.int_2d = _r2d()
        return frame

    # dispatch_live_frame_reduction calls reduce_live_frame from its defining
    # module (xdart.modules.reduction).  Patch the canonical name; the
    # symbol is no longer re-imported into the wrangler-thread modules.
    monkeypatch.setattr(reduction_adapters, "reduce_live_frame", fake_reduce)

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

    frame = nexus_wrangler_thread.nexusThread._integrate_one(
        worker,
        scan,
        _BorrowPool(),
        ReductionPlan(
            integration_1d=Integration1DPlan(),
            integration_2d=Integration2DPlan(),
        ),
        5,
        np.ones((2, 2)),
        {"i0": 1.0},
    )

    assert frame.idx == 5
    assert frame.int_1d is not None
    assert frame.int_2d is not None
    assert calls and calls[0][0] == 5
    assert worker._xye_buffer[0][0] == 5


def test_spec_sequential_standard_path_calls_headless_reduction(monkeypatch) -> None:
    from xdart.gui.tabs.static_scan.wranglers import image_wrangler_thread

    calls = []

    def fake_reduce(frame, plan, *, scan_name, global_mask, integrator):
        calls.append((frame.idx, plan, scan_name, global_mask, integrator))
        frame.int_1d = _r1d()
        frame.int_2d = _r2d()
        return frame

    monkeypatch.setattr(reduction_adapters, "reduce_live_frame", fake_reduce)

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
    assert worker._xye_buffer[0][0] == 1


def test_single_frame_reintegration_uses_headless_reduction(monkeypatch) -> None:
    from xdart.gui.tabs.static_scan.scan_threads import integratorThread
    from xdart.modules.frame_publication import PublicationStore

    calls = []

    def fake_run_reduction(plan_arg, scan_arg):
        calls.append((plan_arg, scan_arg))
        return ReductionResult(
            scan_name=scan_arg.name,
            frames={
                0: FrameReduction(0, result_1d=_r1d(), result_2d=None),
            },
            n_processed=1,
        )

    monkeypatch.setattr(reduction_adapters, "run_reduction", fake_run_reduction)

    frame = LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    scan = LiveScan(
        "scan",
        frames=[frame],
        bai_1d_args={"monitor": "i0"},
    )
    scan._cached_integrator = _FakeIntegrator()
    scan.global_mask = np.array([3])

    data_1d = {}
    store = PublicationStore()
    thread = integratorThread(
        scan,
        None,
        None,
        None,
        [0],
        data_1d,
        {},
        publication_store=store,
    )
    thread.bai_1d_SI()

    assert calls
    assert calls[0][0].integration_1d.monitor_key == "i0"
    assert data_1d[0].int_1d is not None
    pub = store.get(0)
    assert pub is not None
    np.testing.assert_allclose(pub.view.intensity_1d, data_1d[0].int_1d.intensity)


def test_single_frame_2d_reintegration_refreshes_1d(monkeypatch) -> None:
    from xdart.gui.tabs.static_scan.scan_threads import integratorThread
    from xdart.modules.frame_publication import PublicationStore

    calls = []

    def fake_reduce(frame, plan, *, scan_name, global_mask, integrator):
        calls.append(plan)
        frame.int_1d = _r1d()
        frame.int_2d = _r2d()
        return frame

    monkeypatch.setattr(reduction_adapters, "reduce_live_frame", fake_reduce)

    frame = LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    scan = LiveScan("scan", frames=[frame])
    scan._cached_integrator = _FakeIntegrator()
    data_1d = {}
    data_2d = {}
    store = PublicationStore()
    thread = integratorThread(
        scan,
        None,
        None,
        None,
        [0],
        data_1d,
        data_2d,
        publication_store=store,
    )

    thread.bai_2d_SI()

    assert calls
    assert calls[0].integration_1d is not None
    assert calls[0].integration_2d is not None
    assert data_1d[0].int_1d is not None
    assert data_2d[0]["int_2d"] is not None
    pub = store.get(0)
    assert pub is not None
    np.testing.assert_allclose(pub.view.intensity_1d, data_1d[0].int_1d.intensity)
    np.testing.assert_allclose(pub.view.intensity_2d, data_2d[0]["int_2d"].intensity.T)


def test_reintegrate_all_refreshes_publication_store(monkeypatch) -> None:
    # A1: reintegrating ("Integrate 2D") must republish into the
    # PublicationStore so the cake (payload path is preferred) shows the NEW
    # pixels, not the pre-reintegrate ones.  fake_reduce makes the 2D shape
    # depend on bai_2d_args['npt_rad'] so a changed arg is observable.
    from xdart.gui.tabs.static_scan.scan_threads import integratorThread
    from xdart.modules.frame_publication import PublicationStore

    def fake_reduce(frame, plan, *, scan_name, global_mask, integrator):
        npt = int(scan.bai_2d_args.get("npt_rad", 10))
        frame.int_1d = _r1d()
        frame.int_2d = IntegrationResult2D(
            radial=np.linspace(0.0, 1.0, npt),
            azimuthal=np.linspace(-90.0, 90.0, 3),
            intensity=np.ones((npt, 3)),
            unit="q_A^-1", azimuthal_unit="chi_deg",
        )
        return frame

    monkeypatch.setattr(reduction_adapters, "reduce_live_frame", fake_reduce)

    frame = LiveFrame(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    scan = LiveScan("scan", frames=[frame])
    scan._cached_integrator = _FakeIntegrator()
    scan.skip_2d = False
    scan.bai_2d_args = {"npt_rad": 10, "gi_mode_2d": "qip_qoop"}
    store = PublicationStore()

    thread = integratorThread(
        scan, None, None, None, [0], {}, {},
        publication_store=store,
    )

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
