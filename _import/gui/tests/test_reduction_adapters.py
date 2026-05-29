"""Tests for xdart's boundary into ssrl_xrd_tools.reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd
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
    scan_from_live_scan,
)


# Convenience aliases used throughout the tests below to keep wording short.
# These exist only inside the test module — the production aliases on
# xdart.modules.{ewald,reduction} were dropped in the rename release.
EwaldArch = LiveFrame
EwaldSphere = LiveScan
ArchSeries = LiveFrameSeries
frame_from_ewald_arch = frame_from_live_frame
scan_from_ewald_sphere = scan_from_live_scan
plan_from_ewald_sphere = plan_from_live_scan
reduce_ewald_arch = reduce_live_frame
dispatch_arch_reduction = dispatch_live_frame_reduction


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
        "xdart.modules.ewald.arch.LiveFrame",
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


def test_frame_from_ewald_arch_maps_simple_fields(tmp_path) -> None:
    frame = EwaldArch(
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

    frame = frame_from_ewald_arch(frame)

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


def test_frame_from_ewald_arch_can_skip_large_background() -> None:
    frame = EwaldArch(
        idx=4,
        map_raw=np.arange(4).reshape(2, 2),
        poni=_poni(),
        bg_raw=np.ones((2, 2)),
    )

    frame = frame_from_ewald_arch(
        frame,
        include_image=False,
        include_background=False,
    )

    assert frame.image is None
    assert frame.background is None


def test_scan_from_ewald_sphere_uses_scan_frame_names() -> None:
    a2 = EwaldArch(idx=2, map_raw=np.ones((2, 2)), poni=_poni(),
                   scan_info={"th": 2.0})
    a1 = EwaldArch(idx=1, map_raw=np.zeros((2, 2)), poni=_poni(),
                   scan_info={"th": 1.0})
    scan = EwaldSphere(
        "scan42",
        frames=[a2, a1],
        scan_data=pd.DataFrame({"th": [1.0, 2.0]}, index=[1, 2]),
        mg_args={"wavelength": 1e-10},
        data_file="scan42.nxs",
    )

    scan = scan_from_ewald_sphere(scan)

    assert scan.name == "scan42"
    assert [f.index for f in scan.frames] == [1, 2]
    assert scan.poni == _poni()
    assert scan.wavelength == 1.0
    np.testing.assert_allclose(scan.motors["th"], [1.0, 2.0])
    assert scan.output_path.name == "scan42.nxs"


def test_plan_from_ewald_sphere_maps_integration_settings() -> None:
    frame = EwaldArch(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    scan = EwaldSphere(
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

    plan = plan_from_ewald_sphere(scan)

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


def test_plan_from_gi_sphere_requires_incident_angle() -> None:
    scan = EwaldSphere("scan", frames=[EwaldArch(idx=0, map_raw=np.ones((2, 2)))],
                         gi=True)

    try:
        plan_from_ewald_sphere(scan)
    except ValueError as exc:
        assert "gi_incident_angle" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing GI angle to fail")


def test_reduce_ewald_arch_populates_existing_arch(monkeypatch) -> None:
    frame = EwaldArch(
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

    returned = reduce_ewald_arch(
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


def test_mask_conversion_ignores_incompatible_mask() -> None:
    """A matching mask is applied; a shape-incompatible mask is ignored
    (plan.mask is None) with a warning rather than raising — reducing
    unmasked beats aborting the scan."""
    frame = EwaldArch(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    scan = EwaldSphere(
        "scan",
        frames=[frame],
        global_mask=np.array([[1, 0], [0, 1]]),
    )
    plan = plan_from_ewald_sphere(scan)
    np.testing.assert_array_equal(
        plan.mask,
        np.array([[True, False], [False, True]]),
    )

    bad = EwaldSphere(
        "scan",
        frames=[frame],
        global_mask=np.ones((3, 3)),  # wrong shape for a 2x2 image
    )
    bad_plan = plan_from_ewald_sphere(bad)
    assert bad_plan.mask is None


def test_flat_global_mask_is_preserved_until_shape_is_known() -> None:
    scan = EwaldSphere(
        "scan",
        frames=[EwaldArch(idx=0, map_raw=None, poni=_poni())],
        global_mask=np.array([0, 3]),
    )

    plan = plan_from_ewald_sphere(scan)

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
    worker._resolve_frame_mask = lambda _sphere, _img: np.array([0])

    frame = nexus_wrangler_thread.nexusThread._integrate_one(
        worker,
        scan,
        _BorrowPool(),
        None,
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
    from xdart.gui.tabs.static_scan.wranglers import spec_wrangler_thread

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
        _resolve_frame_mask=lambda _sphere, _img: np.array([0]),
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

    spec_wrangler_thread.specThread._process_one(
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

    frame = EwaldArch(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    scan = EwaldSphere(
        "scan",
        frames=[frame],
        bai_1d_args={"monitor": "i0"},
    )
    scan._cached_integrator = _FakeIntegrator()
    scan.global_mask = np.array([3])

    data_1d = {}
    thread = integratorThread(
        scan,
        None,
        None,
        None,
        [0],
        data_1d,
        {},
    )
    thread.bai_1d_SI()

    assert calls
    assert calls[0][0].integration_1d.monitor_key == "i0"
    assert data_1d[0].int_1d is not None


def test_single_frame_2d_reintegration_refreshes_1d(monkeypatch) -> None:
    from xdart.gui.tabs.static_scan.scan_threads import integratorThread

    calls = []

    def fake_reduce(frame, plan, *, scan_name, global_mask, integrator):
        calls.append(plan)
        frame.int_1d = _r1d()
        frame.int_2d = _r2d()
        return frame

    monkeypatch.setattr(reduction_adapters, "reduce_live_frame", fake_reduce)

    frame = EwaldArch(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    scan = EwaldSphere("scan", frames=[frame])
    scan._cached_integrator = _FakeIntegrator()
    data_1d = {}
    data_2d = {}
    thread = integratorThread(
        scan,
        None,
        None,
        None,
        [0],
        data_1d,
        data_2d,
    )

    thread.bai_2d_SI()

    assert calls
    assert calls[0].integration_1d is not None
    assert calls[0].integration_2d is not None
    assert data_1d[0].int_1d is not None
    assert data_2d[0]["int_2d"] is not None


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
