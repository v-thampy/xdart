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
    ReductionPlan,
    ReductionResult,
)

from xdart.modules.ewald import EwaldArch, EwaldSphere
import xdart.modules.reduction as reduction_adapters
from xdart.modules.reduction import (
    frame_from_ewald_arch,
    plan_from_ewald_sphere,
    reduce_ewald_arch,
    scan_from_ewald_sphere,
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


def test_frame_from_ewald_arch_maps_simple_fields(tmp_path) -> None:
    arch = EwaldArch(
        idx=4,
        map_raw=np.arange(4).reshape(2, 2),
        poni=_poni(),
        bg_raw=np.ones((2, 2)),
        mask=np.array([1, 3]),
        scan_info={"th": 1.2, "i0": 99.0},
    )
    arch.source_file = "frame_0004.tif"
    arch.source_frame_idx = 2
    arch._source_root = str(tmp_path)
    arch.map_norm = 5.0

    frame = frame_from_ewald_arch(arch)

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


def test_scan_from_ewald_sphere_uses_scan_frame_names() -> None:
    a2 = EwaldArch(idx=2, map_raw=np.ones((2, 2)), poni=_poni(),
                   scan_info={"th": 2.0})
    a1 = EwaldArch(idx=1, map_raw=np.zeros((2, 2)), poni=_poni(),
                   scan_info={"th": 1.0})
    sphere = EwaldSphere(
        "scan42",
        arches=[a2, a1],
        scan_data=pd.DataFrame({"th": [1.0, 2.0]}, index=[1, 2]),
        mg_args={"wavelength": 1e-10},
        data_file="scan42.nxs",
    )

    scan = scan_from_ewald_sphere(sphere)

    assert scan.name == "scan42"
    assert [f.index for f in scan.frames] == [1, 2]
    assert scan.poni == _poni()
    assert scan.wavelength == 1.0
    np.testing.assert_allclose(scan.motors["th"], [1.0, 2.0])
    assert scan.output_path.name == "scan42.nxs"


def test_plan_from_ewald_sphere_maps_integration_settings() -> None:
    arch = EwaldArch(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    sphere = EwaldSphere(
        "scan",
        arches=[arch],
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

    plan = plan_from_ewald_sphere(sphere, chunk_size=4)

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
    assert plan.chunk_size == 4
    np.testing.assert_array_equal(
        plan.mask,
        np.array([[True, False], [False, True]]),
    )


def test_plan_from_gi_sphere_requires_incident_angle() -> None:
    sphere = EwaldSphere("scan", arches=[EwaldArch(idx=0, map_raw=np.ones((2, 2)))],
                         gi=True)

    try:
        plan_from_ewald_sphere(sphere)
    except ValueError as exc:
        assert "gi_incident_angle" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing GI angle to fail")


def test_reduce_ewald_arch_populates_existing_arch(monkeypatch) -> None:
    arch = EwaldArch(
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
        arch,
        plan,
        scan_name="scan",
        global_mask=np.array([3]),
        integrator="ai",
    )

    assert returned is arch
    assert arch.int_1d is not None
    assert arch.int_2d is not None
    assert arch.map_norm == 33.0


def test_mask_conversion_is_strict() -> None:
    arch = EwaldArch(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    sphere = EwaldSphere(
        "scan",
        arches=[arch],
        global_mask=np.array([[1, 0], [0, 1]]),
    )
    plan = plan_from_ewald_sphere(sphere)
    np.testing.assert_array_equal(
        plan.mask,
        np.array([[True, False], [False, True]]),
    )

    bad = EwaldSphere(
        "scan",
        arches=[arch],
        global_mask=np.ones((3, 3)),
    )
    try:
        plan_from_ewald_sphere(bad)
    except ValueError as exc:
        assert "mask shape" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected mask shape mismatch")


def test_nexus_worker_standard_path_calls_headless_reduction(monkeypatch) -> None:
    from xdart.gui.tabs.static_scan.wranglers import nexus_wrangler_thread

    calls = []

    def fake_reduce(arch, plan, *, scan_name, global_mask, integrator):
        calls.append((arch.idx, plan, scan_name, global_mask, integrator))
        arch.int_1d = _r1d()
        arch.int_2d = _r2d()
        return arch

    monkeypatch.setattr(nexus_wrangler_thread, "reduce_ewald_arch", fake_reduce)

    sphere = SimpleNamespace(
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
    worker._resolve_arch_mask = lambda _sphere, _img: np.array([0])

    arch = nexus_wrangler_thread.nexusThread._integrate_one(
        worker,
        sphere,
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

    assert arch.idx == 5
    assert arch.int_1d is not None
    assert arch.int_2d is not None
    assert calls and calls[0][0] == 5
    assert worker._xye_buffer[0][0] == 5


def test_spec_sequential_standard_path_calls_headless_reduction(monkeypatch) -> None:
    from xdart.gui.tabs.static_scan.wranglers import spec_wrangler_thread

    calls = []

    def fake_reduce(arch, plan, *, scan_name, global_mask, integrator):
        calls.append((arch.idx, plan, scan_name, global_mask, integrator))
        arch.int_1d = _r1d()
        arch.int_2d = _r2d()
        return arch

    monkeypatch.setattr(spec_wrangler_thread, "reduce_ewald_arch", fake_reduce)

    class _Signal:
        def emit(self, *args):
            return None

    sphere = SimpleNamespace(
        name="scan",
        data_file="scan_out.nxs",
        skip_2d=False,
        _cached_integrator=_FakeIntegrator(),
        bai_1d_args={},
        bai_2d_args={},
    )
    worker = SimpleNamespace(
        showLabel=_Signal(),
        _middle_truncate=lambda text: text,
        _apply_threshold_inline=lambda img: img,
        _resolve_arch_mask=lambda _sphere, _img: np.array([0]),
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
    )

    spec_wrangler_thread.specThread._process_one(
        worker,
        sphere,
        "frame_0001.tif",
        1,
        np.ones((2, 2)),
        {"i0": 1.0},
        0,
    )

    assert calls and calls[0][0] == 1
    assert worker._xye_buffer[0][0] == 1


def test_single_frame_reintegration_uses_headless_reduction(monkeypatch) -> None:
    from xdart.gui.tabs.static_scan.sphere_threads import integratorThread

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

    arch = EwaldArch(idx=0, map_raw=np.ones((2, 2)), poni=_poni())
    sphere = EwaldSphere(
        "scan",
        arches=[arch],
        bai_1d_args={"monitor": "i0"},
    )
    sphere._cached_integrator = _FakeIntegrator()
    sphere.global_mask = np.array([3])

    data_1d = {}
    thread = integratorThread(
        sphere,
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
