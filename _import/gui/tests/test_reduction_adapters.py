"""Tests for xdart's boundary into ssrl_xrd_tools.reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.core.containers import PONI
from ssrl_xrd_tools.reduction import FrameReduction, ReductionPlan, ReductionResult

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
    assert frame.normalization_factor == 5.0


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
            "monitor": "i0",
            "chi_offset": 90.0,
        },
        bai_2d_args={
            "npt_rad": 20,
            "npt_azim": 30,
            "method": "csr",
            "azimuth_range": (-10.0, 10.0),
        },
        global_mask=np.array([0, 3]),
    )

    plan = plan_from_ewald_sphere(sphere, chunk_size=4)

    assert plan.integrate_1d
    assert plan.integrate_2d
    assert plan.npt_1d == 123
    assert plan.npt_rad_2d == 20
    assert plan.npt_azim_2d == 30
    assert plan.unit == "2th_deg"
    assert plan.method_1d == "BBox"
    assert plan.method_2d == "csr"
    assert plan.radial_range == (1.0, 2.0)
    assert plan.azimuth_range == (-10.0, 10.0)
    assert plan.monitor_key == "i0"
    assert plan.azimuth_offset == 90.0
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
    plan = ReductionPlan(integrate_1d=True, integrate_2d=True, monitor_key="i0")

    def fake_run_reduction(plan_arg, scan_arg):
        assert plan_arg.monitor_key == "i0"
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
