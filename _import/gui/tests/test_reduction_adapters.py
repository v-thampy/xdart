"""Tests for xdart's boundary into ssrl_xrd_tools.reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ssrl_xrd_tools.core.containers import PONI

from xdart.modules.ewald import EwaldArch, EwaldSphere
from xdart.modules.reduction import (
    frame_from_ewald_arch,
    plan_from_ewald_sphere,
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


def test_frame_from_ewald_arch_maps_simple_fields(tmp_path) -> None:
    arch = EwaldArch(
        idx=4,
        map_raw=np.arange(4).reshape(2, 2),
        poni=_poni(),
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
            "npt": 123,
            "unit": "2th_deg",
            "method": "BBox",
            "radial_range": (1.0, 2.0),
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
