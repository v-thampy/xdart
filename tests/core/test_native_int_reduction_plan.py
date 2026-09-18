from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.session.readiness import (
    build_native_int_reduction_plan_from_args,
    build_native_int_reduction_plan_from_scan,
)


@pytest.mark.parametrize("mode_1d,mode_2d", (
    ("q_total", "qip_qoop"), ("q_ip", "q_chi"),
    ("q_oop", "exit_angles"), ("exit_angle", "qip_qoop"),
    ("chi_gi", "q_chi"),
))
@pytest.mark.parametrize("dimensions", ((True, True), (True, False), (False, True)))
def test_gi_resource_modes_match_actual_required_results(mode_1d, mode_2d, dimensions):
    from types import SimpleNamespace
    from xrd_tools.session import required_result_modes
    from xrd_tools.session.policy import requirements_from

    plan = build_native_int_reduction_plan_from_args(
        {"gi_mode_1d": mode_1d}, {"gi_mode_2d": mode_2d},
        gi_enabled=True, gi_incident_angle=0.2,
        integrate_1d=dimensions[0], integrate_2d=dimensions[1],
    )
    requirements = requirements_from(SimpleNamespace(
        frame_shape=(3072, 3072), dtype=np.dtype("uint16"),
    ), plan)
    modes = required_result_modes(plan)
    assert (requirements.modes_1d, requirements.modes_2d) == (
        sum(mode.kind == "1d" for mode in modes),
        sum(mode.kind == "2d" for mode in modes),
    )


def test_selected_gi_modes_fit_the_windows_eight_gib_run_budget():
    from types import SimpleNamespace
    from xrd_tools.session.policy import requirements_from, resolve_session_policy

    plan = build_native_int_reduction_plan_from_args(
        {"npt": 1000}, {"npt_rad": 500, "npt_azim": 500},
        gi_enabled=True, gi_incident_angle=0.2,
    )
    requirements = requirements_from(SimpleNamespace(
        frame_shape=(3072, 3072), dtype=np.dtype("uint16"),
    ), plan)
    allocation = resolve_session_policy(
        requirements, envelope_bytes=8 * 1024**3,
        requested_workers=4,
        requests={"workers": 4, "reduction_inflight": 8, "staging_items": 64},
        env={},
    ).allocation
    assert allocation.workers == 4
    assert allocation.staging_items == 64
    assert min(allocation.record_items, allocation.publication_items,
               allocation.record_heavy_items, allocation.publication_heavy_items,
               allocation.thumbnail_items) >= 16
    assert allocation.assigned_bytes <= allocation.envelope_bytes


def _plain(value):
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return tuple(_plain(item) for item in value)
    return value


def _plan_snapshot(plan):
    def _snap(obj, attrs):
        if obj is None:
            return None
        return {name: _plain(getattr(obj, name)) for name in attrs}

    def _mask(mask):
        if mask is None:
            return None
        values = getattr(mask, "values", mask)
        array = np.asarray(values)
        return {
            "kind": type(mask).__name__,
            "shape": tuple(array.shape),
            "dtype": str(array.dtype),
            "values": (
                tuple(array.ravel().tolist()) if array.size <= 20 else None
            ),
            "true_count": (
                int(array.astype(bool, copy=False).sum())
                if array.dtype == bool
                else None
            ),
        }

    return {
        "integration_1d": _snap(plan.integration_1d, (
            "npt",
            "npt_rad",
            "unit",
            "method",
            "radial_range",
            "azimuth_range",
            "monitor_key",
            "error_model",
            "polarization_factor",
            "azimuth_offset",
            "extra",
        )),
        "integration_2d": _snap(plan.integration_2d, (
            "npt_rad",
            "npt_azim",
            "unit",
            "method",
            "radial_range",
            "azimuth_range",
            "azimuth_offset",
            "monitor_key",
            "error_model",
            "polarization_factor",
            "extra",
        )),
        "gi": _snap(plan.gi, (
            "incident_angle",
            "incidence_motor",
            "tilt_angle",
            "sample_orientation",
            "method",
            "mode_1d",
            "mode_2d",
            "npt_oop",
        )),
        "mask": _mask(plan.mask),
        "threshold_min": _plain(plan.threshold_min),
        "threshold_max": _plain(plan.threshold_max),
        "mask_saturation": _plain(plan.mask_saturation),
    }


def test_args_builder_preserves_monitor_and_mask_contract() -> None:
    args_1d = {
        "unit": "q_A^-1",
        "method": "csr",
        "numpoints": 250,
        "radial_range": (0.2, 4.4),
        "azimuth_range": (-30.0, 30.0),
        "monitor": "I0",
        "normalization_factor": 5.0,
        "error_model": "poisson",
        "polarization_factor": 0.95,
    }
    args_2d = {
        "unit": "q_A^-1",
        "method": "csr",
        "npt_rad": 80,
        "npt_azim": 90,
        "radial_range": (0.1, 5.0),
        "azimuth_range": (-90.0, 90.0),
        "chi_offset": 2.5,
        "monitor": "mon",
        "normalization_factor": 2.0,
        "error_model": "azimuthal",
        "polarization_factor": 0.9,
    }

    detector_mask = np.array([1, 4])
    actual = build_native_int_reduction_plan_from_args(
        args_1d,
        args_2d,
        gi_enabled=False,
        integrate_1d=True,
        integrate_2d=True,
        detector_mask=detector_mask,
        detector_shape=(2, 3),
    )

    snapshot = _plan_snapshot(actual)
    assert snapshot["integration_1d"]["monitor_key"] == "I0"
    assert snapshot["integration_2d"]["monitor_key"] == "mon"
    assert snapshot["mask"]["kind"] == "ndarray"
    assert snapshot["mask"]["shape"] == (2, 3)
    assert snapshot["mask"]["true_count"] == 2
    assert "normalization_factor" not in snapshot["integration_1d"]["extra"]
    assert "normalization_factor" not in snapshot["integration_2d"]["extra"]


def test_scan_builder_matches_current_args_builder() -> None:
    args_1d = {
        "unit": "2th_deg",
        "method": "BBox",
        "numpoints": 321,
        "radial_range": (1.0, 4.0),
        "azimuth_range": (60.0, 120.0),
        "chi_offset": 90.0,
        "error_model": "poisson",
        "polarization_factor": 0.8,
        "correctSolidAngle": False,
        "dummy": -2.0,
        "delta_dummy": 0.1,
        "safe": False,
    }
    args_2d = {
        "unit": "q_A^-1",
        "method": "csr",
        "npt_rad": 77,
        "npt_azim": 88,
        "radial_range": (0.5, 5.0),
        "azimuth_range": (-45.0, 45.0),
        "chi_offset": 12.0,
        "error_model": "azimuthal",
        "polarization_factor": 0.9,
        "correctSolidAngle": True,
        "dummy": -3.0,
        "delta_dummy": 0.2,
        "safe": True,
    }

    class FakeFrames:
        index = []

    class FakeScan:
        skip_2d = False
        gi = False
        global_mask = np.array([1, 4])
        detector_shape = (2, 3)
        frames = FakeFrames()
        bai_1d_args = dict(args_1d)
        bai_2d_args = dict(args_2d)

    from_scan = build_native_int_reduction_plan_from_scan(
        FakeScan(), integrate_1d=True, integrate_2d=True
    )
    from_args = build_native_int_reduction_plan_from_args(
        args_1d,
        args_2d,
        gi_enabled=False,
        integrate_1d=True,
        integrate_2d=True,
        detector_mask=FakeScan.global_mask,
        detector_shape=FakeScan.detector_shape,
    )

    assert _plan_snapshot(from_scan) == _plan_snapshot(from_args)
    assert from_scan.integration_1d.azimuth_offset == 90.0
    assert from_scan.integration_2d.azimuth_offset == 12.0


def test_gi_plan_defaults_orientation_to_four() -> None:
    args_plan = build_native_int_reduction_plan_from_args(
        {},
        {},
        gi_enabled=True,
        gi_incident_angle=0.1,
        integrate_2d=False,
    )
    assert args_plan.gi.sample_orientation == 4

    class FakeFrames:
        index = []

    class FakeScan:
        skip_2d = True
        gi = True
        _cached_fiber_integrator_angle = 0.1
        incidence_motor = None
        global_mask = None
        detector_shape = (2, 3)
        frames = FakeFrames()
        bai_1d_args = {}
        bai_2d_args = {}
        gi_config = {}

    scan_plan = build_native_int_reduction_plan_from_scan(
        FakeScan(), integrate_1d=True, integrate_2d=False
    )
    assert scan_plan.gi.sample_orientation == 4
