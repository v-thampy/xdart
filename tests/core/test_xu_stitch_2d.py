"""xrayutilities Stitch 2D (STITCH-2D-20260930 item 2): the (radial, χ) map from the same pixels and weights as
I(q), with χ in pyFAI's convention (checked against pyFAI's own χ first, maintainer's list item 2)."""
import os
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
from xrd_tools.integrate.xu_stitch import (
    XuPowderQProvider, resolve_xu_stitch_effective_geometry, run_xu_hist_stitch_1d,
)
from tests.core.test_xu_stitch import _OneFrameSource, _receipt

SHAPE = (195, 1475)


def _run(receipt, **kwargs):
    return run_xu_hist_stitch_1d(
        receipt, _OneFrameSource(monitor=1.0), frame_indices=(7,), q_min_A_inverse=1.0,
        q_max_A_inverse=5.2, npt=64, monitor_key="monitor", max_frame_bytes=4 * 1024 * 1024, **kwargs)


def test_the_map_is_i_of_q_spread_over_chi_and_leaves_i_of_q_alone(tmp_path):
    receipt = _receipt(tmp_path)
    alone = _run(receipt)
    both = _run(receipt, map_grid=(64, 36, (-180.0, 180.0)))
    # I(q) beside the map is the 1-D Stitch's, bit for bit
    assert np.array_equal(both.payload.intensity, alone.payload.intensity, equal_nan=True)
    assert np.array_equal(both.diagnostics.coverage, alone.diagnostics.coverage)
    assert np.array_equal(both.diagnostics.normalization, alone.diagnostics.normalization)
    stitched, diagnostics = both.map
    assert stitched.intensity.shape == diagnostics.coverage.shape == (64, 36)  # (radial, azimuthal)
    assert (stitched.unit, stitched.azimuthal_unit) == ("q_A^-1", "chi_deg")
    np.testing.assert_array_equal(stitched.radial, alone.payload.radial)
    # every pixel of I(q) is in one χ bin of the map: the map summed over χ is I(q)'s histogram
    assert np.array_equal(diagnostics.coverage.sum(axis=1), alone.diagnostics.coverage)
    np.testing.assert_allclose(diagnostics.normalization.sum(axis=1), alone.diagnostics.normalization, rtol=1e-12)
    signal = np.where(diagnostics.normalization > 0, stitched.intensity * diagnostics.normalization, 0.0)
    expected = np.where(alone.diagnostics.normalization > 0,
                        alone.payload.intensity * alone.diagnostics.normalization, 0.0)
    np.testing.assert_allclose(signal.sum(axis=1), expected, rtol=1e-12, atol=1e-300)
    empty = diagnostics.normalization == 0  # as the 1-D Stitch: an empty bin is NaN
    assert empty.any() and np.isnan(stitched.intensity[empty]).all()
    assert np.isfinite(stitched.intensity[~empty]).all()


def _stitching_root() -> Path:
    configured = os.environ.get("XDART_TEST_DATA")
    if not configured or not (Path(configured) / "stitching").is_dir():
        pytest.skip("XDART_TEST_DATA stitching data is unavailable")
    return Path(configured) / "stitching"


@pytest.mark.parametrize(("delta", "nu"), [(0.0, 25.0), (25.0, 0.0)])
def test_the_maps_chi_is_pyfais_on_the_scan14_calibrations(tmp_path, delta, nu):
    """pyFAI's χ (``center_array('chi_rad')``, its discontinuity at ±180°) at single-circle poses, where the
    two calibrations agree (a mirrored or quarter-turned χ is off by 90° or more)."""
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.integrate.multi import create_multigeometry_integrators_from_geometry
    root = _stitching_root()
    receipt = _receipt(tmp_path)
    diff = Diffractometer.from_pyfai_goniometer(
        root / "data/calibration/MG_gonio_object_5images.json",
        source_motors={"del_angle": "del", "nu_angle": "nu"})
    integrator, = create_multigeometry_integrators_from_geometry(
        diff, {"del": np.array([delta]), "nu": np.array([nu])})
    integrator.setChiDiscAtPi()
    expected = np.degrees(np.asarray(integrator.center_array(SHAPE, unit="chi_rad")))
    asset = receipt.projection.value
    with xu_runtime_session() as session:
        geometry = resolve_xu_stitch_effective_geometry(receipt, session)
        lease = XuPowderQProvider(geometry, asset["acquisition"]).frame(nu=nu, del_=delta)
        try:
            actual = lease.chi()
            qx, _qy, qz = (np.array(value) for value in lease._values)
        finally:
            lease.release()
    wrap = lambda values: (values + 180.0) % 360.0 - 180.0  # noqa: E731
    assert actual.shape == expected.shape == SHAPE
    assert float(np.max(np.abs(wrap(actual - expected)))) < 1.5
    mirrored = np.degrees(np.arctan2(qx, qz))
    assert delta == 0.0 or float(np.median(np.abs(wrap(mirrored - expected)))) > 90.0
