"""XU Stitch corrections: pyFAI's polarization per pixel, and the dark and flat identities."""
import os
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
from xrd_tools.integrate.xu_stitch import (
    XuPowderQProvider,
    _polarization_axes,
    resolve_xu_stitch_effective_geometry,
    run_xu_hist_stitch_1d,
)
from tests.core.test_xu_stitch import _OneFrameSource, _receipt

SHAPE = (195, 1475)


def _run(receipt, **kwargs):
    return run_xu_hist_stitch_1d(
        receipt, _OneFrameSource(monitor=1.0), frame_indices=(7,), q_min_A_inverse=1.0,
        q_max_A_inverse=5.2, npt=64, monitor_key="monitor", max_frame_bytes=4 * 1024 * 1024,
        **kwargs)


def test_flat_and_dark_identities_and_unusable_pixels(tmp_path):
    receipt = _receipt(tmp_path)
    plain = _run(receipt)
    occupied = plain.diagnostics.normalization > 0
    assert occupied.any()
    off = _run(receipt, polarization_factor=None, dark=None, flat=None)
    assert np.array_equal(off.payload.intensity, plain.payload.intensity, equal_nan=True)
    halved = _run(receipt, flat=np.full(SHAPE, 2.0))
    np.testing.assert_allclose(halved.payload.intensity[occupied],
                               plain.payload.intensity[occupied] / 2, rtol=1e-12)
    darkened = _run(receipt, dark=np.full(SHAPE, .25))
    # (I - I_dark) · Σ normalization = dark · count, bin by bin
    np.testing.assert_allclose(
        (plain.payload.intensity - darkened.payload.intensity)[occupied]
        * plain.diagnostics.normalization[occupied],
        .25 * plain.diagnostics.coverage[occupied], rtol=1e-9)
    flat = np.full(SHAPE, 2.0)
    flat[:, :700] = 0.0
    dark = np.zeros(SHAPE)
    dark[:20] = np.nan
    masked = _run(receipt, flat=flat, dark=dark)
    assert masked.diagnostics.coverage.sum() < plain.diagnostics.coverage.sum()
    assert np.isfinite(masked.payload.intensity[masked.diagnostics.normalization > 0]).all()


def test_polarization_weights_the_normalization(tmp_path):
    receipt = _receipt(tmp_path)
    plain = _run(receipt)
    polarized = _run(receipt, polarization_factor=.99)
    occupied = plain.diagnostics.normalization > 0
    ratio = plain.diagnostics.normalization[occupied] / polarized.diagnostics.normalization[occupied]
    assert np.all(ratio > 1)  # P < 1 everywhere off the beam
    np.testing.assert_allclose(polarized.payload.intensity[occupied],
                               plain.payload.intensity[occupied] * ratio, rtol=1e-12)
    with pytest.raises(ValueError, match="polarization factor"):
        _run(receipt, polarization_factor=1.5)


def _stitching_root() -> Path:
    configured = os.environ.get("XDART_TEST_DATA")
    if not configured or not (Path(configured) / "stitching").is_dir():
        pytest.skip("XDART_TEST_DATA stitching data is unavailable")
    return Path(configured) / "stitching"


@pytest.mark.parametrize(("delta", "nu", "centre"), [(0., 25., "horizontal"), (25., 0., "vertical")])
def test_xu_polarization_equals_pyfai_on_the_scan14_calibrations(tmp_path, delta, nu, centre):
    """A convention error would mirror the factor (0.57 against 0.995 at the centre)."""
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.integrate.multi import create_multigeometry_integrators_from_geometry
    root = _stitching_root()
    receipt = _receipt(tmp_path)
    diff = Diffractometer.from_pyfai_goniometer(
        root / "data/calibration/MG_gonio_object_5images.json",
        source_motors={"del_angle": "del", "nu_angle": "nu"})
    integrator, = create_multigeometry_integrators_from_geometry(
        diff, {"del": np.array([delta]), "nu": np.array([nu])})
    expected = np.asarray(integrator.polarization(SHAPE, factor=.99))
    asset = receipt.projection.value
    with xu_runtime_session() as session:
        geometry = resolve_xu_stitch_effective_geometry(receipt, session)
        lease = XuPowderQProvider(geometry, asset["acquisition"]).frame(nu=nu, del_=delta)
        try:
            actual = lease.polarization(.99, geometry.projection.xu_mapping_wavelength_A,
                                        *_polarization_axes(asset))
        finally:
            lease.release()
    assert actual.shape == expected.shape == SHAPE
    # The XU and MultiGeometry calibrations of scan 14 differ slightly.
    assert float(np.max(np.abs(actual - expected))) < .02
    middle = actual[97, 737]
    assert (middle < .7) if centre == "horizontal" else (middle > .9)


def test_xu_polarization_is_pyfais_formula_in_the_assets_chi_convention(tmp_path):
    """P = ½(1 + cos²2θ − f·cos2χ·sin²2θ) with the asset's χ = −atan2(qx, qz), at a mixed frame."""
    receipt = _receipt(tmp_path)
    asset = receipt.projection.value
    assert asset["xrayutilities"]["chi_validation_convention"].startswith("minus_atan2_qx_qz")
    with xu_runtime_session() as session:
        geometry = resolve_xu_stitch_effective_geometry(receipt, session)
        lease = XuPowderQProvider(geometry, asset["acquisition"]).frame(nu=15.0, del_=30.0)
        try:
            wavelength = geometry.projection.xu_mapping_wavelength_A
            actual = lease.polarization(.95, wavelength, *_polarization_axes(asset))
            qx, _qy, qz = (np.array(value) for value in lease._values)
            q = lease.q_magnitude()
        finally:
            lease.release()
    two_theta = 2 * np.arcsin(q * wavelength / (4 * np.pi))
    chi = -np.arctan2(qx, qz)
    expected = .5 * (1 + np.cos(two_theta) ** 2 - .95 * np.cos(2 * chi) * np.sin(two_theta) ** 2)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_only_a_corrected_xu_plan_changes_its_identity(tmp_path):
    from xrd_tools.analysis.module_transaction import ModuleKind, module_plan_fingerprint
    from xrd_tools.analysis.stitch_operation import XuStitchOperationPlan
    receipt = _receipt(tmp_path)
    with xu_runtime_session() as session:
        effective = resolve_xu_stitch_effective_geometry(receipt, session).projection
    off = XuStitchOperationPlan(receipt, effective, 1.0, 5.2, npt_1d=64, polarization_factor=None)
    parent = module_plan_fingerprint(ModuleKind.STITCH, (
        "stitch-operation-plan-v2-xu-hist", off.backend, off.mode, off.unit, off.method,
        off.radial_range, off.npt_1d, None, off.use_detector_mask, off.max_frame_bytes,
        receipt.fingerprint, effective.fingerprint, off.frame_screen.identity()))
    assert off.fingerprint == parent  # an uncorrected plan keeps the identity it had before
    default = XuStitchOperationPlan(receipt, effective, 1.0, 5.2, npt_1d=64)
    assert default.polarization_factor == .99 and default.fingerprint != off.fingerprint
    assert XuStitchOperationPlan(receipt, effective, 1.0, 5.2, npt_1d=64,
                                 polarization_factor=.5).fingerprint != default.fingerprint
    dark = tmp_path / "dark.npy"
    np.save(dark, np.zeros(SHAPE))
    with pytest.raises(ValueError, match="native records"):
        XuStitchOperationPlan(receipt, effective, 1.0, 5.2, npt_1d=64, dark_path=dark)
