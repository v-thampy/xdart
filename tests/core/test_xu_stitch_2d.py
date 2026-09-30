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
from tests.core.test_xu_extended_arm import transferred  # noqa: F401 (the extended arm, transferred)

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


def test_a_native_xrayutilities_stitch_2d_is_an_attested_int_2d_record(tmp_path):
    """The operation writes the map and I(q) as an Int 2D record (version 7, attested as a Stitch 1D is);
    I(q) beside the map is the native xrayutilities Stitch 1D's."""
    from dataclasses import replace
    from xrd_tools.analysis.module_transaction import ModuleDisposition, ModuleOutputRequest
    from xrd_tools.analysis.stitch_operation import prepare_stitch_operation, run_stitch_operation
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, read_analysis_artifact
    from xrd_tools.io.output_path import resolve_finite_output_target
    from tests.core.test_stitch_operation import _xu_prepared

    original = _xu_prepared(tmp_path)

    def native(mode):
        grid = dict(npt_rad_2d=48, npt_azim_2d=32, azimuth_range=(-180.0, 180.0)) if mode == "2d" else {}
        plan = replace(original.plan, native_records=True, q_min_A_inverse=None, q_max_A_inverse=None,
                       radial_range=(1.0, 5.2), mode=mode, **grid)
        kind = AnalysisArtifactKind.STITCH_2D if mode == "2d" else AnalysisArtifactKind.STITCH_1D
        target = resolve_finite_output_target(tmp_path, f"xu-{mode}", operation_token=f"stitch-{mode}")
        return prepare_stitch_operation(original.module.source, ModuleOutputRequest(
            target, kind, artifact_family=f"xu-{mode}"), plan, project_root=tmp_path)

    two, one = native("2d"), native("1d")
    for request in (two, one):
        result = run_stitch_operation(request)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    payload = read_analysis_artifact(two.module.output.target, expected_kind=AnalysisArtifactKind.STITCH_2D)
    pattern = read_analysis_artifact(one.module.output.target, expected_kind=AnalysisArtifactKind.STITCH_1D)
    assert payload.schema_version == 7 and payload.execution_attestation_digest is not None
    assert payload.intensity.shape == payload.coverage.shape == (48, 32)
    for name in ("intensity", "coverage", "normalization"):
        np.testing.assert_array_equal(getattr(payload.profile, name), getattr(pattern, name))
    np.testing.assert_array_equal(payload.profile.axes[0][1], pattern.axis("q"))
    assert payload.coverage.sum() == pattern.coverage.sum()  # the same pixels, over (q, χ)
    assert len(payload.contributing_frames) == len(pattern.contributing_frames)


def test_the_maps_chi_is_pyfais_exactly_on_one_geometry_at_combined_poses(tmp_path, transferred):  # noqa: F811
    """Codex terminal review: on ONE geometry (the extended arm transferred to xrayutilities, and the same arm
    as pyFAI's), xrayutilities' χ equals pyFAI's per pixel at combined poses outside the transfer's
    validation set (the scan-14 pair above are two calibrations, which differ at combined poses)."""
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.integrate.multi import create_multigeometry_integrators_from_geometry
    from tests.core.test_xu_extended_arm import _record, _receipt as _extended_receipt
    diff = Diffractometer.from_pyfai_goniometer(_record(), source_motors={"del_angle": "del", "nu_angle": "nu"})
    receipt = _extended_receipt(tmp_path, transferred[0])
    with xu_runtime_session() as session:
        geometry = resolve_xu_stitch_effective_geometry(receipt, session)
        provider = XuPowderQProvider(geometry, receipt.projection.value["acquisition"])
        for delta, nu in ((11., -13.), (23., 7.), (41., 29.)):
            integrator, = create_multigeometry_integrators_from_geometry(
                diff, {"del": np.array([delta]), "nu": np.array([nu])})
            integrator.setChiDiscAtPi()
            expected = np.degrees(integrator.center_array(SHAPE, unit="chi_rad"))
            lease = provider.frame(nu=nu, del_=delta)
            try:
                wrapped = (lease.chi() - expected + 180.) % 360. - 180.
            finally:
                lease.release()
            assert float(np.max(np.abs(wrapped))) < 1e-6
