"""GI Stitch 2D (STITCH-2D-20260930 item 3): the q_ip-q_oop map beside the (chi, q) map, I(q) and the GI cuts,
from the same frames and pixels. The map holds both cuts: summed over q_oop it is the q_ip cut, over q_ip the
q_oop cut (each cut keeps the pixels inside the other coordinate's range, as the map does)."""
import numpy as np
import pytest

from xrd_tools.integrate.stitch_gi import GI_UNITS, StitchGI
from tests.core.test_xu_extended_arm import SHAPE, _receipt, transferred  # noqa: F401


def _holds_both_cuts(gi_map, cuts, *, signal_rtol=1e-12):
    stitched, diagnostics = gi_map
    assert (stitched.unit, stitched.azimuthal_unit) == (GI_UNITS["q_ip"], GI_UNITS["q_oop"])
    assert stitched.intensity.shape == diagnostics.coverage.shape == (stitched.radial.size, stitched.azimuthal.size)
    weight = diagnostics.normalization
    signal = np.where(weight > 0, np.nan_to_num(stitched.intensity) * weight, 0.0)
    for (cut, cut_diagnostics), axis, coordinate in zip(cuts, (1, 0), (stitched.radial, stitched.azimuthal)):
        np.testing.assert_allclose(cut.radial, coordinate, rtol=0, atol=1e-12)
        assert np.array_equal(diagnostics.coverage.sum(axis=axis), cut_diagnostics.coverage)
        assert cut_diagnostics.coverage.sum() > 20
        np.testing.assert_allclose(weight.sum(axis=axis), cut_diagnostics.normalization, rtol=signal_rtol)
        expected = np.where(cut_diagnostics.normalization > 0,
                            np.nan_to_num(cut.intensity) * cut_diagnostics.normalization, 0.0)
        np.testing.assert_allclose(signal.sum(axis=axis), expected, rtol=signal_rtol, atol=1e-300)


def test_the_multigeometry_gi_map_holds_both_cuts_beside_the_chi_q_map():
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm
    from xrd_tools.integrate.multi import (
        create_multigeometry_integrator_series_from_geometry, stitch_1d_streaming, stitch_2d_with_1d_streaming,
    )
    from tests.core.test_stacked_arm_extension import _calibration
    arm = StackedDetectorArm(nu_motor="nu", del_motor="del", nu_offset=.01, del_offset=-.02)
    diff = Diffractometer.from_pyfai_goniometer(arm.to_pyfai_goniometer(_calibration()),
                                                source_motors={"del_angle": "del", "nu_angle": "nu"})
    series = create_multigeometry_integrator_series_from_geometry(
        diff, {"del": np.array([10., 24., 38.]), "nu": np.array([2., 9., 15.])})
    images = [np.random.default_rng(5).uniform(5., 50., (195, 1475)) for _ in range(3)]
    gi = StitchGI(("q_ip", "q_oop"), incident_angle_deg=.4, ip_range=(-2.5, 2.5), oop_range=(0., 6.))
    common = dict(radial_range=(.5, 6.), correct_solid_angle=True, polarization_factor=.99)
    (chi_map, _chi_diagnostics, profile, _profile_diagnostics, cuts, gi_map) = stitch_2d_with_1d_streaming(
        images, 3, series.__iter__, npt_rad=64, npt_azim=64, npt_1d=64, azimuth_range=(-180., 180.),
        gi=gi, gi_angles=[(.4, 0.)] * 3, **common)
    alone = stitch_1d_streaming(images, 3, series.__iter__, npt=64, gi=gi, gi_angles=[(.4, 0.)] * 3, **common)
    np.testing.assert_array_equal(profile.intensity, alone[0].intensity)  # I(q) unchanged
    for (cut, diagnostics), (single, single_diagnostics) in zip(cuts, alone[2], strict=True):
        np.testing.assert_array_equal(cut.intensity, single.intensity)  # the cuts unchanged
        np.testing.assert_array_equal(diagnostics.coverage, single_diagnostics.coverage)
    assert chi_map.azimuthal_unit == "chi_deg" and gi_map is not None
    _holds_both_cuts(gi_map, cuts)


@pytest.mark.parametrize("pose", [{"eta": .4, "chi": 90., "mu": 0.}, {"eta": .4, "chi": 88., "mu": .5}])
def test_the_xrayutilities_gi_map_holds_both_cuts_beside_the_chi_q_map(tmp_path, transferred, pose):  # noqa: F811
    from xrd_tools.integrate.xu_stitch import run_xu_hist_stitch_1d
    angles = {1: (12., -7.), 2: (26., 4.), 3: (38., 16.)}

    class Source:
        frame_indices = (1, 2, 3)

        def metadata_for(self, label):
            return {"del": angles[label][0], "nu": angles[label][1], **pose}

        def load_frame(self, label):
            return np.random.default_rng(label).uniform(5., 50., SHAPE).astype(np.float32)

    receipt = _receipt(tmp_path, transferred[0])
    gi = StitchGI(("q_ip", "q_oop"), incidence_motor="eta", tilt_motor="chi", ip_range=(-4., 4.),
                  oop_range=(0., 1.5), sample_motors=(("mu", "mu"), ("phi", None)))
    common = dict(frame_indices=(1, 2, 3), npt=48, radial_range=(.5, 6.), monitor_key=None,
                  max_frame_bytes=8 * 1024 * 1024, gi=gi)
    result = run_xu_hist_stitch_1d(receipt, Source(), map_grid=(48, 48, (-180., 180.)), **common)
    alone = run_xu_hist_stitch_1d(receipt, Source(), **common)
    np.testing.assert_array_equal(result.payload.intensity, alone.payload.intensity)
    for (cut, diagnostics), (single, single_diagnostics) in zip(result.gi_cuts, alone.gi_cuts, strict=True):
        np.testing.assert_array_equal(cut.intensity, single.intensity)
        np.testing.assert_array_equal(diagnostics.coverage, single_diagnostics.coverage)
    assert result.map is not None and result.gi_map is not None
    _holds_both_cuts(result.gi_map, result.gi_cuts)
