"""I(q)'s χ window in Stitch, as Int 1D's azimuthal range (maintainer, 2026-09-30: "a 1-D χ range" -- "yes,
mirror Int"), on both backends. pyFAI's is MultiGeometry's own ``azimuth_range`` (a pixel reaching into it
counts, split); xrayutilities' takes the pixels whose χ is inside it, the same χ as pyFAI's pixel for pixel
(``test_xu_stitch_2d``), so a window takes the same side of the rings on both."""
import numpy as np
import pytest

from xrd_tools.integrate.multi import stitch_1d_streaming, stitch_2d_with_1d_streaming
from tests.core.test_multi import _make_small_integrators, _synthetic_images
from tests.core.test_xu_extended_arm import transferred  # noqa: F401 (one geometry on both backends)

ANGLES = [0.0, 4.0, 9.0]
SHAPE = (195, 1475)


def _frames(poni_fixture):
    return (_synthetic_images(n=3, shape=(100, 100), seed=191), np.asarray([1.0, 2.5, 0.75]),
            lambda: iter(_make_small_integrators(poni_fixture, ANGLES)))


def _window(poni_fixture):
    """The middle 40 % of the frames' χ: a window that cuts through every frame."""
    chi = np.concatenate([np.degrees(ai.chiArray((100, 100))).ravel()
                          for ai in _make_small_integrators(poni_fixture, ANGLES)])
    return tuple(float(value) for value in np.percentile(chi, (30, 70)))


def test_a_pyfai_window_is_multigeometrys(poni_fixture):
    from pyFAI.multi_geometry import MultiGeometry
    images, monitor, integrators = _frames(poni_fixture)
    window = _window(poni_fixture)
    eager = MultiGeometry(_make_small_integrators(poni_fixture, ANGLES), unit="q_A^-1", radial_range=(0.1, 6.0),
                          azimuth_range=window, threadpoolsize=0).integrate1d(
        images, 120, method="BBox", normalization_factor=monitor, correctSolidAngle=False)
    common = dict(npt=120, method="BBox", radial_range=(0.1, 6.0), normalization=monitor,
                  correct_solid_angle=False)
    streamed, diagnostics, _cuts = stitch_1d_streaming(images, 3, integrators, azimuth_range=window, **common)
    np.testing.assert_array_equal(streamed.radial, eager.radial)
    np.testing.assert_array_equal(streamed.intensity, eager.intensity)
    np.testing.assert_array_equal(diagnostics.coverage, eager.count)
    np.testing.assert_array_equal(diagnostics.normalization, eager.sum_normalization)
    _every, every, _cuts = stitch_1d_streaming(images, 3, integrators, **common)
    assert 0 < diagnostics.coverage.sum() < 0.7 * every.coverage.sum()


def test_i_of_q_beside_a_map_takes_the_window_as_a_1d_stitch_and_the_map_keeps_its_own(poni_fixture):
    images, monitor, integrators = _frames(poni_fixture)
    window = _window(poni_fixture)
    common = dict(method="BBox", radial_range=(0.1, 6.0), normalization=monitor, correct_solid_angle=False)
    alone, alone_diagnostics, _cuts = stitch_1d_streaming(images, 3, integrators, npt=120, azimuth_range=window,
                                                          **common)
    two_d = dict(npt_rad=40, npt_azim=36, npt_1d=120, azimuth_range=(-180.0, 180.0), **common)
    stitched, map_diagnostics, profile, diagnostics, _cuts, _gi = stitch_2d_with_1d_streaming(
        images, 3, integrators, azimuth_range_1d=window, **two_d)
    np.testing.assert_array_equal(profile.intensity, alone.intensity)
    np.testing.assert_array_equal(diagnostics.coverage, alone_diagnostics.coverage)
    every_map, every_map_diagnostics, every_profile, *_rest = stitch_2d_with_1d_streaming(
        images, 3, integrators, **two_d)
    np.testing.assert_array_equal(stitched.intensity, every_map.intensity)  # the map: its own χ range
    np.testing.assert_array_equal(map_diagnostics.coverage, every_map_diagnostics.coverage)
    assert not np.array_equal(profile.intensity, every_profile.intensity, equal_nan=True)


def _xu_chi(receipt, *, nu, delta):
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.integrate.xu_stitch import XuPowderQProvider, resolve_xu_stitch_effective_geometry
    with xu_runtime_session() as session:
        geometry = resolve_xu_stitch_effective_geometry(receipt, session)
        lease = XuPowderQProvider(geometry, receipt.projection.value["acquisition"]).frame(nu=nu, del_=delta)
        try:
            return np.array(lease.chi())
        finally:
            lease.release()


def test_an_xrayutilities_window_bins_the_pixels_whose_chi_is_inside(tmp_path):
    """The same I(q), bit for bit, as the pixels outside the window masked; the GI cuts and the map keep theirs."""
    from tests.core.test_xu_stitch import _receipt
    from tests.core.test_xu_stitch_2d import _run
    receipt = _receipt(tmp_path)
    chi = _xu_chi(receipt, nu=-10.0, delta=14.0)  # _run's frame
    window = tuple(float(value) for value in np.nanpercentile(chi, (30, 70)))
    outside = ~(np.isfinite(chi) & (chi >= window[0]) & (chi <= window[1]))
    windowed, masked, every = (_run(receipt, azimuth_range_1d=window), _run(receipt, mask=outside), _run(receipt))
    np.testing.assert_array_equal(windowed.payload.intensity, masked.payload.intensity)
    np.testing.assert_array_equal(windowed.diagnostics.coverage, masked.diagnostics.coverage)
    np.testing.assert_array_equal(windowed.diagnostics.normalization, masked.diagnostics.normalization)
    assert 0 < windowed.diagnostics.coverage.sum() < 0.7 * every.diagnostics.coverage.sum()
    # beside a map: I(q) is windowed, the map is not
    grid = (64, 36, (-180.0, 180.0))
    both, plain = _run(receipt, azimuth_range_1d=window, map_grid=grid), _run(receipt, map_grid=grid)
    np.testing.assert_array_equal(both.payload.intensity, windowed.payload.intensity)
    np.testing.assert_array_equal(both.map[1].coverage, plain.map[1].coverage)
    with pytest.raises(ValueError, match="finite and increasing"):
        _run(receipt, azimuth_range_1d=(10.0, -10.0))


def test_both_backends_window_the_same_side_of_the_rings(tmp_path, transferred):  # noqa: F811
    """On one geometry (the extended arm and its xrayutilities transfer), a frame ten times brighter where
    pyFAI's χ is above its median: on both backends the window above it takes the bright pixels and the one
    below the dark (half a turn apart, as the STITCH-CHI-180 ledger row supposed, they would swap; measured
    2026-09-30 at del 23°, nu 7°: bright/dark 10.97 on pyFAI, 11.63 on xrayutilities)."""
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.integrate.multi import create_multigeometry_integrators_from_geometry
    from xrd_tools.integrate.xu_stitch import run_xu_hist_stitch_1d
    from tests.core.test_xu_extended_arm import _receipt as extended_receipt, _record
    from tests.core.test_xu_stitch import _OneFrameSource
    delta, nu = 23.0, 7.0
    diff = Diffractometer.from_pyfai_goniometer(_record(), source_motors={"del_angle": "del", "nu_angle": "nu"})

    def integrator():
        ai, = create_multigeometry_integrators_from_geometry(diff, {"del": np.array([delta]), "nu": np.array([nu])})
        ai.setChiDiscAtPi()
        return ai
    reference = integrator()
    chi = np.degrees(np.asarray(reference.center_array(SHAPE, unit="chi_rad")))
    q = np.asarray(reference.center_array(SHAPE, unit="q_A^-1"))
    split = float(np.median(chi))
    image = np.where(chi > split, 10.0, 1.0)
    radial = (float(q.min()), float(q.max()))
    source = _OneFrameSource(del_value=delta, nu_value=nu, monitor=1.0)
    source.image = image.astype(np.float32)
    receipt = extended_receipt(tmp_path, transferred[0])
    windows = ((split, 180.0), (-180.0, split))
    pyfai = [np.nanmedian(stitch_1d_streaming(
        [image], 1, lambda: iter([integrator()]), npt=64, method="full", radial_range=radial,
        azimuth_range=window)[0].intensity) for window in windows]
    xu = [np.nanmedian(run_xu_hist_stitch_1d(
        receipt, source, frame_indices=(7,), radial_range=radial, npt=64, monitor_key="monitor",
        max_frame_bytes=16 * 1024 * 1024, azimuth_range_1d=window).payload.intensity) for window in windows]
    assert pyfai[0] / pyfai[1] > 5 and xu[0] / xu[1] > 5, (pyfai, xu)
