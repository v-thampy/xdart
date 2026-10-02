"""17-2's circle directions in RSM: psic's mirror image, mapped through the mirror (the maintainer approved the fix
on 2026-10-01: "Size: about half a day, tests included. Shall I go ahead? yes").

17-2's record reverses mu, chi and nu, the circles that turn about x and y: the instrument is psic's left-right
mirror image (z → −z through the vertical plane that holds the beam). For now phi turns the other way as well ("at
chi=0, phi and eta move in opposite directions, at chi = 180, they both move in the same direction. that is right
now"). SPEC's psic model, its orientation reflections' labels and its UB are the mirror image's, so RSM computes each
pixel with the circles as they turn and mirrors the sample frame's q into SPEC's handedness (``PixelQMap.mirror``).

The oracle here is independent of that implementation: SPEC's own model, psic as written (or with phi reversed),
with the detector seen in the mirror (its camera's left-right axis flipped; a tilted mount's roll and tilt
reversed).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from xrd_tools.core.geometry.camera import CAMERA_AXES
from xrd_tools.core.geometry.diffractometer import Diffractometer
from xrd_tools.core.geometry.pixel_q import MIRROR, DetectorHeader, PixelQMap
from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
from xrd_tools.rsm.orientation import OrientationReflection, SpecOrientation
from xrd_tools.session.instrument import mirrored_axis

SPEC_SAMPLE, SPEC_DETECTOR = ("x+", "z-", "y+", "z-"), ("x+", "z-")  # mu, eta, chi, phi; nu, del
PHI_REVERSED = ("x+", "z-", "y+", "z+")
HEADER = DetectorHeader(cch1=40.0, cch2=71.0, pwidth1=0.172, pwidth2=0.172, distance=600.0, Nch1=97, Nch2=151)
POSES = ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (3.5, 21.0, 97.0, 33.0, 5.0, 40.0),
         (-6.9, 42.4, 200.3, -71.0, 12.0, 47.8))  # mu, eta, chi, phi, nu, del: every circle away from zero
CAMERAS = [(rows, columns) for rows in CAMERA_AXES for columns in CAMERA_AXES if rows[0] != columns[0]]
ENERGY = 13_000.0


def _seen_in_the_mirror(direction: str) -> str:
    """A camera direction in psic's mirror: its z component flips."""
    return direction[0] + ("-" if direction[1] == "+" else "+") if direction[0] == "z" else direction


def _map(sample, detector, camera, *, mount=(0.0, 0.0, 0.0), mirror=False) -> PixelQMap:
    diffractometer = replace(Diffractometer.psic(), sample_circles=tuple(sample), detector_circles=tuple(detector),
                             camera=tuple(camera))
    return PixelQMap(diffractometer, HEADER, area_mount=mount, mirror=mirror)


def _instrument(sample, camera, *, mount=(0.0, 0.0, 0.0)) -> PixelQMap:
    """SPEC's model ``sample`` as the mirrored instrument turns it (mu, chi, nu reversed), its detector seen in
    the mirror, mapped through the mirror."""
    return _map(tuple(map(mirrored_axis, sample)), tuple(map(mirrored_axis, SPEC_DETECTOR)),
                tuple(map(_seen_in_the_mirror, camera)), mount=mount, mirror=True)


@pytest.mark.parametrize("sample", [SPEC_SAMPLE, PHI_REVERSED], ids=["pure-mirror", "phi-reversed-too"])
@pytest.mark.parametrize("camera", CAMERAS)
def test_through_the_mirror_the_instrument_maps_as_specs_model_pixel_for_pixel(camera, sample):
    """Every pixel's q and HKL, at poses with every circle away from zero, for all 8 cameras: the mirrored
    instrument through the mirror is SPEC's model with the detector as it is. Measured bit for bit."""
    ub = np.array([[1.3, 0.2, -0.1], [0.05, 0.9, 0.3], [-0.2, 0.1, 1.1]])  # not symmetric: order matters
    spec, instrument = _map(sample, SPEC_DETECTOR, camera), _instrument(sample, camera)
    with xu_runtime_session() as session:
        for pose in POSES:
            angles = [np.array([value]) for value in pose]
            for matrix in (None, ub):
                expected = np.stack(spec.pixel_q(angles, ENERGY, UB=matrix, runtime_session=session))
                got = np.stack(instrument.pixel_q(angles, ENERGY, UB=matrix, runtime_session=session))
                assert np.array_equal(got, expected), (pose, matrix is None)


@pytest.mark.parametrize("camera", CAMERAS)
def test_a_tilted_mount_seen_in_the_mirror_reverses_its_roll_and_tilt(camera):
    """The refinement's mount in the mirror: detrot and tilt reverse, the tilt azimuth stays (equivalently, for a
    tilt kept positive as a calibration file holds it, the azimuth turns by 180°)."""
    mount = (1.7, 3.2, 37.0)
    detrot, tilt, azimuth = mount
    spec = _map(SPEC_SAMPLE, SPEC_DETECTOR, camera, mount=mount)
    with xu_runtime_session() as session:
        for pose in POSES:
            angles = [np.array([value]) for value in pose]
            expected = np.stack(spec.pixel_q(angles, ENERGY, runtime_session=session))
            for seen in ((-detrot, -tilt, azimuth), (-detrot, tilt, azimuth + 180.0)):
                got = np.stack(_instrument(SPEC_SAMPLE, camera, mount=seen).pixel_q(
                    angles, ENERGY, runtime_session=session))
                np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_the_mirror_is_one_fixed_matrix_folded_into_the_ub():
    """xrayutilities returns inv(UB) @ q; the mirror's map hands it MIRROR @ UB, so it returns inv(UB) @ MIRROR @ q.
    Without the mirror the caller's matrix passes through untouched (the very object)."""
    ub = np.array([[1.3, 0.2, -0.1], [0.05, 0.9, 0.3], [-0.2, 0.1, 1.1]])
    plain = _map(SPEC_SAMPLE, SPEC_DETECTOR, CAMERAS[0])
    assert plain.frame_ub(ub) is ub
    np.testing.assert_array_equal(replace(plain, mirror=True).frame_ub(ub), MIRROR @ ub)
    assert not MIRROR.flags.writeable and np.array_equal(MIRROR, np.diag([1.0, 1.0, -1.0]))


# Indexing with phi reversed (17-2 for now): SPEC's orientation reflections at phi 0 and a peak reached by turning phi.
LATTICE = 3.905  # Å, cubic
B = np.eye(3) * (2 * np.pi / LATTICE)
U = np.array([[0.98, -0.15, 0.13], [0.16, 0.986, -0.04], [-0.12, 0.06, 0.99]])
U, _ = np.linalg.qr(U)  # a proper rotation, the crystal's orientation in SPEC's (the mirror image's) frame
U *= np.sign(np.linalg.det(U))
WAVELENGTH = 12398.419843320026 / ENERGY


def _angles_for(hkl, phi, model: PixelQMap):
    """The (mu, eta, chi, phi, nu, del) at which ``model``'s crystal (U, B) diffracts ``hkl``, with mu = nu = 0 and
    phi given: eta, chi and del solved by least squares, then checked exact."""
    from scipy.optimize import least_squares

    target = U @ B @ np.asarray(hkl, dtype=np.float64)
    two_theta = 2 * np.degrees(np.arcsin(np.linalg.norm(target) * WAVELENGTH / (4 * np.pi)))
    with xu_runtime_session():
        hxrd = model.diff_config.make_hxrd(ENERGY)

        def residual(free):
            eta, chi, delta = free
            return np.asarray(hxrd.Ang2Q.point(0.0, eta, chi, phi, 0.0, delta, UB=np.eye(3))) - target

        best = None
        for chi0 in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
            for eta0 in (two_theta / 2, two_theta / 2 + 90.0, -two_theta / 2):
                fit = least_squares(residual, (eta0, chi0, two_theta), xtol=1e-15, ftol=1e-15, gtol=1e-15)
                if best is None or fit.cost < best.cost:
                    best = fit
    assert np.max(np.abs(best.fun)) < 1e-10, best.fun
    eta, chi, delta = best.x
    return (0.0, float(eta), float(chi), float(phi), 0.0, float(delta))


def _orientation(reflections):
    return SpecOrientation((LATTICE, LATTICE, LATTICE, 90.0, 90.0, 90.0),
                           (2 * np.pi / LATTICE,) * 3 + (90.0, 90.0, 90.0),
                           tuple(OrientationReflection(tuple(float(v) for v in hkl), angles, WAVELENGTH)
                                 for hkl, angles in reflections))


def _beam_pixel_hkl(model: PixelQMap, angles, ub):
    with xu_runtime_session() as session:
        h, k, l = model.pixel_q([np.array([value]) for value in angles], ENERGY, UB=ub, runtime_session=session)
    row, column = int(HEADER.cch1), int(HEADER.cch2)
    return np.array([h[0, row, column], k[0, row, column], l[0, row, column]])


def test_with_phi_in_the_record_a_peak_at_another_phi_indexes_and_without_it_misses_by_twice_the_turn():
    """The orientation reflections (1 1 4) and (-1 0 4) at phi 0, the (1 1 3) reached at phi 30°, all generated in
    the mirror image's model with phi reversed (17-2 now) and labelled there, as SPEC labels them. The record with
    phi reversed indexes the (1 1 3) exactly; the pure-mirror record (phi as psic's) recomputes the same UB (phi is
    0 at both reflections) and puts the peak at the (1 1 3) turned by twice the phi move about the phi axis."""
    truth = _map(PHI_REVERSED, SPEC_DETECTOR, CAMERAS[0])  # the mirror image's model: psic, phi reversed
    or0, or1, peak = (1, 1, 4), (-1, 0, 4), (1, 1, 3)
    orientation = _orientation([(or0, _angles_for(or0, 0.0, truth)), (or1, _angles_for(or1, 0.0, truth))])
    at = _angles_for(peak, 30.0, truth)
    with_phi, without_phi = _instrument(PHI_REVERSED, CAMERAS[0]), _instrument(SPEC_SAMPLE, CAMERAS[0])
    ub = with_phi.orientation_ub(orientation)
    np.testing.assert_allclose(ub, U @ B, rtol=0, atol=1e-9)
    np.testing.assert_allclose(_beam_pixel_hkl(with_phi, at, ub), peak, rtol=0, atol=1e-9)
    ub_without = without_phi.orientation_ub(orientation)
    np.testing.assert_allclose(ub_without, ub, rtol=0, atol=1e-9)  # phi 0 at both reflections: the same UB
    turn = np.radians(2 * 30.0)  # phi's axis is the sample frame's z
    turned = np.array([[np.cos(turn), -np.sin(turn), 0.0], [np.sin(turn), np.cos(turn), 0.0], [0.0, 0.0, 1.0]])
    missed = _beam_pixel_hkl(without_phi, at, ub_without)
    np.testing.assert_allclose(missed, np.linalg.solve(ub, turned @ ub @ np.asarray(peak, float)), rtol=0, atol=1e-9)
    assert np.linalg.norm(missed - peak) > 0.5
