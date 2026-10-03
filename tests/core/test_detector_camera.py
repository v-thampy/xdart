"""The camera table: each signed-axis pair against xrayutilities and pyFAI, on 2θ and χ at every pixel."""
import numpy as np
import pytest

from xrd_tools.core.geometry import ImageOrientation
from xrd_tools.core.geometry.camera import CAMERA_AXES, DetectorCamera

xu = pytest.importorskip('xrayutilities')

SHAPE, BEAM, DIST_MM, PITCH_MM, ENERGY = (195, 487), (40, 100), 500., .172, 12000.  # off-centre, not square
WAVELENGTH_A = 12398.419843320026 / ENERGY
CAMERAS = [DetectorCamera(rows, columns) for rows in CAMERA_AXES for columns in CAMERA_AXES
           if rows[0] != columns[0]]


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('transpose,vertical,horizontal', [
    (t, v, h) for t in (False, True) for v in (False, True) for h in (False, True)])
def test_fractional_pixels_and_inverse_follow_the_actual_array(rotation, transpose, vertical, horizontal):
    orientation = ImageOrientation(rotation, vertical, horizontal, transpose)
    image = np.arange(7 * 11).reshape(7, 11)
    transformed = orientation.apply(image)
    # The midpoint of four neighbours maps to their midpoint under every camera.
    xy = [np.argwhere(transformed == image[y, x])[0][::-1] for y in (2, 3) for x in (4, 5)]
    expected = np.mean(xy, axis=0)
    actual = orientation.apply_pixel((4.5, 2.5), image.shape)
    assert actual == tuple(expected)
    assert orientation.unapply_pixel(actual, image.shape) == (4.5, 2.5)


def _xu_maps(camera):
    """2θ and pyFAI's χ per raw pixel, from psic's lists with every circle at zero (x up, y beam)."""
    qconv = xu.QConversion(['x+', 'z-', 'y+', 'z-'], ['x+', 'z-'], [0, 1, 0])
    hxrd = xu.HXRD([0, 1, 0], [0, 0, 1], geometry='real', en=ENERGY, qconv=qconv)
    hxrd.Ang2Q.init_area(*camera.to_xu(), cch1=BEAM[0], cch2=BEAM[1], Nch1=SHAPE[0], Nch2=SHAPE[1],
                         pwidth1=PITCH_MM, pwidth2=PITCH_MM, distance=DIST_MM)
    qx, qy, qz = hxrd.Ang2Q.area(0, 0, 0, 0, 0, 0)
    kx, ky, kz = qx, qy + 2 * np.pi / WAVELENGTH_A, qz  # the scattered wave vector
    # pyFAI's frame: t1 up (x), t3 along the beam (y), so t2 = -z; its χ is atan2(t1, t2)
    return np.arccos(ky / np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)), np.arctan2(kx, -kz)


def _pyfai_maps(orientation, detector_orientation=3):
    """2θ and χ per RAW pixel when pyFAI integrates the raw array after ``orientation``."""
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    index = orientation.apply(np.arange(SHAPE[0] * SHAPE[1]).reshape(SHAPE))
    detector = Detector(pixel1=PITCH_MM * 1e-3, pixel2=PITCH_MM * 1e-3, max_shape=index.shape,
                        orientation=detector_orientation)
    i1, i2 = (int(values[0]) for values in np.nonzero(index == BEAM[0] * SHAPE[1] + BEAM[1]))
    poni1, poni2, _ = detector.calc_cartesian_positions(d1=np.array([i1]), d2=np.array([i2]))
    ai = AzimuthalIntegrator(dist=DIST_MM * 1e-3, poni1=float(poni1[0]), poni2=float(poni2[0]),
                             detector=detector, wavelength=WAVELENGTH_A * 1e-10)
    maps = []
    for unit in ('2th_rad', 'chi_rad'):
        raw = np.empty(SHAPE[0] * SHAPE[1])
        raw[index.ravel()] = ai.center_array(index.shape, unit=unit, scale=False).ravel()
        maps.append(raw.reshape(SHAPE))
    return maps


def _differences(expected, actual):
    away = np.ones(SHAPE, bool)
    away[BEAM] = False  # χ is undefined at the beam itself
    return (np.abs(actual[0] - expected[0])[away].max(),
            np.abs(np.angle(np.exp(1j * (actual[1] - expected[1]))))[away].max())


@pytest.mark.parametrize('camera', CAMERAS, ids=lambda camera: f'{camera.rows},{camera.columns}')
def test_each_camera_is_one_image_transform_into_pyfais_frame(camera):
    expected = _xu_maps(camera)
    two_theta, chi = _differences(expected, _pyfai_maps(camera.image_orientation()))
    assert two_theta < 1e-9 and chi < 1e-9
    others = [other for other in CAMERAS if other != camera]  # and no other camera's transform fits
    assert all(max(_differences(expected, _pyfai_maps(other.image_orientation()))) > 1e-3 for other in others)
    assert DetectorCamera.from_image_orientation(camera.image_orientation()) == camera


@pytest.mark.parametrize('code', [1, 2, 3, 4])
def test_first_pixel_orientation_reindexes_the_same_physical_pixels(code):
    """2026.9 orientation reindexes pixels; the PONI and physical rays stay fixed."""
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.geometry import DetectorCalibration
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    poni = PONI(dist=.5, poni1=.01, poni2=.02, rot1=.13, rot2=-.09,
                rot3=.21, wavelength=1e-10, detector='Pilatus100k')
    reference = detector_calibration_to_integrator(
        DetectorCalibration(poni, {'orientation': 3}))
    actual = detector_calibration_to_integrator(
        DetectorCalibration(poni, {'orientation': code}))
    rows, columns = np.array([12.25, 61.5, 170.75]), np.array([33.5, 231.25, 450.75])
    mapped_rows = SHAPE[0] - 1 - rows if code in (1, 2) else rows
    mapped_columns = SHAPE[1] - 1 - columns if code in (1, 4) else columns
    for use_cython in (False, True):
        np.testing.assert_allclose(
            actual.calc_pos_zyx(d1=mapped_rows, d2=mapped_columns, use_cython=use_cython),
            reference.calc_pos_zyx(d1=rows, d2=columns, use_cython=use_cython),
            rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual.tth(mapped_rows, mapped_columns),
                               reference.tth(rows, columns), rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual.qFunction(mapped_rows, mapped_columns),
                               reference.qFunction(rows, columns), rtol=0, atol=1e-9)
    np.testing.assert_allclose(actual.chi(mapped_rows, mapped_columns),
                               reference.chi(rows, columns), rtol=0, atol=1e-12)


def test_the_cameras_in_use_and_equivalent_transforms():
    # XU Stitch's fixed camera on the raw Pilatus arrays: pyFAI would need the array turned 180°
    assert DetectorCamera('x-', 'z+').image_orientation() == ImageOrientation(rotation=180)
    # the RSM notebook / asset camera (a Pilatus 100K mounted with its long side vertical): 90°
    assert DetectorCamera('z-', 'x-').image_orientation() == ImageOrientation(rotation=90)
    # the same transform written differently is the same camera
    both_flips = ImageOrientation(flip_vertical=True, flip_horizontal=True)
    assert DetectorCamera.from_image_orientation(both_flips) == DetectorCamera('x-', 'z+')
    assert DetectorCamera.from_image_orientation(ImageOrientation(transpose=True)) == DetectorCamera('z-', 'x+')
    assert DetectorCamera.from_image_orientation(ImageOrientation()) == DetectorCamera('x+', 'z-')


def test_tilted_another_pyfai_orientation_is_another_geometry():
    """Changing orientation for unchanged image indices changes their physical interpretation,
    even when the PONI is moved to keep the direct beam on the same pixel."""
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    maps = {}
    for code in (3, 2):
        detector = Detector(pixel1=PITCH_MM * 1e-3, pixel2=PITCH_MM * 1e-3, max_shape=SHAPE, orientation=code)
        poni1, poni2, _ = detector.calc_cartesian_positions(d1=np.array([BEAM[0]]), d2=np.array([BEAM[1]]))
        ai = AzimuthalIntegrator(dist=DIST_MM * 1e-3, poni1=float(poni1[0]), poni2=float(poni2[0]), rot1=.13,
                                 rot2=-.09, rot3=.21, detector=detector, wavelength=WAVELENGTH_A * 1e-10)
        rotation = ai.rotation_matrix()  # move the PONI so the true direct beam hits the beam pixel
        ai.poni1 -= ai.dist * rotation[2, 0] / rotation[2, 2]
        ai.poni2 -= ai.dist * rotation[2, 1] / rotation[2, 2]
        assert float(ai.tth(np.array([BEAM[0]]), np.array([BEAM[1]]))[0]) == pytest.approx(0, abs=1e-12)
        maps[code] = [ai.center_array(SHAPE, unit=unit, scale=False) for unit in ('2th_rad', 'chi_rad')]
    two_theta, chi = _differences(maps[3], maps[2])
    assert two_theta > 1e-5 and chi > np.radians(1)


@pytest.mark.parametrize('pair', [('x+', 'x-'), ('z-', 'z+'), ('y+', 'z-'), ('x+', 'up')])
def test_a_camera_is_one_x_and_one_z_direction(pair):
    with pytest.raises(ValueError, match='one x and one z'):
        DetectorCamera(*pair)
