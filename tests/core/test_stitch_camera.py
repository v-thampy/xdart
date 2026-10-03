"""The detector camera in Stitch (maintainer, 2026-09-30): "two drop-down boxes linked together. The first has
options x+, x-, z+, z-. Once the first ... is selected, the other box shows the remaining two transverse options";
given in xrayutilities' form it "should account for the mirroring".

Stitch and RSM calibration turn raw images into pyFAI's native reference, rows x+, columns z-.
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.geometry import ImageOrientation
from xrd_tools.core.geometry.camera import (
    CAMERA_AXES, STITCH_CAMERA, DetectorCamera, stitch_camera, stitch_orientation,
)

CAMERAS = [DetectorCamera(rows, columns) for rows in CAMERA_AXES for columns in CAMERA_AXES if rows[0] != columns[0]]


def test_each_camera_is_one_turn_and_mirror_into_stitchs_frame():
    table = {camera.to_xu(): (stitch_orientation(camera).rotation, stitch_orientation(camera).flip_horizontal)
             for camera in CAMERAS}
    assert table == {('x+', 'z-'): (0, False), ('x+', 'z+'): (0, True), ('x-', 'z+'): (180, False),
                     ('x-', 'z-'): (180, True), ('z-', 'x-'): (90, False), ('z+', 'x-'): (90, True),
                     ('z+', 'x+'): (270, False), ('z-', 'x+'): (270, True)}
    assert STITCH_CAMERA.to_xu() == ('x+', 'z-') and stitch_orientation(STITCH_CAMERA).is_identity
    raw = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(stitch_orientation(DetectorCamera('x+', 'z+')).apply(raw), raw[:, ::-1])
    np.testing.assert_array_equal(stitch_orientation(DetectorCamera('x-', 'z-')).apply(raw), raw[::-1, :])
    np.testing.assert_array_equal(stitch_orientation(DetectorCamera('z-', 'x+')).apply(raw), raw.T)
    for rotation in (0, 90, 180, 270):  # an image rotation saved before the camera: its unmirrored camera
        assert stitch_orientation(stitch_camera(ImageOrientation(rotation=rotation))) == ImageOrientation(rotation=rotation)
    for camera in CAMERAS:
        np.testing.assert_array_equal(stitch_orientation(camera).apply(raw),
                                      camera.image_orientation().apply(raw))
        assert stitch_camera(stitch_orientation(camera)) == camera


@pytest.mark.parametrize('camera', CAMERAS, ids=lambda camera: ','.join(camera.to_xu()))
@pytest.mark.parametrize('nu_sense', [1, -1])
def test_native_rough_arm_preserves_signed_physical_q_for_every_camera(camera, nu_sense):
    """Independent XU physical-circle oracle at combined angles, not just |q| or zero pose."""
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.integrate.calibration_seed import rough_arm_record
    from xrd_tools.integrate.multi import create_multigeometry_integrators_from_geometry
    orientation = stitch_orientation(camera)
    raw_shape, raw_beam = (195, 487), (200.25, 80.75)
    with xu_runtime_session() as runtime:
        xu = runtime.xu
        wavelength_A = float(xu.en2lam(17000.004))
        qconv = xu.QConversion(['x+', 'z-', 'y+', 'z-'], ['x+' if nu_sense == 1 else 'x-', 'z-'], [0, 1, 0])
        hxrd = xu.HXRD([0, 1, 0], [0, 0, 1], en=17000.004, geometry='real', qconv=qconv)
        hxrd.Ang2Q.init_area(*camera.to_xu(), cch1=raw_beam[1], cch2=raw_beam[0], Nch1=195, Nch2=487,
                             pwidth1=.172, pwidth2=.172, distance=350.)
        expected = np.stack(hxrd.Ang2Q.area(0., 0., 0., 0., 11., 23.))
        record = rough_arm_record(distance_m=.35, detector='Pilatus100k',
            beam_px=orientation.apply_pixel(raw_beam, raw_shape), wavelength_m=wavelength_A * 1e-10,
            camera=orientation, senses={'nu': nu_sense})
        diff = Diffractometer.from_pyfai_goniometer(record, source_motors={'del_angle': 'del', 'nu_angle': 'nu'})
        ai, = create_multigeometry_integrators_from_geometry(diff, {'nu': [11.], 'del': [23.]})
        raw_points = [(4, 0), (240, 100), (480, 194)]
        x, y = np.array([orientation.apply_pixel(point, raw_shape) for point in raw_points]).T
        theta, chi = ai.tth(y, x), ai.chi(y, x)
        k = 2 * np.pi / wavelength_A
        actual = k * np.stack((np.sin(theta) * np.sin(chi), np.cos(theta) - 1,
                                -np.sin(theta) * np.cos(chi)))
        target = expected[:, [p[1] for p in raw_points], [p[0] for p in raw_points]]
        np.testing.assert_allclose(actual, target, atol=4e-13, rtol=0)


def _mirrored_run(tmp_path, monkeypatch, name, *, flip, mirror_on_disk):
    from tests.core import test_stitch_operation as harness
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.stitch_operation import (
        capture_stitch_geometry, prepare_stitch_operation, run_stitch_operation,
    )
    from xrd_tools.io.analysis_artifact import read_analysis_artifact
    ring = harness._ring
    if mirror_on_disk:
        monkeypatch.setattr(harness, '_ring', lambda index: np.ascontiguousarray(ring(index)[:, ::-1]))
    root = tmp_path / name
    root.mkdir()
    request = harness._prepared(root)
    monkeypatch.setattr(harness, '_ring', ring)
    geometry = capture_stitch_geometry(replace(request.plan.geometry.request, image_flip=flip))
    plan = replace(request.plan, geometry=geometry)
    request = prepare_stitch_operation(request.module.source, request.module.output, plan, project_root=root)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    return geometry, read_analysis_artifact(request.module.output.target)


def test_a_mirrored_camera_stitches_the_frames_mirrored(tmp_path, monkeypatch):
    """The frames reaching pyFAI are the raw ones mirrored left-right: a mirrored camera's stitch is the
    unmirrored geometry's of the same frames mirrored on disk, value for value. Only a mirrored geometry's
    identity changes."""
    plain, _ = _mirrored_run(tmp_path, monkeypatch, 'plain', flip=False, mirror_on_disk=False)
    flipped, mirrored = _mirrored_run(tmp_path, monkeypatch, 'flip', flip=True, mirror_on_disk=False)
    _, on_disk = _mirrored_run(tmp_path, monkeypatch, 'disk', flip=False, mirror_on_disk=True)
    for name in ('intensity', 'coverage'):
        np.testing.assert_array_equal(getattr(mirrored, name), getattr(on_disk, name))
    assert not np.array_equal(mirrored.intensity, _mirrored_run(tmp_path, monkeypatch, 'again', flip=False,
                                                                mirror_on_disk=False)[1].intensity)
    assert replace(plain.request, image_flip=False) == plain.request and flipped.request.image_flip
    assert flipped.fingerprint != plain.fingerprint


def test_a_mirrored_cameras_mask_and_dark_are_mirrored_like_its_frames(tmp_path):
    from tests.core.test_stitch_operation import _prepared
    from xrd_tools.analysis.stitch_native import native_detector_image, native_mask
    from xrd_tools.analysis.stitch_operation import capture_stitch_geometry
    plan = _prepared(tmp_path).plan
    geometry = capture_stitch_geometry(replace(plan.geometry.request, image_flip=True))
    pattern = np.zeros((195, 487))  # the stored image's layout
    pattern[:40, :7] = 1.
    np.save(tmp_path / 'mask.npy', pattern.astype(np.uint8))
    np.save(tmp_path / 'dark.npy', 5. * pattern)
    mirrored = replace(plan, geometry=geometry, native_records=True, radial_range=None,
                       mask_path=tmp_path / 'mask.npy', dark_path=tmp_path / 'dark.npy')
    dark = native_detector_image(mirrored, mirrored._dark_receipt)
    np.testing.assert_array_equal(dark > 0, pattern[:, ::-1] > 0)
    np.testing.assert_array_equal(native_mask(mirrored), pattern[:, ::-1] > 0)


def test_a_mirrored_calibration_keeps_its_camera_through_the_save(tmp_path):
    """Frames staged mirrored record it: the session (only then), the fit's provenance, the pyFAI geometry
    Stitch captures and the xrayutilities file all carry the flip."""
    from pyFAI.calibrant import get_calibrant
    from xrd_tools.analysis.stitch_operation import (
        StitchGeometryInput, StitchGeometryKind, capture_stitch_geometry,
    )
    from xrd_tools.core.geometry import DetectorCalibration, Diffractometer, StackedDetectorArm
    from xrd_tools.integrate.refine import ArmFitBounds
    from xrd_tools.integrate.stitch_calibration import (
        fit_calibration, load_calibration_session, save_calibration_pair, save_calibration_session,
    )
    from tests.core.test_arm_fit_bounds import SOURCES, ZEROS, _controls_from, _seed, _subpixel_frames, replace_dist
    from tests.core.test_stacked_arm_extension import ANGLES, WAVELENGTH
    arm = StackedDetectorArm(nu_offset=np.deg2rad(ZEROS['nu']), del_offset=np.deg2rad(ZEROS['del']))
    truth = Diffractometer.from_pyfai_goniometer(arm.to_pyfai_goniometer(
        DetectorCalibration(replace_dist(_seed(3), .39), detector_config={'orientation': 3})), source_motors=SOURCES)
    calibrant = get_calibrant('LaB6')
    calibrant.wavelength = WAVELENGTH
    angles = ANGLES[::2]
    frames = tuple(replace(frame, flip=True) for frame in
                   _subpixel_frames(tmp_path / 'frames', _controls_from(truth, angles, calibrant), _seed(3)))
    for name in ('session', 'plain'):
        (tmp_path / name).mkdir()
    session = save_calibration_session(tmp_path / 'session', frames)
    assert all(row['flip'] is True for row in json.loads(Path(session).read_text())['frames'])
    assert load_calibration_session(session) == frames
    plain = save_calibration_session(tmp_path / 'plain', tuple(replace(frame, flip=False) for frame in frames))
    assert not any('flip' in row for row in json.loads(Path(plain).read_text())['frames'])  # as before
    fit = fit_calibration(frames, arm_model='stacked_extended', bounds=ArmFitBounds(zeros_deg=ZEROS))
    assert fit.rms_q < 1e-8 and fit.asset['provenance']['image_flip'] is True
    assert all(row['flip'] for row in fit.asset['provenance']['frames'])
    saved = save_calibration_pair(fit, tmp_path / 'calibration' / 'arm.json', validation_angles=angles)
    assert saved.xu_path is not None, saved.xu_refusal
    assert json.loads(Path(saved.xu_path).read_text())['acquisition']['image_flip'] is True
    captured = capture_stitch_geometry(StitchGeometryInput(saved.pyfai_path, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
                                                           source_motors=tuple(sorted(SOURCES.items())), image_flip=True))
    assert captured.request.image_flip
    with pytest.raises(ValueError, match='unmirrored camera'):  # the xrayutilities-only fit takes none
        fit_calibration(frames, backend='xu_hist')
