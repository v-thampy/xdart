"""XU asset v3: the pyFAI extended arm transferred to xrayutilities, per-frame solid angle."""
import json

import numpy as np
import pytest

from xrd_tools.analysis.xu_stitch_calibration import (
    XuStitchCalibrationInput, XuStitchCalibrationRefused, _canonical_bytes,
    capture_xu_stitch_calibration, parse_xu_stitch_calibration_bytes,
)
from xrd_tools.core.geometry import Diffractometer
from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm
from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
from xrd_tools.corrections.stack import CorrectionStack
from xrd_tools.integrate.multi import create_multigeometry_integrators_from_geometry
from xrd_tools.integrate.xu_stitch import XuPowderQProvider, resolve_xu_stitch_effective_geometry
from xrd_tools.integrate.xu_transfer import XuTransferRefused, transfer_extended_arm_to_xu
from tests.core.test_stacked_arm_extension import ANGLES, TRUTH, _calibration

SHAPE = (195, 1475)


def _record(**changes):
    params = {**TRUTH, **changes}
    arm = StackedDetectorArm(nu_motor='nu', del_motor='del', **params)
    record = arm.to_pyfai_goniometer(_calibration())
    record['detector_config'] = {'orientation': 3}
    return record


@pytest.fixture(scope='module')
def transferred():
    return transfer_extended_arm_to_xu(_record(), validation_angles=ANGLES)


def _receipt(tmp_path, asset):
    project = tmp_path / 'project'
    target = project / 'calibration' / 'xu' / 'extended.json'
    target.parent.mkdir(parents=True)
    target.write_bytes(_canonical_bytes(asset))
    return capture_xu_stitch_calibration(XuStitchCalibrationInput('calibration/xu/extended.json'),
                                         project_root=project)


def test_the_transfer_passes_its_gate_on_every_pixel(transferred):
    asset, diagnostics = transferred
    assert asset['version'] == 3
    assert diagnostics['max_abs_dq_A^-1'] <= 1e-8 == diagnostics['gate_A^-1']
    assert asset['validation']['validation_pixel_count'] == len(ANGLES) * SHAPE[0] * SHAPE[1]
    assert asset['corrections']['solid_angle_policy'] == 'per_frame_pyfai_extended_arm_v1'
    # xrayutilities' sample displacement is the arm's sample offset, (−s1, s3, s2) in mm
    expected = [-TRUTH['sample_offset1'] * 1e3, TRUTH['sample_offset3'] * 1e3, TRUTH['sample_offset2'] * 1e3]
    np.testing.assert_allclose(asset['acquisition']['sampledis_mm'], expected, atol=1e-6)
    parse_xu_stitch_calibration_bytes(_canonical_bytes(asset))


def test_the_runtime_reproduces_the_arm_q_and_its_per_frame_solid_angle(tmp_path, transferred):
    asset, _ = transferred
    receipt = _receipt(tmp_path, asset)
    diff = Diffractometer.from_pyfai_goniometer(_record(), source_motors={'del_angle': 'del', 'nu_angle': 'nu'})
    rows, cols = np.indices(SHAPE, dtype=float)
    with xu_runtime_session() as session:
        geometry = resolve_xu_stitch_effective_geometry(receipt, session)
        assert geometry.solid_angle is None and geometry.projection.asset_version == 3
        provider = XuPowderQProvider(geometry, asset['acquisition'])
        for delta, nu in ((12., -7.), (38., 16.)):  # not transfer frames
            integrator, = create_multigeometry_integrators_from_geometry(
                diff, {'del': np.array([delta]), 'nu': np.array([nu])})
            lease = provider.frame(nu=nu, del_=delta)
            try:
                q = lease.q_magnitude()
            finally:
                lease.release()
            assert float(np.max(np.abs(q - integrator.qFunction(rows, cols) / 10))) < 1e-8
            # pyFAI's relative solid angle, scaled by the frame's distance (see below)
            distance = (asset['corrections']['solid_angle_arm']['dist_m'] / integrator.dist) ** 2
            np.testing.assert_array_equal(geometry.frame_solid_angle(nu, delta),
                                          CorrectionStack().normalization(integrator, SHAPE) * distance)


def test_the_xu_stitch_kernel_runs_on_a_v3_asset(tmp_path, transferred):
    from xrd_tools.integrate.xu_stitch import run_xu_hist_stitch_1d

    class Source:
        frame_indices = (1, 2, 3)

        def metadata_for(self, label):
            return {'del': 6. + 10 * label, 'nu': -3.}

        def load_frame(self, label):
            return np.full(SHAPE, 10.0, dtype=np.float32)

    receipt = _receipt(tmp_path, transferred[0])
    result = run_xu_hist_stitch_1d(receipt, Source(), frame_indices=(1, 2, 3), q_min_A_inverse=1.0,
                                   q_max_A_inverse=6.0, npt=64, monitor_key=None,
                                   max_frame_bytes=8 * 1024 * 1024)
    occupied = result.diagnostics.normalization > 0
    assert occupied.any() and np.isfinite(result.payload.intensity[occupied]).all()


class _OneFrame:
    frame_indices = (1,)

    def __init__(self, delta, nu, image):
        self.pose, self.image = {'del': delta, 'nu': nu}, image

    def metadata_for(self, label):
        return self.pose

    def load_frame(self, label):
        return self.image


def test_v3_keeps_uniform_radiance_uniform_at_every_pose(tmp_path, transferred):
    """A sample offset changes each frame's distance; its solid angle must follow.

    Counts from a uniform radiance integrate to one level at every pose, as in
    pyFAI MultiGeometry. v3 keeps XU's relative units: the level is that of the
    arm's reference distance.
    """
    from xrd_tools.integrate.xu_stitch import run_xu_hist_stitch_1d

    asset = transferred[0]
    receipt = _receipt(tmp_path, asset)
    diff = Diffractometer.from_pyfai_goniometer(_record(), source_motors={'del_angle': 'del', 'nu_angle': 'nu'})
    radiance, distances = 1e7, []
    for delta, nu in ((6., -10.), (42., 18.)):
        integrator, = create_multigeometry_integrators_from_geometry(
            diff, {'del': np.array([delta]), 'nu': np.array([nu])})
        pixel_area = integrator.detector.pixel1 * integrator.detector.pixel2
        counts = radiance * CorrectionStack().normalization(integrator, SHAPE) * pixel_area / integrator.dist ** 2
        distances.append(integrator.dist)
        result = run_xu_hist_stitch_1d(receipt, _OneFrame(delta, nu, counts), frame_indices=(1,), npt=16,
                                       monitor_key=None, max_frame_bytes=8 * 1024 * 1024)
        occupied = result.diagnostics.normalization > 0
        assert occupied.sum() >= 8
        level = radiance * pixel_area / asset['corrections']['solid_angle_arm']['dist_m'] ** 2
        np.testing.assert_allclose(result.payload.intensity[occupied], level, rtol=1e-9)
    assert abs(distances[1] / distances[0] - 1) > 3e-3  # the offset moves the detector


@pytest.mark.parametrize(('changes', 'message'), [
    ({'nu_scale': np.pi / 180 * 1.001}, 'motor scales'),
    ({'beam_rotation': .01}, 'beam rotation'),
])
def test_the_transfer_refuses_what_xu_cannot_express(changes, message):
    with pytest.raises(XuTransferRefused, match=message):
        transfer_extended_arm_to_xu(_record(**changes), validation_angles=ANGLES)


def test_the_transfer_takes_pyfais_default_orientation():
    # an arm fitted in orientation 4 carries that orientation's nu sign (orientation_signs); XU's
    # model takes orientation 3, as the XU fit does, and the refusal says so (not "fitted scales")
    record = _record(nu_scale=-np.pi / 180)
    record['detector_config'] = {'orientation': 4}
    with pytest.raises(XuTransferRefused, match='orientation'):
        transfer_extended_arm_to_xu(record, validation_angles=ANGLES)


def test_a_failed_gate_and_a_changed_arm_refuse():
    with pytest.raises(XuTransferRefused, match='gate'):
        transfer_extended_arm_to_xu(_record(), validation_angles=ANGLES[:6], gate=1e-16)
    five = StackedDetectorArm(nu_motor='nu', del_motor='del').to_pyfai_goniometer(_calibration())
    with pytest.raises(XuTransferRefused, match='extended'):
        transfer_extended_arm_to_xu(five, validation_angles=ANGLES)


def test_the_asset_binds_its_arm(transferred):
    asset = json.loads(json.dumps(transferred[0]))
    asset['corrections']['solid_angle_arm']['mount_tilt'] += 1e-6
    with pytest.raises(XuStitchCalibrationRefused, match='digest'):
        parse_xu_stitch_calibration_bytes(_canonical_bytes(asset))
    asset = json.loads(json.dumps(transferred[0]))
    asset['validation']['transfer_max_abs_dq_A^-1'] = 2e-8
    with pytest.raises(XuStitchCalibrationRefused, match='gate'):
        parse_xu_stitch_calibration_bytes(_canonical_bytes(asset))


def test_a_guided_xu_extended_fit_saves_a_v3_asset_the_runtime_admits(tmp_path):
    from xrd_tools.integrate.stitch_calibration import fit_calibration, save_calibration_fit
    from tests.core.test_stacked_arm_extension import _ring_controls
    from tests.core.test_stitch_guided_calibration import _frames
    controls, _calibrant = _ring_controls(TRUTH, _calibration(), ANGLES[::2])
    (tmp_path / 'frames').mkdir()
    frames = _frames(tmp_path / 'frames', controls, _calibration().poni)
    seen = []
    fit = fit_calibration(frames, backend='xu_hist', arm_model='stacked_extended',
                          validation_angles=ANGLES, progress=lambda *step: seen.append(step))
    assert fit.backend == 'xu_hist' and fit.asset['calibration']['version'] == 3
    transfer = fit.asset['provenance']['xu_transfer']
    assert transfer['max_abs_dq_A^-1'] <= 1e-8 and transfer['validation_frames'] == len(ANGLES)
    assert fit.asset['provenance']['source_motors'] == {'nu': 'nu', 'del': 'del'}
    assert {stage for stage, _done, _total in seen} == {'Transferring to xrayutilities', 'Validating the transfer'}
    path = save_calibration_fit(fit, tmp_path / 'calibration' / 'xu-extended.json')
    assert json.loads(open(path).read())['version'] == 3
