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
from xrd_tools.integrate.xu_transfer import XuTransferRefused, transfer_arm_to_xu
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
    return transfer_arm_to_xu(_record(), validation_angles=ANGLES)


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
    # one fit on the coarse virtual detector, from one tilt azimuth, is enough here
    assert diagnostics['method'] == 'coarse_fit' and len(diagnostics['attempts']) == 1
    # with the motor offsets held at the arm's, the XU mount is the arm's own: unique, not a member
    # of the del-offset/tilt valley (whose far members leave the v3 bounds)
    detector = asset['detector']
    assert detector['detrot_deg'] == pytest.approx(np.degrees(TRUTH['mount_roll']), abs=1e-6)
    assert detector['tilt_deg'] == pytest.approx(np.degrees(TRUTH['mount_tilt']), abs=1e-6)
    assert asset['acquisition']['detector_offsets_deg'] == pytest.approx(
        {'del': -np.degrees(TRUTH['del_offset']), 'nu': -np.degrees(TRUTH['nu_offset'])}, abs=1e-12)
    # and Stitch's own XU path agrees with pyFAI in |q| and in the polarization factor
    assert diagnostics['stitch_path_max_abs_dq_A^-1'] <= 1e-8
    assert diagnostics['polarization_max_relative'] <= 1e-6 == diagnostics['polarization_gate']
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
        transfer_arm_to_xu(_record(**changes), validation_angles=ANGLES)


def test_the_transfer_takes_pyfais_default_orientation():
    # an arm fitted in orientation 4 carries that orientation's nu sign (orientation_signs); XU's
    # model takes orientation 3, as the XU fit does, and the refusal says so (not "fitted scales")
    record = _record(nu_scale=-np.pi / 180)
    record['detector_config'] = {'orientation': 4}
    with pytest.raises(XuTransferRefused, match='orientation'):
        transfer_arm_to_xu(record, validation_angles=ANGLES)


def test_a_failed_gate_refuses(monkeypatch):
    with pytest.raises(XuTransferRefused, match='gate'):
        transfer_arm_to_xu(_record(), validation_angles=ANGLES[:6], gate=1e-16)
    from xrd_tools.integrate import xu_transfer
    monkeypatch.setattr(xu_transfer, 'POLARIZATION_GATE', 1e-16)  # the polarization gate is applied
    with pytest.raises(XuTransferRefused, match='polarization'):
        transfer_arm_to_xu(_record(), validation_angles=ANGLES[:6])


def test_a_five_parameter_arm_transfers_by_formula(tmp_path):
    """No fit: cch = poni/pitch − 0.5, the same distance, the offsets negated (was refused: the
    transfer took only the extended arm)."""
    five = StackedDetectorArm(nu_motor='nu', del_motor='del', nu_offset=.02, del_offset=-.05)
    asset, diagnostics = transfer_arm_to_xu(five.to_pyfai_goniometer(_calibration()), validation_angles=ANGLES)
    assert diagnostics['method'] == 'formula' and diagnostics['fit_frames'] == 0
    assert diagnostics['max_abs_dq_A^-1'] <= 1e-8 and diagnostics['polarization_max_relative'] <= 1e-6
    detector, acquisition = asset['detector'], asset['acquisition']
    assert detector['tilt_deg'] == detector['detrot_deg'] == 0 and acquisition['sampledis_mm'] == [0, 0, 0]
    assert detector['cch1_px'] == pytest.approx(.035 / 172e-6 - .5, abs=1e-9)
    assert acquisition['detector_offsets_deg'] == pytest.approx({'del': 180 / np.pi * .05, 'nu': -180 / np.pi * .02})
    _receipt(tmp_path, asset)  # the runtime admits it


@pytest.mark.parametrize('held', ['nu', 'del'])
def test_a_single_2theta_arm_transfers_by_formula_and_stitches(tmp_path, held):
    """fourc/twoc: vertical (nu held at zero) or horizontal (del held, the scan's del column
    turning the nu circle); the held circle is at zero in the asset and in every frame."""
    from xrd_tools.integrate.xu_stitch import run_xu_hist_stitch_1d
    arm = (StackedDetectorArm(nu_motor=None, del_motor='del', nu_offset=.004, del_offset=-.05) if held == 'nu'
           else StackedDetectorArm(nu_motor='del', del_motor=None, nu_offset=-.03, del_offset=.006))
    motors = ('del', None) if held == 'nu' else (None, 'del')
    angles = [(d, 0.) for d in (6., 14., 22., 30., 38.)] if held == 'nu' else [(0., n) for n in (6., 14., 22., 30., 38.)]
    asset, diagnostics = transfer_arm_to_xu(arm.to_pyfai_goniometer(_calibration()), validation_angles=angles,
                                            source_motors=motors)
    assert diagnostics['method'] == 'formula' and diagnostics['polarization_max_relative'] <= 1e-6
    assert asset['acquisition']['source_motors'] == dict(zip(('del', 'nu'), motors))
    assert asset['validation']['validated_scan_domain_source_deg'][held] == [0, 0]

    class Source:
        frame_indices = (1, 2)

        def metadata_for(self, label):
            return {'del': 10. + 8 * label}  # the only motor the scan needs

        def load_frame(self, label):
            return np.full(SHAPE, 10.0, dtype=np.float32)

    result = run_xu_hist_stitch_1d(_receipt(tmp_path, asset), Source(), frame_indices=(1, 2), npt=64,
                                   monitor_key=None, max_frame_bytes=8 * 1024 * 1024)
    occupied = result.diagnostics.normalization > 0
    assert occupied.any() and np.isfinite(result.payload.intensity[occupied]).all()


def test_an_independent_angle_geometry_is_refused():
    from tests.core.test_single_detector_arm import _affine
    with pytest.raises(XuTransferRefused, match='stacked'):
        transfer_arm_to_xu(_affine(['del_angle', 'nu_angle'], detector='Pilatus300kw'), validation_angles=ANGLES)


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


@pytest.mark.parametrize("pose", [{'eta': .4, 'chi': 90., 'mu': 0.}, {'eta': .4, 'chi': 88., 'mu': .5}])
def test_xu_gi_cuts_bin_pyfais_fiber_coordinates_and_leave_iq_alone(tmp_path, transferred, pose):
    """GI Stitch 1D with xrayutilities (maintainer, 2026-09-29): every frame's sample normal comes
    from psic's four sample circles (17-2's mounting: chi 90, eta the incidence; here also tilted by
    chi and mu), and each cut bins exactly the pixels pyFAI's fiber model puts there (orientation 4,
    the incidence and tilt solved from the same normal), sample offset and all; Qip + Qoop gives
    exactly the two single-cut results; I(q) beside them is unchanged."""
    from pyFAI.units import get_unit_fiber
    from xrd_tools.integrate.gid import _fiber_from_integrator
    from xrd_tools.integrate.stitch_gi import GI_UNITS, StitchGI
    from xrd_tools.integrate.xu_stitch import run_xu_hist_stitch_1d
    angles = {1: (12., -7.), 2: (26., 4.), 3: (38., 16.)}

    class Source:
        frame_indices = (1, 2, 3)

        def metadata_for(self, label):
            return {'del': angles[label][0], 'nu': angles[label][1], **pose}

        def load_frame(self, label):
            return np.full(SHAPE, 10.0, dtype=np.float32)

    ranges = {'q_ip': (-4., 4.), 'q_oop': (0., 1.5)}
    receipt = _receipt(tmp_path, transferred[0])
    common = dict(frame_indices=(1, 2, 3), npt=48, radial_range=(.5, 6.), monitor_key=None,
                  max_frame_bytes=8 * 1024 * 1024)

    def stitch(*cuts):
        gi = None if not cuts else StitchGI(cuts, incidence_motor='eta', tilt_motor='chi', ip_range=ranges['q_ip'],
                                            oop_range=ranges['q_oop'], sample_motors=(('mu', 'mu'), ('phi', None)))
        return run_xu_hist_stitch_1d(receipt, Source(), gi=gi, **common)

    result, plain = stitch('q_ip', 'q_oop'), stitch()
    np.testing.assert_array_equal(result.payload.intensity, plain.payload.intensity)
    assert plain.gi_cuts == () and [cut.unit for cut, _ in result.gi_cuts] == [GI_UNITS['q_ip'], GI_UNITS['q_oop']]
    diff = Diffractometer.from_pyfai_goniometer(_record(), source_motors={'del_angle': 'del', 'nu_angle': 'nu'})
    expected = {name: np.zeros(48) for name in ranges}
    gi = StitchGI(('q_ip',), incidence_motor='eta', tilt_motor='chi', sample_motors=(('mu', 'mu'), ('phi', None)))
    incidence, tilt = gi.pyfai_angles(gi.normal(pose))
    assert abs(tilt) == pytest.approx(0. if pose['chi'] == 90. else 2., abs=.01)
    for (delta, nu) in angles.values():
        integrator, = create_multigeometry_integrators_from_geometry(diff, {'del': np.array([delta]), 'nu': np.array([nu])})
        fiber = _fiber_from_integrator(integrator, incident_angle=incidence, tilt_angle=tilt, sample_orientation=4,
                                       angle_unit='deg')
        unit = dict(incident_angle=incidence, tilt_angle=tilt, sample_orientation=4, angle_unit='deg')
        values = {'q_ip': np.asarray(fiber.array_from_unit(SHAPE, 'center', get_unit_fiber('qip_A^-1', **unit), scale=True)),
                  'q_oop': np.asarray(fiber.array_from_unit(SHAPE, 'center', get_unit_fiber('qoop_A^-1', **unit), scale=True))}
        mask = integrator.detector.mask
        for name, other in (('q_ip', 'q_oop'), ('q_oop', 'q_ip')):
            low, high = ranges[other]
            keep = ((values[other] >= low) & (values[other] <= high) & np.isfinite(values[name])
                    & (True if mask is None else ~np.asarray(mask, bool)))
            expected[name] += np.histogram(values[name][keep], np.linspace(*ranges[name], 49))[0]
    for name, (cut, diagnostics) in zip(('q_ip', 'q_oop'), result.gi_cuts):
        np.testing.assert_array_equal(diagnostics.coverage, expected[name])
        (single, single_diagnostics), = stitch(name).gi_cuts
        np.testing.assert_array_equal(cut.intensity, single.intensity)
        np.testing.assert_array_equal(diagnostics.normalization, single_diagnostics.normalization)


@pytest.mark.parametrize('orientation, poses', [
    (4, [(0., .4, 90., 0.), (.5, .4, 88., -1.7), (0., .3, 85., 3.)]),  # 17-2: chi 90, eta incidence
    (3, [(0., .4, 88., 0.)]),  # the same mounting, q_ip the other way
    (7, [(.4, 0., 0., 0.), (.4, .5, -1.5, -1.7)]),  # chi 0, mu incidence
    (6, [(.4, 0., 2., 0.)])])
def test_pyfai_takes_the_incidence_and_tilt_that_give_the_circles_surface_normal(tmp_path, transferred, orientation,
                                                                               poses):
    """Maintainer, 2026-09-29: "pyfai should get the tilt from the data as well just like the incidence
    angle". psic_normal is xrayutilities' own sample transform; pyFAI's fiber q_ip/q_oop at the incidence
    and tilt StitchGI solves from that normal equal what both backends bin (q_oop = q·n̂, q_ip signed
    along the orientation's horizontal axis in the surface), per pixel, on off-axis frames of the
    extended arm, level and tilted."""
    from pyFAI.units import get_unit_fiber
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.integrate.gid import _fiber_from_integrator
    from xrd_tools.integrate.stitch_gi import StitchGI, psic_normal
    asset = transferred[0]
    frames = [(10., 0.), (30., 14.), (6., -7.)]
    diff = Diffractometer.from_pyfai_goniometer(_record(), source_motors={'del_angle': 'del', 'nu_angle': 'nu'})
    integrators = create_multigeometry_integrators_from_geometry(
        diff, {'del': np.array([d for d, _ in frames]), 'nu': np.array([n for _, n in frames])})
    gi = StitchGI(('q_ip',), incidence_motor='eta', tilt_motor='chi', sample_orientation=orientation,
                  sample_motors=(('mu', 'mu'), ('phi', 'phi')))
    with xu_runtime_session() as runtime:
        geometry = resolve_xu_stitch_effective_geometry(_receipt(tmp_path, asset), runtime)
        provider = XuPowderQProvider(geometry, asset['acquisition'])
        for mu, eta, chi, phi in poses:
            normal = gi.normal({'mu': mu, 'eta': eta, 'chi': chi, 'phi': phi})
            np.testing.assert_allclose(normal, geometry.hxrd._A2QConversion.transformSample2Lab(
                np.array([0., 0., 1.]), mu, eta, chi, phi), rtol=0, atol=1e-15)
            np.testing.assert_allclose(normal, psic_normal(mu, eta, chi, phi), rtol=0, atol=0)
            incidence, tilt = gi.pyfai_angles(normal)
            unit = dict(incident_angle=incidence, tilt_angle=tilt, sample_orientation=orientation, angle_unit='deg')
            for (delta, nu), integrator in zip(frames, integrators):
                lease = provider.frame(nu=nu, del_=delta)  # the XU kernel's own path
                try:
                    q_ip, q_oop = lease.gi_coordinates(normal, gi.in_plane_axis(normal))
                finally:
                    lease.release()
                fiber = _fiber_from_integrator(integrator, **unit)
                for name, ours in (('qip_A^-1', q_ip), ('qoop_A^-1', q_oop)):
                    theirs = np.asarray(fiber.array_from_unit(SHAPE, 'center', get_unit_fiber(name, **unit), scale=True))
                    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-8, err_msg=f'{name} {orientation} {chi}')


def test_an_xu_auto_gi_range_windows_every_pixel_as_a_covering_one_does(tmp_path, transferred):
    """C12 for xrayutilities: a cut's window on Auto keeps every pixel, as a covering range does."""
    from xrd_tools.integrate.stitch_gi import StitchGI
    from xrd_tools.integrate.xu_stitch import run_xu_hist_stitch_1d
    angles = {1: (12., -7.), 2: (26., 4.)}

    class Source:
        frame_indices = (1, 2)

        def metadata_for(self, label):
            return {'del': angles[label][0], 'nu': angles[label][1]}

        def load_frame(self, label):
            return np.full(SHAPE, 10.0, dtype=np.float32)

    receipt = _receipt(tmp_path, transferred[0])
    common = dict(frame_indices=(1, 2), npt=48, radial_range=(.5, 6.), monitor_key=None,
                  max_frame_bytes=8 * 1024 * 1024)
    for name, own, other in (('q_ip', 'ip_range', 'oop_range'), ('q_oop', 'oop_range', 'ip_range')):
        bins = {own: (-4., 4.) if name == 'q_ip' else (-1., 3.)}
        auto = run_xu_hist_stitch_1d(receipt, Source(), gi=StitchGI((name,), incident_angle_deg=.4, **bins), **common)
        wide = run_xu_hist_stitch_1d(receipt, Source(), gi=StitchGI((name,), incident_angle_deg=.4, **bins,
                                                                   **{other: (-50., 50.)}), **common)
        (cut, diagnostics), = auto.gi_cuts
        (wide_cut, wide_diagnostics), = wide.gi_cuts
        assert diagnostics.coverage.sum() > 0
        np.testing.assert_array_equal(diagnostics.coverage, wide_diagnostics.coverage)
        np.testing.assert_array_equal(cut.intensity, wide_cut.intensity)
