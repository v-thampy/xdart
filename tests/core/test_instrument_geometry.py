"""The diffractometer geometry and its motor names (Stitch, its calibration, and RSM)."""
import pytest

from xrd_tools.session.instrument import InstrumentGeometry


def test_each_geometry_names_its_circles_with_the_standard_motors():
    assert InstrumentGeometry().motors == (('mu', 'mu'), ('eta', 'eta'), ('chi', 'chi'), ('phi', 'phi'),
                                           ('nu', 'nu'), ('del', 'del'))
    assert InstrumentGeometry('fourc').circles == ('th', 'chi', 'phi', 'tth')
    assert InstrumentGeometry('twoc').circles == ('th', 'tth')
    with pytest.raises(ValueError, match='unknown'):
        InstrumentGeometry('psic_halpha')  # halpha is a motor name, not a geometry
    with pytest.raises(ValueError, match='no circle'):
        InstrumentGeometry('twoc', (('del', 'del'),))


def test_the_scans_motors_fill_each_circle():
    picked = InstrumentGeometry.pick('psic', ['Delta', 'nu', 'halpha', 'chi', 'phi', 'mu', 'I0'])
    assert picked.motor('del') == 'Delta'  # an alias, spelled as the scan spells it
    assert picked.motor('eta') == 'halpha'  # the incidence circle follows Int's GI preference
    fourc = InstrumentGeometry.pick('fourc', ['tth', 'omega', 'chi', 'phi'])
    assert (fourc.motor('th'), fourc.motor('tth')) == ('omega', 'tth')
    kept = InstrumentGeometry('psic', (('del', 'del2'),))  # a choice the scan has stays
    assert InstrumentGeometry.pick('psic', ['del', 'del2', 'nu'], keep=kept).motor('del') == 'del2'
    assert InstrumentGeometry.pick('psic', ['del', 'nu'], keep=kept).motor('del') == 'del'


def test_a_calibration_files_positions_name_their_circle():
    geometry = InstrumentGeometry('psic', (('del', 'tth2'),))
    names = ('del_angle', 'nu_value', 'delta_pos', 'tth2', 'chi', 'x_motor')
    assert [geometry.circle_for_position(name) for name in names] == ['del', 'nu', 'del', 'del', 'chi', None]


def test_stitch_takes_psics_detector_arm_and_the_core_preset_carries_the_names():
    geometry = InstrumentGeometry('psic', (('del', 'delta'), ('eta', 'halpha')))
    assert geometry.stitch_arm() == ('delta', 'nu')
    diff = geometry.diffractometer()
    assert (diff.rot2.source_motor, diff.rot1.source_motor, diff.incident_angle.source_motor) == (
        'delta', 'nu', 'halpha')
    assert InstrumentGeometry.from_record(geometry.to_record()) == geometry


def test_fourc_and_twoc_are_psics_arm_with_nu_held_at_zero():
    # psic with mu = nu = 0 is the four-circle: the 2θ circle plays del, and there is no nu
    fourc = InstrumentGeometry('fourc', (('tth', 'ttheta'),))
    assert fourc.stitch_arm() == ('ttheta', None)
    assert InstrumentGeometry('twoc').stitch_arm() == ('tth', None)
    # psic's names (a calibration's del_angle, the direct-beam "del") are the circle playing them
    names = ('del_angle', 'delta', 'tth', 'two_theta', 'ttheta', 'eta', 'nu_angle')
    assert [fourc.circle_for_position(name) for name in names] == ['tth'] * 5 + ['th', None]
    assert [fourc.arm_motor(name) for name in ('del', 'tth', 'ttheta', 'nu', 'th')] == ['ttheta'] * 3 + [None] * 2
    assert InstrumentGeometry('psic', (('del', 'delta'),)).arm_motor('del_value') == 'delta'
    assert InstrumentGeometry('psic').arm_motor('nu_angle') == 'nu'
    # a 2θ arm named as psic's del (a psic run as twoc, nu parked) is picked for tth
    picked = InstrumentGeometry.pick('twoc', ['Delta', 'eta', 'nu', 'mu'])
    assert (picked.motor('tth'), picked.motor('th')) == ('Delta', 'eta')
    assert InstrumentGeometry.pick('fourc', ['tth', 'del', 'th']).motor('tth') == 'tth'  # its own name first


def test_the_scattering_plane_says_which_psic_circle_a_2theta_arm_plays():
    vertical, horizontal = InstrumentGeometry('twoc'), InstrumentGeometry('twoc', plane='horizontal')
    assert (vertical.stitch_arm(), horizontal.stitch_arm()) == (('tth', None), (None, 'tth'))
    assert horizontal.as_psic() == {'th': 'mu', 'tth': 'nu'} and InstrumentGeometry().as_psic() == {}
    assert [horizontal.circle_for_position(name) for name in ('nu_angle', 'del_angle', 'mu')] == ['tth', None, 'th']
    assert [horizontal.arm_role(name) for name in ('nu', 'tth', 'del')] == ['nu', 'nu', None]
    picked = InstrumentGeometry.pick('twoc', ['nu', 'mu', 'del'], keep=horizontal)
    assert (picked.plane, picked.motor('tth'), picked.motor('th')) == ('horizontal', 'nu', 'mu')
    fourc = InstrumentGeometry('fourc', (('tth', 'gam'),), 'horizontal')
    assert InstrumentGeometry.from_record(fourc.to_record()) == fourc
    assert fourc.to_record()['plane'] == 'horizontal' and 'plane' not in InstrumentGeometry().to_record()
    older = {'geometry': 'twoc', 'motors': {'th': 'th', 'tth': 'tth'}}  # a round-12 record: no plane
    assert InstrumentGeometry.from_record(older) == vertical
    with pytest.raises(ValueError, match='fourc and twoc'):
        InstrumentGeometry('psic', plane='horizontal')
    with pytest.raises(ValueError, match='unknown scattering plane'):
        InstrumentGeometry('twoc', plane='tilted')


def _affine(positions):
    """An independent-Euler goniometer file: rot1 follows its last position, rot2 its first."""
    names = ['dist', 'poni1', 'poni2', 'rot1_offset', 'rot1_scale', 'rot2_offset', 'rot2_scale']
    rot1 = f'rot1_scale * {positions[-1]} + rot1_offset' if len(positions) > 1 else 'rot1_offset'
    return {'content': 'Goniometer calibration v2', 'detector': 'Pilatus300kw', 'wavelength': 1e-10,
            'param_names': names, 'param': [.39, .035, .045, 0., .0174533, 0., .0174533],
            'pos_names': list(positions),
            'trans_function': {'content': 'GeometryTransformation', 'param_names': names,
                               'pos_names': list(positions), 'dist_expr': 'dist', 'poni1_expr': 'poni1',
                               'poni2_expr': 'poni2', 'rot1_expr': rot1,
                               'rot2_expr': f'rot2_scale * {positions[0]} + rot2_offset', 'rot3_expr': '0.0',
                               'constants': {}}}


def test_a_calibrations_arm_comes_from_its_transformation_not_its_names():
    from xrd_tools.core.containers import PONI
    from xrd_tools.core.geometry import DetectorCalibration
    from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm
    from xrd_tools.session.instrument import calibration_arm
    calibration = DetectorCalibration(PONI(dist=.39, poni1=.035, poni2=.045, wavelength=1e-10,
                                           detector='Pilatus300kw'), detector_config={'orientation': 3})
    record = lambda **motors: StackedDetectorArm(**motors).to_pyfai_goniometer(calibration)  # noqa: E731
    assert calibration_arm(record(), {'del_angle': 'del', 'nu_angle': 'nu'}) == ({'del', 'nu'}, set())
    assert calibration_arm(record(nu_motor=None), {'del_angle': 'tth'}) == ({'del'}, {'nu'})
    assert calibration_arm(record(del_motor=None), {'nu_angle': 'tth'}) == ({'nu'}, {'del'})
    # custom position names bound explicitly (Codex round 12 F1): the transformation says del and nu
    assert calibration_arm(_affine(['dval', 'nval']), {'dval': 'del', 'nval': 'nu'}) == ({'del', 'nu'}, set())
    assert calibration_arm(_affine(['pos']), {'pos': 'tth'}) == ({'del'}, set())  # rot2 alone: del
    turned = _affine(['dval', 'nval'])
    turned['trans_function']['rot3_expr'] = 'rot2_scale * dval'
    with pytest.raises(ValueError, match='beam-axis rotation'):
        calibration_arm(turned, {'dval': 'del', 'nval': 'nu'})
    with pytest.raises(ValueError, match='cannot be read'):
        calibration_arm({'content': 'x'}, {})


def test_a_calibration_must_drive_the_geometrys_arm():
    fourc, twoc_h, psic = InstrumentGeometry('fourc'), InstrumentGeometry('twoc', plane='horizontal'), InstrumentGeometry()
    both, vertical, horizontal = ({'del', 'nu'}, set()), ({'del'}, {'nu'}), ({'nu'}, {'del'})
    assert 'fourc has one detector circle' in fourc.arm_mismatch(*both)
    assert 'twoc has one detector circle' in twoc_h.arm_mismatch(*both)
    assert fourc.arm_mismatch(*vertical) is None and twoc_h.arm_mismatch(*horizontal) is None
    assert 'Choose the horizontal scattering plane' in fourc.arm_mismatch(*horizontal)
    assert 'Choose the vertical scattering plane' in twoc_h.arm_mismatch(*vertical)
    assert 'nu held at zero' in psic.arm_mismatch(*vertical)
    assert 'del held at zero' in psic.arm_mismatch(*horizontal)
    assert psic.arm_mismatch(*both) is None
    # a legacy one-position file moves del and holds nothing: psic keeps admitting it
    assert psic.arm_mismatch({'del'}) is None and fourc.arm_mismatch({'del'}) is None
    assert 'vertical' in twoc_h.arm_mismatch({'del'})


def test_each_circle_turns_about_psics_axis_for_the_circle_it_plays():
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.session.instrument import LAB_FRAME, PSIC_AXES
    psic = Diffractometer.psic()  # the same lists as the RSM notebooks, in one lab frame
    assert tuple(PSIC_AXES[c] for c in ('mu', 'eta', 'chi', 'phi')) == tuple(psic.sample_circles)
    assert tuple(PSIC_AXES[c] for c in ('nu', 'del')) == tuple(psic.detector_circles)
    assert tuple(psic.r_i) == (0., 1., 0.) and LAB_FRAME.startswith('x up, y along the beam')
    assert InstrumentGeometry().circle_axes() == (('mu', 'x+'), ('eta', 'z-'), ('chi', 'y+'), ('phi', 'z-'),
                                                  ('nu', 'x+'), ('del', 'z-'))
    assert InstrumentGeometry('fourc').circle_axes() == (('th', 'z-'), ('chi', 'y+'), ('phi', 'z-'), ('tth', 'z-'))
    assert InstrumentGeometry('fourc', plane='horizontal').circle_axes() == (
        ('th', 'x+'), ('chi', 'y+'), ('phi', 'z-'), ('tth', 'x+'))
    assert InstrumentGeometry('twoc', plane='horizontal').circle_axes() == (('th', 'x+'), ('tth', 'x+'))


def test_reversing_mu_chi_and_nu_is_psics_mirror_image():
    """17-2's record (the maintainer, 2026-09-30) reverses mu, chi and nu, the circles that turn about x and y: psic's
    left-right mirror image. For now phi is reversed too (2026-10-01), which the mirror does not undo."""
    from xrd_tools.session.instrument import MIRROR_CIRCLES, mirrored_axis
    assert MIRROR_CIRCLES == ('mu', 'chi', 'nu')
    assert [mirrored_axis(axis) for axis in ('x+', 'x-', 'y+', 'y-', 'z+', 'z-')] == [
        'x-', 'x+', 'y-', 'y+', 'z+', 'z-']
    pure = InstrumentGeometry('psic', reversed_circles=('mu', 'chi', 'nu'))
    assert pure.is_mirror() and pure.mirror_residual() == ()
    now = InstrumentGeometry('psic', reversed_circles=('phi', 'nu', 'mu', 'chi'))
    assert now.reversed_circles == ('mu', 'chi', 'phi', 'nu') and now.is_mirror()
    assert now.mirror_residual() == ('phi',)
    # the mirror read through the record's own axes: every circle but the residual turns as psic's
    assert {circle: mirrored_axis(axis) for circle, axis in now.circle_axes()} == {
        'mu': 'x+', 'eta': 'z-', 'chi': 'y+', 'phi': 'z+', 'nu': 'x+', 'del': 'z-'}
    partial = InstrumentGeometry('psic', reversed_circles=('mu', 'chi'))
    assert not partial.is_mirror() and partial.mirror_residual() == ('mu', 'chi')
    assert not InstrumentGeometry('psic').is_mirror() and InstrumentGeometry('psic').mirror_residual() == ()
    assert not InstrumentGeometry('fourc', plane='horizontal', reversed_circles=('th', 'chi', 'tth')).is_mirror()
