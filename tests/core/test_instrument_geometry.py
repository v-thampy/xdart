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


def test_a_calibration_must_drive_the_geometrys_arm():
    psic_arm = ('del_angle', 'nu_angle')
    assert 'one detector circle' in InstrumentGeometry('fourc').arm_mismatch(psic_arm)
    assert 'one detector circle' in InstrumentGeometry('twoc').arm_mismatch(('del_value', 'nu_value'))
    assert InstrumentGeometry('twoc').arm_mismatch(('del_angle',), single_arm=True) is None
    assert 'nu held at zero' in InstrumentGeometry('psic').arm_mismatch(('del_angle',), single_arm=True)
    assert InstrumentGeometry('psic').arm_mismatch(psic_arm) is None
    # custom positions carry the user's own binding; a legacy one-position file is unchanged
    assert InstrumentGeometry('fourc').arm_mismatch(('dval', 'nval')) is None
    assert InstrumentGeometry('psic').arm_mismatch(('del_value',)) is None
