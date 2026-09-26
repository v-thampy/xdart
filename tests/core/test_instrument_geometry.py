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
    with pytest.raises(ValueError, match='single 2θ arm'):
        InstrumentGeometry('twoc').stitch_arm()
