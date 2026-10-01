"""Each circle's sense of rotation (maintainer, 2026-09-30: "the sense of rotation for each motor should be
selectable"; "let's just stick with swapping the signs for now"; "yes default to standard now").

17-2's psic turns [mu, eta, chi, phi], [nu, del] about [x-, z-, y-, z-], [x-, z-] until November 2026; psic's
standard is [x+, z-, y+, z-], [x+, z-]. A reversed circle turns about psic's reversed axis: the detector arm's
motor scale for it is negative, its xrayutilities axis is reversed, and grazing incidence negates its angle.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.geometry import DetectorCalibration, Diffractometer, StackedDetectorArm
from xrd_tools.integrate.refine import ArmFitBounds
from xrd_tools.session.instrument import InstrumentGeometry
from tests.core.test_arm_fit_bounds import SOURCES, ZEROS, _controls_from, _seed, _subpixel_frames, replace_dist
from tests.core.test_stacked_arm_extension import ANGLES, WAVELENGTH


def test_17_2s_senses_are_psics_axes_reversed_and_saved_only_when_given():
    standard, beamline = InstrumentGeometry(), InstrumentGeometry('psic', reversed_circles=('nu', 'chi', 'mu'))
    assert [axis for _circle, axis in standard.circle_axes()] == ['x+', 'z-', 'y+', 'z-', 'x+', 'z-']
    assert [axis for _circle, axis in beamline.circle_axes()] == ['x-', 'z-', 'y-', 'z-', 'x-', 'z-']
    assert beamline.reversed_circles == ('mu', 'chi', 'nu')  # in the geometry's order
    assert beamline.arm_senses() == {'nu': -1, 'del': 1} and standard.arm_senses() == {'nu': 1, 'del': 1}
    record = beamline.to_record()
    assert record['reversed'] == ['mu', 'chi', 'nu'] and 'reversed' not in standard.to_record()
    assert InstrumentGeometry.from_record(record) == beamline
    assert InstrumentGeometry.from_record({'geometry': 'psic', 'motors': {}}) == standard  # older records
    # a 2θ arm's sense is that of the psic circle it plays
    assert InstrumentGeometry('twoc', plane='horizontal', reversed_circles=('tth',)).arm_senses() == {'nu': -1}
    assert InstrumentGeometry('fourc', reversed_circles=('tth',)).circle_axes()[-1] == ('tth', 'z+')
    with pytest.raises(ValueError, match='no circle'):
        InstrumentGeometry('twoc', reversed_circles=('nu',))
    # a scan's motors keep the senses chosen
    assert InstrumentGeometry.pick('psic', ('nu', 'del'), keep=beamline).reversed_circles == ('mu', 'chi', 'nu')


@pytest.mark.parametrize('flipped', [('nu',), ('del',), ('nu', 'del')])
def test_a_reversed_detector_circle_is_fitted_saved_and_transferred_to_xrayutilities(tmp_path, flipped):
    """An arm whose circle turns the other way: the fit told so recovers it exactly (its zeros, and 2θ and χ at
    every pixel after a save and reload), and its xrayutilities file turns that circle about the reversed axis,
    through the transfer's every-pixel gates. Told psic's standard sense, the same frames do not fit."""
    from pyFAI.calibrant import get_calibrant
    from xrd_tools.integrate.multi import create_multigeometry_integrators_from_geometry
    from xrd_tools.integrate.stitch_calibration import fit_calibration, save_calibration_pair
    senses = {circle: -1 if circle in flipped else 1 for circle in ('nu', 'del')}
    arm = StackedDetectorArm(nu_scale=senses['nu'] * np.pi / 180, del_scale=senses['del'] * np.pi / 180,
                             nu_offset=-senses['nu'] * np.deg2rad(ZEROS['nu']),
                             del_offset=-senses['del'] * np.deg2rad(ZEROS['del']),
                             mount_tilt=np.deg2rad(.4), mount_roll=np.deg2rad(-.3),
                             sample_offset1=2e-5, sample_offset2=-1e-5, sample_offset3=3e-5)
    assert arm.motor_zeros() == pytest.approx(ZEROS)
    truth = Diffractometer.from_pyfai_goniometer(arm.to_pyfai_goniometer(
        DetectorCalibration(replace_dist(_seed(3), .39), detector_config={'orientation': 3})), source_motors=SOURCES)
    calibrant = get_calibrant('LaB6')
    calibrant.wavelength = WAVELENGTH
    angles = ANGLES[::2]
    frames = _subpixel_frames(tmp_path / 'frames', _controls_from(truth, angles, calibrant), _seed(3))
    instrument = InstrumentGeometry('psic', reversed_circles=flipped).to_record()
    fit = fit_calibration(frames, arm_model='stacked_extended', bounds=ArmFitBounds(zeros_deg=ZEROS),
                          instrument_geometry=instrument)
    assert fit.rms_q < 1e-8, fit.message
    saved = save_calibration_pair(fit, tmp_path / 'calibration' / 'arm.json', validation_angles=angles)
    loaded = Diffractometer.from_pyfai_goniometer(json.loads(Path(saved.pyfai_path).read_text()),
                                                  source_motors=SOURCES)
    assert loaded.stacked_arm.motor_zeros() == pytest.approx(ZEROS, abs=1e-7)
    assert (loaded.stacked_arm.nu_scale, loaded.stacked_arm.del_scale) == (arm.nu_scale, arm.del_scale)
    motors = {'del': np.array([a[0] for a in angles]), 'nu': np.array([a[1] for a in angles])}
    rows, columns = np.indices((195, 1475), dtype=float)
    for mine, theirs in zip(create_multigeometry_integrators_from_geometry(loaded, motors),
                            create_multigeometry_integrators_from_geometry(truth, motors)):
        assert np.abs(mine.twoThetaArray() - theirs.twoThetaArray()).max() < 1e-8
        # every pixel's lab position (so its χ too, which is singular where the beam meets the detector)
        here, there = (np.stack(ai.calc_pos_zyx(d1=rows, d2=columns)) for ai in (mine, theirs))
        assert np.abs(here - there).max() < 1e-9  # metres
    assert saved.xu_path is not None and saved.xu_refusal == '', saved.xu_refusal
    xu = json.loads(Path(saved.xu_path).read_text())
    assert xu['xrayutilities']['detector_axes'] == ['x-' if 'nu' in flipped else 'x+',
                                                    'z+' if 'del' in flipped else 'z-']
    assert fit.asset['provenance']['instrument_geometry']['reversed'] == list(flipped)
    try:  # psic's standard senses: the frames cannot be fitted
        wrong = fit_calibration(frames, arm_model='stacked', bounds=ArmFitBounds(zeros_deg=ZEROS), max_rms_q=10.)
    except ValueError:
        pass
    else:
        assert wrong.rms_q > 1e-3, wrong.message


def test_a_standard_xrayutilities_file_is_unchanged():
    """A transfer with psic's senses writes the reference conventions as before (its axes are the reference's)."""
    from xrd_tools.analysis.xu_stitch_calibration import canonical_surface_resource_bytes
    reference = json.loads(canonical_surface_resource_bytes())['xrayutilities']
    assert reference['detector_axes'] == ['x+', 'z-'] and reference['sample_axes'] == ['x+', 'z-', 'y+', 'z-']


def test_a_refinement_refuses_a_seed_that_turns_a_circle_the_other_way(tmp_path):
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    seed = StackedDetectorArm().to_pyfai_goniometer(
        DetectorCalibration(replace_dist(_seed(3), .39), detector_config={'orientation': 3}))
    spec = SourceSpec(tmp_path / 'scan.spec', SourceKind.SPEC, options={'scan': '1'})
    with pytest.raises(ValueError, match=r'turns nu about x\+, the Diffractometer about x-'):
        refine_arm_from_scan(spec, seed,
                             instrument_geometry=InstrumentGeometry('psic', reversed_circles=('nu',)).to_record())


def test_the_rough_arm_turns_a_reversed_circle_the_other_way():
    """The rough geometry places the beam on its pixel at the zeros either way; 2° of nu then moves it as far
    across the detector, the other way."""
    from xrd_tools.integrate.calibration_seed import _direct_beam_hit, rough_arm_record
    beam = (700.3, 95.6)  # x, y in the oriented image
    moved = {}
    for sense in (1, -1):
        record = rough_arm_record(distance_m=.39, detector='Pilatus300kw', beam_px=beam, zeros_deg=ZEROS,
                                  wavelength_m=WAVELENGTH, senses={'nu': sense})
        hit = [np.asarray(_direct_beam_hit(record, del_motor='del', nu_motor='nu', readings={
            'del': ZEROS['del'], 'nu': ZEROS['nu'] + turn})[0]) for turn in (0., 2.)]
        assert hit[0] == pytest.approx((beam[1], beam[0]), abs=1e-6)  # (row, column)
        moved[sense] = hit[1] - hit[0]
    assert abs(moved[1][1]) > 50 and moved[-1] == pytest.approx(-moved[1], abs=1e-6)  # 0.39 m · tan 2° ≈ 79 px


def test_grazing_incidence_turns_a_reversed_sample_circle_the_other_way():
    from xrd_tools.integrate.stitch_gi import StitchGI, psic_normal
    options = dict(incidence_motor='eta', tilt_motor='chi', motor_zeros=(('chi', 1.),))
    plain, flipped = StitchGI(('q_ip',), **options), StitchGI(('q_ip',), reversed_motors=('chi',), **options)
    metadata = {'eta': .3, 'chi': 91.}
    assert plain.circles(metadata) == (0., .3, 90., 0.) and flipped.circles(metadata) == (0., .3, -90., 0.)
    np.testing.assert_allclose(flipped.normal(metadata), psic_normal(0., .3, -90., 0.))
    assert plain.identity() == StitchGI(('q_ip',), reversed_motors=(), **options).identity()  # older plans
    assert flipped.identity() != plain.identity() and flipped.to_provenance()['reversed_motors'] == ['chi']
    assert 'reversed_motors' not in plain.to_provenance()
    with pytest.raises(ValueError, match='reversed motors'):
        StitchGI(('q_ip',), incidence_motor='eta', reversed_motors=('mu',))  # a motor this GI does not read
