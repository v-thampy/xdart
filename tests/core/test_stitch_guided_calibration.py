"""Guided calibration uses matched real files and the released numerical owners."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import fabio
import numpy as np
import pytest
import tifffile

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.integrate.stitch_calibration import (
    CalibrationFrame, adopt_calibration_pair, capture_calibration_frames,
    fit_calibration, load_calibration_session, load_control_frame,
    save_calibration_fit, suggested_frame_indices,
    save_calibration_session, validate_frame_files, calibration_image_identity,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frames(tmp_path, controls, poni, shape=(195, 1475), orientation=3):
    from pyFAI.control_points import ControlPoints
    from pyFAI.calibrant import get_calibrant
    from pyFAI.detectors import detector_factory
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    frames = []
    for index, control in enumerate(controls):
        directory = tmp_path / str(index)
        directory.mkdir()
        image = directory / 'frame.tif'
        tifffile.imwrite(image, np.zeros(shape, dtype=np.float32))
        points = ControlPoints(calibrant=get_calibrant('LaB6'), wavelength=poni.wavelength)
        for ring in np.unique(control.rings):
            selected = control.rings == ring
            points.append(points=np.column_stack([control.rows[selected], control.cols[selected]]).tolist(), ring=int(ring))
        npt = directory / 'frame.npt'
        points.save(str(npt))
        seed = directory / 'frame.poni'
        AzimuthalIntegrator(dist=poni.dist, poni1=poni.poni1, poni2=poni.poni2,
            rot1=poni.rot1, rot2=poni.rot2, rot3=poni.rot3, wavelength=poni.wavelength,
            detector=detector_factory('Pilatus300kw', config={'orientation': orientation})).save(str(seed))
        frame = CalibrationFrame(str(index), str(image), 'image_file', {}, index,
            str(image), 0, str(image), _digest(image), shape, 0, dict(control.motors), None)
        frames.append(adopt_calibration_pair(frame, seed, npt))
    return tuple(frames)


def test_exact_raw_staging_rotation_angles_and_resume(tmp_path):
    spec = tmp_path / 'beam'
    spec.write_text('#F beam\n#E 1\n#D Thu Jan 1 00:00:00 1970\n#C test\n'
                    '#O0 del nu\n#S 1 ascan del 1 3 2 1\n#D today\n#P0 0 4\n#N 2\n#L del  nu\n1 4\n2 5\n3 6\n')
    images = tmp_path / 'images'; images.mkdir()
    original = np.arange(24, dtype='<i4').reshape(4, 6)
    for index in range(3):
        (images / f'beam_scan1_{index:04d}.raw').write_bytes((original + index * 100).tobytes())
    source = SourceSpec(spec, SourceKind.SPEC, options={'scan': '1', 'image_dir': str(images),
        'read_image_kwargs': {'detector_shape': (4, 6), 'raw_dtype': '<i4'}})
    frames = capture_calibration_frames(source, (2, 0), tmp_path / 'session', rotation=180)
    assert [f.index for f in frames] == [2, 0]
    assert [f.motors for f in frames] == [{'del': 3., 'nu': 6.}, {'del': 1., 'nu': 4.}]
    staged = fabio.open(frames[0].image_path).data  # float64 EDF named after its source image
    assert Path(frames[0].image_path).name == 'beam_scan1_0002.edf' and staged.dtype == np.float64
    assert np.array_equal(staged, np.rot90(original + 200, 2))
    assert load_calibration_session(tmp_path / 'session/calibration-session.json') == frames
    with pytest.raises(ValueError, match='existing SPEC'):
        capture_calibration_frames(replace(source, options={**source.options, 'scan': '999'}), (0,), tmp_path / 'invalid')
    assert len(suggested_frame_indices([(f.index, f.motors, None) for f in frames])) == 2


def test_resumed_control_points_require_retained_image_binding(tmp_path):
    from xrd_tools.core.containers import PONI
    from xrd_tools.integrate.refine import ControlFrame
    control = ControlFrame([1,2], [3,4], [0,0], {'del':10.,'nu':2.})
    frame, = _frames(tmp_path, (control,), PONI(dist=.39, poni1=.01, poni2=.02,
        wavelength=1e-10, detector='Pilatus300kw'))
    session = save_calibration_session(tmp_path, (frame,))
    assert load_calibration_session(session)[0].complete
    value = json.loads(session.read_text())
    value['frames'][0]['control_image_sha256'] = '0' * 64
    session.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='different calibration image'):
        load_calibration_session(session)
    # Older sessions preserve their files but cannot inherit automatic binding.
    value['version'] = 1
    value['frames'][0].pop('control_image_sha256')
    value['frames'][0].pop('pairing_policy')
    session.write_text(json.dumps(value))
    old, = load_calibration_session(session)
    assert not old.complete
    assert old.poni_path == frame.poni_path and old.npt_path == frame.npt_path
    with pytest.raises(ValueError, match='matched PONI'):
        validate_frame_files(old)
    with pytest.raises(ValueError, match='loaded pyFAI image differs'):
        adopt_calibration_pair(old, frame.poni_path, frame.npt_path,
                               model_image_sha256=calibration_image_identity(np.ones(frame.shape))[1])


def test_mg_fit_saved_json_reloads_actual_importer_and_bad_inputs_refuse(tmp_path):
    from tests.core.test_stacked_arm import _independent_stacked_controls
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.integrate.multi import create_multigeometry_integrators_from_geometry
    from pyFAI.goniometer import Goniometer
    controls, calibration = _independent_stacked_controls()
    frames = _frames(tmp_path, controls, calibration.poni)
    geometry = {'geometry': 'psic', 'motors': {'mu': 'mu', 'eta': 'halpha', 'chi': 'chi', 'phi': 'phi',
                                               'nu': 'nu', 'del': 'del'}}
    result = fit_calibration(frames, distance_bounds_m=(.37, .41), instrument_geometry=geometry)
    assert result.asset['provenance']['instrument_geometry'] == geometry  # saved beside the file
    path = save_calibration_fit(result, tmp_path / 'fitted.json')
    loaded = Diffractometer.from_pyfai_goniometer(path,
        source_motors={'nu_angle': 'nu', 'del_angle': 'del'})
    assert .37 <= loaded.calibration.poni.dist <= .41
    assert result.rms_q < .006
    actual = Goniometer.sload(path)
    motors = {'del': np.array([15., 37.]), 'nu': np.array([3., -4.])}
    named = create_multigeometry_integrators_from_geometry(loaded, motors)
    for index, (del_angle, nu_angle) in enumerate(zip(motors['del'], motors['nu'])):
        # The reference notebooks pass (del, nu) positionally to pyFAI.
        positional = actual.get_ai((float(del_angle), float(nu_angle)))
        for name in ('dist', 'poni1', 'poni2', 'rot1', 'rot2', 'rot3', 'wavelength'):
            assert getattr(positional, name) == pytest.approx(getattr(named[index], name), abs=1e-12)
    exported = json.loads((tmp_path / 'fitted.json').read_text())
    assert exported['pos_names'] == exported['trans_function']['pos_names'] == ['del_angle', 'nu_angle']
    from xrd_tools.analysis.stitch_operation import (
        StitchGeometryInput, StitchGeometryKind, capture_stitch_geometry,
    )
    receipt = capture_stitch_geometry(StitchGeometryInput(path, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
        source_motors=(('del_angle','del'),('nu_angle','nu'))))
    assert json.loads(receipt.content)['detector_config'] == {'orientation':3}
    assert json.loads((tmp_path / 'fitted.json.provenance.json').read_text())['frames'][0]['index'] == 0
    with pytest.raises(FileExistsError):
        save_calibration_fit(result, path)
    with pytest.raises(ValueError, match='matched PONI'):
        load_control_frame(replace(frames[0], npt_path=''))
    with pytest.raises(ValueError, match='bounds'):
        fit_calibration(frames, distance_bounds_m=(.4, .3))
    with pytest.raises(ValueError, match='RMS'):
        fit_calibration(frames, max_rms_q=1e-8)
    with open(frames[0].npt_path, 'a') as stream:
        stream.write('\n# changed')
    with pytest.raises(ValueError, match='changed'):
        fit_calibration(frames)


@pytest.mark.parametrize('arm_model', ['stacked', 'stacked_extended'])
def test_a_fit_pins_the_beam_rotation_at_zero_whatever_the_first_poni_holds(tmp_path, arm_model):
    """pyFAI-calib2 leaves each frame's rot3 wherever its seed put it, and |q| cannot measure it:
    the guided fit takes the design value 0, not the first frame's rot3."""
    from tests.core.test_stacked_arm import _independent_stacked_controls
    controls, calibration = _independent_stacked_controls()
    frames = _frames(tmp_path, controls, replace(calibration.poni, rot3=np.radians(-.87)))
    record = fit_calibration(frames, distance_bounds_m=(.37, .41), arm_model=arm_model).asset['calibration']
    assert dict(zip(record['param_names'], record['param']))['beam_rotation'] == 0.


def test_nexus_staging_uses_complete_metadata_provider_and_exact_frame(tmp_path):
    from tests.core.test_metadata_provider import _bluesky
    from xrd_tools.integrate.stitch_calibration import source_frame_choices
    from xrd_tools.sources import open_source
    path, hy, i0, _ = _bluesky(tmp_path / 'raw.nxs')
    spec = SourceSpec(path, SourceKind.NEXUS_STACK)
    choices = source_frame_choices(spec, ('hy', 'i0'))
    assert choices[2][1] == {'hy': float(hy[2]), 'i0': float(i0[2])}
    frames = capture_calibration_frames(spec, (2,), tmp_path/'staged', motor_names=('hy','i0'))
    source = open_source(spec)
    try:
        assert np.array_equal(fabio.open(frames[0].image_path).data, source.load_frame(2))
        assert Path(frames[0].image_path).name == 'raw_2.edf'  # frame 2 inside raw.nxs
    finally:
        source.close()
    assert frames[0].raw_index == 2
    assert frames[0].source_options['dataset_path'] == '/entry/data/eiger_image'


@pytest.mark.parametrize('energy_eV', [None, 17000.004], ids=['canonical', 'fractional-energy'])
def test_xu_notebook_fit_and_authored_admission_preserve_v1(tmp_path, energy_eV):
    from pyFAI.calibrant import get_calibrant
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.core.containers import PONI
    from xrd_tools.integrate.refine import ControlFrame
    from xrd_tools.core.energy import energy_eV_to_wavelength_m
    from xrd_tools.integrate.xu_stitch import (
        resolve_xu_stitch_effective_geometry, XuStitchScienceRefused)
    from xrd_tools.analysis.xu_stitch_calibration import (
        canonical_surface_resource_bytes, capture_xu_stitch_calibration,
        XuStitchCalibrationInput, parse_xu_stitch_calibration_bytes, XuStitchCalibrationRefused)
    calibration = json.loads(canonical_surface_resource_bytes())
    detector, acquisition = calibration['detector'], calibration['acquisition']
    if energy_eV is not None:
        acquisition['energy_eV'] = energy_eV
    wl = energy_eV_to_wavelength_m(acquisition['energy_eV'])
    lab6 = get_calibrant('LaB6'); lab6.wavelength = wl
    q_ref = 2 * np.pi / np.asarray(lab6.dspacing)
    controls = []
    with xu_runtime_session() as runtime:
        xu = runtime.xu
        qc = xu.QConversion(['x+', 'z-', 'y+', 'z-'], ['x+', 'z-'], [0, 1, 0])
        hx = xu.HXRD([0, 1, 0], [0, 0, 1], geometry='real', en=acquisition['energy_eV'], qconv=qc)
        hx.Ang2Q.init_area('x-', 'z+', cch1=detector['cch1_px'], cch2=detector['cch2_px'],
            Nch1=195, Nch2=1475, pwidth1=.172, pwidth2=.172, distance=detector['distance_m'] * 1000)
        for de, nu in [(5., -10.), (20., -5.), (30., 3.), (40., 9.5)]:
            qx, qy, qz = hx.Ang2Q.area(0., 0., 0., 0., nu-acquisition['detector_offsets_deg']['nu'],
                                    de-acquisition['detector_offsets_deg']['del'])
            q = np.hypot(np.hypot(qx, qy), qz)
            rows, cols, rings = [], [], []
            for ri, qr in enumerate(q_ref[:20]):
                yy, xx = np.where(np.abs(q-qr) < .001)
                select = np.linspace(0, max(0, len(yy)-1), min(12, len(yy))).astype(int)
                rows.extend(yy[select]); cols.extend(xx[select]); rings.extend([ri]*len(select))
            controls.append(ControlFrame(rows, cols, rings, {'del': de, 'nu': nu}))
    poni = PONI(dist=detector['distance_m'], poni1=detector['cch1_px']*.000172,
                poni2=detector['cch2_px']*.000172, detector='Pilatus300kw', wavelength=wl)
    frames = _frames(tmp_path, controls, poni)
    fit = fit_calibration(frames, backend='xu_hist', distance_bounds_m=(.38,.40),
                          domain={'del':[0.,50.], 'nu':[-15.,30.]})
    assert fit.rms_q < .002
    (tmp_path / 'turned').mkdir()  # PONIs in another pyFAI detector orientation: refused, not ignored
    with pytest.raises(ValueError, match='default detector orientation'):
        fit_calibration(_frames(tmp_path / 'turned', controls, poni, orientation=2), backend='xu_hist',
                        distance_bounds_m=(.38, .40))
    path = save_calibration_fit(fit, tmp_path/'xu.json')
    receipt = capture_xu_stitch_calibration(XuStitchCalibrationInput(path), project_root=tmp_path)
    assert receipt.projection.value['version'] == 2
    assert .38 <= receipt.projection.value['detector']['distance_m'] <= .40
    with xu_runtime_session() as runtime:
        geometry = resolve_xu_stitch_effective_geometry(receipt, runtime)
    assert geometry.projection.asset_version == 2
    assert geometry.projection.energy_eV == acquisition['energy_eV']
    assert geometry.projection.xdart_wavelength_A == wl * 1e10
    # The parser deliberately permits this one-ulp difference, while the real
    # backend refuses it. Save must validate there before creating final files.
    bad = json.loads(receipt.content)
    bad['acquisition']['xdart_wavelength_A'] = float(np.nextafter(
        bad['acquisition']['xdart_wavelength_A'], np.inf))
    parse_xu_stitch_calibration_bytes(json.dumps(bad, sort_keys=True, separators=(',', ':')).encode())
    refused = tmp_path / 'refused' / 'xu.json'
    with pytest.raises(XuStitchScienceRefused, match='xdart wavelength projection changed'):
        save_calibration_fit(replace(fit, asset={**fit.asset, 'calibration': bad}), refused)
    assert not refused.parent.exists()
    constrained = fit_calibration(frames, backend='xu_hist', distance_seed_m=.395,
        distance_bounds_m=(.390, .400), domain={'del':[0.,50.], 'nu':[-15.,30.]})
    assert .390 <= constrained.asset['calibration']['detector']['distance_m'] <= .400
    bad = json.loads(receipt.content)
    bad['validation']['geometry_rms_q_A^-1'] = .1
    with pytest.raises(XuStitchCalibrationRefused):
        parse_xu_stitch_calibration_bytes(json.dumps(bad, sort_keys=True, separators=(',', ':')).encode())
    original = canonical_surface_resource_bytes()
    changed = json.loads(original); changed['detector']['distance_m'] += .001
    with pytest.raises(XuStitchCalibrationRefused):
        parse_xu_stitch_calibration_bytes(json.dumps(changed, sort_keys=True, separators=(',', ':')).encode())
    assert parse_xu_stitch_calibration_bytes(original).content == original


def test_mg_distance_bounds_constrain_optimizer_not_only_initial_seed():
    from tests.core.test_refine import _truth, _synth_control_frames, _FRAMES
    from xrd_tools.integrate.refine import refine_goniometer
    truth = _truth()
    controls = _synth_control_frames(truth, _FRAMES, per_ring=8)
    distance = truth.calibration.poni.dist
    lower, upper = distance * 1.002, distance * 1.006
    result = refine_goniometer(truth.calibration.poni, controls, rot1_motor='nu', rot2_motor='del',
        detector_config={'orientation':3}, distance_seed_m=(lower+upper)/2,
        distance_bounds_m=(lower,upper))
    assert lower <= result.params['dist'] <= upper
    assert result.params['dist'] > distance


def test_calibration_refuses_same_count_missing_interior_raw_frame(tmp_path):
    from tests.core.test_spec_source import _spec
    from xrd_tools.integrate.stitch_calibration import source_frame_choices
    path = _spec(tmp_path)
    for index in (0, 2, 3):
        np.ones((2, 3), dtype=np.uint16).tofile(tmp_path / f'myscan_scan5_{index}.raw')
    spec = SourceSpec(path, SourceKind.SPEC, options={'scan': '5', 'image_dir': str(tmp_path),
        'read_image_kwargs': {'detector_shape': (2, 3), 'raw_dtype': 'uint16'}})
    with pytest.raises(ValueError, match='number'):
        source_frame_choices(spec, ('th', 'chi'))


def test_a_saved_fit_writes_its_xrayutilities_partner_bound_by_hash(tmp_path):
    """Always both (maintainer, 2026-09-29): the pyFAI file, then its xrayutilities partner beside it,
    checked on the calibrant frames, the scan's frames and the typed range, and bound to the pyFAI bytes."""
    from tests.core.test_stacked_arm import _independent_stacked_controls
    from xrd_tools.analysis.xu_stitch_calibration import XuStitchCalibrationInput, capture_xu_stitch_calibration
    from xrd_tools.integrate.stitch_calibration import (
        pair_validation_angles, paired_xu_calibration, save_calibration_pair,
    )
    controls, calibration = _independent_stacked_controls()
    frames = _frames(tmp_path, controls, calibration.poni)
    fit = fit_calibration(frames, distance_bounds_m=(.37, .41))
    domain = {'del': [5., 42.], 'nu': [-11., 17.]}
    angles = pair_validation_angles(fit, scan_angles=[(10., 0.), (30., 12.), (6., -10.)], domain=domain)
    assert len(angles) == 5 + 2 + 49  # the calibrant frames, two more scan frames (one repeats), the grid
    saved = save_calibration_pair(fit, tmp_path / 'calibration' / 'arm.json', validation_angles=angles)
    assert Path(saved.xu_path).name == 'arm.xu.json' and saved.xu_refusal == ''
    assert saved.xu_transfer['method'] == 'formula' and saved.xu_transfer['validation_frames'] == len(angles)
    asset = json.loads(Path(saved.xu_path).read_text())
    assert asset['version'] == 3 and asset['validation']['validated_scan_domain_source_deg'] == domain
    provenance = json.loads(Path(saved.xu_path + '.provenance.json').read_text())
    assert provenance['pyfai_calibration'] == {'file': 'arm.json', 'sha256': _digest(Path(saved.pyfai_path))}
    assert paired_xu_calibration(saved.pyfai_path) == Path(saved.xu_path)
    capture_xu_stitch_calibration(XuStitchCalibrationInput(Path(saved.xu_path)), project_root=tmp_path)
    assert provenance['xu_calibration'] == {'file': 'arm.xu.json', 'sha256': _digest(Path(saved.xu_path))}
    # another pair's partner copied over this one's (Codex review F5): the pair binds the partner's bytes
    import copy
    from dataclasses import replace
    other = copy.deepcopy(fit.asset['calibration'])
    other['param'][other['param_names'].index('dist')] += .05
    other_saved = save_calibration_pair(replace(fit, asset={**fit.asset, 'calibration': other}),
                                        tmp_path / 'calibration' / 'other.json', validation_angles=angles)
    original = Path(saved.xu_path).read_bytes()
    Path(saved.xu_path).write_bytes(Path(other_saved.xu_path).read_bytes())
    with pytest.raises(ValueError, match='arm.xu.json is not the xrayutilities file saved with arm.json'):
        paired_xu_calibration(saved.pyfai_path)
    Path(saved.xu_path).write_bytes(original)
    assert paired_xu_calibration(saved.pyfai_path) == Path(saved.xu_path)
    Path(saved.pyfai_path).write_text(Path(saved.pyfai_path).read_text() + '\n')  # other bytes: not its pair
    with pytest.raises(ValueError, match='arm.xu.json was not made from this arm.json'):
        paired_xu_calibration(saved.pyfai_path)
    (tmp_path / 'calibration' / 'taken.xu.json').write_text('{}')
    with pytest.raises(FileExistsError):  # the partner's name is taken: nothing is written
        save_calibration_pair(fit, tmp_path / 'calibration' / 'taken.json', validation_angles=angles)
    assert not (tmp_path / 'calibration' / 'taken.json').exists()


def test_a_fit_xrayutilities_cannot_express_saves_the_pyfai_file_alone(tmp_path):
    from tests.core.test_stacked_arm import _independent_stacked_controls
    from xrd_tools.integrate.stitch_calibration import paired_xu_calibration, save_calibration_pair
    controls, calibration = _independent_stacked_controls()
    fit = fit_calibration(_frames(tmp_path, controls, calibration.poni), fit_scales=True)
    saved = save_calibration_pair(fit, tmp_path / 'calibration' / 'scaled.json',
                                  validation_angles=[(6., -10.), (15., 3.), (25., 6.)])
    assert Path(saved.pyfai_path).is_file() and saved.xu_path is None
    assert 'motor scales' in saved.xu_refusal and not (tmp_path / 'calibration' / 'scaled.xu.json').exists()
    with pytest.raises(ValueError, match='scaled.json has no xrayutilities partner .scaled.xu.json.: save'):
        paired_xu_calibration(saved.pyfai_path)


@pytest.mark.parametrize('held', ['nu', 'del'])
def test_a_single_arms_partner_is_checked_with_its_held_circle_at_zero(held):
    from xrd_tools.integrate.stitch_calibration import CalibrationFit, pair_validation_angles
    motors = {'del_angle': 'del'} if held == 'nu' else {'nu_angle': 'tth'}
    moving = 'del' if held == 'nu' else 'tth'
    fit = CalibrationFit('multigeometry', {'calibration': {}, 'provenance': {
        'source_motors': motors, 'frames': [{'motors': {moving: 10., 'other': 3.}}, {'motors': {moving: 20.}}]}},
        0., 1., 1, '')
    angles = pair_validation_angles(fit, scan_angles=[(30., 7.)], domain={'del': [0., 40.], 'nu': [0., 40.]}, grid=3)
    expected = ([(10., 0.), (20., 0.), (30., 0.), (0., 0.), (40., 0.)] if held == 'nu'
                else [(0., 10.), (0., 20.), (0., 7.), (0., 0.), (0., 40.)])
    assert angles == expected


def test_the_automatic_arm_model_takes_the_extension_when_it_applies_and_says_why_not(tmp_path):
    """Decision 40: the detector mount and sample offset whenever the frames can determine them;
    otherwise, or when that fit is refused, the five-parameter arm with the reason."""
    from tests.core.test_stacked_arm_extension import ANGLES, TRUTH, _calibration, _ring_controls
    from xrd_tools.integrate.stitch_calibration import automatic_arm_reason
    controls, _ = _ring_controls(TRUTH, _calibration(), ANGLES[::2])
    (tmp_path / 'wide').mkdir()
    wide = _frames(tmp_path / 'wide', controls, _calibration().poni)
    assert automatic_arm_reason(wide) == ''
    assert fit_calibration(wide, arm_model='auto').asset['provenance']['arm_model'] == 'stacked_extended'
    two_rows = tuple(frame for frame in wide if frame.motors['nu'] in (-10., -3.))
    assert automatic_arm_reason(two_rows) == 'the extension needs frames on at least 3 nu positions (2 so far)'
    assert 'motor scales' in automatic_arm_reason(wide, fit_scales=True)
    assert 'single 2θ arm' in automatic_arm_reason(wide, nu_motor=None)
    # three nu positions 0.3° apart: the extension is refused as poorly determined, five parameters fit
    controls, _ = _ring_controls({'nu_offset': .02, 'del_offset': -.05}, _calibration(),
                                 [(d, n) for n in (-.3, 0., .3) for d in (6., 30.)])
    (tmp_path / 'narrow').mkdir()
    fit = fit_calibration(_frames(tmp_path / 'narrow', controls, _calibration().poni), arm_model='auto')
    provenance = fit.asset['provenance']
    assert provenance['arm_model'] == 'stacked' and 'poorly determined' in provenance['arm_model_reason']
    assert fit.message.endswith('five-parameter arm: the detector mount and sample offset were refused: '
                                'arm extension is poorly determined; add frames spanning nu and del')


def test_stitch_with_xrayutilities_reads_an_xu_file_itself_and_a_pyfai_files_partner(tmp_path):
    from tests.core.test_stacked_arm import _independent_stacked_controls
    from xrd_tools.integrate.stitch_calibration import save_calibration_pair, xu_calibration_path
    controls, calibration = _independent_stacked_controls()
    fit = fit_calibration(_frames(tmp_path, controls, calibration.poni), distance_bounds_m=(.37, .41))
    saved = save_calibration_pair(fit, tmp_path / 'calibration' / 'arm.json',
                                  validation_angles=[(6., -10.), (15., 3.), (25., 6.)])
    assert xu_calibration_path(saved.pyfai_path) == Path(saved.xu_path)
    assert xu_calibration_path('calibration/arm.json', tmp_path) == Path(saved.xu_path)  # the project's
    assert xu_calibration_path(saved.xu_path) == Path(saved.xu_path)  # an xrayutilities file is read as is
    assert xu_calibration_path('calibration/none.json', tmp_path) == Path('calibration/none.json')


def test_an_arm_xrayutilities_refuses_still_saves_and_names_the_bound(tmp_path):
    """The v3 admission bounds the motor offsets (10°): its refusal is the transfer's, so Save keeps
    the pyFAI file and says why instead of failing after writing it."""
    from xrd_tools.core.geometry.stacked_arm import StackedDetectorArm
    from xrd_tools.integrate.stitch_calibration import CalibrationFit, save_calibration_pair
    from tests.core.test_stacked_arm_extension import _calibration
    record = StackedDetectorArm(nu_motor='nu', del_motor='del', del_offset=np.radians(12.)).to_pyfai_goniometer(
        _calibration())
    record['detector_config'] = {'orientation': 3}
    fit = CalibrationFit('multigeometry', {'calibration': record, 'provenance': {
        'source_motors': {'del_angle': 'del', 'nu_angle': 'nu'}, 'image_rotation': 0, 'frames': []}}, 1e-3, 10., 100, '')
    saved = save_calibration_pair(fit, tmp_path / 'calibration' / 'offset.json',
                                  validation_angles=[(6., -10.), (15., 3.), (25., 6.)])
    assert Path(saved.pyfai_path).is_file() and saved.xu_path is None
    assert saved.xu_refusal == ('no xrayutilities file: xrayutilities refuses the transferred calibration: '
                                'motor offsets exceed fit bounds')
