"""A single 2θ detector arm (fourc, twoc) is psic's del-on-nu arm with nu held at zero."""
import hashlib
import json
import math
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.core.containers import PONI
from xrd_tools.core.geometry import DetectorCalibration, Diffractometer
from xrd_tools.core.geometry.stacked_arm import SINGLE_ARM_MARKER, StackedDetectorArm, is_single_arm
from xrd_tools.integrate.multi import create_multigeometry_integrator_series_from_geometry

_POSE = ('dist', 'poni1', 'poni2', 'rot1', 'rot2', 'rot3')


def _arm(nu_motor=None):
    return StackedDetectorArm(nu_motor=nu_motor, del_motor='tth', nu_offset=.0123, del_offset=-.02,
                              del_scale=math.radians(1.001), beam_rotation=.003)


def _calibration():
    return DetectorCalibration(PONI(dist=.39, poni1=.033, poni2=.052, wavelength=.775e-10,
                                    detector='Pilatus300kw'), detector_config={'orientation': 3})


def _worst(first, second):
    return max(abs(float(getattr(a, key)) - float(getattr(b, key)))
               for a, b in zip(first, second, strict=True) for key in _POSE)


def test_a_single_arm_is_psics_arm_with_nu_held_at_zero(tmp_path):
    from pyFAI.goniometer import Goniometer
    tth = np.array([5., 12.5, 33., 71.])
    single, psic = _arm(), _arm('nu')
    assert single.single and not psic.single
    ours, theirs = single.rotations({'tth': tth}), psic.rotations({'tth': tth, 'nu': np.zeros(4)})
    assert all(np.array_equal(ours[key], theirs[key]) for key in ours)
    record = single.to_pyfai_goniometer(_calibration())
    assert record['pos_names'] == record['trans_function']['pos_names'] == ['del_angle']
    assert 'nu_scale' not in record['param_names'] and is_single_arm(record)
    assert not is_single_arm(psic.to_pyfai_goniometer(_calibration()))
    diff = Diffractometer.from_pyfai_goniometer(record, source_motors={'del_angle': 'tth'})
    assert diff.stacked_arm == single and diff.all_referenced_motors() == ('tth',)
    assert Diffractometer.from_json(diff.to_json()).stacked_arm == single
    # pyFAI reads the one-position file as an ordinary goniometer and places each frame alike
    path = tmp_path / 'single.json'
    path.write_text(json.dumps(record))
    native = Goniometer.sload(str(path))
    series = list(create_multigeometry_integrator_series_from_geometry(diff, {'tth': tth}))
    assert _worst(series, [native.get_ai((value,)) for value in tth]) == 0
    two = Diffractometer.from_pyfai_goniometer(psic.to_pyfai_goniometer(_calibration()),
                                               source_motors={'del_angle': 'tth', 'nu_angle': 'nu'})
    at_zero = create_multigeometry_integrator_series_from_geometry(two, {'tth': tth, 'nu': np.zeros(4)})
    assert _worst(series, list(at_zero)) == 0
    with pytest.raises(ValueError, match='no mount or sample-offset extension'):
        replace(single, mount_tilt=.01)
    tampered = json.loads(json.dumps(record))
    tampered['trans_function']['rot3_expr'] = 'beam_rotation'
    with pytest.raises(ValueError, match='stacked arm'):
        Diffractometer.from_pyfai_goniometer(tampered, source_motors={'del_angle': 'tth'})
    assert SINGLE_ARM_MARKER in record['trans_function']['constants']


def test_stitch_admits_a_single_arm_calibration_and_a_del_only_poni(tmp_path):
    from xrd_tools.analysis.stitch_operation import (
        StitchGeometryInput, StitchGeometryKind, StitchOperationRefused, _runtime_geometry,
        capture_stitch_geometry,
    )
    path = tmp_path / 'single.json'
    path.write_text(json.dumps(_arm().to_pyfai_goniometer(_calibration())))
    receipt = capture_stitch_geometry(StitchGeometryInput(
        path, StitchGeometryKind.PYFAI_GONIOMETER_JSON, source_motors=(('del_angle', 'tth'),)))
    assert _runtime_geometry(receipt, receipt.content)[0].stacked_arm == _arm()
    with pytest.raises(StitchOperationRefused) as refused:  # a nu binding the file does not declare
        capture_stitch_geometry(StitchGeometryInput(path, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
                                                    source_motors=(('del_angle', 'tth'), ('nu_angle', 'nu'))))
    assert refused.value.code == 'GEOMETRY_MOTOR_MAPPING_MISMATCH'
    # PONI + offsets: a 2θ arm maps del alone, with one reference, and leaves rot1 at the PONI's
    poni = tmp_path / 'frame.poni'
    poni.write_text('poni_version: 2.1\nDetector: Pilatus300kw\nDetector_config: {"orientation": 3}\n'
                    'Distance: 0.39\nPoni1: 0.033\nPoni2: 0.052\nRot1: 0.01\nRot2: 0.2\nRot3: 0.0\n'
                    'Wavelength: 7.75e-11\n')
    request = StitchGeometryInput(poni, StitchGeometryKind.PONI, source_motors=(('del', 'tth'),),
                                  reference_motor_positions=(('del', 11.5),))
    receipt = capture_stitch_geometry(request)
    diff = _runtime_geometry(receipt, receipt.content)[0]
    assert not diff.rot1.is_active and diff.rot2.source_motor == 'tth' and diff.rot2.offset == -11.5
    # a horizontal 2θ arm maps nu alone: rot1 follows it, rot2 stays at the PONI's
    receipt = capture_stitch_geometry(StitchGeometryInput(poni, StitchGeometryKind.PONI,
        source_motors=(('nu', 'tth'),), reference_motor_positions=(('nu', 11.5),)))
    diff = _runtime_geometry(receipt, receipt.content)[0]
    assert not diff.rot2.is_active and diff.rot1.source_motor == 'tth' and diff.rot1.offset == -11.5
    for motors, references in (((('del', 'tth'),), (('del', 1.), ('nu', 0.))),
                               ((('tth', 'tth'),), (('tth', 0.),))):
        with pytest.raises(ValueError, match='del and/or nu'):
            StitchGeometryInput(poni, StitchGeometryKind.PONI, source_motors=motors,
                                reference_motor_positions=references)


def test_a_horizontal_2theta_arm_is_psics_arm_with_del_held_at_zero(tmp_path):
    from pyFAI.goniometer import Goniometer
    from xrd_tools.core.geometry.stacked_arm import stacked_arm_transformation
    tth = np.array([5., 12.5, 33., 71.])
    horizontal = StackedDetectorArm(nu_motor='tth', del_motor=None, nu_offset=-.02, nu_scale=math.radians(.999),
                                    del_offset=.0117, beam_rotation=.003)
    psic = replace(horizontal, del_motor='del')
    assert horizontal.held == 'del' and _arm().held == 'nu' and psic.held is None
    ours = horizontal.rotations({'tth': tth})
    theirs = psic.rotations({'tth': tth, 'del': np.zeros(4)})
    assert all(np.array_equal(ours[key], theirs[key]) for key in ours)
    record = horizontal.to_pyfai_goniometer(_calibration())
    assert record['pos_names'] == ['nu_angle'] and 'del_scale' not in record['param_names']
    assert is_single_arm(record)
    diff = Diffractometer.from_pyfai_goniometer(record, source_motors={'nu_angle': 'tth'})
    assert diff.stacked_arm == horizontal and diff.all_referenced_motors() == ('tth',)
    assert Diffractometer.from_json(diff.to_json()).stacked_arm == horizontal
    path = tmp_path / 'horizontal.json'
    path.write_text(json.dumps(record))
    series = list(create_multigeometry_integrator_series_from_geometry(diff, {'tth': tth}))
    # pyFAI's numexpr evaluates the same expressions in another order: equal to rounding
    assert _worst(series, [Goniometer.sload(str(path)).get_ai((value,)) for value in tth]) < 1e-15
    two = Diffractometer.from_pyfai_goniometer(psic.to_pyfai_goniometer(_calibration()),
                                               source_motors={'del_angle': 'del', 'nu_angle': 'tth'})
    assert _worst(series, list(create_multigeometry_integrator_series_from_geometry(
        two, {'tth': tth, 'del': np.zeros(4)}))) == 0
    # the vertical record saved since round 12 keeps its exact transformation
    assert stacked_arm_transformation(held='nu') == {
        'content': 'GeometryTransformation',
        'param_names': ['dist', 'poni1', 'poni2', 'nu_offset', 'del_offset', 'del_scale', 'beam_rotation'],
        'pos_names': ['del_angle'], 'dist_expr': 'dist', 'poni1_expr': 'poni1', 'poni2_expr': 'poni2',
        'rot1_expr': 'arctan2(sin((nu_offset)),(cos((nu_offset))*cos((del_offset+del_scale*del_angle))))',
        'rot2_expr': 'arctan2((cos((nu_offset))*sin((del_offset+del_scale*del_angle))),sqrt(sin((nu_offset))**2'
                     '+(cos((nu_offset))*cos((del_offset+del_scale*del_angle)))**2))',
        'rot3_expr': 'beam_rotation+arctan2(sin((nu_offset))*sin((del_offset+del_scale*del_angle)),'
                     'cos((del_offset+del_scale*del_angle)))',
        'constants': {'pi': math.pi, 'xdart_single_arm_v1': 1}}
    # a single-arm record declares one moving position; anything else is refused
    mixed = json.loads(json.dumps(record))
    mixed['pos_names'] = mixed['trans_function']['pos_names'] = ['tth_angle']
    with pytest.raises(ValueError, match='stacked arm'):
        Diffractometer.from_pyfai_goniometer(mixed, source_motors={'tth_angle': 'tth'})
    with pytest.raises(ValueError, match='distinct, nonempty'):
        StackedDetectorArm(nu_motor=None, del_motor=None)


def test_the_fit_and_seeds_take_a_horizontal_arm():
    from xrd_tools.integrate.calibration_seed import predict_frame_seed, rough_arm_record
    from xrd_tools.integrate.refine import refine_goniometer
    from tests.core.test_stacked_arm_extension import _calibration as _truth_calibration, _ring_controls
    controls, calibrant = _ring_controls({'nu_offset': -.03, 'del_offset': .012}, _truth_calibration(),
                                         [(0., nu) for nu in (4., 15., 26., 37.)])
    poni = _truth_calibration().poni
    kwargs = dict(calibrant=calibrant, fit_scales=False, model='stacked_del_on_nu', detector='Pilatus300kw',
                  detector_config={'orientation': 3})
    single = refine_goniometer(poni, [replace(c, motors={'tth': c.motors['nu']}) for c in controls],
                               rot1_motor='tth', rot2_motor=None, **kwargs)
    psic = refine_goniometer(poni, [replace(c, motors={'tth': c.motors['nu'], 'del': 0.}) for c in controls],
                             rot1_motor='tth', rot2_motor='del', **kwargs)
    assert single.success and single.diffractometer.stacked_arm.held == 'del'
    assert {key: psic.params[key] for key in single.params} == single.params
    # this fit reads whole-pixel control points, so it recovers the truth approximately
    assert single.params['nu_offset'] == pytest.approx(-.03, abs=2e-3)
    assert single.params['del_offset'] == pytest.approx(.012, abs=2e-3)  # the detector's fixed pitch
    record = single.diffractometer.stacked_arm.to_pyfai_goniometer(single.diffractometer.calibration)
    frame = _frame({'tth': 20.})
    seeded = predict_frame_seed(record, frame, del_motor=None, nu_motor='tth')
    reference = predict_frame_seed(psic.diffractometer.stacked_arm.to_pyfai_goniometer(
        psic.diffractometer.calibration), _frame({'tth': 20., 'del': 0.}), del_motor='del', nu_motor='tth')
    assert _worst([seeded.integrator], [reference.integrator]) == 0 and seeded.rings == reference.rings
    with pytest.raises(ValueError, match='holds del at zero'):  # a horizontal calibration, del moved
        predict_frame_seed(record, _frame({'tth': 20., 'del': 3.}), del_motor='del', nu_motor='tth')
    rough = rough_arm_record(distance_m=.5, detector='Pilatus300kw', beam_px=(737, 97), zero_deg=2.,
                             zero_motor='tth', del_motor=None, nu_motor='tth')
    assert rough['pos_names'] == ['nu_angle']
    assert dict(zip(rough['param_names'], rough['param']))['nu_offset'] == pytest.approx(-math.radians(2.))


def _affine(positions, detector='Pilatus100k'):
    """An independent-Euler goniometer record: rot2 follows the first position, rot1 the last."""
    names = ['dist', 'poni1', 'poni2', 'rot1_offset', 'rot1_scale', 'rot2_offset', 'rot2_scale']
    rot1 = f'rot1_scale * {positions[-1]} + rot1_offset' if len(positions) > 1 else 'rot1_offset'
    return {'content': 'Goniometer calibration v2', 'detector': detector, 'detector_config': {'orientation': 3},
            'wavelength': 1e-10, 'param': [.2, .01677, .04188, 0., math.pi / 180, 0., math.pi / 180],
            'param_names': names, 'pos_names': list(positions),
            'trans_function': {'content': 'GeometryTransformation', 'param_names': names,
                               'pos_names': list(positions), 'dist_expr': 'dist', 'poni1_expr': 'poni1',
                               'poni2_expr': 'poni2', 'rot1_expr': rot1,
                               'rot2_expr': f'rot2_scale * {positions[0]} + rot2_offset', 'rot3_expr': '0.0',
                               'constants': {'pi': math.pi}}}


def test_screening_rows_follow_the_calibrations_circles_not_its_position_names(tmp_path):
    from types import SimpleNamespace
    from xrd_tools.analysis.stitch_operation import (
        StitchGeometryInput, StitchGeometryKind, _frame_screen_rows, capture_stitch_geometry,
    )
    frames = ((10., -3.), (20., -3.), (30., 4.))

    def rows(record_or_poni, bindings, columns, references=()):
        if isinstance(record_or_poni, dict):
            path = tmp_path / f'{len(list(tmp_path.iterdir()))}.json'
            path.write_text(json.dumps(record_or_poni))
            kind = StitchGeometryKind.PYFAI_GONIOMETER_JSON
        else:
            path, kind = record_or_poni, StitchGeometryKind.PONI
        receipt = capture_stitch_geometry(StitchGeometryInput(path, kind, source_motors=bindings,
                                                              reference_motor_positions=references))
        request = SimpleNamespace(
            plan=SimpleNamespace(geometry=receipt),
            manifest=SimpleNamespace(contributions=[
                SimpleNamespace(label=label, values=tuple((name, 0, value) for name, value in zip(columns, frame)))
                for label, frame in enumerate(frames)]),
            module=SimpleNamespace(source=SimpleNamespace(selected_labels=(0, 1, 2))))
        return [row for _member, row in _frame_screen_rows(request)]
    by_nu, one = [-3., -3., 4.], [0., 0., 0.]
    psic = _arm('nu').to_pyfai_goniometer(_calibration())
    assert rows(psic, (('del_angle', 'del'), ('nu_angle', 'tth')), ('del', 'tth')) == by_nu
    # Codex round 13 F1: a two-circle file whose positions are named otherwise is still rows by nu
    for positions in (('del_angle', 'nu_angle'), ('dval', 'nu_angle'), ('dval', 'nval')):
        assert rows(_affine(positions), tuple(zip(positions, ('del', 'nu'))), ('del', 'nu')) == by_nu
    assert rows(_affine(('pos',)), (('pos', 'del'),), ('del', 'nu')) == one  # rot2 alone
    # a single arm is one row, whichever circle moves (the 2θ column here is tth)
    assert rows(_arm().to_pyfai_goniometer(_calibration()), (('del_angle', 'tth'),), ('tth', 'th')) == one
    horizontal = StackedDetectorArm(nu_motor='tth', del_motor=None).to_pyfai_goniometer(_calibration())
    assert rows(horizontal, (('nu_angle', 'tth'),), ('tth', 'th')) == one
    poni = tmp_path / 'frame.poni'
    poni.write_text('poni_version: 2.1\nDetector: Pilatus300kw\nDetector_config: {"orientation": 3}\n'
                    'Distance: 0.39\nPoni1: 0.033\nPoni2: 0.052\nRot1: 0\nRot2: 0\nRot3: 0\nWavelength: 7.75e-11\n')
    assert rows(poni, (('del', 'del'), ('nu', 'nu')), ('del', 'nu'), (('del', 0.), ('nu', 0.))) == by_nu
    assert rows(poni, (('nu', 'tth'),), ('tth', 'th'), (('nu', 0.),)) == one
    frames_rows = [(index, {'tth': 5. + index}, None) for index in range(20)]
    from xrd_tools.integrate.stitch_autocalibration import stratified_frame_indices
    chosen, coverage = stratified_frame_indices(frames_rows, 5, 'tth', None)
    assert coverage == {'nu_rows_available': 1, 'nu_rows_selected': 1}
    assert len(chosen) == 5 and chosen[0] < 4 and chosen[-1] > 15  # spread over the moving nu


def test_renaming_a_calibration_position_changes_no_screening_or_intensity(tmp_path):
    """Codex round 13 F1, through the operation: frames at nu -5, 0, 5 with counts 10000, 1,
    10000 are three screening rows whatever the del position is called, so none is dropped."""
    import tifffile
    from xrd_tools.analysis.module_transaction import (
        MetadataColumnSelector, ModuleDisposition, ModuleKind, ModuleOutputRequest, ModuleSourceReceipt,
    )
    from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table
    from xrd_tools.analysis.stitch_operation import (
        StitchGeometryInput, StitchGeometryKind, StitchOperationPlan, capture_stitch_geometry,
        prepare_stitch_operation, run_stitch_operation,
    )
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, read_analysis_artifact
    from xrd_tools.sources.selection import image_series_spec
    images = tmp_path / 'images'
    images.mkdir()
    for index, (nu, count) in enumerate(((-5, 10000), (0, 1), (5, 10000)), 1):
        image = images / f'scan_{index:04d}.tif'
        tifffile.imwrite(image, np.full((195, 487), count, dtype=np.uint16))
        image.with_suffix('.txt').write_text(f'# Counters\nI0 = 1\n# Motors\ndel = {index * 5}, nu = {nu}\n'
                                             'User: stitch, time: Mon Jan 15 10:30:00 2024  # Temp\n')
    table = run_metadata_table(MetadataTablePlan(image_series_spec(images / 'scan_0001.tif', metadata_format='txt')))
    source = ModuleSourceReceipt.from_metadata_table(table, kind=ModuleKind.STITCH, selected_labels=(1, 2, 3),
        resolved_selectors=(MetadataColumnSelector('del'), MetadataColumnSelector('nu')))
    curves, screens = [], []
    for delta in ('del_angle', 'dval'):
        path = tmp_path / f'{delta}.json'
        path.write_text(json.dumps(_affine((delta, 'nu_angle'))))
        geometry = capture_stitch_geometry(StitchGeometryInput(path, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
                                                               source_motors=((delta, 'del'), ('nu_angle', 'nu'))))
        output = ModuleOutputRequest(tmp_path / f'{delta}.nexus', AnalysisArtifactKind.STITCH_1D)
        request = prepare_stitch_operation(source, output, StitchOperationPlan(geometry, npt_1d=96,
                                                                               radial_range=(.1, 6.)),
                                           project_root=tmp_path)
        result = run_stitch_operation(request)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
        screens.append(result.frame_screen['excluded'])
        curves.append(read_analysis_artifact(output.target).intensity)
    assert screens == [[], []]
    np.testing.assert_array_equal(curves[0], curves[1])


def test_a_session_records_its_arm(tmp_path):
    from xrd_tools.integrate.stitch_calibration import load_calibration_session, save_calibration_session
    frame = replace(_frame({'tth': 20.}), image_path=str(tmp_path / 'frame.edf'))
    Path(frame.image_path).write_bytes(b'x')
    frame = replace(frame, image_sha256=hashlib.sha256(b'x').hexdigest())
    for arm in ((None, 'tth'), ('tth', None)):  # one motor name, either plane
        path = save_calibration_session(tmp_path, (frame,), arm=arm)
        assert json.loads(path.read_text())['version'] == 3
        assert load_calibration_session(path, with_arm=True) == ((frame,), arm)
        assert load_calibration_session(path) == (frame,)
    path = save_calibration_session(tmp_path, (frame,))  # an older session: version 2, no arm
    assert load_calibration_session(path, with_arm=True) == ((frame,), None)
    path = save_calibration_session(tmp_path, (frame,), arm=('del', 'nu'))  # frames read tth only
    with pytest.raises(ValueError, match='do not read its arm'):
        load_calibration_session(path, with_arm=True)


def _frame(motors, shape=(195, 1475)):
    from xrd_tools.integrate.stitch_calibration import CalibrationFrame
    return CalibrationFrame('frame-0', 'scan', 'spec', {}, 0, 'raw', 0, 'image', 'sha', shape, 0,
                            motors, 16000.)


def test_frame_seeds_and_the_rough_geometry_take_a_single_arm():
    from xrd_tools.integrate.calibration_seed import predict_frame_seed, rough_arm_record
    single = _arm().to_pyfai_goniometer(_calibration())
    psic = _arm('nu').to_pyfai_goniometer(_calibration())
    seeded = predict_frame_seed(single, _frame({'tth': 20.}), del_motor='tth', nu_motor=None)
    reference = predict_frame_seed(psic, _frame({'tth': 20., 'nu': 0.}), del_motor='tth', nu_motor='nu')
    assert _worst([seeded.integrator], [reference.integrator]) == 0 and seeded.rings == reference.rings
    # a psic calibration seeds a 2θ-arm frame with nu held at zero
    held = predict_frame_seed(psic, _frame({'tth': 20.}), del_motor='tth', nu_motor=None)
    assert _worst([held.integrator], [reference.integrator]) == 0
    # a 2θ-arm calibration cannot place a frame whose nu moved
    with pytest.raises(ValueError, match='holds nu at zero'):
        predict_frame_seed(single, _frame({'tth': 20., 'nu': 3.}), del_motor='tth', nu_motor='nu')
    rough = rough_arm_record(distance_m=.5, detector='Pilatus300kw', beam_px=(737, 97), zero_deg=1.5,
                             zero_motor='tth', del_motor='tth', nu_motor=None)
    assert is_single_arm(rough) and rough['pos_names'] == ['del_angle']
    assert dict(zip(rough['param_names'], rough['param']))['del_offset'] == pytest.approx(-math.radians(1.5))
    with pytest.raises(ValueError, match="not 'tth'"):
        rough_arm_record(distance_m=.5, detector='Pilatus300kw', beam_px=(737, 97), zero_motor='nu',
                         del_motor='tth', nu_motor=None)


def test_a_single_arm_fit_refuses_what_needs_nu():
    from xrd_tools.integrate.stitch_autocalibration import stratified_frame_indices
    from xrd_tools.integrate.stitch_calibration import fit_calibration
    frames = (None,) * 3  # refused before any frame is read
    for delta, nu in (('tth', None), (None, 'tth')):
        with pytest.raises(ValueError, match='no nu positions'):
            fit_calibration(frames, del_motor=delta, nu_motor=nu, arm_model='stacked_extended')
    with pytest.raises(ValueError, match='at least one moving circle'):
        fit_calibration(frames, del_motor=None, nu_motor=None)
    with pytest.raises(ValueError, match='MultiGeometry backend'):
        fit_calibration(frames, backend='xu_hist', del_motor='tth', nu_motor=None)
    rows = [(index, {'tth': 5. + index}, None) for index in range(20)]
    chosen, coverage = stratified_frame_indices(rows, 5, None, 'tth')
    assert coverage == {'nu_rows_available': 1, 'nu_rows_selected': 1}
    assert len(chosen) == 5 and chosen[0] < 4 and chosen[-1] > 15  # spread over 2θ


def _powder_scan(root, angles, truth):
    """A twoc SPEC scan (tth and th) of LaB6 ring images from a single 2θ arm."""
    from pyFAI.calibrant import get_calibrant
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from tests.core.test_stacked_arm_extension import WAVELENGTH, _calibration, _ray
    calibrant = get_calibrant('LaB6')
    calibrant.wavelength = WAVELENGTH
    q_rings = 2 * np.pi / np.asarray(calibrant.dspacing)[:25]
    rows, cols = np.indices((195, 1475), dtype=float)
    images = root / 'images'
    images.mkdir(parents=True)
    for index, delta in enumerate(angles):
        q, _ = _ray(truth, _calibration(), delta, 0., rows, cols)
        image = 10 + 5000 * np.exp(-((q[..., None] - q_rings) ** 2) / (2 * .008 ** 2)).sum(-1)
        image.astype('<i4').tofile(images / f'powder_scan1_{index:04d}.raw')
    lines = '\n'.join(f'{delta} 5' for delta in angles)
    (root / 'powder').write_text('#F powder\n#E 1\n#D today\n#O0 tth th\n#S 1 ascan tth 6 34 6 1\n'
                                 f'#D today\n#P0 0 5\n#N 2\n#L tth  th\n{lines}\n')
    return SourceSpec(root / 'powder', SourceKind.SPEC, options={
        'scan': '1', 'image_dir': str(images), 'image_stem': 'powder_scan1_',
        'read_image_kwargs': {'detector_shape': (195, 1475), 'raw_dtype': 'int32'}})


def test_a_twoc_scan_refines_its_arm_automatically(tmp_path):
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    from xrd_tools.session.instrument import InstrumentGeometry
    from tests.core.test_stacked_arm_extension import WAVELENGTH
    truth = {'nu_offset': .02, 'del_offset': -.05}
    spec = _powder_scan(tmp_path / 'scan', (6., 10.7, 15.3, 20., 24.7, 29.3, 34.), truth)
    seed = StackedDetectorArm(nu_motor=None, del_motor='tth', nu_offset=.017, del_offset=-.047)
    record = seed.to_pyfai_goniometer(DetectorCalibration(PONI(
        dist=.392, poni1=.0353, poni2=.0447, wavelength=WAVELENGTH, detector='Pilatus300kw'),
        detector_config={'orientation': 3}))
    geometry = InstrumentGeometry('twoc').to_record()
    fit = refine_arm_from_scan(spec, record, source_motors=(('del_angle', 'tth'),), max_frames=7,
                               instrument_geometry=geometry)
    provenance = fit.asset['provenance']
    assert is_single_arm(fit.asset['calibration']) and provenance['source_motors'] == {'del_angle': 'tth'}
    assert provenance['instrument_geometry'] == geometry and provenance['arm_model'] == 'stacked'
    assert provenance['automatic_refinement']['validation']['reason'] == (
        'a single detector arm holds one circle at zero: the extension does not apply')
    assert all(set(frame['motors']) == {'tth'} for frame in provenance['frames'])
    params = dict(zip(fit.asset['calibration']['param_names'], fit.asset['calibration']['param']))
    assert fit.rms_q < 2e-3
    assert params['nu_offset'] == pytest.approx(truth['nu_offset'], abs=5e-4)
    assert params['del_offset'] == pytest.approx(truth['del_offset'], abs=5e-4)
    assert params['dist'] == pytest.approx(.39, abs=5e-4)


def _del_only_root():
    configured = os.environ.get('XDART_TEST_DATA')
    root = Path(configured or '') / 'stitching/data_del_only'
    if not configured or not root.is_dir():
        pytest.skip('needs XDART_TEST_DATA with stitching/data_del_only')
    return root


def _scan7_controls(root, *, nu):
    """BL 17-2 LaB6 16 keV, a psic whose nu stayed at zero: scan 7's saved control points."""
    from pyFAI.control_points import ControlPoints
    from xrd_tools.integrate.refine import ControlFrame
    from xrd_tools.io.spec import read_spec_scan_columns
    columns = dict(read_spec_scan_columns(root / 'calibration/LaB6_16keV', '7.1')[0])
    controls = []
    for index in (2, 5, 15, 30, 38):
        points = ControlPoints(str(root / f'processed_data/b_thampy_LaB6_16keV_scan7_{index:04d}.npt'))
        values = np.asarray([(y, x, points.get(lbl=label).ring) for label in points.get_labels()
                             for y, x in points.get(lbl=label).points], float)
        motors = {'del': float(columns['del'][index]), **({'nu': 0.} if nu else {})}
        controls.append(ControlFrame(values[:, 0], values[:, 1], values[:, 2], motors))
    return controls, points.calibrant


def _fit_scan7(root, *, nu):
    from xrd_tools.integrate.refine import refine_goniometer
    controls, calibrant = _scan7_controls(root, nu=nu)
    return refine_goniometer(str(root / 'processed_data/b_thampy_LaB6_16keV_scan7_0002.poni'), controls,
                             rot1_motor='nu' if nu else None, rot2_motor='del', calibrant=calibrant,
                             fit_scales=False, model='stacked_del_on_nu')


def test_the_real_del_only_scan_fits_the_same_as_a_2theta_arm_and_as_psic():
    root = _del_only_root()
    single, psic = _fit_scan7(root, nu=False), _fit_scan7(root, nu=True)
    assert single.success and psic.success and single.rms_q < .005
    assert single.diffractometer.stacked_arm.single
    assert {key: psic.params[key] for key in single.params} == single.params
    delta = np.array([5.95, 7.375, 12.125, 19.25, 23.05])
    assert _worst(create_multigeometry_integrator_series_from_geometry(single.diffractometer, {'del': delta}),
                  create_multigeometry_integrator_series_from_geometry(
                      psic.diffractometer, {'del': delta, 'nu': np.zeros(5)})) == 0


@pytest.mark.slow
def test_the_real_del_only_scan_stitches_the_same_as_a_2theta_arm_and_as_psic(tmp_path):
    from xrd_tools.analysis.module_transaction import (
        MetadataColumnSelector, ModuleDisposition, ModuleKind, ModuleOutputRequest, ModuleSourceReceipt,
    )
    from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table
    from xrd_tools.analysis.stitch_operation import (
        StitchGeometryInput, StitchGeometryKind, StitchOperationPlan, capture_stitch_geometry,
        prepare_stitch_operation, run_stitch_operation,
    )
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, AnalysisArtifactOverwrite
    root = _del_only_root()
    fit = _fit_scan7(root, nu=False)
    # a copied Project: the calibration must resolve inside it, and the fixture stays untouched
    project = tmp_path / 'project'
    (project / 'images').mkdir(parents=True)
    spec_file = root / 'calibration/LaB6_16keV'
    shutil.copy2(spec_file, project / spec_file.name)
    images = sorted((root / 'images').glob('b_thampy_LaB6_16keV_scan8_*.raw'))
    assert len(images) == 41
    for image in images:
        shutil.copy2(image, project / 'images' / image.name)
    arm, calibration = fit.diffractometer.stacked_arm, fit.diffractometer.calibration
    (project / 'single.json').write_text(json.dumps(arm.to_pyfai_goniometer(calibration)))
    (project / 'psic.json').write_text(json.dumps(replace(arm, nu_motor='nu').to_pyfai_goniometer(calibration)))
    # scan 8 repeats scan 7's del positions with a valid recorded energy (scan 7's is -2165 eV)
    source = SourceSpec(project / spec_file.name, SourceKind.SPEC, options={
        'scan': '8.1', 'image_dir': str(project / 'images'), 'image_stem': 'b_thampy_LaB6_16keV_scan8_',
        'read_image_kwargs': {'detector_shape': (195, 1475), 'raw_dtype': 'int32', 'threshold': 8e5,
                              'rotation': 0}})
    table = run_metadata_table(MetadataTablePlan(source))
    labels = tuple(range(0, 41, 4))
    curves = {}
    for name, motors in (('single', (('del_angle', 'del'),)),
                         ('psic', (('del_angle', 'del'), ('nu_angle', 'nu')))):
        receipt = ModuleSourceReceipt.from_metadata_table(
            table, kind=ModuleKind.STITCH, selected_labels=labels,
            resolved_selectors=tuple(MetadataColumnSelector(motor) for _position, motor in motors))
        geometry = capture_stitch_geometry(StitchGeometryInput(
            project / f'{name}.json', StitchGeometryKind.PYFAI_GONIOMETER_JSON, source_motors=motors))
        plan = StitchOperationPlan(geometry, mode='1d', npt_1d=1500, radial_range=(1.0, 6.2),
                                   use_detector_mask=True, max_frame_bytes=64 << 20)
        request = prepare_stitch_operation(receipt, ModuleOutputRequest(
            project / f'{name}.nexus', AnalysisArtifactKind.STITCH_1D, AnalysisArtifactOverwrite.CREATE_NEW),
            plan, project_root=project)
        read = {column for item in request.manifest.contributions for column, _occurrence, _value in item.values}
        assert read == {motor for _position, motor in motors}  # the 2θ arm never reads nu
        result = run_stitch_operation(request)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED
        curves[name] = (result.payload.axis('q'), result.payload.intensity)
    (q, single), (q_psic, psic) = curves['single'], curves['psic']
    assert np.array_equal(q, q_psic)
    np.testing.assert_array_equal(single, psic)
    assert np.isfinite(single).sum() > .8 * single.size
    for d_spacing in (4.1569, 2.9394, 2.4000, 2.0785):  # LaB6 100, 110, 111, 200
        expected = 2 * math.pi / d_spacing
        window = (q > expected - .08) & (q < expected + .08)
        assert abs(q[window][np.nanargmax(single[window])] - expected) < .01
    assert hashlib.sha256(spec_file.read_bytes()).hexdigest() == hashlib.sha256(
        (project / spec_file.name).read_bytes()).hexdigest()
