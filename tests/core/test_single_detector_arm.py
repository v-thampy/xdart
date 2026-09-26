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
    for motors, references in (((('del', 'tth'),), (('del', 1.), ('nu', 0.))),
                               ((('nu', 'nu'),), (('nu', 0.),))):
        with pytest.raises(ValueError, match=r'del \(and nu\)'):
            StitchGeometryInput(poni, StitchGeometryKind.PONI, source_motors=motors,
                                reference_motor_positions=references)


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
    with pytest.raises(ValueError, match='no nu positions'):
        fit_calibration(frames, del_motor='tth', nu_motor=None, arm_model='stacked_extended')
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
    q_rings = 2 * np.pi / np.asarray(calibrant.dSpacing)[:25]
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
        'a single detector arm has no nu rows: the extension does not apply')
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
