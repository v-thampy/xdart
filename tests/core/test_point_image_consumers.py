"""Confirmed point grouping through the production selection/metadata/reduction owners."""
from dataclasses import replace

import numpy as np
import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.sources.point_images import PointImagePolicy


def _source(tmp_path):
    path = tmp_path / 'sample'
    path.write_text('#F sample\n#E 1\n#D today\n#O0 del  nu\n\n'
                    '#S 1 ascan del 10 11 1 1\n#D today\n#P0 10 0\n#N 2\n'
                    '#L del  nu\n10 0\n11 0\n')
    for index, value in enumerate((10, 4, 20, 8)):
        np.full((2, 2), value, dtype=np.uint16).tofile(tmp_path / f'sample_scan1_{index:04}.raw')
    return SourceSpec(path, SourceKind.SPEC, options={'scan': '1.1', 'image_dir': str(tmp_path),
        'read_image_kwargs': {'detector_shape': (2, 2), 'raw_dtype': 'uint16'},
        'require_complete_images': True})


def test_selection_waits_for_confirmation_then_keeps_policy(tmp_path):
    from xrd_tools.analysis.stitch_sources import StitchSourceSelection, resolve_stitch_sources
    spec = _source(tmp_path)
    selection = StitchSourceSelection('Spec', (str(spec.uri),), '1')
    pending = resolve_stitch_sources(selection, spec.options['read_image_kwargs'])
    assert not pending.groups and len(pending.pending) == 1
    assert pending.pending[0].point_count == 2 and pending.pending[0].image_count == 4
    policy = PointImagePolicy(2, 'average')
    selection = replace(selection, point_image_choices=((str(spec.uri), '1.1', policy),))
    result = resolve_stitch_sources(selection, spec.options['read_image_kwargs'])
    assert not result.pending and len(result.groups) == 1
    assert result.groups[0][0].options['point_images'] == policy.to_dict()
    assert selection.fingerprint_value != replace(selection, point_image_choices=()).fingerprint_value


def test_grouped_metadata_projects_points_and_retains_source_policy(tmp_path):
    from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table, AnalysisDisposition
    spec = _source(tmp_path)
    spec = replace(spec, options={**spec.options, 'point_images': PointImagePolicy(2, 'average').to_dict()})
    table = run_metadata_table(MetadataTablePlan(spec))
    assert table.disposition is AnalysisDisposition.COMPLETED, table.diagnostics
    assert table.labels == (0, 1)
    assert table.receipt.source_spec.options['point_images']['operation'] == 'average'
    assert 'point_images' not in {column.name for column in table.columns}


def test_signed_difference_is_conditioned_before_combination(tmp_path):
    from xrd_tools.sources.registry import open_source
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning, conditioned_stitch_frame
    spec = _source(tmp_path)
    spec = replace(spec, options={**spec.options, 'point_images': PointImagePolicy(2, 'difference').to_dict()})
    source = open_source(spec)
    try:
        image = conditioned_stitch_frame(source, 0, StitchConditioning('manual', 0, 100))
        np.testing.assert_array_equal(image, np.full((2, 2), -6.))
        image = conditioned_stitch_frame(source, 0, StitchConditioning('manual', 5, 100))
        assert np.isnan(image).all()
    finally:
        getattr(source, 'close', lambda: None)()


@pytest.mark.parametrize('operation,signal,variance,dark', [
    ('average', 8., 4., 2.), ('difference', -4., 16., 0.), ('select', 6., 6., 2.)])
def test_stitch_native_statistics_sigma_and_signed_screen(tmp_path, operation, signal, variance, dark):
    from tests.core.test_stitch_corrections import _ai, SHAPE
    from xrd_tools.sources.point_images import PointImageResult
    from xrd_tools.integrate.multi import stitch_1d_streaming
    from xrd_tools.integrate.frame_screen import FrameScreen, FrameScreenPolicy
    from pyFAI.multi_geometry import MultiGeometry
    image = np.full(SHAPE, signal)
    supplied = np.full(SHAPE, variance)
    result = PointImageResult(image, supplied, 0. if operation == 'difference' else 1.)
    screen = FrameScreen(FrameScreenPolicy(), [0, 1, 2], [0, 0, 0])
    actual, _, _ = stitch_1d_streaming([result] * 3, 3, lambda: iter([_ai(), _ai(), _ai()]),
        npt=24, method='full', radial_range=(0., 2.), azimuth_range=(-180., 180.),
        correct_solid_angle=False, error_model='poisson', dark=np.full(SHAPE, 2.), frame_screen=screen)
    ai = _ai()
    ai.detector.darkcurrent = np.full(SHAPE, dark)
    expected = MultiGeometry([ai] * 3, unit='q_A^-1', radial_range=(0., 2.),
        azimuth_range=(-180., 180.), threadpoolsize=0).integrate1d([image] * 3, 24,
        correctSolidAngle=False, lst_variance=[supplied + dark] * 3, method='full')
    covered = expected.count > 0
    assert covered.any() and (~covered).any()
    np.testing.assert_array_equal(np.isnan(actual.intensity), ~covered)
    np.testing.assert_array_equal(np.isnan(actual.sigma), ~covered)
    np.testing.assert_allclose(actual.intensity[covered], expected.intensity[covered], rtol=1e-12)
    np.testing.assert_allclose(actual.sigma[covered], expected.sigma[covered], rtol=1e-12)
    assert not any(row.excluded for row in screen.verdicts)


def test_calibration_stages_conditioned_point_and_reopens_exact_policy(tmp_path):
    from xrd_tools.integrate.stitch_calibration import capture_calibration_frames, load_calibration_session, calibration_frame_mask
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning
    from xrd_tools.io.image import read_image
    spec = _source(tmp_path)
    spec = replace(spec, options={**spec.options, 'point_images': PointImagePolicy(2, 'difference').to_dict()})
    frames = capture_calibration_frames(spec, [0], tmp_path / 'session', conditioning=StitchConditioning('manual', 0, 100))
    np.testing.assert_equal(read_image(frames[0].image_path), -6.)
    assert frames[0].source_options['options']['point_images']['operation'] == 'difference'
    assert not calibration_frame_mask(frames[0], conditioning=StitchConditioning('manual', 0, 100)).any()
    reopened = load_calibration_session(tmp_path / 'session/calibration-session.json')
    assert reopened == frames


def _point_stitch(tmp_path, operation='average', mode='1d'):
    import json
    import tifffile
    from tests.core.test_point_images import _spec_images, _confirmed
    from tests.core.test_stitch_operation import _goniometer_record
    from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table
    from xrd_tools.analysis.module_transaction import ModuleSourceReceipt, ModuleKind, MetadataColumnSelector, ModuleOutputRequest
    from xrd_tools.analysis.stitch_operation import StitchOperationPlan, StitchGeometryInput, StitchGeometryKind, capture_stitch_geometry, prepare_stitch_operation
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, AnalysisArtifactOverwrite
    from xrd_tools.io.output_path import resolve_finite_output_target
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    spec = _confirmed(_spec_images(inputs), PointImagePolicy(2, operation))
    for index in range(4):
        value = (20 if index % 2 == 0 else 10) if operation == 'difference' else index + 10
        tifffile.imwrite(inputs / f'calibrant_scan001_{index:04}.tif', np.full((195, 487), value, np.uint16))
    table = run_metadata_table(MetadataTablePlan(spec))
    source = ModuleSourceReceipt.from_metadata_table(table, kind=ModuleKind.STITCH,
        selected_labels=(0, 1), resolved_selectors=(MetadataColumnSelector('del'), MetadataColumnSelector('nu')))
    path = tmp_path / 'geometry.json'
    path.write_text(json.dumps(_goniometer_record()))
    geometry = capture_stitch_geometry(StitchGeometryInput(path, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
        source_motors=(('del_angle', 'del'), ('nu_angle', 'nu'))))
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning
    plan = StitchOperationPlan(geometry, native_records=True, mode=mode, npt_1d=24, npt_rad_2d=24, npt_azim_2d=16,
        conditioning=StitchConditioning('manual', 0, 100))
    target = resolve_finite_output_target(tmp_path, 'native', operation_token=f'stitch-{mode}')
    kind = AnalysisArtifactKind.STITCH_1D if mode == '1d' else AnalysisArtifactKind.STITCH_2D
    output = ModuleOutputRequest(target, kind, AnalysisArtifactOverwrite.CREATE_NEW, artifact_family='native')
    request = prepare_stitch_operation(source, output, plan, project_root=tmp_path)
    return spec, request


def test_native_stitch_manifest_has_every_used_constituent(tmp_path):
    from xrd_tools.analysis.stitch_native import frame_seeds
    spec, request = _point_stitch(tmp_path)
    source = request.module.source
    records = frame_seeds(source, request.manifest, request.plan)
    for index, record in enumerate(records):
        assert record['point_images']['operation'] == 'average'
        assert [item['source_frame_index'] for item in record['source_constituents']] == [0, 0]
        assert [item['source_path'].split('_')[-1] for item in record['source_constituents']] == [f'{2*index:04}.tif', f'{2*index+1:04}.tif']
        assert 'source_path' not in record


def test_native_point_contributor_reduces_saved_operands(tmp_path):
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.stitch_operation import run_stitch_operation, read_stitch_contributor_view, _runtime_geometry, _runtime_plan
    from xrd_tools.analysis.plans import run_stitch
    from xrd_tools.sources.registry import open_source
    spec, request = _point_stitch(tmp_path, 'difference', '2d')
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    assert result.frame_screen['positive_total_screening'] is False
    assert result.frame_screen['positive_total_screening_bypassed_labels'] == [0, 1]
    assert all(row['positive_total_screening'] is False for row in result.payload.contributing_frames)
    row = result.payload.contributing_frames[0]
    scan = row['key'].split('/')[0][5:]
    view = read_stitch_contributor_view(result.payload, 0, scan=scan)
    diff, _ = _runtime_geometry(request.plan.geometry, request.plan.geometry.content)
    expected = run_stitch(_runtime_plan(request.plan, diff, None), open_source(spec), frame_indices=(0,), collect_frame_records=False)
    np.testing.assert_allclose(view.intensity_1d, expected.profile[0].intensity)
    np.testing.assert_allclose(view.intensity_2d, expected.payload.intensity.T)
    assert (view.intensity_1d[view.intensity_1d != 0] < 0).all()


def test_automatic_calibration_masks_each_average_operand(tmp_path):
    from tests.core.test_stitch_autocalibration import ring_scan, five_parameter_seed, ROW_ANGLES
    from xrd_tools.integrate.stitch_autocalibration import extract_scan_ring_points
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning
    spec = ring_scan(tmp_path / 'scan', [(d, 0.) for d in (6., 20., 34.)])
    files = sorted((tmp_path / 'scan/images').glob('*.raw'))
    images = [np.fromfile(path, dtype='<i4').reshape(195, 1475) for path in files]
    for path in files:
        path.unlink()
    for point, image in enumerate(images):
        first = image.copy()
        first[:, :700] = 0  # mean would be positive, but one used exposure is invalid
        for offset, pixels in enumerate((first, image)):
            pixels.tofile(tmp_path / f'scan/images/ring_scan1_{2*point+offset:04}.raw')
    spec = replace(spec, options={**spec.options, 'point_images': PointImagePolicy(2, 'average').to_dict()})
    points = extract_scan_ring_points(spec, five_parameter_seed(ROW_ANGLES), max_frames=3,
        conditioning=StitchConditioning('manual', 1.))
    assert points.controls
    assert all(control.col_positions.min() >= 699 for control in points.controls)


def test_xu_difference_cancels_shared_dark_and_its_invalid_mask(tmp_path):
    from tests.core.test_xu_stitch import _receipt
    from xrd_tools.sources.registry import open_source
    from xrd_tools.integrate.xu_stitch import run_xu_hist_stitch_1d
    spec = _source(tmp_path)
    for index, value in enumerate((10, 4, 20, 8)):
        np.full((195, 1475), value, dtype=np.uint16).tofile(tmp_path / f'sample_scan1_{index:04}.raw')
    spec = replace(spec, options={**spec.options, 'point_images': PointImagePolicy(2, 'difference').to_dict(),
        'read_image_kwargs': {'detector_shape': (195, 1475), 'raw_dtype': 'uint16'}})
    source = open_source(spec)
    receipt = _receipt(tmp_path)
    try:
        kwargs = dict(frame_indices=(0, 1), q_min_A_inverse=1., q_max_A_inverse=5.2,
            npt=32, monitor_key=None, max_frame_bytes=4*1024*1024, allow_extrapolation=True)
        plain = run_xu_hist_stitch_1d(receipt, source, **kwargs)
        dark = np.full((195, 1475), 2.)
        dark[:20] = np.nan
        corrected = run_xu_hist_stitch_1d(receipt, source, dark=dark, **kwargs)
        np.testing.assert_array_equal(corrected.payload.intensity, plain.payload.intensity)
        np.testing.assert_array_equal(corrected.diagnostics.coverage, plain.diagnostics.coverage)
        assert (plain.payload.intensity[plain.diagnostics.coverage > 0] < 0).all()
    finally:
        getattr(source, 'close', lambda: None)()


@pytest.mark.parametrize('operation', ['average', 'select'])
def test_native_positive_point_screening_summary_is_unchanged(tmp_path, operation):
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    _, request = _point_stitch(tmp_path, operation)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    assert set(result.frame_screen) == {'policy', 'screened', 'excluded', 'empty', 'partial', 'frames'}
    assert all('positive_total_screening' not in row for row in result.payload.contributing_frames)


def test_mixed_point_screening_bypass_is_recorded_per_label(tmp_path):
    from xrd_tools.analysis.module_transaction import ModuleDisposition, ModuleKind, ModuleOutputRequest, ModuleSourceGroupReceipt
    from xrd_tools.analysis.stitch_operation import prepare_stitch_operation, run_stitch_operation
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind
    from xrd_tools.io.output_path import resolve_finite_output_target
    members = []
    for operation in ('difference', 'average'):
        root = tmp_path / operation
        root.mkdir()
        _, request = _point_stitch(root, operation)
        members.append(request.module.source)
    source = ModuleSourceGroupReceipt.from_members(tuple(members), kind=ModuleKind.STITCH)
    output = ModuleOutputRequest(resolve_finite_output_target(tmp_path, 'mixed', operation_token='stitch-1d'),
        AnalysisArtifactKind.STITCH_1D, artifact_family='mixed')
    request = prepare_stitch_operation(source, output, request.plan, project_root=tmp_path)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    assert result.frame_screen['positive_total_screening_bypassed_labels'] == [0, 1]
    assert 'positive_total_screening' not in result.frame_screen
    records = result.payload.contributing_frames
    assert [row['identity']['frame_index'] for row in records] == [0, 1, 0, 1]
    assert [row['identity']['contribution_index'] for row in records] == [0, 1, 2, 3]
    assert [row.get('positive_total_screening') for row in records] == [False, False, None, None]


@pytest.mark.parametrize('operation', ['difference', 'average', 'select'])
def test_automatic_refinement_records_effective_point_screening(tmp_path, operation):
    from tests.core.test_stitch_autocalibration import ring_scan, five_parameter_seed, ROW_ANGLES
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    spec = ring_scan(tmp_path / 'scan', [(delta, 0.) for delta in (6., 20., 34.)])
    files = sorted((tmp_path / 'scan/images').glob('*.raw'))
    images = [np.fromfile(path, dtype='<i4').reshape(195, 1475) for path in files]
    for path in files:
        path.unlink()
    for index, image in enumerate(images):
        operands = ((np.full_like(image, 10), image + 10) if operation == 'difference'
                    else (image, image))
        for offset, pixels in enumerate(operands):
            pixels.tofile(tmp_path / f'scan/images/ring_scan1_{2*index+offset:04}.raw')
    spec = replace(spec, options={**spec.options, 'point_images': PointImagePolicy(2, operation, image_index=2).to_dict()})
    fit = refine_arm_from_scan(spec, five_parameter_seed(ROW_ANGLES), max_frames=3)
    automatic = fit.asset['provenance']['automatic_refinement']
    assert fit.rms_q < .01
    assert automatic['source']['options']['point_images'] == PointImagePolicy(2, operation, image_index=2).to_dict()
    if operation == 'difference':
        assert automatic['frame_screen']['positive_total_screening'] is False
        assert automatic['frame_screen']['positive_total_screening_bypassed_labels'] == [0, 1, 2]
    else:
        assert set(automatic['frame_screen']) == {'exclude', 'empty_ratio', 'partial_ratio', 'window', 'statistic'}
