"""Scan 14: the arm extension removes the nu-dependent low-q ring shift.

The oracle is the product MultiGeometry Stitch run separately on each nu row:
a LaB6 ring must land at the same q whichever row recorded it. The five-parameter
arm leaves a row-to-row spread of several hundredths of an inverse angstrom at
low q; the extension (guided or automatic) brings it within 0.005.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector,
    ModuleDisposition,
    ModuleKind,
    ModuleOutputRequest,
    ModuleSourceReceipt,
)
from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table
from xrd_tools.analysis.stitch_operation import (
    StitchGeometryInput,
    StitchGeometryKind,
    StitchOperationPlan,
    capture_stitch_geometry,
    prepare_stitch_operation,
    run_stitch_operation,
)
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, AnalysisArtifactOverwrite

_NPT_FRAMES = (0, 9, 18, 27, 35, 55, 65, 75, 130)
#: frames the automatic check may exclude: detector counts collapse while i0 stays normal
_DROPPED = {52, 99, 106, 113, 120, 164, 272}


def _root() -> Path:
    configured = os.environ.get('XDART_TEST_DATA')
    if not configured:
        pytest.skip('XDART_TEST_DATA is not configured')
    root = Path(configured) / 'stitching'
    if not root.is_dir():
        pytest.skip('authenticated Stitching test data is unavailable')
    return root


def _spec(root):
    data = root / 'data'
    return SourceSpec(data / 'calibration/LaB6_17keV', SourceKind.SPEC, options={
        'scan': '14.1', 'image_dir': str(data / 'images'),
        'image_stem': 'b_thampy_LaB6_17keV_scan14_',
        'read_image_kwargs': {'detector_shape': (195, 1475), 'raw_dtype': 'int32',
                              'threshold': 8e5, 'rotation': 0}})


@pytest.fixture(scope='module')
def guided(tmp_path_factory):
    from xrd_tools.integrate.stitch_calibration import (
        adopt_calibration_pair, capture_calibration_frames, fit_calibration, save_calibration_fit,
    )
    root = _root()
    # Stitch geometry must resolve inside its Project; copy the scan like the
    # existing Scan 14 fixture so the original files keep their identities.
    project = tmp_path_factory.mktemp('arm-refinement-project')
    (project / 'data/calibration').mkdir(parents=True)
    (project / 'data/images').mkdir()
    shutil.copy2(root / 'data/calibration/LaB6_17keV', project / 'data/calibration/LaB6_17keV')
    images = sorted((root / 'data/images').glob('b_thampy_LaB6_17keV_scan14_*.raw'))
    assert len(images) == 287
    for image in images:
        shutil.copy2(image, project / 'data/images' / image.name)
    staged = capture_calibration_frames(_spec(project), _NPT_FRAMES, project / 'session')
    controls = root / 'data/LaB6_Calibration_Poni_and_Data'
    frames = tuple(adopt_calibration_pair(
        frame, controls / f'LaB6_17keV_scan14_{frame.index:04d}.poni',
        controls / f'LaB6_17keV_scan14_{frame.index:04d}.npt') for frame in staged)
    five = fit_calibration(frames)
    extended = fit_calibration(frames, arm_model='stacked_extended')
    return {'root': project, 'five': five, 'extended': extended,
            'five_path': Path(save_calibration_fit(five, project / 'fits/five.json')),
            'extended_path': Path(save_calibration_fit(extended, project / 'fits/extended.json'))}


def _row_centres(root, geometry_path, output_dir, rows=None, count=3, xu_receipt=None):
    """Ring-centre error per nu row from the product Stitch, {ring: {nu: error}} (Å⁻¹).

    With ``xu_receipt`` (an XU calibration) the rows are stitched by the XU backend.
    """
    from pyFAI.calibrant import get_calibrant
    spec = _spec(root)
    table = run_metadata_table(MetadataTablePlan(spec))
    nu = np.asarray(next(c for c in table.columns if c.name == 'nu').numeric, dtype=float)
    calibrant = get_calibrant('LaB6')
    if xu_receipt is None:
        record = json.loads(geometry_path.read_text())
        calibrant.wavelength = record['wavelength']
        geometry = capture_stitch_geometry(StitchGeometryInput(
            geometry_path, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
            source_motors=(('del_angle', 'del'), ('nu_angle', 'nu')), image_rotation=0))
        plan = StitchOperationPlan(geometry, mode='1d', npt_1d=1500, radial_range=(1.0, 6.2),
                                   use_detector_mask=True, max_frame_bytes=64 * 1024 * 1024)
        selectors = (MetadataColumnSelector('del'), MetadataColumnSelector('nu'))
    else:
        from xrd_tools.analysis.stitch_operation import XuStitchOperationPlan
        from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
        from xrd_tools.integrate.stitch_conditioning import StitchConditioning
        from xrd_tools.integrate.xu_stitch import resolve_xu_stitch_effective_geometry
        calibrant.wavelength = xu_receipt.projection.value['acquisition']['xdart_wavelength_A'] * 1e-10
        with xu_runtime_session() as runtime:
            effective = resolve_xu_stitch_effective_geometry(xu_receipt, runtime).projection
        plan = XuStitchOperationPlan(
            xu_receipt, effective, npt_1d=1500, radial_range=(1.0, 6.2), native_records=True,
            conditioning=StitchConditioning('manual', threshold_max=800_000.0),
            max_frame_bytes=64 * 1024 * 1024)
        selectors = tuple(MetadataColumnSelector(name) for name in ('del', 'nu', 'energy')
                          if name in {column.name for column in table.columns})
    rings = (2 * np.pi / np.asarray(calibrant.dspacing))[:count]
    centres = {float(q): {} for q in rings}
    for row in (np.unique(nu) if rows is None else rows):
        labels = tuple(int(i) for i in np.flatnonzero(nu == row))
        source = ModuleSourceReceipt.from_metadata_table(
            table, kind=ModuleKind.STITCH, selected_labels=labels, resolved_selectors=selectors)
        if xu_receipt is None:
            output = ModuleOutputRequest(output_dir / f'{geometry_path.stem}-nu{row:+.1f}.nexus',
                                         AnalysisArtifactKind.STITCH_1D, AnalysisArtifactOverwrite.CREATE_NEW)
        else:
            from xrd_tools.io.output_path import resolve_finite_output_target
            family = f'xu-nu{row:+.1f}'.replace('+', 'p').replace('-', 'm').replace('.', 'd')
            output = ModuleOutputRequest(resolve_finite_output_target(root, family, operation_token='stitch-1d'),
                                         AnalysisArtifactKind.STITCH_1D, artifact_family=family)
        result = run_stitch_operation(prepare_stitch_operation(source, output, plan, project_root=root))
        assert result.terminal.disposition is ModuleDisposition.COMMITTED
        q = np.asarray(result.payload.axis('q'), dtype=float)
        intensity = np.asarray(result.payload.intensity, dtype=float)
        for target in rings:
            window = np.abs(q - target) <= 0.035
            x, y = q[window], intensity[window]
            if not np.isfinite(y).all():
                continue  # this row does not cover the ring
            baseline = np.linspace(np.median(y[:3]), np.median(y[-3:]), len(y))
            signal = y - baseline
            i = int(np.argmax(signal))
            if not 0 < i < len(signal) - 1 or signal[i] < 0.3 * baseline[i]:
                continue
            a, b, c = signal[i - 1:i + 2]
            centres[float(target)][float(row)] = (
                x[i] + 0.5 * (a - c) / (a - 2 * b + c) * (x[1] - x[0]) - target)
    return centres


def _spreads(centres):
    assert all(len(values) >= 4 for values in centres.values()), centres
    return [max(values.values()) - min(values.values()) for values in centres.values()]


@pytest.mark.slow
def test_scan14_guided_arm_extension_aligns_the_nu_rows(guided, tmp_path):
    from pyFAI.goniometer import Goniometer
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.integrate.multi import create_multigeometry_integrators_from_geometry
    extended = guided['extended']
    assert extended.asset['provenance']['arm_model'] == 'stacked_extended'
    record = json.loads(guided['extended_path'].read_text())
    assert 'xdart_stacked_arm_v2' in record['trans_function']['constants']
    loaded = Diffractometer.from_pyfai_goniometer(
        record, source_motors={'del_angle': 'del', 'nu_angle': 'nu'})
    ai, = create_multigeometry_integrators_from_geometry(loaded, {'del': np.array([25.]), 'nu': np.array([22.5])})
    stock = Goniometer.sload(str(guided['extended_path'])).get_ai((25., 22.5))
    np.testing.assert_allclose(ai.qArray(), stock.qArray(), atol=2e-12, rtol=0)
    assert extended.rms_q < 0.6 * guided['five'].rms_q
    five = _spreads(_row_centres(guided['root'], guided['five_path'], tmp_path))
    fixed = _spreads(_row_centres(guided['root'], guided['extended_path'], tmp_path))
    print(f'nu-row spread (Å⁻¹, rings 100/110/111): five-parameter {np.round(five, 4).tolist()} '
          f'extended {np.round(fixed, 4).tolist()}; rms {guided["five"].rms_q:.5f} -> {extended.rms_q:.5f}')
    assert five[0] > 0.02, five  # the five-parameter arm shifts LaB6 (100) with nu
    assert max(fixed) <= 0.005, (fixed, five)


@pytest.mark.slow
def test_scan14_automatic_refinement_from_the_calibrant_scan(guided, tmp_path):
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    from xrd_tools.integrate.stitch_calibration import save_calibration_fit
    fit = refine_arm_from_scan(_spec(guided['root']), guided['five_path'], max_counts=8e5)
    provenance = fit.asset['provenance']
    automatic = provenance['automatic_refinement']
    assert provenance['arm_model'] == 'stacked_extended', automatic['validation']
    assert automatic['validation']['improvement'] >= 0.4, automatic['validation']
    assert {frame['motors']['nu'] for frame in provenance['frames']} == {-10., -3.5, 3., 9.5, 16., 22.5, 29.}
    excluded = {frame['index'] for frame in automatic['excluded_frames']}
    assert excluded <= _DROPPED, automatic['excluded_frames']
    assert automatic['validation']['coverage'] == {'nu_rows_available': 7, 'nu_rows_selected': 7}
    assert [fold['nu'] for fold in automatic['validation']['folds']] == [-10., -3.5, 3., 9.5, 16., 22.5, 29.]
    path = Path(save_calibration_fit(fit, guided['root'] / 'fits/automatic.json'))
    fixed = _spreads(_row_centres(guided['root'], path, tmp_path))
    print(f"automatic: {len(provenance['frames'])} frames, excluded {sorted(excluded)}, "
          f"held-out {automatic['validation']['held_out_sigma_A^-1']}, nu-row spread {np.round(fixed, 4).tolist()}")
    assert max(fixed) <= 0.005, fixed


@pytest.mark.slow
def test_scan14_small_frame_budget_still_spans_every_nu_row(guided):
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    fit = refine_arm_from_scan(_spec(guided['root']), guided['five_path'], max_frames=7, max_counts=8e5)
    provenance = fit.asset['provenance']
    validation = provenance['automatic_refinement']['validation']
    assert validation['coverage'] == {'nu_rows_available': 7, 'nu_rows_selected': 7}
    assert len({frame['motors']['nu'] for frame in provenance['frames']}) == 7
    assert 'fewer than four nu rows' not in validation['reason'], validation
    assert len(validation['folds']) == 7


@pytest.mark.slow
def test_scan14_automatic_fit_predicts_held_out_nu_rows(guided, tmp_path):
    """The O4 oracle: rows never seen by the fit land within 0.0025 Å⁻¹.

    The maintainer raised it from 0.002 on 2026-09-24. The fit is reproducible now, but
    pyFAI's ring extraction depends on its random seed: runs before the seeding fix
    ranged 0.0017-0.00205, and the seeded value is 0.0019675. The five-parameter arm
    misses by 0.02-0.04.
    """
    from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
    from xrd_tools.integrate.stitch_calibration import save_calibration_fit
    table = run_metadata_table(MetadataTablePlan(_spec(guided['root'])))
    column = {c.name: np.asarray(c.numeric, dtype=float) for c in table.columns if c.name in ('nu', 'del')}
    train = tuple(int(i) for i in np.flatnonzero(
        np.isin(column['nu'], (-10., 3., 16., 29.)) & np.isin(column['del'], np.arange(5., 46., 5.))))
    fit = refine_arm_from_scan(_spec(guided['root']), guided['five_path'], frame_indices=train, max_counts=8e5)
    assert fit.asset['provenance']['arm_model'] == 'stacked_extended'
    path = Path(save_calibration_fit(fit, guided['root'] / 'fits/held-out.json'))
    centres = _row_centres(guided['root'], path, tmp_path, rows=(-3.5, 9.5, 22.5), count=6)
    errors = [value for ring in centres.values() for value in ring.values()]
    print(f'held-out rows: {len(errors)} ring centres, max |error| {max(abs(e) for e in errors):.4f} Å⁻¹')
    assert len(errors) >= 14, centres
    assert max(abs(e) for e in errors) <= 0.0025, centres


def _stitch(root, geometry_path, labels, target, **plan_fields):
    table = run_metadata_table(MetadataTablePlan(_spec(root)))
    source = ModuleSourceReceipt.from_metadata_table(
        table, kind=ModuleKind.STITCH, selected_labels=labels,
        resolved_selectors=(MetadataColumnSelector('del'), MetadataColumnSelector('nu')))
    geometry = capture_stitch_geometry(StitchGeometryInput(
        geometry_path, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
        source_motors=(('del_angle', 'del'), ('nu_angle', 'nu')), image_rotation=0))
    plan = StitchOperationPlan(geometry, mode='1d', npt_1d=1500, radial_range=(1.0, 6.2),
                               use_detector_mask=True, max_frame_bytes=64 * 1024 * 1024, **plan_fields)
    output = ModuleOutputRequest(target, AnalysisArtifactKind.STITCH_1D, AnalysisArtifactOverwrite.CREATE_NEW)
    result = run_stitch_operation(prepare_stitch_operation(source, output, plan, project_root=root))
    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    return result


@pytest.mark.slow
def test_scan14_sample_on_rotation_axis_drops_the_calibrant_offset(guided, tmp_path):
    from xrd_tools.core.geometry import Diffractometer
    record = json.loads(guided['extended_path'].read_text())
    diff = Diffractometer.from_pyfai_goniometer(record, source_motors={'del_angle': 'del', 'nu_angle': 'nu'})
    assert diff.stacked_arm.has_sample_offset
    zeroed = diff.stacked_arm.without_sample_offset().to_pyfai_goniometer(diff.calibration)
    zeroed_path = guided['root'] / 'fits/extended-without-offset.json'
    zeroed_path.write_text(json.dumps(zeroed))
    labels = tuple(range(0, 287, 7))
    on_axis = _stitch(guided['root'], guided['extended_path'], labels, tmp_path / 'axis.nexus',
                      sample_position='rotation_axis')
    reference = _stitch(guided['root'], zeroed_path, labels, tmp_path / 'zeroed.nexus')
    calibrated = _stitch(guided['root'], guided['extended_path'], labels, tmp_path / 'calibrated.nexus')
    np.testing.assert_array_equal(on_axis.payload.intensity, reference.payload.intensity)
    assert not np.array_equal(on_axis.payload.intensity, calibrated.payload.intensity, equal_nan=True)
    assert json.loads(on_axis.payload.provenance_json)['plan']['sample_position'] == 'rotation_axis'


@pytest.mark.slow
def test_scan14_xu_transfer_holds_on_every_pixel_and_aligns_the_nu_rows(guided, tmp_path):
    """P5 O1/O2: the extended arm in xrayutilities, gated on all 287 frames, and its Stitch."""
    from xrd_tools.analysis.xu_stitch_calibration import (
        XuStitchCalibrationInput, _canonical_bytes, capture_xu_stitch_calibration,
    )
    from xrd_tools.integrate.xu_transfer import transfer_extended_arm_to_xu
    table = run_metadata_table(MetadataTablePlan(_spec(guided['root'])))
    column = {c.name: np.asarray(c.numeric, dtype=float) for c in table.columns if c.name in ('nu', 'del')}
    angles = list(zip(column['del'], column['nu']))
    asset, transfer = transfer_extended_arm_to_xu(guided['extended'].asset['calibration'],
                                                  validation_angles=angles)
    print(f"XU transfer: max |dq| {transfer['max_abs_dq_A^-1']:.3g} over {transfer['validation_frames']} frames")
    assert transfer['validation_frames'] == 287 and transfer['max_abs_dq_A^-1'] <= 1e-8
    target = guided['root'] / 'calibration/xu/extended-v3.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_canonical_bytes(asset))
    receipt = capture_xu_stitch_calibration(XuStitchCalibrationInput('calibration/xu/extended-v3.json'),
                                            project_root=guided['root'])
    xu = _row_centres(guided['root'], None, tmp_path, xu_receipt=receipt)
    mg = _row_centres(guided['root'], guided['extended_path'], tmp_path)
    differences = [xu[ring][row] - mg[ring][row] for ring in mg for row in mg[ring] if row in xu[ring]]
    print(f'XU v3 nu-row spread {np.round(_spreads(xu), 4).tolist()}; '
          f'max |XU - MultiGeometry| ring centre {max(abs(d) for d in differences):.2e} Å⁻¹')
    assert max(_spreads(xu)) <= 0.005
    assert len(differences) >= 10 and max(abs(d) for d in differences) <= 1e-3
