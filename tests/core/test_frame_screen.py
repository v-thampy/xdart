"""Dropped-frame screening: the rule, the streaming window, and both Stitch kernels."""
import json

import numpy as np
import pytest

from xrd_tools.integrate.frame_screen import FrameScreen, FrameScreenPolicy, screen_totals


def test_rule_classifies_against_same_row_neighbours():
    totals = [100., 101., 0.5, 99., 102., 15., 100., 98., 50., 51., 49.]
    rows = [0] * 8 + [1] * 3
    verdicts = screen_totals(totals, list(range(11)), rows, FrameScreenPolicy())
    status = [v.status for v in verdicts]
    assert status[2] == 'empty' and verdicts[2].excluded
    assert status[5] == 'partial' and not verdicts[5].excluded
    assert status[8:] == ['ok', 'ok', 'ok']  # row 1 is judged against row 1 only
    partial = screen_totals(totals, list(range(11)), rows, FrameScreenPolicy(exclude='partial'))
    assert [v.label for v in partial if v.excluded] == [2, 5]
    lone = screen_totals([10., 0., 10.], [0, 1, 2], [0, 1, 2], FrameScreenPolicy())
    assert {v.status for v in lone} == {'unscreened'}


def test_streaming_window_matches_the_complete_rule_and_keeps_order():
    rng = np.random.default_rng(5)
    totals = rng.uniform(50, 150, 60)
    totals[[7, 31, 32]] = [0., 3., 12.]
    rows = np.repeat(np.arange(6), 10)
    policy = FrameScreenPolicy(exclude='partial', window=2)
    screen = FrameScreen(policy, list(range(60)), rows.tolist())
    released = []
    for index, total in enumerate(totals):
        released += screen.push(total, index)
        assert len(screen._pending) <= policy.window
    released += screen.finish()
    assert [item for item, _ in released] == list(range(60))
    assert screen.verdicts == screen_totals(totals, list(range(60)), rows.tolist(), policy)
    assert screen.summary()['excluded'] == [7, 31, 32]


def test_policy_validation():
    with pytest.raises(ValueError):
        FrameScreenPolicy(exclude='all')
    with pytest.raises(ValueError):
        FrameScreenPolicy(empty_ratio=.3, partial_ratio=.2)
    with pytest.raises(ValueError):
        FrameScreenPolicy(window=0)


def _ai(delta):
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    return AzimuthalIntegrator(dist=.39, poni1=3.7 * 75e-6, poni2=5.9 * 172e-6, rot2=np.deg2rad(delta),
                               detector=Detector(pixel1=75e-6, pixel2=172e-6, max_shape=(9, 13)),
                               wavelength=.729e-10)


def test_multigeometry_stream_excludes_an_empty_frame_exactly():
    from xrd_tools.integrate.multi import stitch_1d_streaming
    angles = [2., 4., 6., 8., 10., 12.]
    images = [np.full((9, 13), 100.) for _ in angles]
    images[3] = np.zeros((9, 13))

    def run(indices, screen=None):
        return stitch_1d_streaming([images[i] for i in indices], len(indices),
                                   lambda: (_ai(angles[i]) for i in indices), npt=40,
                                   radial_range=(0., 3.), method='full', frame_screen=screen)[0]

    everything = list(range(6))
    screen = FrameScreen(FrameScreenPolicy(), everything, [0] * 6)
    screened = run(everything, screen)
    assert screen.summary()['excluded'] == [3]
    reference = run([0, 1, 2, 4, 5])
    np.testing.assert_array_equal(screened.intensity, reference.intensity)
    kept = run(everything, FrameScreen(FrameScreenPolicy(exclude='none'), everything, [0] * 6))
    np.testing.assert_array_equal(kept.intensity, run(everything).intensity)
    assert not np.array_equal(kept.intensity, screened.intensity, equal_nan=True)


def test_xu_histogram_stream_excludes_an_empty_frame_exactly(tmp_path):
    from pyFAI.detectors import Detector
    from xrd_tools.analysis.xu_stitch_calibration import (
        XuStitchCalibrationInput, authored_surface_calibration, capture_xu_stitch_calibration,
    )
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.integrate.xu_stitch import resolve_xu_stitch_effective_geometry, run_xu_hist_stitch_1d
    from xrd_tools.sources.memory import MemoryFrameSource
    with xu_runtime_session() as session:
        wavelength = float(session.xu.en2lam(17000.004))
    value = authored_surface_calibration(
        detector=Detector(pixel1=75e-6, pixel2=172e-6, max_shape=(9, 13)), cch1=3.2, cch2=5.4,
        distance_m=.39, del_offset_deg=1.7, nu_offset_deg=-.8, energy_eV=17000.004,
        xu_wavelength_A=wavelength, source_motors={'del': 'del', 'nu': 'nu'}, rotation=0,
        control_domain={'del': [-50., 50.], 'nu': [-35., 35.]}, domain={'del': [-50., 50.], 'nu': [-35., 35.]},
        rms_q=.001, condition=10., n_frames=5, n_points=40)
    (tmp_path / 'geometry.json').write_bytes(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
    receipt = capture_xu_stitch_calibration(XuStitchCalibrationInput('geometry.json'), project_root=tmp_path)
    with xu_runtime_session() as session:
        solid = resolve_xu_stitch_effective_geometry(receipt, session).solid_angle.copy()
    deltas = [2., 4., 6., 8., 10., 12.]
    frames = [ScanFrame(index=i, image=(0. if i == 3 else 100.) * solid,
                        metadata={'del': d, 'nu': 0., 'monitor': 1.}) for i, d in enumerate(deltas)]

    def run(indices, screen=None):
        return run_xu_hist_stitch_1d(receipt, MemoryFrameSource(frames), frame_indices=tuple(indices),
                                     npt=20, radial_range=(0., 3.), monitor_key='monitor',
                                     max_frame_bytes=1 << 20, frame_screen=screen)

    screen = FrameScreen(FrameScreenPolicy(), list(range(6)), [0.] * 6)
    screened = run(range(6), screen)
    assert screen.summary()['excluded'] == [3]
    np.testing.assert_array_equal(screened.payload.intensity, run([0, 1, 2, 4, 5]).payload.intensity)


def test_rows_follow_the_nu_motor_for_both_geometry_kinds(tmp_path):
    from xrd_tools.analysis.module_transaction import (
        AnalysisArtifactKind, AnalysisArtifactOverwrite, ModuleOutputRequest,
    )
    from xrd_tools.analysis.stitch_operation import (
        StitchGeometryInput, StitchGeometryKind, StitchOperationPlan, _frame_screen_rows,
        capture_stitch_geometry, prepare_stitch_operation,
    )
    from tests.core.test_stitch_operation import _poni_record, _prepared
    request = _prepared(tmp_path)
    assert _frame_screen_rows(request) == ((0, 0.0), (0, 2.0))  # goniometer JSON: nu_angle ← nu
    path = tmp_path / 'geometry.poni'
    path.write_text(_poni_record(), encoding='utf-8')
    geometry = capture_stitch_geometry(StitchGeometryInput(
        path, StitchGeometryKind.PONI, source_motors=(('del', 'del'), ('nu', 'nu')),
        reference_motor_positions=(('del', 0.0), ('nu', 0.0))))
    output = ModuleOutputRequest(tmp_path / 'output' / 'poni.nexus', AnalysisArtifactKind.STITCH_1D,
                                 AnalysisArtifactOverwrite.CREATE_NEW)
    poni = prepare_stitch_operation(request.module.source, output,
                                    StitchOperationPlan(geometry, npt_1d=96, radial_range=(.1, 6.)),
                                    project_root=tmp_path)
    assert _frame_screen_rows(poni) == ((0, 0.0), (0, 2.0))


def _constant_series(root, values, nu):
    import tifffile
    from xrd_tools.analysis.module_transaction import (
        MetadataColumnSelector, ModuleKind, ModuleSourceReceipt,
    )
    from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table
    from xrd_tools.sources.selection import image_series_spec
    root.mkdir(parents=True)
    for index, value in enumerate(values):
        image = root / f'scan_{index + 1:04d}.tif'
        tifffile.imwrite(image, np.full((195, 487), value, dtype=np.float32))
        image.with_suffix('.txt').write_text(
            f'# Counters\nI0 = 1\n# Motors\ndel = {5 * index}, nu = {nu}\n'
            'User: stitch, time: Mon Jan 15 10:30:00 2024  # Temp\n', encoding='utf-8')
    table = run_metadata_table(MetadataTablePlan(image_series_spec(root / 'scan_0001.tif', metadata_format='txt')))
    return ModuleSourceReceipt.from_metadata_table(
        table, kind=ModuleKind.STITCH, selected_labels=tuple(range(1, len(values) + 1)),
        resolved_selectors=tuple(MetadataColumnSelector(name) for name in ('I0', 'del', 'nu')))


def test_grouped_scans_are_screened_only_against_themselves(tmp_path):
    import json as _json
    from xrd_tools.analysis.module_transaction import (
        ModuleDisposition, ModuleKind, ModuleOutputRequest, ModuleSourceGroupReceipt,
    )
    from xrd_tools.analysis.stitch_operation import (
        StitchGeometryInput, StitchGeometryKind, StitchOperationPlan, capture_stitch_geometry,
        prepare_stitch_operation, run_stitch_operation,
    )
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind
    from xrd_tools.io.output_path import resolve_finite_output_target
    from tests.core.test_stitch_operation import _goniometer_record
    # Two healthy scans at very different count levels on the same nu row; the dim
    # scan's last frame is genuinely dead.
    bright = _constant_series(tmp_path / 'bright', [10000.] * 4, -10)
    dim = _constant_series(tmp_path / 'dim', [10., 10., 10., 0.], -10)
    group = ModuleSourceGroupReceipt.from_members((bright, dim), kind=ModuleKind.STITCH)
    geometry_path = tmp_path / 'geometry.json'
    geometry_path.write_text(_json.dumps(_goniometer_record()), encoding='utf-8')
    geometry = capture_stitch_geometry(StitchGeometryInput(
        geometry_path, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
        source_motors=(('del_angle', 'del'), ('nu_angle', 'nu'))))
    plan = StitchOperationPlan(geometry, native_records=True, npt_1d=96,
                               conditioning=StitchConditioning('off'))
    target = resolve_finite_output_target(tmp_path, 'grouped', operation_token='stitch-1d')
    output = ModuleOutputRequest(target, AnalysisArtifactKind.STITCH_1D, artifact_family='grouped')
    request = prepare_stitch_operation(group, output, plan, project_root=tmp_path)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    assert result.frame_screen['excluded'] == [7]
    assert result.frame_screen['empty'] == [7] and result.frame_screen['partial'] == []
    return target


def _verdicts(target):
    from xrd_tools.io.analysis_artifact import read_analysis_artifact
    rows = read_analysis_artifact(target).contributing_frames
    return [(row['identity']['scan_id'], row['identity']['frame_index'],
             row.get('screen'), row.get('excluded')) for row in rows]


def test_a_saved_grouped_result_records_each_frames_verdict(tmp_path):
    target = test_grouped_scans_are_screened_only_against_themselves(tmp_path)
    verdicts = _verdicts(target)
    # Both members have a frame 4; only the dim member's (stream position 7) is excluded.
    assert [(frame, screen, excluded) for _scan, frame, screen, excluded in verdicts if excluded] == [
        (4, 'empty', True)]
    assert verdicts[7][2:] == ('empty', True) and verdicts[3][2:] == ('ok', False)
    assert {screen for _s, _f, screen, _e in verdicts} <= {'ok', 'unscreened', 'empty'}


def test_a_tampered_frame_verdict_refuses_an_attested_xu_result(tmp_path):
    # XU's execution attestation binds the result fingerprint, which covers the
    # manifest. MultiGeometry artifacts carry no stored result digest for any dataset.
    import h5py
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.io.analysis_artifact import AnalysisArtifactInvalid, read_analysis_artifact
    from xrd_tools.io.analysis_stitch_auxiliary import canonical_frame_manifest
    from tests.core.test_native_stitch_operation import _native
    request = _native(tmp_path, 'xu_hist', 'q_A^-1', grouped=False)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    target = request.module.output.target
    rows = read_analysis_artifact(target).contributing_frames
    assert all(row['screen'] in {'ok', 'unscreened'} and row['excluded'] is False for row in rows)
    with h5py.File(target, 'r+') as handle:
        manifest = json.loads(bytes(handle['entry/stitch/contributing_frames_json'][()]))
        manifest['records'][0]['screen'] = 'partial'
        del handle['entry/stitch/contributing_frames_json']
        handle['entry/stitch'].create_dataset('contributing_frames_json',
                                       data=np.bytes_(canonical_frame_manifest(manifest).encode()))
    with pytest.raises(AnalysisArtifactInvalid, match='attestation result changed'):
        read_analysis_artifact(target)


def test_a_result_saved_without_verdicts_still_reads(tmp_path, monkeypatch):
    from xrd_tools.analysis import stitch_operation
    monkeypatch.setattr(stitch_operation, '_record_frame_screen', lambda *args: None)
    target = test_grouped_scans_are_screened_only_against_themselves(tmp_path)
    verdicts = _verdicts(target)
    assert all(screen is None and excluded is None for _s, _f, screen, excluded in verdicts)


def test_grouped_xu_rows_keep_their_member(tmp_path):
    from xrd_tools.analysis.stitch_operation import _frame_screen_rows
    from tests.core.test_native_stitch_operation import _native
    rows = _frame_screen_rows(_native(tmp_path, 'xu_hist', 'q_A^-1', grouped=True))
    assert {member for member, _nu in rows} == {0, 1}
