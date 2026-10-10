"""Finite metadata/PONI authority rows through existing reduction interfaces."""
from dataclasses import replace
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.core.energy import energy_eV_to_wavelength_m
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io import open_scan
from xrd_tools.notebooks.int_replay import run_int_replay
from xrd_tools.session.run_intent_profile import dump_run_intent_profile
from tests.core.test_int_workflow_replay import _fixture


@pytest.mark.parametrize('wavelength', [1e-10, 7.3812397e-11])
def test_acquisition_fallback_and_unchanged_metadata_keep_exact_wavelength(wavelength):
    from xrd_tools.core.energy import resolve_acquisition_energy, wavelength_m_to_energy_eV
    assert resolve_acquisition_energy(wavelength)['wavelength_m'] == wavelength
    metadata = (wavelength_m_to_energy_eV(wavelength),)
    assert resolve_acquisition_energy(wavelength, metadata)['wavelength_m'] == wavelength


def _replay(intent, folder):
    return run_int_replay(json.loads(dump_run_intent_profile(intent)),
                         data_root=intent.project_root,
                         output_root=Path(intent.project_root) / folder)[0]


def _record_nexus_energy(intent, values):
    with h5py.File(intent.source_spec.uri, 'r+') as document:
        data = document['entry/data']
        data.create_dataset('energy', data=values)
        document.create_group('entry/instrument/positioners/energy')


@pytest.mark.parametrize('source_kind', ['spec', 'nexus'])
@pytest.mark.parametrize('unit', ['q_A^-1', '2th_deg'])
def test_int_replay_metadata_energy_overrides_poni(tmp_path, source_kind, unit):
    intent = _fixture(tmp_path / 'project')
    intent.processing_mode = 'Int 1D'
    intent.bai_1d_args['unit'] = unit
    if source_kind == 'spec':
        import tifffile
        root = Path(intent.project_root)
        spec = root / 'scan.spec'
        spec.write_text('#F scan.spec\n#O0 energy\n#S 1 ascan del 0 1 1 1\n'
                        '#P0 16000\n#N 2\n#L del  I0\n0 1\n1 1\n')
        for index in range(2):
            tifffile.imwrite(root / f'scan_scan1_{index:04d}.tif',
                             np.full((195, 487), 100, np.uint16))
        intent.source_spec = SourceSpec(spec, SourceKind.SPEC,
            options={'scan': '1.1', 'image_dir': str(root)})
    else:
        _record_nexus_energy(intent, [16000., 16000.])
    def run(folder):
        if source_kind == 'nexus':
            return _replay(intent, folder)
        from xrd_tools.sources import open_source
        from xrd_tools.integrate.calibration import load_detector_calibration
        from xrd_tools.session.readiness import bind_native_int_source_energy, native_int_reduction_plan
        from xrd_tools.reduction import NexusSink, run_reduction
        source = open_source(intent.source_spec)
        scan = source.to_scan()
        bind_native_int_source_energy(scan, load_detector_calibration(intent.poni_file))
        plan = native_int_reduction_plan(intent.freeze(), energy_selection=scan.extra['energy_selection'])
        path = Path(intent.project_root) / (folder + '.nexus')
        run_reduction(plan, scan, NexusSink(path, create_new=True))
        return path
    actual_path = run('metadata')
    actual = open_scan(actual_path)
    assert actual.metadata['wavelength_A'] == pytest.approx(energy_eV_to_wavelength_m(16000.) * 1e10)
    # Same existing reducer, independently configured with the effective PONI.
    poni = Path(intent.poni_file)
    poni.write_text(poni.read_text().replace('Wavelength: 1e-10',
        f'Wavelength: {energy_eV_to_wavelength_m(16000.)!r}'))
    expected = open_scan(run('reference'))
    np.testing.assert_array_equal(actual.get_1d(0).intensity, expected.get_1d(0).intensity)
    np.testing.assert_array_equal(actual.get_1d(0).q, expected.get_1d(0).q)
    from xrd_tools.core.provenance import read_provenance
    energy = read_provenance(actual_path)['config']['energy_selection']
    assert energy['source'] == 'metadata'
    assert energy['energy_eV'] == pytest.approx(16000.)
    assert energy['disagreement'] is True


def test_int_replay_missing_energy_uses_poni_and_stored_axes_stay_fixed(tmp_path):
    intent = _fixture(tmp_path / 'project')
    actual_path = _replay(intent, 'fallback')
    actual = open_scan(actual_path)
    assert actual.metadata['wavelength_A'] == pytest.approx(1.)
    q = actual.get_1d(0).q.copy()
    _record_nexus_energy(intent, [16000., 16000.])
    stored = open_scan(actual_path)
    assert stored.metadata['wavelength_A'] == pytest.approx(1.)
    np.testing.assert_array_equal(stored.get_1d(0).q, q)
    from xrd_tools.core.provenance import read_provenance
    assert read_provenance(actual_path)['config']['energy_selection']['source'] == 'calibration'


@pytest.mark.parametrize('values', [[16000., 17000.], [16000., -1.]])
def test_int_replay_invalid_or_varying_metadata_energy_refuses(tmp_path, values):
    intent = _fixture(tmp_path / 'project')
    _record_nexus_energy(intent, values)
    with pytest.raises(ValueError, match='metadata energy'):
        _replay(intent, 'refused')
    assert not (Path(intent.project_root) / 'refused').exists()


def _native_energy_request(root, unit, metadata_energy, fitted=False, poni=False):
    from tests.core.test_stitch_operation import _prepared, _goniometer_record, _poni_record
    from xrd_tools.analysis.stitch_operation import (capture_stitch_geometry, StitchGeometryInput,
        StitchGeometryKind, prepare_stitch_operation)
    from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table
    from xrd_tools.analysis.module_transaction import (ModuleSourceReceipt, ModuleKind,
        ModuleOutputRequest, MetadataColumnSelector)
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind
    from xrd_tools.io.output_path import resolve_finite_output_target
    original = _prepared(root)
    if metadata_energy is not None:
        for path in (root / 'source').glob('*.txt'):
            path.write_text(path.read_text().replace('nu =', f'energy = {metadata_energy}, nu ='))
    table = run_metadata_table(MetadataTablePlan(original.module.source.analysis.source_spec))
    selectors = original.module.source.resolved_selectors
    if metadata_energy is not None:
        selectors += (MetadataColumnSelector('energy'),)
    source = ModuleSourceReceipt.from_metadata_table(table, kind=ModuleKind.STITCH,
        selected_labels=original.module.source.selected_labels, resolved_selectors=selectors)
    geometry = original.plan.geometry
    if poni:
        path = root / 'cal.poni'
        path.write_text(_poni_record())
        geometry = capture_stitch_geometry(StitchGeometryInput(path, StitchGeometryKind.PONI,
            source_motors=(('del', 'del'), ('nu', 'nu')),
            reference_motor_positions=(('del', 0.), ('nu', 0.))))
    if fitted:
        record = _goniometer_record()
        record.update(wavelength=energy_eV_to_wavelength_m(17100.), source_energy_eV=17000.)
        path = root / 'refined.json'
        path.write_text(json.dumps(record))
        geometry = capture_stitch_geometry(StitchGeometryInput(path,
            StitchGeometryKind.PYFAI_GONIOMETER_JSON,
            source_motors=(('del_angle', 'del'), ('nu_angle', 'nu'))))
    plan = replace(original.plan, geometry=geometry, native_records=True, unit=unit, radial_range=None)
    output = ModuleOutputRequest(resolve_finite_output_target(root, 'energy', operation_token='stitch-1d'),
        AnalysisArtifactKind.STITCH_1D, artifact_family='energy')
    return prepare_stitch_operation(source, output, plan, project_root=root)


@pytest.mark.parametrize('unit', ['q_A^-1', '2th_deg'])
@pytest.mark.parametrize('metadata_energy,fitted,expected', [(None, False, 12398.41984),
    (16000., False, 16000.), (16000., True, 16100.)])
def test_native_pyfai_stitch_energy_authority(tmp_path, unit, metadata_energy, fitted, expected):
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.io.analysis_artifact import read_analysis_artifact
    request = _native_energy_request(tmp_path, unit, metadata_energy, fitted)
    wavelength = energy_eV_to_wavelength_m(expected) * 1e10
    assert request.provenance['wavelength_A'] == pytest.approx(wavelength)
    assert request.provenance['plan']['energy_selection']['energy_eV'] == pytest.approx(expected)
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    payload = read_analysis_artifact(request.module.output.target)
    assert payload.wavelength_A == pytest.approx(wavelength)
    q, tth = payload.radial_axis('q_A^-1'), payload.radial_axis('2th_deg')
    # Both persisted axes are float32; compute the physical relation in float64
    # and allow their independent half-ulp rounding.
    np.testing.assert_allclose(q, 4 * np.pi / wavelength * np.sin(np.deg2rad(tth.astype(float) / 2)),
                               rtol=2 * np.finfo(q.dtype).eps)
    if fitted:
        assert request.provenance['plan']['energy_selection']['fitted_energy_offset_eV'] == pytest.approx(100.)


@pytest.mark.parametrize('unit', ['q_A^-1', '2th_deg'])
def test_native_pyfai_poni_override_cross_checks_effective_calibration(tmp_path, unit):
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.io.analysis_artifact import read_analysis_artifact
    request = _native_energy_request(tmp_path, unit, 16000., poni=True)
    wavelength = energy_eV_to_wavelength_m(16000.)
    assert request.provenance['geometry']['effective_calibration']['wavelength'] == wavelength
    result = run_stitch_operation(request)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    assert read_analysis_artifact(request.module.output.target).wavelength_A == pytest.approx(wavelength * 1e10)


def test_native_pyfai_stitch_mixed_member_metadata_energy_refuses(tmp_path):
    from xrd_tools.analysis.module_transaction import ModuleSourceGroupReceipt, ModuleKind
    from xrd_tools.analysis.stitch_operation import prepare_stitch_operation, StitchOperationRefused
    first_root, second_root = tmp_path / 'first', tmp_path / 'second'
    first_root.mkdir(); second_root.mkdir()
    first = _native_energy_request(first_root, 'q_A^-1', 16000.)
    second = _native_energy_request(second_root, 'q_A^-1', 18000.)
    grouped = ModuleSourceGroupReceipt.from_members((first.module.source, second.module.source),
                                                    kind=ModuleKind.STITCH)
    with pytest.raises(StitchOperationRefused, match='metadata energy'):
        prepare_stitch_operation(grouped, first.module.output, first.plan, project_root=tmp_path)


@pytest.mark.parametrize('unit', ['q_A^-1', '2th_deg'])
def test_native_xu_stitch_metadata_energy_and_axes(tmp_path, unit):
    from tests.core.test_stitch_operation import _xu_prepared
    from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table
    from xrd_tools.analysis.module_transaction import ModuleSourceReceipt, ModuleKind, ModuleOutputRequest, ModuleDisposition
    from xrd_tools.analysis.stitch_operation import prepare_stitch_operation, run_stitch_operation
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, read_analysis_artifact
    from xrd_tools.io.output_path import resolve_finite_output_target
    original = _xu_prepared(tmp_path, energy_eV=None)
    path = tmp_path / 'xu-source/scan_0001.txt'
    path.write_text(path.read_text().replace('I0 = 2', 'I0 = 2, energy = 16000'))
    table = run_metadata_table(MetadataTablePlan(original.module.source.analysis.source_spec))
    source = ModuleSourceReceipt.from_metadata_table(table, kind=ModuleKind.STITCH,
        selected_labels=(1,), resolved_selectors=original.module.source.resolved_selectors)
    plan = replace(original.plan, native_records=True, unit=unit,
                   q_min_A_inverse=None, q_max_A_inverse=None, radial_range=None)
    target = resolve_finite_output_target(tmp_path, 'xu-energy', operation_token='stitch-2d')
    plan = replace(plan, mode='2d', npt_rad_2d=32, npt_azim_2d=16)
    request = prepare_stitch_operation(source, ModuleOutputRequest(target,
        AnalysisArtifactKind.STITCH_2D, artifact_family='xu-energy'), plan, project_root=tmp_path)
    assert request.provenance['plan']['energy_selection']['energy_eV'] == pytest.approx(16000.)
    result = run_stitch_operation(request, workers=2)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
    payload = read_analysis_artifact(target)
    wavelength = energy_eV_to_wavelength_m(16000.) * 1e10
    assert payload.wavelength_A == pytest.approx(wavelength)
    from xrd_tools.io.analysis_stitch_auxiliary import native_stitch_companion
    # The 2-D payload exposes its map and profile's stored native axes; the shared
    # companion conversion supplies the alternate coordinate for either.
    for radial, native_unit in ((payload.axis('q'), payload.native_unit),
                               (payload.profile.axes[0][1], payload.profile.axis_units[0][1])):
        _other_unit, other = native_stitch_companion(radial, native_unit, payload.wavelength_A)
        q, tth = (radial, other) if native_unit == 'q_A^-1' else (other, radial)
        np.testing.assert_allclose(q, 4 * np.pi / wavelength * np.sin(np.deg2rad(tth.astype(float) / 2)),
                                  rtol=2 * np.finfo(q.dtype).eps)


def test_xu_energy_changes_q_without_changing_pose_corrections_or_default_identity(tmp_path):
    from tests.core.test_xu_stitch import _receipt
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.integrate.xu_stitch import resolve_xu_stitch_effective_geometry, XuPowderQProvider, _polarization_axes
    receipt = _receipt(tmp_path)
    acquisition = receipt.projection.value['acquisition']
    axes = _polarization_axes(receipt.projection.value)
    with xu_runtime_session() as session:
        base = resolve_xu_stitch_effective_geometry(receipt, session)
        same = resolve_xu_stitch_effective_geometry(receipt, session, metadata_energy_eV=17000.018)
        assert same.projection.to_provenance() == base.projection.to_provenance()
        shifted = resolve_xu_stitch_effective_geometry(receipt, session, metadata_energy_eV=16000.)
        assert shifted.projection.fingerprint != base.projection.fingerprint
        assert shifted.projection.asset_semantic_fingerprint == base.projection.asset_semantic_fingerprint
        np.testing.assert_array_equal(base.mask, shifted.mask)
        np.testing.assert_array_equal(base.solid_angle, shifted.solid_angle)
        comparisons = []
        for geometry in (base, shifted):
            provider = XuPowderQProvider(geometry, acquisition)
            lease = provider.frame(nu=16., del_=31.)
            comparisons.append((lease.q_magnitude(), lease.chi(),
                lease.polarization(.99, geometry.projection.runtime_xu_wavelength_A, *axes)))
            lease.release()
        q0, chi0, pol0 = comparisons[0]
        q1, chi1, pol1 = comparisons[1]
        np.testing.assert_allclose(q1, q0 * (16000. / 17000.018), rtol=2e-15)
        np.testing.assert_allclose(chi1, chi0, atol=1e-12)
        np.testing.assert_allclose(pol1, pol0, atol=2e-15)


@pytest.mark.parametrize('recorded,units', [(16000., None), (16000., 'eV'), (16., 'keV'), (16., None)])
def test_int_replay_reads_a_nexus_monochromator_energy_in_ev_or_kev(tmp_path, recorded, units):
    """CAL-HELD-BATCH-20261009 (maintainer: "It should be easy enough to figure out if the energy is in eV or keV for
    X-rays"): an Int run of a non-Bluesky container recording its monochromator energy as 16000 (eV) or 16 (keV),
    labelled or not, selects a 16 keV metadata energy instead of refusing or misreading it."""
    from xrd_tools.core.provenance import read_provenance
    intent = _fixture(tmp_path / 'project')
    with h5py.File(intent.source_spec.uri, 'r+') as handle:
        del handle.attrs['creator']  # a non-Bluesky acquisition container
        energy = handle.create_dataset('entry/instrument/monochromator/energy', data=recorded)
        if units is not None:
            energy.attrs['units'] = units
    path = _replay(intent, 'run')
    selection = read_provenance(path)['config']['energy_selection']
    assert (selection['source'], selection['energy_eV']) == ('metadata', pytest.approx(16000., rel=1e-12, abs=0))
    assert open_scan(path).metadata['wavelength_A'] == pytest.approx(energy_eV_to_wavelength_m(16000.) * 1e10)

