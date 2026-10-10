"""B1: raw energy inference must not reinterpret a source's canonical eV."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.core.test_acquisition_energy import _replay
from tests.core.test_average_energy import _average
from tests.core.test_int_workflow_replay import _fixture
from xrd_tools.core.energy import energy_eV_to_wavelength_m
from xrd_tools.core.provenance import read_provenance
from xrd_tools.integrate.calibration import load_detector_calibration
from xrd_tools.session.readiness import bind_native_int_source_energy
from xrd_tools.sources import open_source


def _generic(root, recorded=0.1, *, wavelength=False):
    intent = _fixture(root)
    intent.processing_mode = 'Int 1D'
    with h5py.File(intent.source_spec.uri, 'r+') as handle:
        del handle.attrs['creator']
        mono = handle.create_group('entry/instrument/monochromator')
        if wavelength:
            dataset = mono.create_dataset('wavelength', data=energy_eV_to_wavelength_m(100.0))
            dataset.attrs['units'] = 'm'
        else:
            dataset = mono.create_dataset('energy', data=recorded)
            dataset.attrs['units'] = 'keV'
        for name, values in (('del', [0., 1.]), ('nu', [0., 0.])):
            group = handle.create_group(f'entry/instrument/{name}')
            group.attrs['NX_class'] = 'NXpositioner'
            group.create_dataset('value', data=values).attrs['units'] = 'deg'
    return intent


@pytest.mark.parametrize('recorded,expected', [(0.1, 100.), (0.15, 150.), (0.25, 250.),
                                              (16., 16000.), (150., 150000.), (16000., 16000.)])
def test_generic_source_energy_is_inferred_once(tmp_path, recorded, expected):
    intent = _generic(tmp_path / 'project', recorded)
    source = open_source(intent.source_spec)
    try:
        scan = source.to_scan()
        assert scan.frames[0].metadata['energy'] == pytest.approx(expected)
        bind_native_int_source_energy(scan, load_detector_calibration(intent.poni_file))
        assert scan.extra['energy_selection']['energy_eV'] == pytest.approx(expected)
        assert scan.integrator.wavelength == pytest.approx(energy_eV_to_wavelength_m(expected))
    finally:
        source.close()


@pytest.mark.parametrize('energy', [100., 150., 16000.])
def test_explicit_binder_ev_argument_keeps_its_unit(tmp_path, energy):
    intent = _fixture(tmp_path / 'project')
    source = open_source(intent.source_spec)
    try:
        scan = source.to_scan()
        bind_native_int_source_energy(scan, load_detector_calibration(intent.poni_file),
                                      metadata_energy_eV=(energy,))
        assert scan.extra['energy_selection']['energy_eV'] == pytest.approx(energy)
    finally:
        source.close()


def test_wavelength_derived_provider_energy_keeps_ev(tmp_path):
    intent = _generic(tmp_path / 'project', wavelength=True)
    source = open_source(intent.source_spec)
    try:
        scan = source.to_scan()
        assert scan.frames[0].metadata['energy'] == pytest.approx(100.)
        bind_native_int_source_energy(scan, load_detector_calibration(intent.poni_file))
        assert scan.extra['energy_selection']['energy_eV'] == pytest.approx(100.)
    finally:
        source.close()


def test_generic_soft_energy_replay_matches_the_explicit_wavelength_reference(tmp_path):
    from xrd_tools.io import open_scan
    intent = _generic(tmp_path / 'project')
    actual = _replay(intent, 'run')
    assert read_provenance(actual)['config']['energy_selection']['energy_eV'] == pytest.approx(100.)
    reference = _fixture(tmp_path / 'reference')
    reference.processing_mode = 'Int 1D'
    with h5py.File(reference.source_spec.uri, 'r+') as handle:
        handle.create_dataset('entry/instrument/monochromator/wavelength',
                              data=energy_eV_to_wavelength_m(100.) * 1e10)
    expected = _replay(reference, 'reference')
    first, second = open_scan(actual).get_1d(0), open_scan(expected).get_1d(0)
    np.testing.assert_array_equal(first.q, second.q)
    np.testing.assert_array_equal(first.intensity, second.intensity)


def test_generic_soft_energy_average_keeps_ev(tmp_path):
    intent = _generic(tmp_path / 'project')
    result, _ = _average(intent, tmp_path / 'average.nexus')
    assert result.disposition == 'COMMITTED', result
    assert read_provenance(result.target)['config']['energy_selection']['energy_eV'] == pytest.approx(100.)


def test_generic_calibration_frame_choices_keep_ev(tmp_path):
    from xrd_tools.integrate.stitch_calibration import source_frame_choices
    intent = _generic(tmp_path / 'project')
    rows = source_frame_choices(intent.source_spec, ('del', 'nu'))
    assert [row[2] for row in rows] == pytest.approx([100., 100.])


def test_generic_stitch_table_does_not_infer_canonical_energy_again(tmp_path):
    from xrd_tools.analysis.module_transaction import ModuleKind, ModuleSourceReceipt
    from xrd_tools.analysis.scan_operations import MetadataTablePlan, run_metadata_table
    from xrd_tools.analysis.stitch_native import source_energy_checks
    intent = _generic(tmp_path / 'project')
    table = run_metadata_table(MetadataTablePlan(intent.source_spec))
    source = ModuleSourceReceipt.from_metadata_table(table, kind=ModuleKind.STITCH,
                                                    selected_labels=table.labels)
    checks = source_energy_checks(source, energy_eV_to_wavelength_m(100.) * 1e10, tables=(table,))
    assert checks[0]['range_eV'] == pytest.approx([100., 100.])


def test_composite_frames_keep_each_members_energy_unit(tmp_path):
    from xrd_tools.sources.composite import CompositeFrameSource
    intent = _generic(tmp_path / 'generic')
    member = open_source(intent.source_spec)
    source = CompositeFrameSource([member])
    try:
        scan = source.to_scan()
        bind_native_int_source_energy(scan, load_detector_calibration(intent.poni_file))
        assert scan.extra['energy_selection']['energy_eV'] == pytest.approx(100.)
    finally:
        member.close()
