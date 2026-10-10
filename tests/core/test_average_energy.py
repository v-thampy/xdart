"""New Average operations share Run's acquisition-energy authority."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.core.energy import energy_eV_to_wavelength_m
from xrd_tools.core.provenance import read_provenance
from xrd_tools.integrate.calibration import load_detector_calibration
from xrd_tools.io import open_scan
from xrd_tools.reduction import AverageScanRecipe
from xrd_tools.reduction.average import AverageScanRunner
from xrd_tools.session.experiment_state import CalibrationState, FactStatus, PoniValues
from xrd_tools.session.readiness import native_int_reduction_plan
from tests.core.test_acquisition_energy import _record_nexus_energy, _replay
from tests.core.test_int_workflow_replay import _fixture


def _average(intent, target):
    admitted = load_detector_calibration(intent.poni_file)
    p = admitted.poni
    calibration = CalibrationState(
        PoniValues(p.dist, p.poni1, p.poni2, p.rot1, p.rot2, p.rot3, p.wavelength),
        p.detector, admitted.detector_config, status=FactStatus.PRESENT,
        parallax=admitted.parallax,
    )
    recipe = AverageScanRecipe(intent.source_spec, target,
        native_int_reduction_plan(intent.freeze()), calibration=calibration,
        source_base=intent.project_root)
    runner = AverageScanRunner(recipe)
    result = runner.start()
    assert runner.close() is result
    return result, runner.plan


@pytest.mark.parametrize('unit', ['q_A^-1', '2th_deg'])
def test_real_run_and_average_select_the_same_metadata_energy(tmp_path, unit):
    intent = _fixture(tmp_path / 'project')
    intent.processing_mode = 'Int 1D'
    intent.bai_1d_args['unit'] = unit
    _record_nexus_energy(intent, [16000., 16000.])
    run_path = _replay(intent, 'run')
    result, plan = _average(intent, tmp_path / 'average.nexus')
    assert result.disposition == 'COMMITTED', result
    run, average = open_scan(run_path), open_scan(result.target)
    np.testing.assert_array_equal(run.get_1d(0).q, average.get_1d(1).q)
    np.testing.assert_allclose(run.get_1d(0).intensity, average.get_1d(1).intensity,
                               rtol=2 * np.finfo(np.float32).eps)
    selection = read_provenance(result.target)['config']['energy_selection']
    assert selection == read_provenance(run_path)['config']['energy_selection']
    assert selection['wavelength_m'] == energy_eV_to_wavelength_m(16000.)
    assert plan.recipe.calibration.values.wavelength_m == 1e-10
    # Admission remains immutable; effective energy participates in science identity.
    fallback_root = tmp_path / 'fallback'
    fallback_intent = _fixture(fallback_root)
    fallback_intent.processing_mode = 'Int 1D'
    fallback_intent.bai_1d_args['unit'] = unit
    fallback, fallback_plan = _average(fallback_intent, tmp_path / 'fallback.nexus')
    assert fallback.disposition == 'COMMITTED', fallback
    assert plan.science_identity != fallback_plan.science_identity
    q = average.get_1d(1).q.copy()
    with h5py.File(intent.source_spec.uri, 'r+') as source:
        source['entry/data/energy'][...] = [17000., 17000.]
    np.testing.assert_array_equal(open_scan(result.target).get_1d(1).q, q)


@pytest.mark.parametrize('wavelength', [1e-10, 7.3812397e-11])
def test_average_without_metadata_retains_exact_admitted_wavelength(tmp_path, wavelength):
    intent = _fixture(tmp_path / 'project')
    poni = Path(intent.poni_file)
    poni.write_text(poni.read_text().replace('Wavelength: 1e-10',
                                           f'Wavelength: {wavelength!r}'))
    result, plan = _average(intent, tmp_path / 'average.nexus')
    assert result.disposition == 'COMMITTED', result
    selection = read_provenance(result.target)['config']['energy_selection']
    assert selection['source'] == 'calibration'
    assert selection['wavelength_m'] == wavelength
    assert plan.recipe.calibration.values.wavelength_m == wavelength


@pytest.mark.parametrize('source_kind', ['nexus', 'tiff'])
@pytest.mark.parametrize('second', [17000.])
def test_average_refuses_mixed_energy_before_replacing_output(tmp_path, second, source_kind):
    # PROCESSING-ENERGY-BATCH-20261010: an impossible energy (the former -1 rows) now counts as none; Average still
    # adds frames into one pattern, so a mixed series keeps refusing.
    intent = _fixture(tmp_path / 'project')
    if source_kind == 'nexus':
        _record_nexus_energy(intent, [16000., 16000.])
    else:
        from tests.core.test_p36_average import _series, _txt
        from xrd_tools.core.scan import SourceSpec
        source = _series(Path(intent.project_root),
                        (np.full((195, 487), 100, np.uint16),) * 2)
        intent.source_spec = SourceSpec(source.uri, source.kind,
            options={**source.options, 'metadata_format': 'txt'})
        for file in source.options['files']:
            _txt(file, motors=(('energy', 16000.),))
    target = tmp_path / 'average.nexus'
    first, _ = _average(intent, target)
    assert first.disposition == 'COMMITTED', first
    actual = Path(first.target)
    before = actual.read_bytes()
    if source_kind == 'nexus':
        with h5py.File(intent.source_spec.uri, 'r+') as source:
            source['entry/data/energy'][1] = second
    else:
        _txt(source.options['files'][1], motors=(('energy', second),))
    refused, _ = _average(intent, target)
    assert refused.disposition != 'COMMITTED'
    assert 'metadata energy' in str(refused.diagnostic), refused
    assert actual.read_bytes() == before


def test_average_honors_declared_nexus_wavelength(tmp_path):
    intent = _fixture(tmp_path / 'project')
    intent.processing_mode = 'Int 1D'
    with h5py.File(intent.source_spec.uri, 'r+') as source:
        source.create_dataset('entry/instrument/monochromator/wavelength',
                              data=energy_eV_to_wavelength_m(16000.) * 1e10)
    expected = open_scan(_replay(intent, 'run'))
    result, _ = _average(intent, tmp_path / 'average.nexus')
    assert result.disposition == 'COMMITTED', result
    np.testing.assert_array_equal(expected.get_1d(0).q,
                                  open_scan(result.target).get_1d(1).q)
    assert read_provenance(result.target)['config']['energy_selection']['source'] == 'metadata'
