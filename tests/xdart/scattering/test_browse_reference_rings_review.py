"""Saved-record boundary regressions found in the Browse rings terminal review."""

import numpy as np
import pytest

from tests.xdart.scattering.test_browse_reference_rings_run import (
    _copy, _fresh_output, _json, _labels, _replace_json, qapp,
)
from xrd_tools import geometry_figures as gf
from xrd_tools.session.readiness import ring_basis_for_record


@pytest.mark.parametrize('damage', ('json_null', 'empty', 'missing_wavelength', 'null_wavelength'))
def test_present_energy_record_never_recovers_an_unproven_wavelength(qapp, tmp_path, damage):
    output, _ = _fresh_output(qapp, tmp_path / 'run')
    labels = _labels(output)
    assert ring_basis_for_record(output, labels=labels).frame_snapshot(labels[0]) is not None

    def edit(entry):
        config = entry['reduction/config']
        selection = _json(config['energy_selection'])
        if damage == 'json_null':
            selection = None
        elif damage == 'empty':
            selection = {}
        elif damage == 'missing_wavelength':
            del selection['wavelength_m']
        else:
            selection['wavelength_m'] = None
        _replace_json(config, 'energy_selection', selection)

    damaged = _copy(output, tmp_path / 'damaged.nexus', edit)
    refused = ring_basis_for_record(damaged, labels=labels)
    assert refused.frame_snapshot(labels[0]) is None
    assert refused.frame_reason(labels[0]) == (
        gf.SAVED_NO_ENERGY_REASON if damage == 'null_wavelength' else gf.SAVED_ENERGY_MALFORMED_REASON)


@pytest.mark.parametrize('field', ('wavelength', 'energy'))
@pytest.mark.parametrize('length', (1, 2))
def test_legacy_ring_energy_refuses_numeric_arrays_instead_of_taking_the_first(qapp, tmp_path, field, length):
    output, _ = _fresh_output(qapp, tmp_path / 'run')
    labels = _labels(output)

    def edit(entry):
        del entry['reduction/config/energy_selection']
        mono = entry['instrument/monochromator']
        value = float(mono[field][()])
        del mono['wavelength']
        del mono['energy']
        mono[field] = np.asarray([value * (i + 1) for i in range(length)])

    damaged = _copy(output, tmp_path / 'array.nexus', edit)
    refused = ring_basis_for_record(damaged, labels=labels)
    assert refused.frame_snapshot(labels[0]) is None
    assert refused.frame_reason(labels[0]) == gf.SAVED_ENERGY_MALFORMED_REASON
