"""BROWSE-REFERENCE-RINGS-20261013 frozen headless oracle (Claude, October 13, after Codex's pre-writer review).

Rows H3, H5, H6 and H9 of the kickoff (handoffs/browse_reference_rings_2026-10-13.md). The writer-backed rows (H1,
H2, H4, H7, H8) need real Runs and real saved outputs and live in
tests/xdart/scattering/test_browse_reference_rings_run.py. New APIs are imported inside each row, so at the parent a
missing API fails only its own row (an interface failure, recorded as such) while the preservation rows still run.
"""
import json
from dataclasses import replace

import numpy as np
import pytest

from tests.core.test_predicted_ring_overlays import bound_session, calibration
from xrd_tools.core.energy import energy_eV_to_wavelength_m
from xrd_tools.integrate.calibration import detector_calibration_to_integrator

ENERGIES = (16., 17., 18., 19.)


def _scan_wavelengths(scan):
    selection = scan.extra['energy_selection']
    return [energy_eV_to_wavelength_m(energy) for energy in selection['frame_energies_eV']]


# H3 — one per-label conversion, including Standard 2θ energy scans

def test_h3a_a_two_theta_energy_scan_publishes_each_frames_own_wavelength():
    """Preservation (passes at the parent): in 2θ the reducer keeps one integrator, yet each frame's published ring
    snapshot carries that frame's own wavelength. Sabotage for the freeze: publish through the reducer's policy."""
    accepted, scan, views, store = bound_session(energies=ENERGIES, unit='2th_deg')
    published = [view.geometry.poni['wavelength'] for view in views]
    assert published == _scan_wavelengths(scan)
    assert len(set(published)) == len(ENERGIES)


def test_h3b_the_saved_record_and_the_run_share_one_per_label_conversion():
    from xrd_tools.session.readiness import frame_ring_wavelengths
    for unit in ('2th_deg', 'q_A^-1'):
        accepted, scan, views, store = bound_session(energies=ENERGIES, unit=unit)
        converted = frame_ring_wavelengths(scan.extra['energy_selection'])
        assert converted == {frame.index: view.geometry.poni['wavelength'] for frame, view in zip(scan.frames, views)}


def test_h3c_scan_session_publishes_through_the_shared_conversion():
    """The fact the guard pins: ScanSession's label→wavelength map is the shared helper, not a second decoder."""
    import inspect
    from xrd_tools.session import scan_session
    source = inspect.getsource(scan_session.ScanSession.__init__)
    assert 'frame_ring_wavelengths(' in source and 'energy_eV_to_wavelength_m' not in source


# Saved-record fixtures for the builder rows (decoded mappings, as the bounded reader returns them)

def _saved_configuration(tmp_path, *, mode='Int 2D', one=None, two=None, gi=None):
    from xrd_tools.session.run_configuration import RunIntent
    from xrd_tools.sources import image_series_spec
    intent = RunIntent(source_spec=image_series_spec(tmp_path / 'scan_0001.tif', metadata_format=None),
                       poni_file=str(tmp_path / 'cal.poni'), project_root=str(tmp_path),
                       save_path=str(tmp_path / 'out.nexus'), output_mode='Overwrite', processing_mode=mode,
                       live_mode=False, max_cores=1,
                       bai_1d_args=dict(one or {'npt': 8, 'method': 'numpy'}),
                       bai_2d_args=dict(two or {'npt_rad': 8, 'npt_azim': 4, 'method': 'numpy'}))
    if gi is not None:
        intent.gi = gi
    frozen = intent.freeze()
    saved = json.loads(json.dumps(frozen.as_provenance()))
    poni = calibration().poni
    saved['scientific_signature'] = {'accepted_scientific_assets': {
        'poni_values': {'detector': poni.detector, 'dist': poni.dist, 'poni1': poni.poni1, 'poni2': poni.poni2,
                        'rot1': poni.rot1, 'rot2': poni.rot2, 'rot3': poni.rot3, 'wavelength': poni.wavelength},
        'poni_detector_config_json': json.dumps({'orientation': 3})}}
    return frozen, saved


def _selection(labels, energies_keV, representative_keV=None):
    representative = energies_keV[0] if representative_keV is None else representative_keV
    return {'calibration_energy_eV': 12398.41984, 'disagreement': True, 'energy_eV': representative * 1000,
            'fitted_energy_offset_eV': 0.0, 'frame_energies_eV': [e * 1000 for e in energies_keV],
            'frame_labels': list(labels), 'source': 'metadata', 'source_energy_eV': representative * 1000,
            'wavelength_m': energy_eV_to_wavelength_m(representative * 1000)}


# H5 — the energy map, validated

def test_h5a_a_valid_map_gives_each_label_its_own_wavelength_even_beyond_64_kib(tmp_path):
    from xrd_tools.session.readiness import ring_basis_from_saved
    frozen, saved = _saved_configuration(tmp_path)
    labels = tuple(range(1, 5001))
    energies = [16 + (label % 7) / 10 for label in labels]
    selection = _selection(labels, energies)
    assert len(json.dumps(selection).encode()) > 64 * 1024
    basis = ring_basis_from_saved(saved, selection, labels=labels, epochs=(labels,))
    for label, energy in ((1, energies[0]), (2500, energies[2499]), (5000, energies[4999])):
        snapshot = basis.frame_snapshot(label)
        assert snapshot is not None and basis.frame_reason(label) == ''
        assert snapshot['wavelength'] == energy_eV_to_wavelength_m(energy * 1000)


@pytest.mark.parametrize('damage', ('unequal', 'duplicate', 'negative_label', 'text_label', 'nan', 'zero', 'inf',
                                    'no_wavelength_type'))
def test_h5b_a_malformed_map_refuses_rather_than_falling_back(tmp_path, damage):
    from xrd_tools import geometry_figures as gf
    from xrd_tools.session.readiness import ring_basis_from_saved
    frozen, saved = _saved_configuration(tmp_path)
    selection = _selection((1, 2, 3), (16, 17, 18))
    if damage == 'unequal':
        selection['frame_energies_eV'] = selection['frame_energies_eV'][:2]
    elif damage == 'duplicate':
        selection['frame_labels'] = [1, 1, 3]
    elif damage == 'negative_label':
        selection['frame_labels'] = [1, -2, 3]
    elif damage == 'text_label':
        selection['frame_labels'] = [1, '2', 3]
    elif damage == 'nan':
        selection['frame_energies_eV'][1] = float('nan')
    elif damage == 'zero':
        selection['frame_energies_eV'][1] = 0.0
    elif damage == 'inf':
        selection['frame_energies_eV'][1] = float('inf')
    else:
        selection['wavelength_m'] = 'fast'
    result = ring_basis_from_saved(saved, selection, labels=(1, 2, 3), monochromator_wavelength_m=1e-10,
                                   epochs=((1, 2, 3),))
    for label in (1, 2, 3):
        assert result.frame_snapshot(label) is None
        assert result.frame_reason(label) == gf.SAVED_ENERGY_MALFORMED_REASON


def test_h5c_only_an_absent_map_takes_the_monochromator_fallback(tmp_path):
    from xrd_tools import geometry_figures as gf
    from xrd_tools.session.readiness import ring_basis_from_saved
    frozen, saved = _saved_configuration(tmp_path)
    basis = ring_basis_from_saved(saved, None, labels=(1, 2), monochromator_wavelength_m=0.8e-10)
    assert [basis.frame_snapshot(label)['wavelength'] for label in (1, 2)] == [0.8e-10, 0.8e-10]
    refused = ring_basis_from_saved(saved, None, labels=(1, 2))
    assert refused.frame_snapshot(1) is None and refused.frame_reason(1) == gf.SAVED_NO_ENERGY_REASON
    unreadable = ring_basis_from_saved(saved, None, labels=(1, 2), monochromator_wavelength_m=0.8e-10,
                                       unreadable=('energy_selection',))
    assert unreadable.frame_snapshot(1) is None
    assert unreadable.frame_reason(1) == gf.SAVED_ENERGY_MALFORMED_REASON


def test_h5d_rings_exist_only_for_labels_the_output_holds(tmp_path):
    from xrd_tools import geometry_figures as gf
    from xrd_tools.session.readiness import ring_basis_from_saved
    frozen, saved = _saved_configuration(tmp_path)
    basis = ring_basis_from_saved(saved, _selection((1, 2, 3), (16, 17, 18)), labels=(1, 2), epochs=((1, 2),))
    assert basis.frame_snapshot(2) is not None
    assert basis.frame_snapshot(3) is None and basis.frame_reason(3) == gf.SAVED_LABEL_ABSENT_REASON


# H7b — D5(a) in the builder: an energy scan's earlier Append epochs refuse; one-energy records do not

def test_h7b_only_the_last_append_epoch_of_an_energy_scan_draws(tmp_path):
    from xrd_tools import geometry_figures as gf
    from xrd_tools.session.readiness import ring_basis_from_saved
    frozen, saved = _saved_configuration(tmp_path)
    scan = _selection((1, 2, 3), (16, 17, 16.5))
    basis = ring_basis_from_saved(saved, scan, labels=(1, 2, 3), epochs=((1, 2), (3,)))
    assert [basis.frame_reason(label) for label in (1, 2)] == [gf.SAVED_HISTORY_REASON] * 2
    assert basis.frame_snapshot(1) is None and basis.frame_snapshot(2) is None
    assert basis.frame_snapshot(3)['wavelength'] == energy_eV_to_wavelength_m(16500.0)
    for epochs, unreadable in ((None, ()), (((1, 2), (3,)), ('append_lineage',))):
        refused = ring_basis_from_saved(saved, scan, labels=(1, 2, 3), epochs=epochs, unreadable=unreadable)
        assert [refused.frame_reason(label) for label in (1, 2, 3)] == [gf.SAVED_HISTORY_UNREADABLE_REASON] * 3
    one_energy = {key: value for key, value in scan.items() if key not in ('frame_labels', 'frame_energies_eV')}
    live = ring_basis_from_saved(saved, one_energy, labels=(1, 2, 3), epochs=((1,), (2, 3)))
    assert [live.frame_snapshot(label)['wavelength'] for label in (1, 2, 3)] == [one_energy['wavelength_m']] * 3


# H6 — the grid in Browse: q contours, χ withheld

def _run_snapshot(**plan_args):
    from xrd_tools.session.readiness import build_native_int_reduction_plan_from_args, effective_ring_geometry
    accepted = calibration()
    plan = build_native_int_reduction_plan_from_args({'npt': 8}, {'npt_rad': 8, 'npt_azim': 4}, **plan_args)
    return effective_ring_geometry(accepted, integrator=detector_calibration_to_integrator(accepted), plan=plan)


def _curves(figure):
    return [np.asarray(curve.xy) for curve in figure.curves]


def test_h6_a_saved_snapshot_draws_q_contours_and_withholds_chi():
    from xrd_tools import geometry_figures as gf
    snapshot, shape = _run_snapshot(), (195, 487)
    run_q = gf.predict_detector_grid(snapshot, shape=shape, choice='q', mode='2d')
    saved_q = gf.predict_detector_grid(snapshot, shape=shape, choice='q', mode='2d', saved=True)
    saved_chi = gf.predict_detector_grid(snapshot, shape=shape, choice='χ', mode='2d', saved=True)
    saved_both = gf.predict_detector_grid(snapshot, shape=shape, choice='q + χ', mode='2d', saved=True)
    assert run_q.curves and saved_chi.curves == ()
    for figure in (saved_q, saved_both):
        assert len(figure.curves) == len(run_q.curves)
        for actual, expected in zip(_curves(figure), _curves(run_q)):
            np.testing.assert_array_equal(actual, expected)
        assert all('Run' not in curve.tooltip for curve in figure.curves)
    assert gf.detector_grid_note('q', saved=True) == ''
    assert gf.detector_grid_note('χ', saved=True) == gf.detector_grid_note('q + χ', saved=True) == \
        gf.GRID_CHI_WITHHELD_NOTE
    assert gf.detector_grid_note('q + χ', saved=False) == ''


def test_h6b_gi_keeps_its_grid_refusal_for_saved_snapshots():
    from xrd_tools import geometry_figures as gf
    snapshot = _run_snapshot(gi_enabled=True, gi_incident_angle=0.2)
    reason = gf.detector_grid_unavailable(snapshot, (195, 487))
    assert reason and 'GI' in reason
    with pytest.raises(ValueError):
        gf.predict_detector_grid(snapshot, shape=(195, 487), choice='q', mode='2d', saved=True)


# H9 — one plan route for the frozen configuration and the saved mapping

@pytest.mark.parametrize('mode', ('Int 1D', 'Int 2D'))
@pytest.mark.parametrize('gi', (False, True))
def test_h9_the_saved_mapping_rebuilds_the_runs_plan_facts(tmp_path, mode, gi):
    from xrd_tools.session.readiness import (
        build_native_int_reduction_plan_from_args, effective_ring_geometry, execution_plan_shape,
        native_int_reduction_plan)
    from xrd_tools.session.run_configuration import FrozenGIConfiguration, GIIntent
    frozen, saved = _saved_configuration(
        tmp_path, mode=mode, one={'npt': 8, 'method': 'numpy', 'chi_offset': 17.0},
        two={'npt_rad': 8, 'npt_azim': 4, 'method': 'numpy', 'chi_offset': 43.0},
        gi=GIIntent(enabled=True, incidence_motor='Manual', th_val=0.2) if gi else None)
    accepted = calibration()
    integrator = detector_calibration_to_integrator(accepted)
    expected = effective_ring_geometry(accepted, integrator=integrator, plan=native_int_reduction_plan(frozen))
    one, two, values = execution_plan_shape(saved['bai_1d_args'], saved['bai_2d_args'],
                                            FrozenGIConfiguration.from_dict(saved['gi']), saved['processing_mode'])
    rebuilt = effective_ring_geometry(accepted, integrator=integrator,
                                      plan=build_native_int_reduction_plan_from_args(one, two, **values))
    assert dict(rebuilt) == dict(expected)
    if not gi:
        assert expected['azimuth_offset_1d'] == 17.0
        assert expected['azimuth_offset_2d'] == (43.0 if mode == 'Int 2D' else 0.0)


def test_h9b_the_runs_plan_is_unchanged_by_the_split(tmp_path):
    """Preservation (passes at the parent): the frozen configuration's plan keeps the processing-mode rule. Sabotage
    for the freeze: drop the rule from the shared shape function."""
    from xrd_tools.session.readiness import native_int_reduction_plan
    one = {'npt': 8, 'method': 'numpy', 'chi_offset': 17.0}
    two = {'npt_rad': 8, 'npt_azim': 4, 'method': 'numpy', 'chi_offset': 43.0}
    frozen_1d, _ = _saved_configuration(tmp_path, mode='Int 1D', one=one, two=two)
    frozen_2d, _ = _saved_configuration(tmp_path, mode='Int 2D', one=one, two=two)
    plan_1d, plan_2d = native_int_reduction_plan(frozen_1d), native_int_reduction_plan(frozen_2d)
    assert plan_1d.integration_2d is None and plan_1d.integration_1d.azimuth_offset == 17.0
    assert plan_2d.integration_2d.azimuth_offset == 43.0 and plan_2d.integration_1d.azimuth_offset == 17.0
