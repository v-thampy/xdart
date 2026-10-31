"""BROWSE-REFERENCE-RINGS-20261013 frozen GUI oracle (Claude, October 13, after Codex's pre-writer review).

Rows H1, H1b, H2, H2b, H4, H7, H8 and G1-G11 of the kickoff (handoffs/browse_reference_rings_2026-10-13.md): real
Runs through the production executor and page, the Run's own terminal handoff into Browse (never repaired by hand
before a Browse assertion), real reopen, and real saved outputs - fresh ones, mutated copies of fresh ones, and the
maintainer's own under XDART_TEST_DATA. New APIs are imported inside each row, so a missing API fails only its row.
The parent results are recorded in the kickoff's freeze record.
"""
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import fabio
import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore

from tests.xdart.scattering._output_slots import written
from tests.xdart.scattering.test_detector_coordinate_overlays import choose_grid, control
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _intent, _run_to_terminal, write_poni
from tests.xdart.scattering.test_predicted_ring_overlays import (choice, choose, close_phase_window, qapp,  # noqa: F401
                                                                 rings, select, settle)
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xrd_tools.core.energy import energy_eV_to_wavelength_m
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources import image_series_spec
from xrd_tools.sources.selection import DirectorySourceSpec

SHAPE = (195, 487)
ORIENTED = ('poni_version: 2.1\nDetector: Pilatus100k\nDetector_config: {"orientation": 2}\n'
            'Distance: 0.03\nPoni1: 0.05\nPoni2: 0.06\nRot1: 0.05\nRot2: -0.03\nRot3: 0.2\nWavelength: 1.0e-10\n')
DATA = os.environ.get('XDART_TEST_DATA')
SAVED = None if not DATA else Path(DATA) / 'xdart_processed_data'
needs_data = pytest.mark.skipif(not DATA or not (SAVED / 'eiger_S069Ta_redo_eta2p0_1_scan001_int1d.nexus').is_file(),
                                reason='the maintainer\'s saved outputs need XDART_TEST_DATA')


# ---------------------------------------------------------------------------------------------------- fixtures

def _frame(directory, stem, index, energy_keV):
    """One frame: its sidecar, then its image renamed into place whole (as live acquisition writes it)."""
    target = directory / f'{stem}_{index:04d}.tif'
    target.with_suffix('.txt').write_text(f'# Counters\nI0 = 1\n# Motors\nenergy = {energy_keV}\n'
                                          'User: energy, time: Mon Jan 15 10:30:00 2024  # Temp\n')
    rows, columns = np.mgrid[:SHAPE[0], :SHAPE[1]]
    partial = directory / f'{stem}_{index:04d}.part'
    fabio.tifimage.TifImage(data=(100 + (rows + 2 * columns) % 97).astype(np.uint16)).write(str(partial))
    partial.replace(target)
    return target


def _poni(root, text=None):
    poni = root / 'cal.poni'
    write_poni(poni)
    poni.write_text(text or poni.read_text().replace('Distance: 0.1234', 'Distance: 0.03'))
    return poni


def _series(root, energies, *, mode='Int 2D', unit=None, one=None, two=None, gi=None, poni_text=None,
            output_mode='Overwrite', heavy_window=None):
    """A finite image-series Run's intent over frames recorded at ``energies`` (keV)."""
    root.mkdir(parents=True, exist_ok=True)
    for index, energy in enumerate(energies, 1):
        if not (root / f'escan_{index:04d}.tif').exists():
            _frame(root, 'escan', index, energy)
    raw = root / 'escan_0001.tif'
    poni = root / 'cal.poni' if (root / 'cal.poni').exists() else _poni(root, poni_text)
    intent = _intent(raw, root / 'run.nexus', poni, output_mode=output_mode, processing_mode=mode)
    intent.source_spec = image_series_spec(raw, metadata_format='txt')
    for key, value in (one or {}).items():
        intent.bai_1d_args[key] = value
    for key, value in (two or {}).items():
        intent.bai_2d_args[key] = value
    if unit is not None:
        intent.bai_1d_args['unit'] = unit
        intent.bai_2d_args['unit'] = unit
    if gi is not None:
        intent.gi = gi
    if heavy_window is not None:
        intent.run_options['heavy_window'] = heavy_window
    return intent


def _directory(root, *, live, output_mode='Overwrite', unit='2th_deg', mode='Int 1D'):
    (root / 'raw').mkdir(parents=True, exist_ok=True)
    poni = root / 'cal.poni' if (root / 'cal.poni').exists() else _poni(root)
    return RunIntent(source_spec=DirectorySourceSpec(root / 'raw', suffixes=('.tif',), metadata_format='txt'),
                     poni_file=str(poni), project_root=str(root / 'raw'), save_path=str(root / 'processed'),
                     output_mode=output_mode, processing_mode=mode, live_mode=live, max_cores=1,
                     bai_1d_args={'npt': 8, 'method': 'numpy', 'unit': unit},
                     bai_2d_args={'npt_rad': 8, 'npt_azim': 4, 'method': 'numpy', 'unit': unit})


@contextmanager
def _page(qapp, intent):
    executor = StandardRunExecutor(join_timeout=2.)
    page = ScatteringWorkspace(intents=RunIntentStore(intent), lifecycle=ScatteringCoordinator(),
                               sources=FilesystemSourceAdapter(), executor=executor)
    observed, drain = [], executor.drain_events
    executor.drain_events = lambda: (lambda events: (observed.extend(events), events)[1])(drain())
    try:
        page.show()
        yield page, page.findChild(ScatteringWorkspaceShell).scientific, observed
    finally:
        page.close()
        if page.isVisible():  # its cleanup still pending (a given-up Run's quarantine): off screen for later rows
            page.hide()
        page.deleteLater()
        qapp.processEvents()


def _pump(qapp, predicate, seconds=60.):
    until = time.monotonic() + seconds
    while time.monotonic() < until and not predicate():
        qapp.processEvents()
        time.sleep(.01)
    return predicate()


def _terminal(qapp, page, observed, kind=StandardEventKind.FINISHED):
    assert _pump(qapp, lambda: any(e.kind in _TERMINAL for e in observed)), 'the Run never reached a terminal'
    terminal = [e for e in observed if e.kind in _TERMINAL][-1]
    assert terminal.kind is kind, (page._notice_text, terminal)
    return terminal


def _run(qapp, page, observed):
    page.findChild(ScatteringWorkspaceShell).run_controls.startButton.click()
    return _terminal(qapp, page, observed)


def _browsing(qapp, page):
    """Wait for the real terminal or reopen handoff into Browse, and return its context."""
    controller = page._context_controller
    # A reopen that waits for the shown scan's cleanup leaves that scan loaded meanwhile.
    assert _pump(qapp, lambda: not controller.browse_pending and page._deferred_browse_reopen is None
                 and controller.browse_context is not None
                 and controller.browse_context.loaded), 'Browse never adopted the output'
    page._refresh_shell()
    qapp.processEvents()
    assert not page._last_scientific_projection.acquisition_selected, 'the handoff left the Run selected'
    return controller.browse_context


def _reopen(qapp, page, path):
    page._select_scan(str(path), reopen=True)
    return _browsing(qapp, page)


def _published(qapp, page, terminal):
    """The snapshot the Run's session published for each frame (read after the Browse assertions)."""
    controller = page._context_controller
    controller.adopt_acquisition(terminal.run_identity)
    controller.select_acquisition()
    page._refresh_shell()
    qapp.processEvents()
    keys = controller.frame_keys
    store = controller.acquisition_context.publication_store.artifacts[keys[0].artifact].publications
    return {key.local_frame_label: dict(store.get(key.local_frame_label).view.geometry.poni) for key in keys}


def _labels(path):
    with h5py.File(path, 'r', locking=False) as handle:
        node = handle.get('entry/integrated_2d/frame_index') or handle.get('entry/integrated_1d/frame_index')
        return tuple(int(value) for value in node[()])


def _show(qapp, page, view, key):
    """Select ``key`` and wait for Int 2D's raw pane (Browse has no Int 1D raw display; see G1)."""
    def image_ready():
        state = page._last_scientific_projection
        heavy = None if state is None else state.heavy
        return (
            heavy is not None and heavy.frame is key and heavy.raw is not None
            and view.raw.render_matches(
                heavy.raw,
                detector_shape=heavy.detector_shape,
                color_map=view.color_map.currentText(),
                log_scale=state.log_scale,
            )
        )

    select(qapp, page, key)
    # A cached thumbnail and an idle phase worker can precede this image's paint.
    settle(qapp, view, image_ready)


def _curves(target, name='LaB6'):
    return tuple(np.column_stack(item.getData()) for item in rings(target, name))


def _same(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b))


def _native_points(target, native_shape):
    """Drawn ring points mapped back to native pixel coordinates through the pane's own image transform."""
    image = target.image.image
    columns_image, rows_image = image.shape[0], image.shape[1]
    rows, columns = native_shape
    points = []
    for item in rings(target):
        data = np.column_stack(item.getData())
        for x, y in data[np.isfinite(data).all(axis=1)]:
            local = target.image.mapFromParent(QtCore.QPointF(float(x), float(y)))
            points.append(((local.x() - .5) * (columns - 1) / (columns_image - 1),
                           (rows_image - .5 - local.y()) * (rows - 1) / (rows_image - 1)))
    return np.asarray(points)


def _near(points, curves, tolerance=2.5):
    """Every drawn point lies on one of the independently predicted native curves."""
    expected = np.concatenate([np.asarray(curve.xy) for curve in curves])
    distances = np.sqrt(((points[:, None, :] - expected[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return float(distances.max())


def _copy(source, target, edit):
    """A copy of a real saved output with one bounded edit."""
    shutil.copy2(source, target)
    with h5py.File(target, 'r+', locking=False) as handle:
        edit(handle['entry'])
    return target


def _json(node):
    value = node[()]
    return json.loads(value.decode() if isinstance(value, bytes) else value)


def _replace_json(group, name, value):
    if name in group:
        del group[name]
    group[name] = value if isinstance(value, str) else json.dumps(value, separators=(',', ':'))


def _fresh_output(qapp, root, energies=(16, 17), **series):
    """One finite Run's saved output and the snapshots its session published."""
    with _page(qapp, _series(root, energies, **series)) as (page, view, observed):
        terminal = _run(qapp, page, observed)
        _browsing(qapp, page)
        published = _published(qapp, page, terminal)
    return written(root / 'run.nexus', series.get('mode', 'Int 2D')), published


# ---------------------------------------------------------------------------------------------------- H rows

@pytest.mark.parametrize('case', ('q_scan', '2th_scan', 'offsets_1d', 'offsets_2d', 'gi', 'oriented'))
def test_h1_each_saved_output_rebuilds_its_runs_published_snapshots(qapp, tmp_path, case):
    from xrd_tools.session.readiness import ring_basis_for_record
    offsets = dict(one={'chi_offset': 17.0}, two={'chi_offset': 43.0})
    series = {'q_scan': {}, '2th_scan': dict(mode='Int 1D', unit='2th_deg'),
              'offsets_1d': dict(mode='Int 1D', **offsets), 'offsets_2d': dict(mode='Int 2D', **offsets),
              'gi': dict(gi=GIIntent(enabled=True, incidence_motor='Manual', th_val=0.2)),
              'oriented': dict(poni_text=ORIENTED)}[case]
    output, published = _fresh_output(qapp, tmp_path, **series)
    basis = ring_basis_for_record(output, labels=_labels(output))
    assert set(published) == set(_labels(output))
    for label, snapshot in published.items():
        assert basis.frame_reason(label) == '', basis.frame_reason(label)
        assert dict(basis.frame_snapshot(label)) == snapshot
    assert basis.shape == SHAPE


@needs_data
def test_h1b_a_real_binned_output_yields_its_binned_native_shape():
    from xrd_tools.session.readiness import ring_basis_for_record
    output = SAVED / 'Combi4_Angledependence_samz_4p9_03271005_int2d.nexus'
    labels = _labels(output)
    basis = ring_basis_for_record(output, labels=labels)
    assert basis.shape == (3072, 3072)
    snapshot = basis.frame_snapshot(labels[0])
    with h5py.File(output, 'r', locking=False) as handle:
        recorded = float(handle['entry/instrument/monochromator/wavelength'][()]) * 1e-10
        assets = _json(handle['entry/reduction/config/run_configuration'])['scientific_signature'][
            'accepted_scientific_assets']
    assert snapshot['wavelength'] == pytest.approx(recorded, rel=1e-12, abs=0)
    assert json.loads(snapshot['detector_config_json']) == json.loads(assets['poni_detector_config_json'])


@needs_data
def test_h2_the_maintainers_outputs_refuse_with_their_own_reasons_beside_an_eligible_one():
    from xrd_tools import geometry_figures as gf
    from xrd_tools.session.readiness import ring_basis_for_record
    eligible = SAVED / 'eiger_S069Ta_redo_eta2p0_1_scan001_int1d.nexus'
    labels = _labels(eligible)
    basis = ring_basis_for_record(eligible, labels=labels)
    assert basis.frame_snapshot(labels[0]) is not None and basis.shape == (2167, 2070)
    # The older, slot-less output (schema 2, no detector configuration) is saved under test_data/eiger/.
    for path, reason in ((SAVED / 'eiger_S069Ta_redo_eta2p0_1_scan001_master_average.nexus', gf.SAVED_AVERAGE_REASON),
                         (SAVED / 'eiger_S069Ta_redo_eta2p0_1_scan001_reintegrate1d.nexus',
                          gf.SAVED_REINTEGRATE_REASON),
                         (Path(DATA) / 'eiger' / 'xdart_processed_data' / 'eiger_S069Ta_redo_eta2p0_1_scan001.nexus',
                          gf.SAVED_NO_DETECTOR_CONFIG_REASON)):
        refused = ring_basis_for_record(path, labels=_labels(path))
        first = _labels(path)[0]
        assert refused.frame_snapshot(first) is None and refused.frame_reason(first) == reason, (path.name, reason)


def test_h2b_synthetic_refusals_beside_their_eligible_original(qapp, tmp_path):
    from xrd_tools import geometry_figures as gf
    from xrd_tools.session.readiness import ring_basis_for_record
    output, published = _fresh_output(qapp, tmp_path / 'run')
    labels = _labels(output)
    assert ring_basis_for_record(output, labels=labels).frame_snapshot(labels[0]) is not None

    def assets(edit):
        def apply(entry):
            config = entry['reduction/config']
            run = _json(config['run_configuration'])
            edit(run['scientific_signature']['accepted_scientific_assets'], run)
            _replace_json(config, 'run_configuration', run)
        return apply

    cases = {
        'no_signature': (lambda entry: _replace_json(entry['reduction/config'], 'run_configuration', {
            key: value for key, value in _json(entry['reduction/config/run_configuration']).items()
            if key != 'scientific_signature'}), gf.SAVED_NO_CALIBRATION_REASON),
        'not_json': (lambda entry: _replace_json(entry['reduction/config'], 'run_configuration', '{not json'),
                     gf.SAVED_RECORD_MALFORMED_REASON),
        'no_config': (assets(lambda a, run: a.pop('poni_detector_config_json')), gf.SAVED_NO_DETECTOR_CONFIG_REASON),
        'non_flat': (assets(lambda a, run: a['poni_values'].update(detector='Rapid')),
                     'Predicted rings require a uniform, flat, undistorted detector.'),
    }
    for name, (edit, reason) in cases.items():
        copy = _copy(output, tmp_path / f'{name}.nexus', edit)
        refused = ring_basis_for_record(copy, labels=labels)
        assert refused.frame_snapshot(labels[0]) is None and refused.frame_reason(labels[0]) == reason, name


def test_h4_ring_facts_come_from_the_declared_nodes_and_forbidden_forms_refuse(qapp, tmp_path):
    from xrd_tools import geometry_figures as gf
    from xrd_tools.io.browse_presentation import read_browse_presentation
    from xrd_tools.session.readiness import ring_basis_for_record
    output, published = _fresh_output(qapp, tmp_path / 'run')
    labels = _labels(output)
    with h5py.File(output, 'r', locking=False) as handle:
        selection = _json(handle['entry/reduction/config/energy_selection'])
    basis = ring_basis_for_record(output, labels=labels)
    for label, energy in zip(selection['frame_labels'], selection['frame_energies_eV']):
        assert basis.frame_snapshot(label)['wavelength'] == energy_eV_to_wavelength_m(energy)
    changed = dict(selection, frame_energies_eV=[selection['frame_energies_eV'][0], 17500.0])
    copy = _copy(output, tmp_path / 'changed.nexus',
                 lambda entry: _replace_json(entry['reduction/config'], 'energy_selection', changed))
    assert ring_basis_for_record(copy, labels=labels).frame_snapshot(labels[1])['wavelength'] == \
        energy_eV_to_wavelength_m(17500.0)
    large = dict(selection, frame_labels=list(range(1, 6001)),
                 frame_energies_eV=[selection['frame_energies_eV'][0], selection['frame_energies_eV'][1]]
                 + [16000.0 + label for label in range(3, 6001)])
    assert 64 * 1024 < len(json.dumps(large, separators=(',', ':')).encode()) < 262_144
    copy = _copy(output, tmp_path / 'large.nexus',
                 lambda entry: _replace_json(entry['reduction/config'], 'energy_selection', large))
    assert ring_basis_for_record(copy, labels=labels).frame_snapshot(labels[1])['wavelength'] == \
        energy_eV_to_wavelength_m(selection['frame_energies_eV'][1])
    other = tmp_path / 'other.h5'
    with h5py.File(other, 'w') as handle:
        handle['selection'] = json.dumps(selection)

    def relink(link):
        def apply(entry):
            del entry['reduction/config/energy_selection']
            entry['reduction/config/energy_selection'] = link
        return apply

    forms = {
        'soft': relink(h5py.SoftLink('/entry/reduction/config/run_configuration')),
        'external': relink(h5py.ExternalLink(str(other), '/selection')),
        'array': lambda entry: (entry['reduction/config'].__delitem__('energy_selection'),
                                entry['reduction/config'].create_dataset(
                                    'energy_selection', data=np.array([json.dumps(selection)] * 2,
                                                                      dtype=h5py.string_dtype()))),
        'over_ceiling': lambda entry: _replace_json(entry['reduction/config'], 'energy_selection',
                                                    dict(selection, note='x' * 300_000)),
    }
    for name, edit in forms.items():
        copy = _copy(output, tmp_path / f'{name}.nexus', edit)
        refused = ring_basis_for_record(copy, labels=labels)
        assert refused.frame_snapshot(labels[0]) is None, name
        assert refused.frame_reason(labels[0]) == gf.SAVED_ENERGY_MALFORMED_REASON, name
        presentation, _ = read_browse_presentation(copy)
        assert presentation.get('wavelength_m'), name


def test_h7_append_histories_follow_d5(qapp, tmp_path):
    from xrd_tools import geometry_figures as gf
    from xrd_tools.session.readiness import ring_basis_for_record
    # (A) a finite 2θ energy scan, then a finite Append of a frame inside its energy range
    root = tmp_path / 'A'
    output, published = _fresh_output(qapp, root, (16, 17), mode='Int 1D', unit='2th_deg')
    before = ring_basis_for_record(output, labels=_labels(output))
    assert [dict(before.frame_snapshot(label)) for label in (1, 2)] == [published[1], published[2]]
    _frame(root, 'escan', 3, 16.5)
    executor, identity, events = _run_to_terminal(_series(root, (16, 17, 16.5), mode='Int 1D', unit='2th_deg',
                                                          output_mode='Append'), request_value=9611)
    assert next(e for e in events if e.kind in _TERMINAL).kind is StandardEventKind.FINISHED
    executor.close(identity)
    after = ring_basis_for_record(output, labels=_labels(output))
    assert after.frame_snapshot(3) == {**published[1], 'wavelength': energy_eV_to_wavelength_m(16500.0)}
    assert [after.frame_reason(label) for label in (1, 2)] == [gf.SAVED_HISTORY_REASON] * 2
    # (B) Live: one 16 keV frame at start, two 17 keV frames while live, Stop; then a finite Append
    root = tmp_path / 'B'
    intent = _directory(root, live=True)
    _frame(root / 'raw', 'scan', 1, 16)
    with _page(qapp, intent) as (page, view, observed):
        page.findChild(ScatteringWorkspaceShell).run_controls.startButton.click()
        ready = lambda count: len({e.frame_key.local_frame_label for e in observed
                                   if e.kind is StandardEventKind.FRAME_READY}) >= count
        assert _pump(qapp, lambda: ready(1))
        _frame(root / 'raw', 'scan', 2, 17)
        _frame(root / 'raw', 'scan', 3, 17)
        assert _pump(qapp, lambda: ready(3))
        page.findChild(ScatteringWorkspaceShell).run_controls.stopButton.click()
        terminal = _terminal(qapp, page, observed, StandardEventKind.STOPPED)
        _browsing(qapp, page)
        live = _published(qapp, page, terminal)
    output = root / 'processed' / 'scan_int1d.nexus'
    stopped = ring_basis_for_record(output, labels=_labels(output))
    assert [dict(stopped.frame_snapshot(label)) for label in (1, 2, 3)] == [live[1], live[2], live[3]]
    _frame(root / 'raw', 'scan', 4, 17)
    executor, identity, events = _run_to_terminal(_directory(root, live=False, output_mode='Append'),
                                                  request_value=9612)
    assert next(e for e in events if e.kind in _TERMINAL).kind is StandardEventKind.FINISHED
    executor.close(identity)
    appended = ring_basis_for_record(output, labels=_labels(output))
    assert appended.frame_snapshot(4) == {**live[1], 'wavelength': energy_eV_to_wavelength_m(17000.0)}
    assert [appended.frame_reason(label) for label in (1, 2, 3)] == [gf.SAVED_HISTORY_REASON] * 3


def test_h8_an_older_record_uses_its_recorded_monochromator_wavelength(qapp, tmp_path):
    from xrd_tools import geometry_figures as gf
    from xrd_tools.session.readiness import ring_basis_for_record
    output, published = _fresh_output(qapp, tmp_path / 'run', (16, 16))
    labels = _labels(output)

    def older(entry, *, wavelength_A=0.8, configuration=True):
        del entry['reduction/config/energy_selection']
        monochromator = entry['instrument/monochromator']
        del monochromator['wavelength']
        if wavelength_A is not None:
            monochromator['wavelength'] = wavelength_A
        else:
            del monochromator['energy']  # no recorded energy at all: the monochromator's energy also records one
        if not configuration:
            config = entry['reduction/config']
            run = _json(config['run_configuration'])
            run['scientific_signature']['accepted_scientific_assets'].pop('poni_detector_config_json')
            _replace_json(config, 'run_configuration', run)

    copy = _copy(output, tmp_path / 'older.nexus', older)
    basis = ring_basis_for_record(copy, labels=labels)
    assert [basis.frame_snapshot(label)['wavelength'] for label in labels] == \
        [pytest.approx(0.8e-10, rel=1e-12, abs=0)] * len(labels)
    assert published[labels[0]]['wavelength'] != 0.8e-10
    for name, kwargs, reason in (('no_wavelength', dict(wavelength_A=None), gf.SAVED_NO_ENERGY_REASON),
                                 ('no_configuration', dict(configuration=False), gf.SAVED_NO_DETECTOR_CONFIG_REASON)):
        refused = ring_basis_for_record(_copy(output, tmp_path / f'{name}.nexus',
                                              lambda entry: older(entry, **kwargs)), labels=labels)
        assert refused.frame_snapshot(labels[0]) is None and refused.frame_reason(labels[0]) == reason, name


# ---------------------------------------------------------------------------------------------------- G rows

# Int 1D is not parameterized: Browse has no Int 1D raw display (its raw popup reports "Full Raw is available only
# for an exact acquisition frame"), so terminal Browse rings draw on Int 2D's raw pane (freeze record, F11).
@pytest.mark.parametrize('mode', ('Int 2D',))
def test_g1_terminal_browse_draws_each_frames_own_rings(qapp, tmp_path, mode):
    with _page(qapp, _series(tmp_path, (16, 17), mode=mode)) as (page, view, observed):
        terminal = _run(qapp, page, observed)
        _browsing(qapp, page)
        window = choice(view)
        choose(qapp, view)
        browse = {}
        for key in page._context_controller.frame_keys:
            _show(qapp, page, view, key)
            assert rings(view.raw), ('terminal Browse draws the frame\'s rings', window.rings_note.text())
            browse[key.local_frame_label] = _curves(view.raw)
        _published(qapp, page, terminal)
        for key in page._context_controller.frame_keys:
            _show(qapp, page, view, key)
            assert _same(browse[key.local_frame_label], _curves(view.raw)), key.local_frame_label
        assert not _same(browse[1], browse[2])


def test_g2_moving_between_frames_and_outputs_keeps_each_frames_snapshot(qapp, tmp_path):
    other = tmp_path / 'other'
    executor, identity, events = _run_to_terminal(_series(other, (18, 19)), request_value=9621)
    executor.close(identity)
    with _page(qapp, _series(tmp_path / 'first', (16, 17))) as (page, view, observed):
        _run(qapp, page, observed)
        first = _browsing(qapp, page)
        choice(view)
        choose(qapp, view)
        seen = {}
        for key in page._context_controller.frame_keys:
            _show(qapp, page, view, key)
            assert rings(view.raw), view.phase_rings_reason
            seen['first', key.local_frame_label] = _curves(view.raw)
        second = _reopen(qapp, page, written(other / 'run.nexus', 'Int 2D'))
        assert second is not first and first.released and first.ring_basis is None
        for key in page._context_controller.frame_keys:
            _show(qapp, page, view, key)
            assert rings(view.raw), view.phase_rings_reason
            seen['second', key.local_frame_label] = _curves(view.raw)
        assert not _same(seen['first', 1], seen['second', 1]) and not _same(seen['second', 1], seen['second', 2])


def test_g3_reopen_draws_rings_with_the_source_present_and_thumbnail_only(qapp, tmp_path):
    from xrd_tools import geometry_figures as gf
    output, published = _fresh_output(qapp, tmp_path, (16, 17))
    expected = gf.predict_detector_rings(published[1], 'LaB6', shape=SHAPE).curves
    for state in ('source present', 'thumbnail only'):
        if state == 'thumbnail only':
            for image in tmp_path.glob('escan_*.tif'):
                image.rename(image.with_suffix('.moved'))
        with _page(qapp, _series(tmp_path / 'empty', (16,))) as (page, view, observed):
            _reopen(qapp, page, output)
            choice(view)
            choose(qapp, view)
            key = page._context_controller.frame_keys[0]
            _show(qapp, page, view, key)
            assert rings(view.raw), (state, view.phase_rings_reason)
            assert _near(_native_points(view.raw, SHAPE), expected) < 2.5, state
    from xrd_tools.session.readiness import ring_basis_for_record
    assert ring_basis_for_record(output, labels=_labels(output)).shape == SHAPE


@needs_data
def test_g3b_a_real_binned_output_places_rings_on_its_thumbnail(qapp, tmp_path):
    from xrd_tools import geometry_figures as gf
    from xrd_tools.session.readiness import ring_basis_for_record
    output = SAVED / 'Combi4_Angledependence_samz_4p9_03271005_int2d.nexus'
    labels = _labels(output)
    basis = ring_basis_for_record(output, labels=labels)
    expected = gf.predict_detector_rings(basis.frame_snapshot(labels[0]), 'LaB6', shape=basis.shape).curves
    with _page(qapp, _series(tmp_path / 'empty', (16,))) as (page, view, observed):
        _reopen(qapp, page, output)
        choice(view)
        choose(qapp, view)
        _show(qapp, page, view, page._context_controller.frame_keys[0])
        assert rings(view.raw), view.phase_rings_reason
        assert _near(_native_points(view.raw, basis.shape), expected, tolerance=12.) < 12.


def test_g4_a_stopped_live_run_browses_with_the_energies_live_published(qapp, tmp_path):
    intent = _directory(tmp_path, live=True, mode='Int 2D')
    _frame(tmp_path / 'raw', 'scan', 1, 16)
    with _page(qapp, intent) as (page, view, observed):
        page.findChild(ScatteringWorkspaceShell).run_controls.startButton.click()
        ready = lambda count: len({e.frame_key.local_frame_label for e in observed
                                   if e.kind is StandardEventKind.FRAME_READY}) >= count
        assert _pump(qapp, lambda: ready(1))
        _frame(tmp_path / 'raw', 'scan', 2, 17)
        assert _pump(qapp, lambda: ready(2))
        page.findChild(ScatteringWorkspaceShell).run_controls.stopButton.click()
        terminal = _terminal(qapp, page, observed, StandardEventKind.STOPPED)
        _browsing(qapp, page)
        choice(view)
        choose(qapp, view)
        browse = {}
        for key in page._context_controller.frame_keys:
            _show(qapp, page, view, key)
            assert rings(view.raw), view.phase_rings_reason
            browse[key.local_frame_label] = _curves(view.raw)
        _published(qapp, page, terminal)
        for key in page._context_controller.frame_keys:
            _show(qapp, page, view, key)
            assert _same(browse[key.local_frame_label], _curves(view.raw))


def test_g5_same_path_replacement_retires_the_old_basis_and_drops_late_hydration(qapp, tmp_path):
    with _page(qapp, _series(tmp_path, (16, 16))) as (page, view, observed):
        _run(qapp, page, observed)
        old = _browsing(qapp, page)
        choice(view)
        choose(qapp, view)
        keys = page._context_controller.frame_keys
        _show(qapp, page, view, keys[0])
        assert rings(view.raw), view.phase_rings_reason
        before = _curves(view.raw)
        for index in (1, 2):
            (tmp_path / f'escan_{index:04d}.txt').write_text(
                '# Counters\nI0 = 1\n# Motors\nenergy = 18\nUser: energy, time: Mon Jan 15 10:30:00 2024  # Temp\n')
        executor, identity, events = _run_to_terminal(_series(tmp_path, (18, 18)), request_value=9651)
        executor.close(identity)
        controller = page._context_controller
        controller.select_navigation(keys[1], (keys[1],))  # its hydration is still in flight
        page._refresh_shell()
        target = written(tmp_path / 'run.nexus', 'Int 2D')

        def reopened():
            # Browse refuses a new load while the old context's cleanup is pending (its hydration in flight) and
            # admits it once that cleanup ends, as a user's retry is admitted.
            if controller.browse_context is old and not controller.browse_pending:
                page._select_scan(str(target), reopen=True)
            return controller.browse_context not in (None, old)

        assert _pump(qapp, reopened), page._notice_text
        new = _browsing(qapp, page)
        assert new is not old and old.released and old.ring_basis is None
        _show(qapp, page, view, page._context_controller.frame_keys[0])
        assert rings(view.raw) and not _same(before, _curves(view.raw))


def test_g6_eviction_and_hydration_in_browse_keep_each_frames_snapshot(qapp, tmp_path):
    energies = tuple(range(16, 33))
    with _page(qapp, _series(tmp_path, energies, heavy_window=16)) as (page, view, observed):
        _run(qapp, page, observed)
        _browsing(qapp, page)
        choice(view)
        choose(qapp, view)
        keys = page._context_controller.frame_keys
        seen = {}
        for key in (*keys, keys[0]):
            _show(qapp, page, view, key)
            assert rings(view.raw), view.phase_rings_reason
            seen.setdefault(key.local_frame_label, []).append(_curves(view.raw))
        first, again = seen[keys[0].local_frame_label]
        assert _same(first, again) and not _same(first, seen[keys[-1].local_frame_label][0])


def test_g7_refusals_keep_the_output_usable_and_show_their_reasons(qapp, tmp_path):
    from tests.xdart.scattering.test_poni_v3_start import _V3_PONI
    from xrd_tools import geometry_figures as gf
    root = tmp_path / 'scan'
    output, published = _fresh_output(qapp, root, (16, 17), unit='2th_deg')
    _frame(root, 'escan', 3, 16.5)
    executor, identity, events = _run_to_terminal(_series(root, (16, 17, 16.5), unit='2th_deg',
                                                          output_mode='Append'), request_value=9671)
    executor.close(identity)
    with _page(qapp, _series(tmp_path / 'empty', (16,))) as (page, view, observed):
        _reopen(qapp, page, output)
        window = choice(view)
        choose(qapp, view)
        by_label = {key.local_frame_label: key for key in page._context_controller.frame_keys}
        _show(qapp, page, view, by_label[3])
        assert rings(view.raw), window.rings_note.text()
        _show(qapp, page, view, by_label[1])
        assert not rings(view.raw) and page._last_scientific_projection.heavy is not None
        assert window.rings_note.text() == gf.SAVED_HISTORY_REASON

        def unconfigured(entry):
            config = entry['reduction/config']
            run = _json(config['run_configuration'])
            run['scientific_signature']['accepted_scientific_assets'].pop('poni_detector_config_json')
            _replace_json(config, 'run_configuration', run)

        _reopen(qapp, page, _copy(output, tmp_path / 'unconfigured.nexus', unconfigured))
        _show(qapp, page, view, page._context_controller.frame_keys[0])
        assert page._last_scientific_projection.heavy is not None and not rings(view.raw)
        assert window.rings_note.text() == gf.SAVED_NO_DETECTOR_CONFIG_REASON
    text = _V3_PONI.replace('Parallax: False', 'Parallax: True')
    with _page(qapp, _series(tmp_path / 'parallax', (16, 16), poni_text=text)) as (page, view, observed):
        _run(qapp, page, observed)
        _browsing(qapp, page)
        window = choice(view)
        choose(qapp, view)
        _show(qapp, page, view, page._context_controller.frame_keys[0])
        assert page._last_scientific_projection.heavy is not None and not rings(view.raw)
        assert window.rings_note.text() == gf.PARALLAX_RING_REASON


def test_g8_the_grid_in_browse_draws_q_and_withholds_chi_without_changing_the_choice(qapp, tmp_path):
    from xrd_tools import geometry_figures as gf
    with _page(qapp, _series(tmp_path, (16, 17))) as (page, view, observed):
        terminal = _run(qapp, page, observed)
        _browsing(qapp, page)
        _show(qapp, page, view, page._context_controller.frame_keys[0])
        choose_grid(qapp, view, 'q + χ')
        editor = control(view)
        assert editor.isEnabled(), editor.toolTip()
        assert editor.currentText() == 'q + χ' and gf.GRID_CHI_WITHHELD_NOTE in editor.toolTip()
        browse_items = len(view.raw.overlay_families.get('grid', ()))
        choose_grid(qapp, view, 'q')
        assert len(view.raw.overlay_families.get('grid', ())) == browse_items > 0
        assert gf.GRID_CHI_WITHHELD_NOTE not in editor.toolTip()
        choose_grid(qapp, view, 'χ')
        assert not view.raw.overlay_families.get('grid') and gf.GRID_CHI_WITHHELD_NOTE in editor.toolTip()
        choose_grid(qapp, view, 'q + χ')
        _published(qapp, page, terminal)
        _show(qapp, page, view, page._context_controller.frame_keys[0])
        assert control(view).currentText() == 'q + χ'
        assert len(view.raw.overlay_families.get('grid', ())) > browse_items


def test_g9_a_cif_phases_rings_appear_in_browse_with_hkl_hover(qapp, tmp_path):
    import re
    from tests.xdart.scattering.test_help_residuals_gui import (
        _cif, _hide_tip, _hover, _label, _pane, _pane_items, _segments, _settle, _window)
    from xrd_tools.analysis.phase_references import PhaseLibrary
    entry = PhaseLibrary().add_cif(_cif(tmp_path, 'CsCl', 4.1))
    with _page(qapp, _series(tmp_path / 'run', (16, 16))) as (page, view, observed):
        page.resize(1600, 1000)
        _run(qapp, page, observed)
        _browsing(qapp, page)
        _show(qapp, page, view, page._context_controller.frame_keys[0])
        _window(qapp, view).phase_source.set_selection(((entry.id, True),))
        close_phase_window(view)
        _settle(qapp, view, lambda: bool(_pane_items(view.raw, entry.id)))
        canvas, box = _pane(view.raw)
        segment = _segments(_pane_items(view.raw, entry.id)[0])[0]
        _hide_tip(qapp)
        text = _hover(qapp, canvas, box, segment[len(segment) // 2], want=lambda t: re.search(r'\(\d{3}\)', t))
        expected = view.phase_families[entry.id].rings[0].value
        assert all(_label(hkl) in text for hkl in expected.hkls), text


def test_g10_a_gi_output_browses_with_detector_frame_rings_and_no_grid(qapp, tmp_path):
    gi = GIIntent(enabled=True, incidence_motor='Manual', th_val=0.2)
    with _page(qapp, _series(tmp_path, (16, 16), gi=gi)) as (page, view, observed):
        _run(qapp, page, observed)
        _browsing(qapp, page)
        choice(view)
        choose(qapp, view)
        _show(qapp, page, view, page._context_controller.frame_keys[0])
        assert rings(view.raw), view.phase_rings_reason
        editor = control(view)
        assert not editor.isEnabled() and 'GI' in editor.toolTip()


def test_g11_a_full_raw_of_another_shape_refuses(qapp, tmp_path):
    """Browse shows no full raw today ("Full Raw is available only for an exact acquisition frame"), so the guard is
    driven at the projection that attaches saved rings: a real Browse projection whose heavy is replaced by a full
    raw image of the Run's shape, against the real saved basis of a copy whose detector is wider."""
    from dataclasses import replace
    output, published = _fresh_output(qapp, tmp_path / 'run', (16, 16))

    def wider(entry):
        config = entry['reduction/config']
        run = _json(config['run_configuration'])
        run['scientific_signature']['accepted_scientific_assets']['poni_values']['detector'] = 'Pilatus200k'
        _replace_json(config, 'run_configuration', run)

    copy = _copy(output, tmp_path / 'run' / 'wider.nexus', wider)
    with _page(qapp, _series(tmp_path / 'empty', (16,))) as (page, view, observed):
        _reopen(qapp, page, copy)
        _show(qapp, page, view, page._context_controller.frame_keys[0])
        state = page._last_scientific_projection
        assert state.heavy is not None
        from xrd_tools import geometry_figures as gf
        from xrd_tools.session.readiness import ring_basis_for_record
        from xdart.gui.tabs.scattering.context_projection import attach_saved_rings
        basis = ring_basis_for_record(copy, labels=_labels(copy))
        assert basis.shape != SHAPE
        full = replace(state.heavy, raw=np.zeros(SHAPE, dtype=np.float32), detector_source='full',
                       detector_shape=SHAPE, ring_geometry=None)
        refused = attach_saved_rings(replace(state, heavy=full), basis, full.frame)
        assert refused.heavy.ring_geometry is None and refused.saved_ring_reason == gf.RING_SHAPE_MISMATCH_REASON
        thumbnail = replace(full, detector_source='thumbnail', detector_shape=None)
        attached = attach_saved_rings(replace(state, heavy=thumbnail), basis, thumbnail.frame)
        assert attached.heavy.ring_geometry is not None and attached.heavy.detector_shape == basis.shape
