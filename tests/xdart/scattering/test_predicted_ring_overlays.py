"""Frozen P3a GUI oracle. Real Runs, shared view, and existing PREVIEW transport.

REFERENCE-PHASES-20261011 successor (declared in its kickoff): calibrant rings now come from calibrant entries in
the Tools-panel Reference Phases window, through the shared phase worker, instead of Plot Options' Calibrant row.
Every scientific assertion of the eight original rows is kept; helpers keep their names for D and DP.
"""
from contextlib import contextmanager
from dataclasses import replace
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering.test_energy_scan_run import _series
from tests.xdart.scattering.test_p1b_output_graph import _intent, write_poni
from xrd_tools.sources import image_series_spec
from xrd_tools.session.intent_store import RunIntentStore
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.gui.tabs.scattering.display_values import StandardEventKind


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@contextmanager
def run_page(qapp, root, energies=(16, 17), mode='Int 2D', poni_text=None, source_spec=None, heavy_window=None, parallax_override=None):
    root.mkdir(parents=True, exist_ok=True)
    raw = _series(root, energies)
    poni = root/'cal.poni'; write_poni(poni)
    # Input geometry only; expected positions always come from upstream pyFAI.
    text = poni.read_text()
    text = text.replace('Distance: 0.1234', 'Distance: 0.03')
    if poni_text is not None:
        text = poni_text
    poni.write_text(text)
    intent = _intent(raw, root/'run.nexus', poni, processing_mode=mode)
    intent.source_spec = source_spec or image_series_spec(raw, metadata_format='txt')
    if parallax_override is not None:
        intent.poni_v3_override = parallax_override
    if heavy_window is not None:
        intent.run_options['heavy_window'] = heavy_window
    executor = StandardRunExecutor(join_timeout=2.)
    page = ScatteringWorkspace(intents=RunIntentStore(intent), lifecycle=ScatteringCoordinator(),
                               sources=FilesystemSourceAdapter(), executor=executor)
    shell = page.findChild(ScatteringWorkspaceShell)
    observed = []; drain = executor.drain_events
    def observe():
        events = drain(); observed.extend(events); return events
    executor.drain_events = observe
    try:
        page.show(); shell.run_controls.startButton.click()
        until = time.monotonic()+60
        while time.monotonic()<until and not any(e.kind in (StandardEventKind.FINISHED, StandardEventKind.FAILED) for e in observed):
            qapp.processEvents(); time.sleep(.01)
        terminal = [e for e in observed if e.kind in (StandardEventKind.FINISHED, StandardEventKind.FAILED)]
        assert terminal and terminal[-1].kind is StandardEventKind.FINISHED, (page._notice_text, terminal)
        controller = page._context_controller
        # The terminal follows the output asynchronously into Browse. Complete
        # that owner transition, then explicitly inspect this session's Run.
        while time.monotonic()<until and controller.browse_pending:
            qapp.processEvents(); time.sleep(.01)
        controller.adopt_acquisition(terminal[-1].run_identity)
        controller.select_acquisition()
        page._refresh_shell(); qapp.processEvents()
        yield page, shell.scientific
    finally:
        page.close(); page.deleteLater(); qapp.processEvents()


def choice(view):
    """The window whose calibrant chooser replaced Plot Options' Calibrant row."""
    view.show_reference_phases()
    window = view.reference_phases
    assert window.isVisible() and window.phase_source.calibrant_row.editor.toolTip()
    return window


def choose(qapp, view, value='LaB6', accept=True):
    """Add calibrant ``value`` through the window ('Off' removes every calibrant), then let the shared phase
    worker settle. The window applies live, so a declined choice (``accept=False``) is simply never made."""
    source = choice(view).phase_source
    qapp.processEvents()
    if accept and value == 'Off':
        source.set_selection(tuple(pair for pair in source.selection if not pair[0].startswith('calibrant:')))
    elif accept:
        source.add_calibrant(value)
    settle(qapp, view)
    return source


def rings(target, name='LaB6'):
    """The calibrant's ring items on one pane (one joined item per family)."""
    return tuple(target.overlay_families.get('phase:calibrant:' + name, ()))


def settle(qapp, view, predicate=lambda: True):
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        qapp.processEvents()
        if not view.phase_worker.busy and predicate():
            qapp.processEvents()
            return
        time.sleep(.005)
    raise AssertionError('the shared phase worker did not settle')


def select(qapp, page, key):
    controller = page._context_controller
    assert controller.select_navigation(key, (key,))
    page._refresh_shell()
    until = time.monotonic()+30
    while time.monotonic()<until:
        qapp.processEvents(); page._refresh_shell()
        payload = controller.project(key)
        if payload is not None and payload.view.thumbnail is not None:
            page._refresh_shell(); qapp.processEvents()
            return payload
        time.sleep(.01)
    raise AssertionError('selected frame did not settle')


def pane(view, mode='Int 2D'):
    return view.raw_popup_image if mode == 'Int 1D' else view.raw


@pytest.mark.parametrize('mode', ('Int 1D', 'Int 2D'))
def test_int_rings_follow_current_frame_and_off_clears(qapp, tmp_path, mode):
    with run_page(qapp, tmp_path, mode=mode) as (page, view):
        window = choice(view)
        assert not window.phase_source.selection and window.phase_source.calibrant_row.editor.toolTip()
        if mode == 'Int 1D':
            view.raw_popup_button.click(); qapp.processEvents()
            select(qapp, page, page._context_controller.navigation.current)
            # Opening Int's popup loads full raw asynchronously (as DP waits); compare pixels after it settles.
            settle(qapp, view, lambda: view.raw_popup_image.image.image is not None
                   and view.raw_popup_image.image.image.shape == (487, 195))
        before = pane(view, mode).image.image.copy()
        choose(qapp, view)
        target = pane(view, mode)
        settle(qapp, view, lambda: rings(target))
        first = tuple(item.getData() for item in rings(target))
        choose(qapp, view, 'CeO2', accept=False)
        assert not rings(target, 'CeO2') and rings(target)
        np.testing.assert_array_equal(target.image.image, before)
        keys = page._context_controller.frame_keys
        select(qapp, page, keys[0])
        settle(qapp, view, lambda: rings(target) and (mode != 'Int 1D' or target.image.image.shape == (487, 195)))
        second = tuple(item.getData() for item in rings(target))
        assert len(first) != len(second) or any(not np.array_equal(a[0], b[0], equal_nan=True)
                                                for a, b in zip(first, second))
        current_pixels = target.image.image.copy()
        choose(qapp, view, 'Off')
        assert not rings(target)
        np.testing.assert_array_equal(target.image.image, current_pixels)


@pytest.mark.parametrize('thumbnail', (False, True))
def test_raw_ring_points_share_image_and_hover_mapping(qapp, tmp_path, thumbnail):
    from xrd_tools.io.nexus_record import make_thumbnail_array
    from xrd_tools.geometry_figures import Curve
    with run_page(qapp, tmp_path) as (page, view):
        choice(view)
        choose(qapp, view)
        target = view.raw
        shape = (1043, 981)
        raw = np.zeros(shape, dtype=np.float32)
        raw[[0, -1], :] = 1000; raw[:, [0, -1]] = 1000
        image = make_thumbnail_array(raw, max_size=256) if thumbnail else raw
        target.render(image, detector_shape=shape)
        xy = np.array(((0., 0.), (shape[1]-1., 0.), (shape[1]-1., shape[0]-1.)))
        target.render_overlay_family('edges', (Curve('edges', xy, 'predicted', 'display only'),), native_shape=shape)
        item = target.overlay_families['edges'][0]
        scene = np.column_stack(item.getData())
        for point, native in zip(scene, xy):
            local = target.image.mapFromParent(QtCore.QPointF(*point))
            j = native[0]*(image.shape[1]-1)/(shape[1]-1)+.5
            i = image.shape[0]-.5-native[1]*(image.shape[0]-1)/(shape[0]-1)
            assert local.x() == pytest.approx(j, abs=1e-6)
            assert local.y() == pytest.approx(i, abs=1e-6)
            assert target.image.image[int(np.floor(j)), int(np.floor(i))] == 1000
        for singleton in ((1, 9), (7, 1), (1, 1)):
            target.render(np.ones(singleton), detector_shape=singleton)
            target.render_overlay_family('one', (Curve('one', np.zeros((1,2)), 'predicted', 'display only'),),
                                         native_shape=singleton)
            assert np.isfinite(np.column_stack(target.overlay_families['one'][0].getData())).all()


@pytest.mark.parametrize('kind', ('v3', 'override'))
@pytest.mark.parametrize('enabled', (False, True))
def test_parallax_controls_ring_availability(qapp, tmp_path, enabled, kind):
    from tests.xdart.scattering.test_poni_v3_start import _V3_PONI
    from xrd_tools.session.run_configuration import PoniV3OverrideIntent
    text = _V3_PONI.replace('Parallax: False', f'Parallax: {enabled}') if kind == 'v3' else None
    override = PoniV3OverrideIntent('Si', 450e-6, enabled) if kind == 'override' else None
    with run_page(qapp, tmp_path, energies=(16,16), poni_text=text, parallax_override=override) as (page, view):
        window = choice(view)
        choose(qapp, view)
        assert window.phase_source.calibrant_row.editor.isEnabled(), 'calibrant lines stay available'
        if enabled:
            assert 'parallax' in window.rings_note.text().lower() and not rings(view.raw)
        else:
            settle(qapp, view, lambda: rings(view.raw))
            assert rings(view.raw) and not window.rings_note.text()


def test_unchanged_activation_rebuilds_no_ring_overlay(qapp, tmp_path, monkeypatch):
    import xrd_tools.analysis.phase_references as provider
    calls = []; actual = provider.build_phase_references
    def count(*a, **k):
        calls.append(tuple(entry.id for entry in a)); return actual(*a, **k)
    with run_page(qapp, tmp_path, energies=(16,16)) as (page, view):
        choice(view); choose(qapp, view); settle(qapp, view, lambda: rings(view.raw))
        items = rings(view.raw)
        monkeypatch.setattr(provider, 'build_phase_references', count)
        QtWidgets.QApplication.sendEvent(page, QtCore.QEvent(QtCore.QEvent.WindowActivate))
        qapp.processEvents(); page._refresh_shell()
        select(qapp, page, page._context_controller.frame_keys[0]); settle(qapp, view)
        view.raw.canvas.imageViewBox.setRange(xRange=(10, 20), yRange=(20, 40))
        qapp.processEvents()
        assert not calls and rings(view.raw) == items
    with run_page(qapp, tmp_path/'scan') as (page, view):
        choice(view); choose(qapp, view); settle(qapp, view, lambda: rings(view.raw)); calls.clear()
        select(qapp, page, page._context_controller.frame_keys[0]); settle(qapp, view, lambda: rings(view.raw))
        assert calls and all(call == ('calibrant:LaB6',) for call in calls), 'a new energy lists the calibrant again'
        assert rings(view.raw)


@pytest.mark.parametrize('mode', ('Int 1D', 'Int 2D', '2D Viewer', 'Stitch 2D', 'Average', 'Reintegrate'))
def test_empty_overlays_preserve_every_pane_consumer(qapp, mode):
    from tests.xdart.scattering.e3_shell_support import make_shell_projection
    shell = ScatteringWorkspaceShell(); state = make_shell_projection()
    state = replace(state, scientific=replace(state.scientific, processing_mode=mode))
    try:
        shell.show(); shell.apply_state(state); qapp.processEvents()
        view = shell.scientific
        assert not rings(view.raw)
        if mode.startswith('Int') or mode in ('Average', 'Reintegrate'):
            window = choice(view)
            options = replace(state.scientific.plot_options, phase_references=(('calibrant:LaB6', True),))
            shell.apply_state(replace(state, scientific=replace(state.scientific, plot_options=options)))
            settle(qapp, view)
            assert not rings(view.raw)
            assert ("Predicted rings are drawn only on a Run's frames while that Run is shown. "
                    "Frames opened from a file, and Average or Reintegrate results, carry no ring geometry."
                    ) in window.rings_note.text()
        view.show_raw_popup('Contributing raw')
        raw = np.arange(35.).reshape(5,7)
        view.show_raw_image(raw, status='source', color_map='Default', log_scale=False)
        np.testing.assert_array_equal(view.raw_popup_image.image.image, raw.T[:,::-1])
        assert not rings(view.raw_popup_image)
        view.raw_popup_dialog.close(); qapp.processEvents()
        view.raw.clear(); assert not rings(view.raw)
    finally:
        shell.close(); shell.deleteLater(); qapp.processEvents()


def test_logical_point_frames_keep_their_own_ring_geometry(qapp, tmp_path):
    import h5py
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.sources.point_images import PointImagePolicy
    path = tmp_path/'points.h5'
    with h5py.File(path, 'w') as stream:
        detector = stream.create_group('entry/instrument/detector')
        detector.create_dataset('data', data=np.full((4,195,487), 100, np.uint16))
        detector.create_dataset('detectorSpecific/nimages', data=2)
        detector.create_dataset('detectorSpecific/ntrigger', data=2)
    spec = SourceSpec(path, SourceKind.NEXUS_STACK,
        options={'point_images': PointImagePolicy(2, 'average').to_dict()})
    with run_page(qapp, tmp_path/'run', source_spec=spec) as (page, view):
        controller = page._context_controller
        payload = controller.project(controller.frame_keys[-1])
        assert payload.view.geometry is not None and payload.view.geometry.poni is not None
        choice(view); choose(qapp, view); settle(qapp, view, lambda: rings(view.raw))
        assert len(controller.frame_keys) == 2 and rings(view.raw)
        prior = payload.view.geometry.poni
        other = select(qapp, page, controller.frame_keys[0]); settle(qapp, view, lambda: rings(view.raw))
        assert other.view.geometry.poni == prior
        assert rings(view.raw)


def test_rehydrated_frame_keeps_its_ring_geometry(qapp, tmp_path):
    # The supported Run option funds its existing sixteen-frame checkpoint.
    # Seventeen frames still force frame 0 through real eviction and hydration.
    with run_page(qapp, tmp_path, energies=tuple(range(16,33)), heavy_window=16) as (page, view):
        choice(view); choose(qapp, view)
        controller = page._context_controller
        keys = controller.frame_keys
        art = controller.acquisition_context
        store = art.publication_store.artifacts[keys[0].artifact].publications
        original = store.get(keys[0].local_frame_label).view.geometry.poni
        last = store.get(keys[-1].local_frame_label).view.geometry.poni
        assert store._max_heavy_items == 16
        assert store.get(keys[0].local_frame_label).view.intensity_2d is None
        payload = select(qapp, page, keys[0]); settle(qapp, view, lambda: rings(view.raw))
        assert payload.view.geometry.poni == original
        assert payload.view.geometry.poni != last
        assert rings(view.raw)


def test_same_energy_step_to_an_evicted_frame_reuses_the_prediction(qapp, tmp_path, monkeypatch):
    # O5 beyond the heavy window (P3a terminal review): the frame's own snapshot equals the cached one,
    # so its PREVIEW hydration redraws the cached rings without predicting them again.
    import xrd_tools.analysis.phase_references as provider
    with run_page(qapp, tmp_path, energies=(16,)*17, heavy_window=16) as (page, view):
        choice(view); choose(qapp, view); settle(qapp, view, lambda: rings(view.raw))
        controller = page._context_controller
        keys = controller.frame_keys
        store = controller.acquisition_context.publication_store.artifacts[keys[0].artifact].publications
        assert store.get(keys[0].local_frame_label).view.intensity_2d is None
        calls = []; actual = provider.build_phase_references
        def count(*a, **k):
            if 'two_theta_range' in k:  # a detector-ring listing (1D lines list by q range)
                calls.append(1)
            return actual(*a, **k)
        monkeypatch.setattr(provider, 'build_phase_references', count)
        select(qapp, page, keys[0]); settle(qapp, view, lambda: rings(view.raw))
        assert rings(view.raw) and not calls
