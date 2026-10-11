"""Frozen P3a GUI oracle. Real Runs, shared view, and existing PREVIEW transport."""
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
def run_page(qapp, root, energies=(16, 17), mode='Int 2D', poni_text=None, source_spec=None):
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
        page._refresh_shell(); qapp.processEvents()
        yield page, shell.scientific
    finally:
        page.close(); page.deleteLater(); qapp.processEvents()


def choice(view):
    # Behavioral first assertion: parent has no visible Calibrant control.
    labels = [label for label in view.plot_options_dialog.findChildren(QtWidgets.QLabel)
              if label.text() == 'Calibrant']
    assert len(labels) == 1, 'Plot Options must contain the Calibrant choice'
    return labels[0].parentWidget().findChild(QtWidgets.QComboBox)


def choose(qapp, view, value='LaB6', accept=True):
    view.options.click(); qapp.processEvents()
    editor = choice(view); editor.setCurrentText(value)
    (view.plot_options_dialog.accept_button if accept else view.plot_options_dialog.cancel_button).click()
    qapp.processEvents()
    return editor


def select(qapp, page, key):
    controller = page._context_controller
    assert controller.select_navigation(key, (key,))
    page._refresh_shell()
    until = time.monotonic()+30
    while time.monotonic()<until:
        qapp.processEvents(); page._refresh_shell()
        payload = controller.project(key)
        if payload is not None and payload.view.thumbnail is not None:
            return payload
        time.sleep(.01)
    raise AssertionError('selected frame did not settle')


def pane(view, mode='Int 2D'):
    return view.raw_popup_image if mode == 'Int 1D' else view.raw


@pytest.mark.parametrize('mode', ('Int 1D', 'Int 2D'))
def test_int_rings_follow_current_frame_and_off_clears(qapp, tmp_path, mode):
    with run_page(qapp, tmp_path, mode=mode) as (page, view):
        editor = choice(view)
        assert editor.currentText() == 'Off' and editor.toolTip()
        if mode == 'Int 1D':
            view.show_raw_popup('Detector image'); qapp.processEvents()
        before = pane(view, mode).image.image.copy()
        choose(qapp, view)
        target = pane(view, mode)
        assert target.ring_items
        first = tuple(item.getData() for item in target.ring_items)
        choose(qapp, view, 'CeO2', accept=False)
        assert choice(view).currentText() == 'LaB6'
        assert target.ring_items
        keys = page._context_controller.frame_keys
        select(qapp, page, keys[0])
        second = tuple(item.getData() for item in target.ring_items)
        assert len(first) != len(second) or any(not np.array_equal(a[0], b[0]) for a,b in zip(first, second))
        np.testing.assert_array_equal(target.image.image, before)
        choose(qapp, view, 'Off')
        assert not target.ring_items


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
        target.render_ring_curves((Curve('edges', xy, 'predicted', 'display only'),), native_shape=shape)
        item = target.ring_items[0]
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
            target.render_ring_curves((Curve('one', np.zeros((1,2)), 'predicted', 'display only'),), native_shape=singleton)
            assert np.isfinite(np.column_stack(target.ring_items[0].getData())).all()


@pytest.mark.parametrize('enabled', (False, True))
def test_parallax_controls_ring_availability(qapp, tmp_path, enabled):
    from tests.xdart.scattering.test_poni_v3_start import _V3_PONI
    text = _V3_PONI.replace('Parallax: False', f'Parallax: {enabled}')
    with run_page(qapp, tmp_path, energies=(16,16), poni_text=text) as (page, view):
        editor = choice(view)
        assert editor.isEnabled() is (not enabled)
        if enabled:
            assert 'parallax' in editor.toolTip().lower()
        else:
            choose(qapp, view)
            assert view.raw.ring_items


def test_unchanged_activation_rebuilds_no_ring_overlay(qapp, tmp_path, monkeypatch):
    import xrd_tools.geometry_figures as gf
    with run_page(qapp, tmp_path, energies=(16,16)) as (page, view):
        choice(view); choose(qapp, view)
        assert view.raw.ring_items
        items = tuple(view.raw.ring_items)
        calls = []; actual = gf.predict_detector_rings
        def count(*a, **k):
            calls.append(1); return actual(*a, **k)
        monkeypatch.setattr(gf, 'predict_detector_rings', count)
        QtWidgets.QApplication.sendEvent(page, QtCore.QEvent(QtCore.QEvent.WindowActivate))
        qapp.processEvents(); page._refresh_shell()
        select(qapp, page, page._context_controller.frame_keys[0])
        view.raw.canvas.imageViewBox.setRange(xRange=(10, 20), yRange=(20, 40))
        qapp.processEvents()
        assert not calls and tuple(view.raw.ring_items) == items
    with run_page(qapp, tmp_path/'scan') as (page, view):
        choice(view); choose(qapp, view); calls.clear()
        select(qapp, page, page._context_controller.frame_keys[0])
        assert len(calls) == 1


@pytest.mark.parametrize('mode', ('Int 1D', 'Int 2D', '2D Viewer', 'Stitch 2D', 'Average', 'Reintegrate'))
def test_empty_overlays_preserve_every_pane_consumer(qapp, mode):
    from tests.xdart.scattering.e3_shell_support import make_shell_projection
    shell = ScatteringWorkspaceShell(); state = make_shell_projection()
    state = replace(state, scientific=replace(state.scientific, processing_mode=mode))
    try:
        shell.show(); shell.apply_state(state); qapp.processEvents()
        view = shell.scientific
        assert not getattr(view.raw, 'ring_items', ())
        if mode.startswith('Int') or mode in ('Average', 'Reintegrate'):
            editor = choice(view)
            assert not editor.isEnabled()
            assert 'Predicted rings are drawn only for frames reduced by Run in this session.' in editor.toolTip()
        view.show_raw_popup('Contributing raw')
        raw = np.arange(35.).reshape(5,7)
        view.show_raw_image(raw, status='source', color_map='Default', log_scale=False)
        np.testing.assert_array_equal(view.raw_popup_image.image.image, raw.T[:,::-1])
        assert not getattr(view.raw_popup_image, 'ring_items', ())
        view.raw_popup_dialog.close(); qapp.processEvents()
        view.raw.clear(); assert not getattr(view.raw, 'ring_items', ())
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
        choice(view); choose(qapp, view)
        assert len(controller.frame_keys) == 2 and view.raw.ring_items
        prior = payload.view.geometry.poni
        other = select(qapp, page, controller.frame_keys[0])
        assert other.view.geometry.poni == prior
        assert view.raw.ring_items


def test_rehydrated_frame_keeps_its_ring_geometry(qapp, tmp_path):
    with run_page(qapp, tmp_path, energies=(16,17,18)) as (page, view):
        choice(view); choose(qapp, view)
        controller = page._context_controller
        keys = controller.frame_keys
        art = controller.acquisition_context
        store = art.publication_store
        original = store.get(keys[0].local_frame_label).view.geometry.poni
        store.set_max_heavy_items(1)
        payload = select(qapp, page, keys[0])
        assert payload.view.geometry.poni == original
        assert payload.view.geometry.poni != controller.project(keys[-1]).view.geometry.poni
        assert view.raw.ring_items
