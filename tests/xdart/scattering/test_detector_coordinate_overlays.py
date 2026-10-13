"""P3b frozen GUI oracle through real Run and the shared ScientificView."""
from dataclasses import replace
import numpy as np
import pytest
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering.test_predicted_ring_overlays import qapp, run_page, select, pane, choose, rings, settle


def control(view):
    labels = [x for x in view.plot_options_dialog.findChildren(QtWidgets.QLabel)
              if x.text() == 'Coordinate grid']
    assert len(labels) == 1, 'Plot Options must contain the Coordinate grid choice'
    return labels[0].parentWidget().findChild(QtWidgets.QComboBox)


def choose_grid(qapp, view, value='q + χ', accept=True):
    view.options.click(); qapp.processEvents()
    control(view).setCurrentText(value)
    (view.plot_options_dialog.accept_button if accept else view.plot_options_dialog.cancel_button).click()
    qapp.processEvents()


def test_coordinate_grid_control_and_labels_reuse_ints_owners(qapp, tmp_path):
    with run_page(qapp, tmp_path, energies=(16,16)) as (page, view):
        editor = control(view)
        assert tuple(editor.itemText(i) for i in range(editor.count())) == ('Off', 'q', 'χ', 'q + χ')
        assert editor.currentText() == 'Off' and editor.toolTip()
        choose_grid(qapp, view)
        assert view.raw.overlay_families['grid']
        assert view._grid_prediction.texts
        assert all(x.toolTip() for x in view.raw.overlay_families['grid'])
        assert all(x.acceptedMouseButtons() == QtCore.Qt.MouseButton.NoButton
                   for x in view.raw.overlay_families['grid'])
        # REFERENCE-PHASES-20261011 successor: the calibrant is a phase family from the Reference Phases window.
        choose(qapp, view); settle(qapp, view, lambda: rings(view.raw))
        ring_items = rings(view.raw)
        choose_grid(qapp, view, 'Off', accept=False)
        assert control(view).currentText() == 'q + χ'
        assert rings(view.raw) == ring_items
        choose_grid(qapp, view, 'Off')
        assert not view.raw.overlay_families.get('grid') and rings(view.raw)


@pytest.mark.parametrize('mode', ('Int 1D', 'Int 2D'))
@pytest.mark.parametrize('thumbnail', (False, True))
def test_full_and_thumbnail_grid_share_the_native_pixel_mapping(qapp, tmp_path, mode, thumbnail):
    with run_page(qapp, tmp_path, energies=(16,16), mode=mode) as (page, view):
        control(view)
        if mode == 'Int 1D':
            view.raw_popup_button.click(); qapp.processEvents()
            select(qapp, page, page._context_controller.navigation.current)
        choose_grid(qapp, view)
        target = pane(view, mode)
        original = target.image.image.copy()
        projection = view._grid_prediction
        shape = (195,487)
        image = np.ones((97,243) if thumbnail else shape)
        target.render(image, detector_shape=shape)
        target.render_overlay_family('grid', projection.curves, texts=projection.texts, native_shape=shape)
        items = target.overlay_families['grid']
        c, r = target.image.image.shape
        transform, pos = target.image.transform(), target.image.pos()
        def mapped(xy):
            xy = np.asarray(xy)
            x = xy[:,0]*(c-1)/(shape[1]-1)+.5
            y = r-.5-xy[:,1]*(r-1)/(shape[0]-1)
            return np.column_stack((transform.m11()*x+transform.m21()*y+transform.dx()+pos.x(),
                                    transform.m12()*x+transform.m22()*y+transform.dy()+pos.y()))
        curves = [x for x in items if isinstance(x, pg.PlotDataItem)]
        for curve, item in zip(projection.curves, curves):
            np.testing.assert_allclose(np.column_stack(item.getData()), mapped(curve.xy), atol=.5)
        texts = [x for x in items if isinstance(x, pg.TextItem)]
        for label, item in zip(projection.texts, texts):
            np.testing.assert_allclose((item.pos().x(),item.pos().y()), mapped([label.xy])[0], atol=.5)
        assert len(curves) == len(projection.curves) and len(texts) == len(projection.texts)
        # Real distinct plan facts, published by the headless Run fixture, drive the popup mode.
        from tests.core.test_detector_coordinate_overlays import convention
        _, views, _ = convention()
        from xrd_tools.geometry_figures import predict_detector_grid
        expected = predict_detector_grid(views[0].geometry.poni, shape=shape, choice='χ',
                                         mode='1d' if mode == 'Int 1D' else '2d')
        offset = 17. if mode == 'Int 1D' else 43.
        from xrd_tools.integrate.calibration import detector_calibration_to_integrator
        from tests.core.test_predicted_ring_overlays import calibration
        ai = detector_calibration_to_integrator(calibration(wavelength=views[0].geometry.poni['wavelength']))
        for curve in expected.curves:
            chi = np.degrees(ai.chi(curve.xy[:,1],curve.xy[:,0]))+offset
            np.testing.assert_allclose(chi, float(curve.name.split()[1]), atol=1e-5)


def test_grid_availability_matches_domain_capability_and_host(qapp, tmp_path):
    with run_page(qapp, tmp_path, energies=(16,16)) as (page, view):
        assert control(view).isEnabled()
        from xrd_tools.geometry_figures import detector_grid_unavailable
        controller = page._context_controller
        snapshot = controller.project(controller.navigation.current).view.geometry.poni
        for changed, reason in (({'domain':'Grazing'}, 'GI'), ({'parallax':True}, 'parallax'),
                                ({'uniform_pixel':False}, 'uniform'), ({'chiDiscAtPi':False}, 'π')):
            assert reason in detector_grid_unavailable(dict(snapshot, **changed), (195,487))
        from tests.xdart.scattering.e3_shell_support import make_shell_projection
        from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
        for mode in ('2D Viewer','Stitch 2D','RSM'):
            shell = ScatteringWorkspaceShell()
            try:
                state = make_shell_projection()
                shell.apply_state(replace(state,scientific=replace(state.scientific,processing_mode=mode)))
                editor = control(shell.scientific)
                assert not editor.isEnabled() and editor.parentWidget().isHidden()
            finally:
                shell.close(); shell.deleteLater(); qapp.processEvents()


def test_grid_reuses_one_prediction_through_resident_and_preview_frames(qapp, tmp_path, monkeypatch):
    with run_page(qapp, tmp_path, energies=(16,)*17, heavy_window=16) as (page, view):
        control(view); choose_grid(qapp, view)
        from xrd_tools import geometry_figures as gf
        actual = gf.predict_detector_grid; calls = []
        def count(*a, **k):
            calls.append(1); return actual(*a, **k)
        monkeypatch.setattr(gf, 'predict_detector_grid', count)
        keys = page._context_controller.frame_keys
        cached = view._grid_prediction
        select(qapp, page, keys[0])
        assert view.raw.overlay_families['grid'] and view._grid_prediction is cached and not calls
        QtWidgets.QApplication.sendEvent(page, QtCore.QEvent(QtCore.QEvent.WindowActivate))
        qapp.processEvents(); page._refresh_shell()
        assert not calls
        choose_grid(qapp, view, 'Off'); choose_grid(qapp, view)
        assert not calls
        view.raw.clear()
        assert not view.raw.overlay_families


def test_terminal_browse_average_and_reintegrate_have_no_grid_geometry(qapp, tmp_path):
    with run_page(qapp, tmp_path, energies=(16,16)) as (page, view):
        control(view); choose_grid(qapp, view)
        page._context_controller.select_browse()
        page._refresh_shell(); qapp.processEvents()
        # BROWSE-REFERENCE-RINGS-20261013 (superseded terminal-Browse clause, D4(c)): terminal Browse draws the
        # saved frame's q contours and withholds its χ half-rays; the choice is unchanged and the row says why.
        select(qapp, page, page._context_controller.frame_keys[0])
        assert control(view).isEnabled() and control(view).currentText() == 'q + χ', control(view).toolTip()
        from xrd_tools.geometry_figures import GRID_CHI_WITHHELD_NOTE
        assert GRID_CHI_WITHHELD_NOTE in control(view).toolTip() and view.raw.overlay_families.get('grid')
        from tests.xdart.scattering.e3_shell_support import make_shell_projection
        for mode in ('Average','Reintegrate'):
            state = make_shell_projection()
            page._shell.apply_state(replace(state,scientific=replace(state.scientific,processing_mode=mode)))
            assert not control(view).isEnabled()
            assert not view.raw.overlay_families.get('grid')
