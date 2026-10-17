"""Full ScientificView Figure layout and typed overlay oracle; no test painter."""
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_predicted_and_observed_roles_remain_distinct_without_colour_in_both_themes(qapp, theme):
    from xrd_tools import geometry_figures as gf
    from xdart.gui.tabs.scattering.scientific_view import ScientificView
    from xrd_tools.session.display_logic import PanelKey, PanelRole
    from xdart.gui.tabs.scattering.shell_values import KeyedFigureProjection, KeyedPanelsProjection
    image = np.arange(35.).reshape(5, 7)
    figure = gf.Figure('pixel centres', 'column', 'row', 'px', 'Synthetic; stored array.',
        curves=(gf.Curve('prediction', ((0., 1.), (6., 2.)), 'predicted', 'Predicted ring.'),
                gf.Curve('observed', ((0., 2.), (6., 3.)), 'observed', 'Observed points.')),
        markers=(gf.Marker('beam', (3., 2.), 'beam', 'Direct beam.'),),
        arrows=(gf.Arrow('lab up', (1., 1.), (1., 3.), 'lab', 'Lab +x is up.'),),
        texts=(gf.Text('label', (3., 4.), 'Synthetic frame', 'Synthetic example.'),), image=image)
    view = ScientificView(); view.setProperty('theme', theme)
    try:
        key = PanelKey(PanelRole.FIGURE, 'figure')
        state = KeyedPanelsProjection(((key,),), (KeyedFigureProjection(key, 'Figure', figure),))
        view.reconcile_keyed(state); view.show(); qapp.processEvents()
        panel = view.keyed_panel(key)
        assert panel.curves['prediction'].opts['pen'].style() == QtCore.Qt.PenStyle.DashLine
        assert panel.curves['observed'].opts['pen'].style() == QtCore.Qt.PenStyle.SolidLine
        assert panel.viewbox.state['aspectLocked'] == 1.
        assert panel.viewbox.menuEnabled() and panel.caption.toolTip()
        assert all(w.isHidden() for w in (view.frame_selector, view.previous_frame, view.next_frame))
        assert all(item.toolTip() for item in panel.overlay_items)
        assert len(panel.overlay_items) == 5
        prior = view.keyed_projection
        with pytest.raises((ValueError, TypeError)):
            gf.Curve('bad', ((1., 2., 3.),), 'predicted', 'Bad dimensionality.')
        assert view.keyed_projection is prior
        view._apply_processing_layout('Int 1D')
        assert view._keyed_area.isHidden() and not view.frame_selector.isHidden()
    finally:
        view.close(); view.deleteLater(); qapp.processEvents()
