import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from PySide6.QtTest import QTest

from tests.core.test_rsm_viewer_model import _projection
from xdart.gui.tabs.rsm.page import RSMPage
from tests.xdart.pages.test_rsm_page import _page
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xrd_tools.session.rsm_viewer_model import RSMViewerModel, make_rsm_viewer_values
from xdart.gui.tabs.rsm.display import rsm_keyed_projection


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_keyed_curves_inherit_int_data_style_including_markers(qapp):
    view = ScientificView()
    try:
        values = make_rsm_viewer_values(_projection())
        view.reconcile_keyed(rsm_keyed_projection(values, RSMViewerModel().snapshot(values)))
        for panel in view.keyed_projection.panels[3:]:
            curve = view.keyed_panel(panel.key).curve
            assert curve.opts['symbol'] == 'o'
            assert curve.opts['symbolSize'] == 4
            assert curve.opts['pen'].widthF() == 1.4
            assert curve.opts['connect'] == 'finite'
    finally:
        view.close(); view.deleteLater(); qapp.processEvents()


def test_actual_rsm_page_defaults_to_mean_and_shared_ranges_change_all_six(qapp):
    page = _page(qapp)
    try:
        source = _projection(shape=(5, 7, 9))
        page._show('shown.nexus', source)
        page.resize(1300, 800); page.show(); qapp.processEvents()
        values, snapshot = page.shown
        assert snapshot.state.projection
        assert all(not page.view.keyed_panel(p.key).selector.isVisible() for p in page.view.keyed_projection.panels[:3])
        assert len(page.view.coordinate_ranges) == 3
        row = page.view.coordinate_ranges[2]
        axis = values.axes[2][1]
        assert [e.text() for e in row.editors] == ['2.000', '3.000']
        editor = row.editors[0]
        editor.setFocus(); editor.selectAll(); QTest.keyClicks(editor, format(float(axis[4]), '.9g'))
        QTest.keyClick(editor, QtCore.Qt.Key.Key_Return); qapp.processEvents()
        expected = RSMViewerModel().snapshot(values, projection=True,
            coordinate_ranges=((0., 1.), (1., 2.), (float(axis[4]), 3.)))
        for actual, wanted in zip(page.shown[1].products, expected.products, strict=True):
            np.testing.assert_array_equal(actual.values, wanted.values)
        assert row.slider.toolTip() and row.label.toolTip()
        # The canonical handle widget really receives a mouse gesture.
        before = page.shown[1]
        slider = row.slider
        QTest.mouseClick(slider, QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(slider.width()//3, slider.height()//2))
        qapp.processEvents()
        assert page.shown[1] is not before
        assert values.intensity.tobytes() == make_rsm_viewer_values(source).intensity.tobytes()
    finally:
        page.close(); page.deleteLater(); qapp.processEvents()
