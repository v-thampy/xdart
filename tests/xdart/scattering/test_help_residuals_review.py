"""Terminal-review discriminators, frozen before the HELP-RESIDUALS corrections."""
from dataclasses import replace

import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize('canvas_type', (pg.PlotWidget, pg.GraphicsLayoutWidget))
def test_reference_hover_clears_on_a_real_viewport_leave(qapp, canvas_type):
    from xdart.gui.widgets.scientific_plot import HoverTooltip
    holder = QtWidgets.QWidget()
    canvas = canvas_type(parent=holder)
    holder.show()
    qapp.processEvents()
    hover = HoverTooltip(canvas)
    try:
        hover.show(canvas.mapToScene(20, 20), 'phase (111)')
        QtWidgets.QApplication.sendEvent(canvas.viewport(), QtCore.QEvent(QtCore.QEvent.Type.Leave))
        assert hover.text == '', 'leaving the viewport retires its reference tooltip'
    finally:
        hover.hide()
        holder.close()
        holder.deleteLater()
        qapp.processEvents()


def test_browse_loading_does_not_claim_the_retained_runs_ring_geometry(qapp, tmp_path):
    from tests.xdart.scattering.test_predicted_ring_overlays import choice, choose, rings, run_page, settle
    from xrd_tools.geometry_figures import LOADING_RING_REASON
    with run_page(qapp, tmp_path, energies=(16, 16)) as (page, view):
        window = choice(view)
        choose(qapp, view)
        settle(qapp, view, lambda: rings(view.raw))
        controller = page._context_controller
        controller.select_browse()
        page._refresh_shell()
        qapp.processEvents()
        # Browse intentionally retains the acquisition's RunIdentity. Its still-loading presentation must
        # nevertheless have Browse's reason, rather than promise a Run snapshot will arrive from disk.
        # BROWSE-REFERENCE-RINGS-20261013 (superseded clause): the saved output now supplies this frame's rings,
        # so a Browse frame whose image is loading says it waits for that image, never for the Run's.
        state = replace(page._last_scientific_projection, heavy=None, retain_display=False)
        view.reconcile(state, controller.navigation, completed=2, total=2, detail='Loading Browse frame')
        assert window.rings_note.text() != LOADING_RING_REASON, window.rings_note.text()
        from xrd_tools.geometry_figures import BROWSE_LOADING_RING_REASON
        assert window.rings_note.text() == BROWSE_LOADING_RING_REASON, window.rings_note.text()
