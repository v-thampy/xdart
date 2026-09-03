from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.workspace_shell import (
    ScatteringWorkspaceShell,
)

from tests.xdart.scattering.e3_shell_support import make_shell_projection


def test_one_d_plot_restores_labels_legend_and_connected_lines() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = ScatteringWorkspaceShell()
    shell.resize(1600, 1000)
    shell.show()
    try:
        shell.apply_state(make_shell_projection(plot_mode="Overlay"))
        app.processEvents()
        scientific = shell.scientific
        assert [
            scientific.plot_axis.itemText(index)
            for index in range(scientific.plot_axis.count())
        ] == ["Q (Å⁻¹)", "2θ (°)", "χ (°)"]
        assert scientific.slice.text() == "χ (c/w)"

        items = scientific.curve.listDataItems()
        assert len(items) == 5
        assert scientific.legend is not None
        assert len(scientific.legend.items) == 5
        assert all(item.opts["symbol"] == "o" for item in items)
        assert all(item.opts["symbolSize"] == 4 for item in items)
        assert all(item.opts["pen"].widthF() == 1.4 for item in items)
        axis = scientific.curve.getPlotItem().getAxis("left")
        assert axis.labelText == "I / Monitor (a.u.)"
    finally:
        shell.close()
        shell.deleteLater()
        app.processEvents()


def test_one_d_plot_reports_cursor_coordinates() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = ScatteringWorkspaceShell()
    shell.resize(1600, 1000)
    shell.show()
    try:
        shell.apply_state(make_shell_projection(plot_mode="Single"))
        app.processEvents()
        scientific = shell.scientific
        point = scientific.curve.getPlotItem().vb.mapViewToScene(
            QtCore.QPointF(1.0, 0.5)
        )
        scientific._mouse_moved((point,))
        assert scientific.cursor_position.text.startswith("x=")
        assert ", y=" in scientific.cursor_position.text
    finally:
        shell.close()
        shell.deleteLater()
        app.processEvents()
