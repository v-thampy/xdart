"""The committed RSM's six products on the shared scientific panes (AUD-9).

Three slices on Int's ``ScientificImagePane`` (the image pane its Int 2D cake uses) and three mean
projections on pyqtgraph plots with the shared cursor readout, laid out as the viewer model's 2×3
(``session.rsm_viewer_model``): the slices through the selected indices in the first row, the mean
projections in the second. ``ScientificView`` is one image beside one 1-D axis; six panels on six axes are
composed from its panes rather than copied.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.shell_values import AxisProjection
from xdart.gui.tabs.scattering.shell_widgets import ScientificImagePane
from xdart.gui.widgets.scientific_plot import PlotCursorReadout
from xrd_tools.session.display_logic import PanelRole
from xrd_tools.session.rsm_viewer_model import rsm_axis_bounds


def _labels(values):
    """Each axis's label as the frame and the RSM dialog write it (``H``; ``Qx (Å⁻¹)``), with no separate
    unit: the stored unit key (``q_A^-1``) would make Int's pane name every Q axis "Q"."""
    return values.coordinate_frame.axis_labels


def _titles(values, snapshot):
    """The six panels' titles: each slice names the index it is taken at."""
    state = snapshot.state
    first, second, third = values.coordinate_frame.axis_symbols
    return (
        f"{first}{second} slice · {third} index {state.l_index}",
        f"{first}{third} slice · {second} index {state.k_index}",
        f"{second}{third} slice · {first} index {state.h_index}",
        f"{first} mean projection",
        f"{second} mean projection",
        f"{third} mean projection",
    )


def _axes(values):
    """Each panel's (horizontal, vertical) axis indices: the slices' pairs, then each projection's axis."""
    return ((0, 1), (0, 2), (1, 2), (0, None), (1, None), (2, None))


class RSMResultView(QtWidgets.QWidget):
    """Six panes and the slice indices; a paint is all six or none (the prior six stay on a failure)."""

    indicesChanged = QtCore.Signal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        indices = QtWidgets.QHBoxLayout()
        indices.addWidget(QtWidgets.QLabel("Slice indices"))
        self.index_labels, self.index_controls = [], []
        for name in ("H", "K", "L"):
            label = QtWidgets.QLabel(name)
            control = QtWidgets.QSpinBox()
            control.setObjectName(f"rsmPage{name}Index")
            control.setRange(0, 0)
            control.setEnabled(False)
            control.valueChanged.connect(self._index_edited)
            indices.addWidget(label)
            indices.addWidget(control)
            self.index_labels.append(label)
            self.index_controls.append(control)
        self.facts = QtWidgets.QLabel("No RSM result shown")
        self.facts.setObjectName("rsmPageFacts")
        self.facts.setWordWrap(True)
        indices.addWidget(self.facts, 1)
        layout.addLayout(indices)
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self.slices = tuple(ScientificImagePane(lock_aspect=False) for _ in range(3))
        self.projections = tuple(pg.PlotWidget() for _ in range(3))
        self.curves = tuple(plot.plot(connect="finite") for plot in self.projections)
        self._readouts = tuple(PlotCursorReadout(plot) for plot in self.projections)  # Int's cursor label
        for index, pane in enumerate(self.slices):
            pane.setObjectName(f"rsmPageSlice{index}")
            grid.addWidget(pane, 0, index)
        for index, plot in enumerate(self.projections):
            plot.setObjectName(f"rsmPageProjection{index}")
            grid.addWidget(plot, 1, index)
        layout.addLayout(grid, 1)
        self._shown = None  # (values, snapshot) on the panes
        self._suppress = False

    @property
    def shown(self):
        return self._shown

    def _index_edited(self, *_args):
        if not self._suppress and self._shown is not None:
            self.indicesChanged.emit(*(control.value() for control in self.index_controls))

    def show_snapshot(self, values, snapshot, *, color_map="viridis", log_scale=False):
        """Paint the six products; on any failure repaint the prior six (or clear) and raise."""
        prior = self._shown
        try:
            self._paint(values, snapshot, color_map=color_map, log_scale=log_scale)
        except BaseException:
            try:
                if prior is None:
                    self.clear()
                else:
                    self._paint(*prior, color_map=color_map, log_scale=log_scale)
            except BaseException:  # the primary failure is the one reported
                pass
            raise
        self._shown = (values, snapshot)

    def _paint(self, values, snapshot, *, color_map, log_scale):
        labels = _labels(values)
        titles = _titles(values, snapshot)
        products = snapshot.products
        if len(products) != 6:
            raise ValueError("an RSM snapshot holds six products")
        for index, (product, (horizontal, vertical)) in enumerate(zip(products, _axes(values), strict=True)):
            x_label = labels[horizontal]
            if product.panel_key.role is PanelRole.SLICE_2D:
                y_label = labels[vertical]
                pane = self.slices[index]
                # the pane's image spans its axes' first to last values: the bins' outer edges, so each
                # pixel sits at its bin centre, as in the RSM dialog (Codex's stage-1 review F3)
                pane.render(np.asarray(product.values),
                            x_axis=AxisProjection(np.array(rsm_axis_bounds(product.x_axis)), x_label, ""),
                            y_axis=AxisProjection(np.array(rsm_axis_bounds(product.y_axis_or_none)), y_label, ""),
                            color_map=color_map, log_scale=log_scale)
                pane.plot.setTitle(titles[index])
            else:
                plot = self.projections[index - 3]
                self.curves[index - 3].setData(np.asarray(product.x_axis, dtype=float),
                                               np.asarray(product.values, dtype=float))
                plot.setTitle(titles[index])
                plot.setLabel("bottom", x_label)
                plot.setLabel("left", "Mean I")
        state = snapshot.state
        self._suppress = True
        try:
            for label, control, symbol, size, at in zip(
                    self.index_labels, self.index_controls, values.coordinate_frame.axis_symbols, values.shape,
                    (state.h_index, state.k_index, state.l_index), strict=True):
                label.setText(symbol)
                control.setRange(0, size - 1)
                control.setValue(at)
                control.setEnabled(True)
        finally:
            self._suppress = False
        axes = tuple(axis for _name, axis in values.axes)
        symbols = values.coordinate_frame.axis_symbols
        self.facts.setText(
            f"{values.coordinate_frame.display_name} · shape {values.shape} · "
            + " · ".join(f"{symbol} {axis[0]:.6g}..{axis[-1]:.6g}" for symbol, axis in zip(symbols, axes,
                                                                                         strict=True)))

    def clear(self, detail="No RSM result shown"):
        for pane in self.slices:
            pane.clear_image_buffers()
            pane.plot.setTitle("")
        for plot, curve in zip(self.projections, self.curves, strict=True):
            curve.setData([], [])
            plot.setTitle("")
        self._suppress = True
        try:
            for control in self.index_controls:
                control.setRange(0, 0)
                control.setValue(0)
                control.setEnabled(False)
        finally:
            self._suppress = False
        self.facts.setText(str(detail))
        self._shown = None


__all__ = ["RSMResultView"]
