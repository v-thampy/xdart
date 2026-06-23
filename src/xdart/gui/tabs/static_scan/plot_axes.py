# -*- coding: utf-8 -*-
"""Shared pyqtgraph helper: a second, right-hand Y axis on a PlotWidget.

Columns of very different magnitude (peak center ~2 vs amplitude ~1e5, or
temperature vs an ROI sum) read cleanly only when the small one gets its own
scale.  :func:`attach_right_axis` wires the standard pyqtgraph linked-``ViewBox``
pattern (a second ViewBox sharing the x range, its geometry kept in step with
the main view), returning the ViewBox; add right-axis curves with
``vb.addItem(item)``.  Used by the Scan Plot overlay and the fitting trend.
"""

import pyqtgraph as pg


def attach_right_axis(plot_widget):
    """Attach a right-hand Y axis backed by a linked ViewBox sharing the x
    range.  Returns ``(view_box, axis)``; the axis starts hidden — call
    ``axis.setVisible(True)`` (or :func:`set_right_axis_visible`) when a series
    uses it.  Add right-axis curves with ``view_box.addItem(item)`` and clear
    them with ``view_box.clear()``."""
    plot_item = plot_widget.getPlotItem()
    vb = pg.ViewBox()
    plot_item.showAxis("right")
    plot_item.scene().addItem(vb)
    right_axis = plot_item.getAxis("right")
    right_axis.linkToView(vb)
    vb.setXLink(plot_item)
    right_axis.setVisible(False)
    # The right ViewBox shares the x range and AUTO-RANGES its y to fit the
    # right-axis series on every redraw.  Leaving it independently mouse/menu
    # interactive is the classic linked-ViewBox footgun: a stray drag/wheel
    # would pan-or-zoom (and latch off autorange on) just this axis, so the
    # right series silently drifts off-scale.  Disable its own mouse handling —
    # the user interacts with the primary (left) view; x stays linked, y keeps
    # tracking the data.
    vb.setMouseEnabled(x=False, y=False)
    vb.setMenuEnabled(False)

    def _sync_geometry():
        vb.setGeometry(plot_item.vb.sceneBoundingRect())
        vb.linkedViewChanged(plot_item.vb, vb.XAxis)

    plot_item.vb.sigResized.connect(_sync_geometry)
    _sync_geometry()
    return vb, right_axis


def set_right_axis_visible(view_box, axis, visible, label=None):
    """Show/hide the right axis + its ViewBox; set its label when shown."""
    axis.setVisible(bool(visible))
    view_box.setVisible(bool(visible))
    if visible and label is not None:
        axis.setLabel(label)
