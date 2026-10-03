"""ScientificView's typed rectangle region and per-panel presentation (FIT-TOOLS-2026-10, Codex's design review F7:
"The actual missing capabilities are rectangle interaction and per-panel presentation policy"). A fit's data, model
and residual are a keyed 1×3: data and model share the intensity controls' levels; the residual is linear and signed,
on a diverging map with its own symmetric limits, so a global Log never hides its sign. One region type serves a
detector's pixels and a slice's coordinates. Int's keyed views keep their behaviour (``test_scientific_view_keyed.py``,
``test_rsm_page.py``)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.Qt import QtCore

from xdart.gui.tabs.scattering.scientific_view import SIGNED_COLOR_MAP, ScientificView
from xdart.gui.tabs.scattering.shell_values import (
    AxisProjection, KeyedImageProjection, KeyedPanelsProjection, RectRegion,
)
from xrd_tools.session.display_logic import PanelKey, PanelRole


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


_DATA, _MODEL, _RESIDUAL = (PanelKey(PanelRole.SLICE_2D, name) for name in ("data", "model", "residual"))
_H, _K = ("H", "r.l.u."), ("K", "r.l.u.")


def _fit(*, log_scale=False, residual_scale=1.0, own_model=False, x_label="H"):
    x, y = np.linspace(0.9, 1.1, 5), np.linspace(1.9, 2.1, 4)
    data = 10.0 + 100.0 * np.exp(-((x[None, :] - 1.0) ** 2 + (y[:, None] - 2.0) ** 2) / 0.005)
    model = data * 0.98
    residual = (data - model) * residual_scale
    residual[0, 0] = -2.0 * abs(float(residual.max()))  # one negative pixel larger than any positive one

    def panel(key, image, presentation="shared"):
        return KeyedImageProjection(key, key.instance, image, AxisProjection(x, x_label, "r.l.u."),
                                    AxisProjection(y, "K", "r.l.u."), 0, (), navigation=False,
                                    presentation=presentation)
    return KeyedPanelsProjection(((_DATA, _MODEL, _RESIDUAL),), (
        panel(_DATA, data), panel(_MODEL, model, "own" if own_model else "shared"),
        panel(_RESIDUAL, residual, "signed")), color_map="magma", log_scale=log_scale), residual


def _levels(view, key):
    return tuple(view.keyed_panel(key).pane.canvas.histogram.levels())


@pytest.mark.parametrize("log_scale", (False, True))
def test_a_residual_is_linear_signed_and_symmetric_whatever_log_says_and_data_and_model_share_levels(qapp, log_scale):
    view = ScientificView()
    try:
        state, residual = _fit(log_scale=log_scale)
        view.reconcile_keyed(state)
        bound = float(np.max(np.abs(residual)))
        canvas = view.keyed_panel(_RESIDUAL).pane.canvas
        assert canvas.histogram.colorMap().name == SIGNED_COLOR_MAP
        assert view.keyed_panel(_DATA).pane.canvas.histogram.colorMap().name == "magma"
        np.testing.assert_allclose(_levels(view, _RESIDUAL), (-bound, bound))
        assert float(np.nanmin(canvas.displayed_image)) < 0.0  # its sign shown, never shifted for a log
        np.testing.assert_allclose(canvas.displayed_image, residual.T)
        assert _levels(view, _DATA) == _levels(view, _MODEL)  # data and model: one set of levels
        typed = (1.2, 1.8) if log_scale else (20.0, 60.0)
        view.viewer_intensity._open_entry()
        view.viewer_intensity._entry_popup.findChild(QtWidgets.QComboBox, "intensityMode").setCurrentText("Absolute")
        view.viewer_intensity.autoscale.setChecked(False)
        view.viewer_intensity.slider.setValues(*typed, emit=True)  # dragged: data and model, not the residual
        assert _levels(view, _DATA) == _levels(view, _MODEL) == pytest.approx(typed)
        np.testing.assert_allclose(_levels(view, _RESIDUAL), (-bound, bound))
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_a_panel_on_its_own_keeps_its_automatic_levels_and_a_zero_residual_still_has_limits(qapp):
    view = ScientificView()
    try:
        state, _residual = _fit(own_model=True, residual_scale=0.0)
        view.reconcile_keyed(state)
        own = _levels(view, _MODEL)
        view.viewer_intensity._open_entry()
        view.viewer_intensity._entry_popup.findChild(QtWidgets.QComboBox, "intensityMode").setCurrentText("Absolute")
        view.viewer_intensity.autoscale.setChecked(False)
        view.viewer_intensity.slider.setValues(20.0, 60.0, emit=True)
        assert _levels(view, _MODEL) == own and _levels(view, _DATA) == pytest.approx((20.0, 60.0))
        assert _levels(view, _RESIDUAL)[1] > _levels(view, _RESIDUAL)[0]
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_a_region_is_typed_by_its_panels_coordinates_stays_over_redraws_and_says_where_the_user_moved_it(qapp):
    view = ScientificView()
    moved = []
    view.keyedRegionChanged.connect(lambda key, region: moved.append((key, region)))
    try:
        state, _residual = _fit()
        view.reconcile_keyed(state)
        region = RectRegion((0.95, 1.05), (1.95, 2.05), _H, _K)
        view.set_keyed_region(_DATA, region)
        roi = view.keyed_panel(_DATA).region
        assert isinstance(roi, pg.RectROI) and roi in view.keyed_panel(_DATA).pane.canvas.imageViewBox.addedItems
        assert view.keyed_region(_DATA) == region and not moved and view.keyed_region(_MODEL) is None
        with pytest.raises(TypeError, match="cannot be drawn"):
            view.set_keyed_region(_MODEL, RectRegion((0.0, 10.0), (0.0, 10.0), ("column", "px"), ("row", "px")))
        roi.setPos((0.93, 1.92))  # the user drags it (the ROI's own move, finished)
        assert moved[-1][0] == _DATA and moved[-1][1].x == pytest.approx((0.93, 1.03))
        assert moved[-1][1].y == pytest.approx((1.92, 2.02)) and moved[-1][1].x_axis == _H
        view.reconcile_keyed(_fit(log_scale=True)[0])  # a redraw in the same coordinates keeps it
        assert view.keyed_region(_DATA).x == pytest.approx((0.93, 1.03))
        view.set_keyed_region(_DATA, region)  # set by code: where it was asked, nothing emitted
        assert view.keyed_region(_DATA) == region and len(moved) == 1
        view.reconcile_keyed(_fit(x_label="L")[0])  # a redraw in other coordinates drops it
        assert view.keyed_region(_DATA) is None and view.keyed_panel(_DATA).region is None
        view.set_keyed_region(_DATA, RectRegion((0.95, 1.05), (1.95, 2.05), ("L", "r.l.u."), _K))
        view.set_keyed_region(_DATA, None)
        assert view.keyed_region(_DATA) is None
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_one_region_type_gives_a_detectors_pixels_or_a_slices_bins_as_index_ranges():
    pixels = RectRegion((2.0, 4.0), (0.0, 1.0), ("column", "px"), ("row", "px"))
    rows, columns = pixels.index_bounds(np.arange(8.0), np.arange(3.0))
    assert (rows, columns) == (slice(0, 2), slice(2, 5))
    image = np.arange(24.0).reshape(3, 8)
    assert np.shares_memory(image[rows, columns], image)  # a view of the image, not a copy
    descending = RectRegion((0.95, 1.05), (1.0, 3.0), _H, _K)
    rows, columns = descending.index_bounds(np.array([1.2, 1.1, 1.0, 0.9]), np.array([1.0, 2.0, 3.0]))
    assert (rows, columns) == (slice(0, 3), slice(2, 3))
    assert descending.index_bounds(np.array([5.0, 6.0]), np.array([9.0]))[1] == slice(0, 0)
    for bad in (((1.0, 1.0), (0.0, 1.0)), ((0.0, float("nan")), (0.0, 1.0)), ((2.0, 1.0), (0.0, 1.0))):
        with pytest.raises(TypeError):
            RectRegion(*bad, _H, _K)
    with pytest.raises(TypeError):
        KeyedImageProjection(_DATA, "d", np.zeros((1, 1)), AxisProjection(np.zeros(1), "H"),
                             AxisProjection(np.zeros(1), "K"), 0, (), navigation=False, presentation="log")


def test_named_signal_and_background_marks_coexist_move_live_and_reconcile_silently(qapp):
    view = ScientificView()
    moved = []
    view.keyedNamedRegionChanged.connect(lambda key, name, region: moved.append((key, name, region)))
    try:
        view.reconcile_keyed(_fit()[0])
        region = RectRegion((0.95, 1.05), (1.95, 2.05), _H, _K)
        view.set_keyed_named_region(_DATA, "signal", region, color="red")
        view.set_keyed_named_region(_DATA, "background", region, color="blue", dashed=True, visible=False)
        signal = view.keyed_region_item(_DATA, "signal")
        background = view.keyed_region_item(_DATA, "background")
        assert signal is not background and len(view.keyed_panel(_DATA).regions) == 2
        assert background.pen.style() == QtCore.Qt.PenStyle.DashLine and not background.isVisible()
        assert not moved
        signal.setPos((0.92, 1.90), finish=False)  # live drag, before the gesture has finished
        assert moved[-1][:2] == (_DATA, "signal")
        assert moved[-1][2].x == pytest.approx((0.92, 1.02))
        count = len(moved)
        view.set_keyed_named_region(_DATA, "signal", region, color="red")
        view.set_keyed_named_region(_DATA, "background", region, dashed=True, visible=True)
        assert len(moved) == count and background.isVisible()
        view.reconcile_keyed(_fit(log_scale=True)[0])
        assert view.keyed_region_item(_DATA, "signal") is signal
        view.reconcile_keyed(_fit(x_label="L")[0])
        assert view.keyed_named_region(_DATA, "signal") is None
        assert not view.keyed_panel(_DATA).regions
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_detector_picker_reuses_image_and_scale_with_pixel_edges_and_row_zero_at_top(qapp):
    view = ScientificView(detector_picker=True)
    image = np.arange(12.0).reshape(3, 4)
    try:
        key = view.reconcile_detector_image(image)
        pane = view.keyed_panel(key).pane
        np.testing.assert_array_equal(pane.canvas.displayed_image, image.T)
        assert pane.image.boundingRect() == QtCore.QRectF(0, 0, 4, 3)
        assert pane.canvas.imageViewBox.state["yInverted"]
        assert pane.canvas.imageViewBox.state["aspectLocked"]
        region = RectRegion((0, 4), (0, 3), ("column", "px"), ("row", "px"))
        view.set_keyed_named_region(key, "full", region)
        view.detector_log.setChecked(True)
        np.testing.assert_allclose(pane.canvas.displayed_image, np.log10(image.T + 1))
        assert view.keyed_named_region(key, "full") == region
        view.default_scale.setChecked(True)
        np.testing.assert_array_equal(pane.canvas.displayed_image, image.T)
    finally:
        view.deleteLater()
        qapp.processEvents()
