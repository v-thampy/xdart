"""LF-012: continuous trackpad zoom paints each plot once per step, and the plots still draw the same pixels.

Each wheel step used to cost two paints of an expensive 1-D plot at Retina scale: the antialiased 1.4-px data line
through Qt's general path stroker, translucent grid lines blended one pixel at a time, a second paint queued by
pyqtgraph's in-paint view update, and a rebuild of every curve from its range limit. These rows drive real
QWheelEvents through real views, and compare renders into device-pixel-ratio-tagged images against pyqtgraph's own
drawing of the same items: its stock axis, and the curves' parent settings.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401 - pg.exporters.ImageExporter
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtTest, QtWidgets

from xdart.gui.gui_utils import RectViewBox
from xdart.gui.tabs.scattering.fit_view import FitView
from xdart.gui.tabs.scattering.scientific_curves import draw_reference_lines
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import (
    AxisProjection,
    BrowseTraceSnapshot,
    HeavyProjection,
    KeyedImageProjection,
    KeyedPanelsProjection,
    KeyedTraceProjection,
)
from xdart.gui.tabs.scattering.trend_view import TrendView
from xdart.gui.themes.dark import SEABORN_BG, SEABORN_FG, apply_seaborn_plot_style
from xdart.gui.widgets.scientific_plot import data_curve_style
from xrd_tools.session.display_logic import PanelKey, PanelRole

from tests.xdart.scattering.e3_shell_support import make_shell_projection

PEAKS = (1.1, 1.6, 2.2, 2.7, 3.1, 3.8, 4.4, 5.0, 5.6)
_HK, _HL = PanelKey(PanelRole.SLICE_2D, "HK"), PanelKey(PanelRole.SLICE_2D, "HL")
_H, _K = PanelKey(PanelRole.PROJ_1D, "H"), PanelKey(PanelRole.PROJ_1D, "K")


@pytest.fixture(autouse=True)
def app():
    """pyqtgraph's global options as the app's theme sets them (antialiasing decides how a curve is stroked);
    the previous values come back afterwards."""
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    saved = {name: pg.getConfigOption(name) for name in ("antialias", "background", "foreground")}
    pg.setConfigOptions(antialias=True, background=SEABORN_BG, foreground=SEABORN_FG)
    yield application
    pg.setConfigOptions(**saved)


def _profile(points, index=0):
    """A powder profile: a falling background, sharp peaks, noise; read-only like a published trace."""
    q = np.linspace(.5, 6., points)
    rng = np.random.default_rng(7 + index)
    intensity = 50. + 400. * np.exp(-q) + sum(1e3 / (k + 1) * np.exp(-((q - c) / .02) ** 2)
                                              for k, c in enumerate(PEAKS))
    intensity = intensity * (1. + .02 * index) + rng.normal(0., 3., points)
    return q, intensity


def _frozen(*arrays):
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _state(points=3000, traces=1, *, color_map="Default", gap=None, intensity_scale="Linear", int_2d=False):
    """A Browse state of Int's view: ``traces`` saved profiles in Single (one) or Overlay (more)."""
    projection = make_shell_projection(frame_count=traces, selected_index=0, heavy_indices=(0,) if int_2d else (),
                                       plot_mode="Single" if traces == 1 else "Overlay")
    rows = []
    for index, trace in enumerate(projection.scientific.traces):
        q, intensity = _profile(points, index)
        if gap is not None:
            intensity[gap] = np.nan
        rows.append(replace(trace, axis=AxisProjection(*_frozen(q), "Q", "Å⁻¹"), intensity=_frozen(intensity)[0]))
    state = replace(projection.scientific, processing_mode="Int 2D" if int_2d else "Int 1D", traces=tuple(rows),
                    share_axis=False, color_map=color_map,
                    plot_options=replace(projection.scientific.plot_options, intensity_scale=intensity_scale))
    if int_2d:
        cake_q, chi = _frozen(np.linspace(.5, 6., 400), np.linspace(-180., 180., 180))
        cake = 1e3 * np.exp(-((cake_q[None, :] - 2.2) / .02) ** 2) + np.random.default_rng(3).random((180, 400))
        raw = np.random.default_rng(4).random((300, 280)) * 100.
        state = replace(state, heavy=HeavyProjection(projection.navigation.current, *_frozen(raw, cake),
                                                     AxisProjection(cake_q, "Q", "Å⁻¹"),
                                                     AxisProjection(chi, "χ", "°")))
    shown = tuple(projection.navigation.frames[:traces])
    snapshot = BrowseTraceSnapshot(shown, shown, tuple(range(1, traces + 1)), plot_mode=state.plot_mode,
                                   science_contract=state.browse_science_contract, stacked_options_applied=traces > 1)
    return replace(state, browse_trace_snapshot=snapshot), projection.navigation


def _view(state, navigation):
    view = ScientificView()
    view.resize(1200, 800)
    view.show()
    _reconcile(view, state, navigation)
    return view


def _reconcile(view, state, navigation):
    view.reconcile(state, navigation, completed=1, total=len(navigation.frames), detail="Ready")
    QtTest.QTest.qWait(100)


def _keyed_state(points=1500, color_map="Default"):
    x, y = np.array([1., 2., 3.]), np.array([10., 20.])

    def image(key):
        return KeyedImageProjection(key, f"{key.instance} slice", np.arange(6.).reshape(2, 3),
                                    AxisProjection(x, "H", "", edges=(.5, 3.5)),
                                    AxisProjection(y, "K", "", edges=(5., 25.)), 1, ("0", "1", "2"))

    def trace(key, index):
        q, intensity = _profile(points, index)
        return KeyedTraceProjection(key, f"{key.instance} mean", AxisProjection(q, key.instance, ""), intensity,
                                    "Mean I")
    return KeyedPanelsProjection(((_HK, _HL), (_H, _K)), (image(_HK), image(_HL), trace(_H, 0), trace(_K, 1)),
                                 title="result", status="facts", color_map=color_map, log_scale=False)


class _Paints(QtCore.QObject):
    """Counts the paint events a widget receives."""

    def __init__(self, widget):
        super().__init__(widget)
        self.count = 0
        widget.installEventFilter(self)

    def eventFilter(self, _watched, event):  # noqa: N802 - Qt override
        if event.type() == QtCore.QEvent.Type.Paint:
            self.count += 1
        return False


def _settle(paints):
    """Turn the event loop until no paint has arrived for 100 ms: a second pass queued by the first one lands in
    the next loop turn."""
    seen, deadline = -1, QtCore.QDeadlineTimer(3000)
    while paints.count != seen and not deadline.hasExpired():
        seen = paints.count
        QtTest.QTest.qWait(100)


def _wheel(widget, point, sign=1):
    """One macOS-trackpad-shaped scroll step (pixel and angle deltas, phase ScrollUpdate) at viewport ``point``."""
    viewport = widget.viewport()
    event = QtGui.QWheelEvent(QtCore.QPointF(point), QtCore.QPointF(viewport.mapToGlobal(point)),
                              QtCore.QPoint(0, 4 * sign), QtCore.QPoint(0, 8 * sign), QtCore.Qt.MouseButton.NoButton,
                              QtCore.Qt.KeyboardModifier.NoModifier, QtCore.Qt.ScrollPhase.ScrollUpdate, False)
    QtWidgets.QApplication.sendEvent(viewport, event)


def _paints_per_step(widget, point, steps=6):
    paints = _Paints(widget.viewport())
    _settle(paints)
    counts = []
    for step in range(steps):
        before = paints.count
        _wheel(widget, point, 1 if step < steps // 2 else -1)
        _settle(paints)
        counts.append(paints.count - before)
    return counts


def _render(widget, dpr, offset=(0, 0)):
    """``widget`` as it paints itself into an image of device pixel ratio ``dpr``, ``offset`` logical pixels in."""
    image = QtGui.QImage(int((widget.width() + offset[0]) * dpr) + 1, int((widget.height() + offset[1]) * dpr) + 1,
                         QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr)
    image.fill(0)
    QtWidgets.QWidget.render(widget, image, QtCore.QPoint(*offset))  # pyqtgraph's views shadow render
    return image


def _pixels(image):
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    return np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())[
        :, :4 * image.width()].reshape(image.height(), image.width(), 4).copy()


def _differing(first, second):
    """The number of device pixels that differ between two renders of one size."""
    return int(np.any(_pixels(first) != _pixels(second), axis=2).sum())


def _data_items(plot_widget):
    return plot_widget.getPlotItem().listDataItems()


def _assert_data_drawing(items):
    """Data curves draw as segments with no range limit; the segments meet under an opaque point symbol of the
    line's own colour, which is why their overlaps cannot show."""
    assert items
    for item in items:
        assert item.curve.opts["segmentedLineMode"] == "on"
        assert item.opts["dynamicRangeLimit"] is None
        pen = pg.mkPen(item.opts["pen"])
        assert item.opts["symbol"] == "o" and pen.color().alpha() == 255
        assert pg.mkBrush(item.opts["symbolBrush"]).color() == pen.color()


def _assert_parent_pixels(plot_widget, items, dpr=2.):
    """The plot renders as it does with its curves on the parent's settings (pyqtgraph's own stroke for an
    antialiased pen, 'auto', and its default range limit), up to antialiasing seams: where a dense, noisy curve's
    strokes overlap away from the symbols, the segments blend their edges one after another and the path blends
    their union, so a few edge pixels inside the curve's band differ by a fraction of the contrast."""
    fixed = _pixels(_render(plot_widget, dpr))
    for item in items:
        item.curve.setSegmentedLineMode("auto")
        item.setDynamicRangeLimit(1e6)
    QtTest.QTest.qWait(50)
    delta = np.abs(fixed.astype(int) - _pixels(_render(plot_widget, dpr)).astype(int)).max(axis=2)
    assert delta.max() <= 96 and np.count_nonzero(delta) <= .001 * delta.size, (int(delta.max()),
                                                                                 np.count_nonzero(delta))


# -- Z1: one paint per wheel step (F2) ------------------------------------------------------------------------------

@pytest.mark.parametrize("where", ("plot area", "left axis"))
def test_a_wheel_step_paints_the_one_d_plot_once(where):
    view = _view(*_state())
    try:
        plot = view.curve.getPlotItem()
        if where == "plot area":
            point = view.curve.mapFromScene(plot.getViewBox().sceneBoundingRect().center())
        else:  # over the axis itself: pyqtgraph zooms y alone
            axis = plot.getAxis("left")
            point = view.curve.mapFromScene(axis.mapToScene(QtCore.QPointF(axis.width() / 2., axis.height() / 2.)))
        before = plot.getViewBox().viewRange()
        assert _paints_per_step(view.curve, point) == [1] * 6
        assert plot.getViewBox().viewRange() != before  # it zoomed
    finally:
        view.close()


def test_a_wheel_step_paints_the_cake_image_once():
    view = _view(*_state(int_2d=True))
    try:
        canvas = view.cake.canvas
        point = canvas.image_win.mapFromScene(canvas.imageViewBox.sceneBoundingRect().center())
        before = canvas.imageViewBox.viewRange()
        assert _paints_per_step(canvas.image_win, point) == [1] * 6
        assert canvas.imageViewBox.viewRange() != before
    finally:
        view.close()


# -- Z2, Z3: no range limit on data curves (F3) ----------------------------------------------------------------------

def test_a_wheel_step_rebuilds_no_browse_curve():
    view = _view(*_state(points=1500, traces=10))
    try:
        items = _data_items(view.curve)
        assert len(items) == 10 and not any(item.opts["clipToView"] for item in items)
        rebuilds = []
        for item in items:
            original = item.updateItems
            item.updateItems = lambda *args, _original=original, **kwargs: (rebuilds.append(1),
                                                                             _original(*args, **kwargs))[1]
        point = view.curve.mapFromScene(view.curve.getViewBox().sceneBoundingRect().center())
        _paints_per_step(view.curve, point, steps=4)
        assert rebuilds == []
        _assert_data_drawing(items)
    finally:
        view.close()


@pytest.mark.parametrize("height", (1e-6, 1e-9))
def test_a_curve_crosses_a_tiny_y_window_where_its_data_do(height):
    """A y window ``height`` high across a steep flank, a quarter of the way between two points (the data span
    about 1e9 to 1e12 view heights): the line crosses it where the data do. pyqtgraph's range limit clipped both
    points to one view height around the window, which moved the crossing to their midpoint."""
    view = _view(*_state())
    try:
        view.legend.setVisible(False)
        item = _data_items(view.curve)[0]
        q, intensity = item.getOriginalDataset()
        index = int(np.argmin(np.abs(q - 2.18)))
        assert intensity[index + 1] - intensity[index] > 10.  # both points lie far outside the window
        level = intensity[index] + .25 * (intensity[index + 1] - intensity[index])
        crossing = q[index] + .25 * (q[index + 1] - q[index])
        box = view.curve.getViewBox()
        box.setRange(xRange=(crossing - .01, crossing + .01), yRange=(level - height / 2., level + height / 2.),
                     padding=0.)
        QtTest.QTest.qWait(100)
        shown = _pixels(_render(view.curve, 2.))
        item.setVisible(False)
        QtTest.QTest.qWait(50)
        lit = np.any(shown != _pixels(_render(view.curve, 2.)), axis=2)  # the device pixels the curve drew
        expected = view.curve.viewport().mapTo(view.curve,
                                               view.curve.mapFromScene(box.mapViewToScene(QtCore.QPointF(crossing,
                                                                                                         level))))
        columns = np.flatnonzero(lit[int(expected.y() * 2.)])
        assert columns.size and abs(columns.mean() / 2. - expected.x()) < 3., (columns, expected)
    finally:
        view.close()


# -- Z4: data curves draw as segments, same pixels (F1 + F3) ---------------------------------------------------------

@pytest.mark.parametrize("case", ("single", "zoomed in", "log", "overlay viridis", "nan gap"))
def test_data_curves_draw_as_segments_with_the_parents_pixels(case):
    state = _state(**{"single": {}, "zoomed in": {}, "log": {"intensity_scale": "Log"},
                      "overlay viridis": {"points": 1500, "traces": 10, "color_map": "viridis"},
                      "nan gap": {"gap": slice(1000, 1100)}}[case])
    view = _view(*state)
    try:
        if case == "zoomed in":  # symbols about 25 px apart: the line between them shows
            view.curve.getViewBox().setRange(xRange=(2.13, 2.27), yRange=(-50., 1100.), padding=0.)
            QtTest.QTest.qWait(100)
        items = _data_items(view.curve)
        _assert_data_drawing(items)
        _assert_parent_pixels(view.curve, items)
    finally:
        view.close()


def test_a_reused_curve_keeps_its_drawing_through_new_data():
    """Single Browse keeps its one curve item through a steady scope and gives it the new arrays (``setData``)."""
    state, navigation = _state(points=1500)
    view = _view(state, navigation)
    try:
        first = _data_items(view.curve)
        _assert_data_drawing(first)
        traces = tuple(replace(trace, intensity=_frozen(trace.intensity * 1.5)[0]) for trace in state.traces)
        _reconcile(view, replace(state, traces=traces), navigation)
        items = _data_items(view.curve)
        assert items == first
        np.testing.assert_array_equal(items[0].getOriginalDataset()[1], traces[0].intensity)
        _assert_data_drawing(items)
        _assert_parent_pixels(view.curve, items)
    finally:
        view.close()


def test_the_keyed_trace_draws_as_segments_with_the_parents_pixels():
    view = ScientificView()
    view.resize(1200, 800)
    view.show()
    try:
        view.reconcile_keyed(_keyed_state())
        view.reconcile_keyed(_keyed_state(color_map="magma"))  # the keyed panel's repeated draw
        QtTest.QTest.qWait(100)
        panel = view.keyed_panel(_H)
        _assert_data_drawing([panel.curve])
        _assert_parent_pixels(panel.plot, [panel.curve])
    finally:
        view.close()


# -- Z5: grid rows filled as the pixels pyqtgraph's axis lights (F4) -------------------------------------------------

def _grid_twins():
    """Two seaborn-styled plots of one profile, pyqtgraph's stock left axis and the grid-row axis."""
    from xdart.gui.gui_utils import GridAxisItem
    plots = []
    for axis_items in ({}, {"left": GridAxisItem("left")}):
        widget = pg.PlotWidget(viewBox=RectViewBox(), axisItems=axis_items)
        apply_seaborn_plot_style(widget.getPlotItem(), grid=True)
        widget.plot(*_profile(1500), **data_curve_style((80, 160, 220)))
        widget.resize(900, 420)
        widget.show()
        plots.append(widget)
    QtTest.QTest.qWait(100)
    return plots


@pytest.mark.parametrize("dpr", (1., 1.25, 1.5, 2., 3.))
def test_grid_rows_light_the_pixels_of_pyqtgraphs_axis(dpr):
    stock, fixed = _grid_twins()
    try:
        assert type(stock.getPlotItem().getAxis("left")) is pg.AxisItem
        for offset in ((0, 0), (3, 5), (1, 1)):
            assert _differing(_render(stock, dpr, offset), _render(fixed, dpr, offset)) == 0
        assert fixed.getPlotItem().getAxis("left").grid_rows  # filled by the axis, not stroked
        for widget in (stock, fixed):  # the layout moves the axis: a title, a left label with units
            widget.setTitle("A title")
            widget.setLabel("left", "Intensity", units="counts")
        QtTest.QTest.qWait(100)
        for offset in ((0, 0), (3, 5)):
            assert _differing(_render(stock, dpr, offset), _render(fixed, dpr, offset)) == 0
    finally:
        for widget in (stock, fixed):
            widget.close()


def test_grid_rows_export_and_record_as_pyqtgraph_draws_them():
    stock, fixed = _grid_twins()
    try:
        images = []
        for widget in (stock, fixed):  # Export... to an image: a scene render at another scale
            exporter = pg.exporters.ImageExporter(widget.getPlotItem())
            exporter.parameters()["width"] = 731
            images.append(exporter.export(toBytes=True))
        assert _differing(*images) == 0
        images = []
        for widget in (stock, fixed):  # a recording paint device: the rows stay lines
            picture = QtGui.QPicture()
            QtWidgets.QWidget.render(widget, picture)
            image = QtGui.QImage(widget.width(), widget.height(), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(0)
            painter = QtGui.QPainter(image)
            painter.drawPicture(0, 0, picture)
            painter.end()
            images.append(image)
        assert _differing(*images) == 0
    finally:
        for widget in (stock, fixed):
            widget.close()


def test_every_grid_plot_fills_its_grid_rows():
    from xdart.gui.gui_utils import GridAxisItem
    view, fit, trend = ScientificView(), FitView(), TrendView()
    try:
        view.reconcile_keyed(_keyed_state())
        plots = (view.curve, view.angular_support_plot, view.keyed_panel(_H).plot, fit.plot, fit.residuals,
                 trend.plot)
        for plot in plots:
            plot.resize(600, 300)
            assert isinstance(plot.getPlotItem().getAxis("left"), GridAxisItem)
            assert type(plot.getPlotItem().getAxis("bottom")) is pg.AxisItem
            _render(plot, 2.)
            assert plot.getPlotItem().getAxis("left").grid_rows
    finally:
        for widget in (view, fit, trend):
            widget.close()


# -- Z6: reference (phase) lines draw as segments, same pixels (F5) --------------------------------------------------

@pytest.mark.parametrize("zoomed", (False, True))
def test_reference_lines_draw_as_segments_with_the_parents_pixels(zoomed):
    view = _view(*_state(points=1500))
    try:
        plot = view.curve.getPlotItem()
        families = [draw_reference_lines(plot, np.linspace(1., 5.9, count), color=color, family=f"lines-{count}")
                    for count, color in ((40, (230, 120, 40)), (20, (150, 80, 200)))]
        if zoomed:
            plot.getViewBox().setRange(xRange=(2.13, 2.27), yRange=(-50., 1100.), padding=0.)
        QtTest.QTest.qWait(100)
        fixed = _render(view.curve, 2.)
        for family in families:
            assert family.lines.opts["segmentedLineMode"] == "on"
            family.lines.setSegmentedLineMode("auto")
        QtTest.QTest.qWait(50)
        assert _differing(fixed, _render(view.curve, 2.)) == 0
        assert pg.getConfigOption("segmentedLineMode") == "auto"  # every other curve keeps pyqtgraph's choice
    finally:
        view.close()
