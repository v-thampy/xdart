"""The shared 3-D volume view, the RSM's "3D Viewer" (AUD-9 stage 2b G9): the maintainer, "A button to bring up a
3d viewer for the entire 3d RSM volume". Int's colour-map list, Log and intensity controls over the headless
``volume_rgba``; a volume above the limit refused with its 2× view; offscreen, no GL widget but the same RGBA.
Each draw is computed on the view's worker (item 6): ``_settled`` lets the event loop deliver it."""
from __future__ import annotations

import gc
import threading
import time
import weakref

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.shell_widgets import color_map_choices
from xdart.gui.tabs.scattering.volume_view import VolumeProjection, VolumeView, opengl_available
from xrd_tools.rsm.volume_render import log_transform, volume_levels, volume_rgba


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _settled(view, seconds=20.0):
    """The view once the draws asked for are done: the worker's last result shown (or dropped) as the event loop
    delivers it, and the worker idle."""
    deadline = time.monotonic() + seconds
    while (view.computing or view._running is not None) and time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.002)
    assert not view.computing and view._running is None, view.notice.text()
    return view


def _projection(shape=(4, 5, 6)):
    values = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0) ** 1.5
    values[0, 0, 0] = np.nan
    values.setflags(write=False)
    axes = tuple(np.linspace(index, index + 1.0, size) for index, size in enumerate(shape))
    return VolumeProjection(values, axes, ("H", "K", "L"), "a_rsm.nexus")


def test_the_volume_is_drawn_through_volume_rgba_under_ints_colour_map_and_log(qapp):
    """The view's RGBA is ``volume_rgba`` at Int's auto levels; its colour-map list is Int's own (shared
    builder); the colour map and Log draw it again, Log resetting the levels to their automatic ones."""
    view = VolumeView()
    projection = _projection()
    try:
        assert tuple(view.color_map.itemText(i) for i in range(view.color_map.count())) == color_map_choices()
        view.reconcile(projection)
        _settled(view)
        levels = volume_levels(projection.values)
        np.testing.assert_array_equal(view.rgba, volume_rgba(projection.values, "Default", False, levels=levels))
        assert view.rgba[0, 0, 0, 3] == 0  # an empty voxel is transparent
        view.set_color_map("magma")
        _settled(view)
        np.testing.assert_array_equal(view.rgba, volume_rgba(projection.values, "magma", False, levels=levels))
        view.log_scale.click()
        _settled(view)
        log_levels = volume_levels(log_transform(projection.values), log=True)
        np.testing.assert_array_equal(view.rgba,
                                      volume_rgba(projection.values, "magma", True, levels=log_levels))
        assert view.intensity.autoscale.isChecked() and view.title.text() == "a_rsm.nexus"
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_manual_levels_draw_the_volume_between_them(qapp):
    """With Autoscale off, the intensity controls' window is the levels (and the opacity floor)."""
    view = VolumeView()
    projection = _projection()
    try:
        view.reconcile(projection)
        _settled(view)
        view.intensity.autoscale.setChecked(False)
        view.intensity._set_window((10.0, 200.0))
        view.intensity.rangeChanged.emit(10.0, 200.0)
        _settled(view)
        low, high = view.intensity.values()
        np.testing.assert_array_equal(view.rgba, volume_rgba(projection.values, "Default", False,
                                                             levels=(low, high)))
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_a_volume_above_the_limit_is_refused_with_its_2x_view(qapp):
    """Above the 3D Viewer's limit the view draws nothing and says why, offering the 2× view, which it then
    draws at half resolution."""
    view = VolumeView()
    view.max_voxels = 100  # a 6×6×6 volume (216 voxels) is above it; its 2× view (27) is not
    try:
        view.reconcile(_projection((6, 6, 6)))
        assert view.rgba is None and "2× view" in view.notice.text() and not view.half_view.isHidden()
        view.half_view.click()
        _settled(view)
        assert view.drawn_shape == (3, 3, 3) and view.notice.isHidden() and view.half_view.isHidden()
        view.reconcile(_projection((6, 6, 6)))  # a new volume starts at full resolution again
        assert view.rgba is None and not view.half_view.isHidden()
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_without_opengl_no_gl_widget_is_built_and_release_lets_the_volume_go(qapp):
    view = VolumeView()
    try:
        if not opengl_available():  # an offscreen display: the view says so and still computes the RGBA
            assert view.gl is None
            assert view.findChild(QtWidgets.QLabel, "volumeViewNoOpenGL") is not None
        else:
            assert view.gl is not None
        view.reconcile(_projection())
        _settled(view)
        assert view.rgba is not None and view.projection is not None
        view.release()
        assert view.rgba is None and view.projection is None and view.title.text() == ""
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_the_gl_items_are_built_where_opengl_is_available(qapp, monkeypatch):
    """The GL path's items (the volume, its box and the three axis labels), built as on a display with
    OpenGL; offscreen nothing is painted, but the items' construction and the camera call are exercised."""
    from xdart.gui.tabs.scattering import volume_view
    monkeypatch.setattr(volume_view, "_GL_AVAILABLE", True)
    view = VolumeView()
    try:
        assert view.gl is not None
        view.reconcile(_projection())
        _settled(view)
        import pyqtgraph.opengl as gl
        kinds = sorted(type(item).__name__ for item in view.gl.items)
        assert kinds == ["GLBoxItem", "GLTextItem", "GLTextItem", "GLTextItem", "GLVolumeItem"]
        assert [item.text for item in view.gl.items if isinstance(item, gl.GLTextItem)][0].startswith("H 0…")
        view.release()
        assert view.gl.items == []
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_log_on_or_off_auto_or_manual_draws_the_bytes_and_levels_of_volume_rgbas_own_log(qapp):
    """Codex's combined stage-2b review, priority 5: with Log on the view transforms the volume once, for its
    domain and levels and for its colours alike, where volume_rgba transformed it a second time. The RGBA bytes,
    the domain and the levels are those of volume_rgba transforming the volume itself (log=True) at the levels
    taken in the transformed space: Log off and on, automatic and manual levels, empty voxels, and values below 1
    (Log's shift)."""
    shape = (6, 7, 8)
    values = ((np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 40.0) / 3.0) ** 3 / 50.0
    values[0, 0, 0] = values[3, 4, 5] = np.nan
    values.setflags(write=False)
    assert np.nanmin(values) < 0.0 < 1.0 < np.nanmax(values)
    axes = tuple(np.linspace(index, index + 1.0, size) for index, size in enumerate(shape))
    view = VolumeView()
    try:
        view.reconcile(VolumeProjection(values, axes, ("H", "K", "L"), "a_rsm.nexus"))
        _settled(view)
        for log in (False, True):
            if view.log_scale.isChecked() is not log:
                view.log_scale.click()  # Log draws again at its automatic levels
                _settled(view)
            shown = log_transform(values) if log else values
            finite = shown[np.isfinite(shown)]
            auto = volume_levels(shown, log=log)
            assert view.intensity.autoscale.isChecked() and view.intensity.values() == auto, log
            assert view.intensity.slider.domain() == (float(finite.min()), float(finite.max())), log
            assert view.rgba.tobytes() == volume_rgba(values, "Default", log, levels=auto).tobytes(), log
            assert view.rgba[0, 0, 0, 3] == view.rgba[3, 4, 5, 3] == 0  # empty voxels are transparent
            manual = (auto[0] + 0.25 * (auto[1] - auto[0]), auto[1] - 0.125 * (auto[1] - auto[0]))
            view.intensity.autoscale.setChecked(False)
            view.intensity._set_window(manual)
            view.intensity.rangeChanged.emit(*manual)
            _settled(view)
            assert view.intensity.values() == manual, log
            assert view.rgba.tobytes() == volume_rgba(values, "Default", log, levels=manual).tobytes(), log
            view.intensity.autoscale.setChecked(True)
            _settled(view)
            assert view.rgba.tobytes() == volume_rgba(values, "Default", log, levels=auto).tobytes(), log
    finally:
        view.deleteLater()
        qapp.processEvents()


# AUD-9 stage 2b item 6 (the maintainer: "yes go ahead with both"): each draw on the view's worker, the latest wins


class _Held:
    """The view's real draw, its result held before it returns until ``release`` is set: a draw finished but not
    yet delivered, so the test decides what is asked for meanwhile. Keeps weak references to every result."""

    def __init__(self, job):
        self.job, self.results = job, []
        self.computed, self.release = threading.Event(), threading.Event()

    def __call__(self, request, cancel, progress):
        result = self.job(request, cancel, progress)
        self.results.append(weakref.ref(result))
        self.computed.set()
        assert self.release.wait(20)
        return result


def _held(monkeypatch):
    from xdart.gui.tabs.scattering import volume_view
    held = _Held(volume_view._draw_volume)
    monkeypatch.setattr(volume_view, "_draw_volume", held)
    return held


def test_only_the_latest_draw_is_shown_and_a_stale_one_is_let_go(qapp, monkeypatch):
    """A draw finished after a later one was asked for is dropped, its buffers let go: one draw on the worker at a
    time, the image shown kept (and the view saying it draws) until the latest arrives."""
    held = _held(monkeypatch)
    view = VolumeView()
    projection = _projection()
    try:
        view.reconcile(projection)  # draw A, held on the worker once computed
        assert held.computed.wait(20)
        assert view.computing and view.rgba is None and view.notice.text() == "Drawing the 3D view…"
        view.set_color_map("magma")  # draw B, asked for while A is on the worker
        assert view._running[1].color_map == "Default" and view._wanted.color_map == "magma"
        held.release.set()
        _settled(view)
        levels = volume_levels(projection.values)
        assert view.rgba.tobytes() == volume_rgba(projection.values, "magma", False, levels=levels).tobytes()
        assert view.rgba.tobytes() != volume_rgba(projection.values, "Default", False, levels=levels).tobytes()
        assert len(held.results) == 2 and view.notice.isHidden()
        gc.collect()
        assert held.results[0]() is None  # A's RGBA and index, dropped
    finally:
        held.release.set()
        _settled(view)
        view.deleteLater()
        qapp.processEvents()


def test_a_colour_map_change_colours_the_kept_index_and_a_log_change_makes_a_new_one(qapp):
    """The drawn image's 8-bit index is kept: a colour-map change applies the new table to it (the same index
    object), a Log change makes a new index; the bytes are volume_rgba's either way."""
    view = VolumeView()
    projection = _projection()
    try:
        view.reconcile(projection)
        _settled(view)
        index, alpha = view._index, view._alpha
        assert index.dtype == np.uint8 and index.shape == projection.values.shape
        view.set_color_map("magma")
        _settled(view)
        assert view._index is index and view._alpha is alpha
        levels = volume_levels(projection.values)
        assert view.rgba.tobytes() == volume_rgba(projection.values, "magma", False, levels=levels).tobytes()
        view.log_scale.click()
        _settled(view)
        assert view._index is not index
        log_levels = volume_levels(log_transform(projection.values), log=True)
        assert view.rgba.tobytes() == volume_rgba(projection.values, "magma", True, levels=log_levels).tobytes()
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_closing_while_a_draw_computes_drops_it_and_lets_its_buffers_go(qapp, monkeypatch):
    """The window closed (``release``) while a draw is on the worker: nothing is drawn when it returns, and its
    result is let go; the worker is free for the next volume."""
    held = _held(monkeypatch)
    view = VolumeView()
    try:
        view.reconcile(_projection())
        assert held.computed.wait(20)
        view.release()
        assert view.projection is None and view.rgba is None and not view.computing and view.notice.isHidden()
        held.release.set()
        _settled(view)
        assert view.rgba is None and view.projection is None and view._index is None
        gc.collect()
        assert held.results[0]() is None
        view.reconcile(_projection())
        _settled(view)
        assert view.rgba is not None
    finally:
        held.release.set()
        _settled(view)
        view.deleteLater()
        qapp.processEvents()


def test_a_draw_refused_or_failed_on_the_worker_is_named_and_nothing_is_drawn(qapp, monkeypatch):
    """A volume with no finite value is refused as before, now from the worker; a draw that fails names the
    failure; a volume above the limit is still refused at once, before any worker."""
    view = VolumeView()
    empty = np.full((3, 4, 5), np.nan)
    try:
        view.reconcile(VolumeProjection(empty, tuple(np.arange(float(n)) for n in empty.shape), ("H", "K", "L")))
        _settled(view)
        assert view.rgba is None and view.notice.text() == "The volume holds no finite value to draw."
        from xdart.gui.tabs.scattering import volume_view

        def fails(request, cancel, progress):
            raise MemoryError("no room")
        monkeypatch.setattr(volume_view, "_draw_volume", fails)
        view.reconcile(_projection())
        _settled(view)
        assert view.rgba is None and view.notice.text() == "the 3D view could not be drawn (MemoryError: no room)"
        view.max_voxels = 10
        view.reconcile(_projection())
        assert not view.computing and "2× view" in view.notice.text() and not view.half_view.isHidden()
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_shutdown_while_a_draw_computes_is_pending_then_clean(qapp, monkeypatch):
    """The view's worker closes with its page: PENDING while a draw runs (the page asks again), then CLEAN, the
    draw never shown and its result let go."""
    from xdart.gui.pages.values import PageCleanup
    held = _held(monkeypatch)
    view = VolumeView()
    try:
        view.reconcile(_projection())
        assert held.computed.wait(20)
        assert view.shutdown().status is PageCleanup.PENDING
        held.release.set()
        deadline = time.monotonic() + 20
        receipt = view.shutdown()
        while receipt.status is PageCleanup.PENDING and time.monotonic() < deadline:
            time.sleep(0.01)
            receipt = view.shutdown()
        assert receipt.status is PageCleanup.CLEAN and view.rgba is None and view._running is None
        QtWidgets.QApplication.processEvents()
        assert view.rgba is None
        gc.collect()
        assert held.results[0]() is None
    finally:
        held.release.set()
        view.deleteLater()
        qapp.processEvents()
