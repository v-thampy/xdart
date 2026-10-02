"""The shared 3-D volume view, the RSM's "3D Viewer" (AUD-9 stage 2b G9): the maintainer, "A button to bring up a
3d viewer for the entire 3d RSM volume". Int's colour-map list, Log and intensity controls over the headless
``volume_rgba``; a volume above the limit refused with its 2× view; offscreen, no GL widget but the same RGBA."""
from __future__ import annotations

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
        levels = volume_levels(projection.values)
        np.testing.assert_array_equal(view.rgba, volume_rgba(projection.values, "Default", False, levels=levels))
        assert view.rgba[0, 0, 0, 3] == 0  # an empty voxel is transparent
        view.set_color_map("magma")
        np.testing.assert_array_equal(view.rgba, volume_rgba(projection.values, "magma", False, levels=levels))
        view.log_scale.click()
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
        view.intensity.autoscale.setChecked(False)
        view.intensity._set_window((10.0, 200.0))
        view.intensity.rangeChanged.emit(10.0, 200.0)
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
        for log in (False, True):
            if view.log_scale.isChecked() is not log:
                view.log_scale.click()  # Log draws again at its automatic levels
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
            assert view.intensity.values() == manual, log
            assert view.rgba.tobytes() == volume_rgba(values, "Default", log, levels=manual).tobytes(), log
            view.intensity.autoscale.setChecked(True)
            assert view.rgba.tobytes() == volume_rgba(values, "Default", log, levels=auto).tobytes(), log
    finally:
        view.deleteLater()
        qapp.processEvents()
