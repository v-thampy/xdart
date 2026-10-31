"""DISPLAY-FLOAT32-128, the view side (design note v3): the mounted image widget keeps float32 display copies, paints
the Log extremes at the ends of the scale, and its owned arrays stay within the ledger's renderer charge (F3); the 2D
Viewer holds one frame at a time, a refused prepared frame included, and a tight budget shows each frame after the
previous one is released (F5); and a diagnostic timing of a real 4096x4096 frame (F4, no bound).

Rows named ``preservation`` hold on dev 7ed95ef7 too, and so does the ordinary render-peak bound (that tree charged
more); every other row is new behaviour and fails there.
"""
from __future__ import annotations

import hashlib
import os
import time
import tracemalloc
import weakref
from contextlib import contextmanager
from functools import partial
from threading import Event, current_thread
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.scientific_view import MAX_VIEWER_LOADING_SNAPSHOT_PIXELS, ScientificView
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.modules.display_context import Viewer2DCleanupReceipt, Viewer2DCleanupState
from xrd_tools.io import viewer_2d as api
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from tests.xdart.scattering import test_p2a1_viewer_context_page as p2a1
from tests.xdart.scattering.test_viewer_2d_navigation import _settle_2d_layout, _wait, _wait_cleanup

F32 = np.dtype(np.float32)
#: allocations that do not scale with the frame (the colour map, Qt and pyqtgraph objects)
FIXED_SLACK = 2**20


def _readonly(value):
    value.setflags(write=False)
    return value


def _log_reference(image):
    shifted = np.asarray(image, dtype=float).copy()
    low = np.nanmin(shifted)
    if low < 1:
        shifted -= low - 1
    return np.log10(shifted)


def _painted(pane, source, index):
    """The colour the pane painted for ``source[index]``: the raw pane shows the frame transposed and mirrored in Y."""
    pane.image.render()
    row, column = index
    colour = pane.image.qimage.pixelColor(column, source.shape[0] - 1 - row)
    return colour.red(), colour.green(), colour.blue(), colour.alpha()


def _frame(shape, nan_fraction, seed=1):
    rng = np.random.default_rng(seed)
    value = rng.uniform(1.0, 60000.0, size=shape).astype(np.float32)
    value[rng.uniform(size=shape) < nan_fraction] = np.nan
    return _readonly(value)


def _render_peaks(pane, source, zoom=None):
    """Owned-array peaks, the canonical frame included, of a first render and of each scale and colour change after it,
    each painted; everything drawn from the frame before them is released first."""
    pane.clear_image_buffers(keep_chrome=True)
    base = tracemalloc.get_traced_memory()[0] - source.nbytes
    peaks = {}
    for step, log, cmap in (("first Linear", False, "viridis"), ("Linear to Log", True, "viridis"),
                            ("Log recolour", True, "magma"), ("Log to Linear", False, "magma"),
                            ("Linear recolour", False, "viridis"), ("first Log", True, "viridis")):
        if step == "first Log":
            pane.clear_image_buffers(keep_chrome=True)
        tracemalloc.reset_peak()
        pane.render(source, log_scale=log, color_map=cmap)
        if zoom is not None:
            pane.canvas.imageViewBox.setRange(zoom, padding=0.0)
        pane.canvas.image_win.grab()
        peaks[step] = tracemalloc.get_traced_memory()[1] - base
    return peaks


# F3: the mounted _ScientificImageWidget keeps float32 display copies; float64 cakes are never downcast.

def test_float32_detector_image_stays_float32_in_linear_log_and_sqrt():
    view = ScientificView()
    pane, canvas = view.raw, view.raw.canvas
    source = _readonly((np.arange(64 * 48, dtype=np.float32).reshape(64, 48) % 997) - 5)
    digest = hashlib.sha256(source.tobytes()).hexdigest()
    pane.render(source, log_scale=False)
    assert canvas.displayed_image.dtype == F32
    assert np.shares_memory(canvas.displayed_image, source), "Linear shows the frame itself (preservation)"
    pane.render(source, log_scale=True)
    shown = canvas.displayed_image
    assert shown.dtype == F32 and not np.shares_memory(shown, source)
    np.testing.assert_allclose(shown, _log_reference(canvas.raw_image), rtol=1e-6, atol=1e-6)
    levels = canvas.imageItem.levels
    assert np.isfinite(levels).all()
    np.testing.assert_allclose(levels, np.percentile(_log_reference(canvas.raw_image), (0.1, 99.9)),
                               rtol=1e-5, atol=1e-5)
    canvas.setImage(canvas.raw_image, scale="Sqrt")
    signed = np.asarray(canvas.raw_image, dtype=float)
    assert canvas.displayed_image.dtype == F32
    np.testing.assert_allclose(canvas.displayed_image, np.sign(signed) * np.sqrt(np.abs(signed)), rtol=1e-6, atol=1e-6)
    assert hashlib.sha256(source.tobytes()).hexdigest() == digest


def test_float64_cake_keeps_float64_display_copies():
    """Preservation: a scientific float64 cake is never downcast for display."""
    view = ScientificView()
    cake = _readonly(np.linspace(0.0, 50.0, 30 * 20).reshape(30, 20))
    view.cake.render(cake, log_scale=False)
    assert view.cake.canvas.displayed_image.dtype == np.dtype(float)
    assert np.shares_memory(view.cake.canvas.displayed_image, cake)
    view.cake.render(cake, log_scale=True)
    assert view.cake.canvas.displayed_image.dtype == np.dtype(float)


@pytest.mark.parametrize("case", ("spread", "mostly NaN", "wide, zoomed in"))
def test_log_extremes_paint_at_the_ends_of_the_scale_and_nans_stay_transparent(case):
    """A float32 span beyond 3.4e38 overflows the Log shift: the largest pixel becomes +inf and paints in the top colour;
    a minimum below -2**24 loses the shift's +1, so the smallest becomes -inf and paints in the bottom colour. NaNs stay
    transparent, the levels stay finite and the frame's bytes are unchanged; also on pyqtgraph's route for an image
    wider than 32767 pixels, drawn pixel for pixel when zoomed in (REVIEW-2 R3)."""
    view = ScientificView()
    view.resize(1200, 900)
    pane = view.raw
    shape = (2, 40000) if case == "wide, zoomed in" else (8, 6)
    if case == "spread":
        source = np.linspace(-2e38, 1e37, shape[0] * shape[1], dtype=np.float32).reshape(shape)
    else:
        source = np.full(shape, np.nan if case == "mostly NaN" else 10.0, dtype=np.float32)
    bottom, top, hole, finite = (0, 0), (shape[0] - 1, shape[1] - 1), (1, 3), (shape[0] - 1, 1)
    source[bottom], source[top], source[hole], source[finite] = -3e38, 3e38, np.nan, 20.0
    digest = hashlib.sha256(source.tobytes()).hexdigest()
    pane.render(_readonly(source), log_scale=True)
    if case == "wide, zoomed in":
        assert pane.image.autoDownsample
        pane.canvas.imageViewBox.setRange(QtCore.QRectF(0, 0, 300, 2), padding=0.0)
    shown = pane.canvas.displayed_image
    assert shown.dtype == F32
    finite_shown = shown[np.isfinite(shown)]
    assert np.isposinf(shown).sum() == 1 and np.isneginf(shown).sum() == 1 and finite_shown.size
    assert np.isfinite(pane.image.levels).all()
    lut = pane.image.lut
    assert _painted(pane, source, top) == (*(int(value) for value in lut[-1][:3]), 255)
    assert _painted(pane, source, bottom) == (*(int(value) for value in lut[0][:3]), 255)
    assert _painted(pane, source, hole)[3] == 0
    if case == "wide, zoomed in":
        assert pane.image._lastDownsample == (1, 1)
        assert pane.image.qimage.width() == shape[1]
    assert hashlib.sha256(source.tobytes()).hexdigest() == digest


@pytest.mark.parametrize("nan_fraction", (0.0, 0.001, 0.999))
def test_render_peaks_stay_within_the_ledgers_renderer_charge(nan_fraction):
    """The worst owned-array peak of drawing one frame (a recolour in Log, while the old display copy, QImage and NaN
    mask are still alive) stays within the renderer charge the ledger admits (REVIEW-2 R2)."""
    view = ScientificView()
    view.resize(1200, 900)
    source = _frame((2048, 2048), nan_fraction)
    charge = api.viewer_2d_memory_ledger(*source.shape).renderer_peak
    tracemalloc.start()
    try:
        peaks = _render_peaks(view.raw, source)
    finally:
        tracemalloc.stop()
    worst = max(peaks.values())
    assert worst <= charge + FIXED_SLACK, {step: round(peak / source.size, 2) for step, peak in peaks.items()}


def test_a_wide_frame_render_peaks_stay_within_the_wide_charge():
    """Wider than 32767 pixels, pyqtgraph downsamples to the view and keeps NaN coordinates, 16 bytes per NaN pixel;
    zoomed in, nothing is downsampled. The worst peak stays within the wider renderer charge for such frames."""
    view = ScientificView()
    view.resize(1200, 900)
    source = _frame((64, 40000), 0.999)
    charge = api.viewer_2d_memory_ledger(*source.shape).renderer_peak
    assert charge > api.viewer_2d_memory_ledger(2048, 1250).renderer_peak, "same pixels, ordinary shape"
    tracemalloc.start()
    try:
        peaks = _render_peaks(view.raw, source, zoom=QtCore.QRectF(0, 0, 300, 64))
    finally:
        tracemalloc.stop()
    assert view.raw.image._lastDownsample == (1, 1)
    worst = max(peaks.values())
    assert worst <= charge + FIXED_SLACK, {step: round(peak / source.size, 2) for step, peak in peaks.items()}


def test_an_ordinary_frame_after_a_wide_one_is_drawn_pixel_for_pixel():
    """pyqtgraph switches downsampling on for an image wider than 32767 pixels and used to leave it on: the next
    ordinary frame was then averaged in blocks, NaN blocks included, and kept NaN coordinates."""
    view = ScientificView()
    view.resize(1200, 900)
    pane = view.raw
    pane.render(_frame((2, 40000), 0.5), log_scale=False)
    pane.canvas.image_win.grab()
    assert pane.image.autoDownsample
    pane.clear_image_buffers(keep_chrome=True)
    source = _frame((2048, 2048), 0.9)
    pane.render(source, log_scale=True)
    pane.canvas.image_win.grab()
    assert not pane.image.autoDownsample
    assert (pane.image.qimage.width(), pane.image.qimage.height()) == (2048, 2048)
    assert type(pane.image._imageNanLocations) is np.ndarray, "the widget's one-byte mask, not NaN coordinates"


# F5: one frame at a time. Weak references are read as they are, without forcing a collection.

def test_switching_frames_releases_the_previous_frame_before_the_next_read(tmp_path, monkeypatch):
    """Preservation: the page's clear drops the shown frame and its Log copy before the next read is submitted."""
    path = tmp_path / "stack.npy"
    np.save(path, (np.arange(3 * 64 * 48, dtype=np.uint32) % 997).astype(np.uint16).reshape(3, 64, 48))
    controller = p2a1._controller()
    controller.open_viewer_2d(str(path))
    p2a1._await_viewer(controller)
    view = ScientificView()
    payload = controller.project(controller.navigation.current)
    view.raw.render(payload.view.raw, log_scale=True)
    canonical = weakref.ref(controller.viewer_2d_frame.array)
    display = weakref.ref(view.raw.canvas.displayed_image)
    assert display() is not None and not np.shares_memory(display(), canonical())
    del payload
    seen = []
    submit = controller._viewer_2d_submit

    def observed(request):
        seen.append((canonical() is None, display() is None))
        return submit(request)

    monkeypatch.setattr(controller, "_viewer_2d_submit", observed)
    page = SimpleNamespace(_context_controller=controller, _shell=SimpleNamespace(scientific=view),
                           _last_scientific_projection=object(), _ensure_timer=lambda: None)
    page._clear_viewer_2d_renderer = partial(ScatteringWorkspace._clear_viewer_2d_renderer, page)
    following = controller.navigation.frames[1]
    ScatteringWorkspace._select_frames(page, ShellCommand(ShellCommandKind.SELECT_FRAME, frame=following,
                                                          frames=(following,)))
    assert seen == [(True, True)], "both the canonical frame and its Log copy were gone when the next read began"
    p2a1._await_viewer(controller)
    assert controller.viewer_2d_frame.label == following.local_frame_label


def _shows(page, label):
    frame = page._context_controller.viewer_2d_frame
    return (frame is not None and frame.label == label
            and page._shell.scientific._viewer_2d_payload is frame.array)


@contextmanager
def _viewer_page(tmp_path, values):
    """The real 2D Viewer page, offscreen, showing frame 0 of ``values`` saved as a.h5 (and again as c.h5)."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    paths = (tmp_path / "a.h5", tmp_path / "c.h5")
    for path in paths:
        with h5py.File(path, "w") as handle:
            handle.create_dataset("entry/data/data", data=values)
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(processing_mode="2D Viewer", project_root=str(tmp_path),
                                         save_path=str(tmp_path))),
        lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter())
    page.resize(1200, 800)
    page.show()
    try:
        page._set_browser_directory(str(tmp_path), explicit=True)
        page._open_viewer_2d_path(str(paths[0]))
        _wait(page, app, lambda: _shows(page, 0))
        _settle_2d_layout(page, app)
        yield page, app, paths
    finally:
        page.close_workspace()
        app.processEvents()
        page.deleteLater()
        app.processEvents()


def _gate_frame_one(monkeypatch, controller, prepared, entered, release):
    """Hold frame 1's prepared commit until ``release``, keeping only a weak reference to its array."""
    commit = type(controller._viewer_2d).commit

    def gated_commit(owner, value):
        if getattr(value.request, "label", None) == 1 and not prepared:
            prepared.append(weakref.ref(value.frame.array))
            entered.set()
            assert release.wait(6)
        return commit(owner, value)

    monkeypatch.setattr(type(controller._viewer_2d), "commit", gated_commit)


def _observe_reads(monkeypatch, prepared, seen):
    import xdart.gui.tabs.scattering.hydration_transport as hydration
    read = hydration.read_viewer_2d_frame

    def observed(catalog, label, **kwargs):
        if prepared:  # once frame 1 is prepared: is it gone when this read begins?
            seen.append((label, prepared[0]() is None))
        return read(catalog, label, **kwargs)

    monkeypatch.setattr(hydration, "read_viewer_2d_frame", observed)


def test_a_refused_prepared_frame_is_released_before_the_next_read(tmp_path, monkeypatch):
    """Frame 1 is read and prepared, then refused at its commit because the operator stepped on to frame 2. The worker
    meets frame 2's request as soon as it has delivered frame 1's outcome; frame 1's array must already be gone when
    frame 2's read begins (REVIEW-2 R1)."""
    values = np.arange(4 * 12 * 16, dtype=np.uint16).reshape(4, 12, 16)
    with _viewer_page(tmp_path, values) as (page, app, _paths):
        controller = page._context_controller
        transport = controller._viewer_2d_standalone
        entered, release, armed = Event(), Event(), Event()
        prepared, seen = [], []
        _gate_frame_one(monkeypatch, controller, prepared, entered, release)
        _observe_reads(monkeypatch, prepared, seen)
        dispatch = transport.dispatch_detached

        def queued_at_the_boundary(mutation):
            # The page queues frame 2 on its next drain; hold the worker just after it delivered frame 1's outcome
            # until then, as a busy GUI thread does.
            dispatch(mutation)
            if armed.is_set() and current_thread().name == "scattering-preview-transport":
                armed.clear()
                deadline = time.monotonic() + 6
                while transport._queued is None and time.monotonic() < deadline:
                    time.sleep(0.005)

        monkeypatch.setattr(transport, "dispatch_detached", queued_at_the_boundary)
        view = page._shell.scientific
        try:
            QtTest.QTest.mouseClick(view.next_frame, QtCore.Qt.LeftButton)
            _wait(page, app, entered.is_set)
            QtTest.QTest.mouseClick(view.next_frame, QtCore.Qt.LeftButton)
            armed.set()
            release.set()
            _wait(page, app, lambda: _shows(page, 2))
        finally:
            release.set()
        assert seen == [(2, True)], "frame 1's refused array was still alive when frame 2's read began"
        np.testing.assert_array_equal(controller.viewer_2d_frame.array, values[2])


def test_pending_cleanup_holds_the_next_read_until_the_prepared_frame_is_released(tmp_path, monkeypatch):
    """Preservation: while the refused frame's cleanup is pending, its array is still owned and no other read begins;
    once the cleanup succeeds, the array is gone before the next read."""
    values = np.arange(4 * 12 * 16, dtype=np.uint16).reshape(4, 12, 16)
    with _viewer_page(tmp_path, values) as (page, app, paths):
        controller = page._context_controller
        entered, release = Event(), Event()
        prepared, seen, disposals = [], [], []
        _gate_frame_one(monkeypatch, controller, prepared, entered, release)
        _observe_reads(monkeypatch, prepared, seen)
        dispose = type(controller._viewer_2d).dispose

        def pending_once(owner, disposal):
            disposals.append(disposal.prepared is not None)
            if len(disposals) == 1:
                return Viewer2DCleanupReceipt(disposal.token, Viewer2DCleanupState.CLEANUP_PENDING)
            return dispose(owner, disposal)

        monkeypatch.setattr(type(controller._viewer_2d), "dispose", pending_once)
        view = page._shell.scientific
        try:
            QtTest.QTest.mouseClick(view.next_frame, QtCore.Qt.LeftButton)
            _wait(page, app, entered.is_set)
            QtTest.QTest.mouseClick(view.next_frame, QtCore.Qt.LeftButton)
            release.set()
            _wait(page, app, lambda: len(disposals) == 1)
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                app.processEvents()
                page._drain_executor()
                time.sleep(0.005)
            assert disposals == [True] and seen == []
            assert prepared[0]() is not None, "the blocked cleanup still owns the refused frame"
            assert controller.viewer_2d_frame is None
            page._open_viewer_2d_path(str(paths[1]))
            assert _wait_cleanup(page, app), controller.viewer_2d_diagnostic
            _wait(page, app, lambda: _shows(page, 0)
                  and controller.viewer_2d_context.original_path == str(paths[1]))
        finally:
            release.set()
        assert disposals == [True, True]
        assert seen and seen[0] == (0, True), "the refused frame was released before the next read"


def test_a_failed_renderer_clear_refuses_the_next_read(tmp_path, monkeypatch):
    """Preservation: when the view cannot clear the shown frame, the frame stays owned and no other read begins."""
    import xdart.gui.tabs.scattering.hydration_transport as hydration
    values = np.arange(4 * 12 * 16, dtype=np.uint16).reshape(4, 12, 16)
    with _viewer_page(tmp_path, values) as (page, app, _paths):
        controller = page._context_controller
        shown = controller.viewer_2d_frame
        reads = []
        read = hydration.read_viewer_2d_frame
        monkeypatch.setattr(hydration, "read_viewer_2d_frame",
                            lambda catalog, label, **kwargs: (reads.append(label), read(catalog, label, **kwargs))[1])

        def refuse(*_args, **_kwargs):
            raise RuntimeError("the view kept its frame")

        monkeypatch.setattr(page._shell.scientific, "clear_viewer_2d", refuse)
        QtTest.QTest.mouseClick(page._shell.scientific.next_frame, QtCore.Qt.LeftButton)
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            app.processEvents()
            page._drain_executor()
            time.sleep(0.005)
        assert reads == []
        assert controller.viewer_2d_frame is shown and controller.viewer_2d_cleanup_pending


def test_a_tight_budget_shows_each_frame_after_the_previous_is_released(tmp_path, monkeypatch):
    """The budget holds one frame's charge but not that charge plus the shown frame: every frame still displays, since
    the shown frame is released before the next read (his rule; REVIEW-2 R2). The read's receipt reserves exactly the
    ledger's admission."""
    monkeypatch.setattr(api, "_physical_ram_bytes", lambda: 450_000_000)
    budget = api.viewer_2d_memory_ledger(1, 1).budget
    ledger = api.viewer_2d_memory_ledger(1024, 1024)
    assert ledger.admission <= budget < ledger.admission + ledger.canonical_bytes
    values = (np.arange(3 * 1024 * 1024, dtype=np.uint32) % 65521).astype(np.uint16).reshape(3, 1024, 1024)
    with _viewer_page(tmp_path, values) as (page, app, _paths):
        controller = page._context_controller
        for label in (1, 2):
            QtTest.QTest.mouseClick(page._shell.scientific.next_frame, QtCore.Qt.LeftButton)
            _wait(page, app, lambda: _shows(page, label))
            np.testing.assert_array_equal(controller.viewer_2d_frame.array, values[label])
            assert controller._viewer_2d.receipt.reserved == api.viewer_2d_selected_ledger(
                controller._viewer_2d.catalog, label).admission == ledger.admission
        assert controller.viewer_2d_diagnostic == ""
    assert api.LOADING_SNAPSHOT_RESERVATION == 4 * MAX_VIEWER_LOADING_SNAPSHOT_PIXELS, "the ledger charges the snapshot"


# F4: diagnostic only, never a red. Run with XDART_F4_TIMING=1 and -s, once per tree, in its own process.

@pytest.mark.skipif(os.environ.get("XDART_F4_TIMING") != "1", reason="diagnostic timing: XDART_F4_TIMING=1")
def test_f4_read_and_draw_timing_of_a_4096_square_frame(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    import gc
    native = (np.arange(4096 * 4096, dtype=np.uint32) % 65521).astype(np.uint16).reshape(4096, 4096)
    path = tmp_path / "frame_4096.tif"
    tifffile.imwrite(path, native)
    nan_source = native.astype(np.float64)
    nan_source[::7, ::5] = np.nan
    nan_path = tmp_path / "frame_4096_nan.npy"
    np.save(nan_path, nan_source)
    del nan_source
    app = QtWidgets.QApplication.instance()
    view = ScientificView()
    view.resize(1200, 900)
    view.show()
    for name, source in (("uint16 TIFF", path), ("float64 NaN NPY", nan_path)):
        for attempt in range(3):
            catalog = api.catalog_viewer_2d(source)
            tracemalloc.start()
            start = time.perf_counter()
            frame = api.read_viewer_2d_frame(catalog, 0)
            read_s = time.perf_counter() - start
            read_peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            assert frame.array.shape == (4096, 4096)
            if name == "uint16 TIFF":
                assert frame.array[100, 100] == native[100, 100]
            for log_scale in (False, True):
                tracemalloc.start()
                start = time.perf_counter()
                view.raw.render(frame.array, log_scale=log_scale)
                view.raw.canvas.imageItem.render()
                view.raw.canvas.image_win.viewport().repaint()
                app.processEvents()
                draw_s = time.perf_counter() - start
                draw_peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                print(f"F4 {name} attempt {attempt} {'Log' if log_scale else 'Linear'}: read {read_s:.3f} s "
                      f"(traced peak {read_peak / 2**20:.0f} MiB), draw {draw_s:.3f} s (traced peak "
                      f"{draw_peak / 2**20:.0f} MiB), canonical {frame.array.dtype} {frame.array.nbytes / 2**20:.0f} MiB")
            view.raw.clear_image_buffers(keep_chrome=True)
            del frame
            gc.collect()
