"""LIVE-1218-VIEWER2D-BG: a 2D Viewer raw background stays set across frames and files.

His report: "In 2d viewer, setting raw BG is immediately reset when a different frame is selected". His design
word: "if the shapes don't match the BG should be ignored. No need to throw an error".
"""

from __future__ import annotations

import time

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


def _wait(page, qapp, predicate, what="the 2D Viewer"):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        qapp.processEvents()
        page._drain_executor()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"{what} did not settle")


@pytest.fixture
def viewer(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rng = np.random.default_rng(1218)
    stacks = {
        "a.h5": rng.integers(0, 1000, size=(4, 12, 16), dtype=np.uint16),
        "c.h5": rng.integers(0, 1000, size=(3, 12, 16), dtype=np.uint16),
        "d.h5": rng.integers(0, 1000, size=(2, 10, 16), dtype=np.uint16),
    }
    for name, values in stacks.items():
        with h5py.File(tmp_path / name, "w") as handle:
            handle.create_dataset("entry/data/data", data=values)
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            processing_mode="2D Viewer", project_root=str(tmp_path), save_path=str(tmp_path),
        )),
        lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(),
    )
    page.resize(1200, 800)
    page.show()
    page._set_browser_directory(str(tmp_path), explicit=True)
    try:
        yield page, app, tmp_path, stacks
    finally:
        page.close_workspace()
        app.processEvents()
        page.deleteLater()
        app.processEvents()


def _shown(page) -> np.ndarray:
    """The 2D Viewer's drawn image, in the frame's own (rows, columns) orientation (drawn as ``frame.T[:, ::-1]``)."""
    drawn = np.asarray(page._shell.scientific.raw.canvas.raw_image, dtype=np.float64)
    return drawn[:, ::-1].T if drawn.ndim == 2 else np.empty((0, 0))


def _open(page, app, path, label=0):
    page._handle_shell_command(ShellCommand(ShellCommandKind.SELECT_SCAN, str(path)))
    controller = page._context_controller
    _wait(page, app, lambda: (
        controller.viewer_2d_context is not None
        and controller.viewer_2d_context.original_path == str(path)
        and controller.viewer_2d_frame is not None
        and controller.viewer_2d_frame.label == label
        and page._shell.scientific._viewer_2d_payload is not None), f"opening {path.name}")


def _step(page, app, label):
    view, controller = page._shell.scientific, page._context_controller
    QtTest.QTest.mouseClick(view.next_frame, QtCore.Qt.LeftButton)
    _wait(page, app, lambda: (
        controller.viewer_2d_frame is not None and controller.viewer_2d_frame.label == label
        and page._shell.scientific._viewer_2d_payload is not None), f"frame {label}")


def _settle_drawn(page, app, expected):
    _wait(page, app, lambda: _shown(page).shape == expected.shape
          and np.allclose(_shown(page), expected, rtol=1e-6, atol=1e-3, equal_nan=True), "the drawn image")


def test_raw_background_survives_idle_ticks(viewer):
    """Every idle drain tick released the background once the view had drawn it (the 'immediately reset')."""
    page, app, root, stacks = viewer
    view, owner = page._shell.scientific, page._background_owner
    _open(page, app, root / "a.h5")
    _settle_drawn(page, app, stacks["a.h5"][0].astype(np.float64))
    QtTest.QTest.mouseClick(view.background, QtCore.Qt.LeftButton)
    _wait(page, app, lambda: owner.phase == "ACTIVE" and view.background.text() == "Clear Raw BG",
          "Set Raw BG")
    _settle_drawn(page, app, np.zeros((12, 16)))
    key, deadline = owner.active_key, time.monotonic() + 1.0
    while time.monotonic() < deadline:
        app.processEvents(); page._drain_executor(); time.sleep(0.01)
        assert owner.phase == "ACTIVE" and owner.active_key == key
    assert view.background.text() == "Clear Raw BG"
    np.testing.assert_allclose(_shown(page), np.zeros((12, 16)), atol=1e-3)


def _no_refusal(page) -> None:
    shown = " ".join((page._shell.scientific.status.text(), page._context_controller.viewer_2d_diagnostic,
                      page._notice_text)).lower()
    assert not any(word in shown for word in ("refus", "error", "mismatch", "does not fit")), shown


@pytest.mark.parametrize("control", ("show_all", "auto_last", "plot_mode"))
def test_raw_background_survives_viewer_navigation_controls(viewer, control):
    page, app, root, stacks = viewer
    view = page._shell.scientific
    _open(page, app, root / "a.h5")
    view.background.click()
    _wait(page, app, lambda: view.background.text() == "Clear Raw BG")
    key = page._background_owner.active_key
    if control == "plot_mode":
        view.plot_mode.setCurrentText("Overlay")
    else:
        getattr(page._shell.browser, control).click()
    app.processEvents()
    assert page._background_owner.active_key == key
    _step(page, app, 1)
    background = stacks["a.h5"][0].astype(np.float64)
    _settle_drawn(page, app, stacks["a.h5"][1] - background)


def test_raw_background_stays_across_frames_and_files_and_skips_other_shapes(viewer):
    page, app, root, stacks = viewer
    view, owner = page._shell.scientific, page._background_owner
    a, c, d = stacks["a.h5"], stacks["c.h5"], stacks["d.h5"]
    _open(page, app, root / "a.h5")
    _settle_drawn(page, app, a[0].astype(np.float64))
    assert view.background.text() == "Set Raw BG" and "every frame and file" in view.background.toolTip()

    QtTest.QTest.mouseClick(view.background, QtCore.Qt.LeftButton)
    _wait(page, app, lambda: owner.phase == "ACTIVE" and view.background.text() == "Clear Raw BG",
          "Set Raw BG")
    background = a[0].astype(np.float64)
    _settle_drawn(page, app, np.zeros_like(background))
    charged = owner.active_bytes

    # another frame of the same file: drawn minus the background, which stays set
    _step(page, app, 1)
    _settle_drawn(page, app, a[1] - background)
    assert owner.phase == "ACTIVE" and view.background.text() == "Clear Raw BG"
    # the same frame keeps the same displayed array (no new upload per refresh), in the one charged frame
    displayed, drawn = owner.projection()[1][0][1], view.raw.canvas.raw_image
    page._refresh_shell(); app.processEvents(); page._refresh_shell()
    assert owner.projection()[1][0][1] is displayed and view.raw.canvas.raw_image is drawn
    assert owner.active_bytes == charged

    # another file of the same shape: still subtracted
    _open(page, app, root / "c.h5")
    _settle_drawn(page, app, c[0] - background)
    assert owner.phase == "ACTIVE" and view.background.text() == "Clear Raw BG"
    assert "does not fit" not in view.background.toolTip()

    # a file of another shape: drawn as it is, no refusal, the background stays set
    _open(page, app, root / "d.h5")
    _settle_drawn(page, app, d[0].astype(np.float64))
    assert owner.phase == "ACTIVE" and view.background.text() == "Clear Raw BG"
    _no_refusal(page)
    assert "The background (12 × 16) does not fit this frame (10 × 16)" in view.background.toolTip()

    # back to a fitting file: subtracted again
    _open(page, app, root / "a.h5")
    _settle_drawn(page, app, a[0] - background)
    assert "does not fit" not in view.background.toolTip()
    _step(page, app, 1)
    _settle_drawn(page, app, a[1] - background)
    assert owner.active_bytes == charged

    # Clear Raw BG while a frame of another shape is shown: let go of, the frame still drawn as it is
    _open(page, app, root / "d.h5")
    _settle_drawn(page, app, d[0].astype(np.float64))
    QtTest.QTest.mouseClick(view.background, QtCore.Qt.LeftButton)
    _wait(page, app, lambda: owner.phase == "RELEASED" and view.background.text() == "Set Raw BG",
          "Clear Raw BG")
    _settle_drawn(page, app, d[0].astype(np.float64))
    assert owner.active_bytes == 0 and owner.projection() is None
    assert "does not fit" not in view.background.toolTip()
    _no_refusal(page)

    # and the fitting file is drawn as it is again
    _open(page, app, root / "a.h5")
    _settle_drawn(page, app, a[0].astype(np.float64))
    assert view.background.text() == "Set Raw BG"
