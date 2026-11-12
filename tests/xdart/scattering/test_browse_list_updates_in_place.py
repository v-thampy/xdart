"""C-9 of NEXT-1210-RESPONSIVENESS: the Browse list updates in place when files arrive.

His report (dev, 2026-11-12, with XDART_VIEWER_DEBUG=1): "i have this output continuously in 2d viewer mode even when
i'm not scrolling", the viewer_artifact_projection line, on a folder of about 9,000 raw images while files were
arriving ("Yes, files were arriving"). The page's catalog poll found the folder changed and the Browse list was
cleared and rebuilt whole on the GUI thread, losing its current row, so its selection sweep ran again. His pick:
"Add it as C-9 (Recommended)": rows change in place, the current row stays, and the selection is swept only when
the list's own selection differs from what the page projects.
"""
from __future__ import annotations

import time

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent

_RAW_BYTES = 195 * 487 * 4  # one Pilatus 100K frame of 32-bit counts, as his .raw files are
_MARKER = QtCore.Qt.ItemDataRole.UserRole + 77


def _raw(path):
    with open(path, "wb") as handle:
        handle.truncate(_RAW_BYTES)


def test_a_file_arriving_keeps_the_current_row_and_sweeps_no_selection(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDART_VIEWER_DEBUG", "1")  # the line he saw says each selection sweep
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    folder = tmp_path / "images"
    folder.mkdir()
    for index in range(200):
        _raw(folder / f"s_{index:04d}.raw")
    target = str(folder / "s_0150.raw")
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(processing_mode="2D Viewer", project_root=str(folder),
                                         save_path=str(folder))),
        lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter())
    page.resize(1200, 800)
    page.show()
    browser, controller = page._shell.browser, page._context_controller

    def wait(done, seconds=10.0):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            app.processEvents()
            page._drain_executor()
            if done():
                return True
            time.sleep(.005)
        return False

    try:
        page._set_browser_directory(str(folder), explicit=True)
        page._open_viewer_2d_path(target)
        assert wait(lambda: controller.viewer_2d_frame is not None and browser.scans.count() == 201)
        assert wait(lambda: browser.scans.currentItem() is not None
                    and browser.scans.currentItem().data(QtCore.Qt.ItemDataRole.UserRole) == target)
        browser.scans.currentItem().setData(_MARKER, "kept")
        capsys.readouterr()
        _raw(folder / "s_9999.raw")  # a frame arrives, as the detector writes them
        assert wait(lambda: (page._poll_browser_catalog(), browser.scans.count() == 202)[1])
        wait(lambda: False, .3)
        current = browser.scans.currentItem()
        assert current is not None and current.data(QtCore.Qt.ItemDataRole.UserRole) == target
        assert current.data(_MARKER) == "kept"  # the same row: the list was not cleared and rebuilt
        assert "viewer_artifact_projection" not in capsys.readouterr().out  # no selection sweep
    finally:
        assert wait(lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
        page.deleteLater()
        app.processEvents()
