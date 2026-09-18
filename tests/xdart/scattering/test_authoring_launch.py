"""Real pyFAI child startup and visible authoring failures."""
from pathlib import Path
from threading import Event
import os
import time

import h5py
import numpy as np
import pytest
import tifffile
from pyqtgraph.Qt import QtTest, QtWidgets
from silx.io.url import DataUrl
from fabio.edfimage import EdfImage

from xdart.gui.tabs.scattering import experiment_authoring as authoring
from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.operation_values import OperationContextStamp, OperationIdentity, OperationTerminalStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xrd_tools.io.image import load_mask
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


def _windows_permissions(monkeypatch):
    monkeypatch.setattr(authoring, "_WINDOWS", True)
    monkeypatch.setattr(authoring, "_DIR_FD_PUBLICATION", False)
    if os.name != "nt":
        real_fchmod = os.fchmod

        def windows_fchmod(fd, mode):
            # Python 3.13 exposes fchmod on Windows, but only the read-only
            # attribute changes. Model its mode bits on a real local file.
            real_fchmod(fd, 0o666 if mode & 0o200 else 0o444)

        monkeypatch.setattr(os, "fchmod", windows_fchmod)


@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize("source_kind", ["tiff", "hdf"])
@pytest.mark.parametrize("action", ["save", "close", "failure"])
def test_real_drawmask_child_saves_and_cleans(
    tmp_path, monkeypatch, _xdart_qt_harness, source_kind, portable, action,
):
    # Use the native launcher and actual pyFAI window. Automate save/close/exit
    # at event-loop entry; no fake Popen, image reader or publisher.
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text('''
import os
from pathlib import Path
from silx.gui import qt
original_exec = qt.QApplication.exec

def automated_exec(app):
    def save():
        for widget in app.topLevelWidgets():
            if widget.windowTitle() == "pyFAI drawmask":
                (Path(__file__).parent / "opened").write_text("mask")
                action = os.environ["XDART_TEST_MASK_ACTION"]
                if action == "save":
                    widget.saveAndClose()
                elif action == "close":
                    widget.close()
                else:
                    app.exit(7)
                return
        app.exit(98)
    qt.QTimer.singleShot(0, save)
    qt.QTimer.singleShot(10000, lambda: app.exit(99))
    return original_exec()
qt.QApplication.exec = automated_exec
''')
    monkeypatch.setenv("PYTHONPATH", str(hooks) + os.pathsep + str(Path(__file__).resolve().parents[3] / "src"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("XDART_TEST_MASK_ACTION", action)
    # Keep fchmod available: Windows Python 3.13 supports it without POSIX
    # permission semantics. Native Windows CI uses its real implementation.
    if portable:
        _windows_permissions(monkeypatch)
    pixels = np.arange(80, dtype=np.uint16).reshape(8, 10)
    source = tmp_path / ("Image.tiff" if source_kind == "tiff" else "Image.h5")
    if source_kind == "tiff":
        tifffile.imwrite(source, pixels)
        selected = str(source)
    else:
        with h5py.File(source, "w") as handle:
            handle.create_dataset("/entry/data/data", data=pixels[None])
        selected = DataUrl(file_path=str(source), data_path="/entry/data/data",
                           data_slice=(0,), scheme="silx").path()
    original = source.read_bytes()
    final = source.with_name(source.stem + "-mask.edf")
    EdfImage(data=np.ones(pixels.shape, dtype=np.uint8)).write(str(final))
    saved = final.read_bytes()
    request = authoring.prepare_mask_request(selected, current_mask=str(final))
    slot = OperationSlot()
    try:
        identity = slot.begin_mask(request, OperationContextStamp(0))
        assert identity is not None
        terminal = None
        deadline = time.monotonic() + 25
        while terminal is None and time.monotonic() < deadline:
            update = slot.poll(identity)
            if update is not None:
                terminal = update.terminal
            time.sleep(0.01)
        assert terminal is not None
        assert not slot.owned
    finally:
        slot.close()
    assert (hooks / "opened").read_text() == "mask"
    assert authoring.mask_terminal_result_valid(terminal, request)
    if action == "save":
        assert terminal.status is OperationTerminalStatus.RETURNED, terminal.diagnostic
        assert terminal.payload.exit_code == 0
        np.testing.assert_array_equal(load_mask(final), np.zeros(pixels.shape, dtype=bool))
    else:
        expected = (OperationTerminalStatus.CANCELLED if action == "close"
                    else OperationTerminalStatus.FAILED)
        assert terminal.status is expected, terminal.diagnostic
        assert terminal.payload.exit_code == (7 if action == "failure" else 0)
        assert final.read_bytes() == saved
        store = RunIntentStore(RunIntent(project_root=str(tmp_path), mask_file=str(final)))
        page = ScatteringWorkspace(intents=store, lifecycle=ScatteringCoordinator(),
                                   sources=FilesystemSourceAdapter())
        before = store.snapshot()
        app = _xdart_qt_harness.app
        try:
            page._authored_assets.adopt_operation(
                "mask", request, page._operation_context_stamp(), identity)
            assert page._consume_authored_asset_update(update)
            app.processEvents()
            dialogs = page.findChildren(QtWidgets.QMessageBox)
            assert len(dialogs) == (0 if action == "close" else 1)
            assert page._authored_asset_dialog is None
            assert store.snapshot() == before
            if action == "close":
                assert not terminal.diagnostic
        finally:
            for dialog in page.findChildren(QtWidgets.QMessageBox):
                dialog.close()
            page.close_workspace()
            page.deleteLater()
            app.processEvents()
    assert not terminal.payload.recovery_path
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".xdart-mask-*"))


@pytest.mark.parametrize("failure", ["preparation", "worker"])
@pytest.mark.parametrize("asset", ["poni", "mask"])
def test_authoring_failure_is_visible_and_logged(
    tmp_path, monkeypatch, _xdart_qt_harness, caplog, asset, failure,
):
    app = _xdart_qt_harness.app
    source = tmp_path / "missing.tiff"
    if failure == "worker":
        source.write_bytes(b"not a TIFF")
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(project_root=str(tmp_path))),
        lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(),
        authoring_source_chooser=lambda *_: str(source),
    )
    try:
        (page._calibrate_action if asset == "poni" else page._mask_action)()
        deadline = time.monotonic() + 10
        while not page.findChildren(QtWidgets.QMessageBox) and time.monotonic() < deadline:
            QtTest.QTest.qWait(10)
        app.processEvents()
        dialogs = page.findChildren(QtWidgets.QMessageBox)
        assert len(dialogs) == 1
        assert dialogs[0].isVisible()
        assert dialogs[0].informativeText()
        assert ("not started" if failure == "preparation" else "tool failed") in caplog.text
        assert page._workspace_operations.current_identity is None
    finally:
        for dialog in page.findChildren(QtWidgets.QMessageBox):
            dialog.close()
        page.close_workspace()
        page.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("portable", [False, True])
def test_real_calibration_child_opens_and_exits(tmp_path, monkeypatch, portable):
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text('''
from pathlib import Path
from silx.gui import qt
scratch = Path(__file__).parent
qt.QSettings.setPath(qt.QSettings.IniFormat, qt.QSettings.UserScope, str(scratch))
original_exec = qt.QApplication.exec

def automated_exec(app):
    def finish():
        from pyFAI.gui.CalibrationWindow import CalibrationWindow
        if any(isinstance(w, CalibrationWindow) and w.isVisible()
               for w in app.topLevelWidgets()):
            (scratch / "opened").write_text("calibration")
            app.exit(0)
        else:
            app.exit(98)
    qt.QTimer.singleShot(0, finish)
    qt.QTimer.singleShot(10000, lambda: app.exit(99))
    return original_exec()
qt.QApplication.exec = automated_exec
''')
    monkeypatch.setenv("PYTHONPATH", str(hooks) + os.pathsep + str(Path(__file__).resolve().parents[3] / "src"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    if portable:
        _windows_permissions(monkeypatch)
    source = tmp_path / "calibration.tiff"
    tifffile.imwrite(source, np.arange(80, dtype=np.uint16).reshape(8, 10))
    original = source.read_bytes()
    request = authoring.prepare_calibration_request(str(source))
    terminal = authoring.run_calibration(request, OperationIdentity(1), Event(),
                                        lambda *_: None, lambda _: True)
    assert terminal.status is OperationTerminalStatus.RETURNED, terminal.diagnostic
    assert terminal.payload.exit_code == 0
    assert (hooks / "opened").read_text() == "calibration"
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".xdart-authoring-stderr-*"))
