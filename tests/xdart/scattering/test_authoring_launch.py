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

from xdart.gui.tabs.scattering import experiment_authoring as authoring
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.operation_values import OperationIdentity, OperationTerminalStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xrd_tools.io.image import load_mask
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent


@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize("source_kind", ["tiff", "hdf"])
def test_real_drawmask_child_saves_and_cleans(
    tmp_path, monkeypatch, source_kind, portable,
):
    # Use the native launcher and actual pyFAI window. Only automate its Save
    # button at event-loop entry; no fake Popen, image reader or publisher.
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text('''
from silx.gui import qt
original_exec = qt.QApplication.exec

def automated_exec(app):
    def save():
        for widget in app.topLevelWidgets():
            if widget.windowTitle() == "pyFAI drawmask":
                widget.saveAndClose()
                return
        app.exit(98)
    qt.QTimer.singleShot(0, save)
    qt.QTimer.singleShot(10000, lambda: app.exit(99))
    return original_exec()
qt.QApplication.exec = automated_exec
''')
    monkeypatch.setenv("PYTHONPATH", str(hooks) + os.pathsep + str(Path(__file__).resolve().parents[3] / "src"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    # Exercise the Windows filesystem branch on POSIX too. On Windows these
    # capabilities really are absent, and the same test runs in locked Pixi CI.
    if portable:
        monkeypatch.setattr(authoring, "_WINDOWS", True)
        monkeypatch.setattr(authoring, "_DIR_FD_PUBLICATION", False)
        monkeypatch.delattr(os, "fchmod", raising=False)
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
    request = authoring.prepare_mask_request(selected)
    terminal = authoring.run_mask(request, OperationIdentity(1), Event(),
                                 lambda *_: None, lambda _: True)
    assert terminal.status is OperationTerminalStatus.RETURNED, terminal.diagnostic
    assert authoring.mask_terminal_result_valid(terminal, request)
    assert terminal.payload.exit_code == 0
    assert not terminal.payload.recovery_path
    np.testing.assert_array_equal(load_mask(request.final_path), np.zeros(pixels.shape, dtype=bool))
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".xdart-mask-*"))
    saved = Path(request.final_path).read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        authoring.prepare_mask_request(selected)
    assert Path(request.final_path).read_bytes() == saved
    if portable and source_kind == "tiff":
        Path(request.final_path).unlink()
        def occupy_final(stage, *_):
            if stage == "publish":
                Path(request.final_path).write_bytes(b"another mask")
        terminal = authoring.run_mask(request, OperationIdentity(2), Event(),
                                     occupy_final, lambda _: True)
        assert terminal.status is OperationTerminalStatus.FAILED
        assert "already exists" in terminal.diagnostic
        assert Path(request.final_path).read_bytes() == b"another mask"
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
