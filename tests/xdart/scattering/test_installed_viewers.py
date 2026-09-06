"""Exercise the installed GUI entry points, with isolated user state and files."""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest


@pytest.mark.parametrize("viewer", ["nexpy", "silx"])
def test_installed_viewer_loads_hdf5_readonly_and_exits(viewer, tmp_path):
    importlib.metadata.version(viewer)  # A GUI install must supply both apps.
    target = tmp_path / "viewer test.nexus"
    with h5py.File(target, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("counts", data=np.arange(12).reshape(3, 4))
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    for key in ("QT_API", "PYQTGRAPH_QT_LIB", "MPLBACKEND"):
        environment.pop(key, None)
    environment.update({key: str(tmp_path) for key in (
        "TMPDIR", "MPLCONFIGDIR", "IPYTHONDIR", "JUPYTER_CONFIG_DIR",
        "JUPYTER_RUNTIME_DIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
    )})
    environment["XDART_SESSION_FILE"] = str(tmp_path / "session.json")
    # Use the distribution's actual console entry point. Only its event-loop
    # lifetime and state location are adapted: CLI parsing and file loading
    # stay real. Inspect the populated application model, not just imports.
    script = r'''
import importlib.metadata
from pathlib import Path
import sys
import time
from PySide6 import QtCore, QtWidgets

viewer, filename, scratch = sys.argv[1:]
Path.home = classmethod(lambda cls: Path(scratch))
QtCore.QSettings.setPath(QtCore.QSettings.IniFormat,
                        QtCore.QSettings.UserScope, scratch)
QtCore.QSettings.setPath(QtCore.QSettings.NativeFormat,
                        QtCore.QSettings.UserScope, scratch)
original_exec = QtWidgets.QApplication.exec
checked = []
if viewer == "nexpy":
    app = QtWidgets.QApplication([])
    startup = QtCore.QTimer()
    def acknowledge_dialogs():
        for widget in app.topLevelWidgets():
            if not isinstance(widget, QtWidgets.QMessageBox):
                continue
            if widget.text() == "New plugins are available":
                widget.accept()
            elif widget.text() == "Are you sure you want to quit NeXpy?":
                widget.button(QtWidgets.QMessageBox.Ok).click()
    startup.timeout.connect(acknowledge_dialogs)
    startup.start(50)

def bounded_exec(self):
    deadline = time.monotonic() + 15
    timer = QtCore.QTimer()
    def inspect():
        try:
            if viewer == "nexpy":
                from nexpy.gui.mainwindow import MainWindow
                window = next(w for w in self.topLevelWidgets()
                              if isinstance(w, MainWindow))
                loaded = [window.tree[key] for key in window.tree
                          if window.tree[key].nxfilename == filename]
                assert len(loaded) == 1
                data = loaded[0]["entry/counts"]
                assert data.shape == (3, 4)
                assert data.nxdata.sum() == 66
                assert loaded[0].nxfilemode == "r"
            else:
                from silx.app.view.Viewer import Viewer
                from silx.gui.hdf5 import Hdf5TreeView
                window = next(w for w in self.topLevelWidgets()
                              if isinstance(w, Viewer))
                model = window.findChild(Hdf5TreeView).findHdf5TreeModel()
                data = model.data(model.index(0, 0), model.H5PY_OBJECT_ROLE)
                assert data is not None
                assert data["entry/counts"].shape == (3, 4)
                assert data["entry/counts"][()].sum() == 66
                assert data.file.mode == "r"
            checked.append(True)
            timer.stop()
            self.quit()
        except (AssertionError, StopIteration):
            if time.monotonic() >= deadline:
                timer.stop()
                self.quit()
    timer.timeout.connect(inspect)
    timer.start(50)
    result = original_exec()
    assert checked, "viewer did not expose the requested read-only dataset"
    print("VIEWER_FILE_LOADED", viewer, file=sys.__stdout__, flush=True)
    return result

QtWidgets.QApplication.exec = bounded_exec
sys.argv = [viewer] + (["view"] if viewer == "silx" else []) + [filename]
distribution = importlib.metadata.distribution(viewer)
next(ep for ep in distribution.entry_points if ep.name == viewer).load()()
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, viewer, str(target), str(tmp_path)],
        env=environment, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"VIEWER_FILE_LOADED {viewer}" in completed.stdout
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before
