"""Shared pytest setup for xdart.

Keep pyqtgraph on the same Qt binding as the generated UI modules before test
modules import ``pyqtgraph.Qt`` directly.
"""

import gc
import os
import tempfile

import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_API", "PySide6")
os.environ.setdefault("MPLBACKEND", "Agg")

# Isolate session persistence: staticWidget restores ~/.xdart/session.json at
# construction and SAVES it at close() -- without this, GUI-test fixtures
# polluted the user's real session (and inherited the user's state, making
# tests order/machine dependent).
os.environ.setdefault(
    "XDART_SESSION_FILE",
    os.path.join(tempfile.mkdtemp(prefix="xdart_test_session_"),
                 "session.json"),
)


@pytest.fixture(scope="session", autouse=True)
def _qt_session_teardown():
    """Deterministic Qt/HDF5 teardown at session end.

    The suite has no session-owned QApplication: the first GUI module creates
    the singleton and nothing ever destroys it, tests never run exec() (so
    deleteLater is never delivered -- processEvents() does NOT deliver
    QEvent.DeferredDelete), and ~1600 tests' residue (pending deferred
    deletions, orphaned slow-close QThreads, the open-handle H5FilePool) used
    to ride into interpreter shutdown, where shiboken destroying the giant
    C++ graph during Python finalization is the canonical PySide6 exit
    SIGSEGV (linux CI exit 139 AFTER a green summary).  Tear it all down
    HERE, while the interpreter is fully alive.  Every step is best-effort:
    a teardown helper must never fail the suite.
    """
    yield
    try:
        from PySide6 import QtCore, QtWidgets
    except Exception:
        return
    app = QtWidgets.QApplication.instance()

    def _drain():
        # Two passes: deliver pending deleteLater, let their destructors queue
        # more, deliver again.
        for _ in range(2):
            QtCore.QCoreApplication.sendPostedEvents(
                None, QtCore.QEvent.Type.DeferredDelete)
            if app is not None:
                app.processEvents()

    # 1. Bounded-wait the deliberately orphaned slow-close QThreads so a
    #    still-running native thread isn't destroyed at module teardown
    #    (Qt qFatal) -- then drop the retention lists.
    try:
        from xdart.gui.tabs.static_scan import h5viewer as _h5v
        from xdart.gui.tabs.static_scan import static_scan_widget as _ssw
        for lst in (getattr(_ssw, "_ORPHANED_STITCH_THREADS", []),
                    getattr(_h5v, "_ORPHANED_FILE_THREADS", []),
                    getattr(_h5v, "_ORPHANED_LOAD_WORKERS", [])):
            for th in list(lst):
                try:
                    if hasattr(th, "isRunning") and th.isRunning():
                        th.wait(5000)
                except Exception:
                    pass
            try:
                lst.clear()
            except Exception:
                pass
    except Exception:
        pass
    # 2. Close every surviving top-level widget and actually deliver the
    #    deferred deletions (what a running event loop would have done).
    try:
        if app is not None:
            for w in QtWidgets.QApplication.topLevelWidgets():
                try:
                    w.close()
                except Exception:
                    pass
            _drain()
    except Exception:
        pass
    # 3. Close the process-wide H5 read pool while h5py is fully alive --
    #    never leave HDF5 handle finalization to interpreter-exit ordering.
    try:
        from xdart.utils.h5pool import get_pool
        get_pool().close_all()
    except Exception:
        pass
    # 4. Collect Python-side garbage NOW (widget wrappers dropped above), then
    #    deliver any deleteLater that the destructors queued.
    try:
        gc.collect()
        _drain()
    except Exception:
        pass
