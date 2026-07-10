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
    """Session-end thread/handle cleanup — the SAFE subset only.

    Runs while the interpreter is fully alive: bounded-wait the deliberately
    orphaned slow-close QThreads (a still-running native QThread destroyed at
    module teardown is a Qt qFatal), close surviving top-level widgets (their
    closeEvent handlers stop workers/timers), and close the process-wide
    H5FilePool (never leave HDF5 handle finalization to interpreter-exit
    ordering against Qt teardown).

    Deliberately does NOT deliver the session's accumulated DeferredDelete
    backlog: a previous version drained it here
    (sendPostedEvents(None, DeferredDelete)) and that mass delivery ITSELF
    segfaulted on linux CI (faulthandler pinned the crash to the drain, PR
    run 29104018293) — ~1600 tests' worth of delete-order hazards is the same
    minefield whether walked at Py_Finalize or here, and here it fires
    DURING the last test's teardown where scripts/ci_pytest.py's hard exit
    cannot skip it.  The interpreter-shutdown crash is the wrapper's job;
    this fixture only prevents the qFatal/HDF5 variants.  Every step is
    best-effort: a teardown helper must never fail the suite.
    """
    yield
    try:
        from PySide6 import QtWidgets
    except Exception:
        return
    app = QtWidgets.QApplication.instance()
    # 1. Bounded-wait the orphaned slow-close QThreads, then drop the lists.
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
    # 2. Close surviving top-level widgets (runs closeEvent shutdown hooks;
    #    no event delivery — see the docstring).
    try:
        if app is not None:
            for w in QtWidgets.QApplication.topLevelWidgets():
                try:
                    w.close()
                except Exception:
                    pass
    except Exception:
        pass
    # 3. Close the process-wide H5 read pool while h5py is fully alive.
    try:
        from xdart.utils.h5pool import get_pool
        get_pool().close_all()
    except Exception:
        pass
    # 4. Python-side garbage only (no Qt event delivery).
    try:
        gc.collect()
    except Exception:
        pass
