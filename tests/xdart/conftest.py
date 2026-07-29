"""Shared pytest setup for xdart.

Keep pyqtgraph on the same Qt binding as the generated UI modules before test
modules import ``pyqtgraph.Qt`` directly.
"""

import gc
import os
import sys
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

# Isolate application PREFERENCES the same way.  The theme/typography owner is
# the only QSettings reader in the GUI and it honours XDART_SETTINGS_FILE, so
# this one line keeps every test -- and every standalone probe that sets it --
# out of the maintainer's real `com.xdart.xdart` preferences.  QSettings cannot
# be redirected after the fact on macOS (the two-argument constructor ignores
# setDefaultFormat and always resolves to the native plist), which is exactly
# why the override lives at the accessor instead.
os.environ.setdefault(
    "XDART_SETTINGS_FILE",
    os.path.join(tempfile.mkdtemp(prefix="xdart_test_settings_"),
                 "xdart.ini"),
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


# --------------------------------------------------------------------------
# Skip PySide6's pathological interpreter-shutdown teardown of the accumulated
# Qt object graph.
#
# Every GUI test builds a ``staticWidget`` whose sub-widgets (pyqtgraph
# ViewBoxMenus, combobox popups, context QMenus, card QFrames, ...) create
# ~190 PARENTLESS top-level widgets.  ``widget.close() + deleteLater()`` cannot
# reap them: no Qt event loop runs during the tests, so the posted
# ``DeferredDelete`` events are never delivered (the per-test
# ``qapp.processEvents()`` drain does not flush level-0 DeferredDelete), and the
# widgets are parentless so nothing cascade-deletes them.  All ~190 per test
# accumulate for the whole file (>30k live QObjects, GBs of RSS) and are
# destroyed in a single avalanche at ``Py_Finalize`` ->
# ``PySide::destroyQCoreApplication`` -> ``visitAllPyObjects``.  Each
# destruction walks PySide6's GLOBAL signal-connection QHash
# (``onPysideReceiverSlotDestroyed``), so the mass teardown is O(N^2): measured
# ~286 s of pure post-session hang for test_controls_panel_v2.py (body ~85 s).
#
# We CANNOT reduce N by reaping per test: a ``sendPostedEvents(DeferredDelete)``
# drain is banned (it segfaulted linux CI mid-run -- see _qt_session_teardown),
# and a blind ``shiboken6.delete`` of the parentless popup graph double-frees
# (verified: SIGSEGV).  So we do what scripts/ci_pytest.py already does on linux
# CI (and what pyqtgraph's ``pg.exit()`` does): once pytest has finished — the
# terminal summary printed and any JUnit XML written in pytest_sessionfinish —
# hard-exit with pytest's own verdict, skipping the Py_Finalize object-graph
# walk where the hang (macOS) / segfault (linux) lives.  The safe session
# cleanups above (thread waits, H5 pool close) run as fixture finalizers BEFORE
# pytest_unconfigure, so they are preserved.
#
# Opt out with XDART_KEEP_TEARDOWN=1 (e.g. debugging teardown itself).
def pytest_sessionfinish(session, exitstatus):
    session.config._xdart_exit_status = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if os.environ.get("XDART_KEEP_TEARDOWN"):
        return
    status = int(getattr(config, "_xdart_exit_status", 0))
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(status)
