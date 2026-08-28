"""Shared pytest setup for xdart.

Keep pyqtgraph on the same Qt binding as the generated UI modules before test
modules import ``pyqtgraph.Qt`` directly.
"""

import gc
import os
import sys
import tempfile

import pytest


_GC_WAS_ENABLED = gc.isenabled()
# Disable the threshold-driven collector as soon as pytest imports this
# package-local conftest.  Reference counting remains active; the harness
# below owns every explicit cyclic collection on QApplication's thread.
gc.disable()

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_API", "PySide6")
os.environ.setdefault("MPLBACKEND", "Agg")

# Isolate GUI session persistence so tests neither read nor overwrite the
# maintainer's real session state.
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
def _xdart_qt_harness():
    """Own QApplication strongly and make cyclic GC main-thread explicit."""

    from tests.xdart.qt_test_harness import QtTestHarness

    harness = QtTestHarness.create()
    try:
        yield harness
    finally:
        # Dependent session finalizers have completed before this owner retires.
        if _GC_WAS_ENABLED:
            gc.enable()


@pytest.fixture(autouse=True)
def _qt_test_boundary(_xdart_qt_harness):
    """Retire one test's Qt roots, then collect its cycles on the GUI thread."""

    baseline = _xdart_qt_harness.top_level_snapshot()
    yield
    _xdart_qt_harness.retire_new_top_levels(baseline)
    _xdart_qt_harness.collect()


@pytest.fixture(scope="session", autouse=True)
def _qt_session_teardown(_xdart_qt_harness):
    """Session-end thread/handle cleanup — the SAFE subset only.

    Runs while the interpreter is fully alive: close surviving top-level
    widgets (their closeEvent handlers stop workers/timers) and close the
    process-wide H5FilePool (never leave HDF5 handle finalization to
    interpreter-exit ordering against Qt teardown).

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
    # 1. Close surviving top-level widgets (runs closeEvent shutdown hooks;
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
    # 2. Close the process-wide H5 read pool while h5py is fully alive.
    try:
        from xrd_tools.session import get_pool
        get_pool().close_all()
    except Exception:
        pass
    # 3. Python-side garbage only (no Qt event delivery).
    try:
        _xdart_qt_harness.collect()
    except Exception:
        pass


# --------------------------------------------------------------------------
# Skip PySide6's pathological interpreter-shutdown teardown of the accumulated
# Qt object graph.
#
# GUI tests create parentless Qt/pyqtgraph helper widgets whose DeferredDelete
# events cannot be safely mass-drained under PySide6.  Letting the accumulated
# object graph fall through normal interpreter finalization has historically
# caused a long O(N^2) shutdown or a native crash.
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
