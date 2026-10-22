"""Deterministic Qt lifecycle boundaries for the xdart pytest process.

Python's cyclic collector may run on whichever Python thread crosses its
allocation threshold.  That is unsafe for cycles containing GUI-thread Qt
objects: finalizing one from a loader worker can stop a ``QTimer`` from the
wrong thread and leave Qt's timer dispatcher holding a stale recipient.

The test process therefore disables *automatic* cyclic collection and calls
``collect`` only through this main-thread-owned harness.  Reference counting
is unaffected.  The harness also closes top-level widgets created by one test
before collecting its unreachable Python cycles.  DeferredDelete delivery at
the per-test boundary is the default (``XDART_QT_DELIVER_DELETES=0`` opts
out); the session-end backlog drain remains banned because of the hazard
documented in ``tests/xdart/conftest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import threading

import shiboken6
from PySide6 import QtCore, QtWidgets


def _cpp_identity(widget: QtWidgets.QWidget) -> int:
    """Return the stable identity of the wrapped C++ QObject."""

    return int(shiboken6.getCppPointer(widget)[0])


@dataclass(frozen=True, slots=True)
class QtTestHarness:
    """One strongly-owned QApplication and its only GC boundary."""

    app: QtWidgets.QApplication
    main_thread_ident: int

    @classmethod
    def create(cls) -> "QtTestHarness":
        """Create or retain the process QApplication on the main thread."""

        main_ident = threading.main_thread().ident
        if main_ident is None or threading.get_ident() != main_ident:
            raise RuntimeError("Qt test harness must start on the main thread")
        app = (
            QtWidgets.QApplication.instance()
            or QtWidgets.QApplication([])
        )
        harness = cls(app=app, main_thread_ident=main_ident)
        harness.assert_main_thread()
        return harness

    def assert_main_thread(self) -> None:
        """Refuse Qt cleanup or Python collection from a worker thread."""

        if threading.get_ident() != self.main_thread_ident:
            raise RuntimeError("Qt test cleanup must run on the main thread")
        if QtCore.QThread.currentThread() != self.app.thread():
            raise RuntimeError("Qt test cleanup must run on QApplication.thread()")

    def top_level_snapshot(self) -> frozenset[int]:
        """Record stable C++ top-level identities without retaining wrappers."""

        self.assert_main_thread()
        return frozenset(
            _cpp_identity(widget)
            for widget in QtWidgets.QApplication.topLevelWidgets()
        )

    def retire_new_top_levels(self, baseline: frozenset[int]) -> None:
        """Close top-level widgets introduced after ``baseline``.

        Module/session fixtures that predate the function boundary remain
        alive.  Ordinary event processing lets close handlers stop their
        workers and timers, but no explicit ``DeferredDelete`` delivery occurs.
        """

        self.assert_main_thread()
        for widget in tuple(QtWidgets.QApplication.topLevelWidgets()):
            if _cpp_identity(widget) in baseline:
                continue
            try:
                widget.close()
            except RuntimeError:
                pass
        self.app.processEvents(
            QtCore.QEventLoop.ProcessEventsFlag.AllEvents,
        )

    def deliver_deferred_deletes(self) -> None:
        """Delete the objects this test scheduled with ``deleteLater()``.

        Qt delivers a deletion posted outside any event loop only when a later
        nested loop runs, so without this one process keeps every such object
        until some test destroys the whole backlog at once (QT-TEST-LIFECYCLE).
        The test boundary delivers pending deletions on the GUI thread after
        each test's retirement unless ``XDART_QT_DELIVER_DELETES=0``.  It does
        not delete windows merely because they were closed, or bound the number
        of live widgets.
        """

        self.assert_main_thread()
        QtCore.QCoreApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete,
        )

    def collect(self) -> int:
        """Collect Python cycles on the QApplication's owning thread."""

        self.assert_main_thread()
        return gc.collect()
