from __future__ import annotations

from pathlib import Path
from threading import Thread, current_thread

from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor, _StandardRun
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity


def _identity() -> RunIdentity:
    return RunIdentity(1, "f" * 64)


def test_dead_worker_cleanup_retry_never_runs_on_the_qt_thread() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert QtWidgets.QApplication.instance().thread() is app.thread()

    class OneShotCloseFailure:
        def __init__(self) -> None:
            self.threads = []

        def close(self) -> None:
            self.threads.append(current_thread())
            if len(self.threads) == 1:
                raise RuntimeError("first close fails")

    source = OneShotCloseFailure()
    run = _StandardRun(
        None, _identity(), None, source, None, None, Path("unused.nxs")
    )
    executor = StandardRunExecutor()
    executor._active = run
    worker = Thread(target=executor._cleanup, args=(run,), name="initial-cleanup")
    run.worker = worker
    worker.start()
    worker.join(2)
    assert not worker.is_alive()
    assert run.cleanup_status is CleanupStatus.CLEANUP_PENDING

    qt_python_thread = current_thread()
    receipt = executor.close(run.identity)

    assert receipt.cleanup_status is CleanupStatus.CLEANED
    assert len(source.threads) == 2
    assert source.threads[0] is not qt_python_thread
    assert source.threads[1] is not qt_python_thread
