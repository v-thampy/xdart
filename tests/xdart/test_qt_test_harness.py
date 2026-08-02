from __future__ import annotations

import gc
import threading

from PySide6 import QtWidgets


_GC_ENABLED_DURING_TEST_MODULE_IMPORT = gc.isenabled()


def test_qt_harness_disables_automatic_cyclic_collection(
    _xdart_qt_harness,
) -> None:
    assert not _GC_ENABLED_DURING_TEST_MODULE_IMPORT
    assert not gc.isenabled()
    assert QtWidgets.QApplication.instance() is _xdart_qt_harness.app
    _xdart_qt_harness.assert_main_thread()


def test_qt_harness_retires_only_test_owned_top_levels(
    _xdart_qt_harness,
) -> None:
    retained = QtWidgets.QWidget()
    retained.show()
    baseline = _xdart_qt_harness.top_level_snapshot()
    transient = QtWidgets.QWidget()
    transient.show()

    _xdart_qt_harness.retire_new_top_levels(baseline)

    assert retained.isVisible()
    assert not transient.isVisible()
    retained.close()


def test_qt_harness_refuses_worker_thread_collection(
    _xdart_qt_harness,
) -> None:
    errors: list[BaseException] = []

    def collect() -> None:
        try:
            _xdart_qt_harness.collect()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=collect, name="qt-harness-oracle")
    worker.start()
    worker.join(timeout=5.0)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "main thread" in str(errors[0])
