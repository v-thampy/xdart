"""Terminal discriminators for pending Save context and scene-first retirement.

Use the frozen real right-click, format and filename helpers; no exporter/dialog doubles.
"""
import numpy as np
import pytest
import shiboken6
from pyqtgraph.Qt import QtCore, QtTest

from tests.xdart.test_shared_plot_export import (
    CSV, _answer, _click_format, _export, _open, _plot, hooked, qapp,  # noqa: F401
)


@pytest.mark.parametrize("cancel_latest", [False, True])
def test_pending_saves_keep_their_own_filter_and_callback_context(
        qapp, hooked, tmp_path, _xdart_qt_harness, cancel_latest):
    host, view, plot, _curves = _plot(qapp, [(range(3), [1, 2, 3])])
    try:
        dialog = _open(qapp, view, plot.getViewBox())
        _click_format(qapp, dialog, CSV)
        first = _export(qapp, dialog)
        first.selectNameFilter("*.tsv")
        second = _export(qapp, dialog)
        assert first is not second and first.isVisible() and second.isVisible()
        if cancel_latest:
            QtTest.QTest.keyClick(second, QtCore.Qt.Key.Key_Escape)
            _xdart_qt_harness.deliver_deferred_deletes()
            assert not shiboken6.isValid(second)
        _answer(qapp, first, tmp_path, "first")
        assert hooked == [], "the older Save must not dereference a different, possibly deleted dialog"
        assert (tmp_path / "first.tsv").exists(), "extension belongs to the Save that selected the filename"
        np.testing.assert_array_equal(
            np.loadtxt(tmp_path / "first.tsv", delimiter=",", skiprows=1),
            np.column_stack([np.arange(3.0), [1, 2, 3]]))
        if not cancel_latest:
            _xdart_qt_harness.deliver_deferred_deletes()
            _answer(qapp, second, tmp_path, "second")
            assert hooked == [] and (tmp_path / "second.csv").exists()
        assert sorted(path.name for path in tmp_path.iterdir()) == (
            ["first.tsv"] if cancel_latest else ["first.tsv", "second.csv"])
    finally:
        host.deleteLater()
        _xdart_qt_harness.deliver_deferred_deletes()


def test_scene_destruction_retires_export_and_pending_save_before_host(
        qapp, hooked, tmp_path, _xdart_qt_harness):
    host, view, plot, _curves = _plot(qapp, [(range(3), [1, 2, 3])])
    try:
        dialog = _open(qapp, view, plot.getViewBox())
        _click_format(qapp, dialog, CSV)
        pending = _export(qapp, dialog)
        select_box = dialog.selectBox
        scene = view.scene()
        dialog.close()
        scene.deleteLater()
        _xdart_qt_harness.deliver_deferred_deletes()
        assert shiboken6.isValid(view), "scene-first destruction is distinct from host deletion"
        assert not any(shiboken6.isValid(item) for item in (dialog, pending, select_box))
        assert hooked == [] and list(tmp_path.iterdir()) == []
    finally:
        host.deleteLater()
        _xdart_qt_harness.deliver_deferred_deletes()
