"""COVERAGE-EXPORT-FOLLOWUP-20261014 Stage B, frozen before the fix: the shared plot context-menu export.

PLOT-EXPORT-TARGET-TYPE and PLOT-EXPORT-EMPTY-HDF5, driven the way a user meets them: a RectViewBox plot, a right-click
in its viewport, the menu's own "Export..." action and pyqtgraph's Export dialog, real clicks on its target tree and
format list, and the real Save dialog answered by typing a file name and Return (its own fileSelected callback). No
dialog, exporter or serialization double; a raise inside a Qt slot lands in sys.excepthook, which a spy records.

- B1: Entire Scene and a bare ViewBox take only the visual formats, and each data format says why not. A chosen
  format stays chosen while it is available; otherwise the first available one is, with Export/Copy reconciled.
- B2: an empty or image-only plot offers no HDF5 (nor CSV): nothing is written and nothing raises.
- B3: valid CSV keeps the original data and HDF5 the displayed data (log x, downsampled), exactly as upstream writes.
- B4: error-bar-only CSV, ragged curves and image-plus-curve plots export. HDF5 with a cleared curve beside a valid one
  is refused with a reason and no file, while CSV writes the valid curve.
- B5: a pending Save checks the plot it was opened for. Cleared meanwhile, with the Export dialog closed, it is refused
  in that reopened dialog and writes nothing; a changed tree target cannot redirect it; deleting the host while Save is
  open writes nothing and raises nothing.
- B6: no Export dialog before the first Export click, then one per scene however often and from whichever view box it
  opens; PNG and SVG write rendered content and Cancel writes nothing; deleting the host retires the Export and
  Save dialogs and the selection box.
"""
import sys

import h5py
import numpy as np
import pytest
import shiboken6
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

LEFT, NONE = QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier
CSV, HDF5 = "CSV of original plot data", "HDF5 Export: plot (x,y)"
IMAGE, MPL, SVG = "Image File (PNG, TIF, JPG, ...)", "Matplotlib Window", "Scalable Vector Graphics (SVG)"


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def hooked(monkeypatch):
    """Every exception that reaches sys.excepthook: a raise inside a Qt slot lands there."""
    seen = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, value, trace: seen.append(f"{kind.__name__}: {value}"))
    return seen


def _plot(qapp, curves=(), *, image=False, errorbars=None):
    """A shown host holding one PlotWidget on a RectViewBox, with ``curves`` and optional image or error bars."""
    import pyqtgraph as pg
    from xdart.gui.gui_utils import RectViewBox
    host = QtWidgets.QWidget()
    view = pg.PlotWidget(viewBox=RectViewBox())
    QtWidgets.QVBoxLayout(host).addWidget(view)
    plot = view.getPlotItem()
    made = [plot.plot(np.asarray(x, float), np.asarray(y, float)) for x, y in curves]
    if image:
        plot.addItem(pg.ImageItem(np.arange(12.0).reshape(3, 4)))
    if errorbars is not None:
        plot.addItem(pg.ErrorBarItem(**errorbars))
    host.resize(480, 360)
    host.show()
    qapp.processEvents()
    return host, view, plot, made


def _layout(qapp, rows):
    """A shown host whose one scene holds a RectViewBox plot per row of curves and one bare RectViewBox."""
    import pyqtgraph as pg
    from xdart.gui.gui_utils import RectViewBox
    host = QtWidgets.QWidget()
    view = pg.GraphicsLayoutWidget()
    QtWidgets.QVBoxLayout(host).addWidget(view)
    plots = []
    for curves in rows:
        plot = view.addPlot(viewBox=RectViewBox())
        for x, y in curves:
            plot.plot(np.asarray(x, float), np.asarray(y, float))
        plots.append(plot)
        view.nextRow()
    bare = RectViewBox()  # addViewBox() would make pyqtgraph's own ViewBox, not Int's RectViewBox
    view.addItem(bare)
    host.resize(480, 640)
    host.show()
    qapp.processEvents()
    return host, view, plots, bare


def _open(qapp, view, viewbox):
    """Right-click ``viewbox`` in ``view``'s viewport and choose its menu's own Export... action."""
    point = view.mapFromScene(viewbox.sceneBoundingRect().center())
    QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.MouseButton.RightButton, NONE, point)
    qapp.processEvents()
    menu = QtWidgets.QApplication.activePopupWidget()
    assert menu is not None, "the right-click raised no menu"
    action = next(item for item in menu.actions() if item.text().startswith("Export"))
    menu.close()
    action.trigger()
    qapp.processEvents()
    return view.scene().exportDialog


def _click_target(qapp, dialog, gitem):
    """Click the export tree's row for ``gitem`` (a plot or view box, or the scene for Entire Scene)."""
    tree, rows = dialog.ui.itemTree, []

    def walk(item):
        rows.append(item)
        for index in range(item.childCount()):
            walk(item.child(index))

    for index in range(tree.topLevelItemCount()):
        walk(tree.topLevelItem(index))
    (row,) = [item for item in rows if item.gitem is gitem]
    tree.scrollToItem(row)
    QtTest.QTest.mouseClick(tree.viewport(), LEFT, NONE, tree.visualItemRect(row).center())
    qapp.processEvents()
    assert tree.currentItem() is row


def _format(dialog, name):
    formats = dialog.ui.formatList
    (item,) = [formats.item(index) for index in range(formats.count()) if formats.item(index).text() == name]
    return item


def _click_format(qapp, dialog, name):
    formats = dialog.ui.formatList
    QtTest.QTest.mouseClick(formats.viewport(), LEFT, NONE, formats.visualItemRect(_format(dialog, name)).center())
    qapp.processEvents()


def _available(item):
    return bool(item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled)


def _current(dialog):
    item = dialog.ui.formatList.currentItem()
    return None if item is None else item.text()


def _export(qapp, dialog):
    """Click Export; the Save dialog it opens, if any."""
    QtTest.QTest.mouseClick(dialog.ui.exportBtn, LEFT)
    qapp.processEvents()
    save = getattr(dialog.currentExporter, "fileDialog", None)
    return save if save is not None and save.isVisible() else None


def _answer(qapp, save, folder, name):
    """Answer a Save dialog as a user does: its folder, then the name typed into its own name field, then Return."""
    save.setDirectory(str(folder))
    field = save.findChild(QtWidgets.QLineEdit, "fileNameEdit")
    field.setFocus()
    QtTest.QTest.keyClicks(field, name)
    QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Return)
    qapp.processEvents()


def _status(dialog):
    return dialog.findChild(QtWidgets.QLabel, "plotExportStatus")


def test_b1_scene_and_bare_viewbox_targets_take_only_visual_formats(qapp, hooked):
    host, view, plot, _made = _plot(qapp, [(range(5), [1, 4, 9, 16, 25])])
    dialog = _open(qapp, view, plot.getViewBox())
    assert dialog.ui.itemTree.currentItem().gitem is plot
    assert all(_available(_format(dialog, name)) for name in (CSV, HDF5, IMAGE, MPL, SVG)), "a plot takes every format"
    _click_format(qapp, dialog, SVG)
    _click_target(qapp, dialog, view.scene())
    for name in (CSV, HDF5, MPL):
        item = _format(dialog, name)
        assert not _available(item) and item.toolTip(), (name, item.toolTip())
    assert _available(_format(dialog, IMAGE)) and _available(_format(dialog, SVG))
    assert _current(dialog) == SVG, "an available choice stays chosen"
    _click_format(qapp, dialog, CSV)
    assert _current(dialog) == SVG, "an unavailable format cannot be chosen"
    _click_target(qapp, dialog, plot)
    _click_format(qapp, dialog, CSV)
    assert _current(dialog) == CSV and dialog.ui.exportBtn.isEnabled() and not dialog.ui.copyBtn.isEnabled()
    _click_target(qapp, dialog, plot.getViewBox())
    assert not _available(_format(dialog, CSV)) and _current(dialog) == IMAGE, "falls to the first available format"
    assert dialog.ui.exportBtn.isEnabled() and dialog.ui.copyBtn.isEnabled()
    # a view box outside any plot
    host2, layout_view, _plots, bare = _layout(qapp, [[(range(3), [1, 2, 3])]])
    dialog2 = _open(qapp, layout_view, bare)
    assert dialog2.ui.itemTree.currentItem().gitem is bare
    assert [_available(_format(dialog2, name)) for name in (CSV, HDF5, IMAGE, MPL, SVG)] == [
        False, False, True, False, True]
    assert hooked == []


def test_b2_empty_and_image_only_plots_offer_no_data_formats_and_write_nothing(qapp, hooked, tmp_path):
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("kept")
    for kind in ("empty", "image-only"):
        host, view, plot, _made = _plot(qapp, image=(kind == "image-only"))
        dialog = _open(qapp, view, plot.getViewBox())
        before = _current(dialog)
        for name in (HDF5, CSV):
            item = _format(dialog, name)
            assert not _available(item) and item.toolTip(), (kind, name)
        _click_format(qapp, dialog, HDF5)
        assert _current(dialog) == before != HDF5, kind
    assert sorted(path.name for path in tmp_path.iterdir()) == ["sentinel.txt"] and sentinel.read_text() == "kept"
    assert hooked == []


def test_b3_valid_csv_keeps_the_original_data_and_hdf5_the_displayed_data(qapp, hooked, tmp_path):
    x = np.array([1.0, 10.0, 100.0, 1000.0, 1.0e4, 1.0e5])
    y = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    host, view, plot, (curve,) = _plot(qapp, [(x, y)])
    plot.setLogMode(x=True)
    curve.setDownsampling(ds=2, auto=False, method="subsample")
    qapp.processEvents()
    shown_x, shown_y = curve.getData()
    original_x, original_y = curve.getOriginalDataset()
    assert len(shown_x) < len(original_x), "the displayed data is downsampled and in log x"
    dialog = _open(qapp, view, plot.getViewBox())
    _click_format(qapp, dialog, CSV)
    _answer(qapp, _export(qapp, dialog), tmp_path, "curve.csv")
    _click_format(qapp, dialog, HDF5)
    _answer(qapp, _export(qapp, dialog), tmp_path, "curve.h5")
    rows = np.loadtxt(tmp_path / "curve.csv", delimiter=",", skiprows=1)
    np.testing.assert_array_equal(rows[:, 0], original_x)
    np.testing.assert_array_equal(rows[:, 1], original_y)
    with h5py.File(tmp_path / "curve.h5", "r") as handle:
        data = np.asarray(handle["Export"])
    np.testing.assert_array_equal(data[0], shown_x)
    np.testing.assert_array_equal(data[1], shown_y)
    assert hooked == []


def test_b4_errorbars_ragged_and_mixed_plots_export_and_hdf5_refuses_a_cleared_curve(qapp, hooked, tmp_path):
    # a cleared curve beside a valid one: HDF5 says why and writes nothing; CSV writes the valid curve
    host, view, plot, (_kept, cleared) = _plot(qapp, [(range(4), [1, 2, 3, 4]), (range(3), [5, 6, 7])])
    cleared.clear()
    qapp.processEvents()
    dialog = _open(qapp, view, plot.getViewBox())
    item = _format(dialog, HDF5)
    assert not _available(item) and item.toolTip(), "HDF5 writes every curve, and one has no data"
    _click_format(qapp, dialog, CSV)
    _answer(qapp, _export(qapp, dialog), tmp_path, "kept.csv")
    np.testing.assert_array_equal(np.loadtxt(tmp_path / "kept.csv", delimiter=",", skiprows=1),
                                  np.column_stack([np.arange(4.0), [1.0, 2.0, 3.0, 4.0]]))
    # error bars alone: CSV writes them; HDF5 has no curve to write
    host2, view2, plot2, _made = _plot(qapp, errorbars={"x": np.arange(3.0), "y": np.array([1.0, 2.0, 3.0]),
                                                        "height": np.array([0.5, 0.5, 0.5])})
    dialog2 = _open(qapp, view2, plot2.getViewBox())
    assert not _available(_format(dialog2, HDF5)) and _format(dialog2, HDF5).toolTip()
    _click_format(qapp, dialog2, CSV)
    _answer(qapp, _export(qapp, dialog2), tmp_path, "errorbars.csv")
    errors = np.genfromtxt(tmp_path / "errorbars.csv", delimiter=",", skip_header=1)
    np.testing.assert_array_equal(errors[:, :2], np.column_stack([np.arange(3.0), [1.0, 2.0, 3.0]]))
    # ragged curves, and an image beside a curve: both data formats export
    host3, view3, plot3, _made = _plot(qapp, [(range(5), [1, 2, 3, 4, 5]), (range(3), [9, 8, 7])], image=True)
    dialog3 = _open(qapp, view3, plot3.getViewBox())
    assert _available(_format(dialog3, CSV)) and _available(_format(dialog3, HDF5))
    _click_format(qapp, dialog3, HDF5)
    _answer(qapp, _export(qapp, dialog3), tmp_path, "ragged.h5")
    with h5py.File(tmp_path / "ragged.h5", "r") as handle:
        group = handle["Export"]
        stored = sorted((np.asarray(group[name]).tolist() for name in group), key=lambda rows: len(rows[0]))
    assert stored == [[[0.0, 1.0, 2.0], [9.0, 8.0, 7.0]], [[0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0]]]
    _click_format(qapp, dialog3, CSV)
    _answer(qapp, _export(qapp, dialog3), tmp_path, "ragged.csv")
    ragged = np.genfromtxt(tmp_path / "ragged.csv", delimiter=",", skip_header=1)
    np.testing.assert_array_equal(ragged[:, :2], np.column_stack([np.arange(5.0), [1.0, 2.0, 3.0, 4.0, 5.0]]))
    np.testing.assert_array_equal(ragged[:3, 2:4], np.column_stack([np.arange(3.0), [9.0, 8.0, 7.0]]))
    assert hooked == []


def test_b5_a_pending_save_checks_its_own_plot_and_reports_in_the_export_dialog(
        qapp, hooked, tmp_path, _xdart_qt_harness):
    # cleared while Save is open, with the Export dialog closed: refused in that reopened dialog, nothing written
    host, view, plot, _made = _plot(qapp, [(range(4), [1, 2, 3, 4])])
    dialog = _open(qapp, view, plot.getViewBox())
    _click_format(qapp, dialog, CSV)
    save = _export(qapp, dialog)
    plot.clear()
    qapp.processEvents()
    QtTest.QTest.mouseClick(dialog.ui.closeBtn, LEFT)
    qapp.processEvents()
    assert not dialog.isVisible() and save.isVisible()
    _answer(qapp, save, tmp_path, "cleared.csv")
    assert list(tmp_path.iterdir()) == [], "nothing is created, truncated or appended"
    status = _status(dialog)
    assert dialog.isVisible() and status is not None and status.isVisible(), "the reason returns in its own dialog"
    assert status.text() and status.toolTip()
    # a changed tree target cannot redirect a pending Save: it checks the plot it was opened for
    host2, view2, (first, second), _bare = _layout(qapp, [[(range(4), [1, 2, 3, 4])], [(range(3), [7, 8, 9])]])
    dialog2 = _open(qapp, view2, first.getViewBox())
    _click_format(qapp, dialog2, CSV)
    save2 = _export(qapp, dialog2)
    first.clear()
    qapp.processEvents()
    _click_target(qapp, dialog2, second)
    _answer(qapp, save2, tmp_path, "redirected.csv")
    status2 = _status(dialog2)
    assert status2 is not None and status2.isVisible() and status2.text(), "the pending Save reports its own refusal"
    assert list(tmp_path.iterdir()) == []
    # the host goes while Save is open: the Save dialog goes with it, and nothing is written or raised
    host3, view3, plot3, _made = _plot(qapp, [(range(4), [1, 2, 3, 4])])
    dialog3 = _open(qapp, view3, plot3.getViewBox())
    _click_format(qapp, dialog3, CSV)
    save3 = _export(qapp, dialog3)
    host3.deleteLater()
    _xdart_qt_harness.deliver_deferred_deletes()
    assert not shiboken6.isValid(save3) and not shiboken6.isValid(dialog3)
    assert list(tmp_path.iterdir()) == []
    assert hooked == []


def test_b6_one_lazy_dialog_per_scene_visual_exports_and_cleanup_with_the_host(
        qapp, hooked, tmp_path, _xdart_qt_harness):
    from pyqtgraph.GraphicsScene.exportDialog import ExportDialog
    windows_before = {shiboken6.getCppPointer(widget)[0] for widget in QtWidgets.QApplication.topLevelWidgets()}
    host, view, (top, bottom), bare = _layout(qapp, [[(range(4), [1, 2, 3, 4])], [(range(3), [1, 0, 1])]])
    assert view.scene().exportDialog is None, "no Export dialog before the first Export click"
    dialog = _open(qapp, view, top.getViewBox())
    assert _open(qapp, view, bottom.getViewBox()) is dialog and _open(qapp, view, bare) is dialog
    assert _open(qapp, view, top.getViewBox()) is dialog
    _click_target(qapp, dialog, top)
    _click_format(qapp, dialog, IMAGE)
    _answer(qapp, _export(qapp, dialog), tmp_path, "plot.png")
    assert (tmp_path / "plot.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # Exercise the real Save callback; defer the success assertion until after host retirement so the parent red
    # also completes the lifetime/cancellation checks.
    _click_format(qapp, dialog, SVG)
    _answer(qapp, _export(qapp, dialog), tmp_path, "plot.svg")
    status = _status(dialog)
    svg_refusal = status.text() if status is not None and status.isVisible() else ""
    files_after_svg = sorted(path.name for path in tmp_path.iterdir())
    _click_format(qapp, dialog, CSV)
    cancelled = _export(qapp, dialog)
    QtTest.QTest.keyClick(cancelled, QtCore.Qt.Key.Key_Escape)
    qapp.processEvents()
    assert not cancelled.isVisible() and sorted(path.name for path in tmp_path.iterdir()) == files_after_svg
    pending, select_box = _export(qapp, dialog), dialog.selectBox
    assert pending is not None
    host.deleteLater()
    _xdart_qt_harness.deliver_deferred_deletes()
    assert not any(shiboken6.isValid(item) for item in (dialog, cancelled, pending, select_box)), "retired with the host"
    added = [widget for widget in QtWidgets.QApplication.topLevelWidgets()
             if shiboken6.getCppPointer(widget)[0] not in windows_before
             and isinstance(widget, (ExportDialog, QtWidgets.QFileDialog))]
    assert added == [] and sorted(path.name for path in tmp_path.iterdir()) == files_after_svg
    assert hooked == []
    assert (tmp_path / "plot.svg").exists(), svg_refusal
    assert not svg_refusal and files_after_svg == ["plot.png", "plot.svg"]
    from xml.etree import ElementTree
    svg = ElementTree.parse(tmp_path / "plot.svg").getroot()
    assert svg.tag == "{http://www.w3.org/2000/svg}svg"
    assert any(node.get("d") or node.get("points") for node in svg.iter()
               if node.tag.rsplit("}", 1)[-1] in {"path", "polyline"}), "SVG contains rendered geometry"
