"""The SPEC file picker: folders and SPEC-named files only, until "All files" is chosen."""
import time

import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from PySide6.QtTest import QTest

from xdart.gui.widgets import file_dialogs
from xdart.gui.widgets.file_dialogs import ANY_FILTER, NEXUS_FILTER, is_spec_name, spec_dialog

# a direct-beam folder as the beamline writes it: the SPEC file beside its images and notes
FOLDER = ("direct_beam", "b_thampy_direct_beam_scan3_0094.raw", "b_thampy_direct_beam_scan3_0094.raw.pdi",
          "direct_beam_scan1.csv", "run.spec")


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize(("name", "spec"), [
    ("direct_beam", True), ("b_thampy_LaB6_17keV", True), ("run.spec", True), ("RUN.SPEC", True),
    ("b_thampy_direct_beam_scan3_0094.raw", False), ("b_thampy_direct_beam_scan3_0094.raw.pdi", False),
    ("direct_beam_scan1.csv", False), ("frame_0000.edf", False), (".hidden", False)])
def test_a_spec_file_has_no_extension_or_spec(name, spec):
    assert is_spec_name(name) is spec


def _listed(dialog, folder, qapp, count):
    """The names the dialog lists in ``folder`` once its file system model has read it."""
    proxy = dialog.proxyModel()
    model = proxy.sourceModel()
    until = time.monotonic() + 10
    while model.rowCount(model.index(str(folder))) < count and time.monotonic() < until:
        qapp.processEvents()
        time.sleep(.01)
    root = proxy.mapFromSource(model.index(str(folder)))
    return {proxy.index(row, 0, root).data() for row in range(proxy.rowCount(root))}


def test_the_spec_picker_lists_folders_and_spec_files_until_all_files_is_chosen(qapp, tmp_path):
    for name in FOLDER:
        (tmp_path / name).write_text("x")
    (tmp_path / "images").mkdir()
    dialog = spec_dialog(None, "Select SPEC file", str(tmp_path), many=True)
    try:
        assert not dialog.testOption(QtWidgets.QFileDialog.Option.ShowDirsOnly)
        assert _listed(dialog, tmp_path, qapp, len(FOLDER) + 1) == {"direct_beam", "run.spec", "images"}
        types = dialog.findChild(QtWidgets.QComboBox, "fileTypeCombo")
        QTest.keyClick(types, QtCore.Qt.Key.Key_Down)  # the user picks "All files"
        qapp.processEvents()
        assert dialog.selectedNameFilter() == ANY_FILTER
        assert _listed(dialog, tmp_path, qapp, len(FOLDER) + 1) == {*FOLDER, "images"}
        QTest.keyClick(types, QtCore.Qt.Key.Key_Up)  # and "SPEC files" again
        qapp.processEvents()
        assert _listed(dialog, tmp_path, qapp, len(FOLDER) + 1) == {"direct_beam", "run.spec", "images"}
    finally:
        dialog.deleteLater()


def test_stitch_spec_sources_use_the_spec_picker_and_keep_every_pick(qapp, tmp_path, monkeypatch):
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.stitch.page import StitchPage

    class Status:
        def show(self, text, timeout_ms=0):
            pass
    calls = []
    picks = [str(tmp_path / "one"), str(tmp_path / "two")]
    monkeypatch.setattr(file_dialogs, "browse_spec", lambda *a, **k: calls.append(("spec", k)) or picks)
    monkeypatch.setattr(file_dialogs, "browse", lambda *a, **k: calls.append(("any", k)) or picks)
    page = StitchPage(empty_host_services(Status()))
    try:
        page.set_field("source_kind", "Spec")
        page._browse("spec_path")
        assert calls[-1][0] == "spec" and page._source_paths == tuple(picks)  # both files kept
        page.set_field("source_kind", "NeXus")
        page._browse("spec_path")
        kind, options = calls[-1]
        assert kind == "any" and options["file_filter"] == NEXUS_FILTER
    finally:
        page.shutdown()


def test_calibration_source_fields_pick_spec_or_nexus_by_their_source_type(qapp, tmp_path, monkeypatch):
    from xdart.gui.tabs.stitch.calibration import StitchCalibrationDialog
    calls = []
    monkeypatch.setattr(file_dialogs, "browse_spec", lambda *a, **k: calls.append("spec") or [])
    monkeypatch.setattr(file_dialogs, "browse", lambda *a, **k: calls.append(k.get("file_filter")) or [])
    dialog = StitchCalibrationDialog(tmp_path, backend="multigeometry")
    try:
        def click(key):
            dialog._rows[key][1].findChild(QtWidgets.QAbstractButton).click()
            return calls[-1]
        for key, combo, spec_kind, nexus_kind in (("source", dialog.source_kind, "spec", "nexus_stack"),
                                                  ("beam_source", dialog.beam_source_kind, "spec_series", "nexus")):
            combo.setCurrentIndex(combo.findData(spec_kind))
            assert click(key) == "spec"
            combo.setCurrentIndex(combo.findData(nexus_kind))
            assert click(key) == NEXUS_FILTER
    finally:
        dialog.close()
        dialog.deleteLater()
