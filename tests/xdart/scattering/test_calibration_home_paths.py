"""A ~ typed in the calibration window's Image Path or the Refine card's Geometry file is the home folder
(LIVE-FIX-IMAGE-FOLDER-TILDE; a Known limitation in the 1.2.9 notes).

1.2.9 expanded a typed ~ in the SPEC path only; the image folder and the geometry file were read as typed, so a
"~" folder under the current directory was searched. The pickers give absolute paths; typing is the only way in.
HOME and USERPROFILE (Windows) both point at the test's home folder.
"""
from __future__ import annotations

import shutil

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_calibration_quick_chain import _click, _close, _commit, _quick_card, _window
from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _wait


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _home(tmp_path, monkeypatch):
    home = (tmp_path / "home").resolve()
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


@pytest.mark.parametrize("page_name", ["Stitch", "RSM"])
def test_a_typed_home_image_path_is_the_home_folder(qapp, tmp_path, monkeypatch, page_name):
    """Image Path "~/calibrant/images" is home's images folder in the source the window reads, and Use for both
    copies that folder to the other scan."""
    from xdart.gui.widgets.controls_panel import ActionButton

    home = _home(tmp_path, monkeypatch)
    spec = home / "calibrant" / "scan"
    images = home / "calibrant" / "images"
    images.mkdir(parents=True)
    spec.write_text("#F scan\n")
    page, dialog = _window(qapp, tmp_path / "project", page_name)
    try:
        _quick_card(dialog)
        rows = dialog._quick_rows
        _commit(rows["source"].editor, str(spec))
        _wait(dialog)
        _commit(rows["images"].editor, "~/calibrant/images")
        _wait(dialog)
        assert dialog._spec().options["image_dir"] == str(images)
        copy = [button for button in rows["images"].findChildren(ActionButton) if button.text() == "Use for both"]
        assert len(copy) == 1
        _click(copy[0])
        _wait(dialog)
        assert dialog._text("beam_images") == str(images)
    finally:
        _close(qapp, page, dialog)


@pytest.mark.parametrize("typed_home", [True, False], ids=["home", "absolute-control"])
def test_a_typed_home_geometry_file_starts_the_refinement(qapp, tmp_path, monkeypatch, typed_home):
    """The typed seed qualifies in the UI, reaches guided seeding and starts the real Refine job."""
    from tests.xdart.scattering.test_lt_calibration_compact import _checklist, _orientation_row, _show, _staged

    home = _home(tmp_path, monkeypatch)
    dialog, saved, _value = _staged(qapp, tmp_path)
    try:
        seed = home / "calibration" / "fixed-arm.json"
        seed.parent.mkdir()
        shutil.copyfile(saved, seed)
        table = _checklist(dialog)
        _show(dialog, "4 · Calibrate and fit")
        dialog.refine_seed.setCurrentIndex(dialog.refine_seed.findData("file"))
        _commit(dialog._fields["refine_seed_path"], "~/calibration/fixed-arm.json" if typed_home else str(seed))
        _wait(dialog)
        assert dialog.refine_seed_note.text() == ""
        assert dialog.orientation_button.isEnabled(), _orientation_row(table)
        assert dialog._seed_options()["file"] == str(seed)
        dialog.refine_button.click()
        assert dialog.active() and dialog._job == "refine", dialog.status.text()
        assert dialog._point_request[1][1] == str(seed)
    finally:
        if dialog.active():
            dialog.cancel_button.click()
            _wait(dialog)
        dialog.close()
        _wait(dialog)
        dialog.deleteLater()
        qapp.processEvents()
