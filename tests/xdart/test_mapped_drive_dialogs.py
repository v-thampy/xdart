"""A folder picked in xdart's dialog on a mapped share reads as the drive, as a path pasted from Explorer does, and the
Save Path's follow-the-Project check takes the two spellings as one folder (maintainer, 2026-10-26: "if i use the open
file dialog, it gives \\ssrl... but if i copy and paste from windows explorer - i get x:. they should point to the same
path in xdart", his pick "Drive letter, X:\\... (Recommended)")."""
from __future__ import annotations

import ntpath
import types

import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering import controls_editing
from xdart.gui.widgets import file_dialogs
from xrd_tools.io import mapped_drives

SHARE = r"\\ssrlblfs1.slac.stanford.edu\hardx"


@pytest.fixture
def mapped(monkeypatch):
    monkeypatch.setattr(mapped_drives, "drive_remotes", lambda: {"X:": SHARE})


def test_a_folder_picked_on_the_mapped_share_reads_as_the_drive(qapp, mapped, monkeypatch):  # noqa: F811
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", staticmethod(
        lambda *args, **kwargs: "//ssrlblfs1.slac.stanford.edu/hardx/bl11-3/test_Oct2026"))
    assert file_dialogs.ask(None, "Choose folder", folder=True) == [r"X:\bl11-3\test_Oct2026"]


def test_files_picked_on_the_mapped_share_read_as_the_drive_and_others_as_picked(qapp, mapped, monkeypatch):  # noqa: F811
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", staticmethod(lambda *args, **kwargs: (
        ["//ssrlblfs1.slac.stanford.edu/hardx/cal/LaB6.poni", "C:/local/mask.edf"], "")))
    assert file_dialogs.ask(None, "Choose", many=True) == [r"X:\cal\LaB6.poni", "C:/local/mask.edf"]


def test_a_cancelled_dialog_still_returns_nothing(qapp, mapped, monkeypatch):  # noqa: F811
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", staticmethod(lambda *args, **kwargs: ""))
    assert file_dialogs.ask(None, "Choose folder", folder=True) == []


def test_the_save_path_check_takes_the_drive_and_the_share_as_one_folder(mapped, monkeypatch):
    monkeypatch.setattr(controls_editing, "os", types.SimpleNamespace(path=ntpath))  # Windows spelling rules
    assert controls_editing._same_folder(r"X:\bl11-3\test_Oct2026\xdart_processed_data",
                                         "//ssrlblfs1.slac.stanford.edu/hardx/bl11-3/test_Oct2026/xdart_processed_data")
    assert not controls_editing._same_folder(r"X:\bl11-3\Wu_Oct2026\xdart_processed_data",
                                             SHARE + r"\bl11-3\test_Oct2026\xdart_processed_data")
