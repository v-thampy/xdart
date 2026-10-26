"""A Windows mapped drive and its share's network path are one place (xrd_tools.io.mapped_drives; maintainer,
2026-10-26: "x: maps to \\ssrlblfs1.slac.stanford.edu\\hardx ... if i use the open file dialog, it gives \\ssrl... but
if i copy and paste from windows explorer - i get x:. they should point to the same path in xdart"). The rules are
exercised with ntpath and an explicit drive table, so they run on every platform; off Windows there are no mappings."""
from __future__ import annotations

import ntpath
import posixpath

import pytest

from xrd_tools.io import mapped_drives
from xrd_tools.io.mapped_drives import as_mapped_drive, as_network_path, relative_within

SHARE = r"\\ssrlblfs1.slac.stanford.edu\hardx"
REMOTES = {"X:": SHARE}


@pytest.mark.parametrize("path, expected", [
    ("//ssrlblfs1.slac.stanford.edu/hardx/bl11-3/test_Oct2026", r"X:\bl11-3\test_Oct2026"),  # Qt's spelling
    (r"\\ssrlblfs1.slac.stanford.edu\hardx\bl11-3\test_Oct2026", r"X:\bl11-3\test_Oct2026"),
    (r"\\SSRLBLFS1.SLAC.stanford.edu\HARDX\bl11-3", r"X:\bl11-3"),  # a share's name ignores case
    (SHARE, "X:\\"),
    (r"\\ssrlblfs1.slac.stanford.edu\hardxy\bl11-3", r"\\ssrlblfs1.slac.stanford.edu\hardxy\bl11-3"),  # other share
    (r"\\otherhost\hardx\bl11-3", r"\\otherhost\hardx\bl11-3"),
    (r"X:\bl11-3", r"X:\bl11-3"),  # already a drive
    (r"\\?\UNC\ssrlblfs1.slac.stanford.edu\hardx\a", r"\\?\UNC\ssrlblfs1.slac.stanford.edu\hardx\a"),  # a device path
    ("", ""),
])
def test_a_path_on_a_mapped_share_is_spelled_with_its_drive(path, expected):
    assert as_mapped_drive(path, REMOTES) == expected


def test_the_longest_mapped_folder_names_the_drive():
    remotes = {"X:": r"\\host\share", "Y:": r"\\host\share\sub"}
    assert as_mapped_drive(r"\\host\share\sub\a", remotes) == r"Y:\a"
    assert as_mapped_drive(r"\\host\share\other\a", remotes) == r"X:\other\a"


@pytest.mark.parametrize("path, expected", [
    (r"X:\bl11-3\test_Oct2026", SHARE + r"\bl11-3\test_Oct2026"),
    ("x:/bl11-3", SHARE + r"\bl11-3"),
    ("X:", SHARE),
    (r"C:\data", r"C:\data"),  # not mapped
    (SHARE + r"\bl11-3", SHARE + r"\bl11-3"),
])
def test_a_mapped_drive_spells_its_network_path_for_comparing(path, expected):
    assert as_network_path(path, REMOTES) == expected


def test_without_a_drive_table_every_spelling_stays_as_it_is():
    for path in (r"\\host\share\a", r"X:\a", "/data/a"):
        assert as_mapped_drive(path, {}) == as_network_path(path, {}) == path


def test_off_windows_there_are_no_mapped_drives(monkeypatch):
    monkeypatch.setattr(mapped_drives, "_is_windows", lambda: False)
    assert mapped_drives.drive_remotes() == {}
    assert as_mapped_drive(SHARE + r"\a") == SHARE + r"\a"


def test_the_beamline_source_on_the_share_is_inside_its_project_on_the_drive():
    """His log: "source \\\\ssrlblfs1...\\bl11-3\\test_Oct2026\\sub1\\LaB6_..._0001.tif is outside the project root
    x:\\bl11-3\\test_oct2026; storing an absolute (non-portable) path"."""
    source = SHARE + r"\bl11-3\test_Oct2026\sub1\LaB6_4deg_300mm_15s_10261309_0001.tif"
    project = r"x:\bl11-3\test_oct2026"
    assert relative_within(source, project, pathmod=ntpath, remotes=REMOTES) == (
        "sub1/LaB6_4deg_300mm_15s_10261309_0001.tif")
    assert relative_within(r"X:\bl11-3\test_Oct2026\a.tif", SHARE + r"\bl11-3\test_Oct2026", pathmod=ntpath,
                           remotes=REMOTES) == "a.tif"
    # without the table the two spellings are two places, as before
    assert relative_within(source, project, pathmod=ntpath, remotes={}) is None


@pytest.mark.parametrize("source", [
    SHARE + r"\bl11-3\Wu_Oct2026\calibration\LaB6.tif",  # another folder on the same share
    r"\\otherhost\data\LaB6.tif",
    r"C:\data\LaB6.tif",
])
def test_a_source_truly_outside_the_project_stays_outside(source):
    assert relative_within(source, r"X:\bl11-3\test_Oct2026", pathmod=ntpath, remotes=REMOTES) is None


def test_on_posix_inside_and_outside_are_as_before():
    assert relative_within("/data/project/sub/a.tif", "/data/project", pathmod=posixpath, remotes={}) == "sub/a.tif"
    assert relative_within("/data/other/a.tif", "/data/project", pathmod=posixpath, remotes={}) is None
