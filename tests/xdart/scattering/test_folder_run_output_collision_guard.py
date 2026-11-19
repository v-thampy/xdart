"""A Folder Run never writes its output over a raw file, however the file system answers (NEXT-1215-SHARE-LOOKUPS).

Admission matches raw files by their (device, inode) identity and resolves a path only where an identity is missing;
on a network share every resolve is a round trip. A Save Path output that is a raw TIFF through a symlink or a hardlink
is refused through the real page whether the identity is usable, reported as 0 or not readable, on the raw side or on
the target's. The altered answers go to the guard's own stat calls only (other owners read the same files)."""
import os
import sys

import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _pump
from tests.xdart.scattering.test_folder_run_frame_read_at_scan_boundary import _scans
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401


class _ZeroInode:
    """A stat answer that reports no file identity (st_ino 0), as some file systems and shares do."""

    def __init__(self, result):
        self._result = result
        self.st_ino = 0

    def __getattr__(self, name):
        return getattr(self._result, name)


def _asked_by_the_guard() -> bool:
    frame = sys._getframe(2)
    for _ in range(4):
        if frame is None:
            return False
        if frame.f_code.co_name == "_validate_deferred_targets":
            return True
        frame = frame.f_back
    return False


@pytest.mark.parametrize("link, answer", (
    ("symlink", None),
    ("hardlink", None),
    ("symlink", "target-inode-0"),
    ("symlink", "target-unreadable"),
    ("symlink", "raw-inode-0"),
    ("symlink", "raw-unreadable"),
))
def test_an_output_that_is_a_raw_file_is_refused_at_admission(qapp, tmp_path, monkeypatch, link, answer):  # noqa: F811
    _stems, intent = _scans(tmp_path, 2, 2)
    raw = tmp_path / "raw" / "s00_0001.tif"
    before = raw.read_bytes()
    target = tmp_path / "processed" / "s00_int1d.nexus"
    target.parent.mkdir()
    try:
        target.symlink_to(raw) if link == "symlink" else os.link(raw, target)
    except OSError as error:
        pytest.skip(f"{link}s unavailable: {error}")
    asked = os.path.normcase(os.fspath(target if answer and answer.startswith("target") else raw))
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if (answer and not isinstance(path, int) and os.path.normcase(os.fsdecode(os.fspath(path))) == asked
                and _asked_by_the_guard()):
            if answer.endswith("unreadable"):
                raise PermissionError(13, "Permission denied", os.fspath(path))
            return _ZeroInode(real_stat(path, *args, **kwargs))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    with _page(qapp, intent) as (page, _view, _observed):
        page._shell.run_controls.startButton.click()
        assert _pump(qapp, lambda: "same file as a raw directory input" in (page._notice_text or ""), seconds=30), \
            page._notice_text
    assert raw.read_bytes() == before
