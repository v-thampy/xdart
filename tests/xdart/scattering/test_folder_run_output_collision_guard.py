"""A Folder Run never writes its output over a raw file, however the file system answers (NEXT-1215-SHARE-LOOKUPS).

Admission checks file identity and compares path spellings when identity is insufficient. A Save Path output that is
a raw TIFF through a symlink or hardlink is refused through the real page. The altered filesystem answers go to the
guard's own stat calls only (other owners read the same files)."""
import os
import sys

import pytest

pytest.importorskip("pyqtgraph")

from tests.xdart.scattering.test_browse_reference_rings_run import _page, _pump
from tests.xdart.scattering.test_folder_run_frame_read_at_scan_boundary import _scans
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering import output_preflight


@pytest.fixture(autouse=True, params=(False, True), ids=("platform-paths", "windows-paths"))
def _guard_branch(request, monkeypatch):
    """Run each row with the guard's Windows branch too (raw files resolved folder by folder, NEXT-1215), and with
    this platform's."""
    if request.param:
        monkeypatch.setattr(output_preflight, "_CHECK_RESOLVED_RAW_PATHS", True)
    return request.param


class _StatAnswer:
    """Override fields that a filesystem provider can report differently."""

    def __init__(self, result, **fields):
        self._result = result
        self.__dict__.update(fields)

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
    ("symlink", "target-not-found"),
    ("symlink", "target-different-device"),
    ("hardlink", "both-inodes-0"),
    ("two-symlinks", "target-different-device"),
    ("two-symlinks", "target-not-found"),
    ("two-symlinks", "target-unreadable"),
    ("two-symlinks", "raw-unreadable"),
))
def test_an_output_that_is_a_raw_file_is_refused_at_admission(qapp, tmp_path, monkeypatch, link, answer):  # noqa: F811
    _stems, intent = _scans(tmp_path, 2, 2)
    raw = tmp_path / "raw" / "s00_0001.tif"
    before = raw.read_bytes()
    target = tmp_path / "processed" / "s00_int1d.nexus"
    target.parent.mkdir()
    try:
        if link == "two-symlinks":
            original = raw.rename(tmp_path / "detector.tif")
            raw.symlink_to(original)
            target.symlink_to(original)
            if answer == "target-different-device":
                # Exercise the Windows collision policy with real files and both symlinks.
                monkeypatch.setattr(output_preflight, "_CHECK_RESOLVED_RAW_PATHS", True, raising=False)
        elif link == "symlink":
            target.symlink_to(raw)
        else:
            os.link(raw, target)
    except OSError as error:
        pytest.skip(f"{link}s unavailable: {error}")
    if answer == "both-inodes-0":
        altered = (raw, target)
    else:
        altered = (target if answer and answer.startswith("target") else raw,)
    asked = {os.path.normcase(os.fspath(path)) for path in altered}
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if (answer and not isinstance(path, int) and os.path.normcase(os.fsdecode(os.fspath(path))) in asked
                and _asked_by_the_guard()):
            if answer.endswith("unreadable"):
                raise PermissionError(13, "Permission denied", os.fspath(path))
            if answer == "target-not-found":
                raise FileNotFoundError(2, "No such file or directory", os.fspath(path))
            result = real_stat(path, *args, **kwargs)
            if answer == "target-different-device":
                return _StatAnswer(result, st_dev=result.st_dev + 1)
            return _StatAnswer(result, st_ino=0)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    with _page(qapp, intent) as (page, _view, observed):
        def collision_reported():
            return "same file as a raw directory input" in (page._notice_text or "")

        page._shell.run_controls.startButton.click()
        assert _pump(qapp, lambda: collision_reported() or bool(observed), seconds=30)
        assert collision_reported(), (page._notice_text, [event.kind for event in observed])
    assert raw.read_bytes() == before
