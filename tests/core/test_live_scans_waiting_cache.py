"""A SPEC scan without detector files waits without opening its source, and an unchanged wait is not redone.

Found on real data in Claude's terminal review of per-scan Live: STO_align has 1,196 scans, most without
images, and each 2 s poll reopened every waiting scan (about 2.8 s per poll on an M3 Max).
"""
from xrd_tools.analysis import live_scans
from xrd_tools.analysis.live_scans import LiveScanObserver
from xrd_tools.io import stat_identity

from tests.core.test_live_per_scan import DECODE, SCAN, _append, _raw, _selection, _spec, _spec_rows


def test_waiting_scans_without_images_are_not_reopened(tmp_path, monkeypatch):
    path = _spec(tmp_path)  # scan 2, two complete rows
    _append(path, _spec_rows(3, SCAN, 2))
    _append(path, _spec_rows(4, SCAN, 2))
    _raw(path, 2, 0, 1)  # only scan 2 has its detector files
    opened = []
    real_open = live_scans.open_source

    def counting_open(spec, *args, **kwargs):
        opened.append(spec.options.get("scan"))
        return real_open(spec, *args, **kwargs)

    monkeypatch.setattr(live_scans, "open_source", counting_open)
    observer = LiveScanObserver(_selection(path, stitch_all=True), DECODE)

    first = observer.poll(now=0).groups
    assert [row.status for row in first] == ["waiting", "waiting", "waiting"]
    assert opened == ["2.1"]  # scans 3 and 4 have no files: never opened
    assert all("detector coverage" in row.reason for row in first[1:])

    opened.clear()
    second = observer.poll(now=2).groups
    assert [row.status for row in second] == ["ready", "waiting", "waiting"]
    if stat_identity.IDENTITY_CARRIES_CTIME:
        assert opened == []  # unchanged SPEC file and image files: the same verdicts, no reopen

    _raw(path, 3, 0, 1)  # scan 3's files arrive: it is examined again
    opened.clear()
    observer.poll(now=4)
    assert "3.1" in opened and "4.1" not in opened
    assert [row.status for row in observer.poll(now=6).groups] == ["ready", "ready", "waiting"]
