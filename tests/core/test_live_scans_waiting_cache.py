"""A SPEC scan without detector files waits without opening its source, and an unchanged wait is not redone.

Found on real data in Claude's terminal review of per-scan Live: STO_align has 1,196 scans, most without
images, and each 2 s poll reopened every waiting scan (about 2.8 s per poll on an M3 Max).
"""
import pytest

from xrd_tools.analysis import live_scans
from xrd_tools.analysis.live_scans import LiveScanObserver
from xrd_tools.io import spec, stat_identity

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


@pytest.mark.parametrize("fallback", (False, True))
def test_waiting_spec_proofs_follow_their_own_scan_when_the_log_grows(
        tmp_path, monkeypatch, ctime_seam, win32_pathname_ctime, fallback):
    path = _spec(tmp_path)
    for number in range(3, 6):
        _append(path, _spec_rows(number, SCAN, 2))
    _append(path, _spec_rows(6, SCAN, 1))
    _raw(path, 6, 0)
    if fallback:
        monkeypatch.setattr(ctime_seam, "IDENTITY_CARRIES_CTIME", False)
        win32_pathname_ctime(path)
    elif not stat_identity.IDENTITY_CARRIES_CTIME:
        pytest.skip("the POSIX cache requires trustworthy pathname ctime")

    checked = []
    real_proof = spec.spec_occurrence

    def count_proof(selected_path, key):
        checked.append(key)
        return real_proof(selected_path, key)

    monkeypatch.setattr(spec, "spec_occurrence", count_proof)
    observer = LiveScanObserver(_selection(path, stitch_all=True), DECODE)
    keys = [f"{number}.1" for number in range(2, 7)]
    initial = observer.poll(now=0).groups
    assert checked == keys
    assert all(row.status == "waiting" for row in initial)
    assert all("detector coverage" in row.reason for row in initial[:-1])
    assert "completion" in initial[-1].reason

    checked.clear()
    _append(path, "15 -10\n")
    _raw(path, 6, 1)
    growing = observer.poll(now=2).groups
    assert checked == (keys if fallback else ["6.1"])
    assert [row.reason for row in growing[:-1]] == [row.reason for row in initial[:-1]]
    assert observer.poll(now=4).groups[-1].status == "ready"

    # A changed selected block invalidates its own historical wait as well.
    checked.clear()
    contents = path.read_bytes()
    path.write_bytes(contents.replace(b"#P0 14 -10", b"#P0 14 -11", 1))
    rewritten = observer.poll(now=6).groups
    assert rewritten[0].status == "waiting" and "detector coverage" in rewritten[0].reason
    assert "2.1" in checked
    if not fallback:
        assert not set(keys[1:-1]).intersection(checked)

    _raw(path, 2, 0, 1)
    row = observer.poll(now=8).groups[0]
    assert row.status == "waiting" and "stillness" in row.reason
    assert observer.poll(now=10).groups[0].status == "ready"


def test_waiting_spec_rechecks_when_a_partial_successor_becomes_valid(tmp_path):
    path = _spec(tmp_path, command="ascan del 14 17 3 1", rows=2)
    _raw(path, 2, 0, 1)
    _append(path, "#S 3")
    observer = LiveScanObserver(_selection(path), DECODE)

    row, = observer.poll(now=0).groups
    assert row.status == "waiting" and "completion" in row.reason
    row, = observer.poll(now=2).groups
    assert row.status == "waiting" and "completion" in row.reason
    _append(path, f" {SCAN}\n")
    row, = observer.poll(now=4).groups
    assert row.status == "waiting" and "stillness" in row.reason
    row, = observer.poll(now=6).groups
    assert row.status == "ready"
