"""Real metadata/file-I/O discriminators for N-69301b5f: Codex's source-read denial row, copied unchanged. Its NW-R1
row, which pinned processing without metadata for a .txt present but never settling, is replaced by the declared
rare-case skip row test_live_never_waits.py::test_a_sidecar_that_never_settles_skips_its_scan_from_that_frame_and_the_
run_goes_on."""
from pathlib import Path

import pytest

from tests.xdart.scattering.test_live_never_waits import _intent, _live
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _labels, _output, _stop, _wait
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal
from xrd_tools.sources.image import TiffSeriesSource


@pytest.mark.parametrize("live, denied_label", [(True, 1), (True, 2), (False, 1)])
def test_a_real_source_read_denial_skips_its_scan_and_processes_the_next(tmp_path, monkeypatch, live, denied_label):
    raw = tmp_path / "raw"
    raw.mkdir()
    for stem in ("bad", "good"):
        for label in (1, 2):
            _frame(raw, stem, label, "txt", float(label))
    denied = raw / f"bad_{denied_label:04d}.tif"
    actual_read = TiffSeriesSource._read_path
    attempts = []

    def read(source, path):
        if Path(path) != denied:
            return actual_read(source, path)
        attempts.append(str(path))
        mode = denied.stat().st_mode & 0o777
        denied.chmod(0)  # a real unreadable source; no exception or return value is fabricated
        try:
            return actual_read(source, path)
        finally:
            denied.chmod(mode)

    monkeypatch.setattr(TiffSeriesSource, "_read_path", read)
    if live:
        executor, identity = _live(tmp_path, raw, 8911 + denied_label)
        events = []
        try:
            _wait(executor, events, lambda values: 2 in _labels(values, "good") or any(
                event.kind in _TERMINAL for event in values), 20.)
        finally:
            terminal = _stop(executor, identity, events)
        expected = "STOPPED"
    else:
        executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=8914)
        executor.close(identity)
        terminal = [(event.kind.name, event.detail) for event in events if event.kind in _TERMINAL]
        expected = "FINISHED"
    print("SOURCE-READ-DENIAL", live, denied_label, attempts, terminal)
    assert attempts, "the real source decoder must be reached"
    assert [kind for kind, _detail in terminal] == [expected], terminal
    assert _output(tmp_path, "good")[0] == (1, 2)
