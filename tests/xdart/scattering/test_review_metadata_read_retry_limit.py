"""A metadata writer changing during each guarded read must still reach the wait limit."""
from pathlib import Path

import xrd_tools.io.metadata as metadata
from tests.xdart.scattering.test_live_never_waits import _live
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _labels, _said, _settle, _stop, _wait
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL
from xdart.gui.tabs.scattering.adapters import run_executor


def test_guarded_read_retries_do_not_reset_the_same_frames_metadata_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(run_executor, "_METADATA_WAIT_NOTE_S", .05)
    monkeypatch.setattr(run_executor, "_METADATA_WAIT_LIMIT_S", {"sidecar": .4, "spec": .4})
    raw = tmp_path / "raw"
    raw.mkdir()
    parse = metadata.read_txt_metadata
    reads = []

    def changing(path, *args, **kwargs):
        result = parse(path, *args, **kwargs)
        if Path(path).name == "grow_0002.txt":
            sidecar = Path(path)
            sidecar.write_text(sidecar.read_text() + "# changed during guarded metadata read\n")
            reads.append(1)
        return result

    monkeypatch.setattr(metadata, "read_txt_metadata", changing)
    executor, identity = _live(tmp_path, raw, 8920)
    events = []
    try:
        _frame(raw, "grow", 1, "txt", 1.)
        _wait(executor, events, lambda values: 1 in _labels(values, "grow") or any(
            event.kind in _TERMINAL for event in values), 15.)
        _frame(raw, "grow", 2, "txt", 7.)
        _settle(executor, events, 3.)  # several times the shortened deadline
        print("GUARDED-METADATA-RETRIES", len(reads), [(e.kind.name, e.detail) for e in events
                                                    if "metadata" in str(e.detail)])
        assert len(reads) >= 2, "the real guarded read must be exercised repeatedly"
        assert _said(events, "grow_int1d.nexus stopped growing"), "same-frame retries never reached the deadline"
        _frame(raw, "next", 1, "txt", 5.)
        _wait(executor, events, lambda values: 1 in _labels(values, "next") or any(
            event.kind in _TERMINAL for event in values), 15.)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
