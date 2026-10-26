"""A Live Run ends on the last output it wrote, though that writer closed before a saved scan it then showed (Codex
review N-d7f95ae0 TT-R1, its discriminator; maintainer 2026-10-26: "Last frame processed"). Real Live Run, real
files: scanb's writer retires when scanb stops growing, then already-processed scanc appears, then Stop."""
from pathlib import Path

from tests.xdart.scattering.test_live_changed_images import _finished, _live, _reacquire
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _labels, _said, _settle, _stop, _terminal, _wait
from xdart.gui.tabs.scattering.display_values import StandardEventKind


def test_last_processed_target_survives_live_retirement_then_saved_presentation(tmp_path):
    raw = _finished(tmp_path, "scanc", 1)
    hidden = tmp_path / "not_yet_listed"
    hidden.mkdir()
    saved_sources = tuple(raw.glob("scanc_*"))
    assert len(saved_sources) == 2
    for path in saved_sources:
        path.rename(hidden / path.name)
    executor, identity = _live(tmp_path, raw, 9781)
    events = []
    before_saved = None
    try:
        for label in (1, 2):
            _frame(raw, "scanb", label, "txt", float(label))
            _wait(executor, events, lambda values: label in _labels(values, "scanb") or _terminal(values))
        _settle(executor, events, .5)
        _reacquire(raw, "scanb", 1)
        _frame(raw, "scanb", 3, "txt", 3.)
        _wait(executor, events, lambda values: _said(values, "stopped growing") or _terminal(values), timeout=30.)
        assert not _terminal(events)
        run = executor._exact_run(identity)
        assert run.output is not None and not run.output.has_current
        before_saved = (str(run.artifact), run.current_completed, run.last_written)
        for path in saved_sources:
            (hidden / path.name).rename(path)
        _wait(executor, events, lambda values: any(key.source_scan == "scanc"
              for key in executor.frame_catalog(identity).entries) or _terminal(values), timeout=30.)
        _settle(executor, events, .3)
    finally:
        terminal = _stop(executor, identity, events)
    stopped = next(event for event in events if event.kind is StandardEventKind.STOPPED)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert _labels(events, "scanb") == [1, 2]
    assert Path(stopped.artifact).name == "scanb_int1d.nexus", (before_saved, stopped)
    assert stopped.artifact_completed == 2
