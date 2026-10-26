"""A Live Run ends on the last output it wrote, though that writer closed before a saved scan it then showed (Codex
reviews N-d7f95ae0 TT-R1 and N-85413c4d TT-R2, their discriminators; maintainer 2026-10-26: "Last frame processed").
Real Live Runs, real files: scanb's writer retires when scanb stops growing, or its container fails on its own third
frame with two committed; then already-processed scanc appears, then Stop. A container that commits nothing is not
the last written."""
from pathlib import Path

from tests.xdart.scattering.test_live_already_processed import _intent
from tests.xdart.scattering.test_live_changed_images import _finished, _lineage, _live, _reacquire
from tests.xdart.scattering.test_live_never_waits import _stack
from tests.xdart.scattering.test_live_qualify_refusal_run import (
    _frame, _labels, _output, _said, _settle, _stop, _terminal, _wait,
)
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xrd_tools.sources.selection import DirectorySourceSpec


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


def test_retained_container_rows_choose_the_last_processed_output(tmp_path):
    """Codex review N-85413c4d TT-R2, its discriminator: a container whose third frame fails keeps frames 0 and 1
    committed; then saved scanc appears, then Stop. The end names scanb with its two durable rows."""
    raw = _finished(tmp_path, "scanc", 1)
    saved_sources = tuple(raw.glob("scanc_*"))
    hidden = tmp_path / "not_yet_listed"
    hidden.mkdir()
    for saved in saved_sources:
        saved.rename(hidden / saved.name)
    intent = _intent(tmp_path, raw, live=True)
    intent.source_spec = DirectorySourceSpec(raw, suffixes=(".tif", ".nxs"), metadata_format="txt")
    executor = StandardRunExecutor(join_timeout=2.)
    identity, events = _start(executor, intent, request_value=9784), []
    try:
        _stack(raw / "scanb.nxs", 3, corrupt=2)
        _wait(executor, events, lambda values: _said(values, "stopped growing") or any(
            e.kind in _TERMINAL for e in values), timeout=20.)
        assert not any(e.kind in _TERMINAL for e in events)
        for saved in saved_sources:
            (hidden / saved.name).rename(saved)
        _wait(executor, events, lambda values: any(key.source_scan == "scanc"
              for key in executor.frame_catalog(identity).entries), timeout=20.)
        _settle(executor, events, .3)
    finally:
        terminal = _stop(executor, identity, events)
    stopped = next(e for e in events if e.kind is StandardEventKind.STOPPED)
    lineage = _lineage(tmp_path / "processed" / "scanb_int1d.nexus")[0]
    assert [kind for kind, _detail in terminal] == ["STOPPED"]
    assert _output(tmp_path, "scanb")[0] == (0, 1)
    assert lineage == "committed"
    assert _labels(events, "scanc") == []
    assert Path(stopped.artifact).name == "scanb_int1d.nexus"
    assert stopped.artifact_completed == 2

def test_a_container_that_fails_before_committing_a_frame_is_not_the_last_written(tmp_path):
    """The control: its first frame fails, so its sink commits nothing in this Run; the end keeps the last scan shown."""
    raw = _finished(tmp_path, "scanc", 1)
    saved_sources = tuple(raw.glob("scanc_*"))
    hidden = tmp_path / "not_yet_listed"
    hidden.mkdir()
    for saved in saved_sources:
        saved.rename(hidden / saved.name)
    intent = _intent(tmp_path, raw, live=True)
    intent.source_spec = DirectorySourceSpec(raw, suffixes=(".tif", ".nxs"), metadata_format="txt")
    executor = StandardRunExecutor(join_timeout=2.)
    identity, events = _start(executor, intent, request_value=9785), []
    try:
        _stack(raw / "scanb.nxs", 3, corrupt=0)
        _wait(executor, events, lambda values: _said(values, "stopped growing") or _said(values, "Skipped") or any(
            e.kind in _TERMINAL for e in values), timeout=20.)
        assert not any(e.kind in _TERMINAL for e in events)
        for saved in saved_sources:
            (hidden / saved.name).rename(saved)
        _wait(executor, events, lambda values: any(key.source_scan == "scanc"
              for key in executor.frame_catalog(identity).entries), timeout=20.)
        _settle(executor, events, .3)
    finally:
        terminal = _stop(executor, identity, events)
    stopped = next(e for e in events if e.kind is StandardEventKind.STOPPED)
    assert [kind for kind, _detail in terminal] == ["STOPPED"]
    assert Path(stopped.artifact).name == "scanc_int1d.nexus", stopped
