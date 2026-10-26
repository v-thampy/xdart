"""Normalization coverage across anchor changes and saved-output replacement."""
from tests.xdart.scattering.test_e6_norm_consumers import _acquisition_parts, _add_artifact, _retain
from tests.xdart.scattering.test_live_already_processed import _intent
from tests.xdart.scattering.test_live_changed_images import _catalog, _live, _reacquire
from tests.xdart.scattering.test_live_qualify_refusal_run import (
    SIDECARS, _frame, _labels, _output, _settle, _stop, _terminal, _wait,
)
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal
from xdart.gui.tabs.scattering import context_runtime
from xdart.gui.tabs.scattering.scientific_axes import NormCoverage
from xrd_tools.session.scan_norm import channel_is_partial
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor


def test_anchor_change_keeps_each_scans_accepted_gate_predecessor(monkeypatch):
    parts = _acquisition_parts([(1, 1., {"mon": 2.})], partitions=2)
    owner_b = _add_artifact(parts.display, "/out/b.nxs", "run.b")
    last, delta = _retain(parts.display, owner_b, "run.b", "/out/b.nxs", 2, 2., {"mon": 8.})
    assert parts.runtime.accept_navigation(delta, plot_mode="Overlay")
    frames = parts.runtime.navigation.frames
    assert parts.runtime.select_navigation(frames[0], frames)
    parts.runtime.capture_norm_aggregate_for_refresh()
    first = parts.runtime.norm_aggregate
    assert type(first) is NormCoverage
    accepted = {owner.identity: owner for owner in (first.current, *first.others)}
    actual_gate, calls = context_runtime.accepts_norm_aggregate, []

    def gate(held, candidate, expected):
        calls.append((held, candidate, expected))
        return actual_gate(held, candidate, expected)

    monkeypatch.setattr(context_runtime, "accepts_norm_aggregate", gate)
    assert parts.runtime.select_navigation(last, frames)
    parts.runtime.capture_norm_aggregate_for_refresh()
    assert len(calls) == 2
    for held, _candidate, expected in calls:
        assert held is accepted[expected], "an anchor change must not erase the scan's accepted predecessor"


def test_saved_scan_whole_replacement_rebuilds_its_norm_rows(tmp_path, monkeypatch):
    # A Norm Channel is chosen: only then does a Run read a saved output's scalar rows.
    monkeypatch.setattr(StandardRunExecutor, "display_normalization", True)
    raw = tmp_path / "raw"
    raw.mkdir()
    for label in (1, 2):
        path = _frame(raw, "done", label, "txt", float(label))
        sidecar = path.with_suffix(".txt")
        sidecar.write_text(sidecar.read_text().replace(f"I0 = {float(label)}\n", ""))
    seed, seed_id, seeded = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=9785)
    seed.close(seed_id)
    assert [event.kind.name for event in seeded if event.kind in _TERMINAL] == ["FINISHED"]
    executor, identity = _live(tmp_path, raw, 9786)
    events = []
    try:
        _wait(executor, events, lambda values: _catalog(executor, identity, "done") == {1: 1, 2: 1}
              or _terminal(values))
        saved_frame = next(key for key in executor.frame_catalog(identity).entries
                           if key.source_scan == "done")
        prior = executor._exact_run(identity).display.frame_norm_aggregate(saved_frame)
        assert prior.row_count == 2
        shown = len(_labels(events, "done"))
        for label in (1, 2):
            (raw / f"done_{label:04d}.txt").write_text(SIDECARS["txt"](float(label)))
        _reacquire(raw, "done", 2)
        _wait(executor, events, lambda values: len(_labels(values, "done")) >= shown + 2
              or _terminal(values), timeout=30.)
        _settle(executor, events, .3)
        frame = next(key for key in executor.frame_catalog(identity).entries
                     if key.source_scan == "done" and key.local_frame_label == 2)
        aggregate = executor._exact_run(identity).display.frame_norm_aggregate(frame)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert _output(tmp_path, "done")[:2] == ((1, 2), (1., 2.))
    assert aggregate.identity == prior.identity
    assert aggregate.revision > prior.revision
    assert aggregate.row_count == 2
    assert not channel_is_partial(aggregate, "i0")


def test_replacement_norm_revision_is_accepted_after_the_old_rows_leave():
    parts = _acquisition_parts([(1, 1., {"mon": 2.}), (2, 2., {})])
    parts.runtime.capture_norm_aggregate_for_refresh()
    prior = parts.runtime.norm_aggregate
    assert prior.row_count == 2 and channel_is_partial(prior, "mon")
    parts.display.retire_artifact_navigation(str(parts.owner.artifact))
    frame, delta = _retain(parts.display, parts.owner, parts.owner.source_scan,
                           str(parts.owner.artifact), 1, 3., {"mon": 8.})
    assert parts.runtime.accept_navigation(delta, plot_mode="Overlay")
    assert parts.runtime.select_navigation(frame, (frame,))
    parts.runtime.capture_norm_aggregate_for_refresh()
    replacement = parts.runtime.norm_aggregate
    assert replacement is parts.owner.norm_aggregate
    assert replacement.identity == prior.identity
    assert replacement.revision > prior.revision
    assert replacement.row_count == 1
    assert dict(replacement.channels) == {"mon": (8., 1)}
    assert not channel_is_partial(replacement, "mon")


def test_saved_scan_extension_keeps_its_prefix_norm_rows(tmp_path, monkeypatch):
    # A Norm Channel is chosen: only then does a Run read a saved output's scalar rows.
    monkeypatch.setattr(StandardRunExecutor, "display_normalization", True)
    raw = tmp_path / "raw"
    raw.mkdir()
    for label in (1, 2):
        _frame(raw, "done", label, "txt", float(label))
    seed, seed_id, seeded = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=9787)
    seed.close(seed_id)
    assert [event.kind.name for event in seeded if event.kind in _TERMINAL] == ["FINISHED"]
    executor, identity = _live(tmp_path, raw, 9788)
    events = []
    try:
        _wait(executor, events, lambda values: _catalog(executor, identity, "done") == {1: 1, 2: 1}
              or _terminal(values))
        _frame(raw, "done", 3, "txt", 3.)
        _wait(executor, events, lambda values: 3 in _labels(values, "done") or _terminal(values))
        frame = next(key for key in executor.frame_catalog(identity).entries
                     if key.source_scan == "done" and key.local_frame_label == 3)
        aggregate = executor._exact_run(identity).display.frame_norm_aggregate(frame)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert _output(tmp_path, "done")[:2] == ((1, 2, 3), (1., 2., 3.))
    assert aggregate.row_count == 3
    assert aggregate.channels["i0"] == (6., 3)
    assert not channel_is_partial(aggregate, "i0")
