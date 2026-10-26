"""A Live attempt reads no image of a scan whose metadata is not ready yet (beamline 2026-10-26: a Live Run's first poll
probed the images of all 148 scans in the folder, 32 s on the share, then admitted none, each sidecar having been seen
only once; the first frame came 52 s after the Run began). A Run over a folder that already holds its scans: every
attempt that finds no member ready probes nothing, and each scan is still processed with its own metadata. The real
executor, session and admission; the attempt, readiness and probe calls are only observed."""
from __future__ import annotations

from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _labels, _output, _run, _settle, _stop, _wait
from xdart.gui.tabs.scattering import output_preflight
from xdart.gui.tabs.scattering.adapters import run_executor
from xrd_tools.sources.directory_session import DirectoryIndexSession

STEMS = tuple(f"s{index}" for index in range(6))


def test_an_attempt_whose_metadata_is_not_ready_probes_no_image(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    for index, stem in enumerate(STEMS):
        _frame(raw, stem, 1, "txt", float(index + 1))
    calls: list[tuple[str, int]] = []
    attempt, ready, probe = (run_executor.materialize_live_directory_group, output_preflight._live_metadata_ready,
                             DirectoryIndexSession.probe_candidates)

    def observed_attempt(*args, **kwargs):
        calls.append(("attempt", 0))
        return attempt(*args, **kwargs)

    def observed_ready(*args, **kwargs):
        result = ready(*args, **kwargs)
        calls.append(("ready", result[0]))
        return result

    def observed_probe(self, candidates, *args, **kwargs):
        candidates = tuple(candidates)
        calls.append(("probe", len(candidates)))
        return probe(self, candidates, *args, **kwargs)

    monkeypatch.setattr(run_executor, "materialize_live_directory_group", observed_attempt)
    monkeypatch.setattr(output_preflight, "_live_metadata_ready", observed_ready)
    monkeypatch.setattr(DirectoryIndexSession, "probe_candidates", observed_probe)
    executor, identity = _run(tmp_path, raw, "txt", 3601)
    events = []
    try:
        _wait(executor, events, lambda values: all(1 in _labels(values, stem) for stem in STEMS), 60.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    attempts, segment = [], None
    for kind, value in calls:
        if kind == "attempt":
            segment = []
            attempts.append(segment)
        elif segment is not None:
            segment.append((kind, value))
    waited = [segment for segment in attempts if [value for kind, value in segment if kind == "ready"][-1:] == [0]]
    assert waited, "no attempt found its metadata unready: the first look at each sidecar should wait"
    assert not any(kind == "probe" for segment in waited for kind, _value in segment), waited
    for index, stem in enumerate(STEMS):
        assert _output(tmp_path, stem)[:2] == ((1,), (float(index + 1),)), stem
