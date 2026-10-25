"""A directory Append over a scan that grew reads only its new frames (maintainer, 2026-10-23: "the idea of append is
to quickly skip over already processed scans/frames ... If a frame has already been processed that means the image data
as well as the metadata if any were already verified. It seems wasteful to recheck that").

The frames its output already holds are recognized from the directory listing and the output's committed record, and
their graph is the one that record wrote: they are not probed, captured or qualified again (the scan's construction still
reads each frame's metadata once). Real Runs over real TIFF + .txt scans."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import h5py

import xrd_tools.io.metadata as metadata
import xrd_tools.sources.registry as registry
from tests.xdart.scattering.test_live_already_processed import _intent
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _output
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal
from xdart.gui.tabs.scattering import output_preflight
from xrd_tools.sources.execution_graph import SourceFileState

N = 12


def _run(tmp_path, raw, request_value):
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=request_value)
    executor.close(identity)
    return [event for event in events if event.kind in _TERMINAL]


def _record(tmp_path):
    with h5py.File(tmp_path / "processed" / "grow_int1d.nexus", "r") as handle:
        config = handle["entry/reduction/config"]
        return json.loads(config["source_execution"][()]), json.loads(config["append_lineage"][()])


def _old(path, raw):
    name = Path(str(path)).name
    return name.startswith("grow_") and int(name[5:9]) <= N


def test_a_grown_scans_processed_frames_are_not_probed_captured_or_qualified_again(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(1, N + 1):
        _frame(raw, "grow", index, "txt", float(index))
    assert [event.kind.name for event in _run(tmp_path, raw, 9801)] == ["FINISHED"]
    _frame(raw, "grow", N + 1, "txt", float(N + 1))
    reads, probes, captures, grown = Counter(), Counter(), Counter(), []
    real_read, real_probe = metadata._read_image_metadata_observed_once, registry._image_probe
    real_capture, real_prefix = SourceFileState.capture.__func__, output_preflight._committed_prefix

    def read(path, *args, **kwargs):
        if _old(path, raw):
            reads["read"] += 1
        return real_read(path, *args, **kwargs)

    def probe(path, *args, **kwargs):
        if _old(path, raw):
            probes["probe"] += 1
        return real_probe(path, *args, **kwargs)

    def capture(cls, path):
        if _old(path, raw):
            captures["capture"] += 1
        return real_capture(cls, path)

    def prefix(*args, **kwargs):
        grown.append(real_prefix(*args, **kwargs))
        return grown[-1]
    monkeypatch.setattr(metadata, "_read_image_metadata_observed_once", read)
    monkeypatch.setattr(registry, "_image_probe", probe)
    monkeypatch.setattr(SourceFileState, "capture", classmethod(capture))
    monkeypatch.setattr(output_preflight, "_committed_prefix", prefix)
    terminals = _run(tmp_path, raw, 9802)
    monkeypatch.undo()
    assert [event.kind.name for event in terminals] == ["FINISHED"], terminals
    assert len(grown) == 1 and grown[0] is not None and len(grown[0].stamp.members) == N
    # Each processed frame: no probe, no capture, one metadata read (the scan's construction), not seven.
    assert (probes["probe"], captures["capture"], reads["read"]) == (0, 0, N), (probes, captures, reads)
    labels, i0, _epoch = _output(tmp_path, "grow")
    assert labels == tuple(range(1, N + 2)) and i0 == tuple(float(index) for index in range(1, N + 2)), (labels, i0)
    # What it recorded is what the full way records for unchanged files: every member's and sidecar's exact state.
    execution, lineage = _record(tmp_path)
    images = sorted(raw.glob("grow_*.tif"))
    assert execution["member_stamps"] == [SourceFileState.capture(path).as_dict() for path in images]
    assert [source["metadata_file"] for source in execution["metadata_sources"]] == [
        SourceFileState.capture(path.with_suffix(".txt")).as_dict() for path in images]
    assert [(member["path"], member["size"], member["mtime_ns"]) for member in lineage["epochs"][-1]["source"][
        "image_members"]] == [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in images]
    # The next Run finds the whole scan processed (1.2.5's skip, from the record this one wrote).
    terminals = _run(tmp_path, raw, 9803)
    assert [event.kind.name for event in terminals] == ["FINISHED"], terminals
    assert _output(tmp_path, "grow")[0] == tuple(range(1, N + 2))


def test_a_grown_scan_whose_processed_image_changed_goes_the_full_way(tmp_path, monkeypatch):
    """The listing no longer matches the record for that frame: no prefix; the scan is looked at whole, as before."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(1, N + 1):
        _frame(raw, "grow", index, "txt", float(index))
    assert [event.kind.name for event in _run(tmp_path, raw, 9804)] == ["FINISHED"]
    _frame(raw, "grow", 3, "txt", 3.0, sidecar=False)  # the same image, rewritten: another time
    _frame(raw, "grow", N + 1, "txt", float(N + 1))
    grown, real_prefix = [], output_preflight._committed_prefix

    def prefix(*args, **kwargs):
        grown.append(real_prefix(*args, **kwargs))
        return grown[-1]
    monkeypatch.setattr(output_preflight, "_committed_prefix", prefix)
    terminals = _run(tmp_path, raw, 9805)
    assert grown == [None], grown
    assert [event.kind.name for event in terminals] == ["FINISHED"], terminals
    assert _output(tmp_path, "grow")[0] == tuple(range(1, N + 2))
