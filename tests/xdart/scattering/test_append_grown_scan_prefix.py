"""A directory Append over a scan that grew reads only its new frames (maintainer, 2026-10-23: "the idea of append is
to quickly skip over already processed scans/frames ... If a frame has already been processed that means the image data
as well as the metadata if any were already verified. It seems wasteful to recheck that").

The frames its output already holds are recognized from the directory listing and the output's committed record, and
their graph is the one that record wrote: they are not probed, captured, qualified or read again; their energy is the
selection the output saved, extended by the new frames when it settles the result exactly, else they are read as before.
Real Runs over real TIFF + .txt scans."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import fabio
import h5py
import numpy as np
import pytest

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
    assert len(grown) == 1 and grown[0] is not None and len(grown[0][0].stamp.members) == N
    # Each processed frame: no probe, no capture, no metadata read (were seven).
    assert (probes["probe"], captures["capture"], reads["read"]) == (0, 0, 0), (probes, captures, reads)
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


def _txt_frame(raw, index, i0, energy):
    """One frame: its .txt (``energy`` in keV, none when None), then its image renamed into place whole."""
    motors = "" if energy is None else f"# Motors\nenergy = {energy!r}\n"
    raw.joinpath(f"grow_{index:04d}.txt").write_text(
        f"# Counters\nI0 = {i0}\n{motors}User: energy, time: Mon Jan 15 10:30:00 2024  # Temp\n")
    partial = raw / f"grow_{index:04d}.part"
    fabio.tifimage.TifImage(data=np.full((195, 487), index, dtype=np.uint16)).write(str(partial))
    partial.replace(raw / f"grow_{index:04d}.tif")


def _grown_energy(tmp_path, monkeypatch, before, after, *, prefix):
    """(terminal kinds, reads of the processed frames' metadata, the saved energy selection) after a scan of
    ``before`` energies grows by ``after`` ones; ``prefix`` False sends it the full way."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    for index, energy in enumerate(before, 1):
        _txt_frame(raw, index, float(index), energy)
    assert [event.kind.name for event in _run(tmp_path, raw, 9811)] == ["FINISHED"]
    for index, energy in enumerate(after, len(before) + 1):
        _txt_frame(raw, index, float(index), energy)
    reads, real_read = Counter(), metadata._read_image_metadata_observed_once

    def read(path, *args, **kwargs):
        if int(Path(str(path)).name[5:9]) <= len(before):
            reads["read"] += 1
        return real_read(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(metadata, "_read_image_metadata_observed_once", read)
        if not prefix:
            patch.setattr(output_preflight, "_committed_prefix", lambda *_args, **_kwargs: None)
        terminals = _run(tmp_path, raw, 9812)
    with h5py.File(tmp_path / "processed" / "grow_int1d.nexus", "r") as handle:
        selection = json.loads(handle["entry/reduction/config/energy_selection"][()])
    return [event.kind.name for event in terminals], reads["read"], selection


@pytest.mark.parametrize("case, before, after, settled", [
    ("fixed", [16.0] * 6, [16.0], True),
    ("energy scan", [16.0, 16.1, 16.2, 16.3, 16.4, 16.5], [16.2, 16.4], True),  # within its span: no wider grid
    ("energy scan, a glitched new frame", [16.0, 16.1, 16.2, 16.3, 16.4, 16.5], [0.0, 16.3], True),
    ("no energy recorded", [None] * 6, [None], True),
    ("a common energy that now varies", [16.0] * 6, [16.2], False),  # read again, then reduced whole on its grid
])
def test_a_grown_scans_energy_is_the_full_ways_without_reading_its_processed_frames(
        tmp_path, monkeypatch, case, before, after, settled):
    """The Run ends as the full way's does, with the same saved energy selection; its processed frames' .txt are read
    only when the saved selection cannot settle it (test_energy_scan_append_grid.py pins the wider-grid rebuild)."""
    kinds, read, selection = _grown_energy(tmp_path / "grown", monkeypatch, before, after, prefix=True)
    full_kinds, _full_read, full = _grown_energy(tmp_path / "full", monkeypatch, before, after, prefix=False)
    assert kinds == full_kinds == ["FINISHED"], (case, kinds, full_kinds)
    assert (read == 0) is settled, (case, read)
    assert selection == full, (case, selection, full)
