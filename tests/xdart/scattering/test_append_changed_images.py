"""A finite Append reduces a scan again whole when a processed image changed on disk, and goes on when one was only
re-dated (Codex N-a8a6c887's changed-image design, amended N-0243a1bb to the maintainer's rule; stages 1 and 2).

Maintainer, 2026-10-23: a processed image overwritten on disk is reprocessed ("reprocess the new image and overwrite
the old result"), and the start time its sidecar records tells a re-copied file from a new image. An image whose
size moved, or whose time moved without its sidecar's recorded acquisition time, makes its scan's Append a refusal
typed for a rebuild, which the Run carries out as an ordinary replacement of that one output (every current label,
the same target); one whose time alone moved while its sidecar still records the frame's acquisition time is the
same acquisition. The recorded value is the .txt "time:" clock time with no zone applied, so another host reads the
same number. A workflow rule, not byte proof. Real directory Runs over real TIFF + .txt scans."""
from __future__ import annotations

import calendar
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import fabio
import h5py
import numpy as np
import pytest

import xrd_tools.io.append as append
import xrd_tools.io.metadata as metadata_io
import xrd_tools.io.nexus as nexus
import xrd_tools.io.output_transaction as output_transaction
import xrd_tools.reduction.core as core
from tests.xdart.scattering.test_append_already_processed import _labels, _run
from tests.xdart.scattering.test_live_revalidation_cost_run import TINY, _poni, _tiff
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal
from xdart.gui.tabs.scattering.adapters import dynamic_output
from xrd_tools.core.metadata import monitor_candidates
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.append import AppendDisposition
from xrd_tools.session.run_configuration import RunIntent
from xdart.utils import log_events

_REFUSED = "source does not extend exact image-member history"


def _txt(raw, stem, k, second):
    (raw / f"{stem}_{k:04d}.txt").write_text(
        f"# Counters\nI0 = {float(k)}\n# Motors\nenergy = 16\n"
        f"User: energy, time: Mon Jan 15 10:30:{second:02d} 2024  # Temp\n")


def _frame(raw, stem, k, second=None):
    _txt(raw, stem, k, 10 + k if second is None else second)
    _tiff(raw / f"{stem}_{k:04d}.tif", k, TINY)


def _processed(tmp_path, frames=3, others=()):
    raw = tmp_path / "raw"
    raw.mkdir()
    for k in range(frames):
        _frame(raw, "scana", k)
    for stem in others:  # scans left as they are
        for k in range(2):
            _frame(raw, stem, k)
    events, _catalog = _run(tmp_path, raw, 9201)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    return raw, tmp_path / "processed" / "scana_int1d.nexus"


def _redate(path, seconds=5):
    """A re-copied file: the same bytes under a new modified time."""
    state = path.stat()
    os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns + seconds * 1_000_000_000))


def _again(tmp_path, raw, monkeypatch, request_value, **intent):
    """Append again: the Run's terminals, its Append decisions (output name, decision), each qualified (target,
    intent), its sidecar reads for moved images, and its end-of-run notes."""
    seen = SimpleNamespace(decisions=[], intents=[], reads=[])
    preview, qualify = dynamic_output.preview_append_decision, dynamic_output.qualify_append

    def recording(configuration, item, signed):
        decision = preview(configuration, item, signed)
        seen.decisions.append((Path(str(item.target)).name, decision))
        return decision

    monkeypatch.setattr(dynamic_output, "preview_append_decision", recording)
    monkeypatch.setattr(dynamic_output, "qualify_append", lambda target, value, **kwargs: (
        seen.intents.append((target, value)), qualify(target, value, **kwargs))[1])
    acquired = getattr(append, "_acquired", None)
    monkeypatch.setattr(append, "_acquired", lambda path: (seen.reads.append(Path(path).name), acquired(path))[1],
                        raising=False)
    events, _catalog = _run(tmp_path, raw, request_value, **intent)
    terminal = [event for event in events if event.kind in _TERMINAL]
    seen.terminals = [event.kind.name for event in terminal]
    seen.warnings = [warning for event in terminal for warning in event.warnings]
    seen.rebuilt = [note for event in terminal for note in getattr(event, "rebuilt", ())]
    return seen


def _rows(output):
    with h5py.File(output, "r") as handle:
        group = handle["entry/integrated_1d"]
        return group["frame_index"][()].tolist(), group["axis_1"][()], group["intensity"][()]


def _lineage(output):
    with h5py.File(output, "r") as handle:
        lineage = append._read_lineage(handle["entry"])
        value = handle["entry/reduction/config/run_configuration"][()]
    provenance = json.loads(value.decode() if isinstance(value, bytes) else value)
    return lineage["state"], [epoch["labels"] for epoch in lineage["epochs"]], provenance


def test_a_txt_scan_records_each_frames_clock_time_at_full_precision(tmp_path):
    _raw, output = _processed(tmp_path)
    with h5py.File(output, "r") as handle:
        group = handle["entry/scan_data"]
        recorded = dict(zip(group["frame_index"][()].tolist(), group["epoch_local"][()].tolist()))
        assert group["epoch_local"].dtype == np.float64
    expected = [float(calendar.timegm((2024, 1, 15, 10, 30, 10 + k, 0, 0, 0))) for k in range(3)]
    assert [recorded[label] for label in sorted(recorded)] == expected


def test_a_redated_image_whose_sidecar_records_the_same_acquisition_is_still_processed(tmp_path, monkeypatch):
    raw, output = _processed(tmp_path)
    _redate(raw / "scana_0001.tif")
    before = output.read_bytes()
    seen = _again(tmp_path, raw, monkeypatch, 9202)
    assert seen.terminals == ["FINISHED"] and seen.warnings == [] and seen.rebuilt == []
    assert [(name, decision.disposition, getattr(decision, "rebuild", None)) for name, decision in seen.decisions] == [
        ("scana_int1d.nexus", AppendDisposition.SKIP, False)]
    assert seen.reads and set(seen.reads) == {"scana_0001.tif"}  # only the moved image's sidecar is read
    assert output.read_bytes() == before
    # a Live Run (and a headless one) keeps the exact image history: the same check without the finite opt-in
    # refuses as before
    target, qualified = seen.intents[-1]
    exact = append.qualify_append(target, qualified)
    assert exact.disposition is AppendDisposition.REFUSE and not getattr(exact, "rebuild", False)
    assert _REFUSED in exact.reason


def test_a_redated_image_beside_new_frames_appends_only_the_new_frames(tmp_path, monkeypatch):
    raw, output = _processed(tmp_path)
    committed = _labels(tmp_path, "scana")
    _redate(raw / "scana_0001.tif")
    _frame(raw, "scana", 3)
    seen = _again(tmp_path, raw, monkeypatch, 9203)
    assert seen.warnings == [] and seen.rebuilt == []
    [(_name, decision)] = seen.decisions
    assert decision.disposition is AppendDisposition.WRITE and not getattr(decision, "rebuild", False)
    assert decision.labels == (committed[-1] + 1,) and _labels(tmp_path, "scana") == (*committed, committed[-1] + 1)
    # the new epoch records the moved image as it was reduced, so the next Append asks its sidecar again
    before = output.read_bytes()
    seen = _again(tmp_path, raw, monkeypatch, 9204)
    assert seen.warnings == [] and set(seen.reads) == {"scana_0001.tif"} and output.read_bytes() == before
    assert [decision.disposition for _name, decision in seen.decisions] == [AppendDisposition.SKIP]


@pytest.mark.parametrize("change", ["acquired again", "size", "acquired again with new frames",
                                    "first image smaller with new frames"])
def test_a_changed_image_reduces_its_scan_again_whole_into_the_same_output(tmp_path, monkeypatch, change):
    raw, output = _processed(tmp_path, others=("scanb",))
    committed = _labels(tmp_path, "scana")
    processed = tmp_path / "processed"
    untouched = {path: path.read_bytes() for path in sorted(processed.rglob("*scanb*")) if path.is_file()}
    assert {path.suffix for path in untouched} == {".nexus", ".xye"}
    image = raw / "scana_0001.tif"
    if change == "size":
        fabio.tifimage.TifImage(data=np.full(TINY, 7, dtype=np.uint32)).write(str(image))  # same name, new size
    elif change.startswith("first image"):  # the series' own size and time are its first image's
        fabio.tifimage.TifImage(data=np.full(TINY, 7, dtype=np.uint8)).write(str(raw / "scana_0000.tif"))
    else:
        _txt(raw, "scana", 1, second=59)  # a new acquisition time
        _tiff(image, 99, TINY)  # and new pixels, same size
        _redate(image)
    current = committed
    if change.endswith("new frames"):
        _frame(raw, "scana", 3)
        current = (*committed, committed[-1] + 1)
    hashed, capture = [], output_transaction._capture_target
    monkeypatch.setattr(output_transaction, "_capture_target", lambda target, *, hash_content=True: (
        hashed.append(hash_content) if Path(target).name == output.name else None, capture(
            target, hash_content=hash_content))[1])
    seen = _again(tmp_path, raw, monkeypatch, 9205)
    assert hashed and not any(hashed)  # an Overwrite's fast route: the old output is never hashed whole
    [(_name, decision)] = seen.decisions
    assert decision.disposition is AppendDisposition.REFUSE and getattr(decision, "rebuild", None) is True
    assert decision.labels == current and decision.committed_labels == committed
    # the refusal's own text (today's) names what changed; the Run reduces the scan again instead of skipping it
    today = "source size regressed" if change.startswith("first image") else f"{_REFUSED}: image 2 (scana_0001.tif)"
    assert seen.terminals == ["FINISHED"] and seen.warnings == []
    assert len(seen.rebuilt) == 1 and seen.rebuilt[0].startswith("Append reprocessed scana_int1d.nexus whole: ")
    assert today in seen.rebuilt[0]
    state, epochs, provenance = _lineage(output)
    assert state == "committed" and epochs == [list(current)]  # one fresh epoch, every current label
    assert provenance["output_mode"] == "Append" and provenance["append_replacement"] == {
        "requested_output_mode": "Append", "effective_output_mode": "Overwrite", "reason": decision.reason}
    rebuilt = _rows(output)
    for label, row in zip(rebuilt[0], rebuilt[2]):  # every label's side output follows the new rows
        xye = np.loadtxt(processed / "scana" / f"iq_scana_{label:04d}.xye")
        assert np.allclose(xye[:, 1], row, rtol=1e-6, atol=0)
    assert {path: path.read_bytes() for path in untouched} == untouched  # the other scan's outputs, as they were
    fresh = _run(tmp_path, raw, 9206, mode="Overwrite")  # the same images, an ordinary Overwrite
    assert [event.kind.name for event in fresh[0] if event.kind in _TERMINAL] == ["FINISHED"]
    expected = _rows(output)
    assert rebuilt[0] == expected[0] == list(current)
    assert np.array_equal(rebuilt[1], expected[1]) and np.array_equal(rebuilt[2], expected[2])


def test_the_log_frames_the_run_and_each_scan_and_says_plainly_what_it_reprocessed(tmp_path, monkeypatch, caplog):
    """Maintainer, 2026-10-24: dated banners for the Run and each scan (frames and time on each scan's end), and
    plain messages."""
    raw, _output = _processed(tmp_path)
    for k in range(2):
        _frame(raw, "scanb", k)
    _txt(raw, "scana", 1, second=59)
    _redate(raw / "scana_0001.tif")
    for flag in ("XDART_PERF", "XDART_PERF_QUARTILES"):  # the regular log: performance monitoring off
        monkeypatch.delenv(flag, raising=False)
    caplog.set_level(logging.INFO)
    caplog.clear()
    seen = _again(tmp_path, raw, monkeypatch, 9216)
    assert seen.terminals == ["FINISHED"]
    banners = [(getattr(record, log_events.BANNER), record.getMessage()) for record in caplog.records
               if getattr(record, log_events.BANNER, None)]
    assert [kind for kind, _message in banners] == ["run", "scan", "scan-end", "scan", "scan-end", "run-end"], banners
    assert banners[0][1].startswith("New Run · Int 1D, Append, 1 core · ") and " → " in banners[0][1]
    assert banners[1][1] == "New Scan · scana_int1d.nexus (reprocessed whole)"
    assert banners[2][1].startswith("End of Scan · scana_int1d.nexus · 3 frames · ") and banners[2][1].endswith(" s")
    assert banners[3][1] == "New Scan · scanb_int1d.nexus"
    assert banners[4][1].startswith("End of Scan · scanb_int1d.nexus · 2 frames · ")
    assert banners[5][1].startswith("End of Run · Finished · 5 frames · 5 files processed (5 new), 0 skipped · ")
    # diagnostics are not the regular log's (maintainer, 2026-10-24): Debug Logging or Performance Monitoring
    assert not [record.getMessage() for record in caplog.records if record.getMessage().startswith((
        "Total Frames Processed", "Total Time", "[PERF-SUMMARY]", "[PERF-RESOURCES]", "[RUN-RESOURCES]",
        "[RUN-PIPELINE", "[SOURCE-READ]"))]
    [rebuilt] = [record for record in caplog.records if "[APPEND-REBUILD]" in record.getMessage()]
    assert getattr(rebuilt, log_events.EVENT, False) and rebuilt.getMessage().startswith(
        "scana_int1d.nexus: images it already processed changed on disk, so the whole scan is reprocessed and its "
        "output replaced. Details: ")


@pytest.mark.parametrize("record", ["absent", "float32"])
def test_without_a_full_precision_recorded_time_a_redated_image_is_a_change(tmp_path, monkeypatch, record):
    """An older output records no clock time, or (before 2026-10-23) times rounded to 128 s: no proof."""
    with monkeypatch.context() as patch:
        if record == "absent":
            reader = metadata_io.read_txt_metadata
            patch.setattr(metadata_io, "read_txt_metadata", lambda *a, **k: {
                key: value for key, value in reader(*a, **k).items() if key != "epoch_local"})
        else:
            patch.setattr(nexus, "_FULL_PRECISION_SCAN_COLUMNS", frozenset({"epoch"}))
        raw, output = _processed(tmp_path)
    with h5py.File(output, "r") as handle:
        group = handle["entry/scan_data"]
        assert ("epoch_local" in group) is (record == "float32")
        assert record == "absent" or group["epoch_local"].dtype == np.float32
    _redate(raw / "scana_0001.tif")
    seen = _again(tmp_path, raw, monkeypatch, 9207)
    [(_name, decision)] = seen.decisions
    assert decision.disposition is AppendDisposition.REFUSE and getattr(decision, "rebuild", None) is True
    assert seen.terminals == ["FINISHED"] and seen.warnings == [] and len(seen.rebuilt) == 1
    with h5py.File(output, "r") as handle:  # the replacement records the clock time at full precision
        assert handle["entry/scan_data/epoch_local"].dtype == np.float64


@pytest.mark.parametrize("other", ["an image removed", "other settings"])
def test_a_moved_image_beside_another_difference_is_refused_as_before(tmp_path, monkeypatch, caplog, other):
    """Image drift never excuses another difference: the refusal stays untyped and the scan is skipped (Codex
    N-a8a6c887)."""
    raw, output = _processed(tmp_path)
    _redate(raw / "scana_0001.tif")
    if other == "an image removed":
        (raw / "scana_0002.tif").unlink()
        (raw / "scana_0002.txt").unlink()
    before = output.read_bytes()
    caplog.set_level(logging.INFO)
    seen = _again(tmp_path, raw, monkeypatch, 9208, **({"npt": 16} if other == "other settings" else {}))
    assert len(seen.warnings) == 1 and seen.rebuilt == [] and output.read_bytes() == before
    # Said plainly (maintainer, 2026-10-26: "user facing warnings and info should be simpler and more direct"); the
    # technical refusal is a debug line.
    plain = ("it was made with different integration settings" if other == "other settings"
             else "the scan now has fewer images than its output holds")
    assert any(record.getMessage().startswith(f"scana_int1d.nexus skipped: {plain}")
               and record.getMessage().endswith("its existing results are left unchanged (Replace reprocesses it). "
                                                "[APPEND-SKIPPED]")
               for record in caplog.records), [record.getMessage() for record in caplog.records][-6:]
    assert all(decision.disposition is AppendDisposition.REFUSE and not getattr(decision, "rebuild", False)
               for _name, decision in seen.decisions)


def test_a_rebuild_that_fails_leaves_the_old_output_as_an_ordinary_overwrite_does(tmp_path, monkeypatch):
    raw, output = _processed(tmp_path)
    before = output.read_bytes()
    _txt(raw, "scana", 1, second=59)
    _redate(raw / "scana_0001.tif")
    real, calls = core.NexusSink._write_batch, []

    def failing(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk full")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(core.NexusSink, "_write_batch", failing)
    seen = _again(tmp_path, raw, monkeypatch, 9209)
    assert seen.terminals == ["FAILED"] and seen.rebuilt == []
    assert output.read_bytes() == before  # restored: the old result, its old epoch


def test_an_output_substituted_after_admission_is_never_replaced(tmp_path, monkeypatch):
    """The replacement takes the Overwrite target guard: an output that changed after its classification fails the
    construction, and the file now there is left as it is (Codex N-a8a6c887: never overwrite a newly substituted
    file because its old path was eligible)."""
    raw, output = _processed(tmp_path)
    _txt(raw, "scana", 1, second=59)
    _redate(raw / "scana_0001.tif")
    prepare = dynamic_output.DynamicOutputAdapter.prepare_admission
    substitute, substituted = tmp_path / "substitute.nexus", []

    def substituting(self, *args, **kwargs):
        prepared = prepare(self, *args, **kwargs)
        shutil.copyfile(output, substitute)
        os.replace(substitute, output)  # another file at the same path, the same bytes
        substituted.append((output.stat().st_dev, output.stat().st_ino))
        return prepared

    monkeypatch.setattr(dynamic_output.DynamicOutputAdapter, "prepare_admission", substituting)
    before = output.read_bytes()
    seen = _again(tmp_path, raw, monkeypatch, 9210)
    assert seen.terminals == ["FINISHED"] and len(seen.warnings) == 1 and seen.rebuilt == [] and len(substituted) == 1
    assert seen.warnings[0].startswith("Skipped scana_int1d.nexus: ValueError: output target changed after Background")
    assert output.read_bytes() == before and (output.stat().st_dev, output.stat().st_ino) == substituted[0]


def test_a_one_scan_run_says_at_its_end_that_it_reduced_its_scan_again(tmp_path):
    """A Run of one TIFF series, not a directory, reports its rebuild once it FINISHED."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for k in range(3):
        _frame(raw, "scana", k)
    files = tuple(str(raw / f"scana_{k:04d}.tif") for k in range(3))

    def run(request_value):
        intent = RunIntent(
            source_spec=SourceSpec(Path(files[0]), SourceKind.TIFF_SERIES,
                                   options={"files": files, "metadata_format": "txt"}),
            poni_file=str(_poni(tmp_path, TINY)), project_root=str(raw), save_path=str(tmp_path / "processed"),
            output_mode="Append", processing_mode="Int 1D", live_mode=False, max_cores=1,
            bai_1d_args={"npt": 8, "method": "numpy"}, bai_2d_args={"npt_rad": 8, "npt_azim": 8, "method": "numpy"})
        executor, identity, events = _run_to_terminal(intent, request_value=request_value)
        executor.close(identity)
        return [event for event in events if event.kind in _TERMINAL]

    [first] = run(9211)
    [output] = list((tmp_path / "processed").glob("*.nexus"))
    assert first.kind.name == "FINISHED" and getattr(first, "rebuilt", ()) == ()
    _txt(raw, "scana", 1, second=59)
    _tiff(raw / "scana_0001.tif", 99, TINY)
    [second] = run(9212)
    assert second.kind.name == "FINISHED" and second.warnings == ()
    assert len(getattr(second, "rebuilt", ())) == 1
    assert second.rebuilt[0].startswith(f"Append reprocessed {output.name} whole: ")
    assert _lineage(output)[:2] == ("committed", [[1, 2, 3]])


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="POSIX time zones")
def test_the_recorded_clock_time_reads_the_same_in_another_time_zone(tmp_path):
    sidecar = tmp_path / "scan_0001.txt"
    sidecar.write_text("# Counters\nI0 = 1\n# Motors\nenergy = 16\nUser: energy, time: Mon Jan 15 10:30:11 2024  # Temp\n")
    zone = os.environ.get("TZ")
    read = []
    try:
        for name in ("UTC", "America/Los_Angeles", "Asia/Tokyo"):
            os.environ["TZ"] = name
            time.tzset()
            read.append(metadata_io.read_txt_metadata(sidecar))
    finally:
        if zone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = zone
        time.tzset()
    assert len({values["epoch"] for values in read}) == 3  # the zone-dependent instant
    assert {values["epoch_local"] for values in read} == {float(calendar.timegm((2024, 1, 15, 10, 30, 11, 0, 0, 0)))}


def test_the_recorded_clock_time_is_never_offered_as_a_monitor(tmp_path):
    _raw, output = _processed(tmp_path)
    with h5py.File(output, "r") as handle:
        columns = [name for name in handle["entry/scan_data"] if name != "frame_index"]
    assert "epoch_local" in columns and monitor_candidates(columns) == ("I0", "energy")


def test_a_replacement_admits_only_the_output_its_classification_read(tmp_path, monkeypatch):
    """Codex N-d3ec12d9 R1 (its reviewer regression, adopted): another file put at the output's path after the
    executor's guard, as the adapter activates, is never overwritten; the replacement's transaction compares the
    file it admits with the one classified."""
    raw, output = _processed(tmp_path)
    _txt(raw, "scana", 1, second=59)
    _redate(raw / "scana_0001.tif")
    activate, substitute, observed = dynamic_output.DynamicOutputAdapter.activate, tmp_path / "substitute.nexus", []

    def substituting(self, preparation, *args, **kwargs):
        assert preparation.effective.disposition.name == "REPLACE"
        shutil.copyfile(output, substitute)
        with h5py.File(substitute, "r+") as handle:
            handle["entry"].attrs["substitute"] = "arrived after the executor guard"
        os.replace(substitute, output)
        observed.append((output.read_bytes(), output.stat().st_ino))
        return activate(self, preparation, *args, **kwargs)

    monkeypatch.setattr(dynamic_output.DynamicOutputAdapter, "activate", substituting)
    seen = _again(tmp_path, raw, monkeypatch, 9213)
    # the scan is skipped and named; the Run finishes (maintainer, 2026-10-24: "a directory run shouldn't stop")
    assert len(observed) == 1 and seen.terminals == ["FINISHED"] and seen.rebuilt == []
    assert [warning for warning in seen.warnings if "scana_int1d.nexus" in warning], seen.warnings
    assert output.read_bytes() == observed[0][0] and output.stat().st_ino == observed[0][1]


@pytest.mark.parametrize(("clock", "unambiguous"), [
    ((2026, 11, 1, 1, 30, 11), False),   # repeated by the US fall-back (and others that day)
    ((2026, 10, 25, 2, 30, 0), False),   # repeated by the European fall-back
    ((2026, 3, 8, 2, 30, 0), False),     # skipped by the US spring-forward
    ((2026, 11, 1, 12, 0, 0), True),     # the same day, past every change
    ((2024, 1, 15, 10, 30, 11), True),
])
def test_only_a_clock_time_that_names_one_instant_in_every_zone_is_recorded(tmp_path, clock, unambiguous):
    """Codex N-d3ec12d9 R2: the .txt "time:" has no zone, so a time some zone's daylight-saving change repeats or
    skips is no proof of an acquisition, wherever the review runs."""
    when = datetime(*clock)
    sidecar = tmp_path / "scan_0001.txt"
    sidecar.write_text("# Counters\nI0 = 1\n# Motors\nenergy = 16\n"
                       f"User: energy, time: {when.strftime('%a %b %d %H:%M:%S %Y')}  # Temp\n")
    values = metadata_io.read_txt_metadata(sidecar)
    assert "epoch" in values and ("epoch_local" in values) is unambiguous


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="POSIX time zones")
def test_a_new_acquisition_at_a_repeated_clock_time_is_reduced_again(tmp_path, monkeypatch):
    """Codex N-d3ec12d9 R2 (its reviewer regression, adopted): America/Los_Angeles, 2026-11-01 01:30:11 names two
    instants an hour apart; a same-size TIFF with new pixels whose new sidecar repeats that clock time is a change."""
    zone = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        raw = tmp_path / "raw"
        raw.mkdir()
        repeated = lambda k: (raw / f"scana_{k:04d}.txt").write_text(
            f"# Counters\nI0 = {float(k)}\n# Motors\nenergy = 16\n"
            f"User: energy, time: Sun Nov 01 01:30:{10 + k:02d} 2026  # Temp\n")
        for k in range(3):
            repeated(k)
            _tiff(raw / f"scana_{k:04d}.tif", k, TINY)
        events, _catalog = _run(tmp_path, raw, 9214)
        assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
        output = tmp_path / "processed" / "scana_int1d.nexus"
        image = raw / "scana_0001.tif"
        size = image.stat().st_size
        _tiff(image, 99, TINY)
        assert image.stat().st_size == size
        _redate(image, seconds=3600)
        repeated(1)
        before = output.read_bytes()
        seen = _again(tmp_path, raw, monkeypatch, 9215)
        assert seen.terminals == ["FINISHED"] and len(seen.rebuilt) == 1 and output.read_bytes() != before
    finally:
        if zone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = zone
        time.tzset()
