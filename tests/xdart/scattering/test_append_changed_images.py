"""An Append tells a processed image that changed on disk from one that was only re-dated (Codex N-a8a6c887's
changed-image design, amended N-0243a1bb to the maintainer's rule; stage 1, the headless classification).

Maintainer, 2026-10-23: a processed image overwritten on disk is reprocessed ("reprocess the new image and overwrite
the old result"), and the start time its sidecar records tells a re-copied file from a new image. An image whose
size moved, or whose time moved without its sidecar's recorded acquisition time, makes its scan's Append a refusal
typed for a whole-scan rebuild (every current label); one whose time alone moved while its sidecar still records the
frame's acquisition time is the same acquisition. The recorded value is the .txt "time:" clock time with no zone
applied, so another host reads the same number. A workflow rule, not byte proof. Real directory Runs over real TIFF +
.txt scans."""
from __future__ import annotations

import calendar
import os
import time
from pathlib import Path

import fabio
import h5py
import numpy as np
import pytest

import xrd_tools.io.append as append
import xrd_tools.io.metadata as metadata_io
import xrd_tools.io.nexus as nexus
from tests.xdart.scattering.test_append_already_processed import _labels, _run
from tests.xdart.scattering.test_live_revalidation_cost_run import TINY, _tiff
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL
from xdart.gui.tabs.scattering.adapters import dynamic_output
from xrd_tools.core.metadata import monitor_candidates
from xrd_tools.io.append import AppendDisposition

_REFUSED = "source does not extend exact image-member history"


def _txt(raw, stem, k, second):
    (raw / f"{stem}_{k:04d}.txt").write_text(
        f"# Counters\nI0 = {float(k)}\n# Motors\nenergy = 16\n"
        f"User: energy, time: Mon Jan 15 10:30:{second:02d} 2024  # Temp\n")


def _frame(raw, stem, k, second=None):
    _txt(raw, stem, k, 10 + k if second is None else second)
    _tiff(raw / f"{stem}_{k:04d}.tif", k, TINY)


def _processed(tmp_path, frames=3):
    raw = tmp_path / "raw"
    raw.mkdir()
    for k in range(frames):
        _frame(raw, "scana", k)
    events, _catalog = _run(tmp_path, raw, 9201)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    return raw, tmp_path / "processed" / "scana_int1d.nexus"


def _redate(path, seconds=5):
    """A re-copied file: the same bytes under a new modified time."""
    state = path.stat()
    os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns + seconds * 1_000_000_000))


intents = []


def _again(tmp_path, raw, monkeypatch, request_value, **intent):
    """Append again; the Run's Append decisions (output name, decision) and its sidecar reads for moved images
    (``intents`` keeps each qualified (target, intent))."""
    decisions, reads = [], []
    preview = dynamic_output.preview_append_decision

    def recording(configuration, item, signed):
        decision = preview(configuration, item, signed)
        decisions.append((Path(str(item.target)).name, decision))
        return decision

    monkeypatch.setattr(dynamic_output, "preview_append_decision", recording)
    qualify = dynamic_output.qualify_append
    monkeypatch.setattr(dynamic_output, "qualify_append", lambda target, intent, **kwargs: (
        intents.append((target, intent)), qualify(target, intent, **kwargs))[1])
    acquired = getattr(append, "_acquired", None)
    monkeypatch.setattr(append, "_acquired", lambda path: (reads.append(Path(path).name), acquired(path))[1],
                        raising=False)
    events, _catalog = _run(tmp_path, raw, request_value, **intent)
    warnings = [warning for event in events if event.kind in _TERMINAL for warning in event.warnings]
    return decisions, reads, warnings


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
    decisions, reads, warnings = _again(tmp_path, raw, monkeypatch, 9202)
    assert warnings == []
    assert [(name, decision.disposition, getattr(decision, "rebuild", None)) for name, decision in decisions] == [
        ("scana_int1d.nexus", AppendDisposition.SKIP, False)]
    assert reads and set(reads) == {"scana_0001.tif"}  # only the moved image's sidecar is read
    assert output.read_bytes() == before
    # a Live Run (and a headless one) keeps the exact image history, so the same check without the finite opt-in
    # refuses as before
    target, qualified = intents[-1]
    exact = append.qualify_append(target, qualified)
    assert exact.disposition is AppendDisposition.REFUSE and not getattr(exact, "rebuild", False)
    assert _REFUSED in exact.reason


def test_a_redated_image_beside_new_frames_appends_only_the_new_frames(tmp_path, monkeypatch):
    raw, output = _processed(tmp_path)
    committed = _labels(tmp_path, "scana")
    _redate(raw / "scana_0001.tif")
    _frame(raw, "scana", 3)
    decisions, _reads, warnings = _again(tmp_path, raw, monkeypatch, 9203)
    assert warnings == []
    [(_name, decision)] = decisions
    assert decision.disposition is AppendDisposition.WRITE and not getattr(decision, "rebuild", False)
    assert decision.labels == (committed[-1] + 1,) and _labels(tmp_path, "scana") == (*committed, committed[-1] + 1)
    # the new epoch records the moved image as it was reduced, so the next Append asks its sidecar again
    before = output.read_bytes()
    decisions, reads, warnings = _again(tmp_path, raw, monkeypatch, 9204)
    assert warnings == [] and set(reads) == {"scana_0001.tif"} and output.read_bytes() == before
    assert [decision.disposition for _name, decision in decisions] == [AppendDisposition.SKIP]


@pytest.mark.parametrize("change", ["acquired again", "size", "acquired again with new frames",
                                    "first image smaller with new frames"])
def test_a_changed_image_refuses_its_scan_as_a_whole_rebuild(tmp_path, monkeypatch, change):
    raw, output = _processed(tmp_path)
    committed = _labels(tmp_path, "scana")
    image = raw / "scana_0001.tif"
    if change == "size":
        fabio.tifimage.TifImage(data=np.full(TINY, 7, dtype=np.uint32)).write(str(image))  # same name, new size
    elif change.startswith("first image"):  # the series' own size and time are its first image's
        fabio.tifimage.TifImage(data=np.full(TINY, 7, dtype=np.uint8)).write(str(raw / "scana_0000.tif"))
    else:
        _txt(raw, "scana", 1, second=59)  # a new acquisition time
        _redate(image)
    current = committed
    if change.endswith("new frames"):
        _frame(raw, "scana", 3)
        current = (*committed, committed[-1] + 1)
    before = output.read_bytes()
    decisions, _reads, warnings = _again(tmp_path, raw, monkeypatch, 9205)
    # the refusal reads as today's; its type says a rebuild may follow
    today = "source size regressed" if change.startswith("first image") else f"{_REFUSED}: image 2 (scana_0001.tif)"
    assert len(warnings) == 1 and today in warnings[0]
    [(_name, decision)] = decisions
    assert decision.disposition is AppendDisposition.REFUSE and getattr(decision, "rebuild", None) is True
    assert decision.labels == current and decision.committed_labels == committed
    assert output.read_bytes() == before  # stage 1 classifies; nothing consumes the rebuild yet


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
    decisions, _reads, warnings = _again(tmp_path, raw, monkeypatch, 9206)
    assert len(warnings) == 1 and _REFUSED in warnings[0]
    [(_name, decision)] = decisions
    assert decision.disposition is AppendDisposition.REFUSE and getattr(decision, "rebuild", None) is True


@pytest.mark.parametrize("other", ["an image removed", "other settings"])
def test_a_moved_image_beside_another_difference_is_refused_as_before(tmp_path, monkeypatch, other):
    """Image drift never excuses another difference: the refusal stays untyped (Codex N-a8a6c887)."""
    raw, _output = _processed(tmp_path)
    _redate(raw / "scana_0001.tif")
    if other == "an image removed":
        (raw / "scana_0002.tif").unlink()
        (raw / "scana_0002.txt").unlink()
    decisions, _reads, warnings = _again(tmp_path, raw, monkeypatch, 9207, **(
        {"npt": 16} if other == "other settings" else {}))
    assert len(warnings) == 1
    assert all(decision.disposition is AppendDisposition.REFUSE and not getattr(decision, "rebuild", False)
               for _name, decision in decisions)


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="POSIX time zones")
def test_the_recorded_clock_time_reads_the_same_in_another_time_zone(tmp_path, monkeypatch):
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

