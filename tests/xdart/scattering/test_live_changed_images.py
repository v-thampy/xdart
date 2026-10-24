"""A Live Run reduces a scan again whole, into its output, when it finds one of that scan's processed images changed
on disk (Codex N-a8a6c887's changed-image design, stage 3).

An output this Run did not write, or one it showed from its saved record, is rebuilt once into a fresh lineage (its
old frames leave navigation and a new display generation starts); later quiet polls do nothing and later frames
extend the new lineage. In Live any moved image is a change (the re-dated-image proof is a finite Append's). A scan
this Run is growing itself still stops growing when one of its reduced images changes (maintainer, 2026-10-24: "Stop
growing it"), and the next Run finds it changed and rebuilds it. Every other refusal still stops that output growing.
Real Live Runs over real TIFF + .txt scans."""
from __future__ import annotations

import logging
import os
import threading
from collections import Counter
from pathlib import Path

import fabio
import h5py
import numpy as np
import pytest

import xrd_tools.io.append as append
import xrd_tools.io.metadata as metadata
from tests.xdart.scattering.test_live_already_processed import _intent
from tests.xdart.scattering.test_live_qualify_refusal_run import (
    _frame, _labels, _said, _settle, _stop, _terminals, _wait,
)
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal, _start
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor


def _reacquire(raw, stem, index, value=99):
    """A new image under the same name and size (its time moves)."""
    target, partial = raw / f"{stem}_{index:04d}.tif", raw / f"{stem}_{index:04d}.part"
    fabio.tifimage.TifImage(data=np.full((195, 487), value, dtype=np.uint16)).write(str(partial))
    partial.replace(target)


def _held_poll(monkeypatch):
    """Once armed, the Live loop's next listing waits for ``released``: what changes meanwhile, one poll sees."""
    armed, inside, released = threading.Event(), threading.Event(), threading.Event()
    real = DirectoryIndexSession.observe

    def observe(self, *, refresh=True):
        if armed.is_set() and not inside.is_set() and threading.current_thread() is not threading.main_thread():
            inside.set()
            released.wait(30.)
        return real(self, refresh=refresh)
    monkeypatch.setattr(DirectoryIndexSession, "observe", observe)
    return armed, inside, released


def _acquired_again_with_its_sidecar(raw, stem, index, held, then=None):
    """A new image and a new acquisition time in its sidecar (and ``then``), all seen by one poll."""
    armed, inside, released = held
    armed.set()
    assert inside.wait(30.), "the Live loop waits at its next listing"
    _reacquire(raw, stem, index)
    sidecar = raw / f"{stem}_{index:04d}.txt"
    sidecar.write_text(sidecar.read_text().replace("10:30:00", "10:31:00"))
    if then is not None:
        then()
    released.set()


def _finished(tmp_path, stem, frames):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    for index in range(1, frames + 1):
        _frame(raw, stem, index, "txt", float(index))
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=8701)
    executor.close(identity)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ["FINISHED"]
    return raw


def _lineage(output):
    with h5py.File(output, "r") as handle:
        lineage = append._read_lineage(handle["entry"])
        rows = handle["entry/integrated_1d/frame_index"][()].tolist(), handle["entry/integrated_1d/intensity"][()]
    return lineage["state"], [epoch["labels"] for epoch in lineage["epochs"]], rows


def _live(tmp_path, raw, request_value):
    executor = StandardRunExecutor(join_timeout=2.)
    return executor, _start(executor, _intent(tmp_path, raw, live=True), request_value=request_value)


def _rebuilt(events):
    return [note for event in events if event.kind in _TERMINAL for note in getattr(event, "rebuilt", ())]


def _catalog(executor, identity, stem):
    return Counter(key.local_frame_label for key in executor.frame_catalog(identity).entries
                   if f"{stem}_int1d" in str(key.artifact))


def test_live_rebuilds_a_finished_scan_whose_image_changed_once_then_extends_the_new_lineage(tmp_path):
    raw = _finished(tmp_path, "done", 2)
    output = tmp_path / "processed" / "done_int1d.nexus"
    old = _lineage(output)[2][1]
    _reacquire(raw, "done", 1)
    executor, identity = _live(tmp_path, raw, 8702)
    events = []
    try:
        _wait(executor, events, lambda values: sorted(set(_labels(values, "done"))) == [1, 2] or any(
            event.kind in _TERMINAL for event in values))
        _settle(executor, events, 1.)
        quiet = len(_labels(events, "done"))
        _frame(raw, "done", 3, "txt", 3.0)
        _wait(executor, events, lambda values: 3 in _labels(values, "done") or any(
            event.kind in _TERMINAL for event in values))
        _settle(executor, events, .5)
        catalog = _catalog(executor, identity, "done")
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert not _said(events, "stopped growing"), _terminals(events)
    assert quiet == 2 and _labels(events, "done") == [1, 2, 3]  # rebuilt once; the quiet polls wrote nothing
    [note] = _rebuilt(events)
    assert note.startswith("Append reprocessed done_int1d.nexus whole: ")
    state, epochs, (labels, intensity) = _lineage(output)
    assert state == "committed" and epochs == [[1, 2], [3]] and labels == [1, 2, 3]
    assert not np.array_equal(intensity[0], old[0]) and np.array_equal(intensity[1], old[1])  # the new image's row
    assert catalog == {1: 1, 2: 1, 3: 1}, catalog


@pytest.mark.parametrize("change", ["acquired again", "re-dated", "acquired again with its sidecar"])
def test_a_scan_this_live_run_grows_stops_growing_when_a_reduced_image_changes_and_the_next_run_rebuilds_it(
        tmp_path, caplog, monkeypatch, change):
    """Maintainer, 2026-10-24 ("Stop growing it"): its frames so far kept, its later frames skipped this Run and
    named; a later Run finds the scan changed and reduces it whole."""
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    held = _held_poll(monkeypatch)
    executor, identity = _live(tmp_path, raw, 8703)
    output = tmp_path / "processed" / "grow_int1d.nexus"
    events = []
    try:
        for index in (1, 2):
            _frame(raw, "grow", index, "txt", float(index))
            _wait(executor, events, lambda values: index in _labels(values, "grow") or any(
                event.kind in _TERMINAL for event in values))
        _settle(executor, events, .5)
        if change == "re-dated":  # in Live a re-dated image is a change too
            image = raw / "grow_0001.tif"
            state = image.stat()
            os.utime(image, ns=(state.st_atime_ns, state.st_mtime_ns + 5_000_000_000))
        elif change == "acquired again":
            _reacquire(raw, "grow", 1)  # a reduced image changes while the Run grows its scan
        if change != "acquired again with its sidecar":
            _frame(raw, "grow", 3, "txt", 3.0)
        else:  # its second, with the next frame: while that sidecar settles the scan is not one frame long (S3-R1)
            _acquired_again_with_its_sidecar(raw, "grow", 2, held, then=lambda: _frame(raw, "grow", 3, "txt", 3.0))
        _wait(executor, events, lambda values: _said(values, "stopped growing") or any(
            event.kind in _TERMINAL for event in values), timeout=30.)
        _settle(executor, events, .5)
        kept = output.read_bytes()
    finally:
        held[2].set()
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert "grow_int1d.nexus stopped growing: its Append was refused" in terminal[0][1], terminal
    assert _labels(events, "grow") == [1, 2] and _rebuilt(events) == []  # frame 3 skipped, nothing rebuilt
    assert any(record.levelno == logging.WARNING and record.getMessage().startswith(
        "grow_int1d.nexus stops growing: an image it already processed changed on disk; its frames so far are kept "
        "and its later frames are skipped in this Run; the next Run reprocesses the whole scan. Details: ")
        for record in caplog.records), [record.getMessage() for record in caplog.records][-10:]
    assert output.read_bytes() == kept and _lineage(output)[2][0] == [1, 2]
    executor, identity = _live(tmp_path, raw, 8707)  # the next Run: a changed scan it finds, rebuilt whole
    events = []
    try:
        _wait(executor, events, lambda values: sorted(set(_labels(values, "grow"))) == [1, 2, 3] or any(
            event.kind in _TERMINAL for event in values))
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert len(_rebuilt(events)) == 1 and not _said(events, "stopped growing")
    assert _lineage(output)[:2] == ("committed", [[1, 2, 3]])


def test_live_rebuilds_a_scan_it_showed_from_its_saved_record_when_an_image_changes(tmp_path):
    raw = _finished(tmp_path, "done", 2)
    executor, identity = _live(tmp_path, raw, 8704)
    events = []
    try:
        _wait(executor, events, lambda values: _catalog(executor, identity, "done") == {1: 1, 2: 1} or any(
            event.kind in _TERMINAL for event in values))
        shown = len(_labels(events, "done"))  # shown from its committed record, nothing written
        _reacquire(raw, "done", 2)
        _wait(executor, events, lambda values: len(_labels(values, "done")) >= shown + 2 or any(
            event.kind in _TERMINAL for event in values), timeout=30.)
        _settle(executor, events, .5)
        catalog = _catalog(executor, identity, "done")
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert not _said(events, "stopped growing"), _terminals(events)
    assert len(_rebuilt(events)) == 1 and catalog == {1: 1, 2: 1}, catalog


def test_a_scan_this_live_run_finished_before_the_next_began_is_rebuilt_when_an_image_changes(tmp_path):
    """No longer growing (the next scan's output took over), it is a scan the Run finds changed."""
    raw = tmp_path / "raw"
    raw.mkdir()
    executor, identity = _live(tmp_path, raw, 8708)
    output = tmp_path / "processed" / "first_int1d.nexus"
    events = []
    try:
        for index in (1, 2):
            _frame(raw, "first", index, "txt", float(index))
            _wait(executor, events, lambda values: index in _labels(values, "first") or any(
                event.kind in _TERMINAL for event in values))
        _frame(raw, "second", 1, "txt", 1.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "second") or any(
            event.kind in _TERMINAL for event in values))
        _settle(executor, events, .5)
        _reacquire(raw, "first", 2)
        _wait(executor, events, lambda values: _labels(values, "first").count(2) >= 2 or any(
            event.kind in _TERMINAL for event in values), timeout=30.)
        _settle(executor, events, .5)
        catalog = _catalog(executor, identity, "first")
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert not _said(events, "stopped growing") and len(_rebuilt(events)) == 1
    assert _lineage(output)[:2] == ("committed", [[1, 2]]) and catalog == {1: 1, 2: 1}, catalog


def test_a_finished_scan_whose_image_and_its_sidecar_were_acquired_again_is_rebuilt_once_its_sidecar_settles(
        tmp_path, monkeypatch):
    """Codex N-cdab3fe2 S3-R1: a sidecar still settling holds the scan; it is not a shorter scan to refuse."""
    raw = tmp_path / "raw"
    raw.mkdir()
    held = _held_poll(monkeypatch)
    executor, identity = _live(tmp_path, raw, 8791)
    events = []
    try:
        for index in (1, 2):
            _frame(raw, "first", index, "txt", float(index))
            _wait(executor, events, lambda values: index in _labels(values, "first") or any(
                event.kind in _TERMINAL for event in values))
        _frame(raw, "second", 1, "txt", 1.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "second") or any(
            event.kind in _TERMINAL for event in values))
        _settle(executor, events, .5)
        _acquired_again_with_its_sidecar(raw, "first", 2, held)
        _wait(executor, events, lambda values: _labels(values, "first").count(2) >= 2 or _said(
            values, "stopped growing") or any(event.kind in _TERMINAL for event in values), timeout=15.)
        _settle(executor, events, .5)
    finally:
        held[2].set()
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert not _said(events, "stopped growing"), terminal
    assert len(_rebuilt(events)) == 1
    assert _lineage(tmp_path / "processed" / "first_int1d.nexus")[:2] == ("committed", [[1, 2]])


def test_an_earlier_runs_changed_output_waits_for_its_reacquired_frames_sidecar_then_is_rebuilt(tmp_path, monkeypatch):
    """Codex N-f8157cee: with no graph of its own in this Run, a found output's re-acquired frame whose sidecar changes
    between the Run's first two looks is not a shorter scan to refuse; it waits, then is rebuilt whole."""
    raw = _finished(tmp_path, "done", 2)
    _reacquire(raw, "done", 2)
    real, observed = metadata.live_sidecar_observation, []

    def observation(image, *args, **kwargs):
        if Path(image).name == "done_0002.tif":
            observed.append(str(image))
            if len(observed) == 2:  # its new acquisition clock lands between the first two looks
                sidecar = Path(image).with_suffix(".txt")
                sidecar.write_text(sidecar.read_text().replace("10:30:00", "10:31:00"))
        return real(image, *args, **kwargs)

    monkeypatch.setattr(metadata, "live_sidecar_observation", observation)
    executor, identity = _live(tmp_path, raw, 8792)
    events = []
    try:
        _wait(executor, events, lambda values: 2 in _labels(values, "done") or _said(values, "stopped growing") or any(
            event.kind in _TERMINAL for event in values), timeout=15.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert len(observed) >= 2
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert not _said(events, "stopped growing"), terminal
    assert len(_rebuilt(events)) == 1
    assert _lineage(tmp_path / "processed" / "done_int1d.nexus")[:2] == ("committed", [[1, 2]])


def test_live_rebuilds_a_finished_scan_whose_image_was_only_re_dated(tmp_path):
    """In Live the re-dated-image proof is not taken: the scan is reduced again once."""
    raw = _finished(tmp_path, "done", 2)
    image = raw / "done_0001.tif"
    state = image.stat()
    os.utime(image, ns=(state.st_atime_ns, state.st_mtime_ns + 5_000_000_000))
    executor, identity = _live(tmp_path, raw, 8705)
    events = []
    try:
        _wait(executor, events, lambda values: sorted(set(_labels(values, "done"))) == [1, 2] or any(
            event.kind in _TERMINAL for event in values))
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert len(_rebuilt(events)) == 1 and not _said(events, "stopped growing")


def test_live_still_stops_growing_a_finished_scan_that_lost_an_image(tmp_path):
    """Not image drift: a refusal as before (the output stays, the status names it)."""
    raw = _finished(tmp_path, "done", 3)
    output = tmp_path / "processed" / "done_int1d.nexus"
    before = output.read_bytes()
    (raw / "done_0002.tif").unlink()
    (raw / "done_0002.txt").unlink()
    executor, identity = _live(tmp_path, raw, 8706)
    events = []
    try:
        _wait(executor, events, lambda values: _said(values, "stopped growing") or any(
            event.kind in _TERMINAL for event in values), timeout=30.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert _rebuilt(events) == [] and output.read_bytes() == before
