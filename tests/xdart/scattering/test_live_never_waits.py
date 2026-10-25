"""A Live or directory Run never stops or waits forever on one scan (maintainer, 2026-10-24: "I don't want a live run
to stop or be waiting forever"; "if something fails, a live run or a directory run shouldn't stop. That scan/frame
should be logged and skipped and the remaining scan processing should continue").

A frame waiting for its own metadata is noted in the log and the Live status ("the log output should show it is
waiting"); after the limit ("10 s for .txt, 30 s for SPEC") it is processed without it ("a give up and processing
without the metadata is preferable"), or, when the Run normalizes by a monitor from it, its scan is skipped from that
frame on ("it is fine to skip the entire scan altogether"). A scan whose admission or construction fails is skipped
and listed; the other scans continue ("Skip that scan, keep the Run"); an output that cannot be written still fails
the Run ("Still fail the Run"). Real Runs over real TIFF + .txt or SPEC scans; the limits are shortened here."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from tests.xdart.scattering import test_live_spec_metadata_run as spec_rows
from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_live_qualify_refusal_run import (
    _frame, _labels, _output, _said, _settle, _stop, _wait,
)
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal, _start
from xdart.gui.tabs.scattering import output_preflight
from xdart.gui.tabs.scattering.adapters import run_executor
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xrd_tools.reduction.core import NexusSink
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources.image import TiffSeriesSource
from xrd_tools.sources.selection import DirectorySourceSpec


@pytest.fixture
def quick(monkeypatch):
    """The note after 0.2 s and the limit after 1 s, for every kind."""
    monkeypatch.setattr(run_executor, "_METADATA_WAIT_NOTE_S", .2)
    monkeypatch.setattr(run_executor, "_METADATA_WAIT_LIMIT_S", {"sidecar": 1., "spec": 1.})


def _intent(tmp_path, raw, *, live, monitor=None):
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    args = {"npt": 8, "method": "numpy", **({} if monitor is None else {"monitor": monitor})}
    return RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="txt"), poni_file=str(poni),
        project_root=str(raw), save_path=str(tmp_path / "processed"), output_mode="Append",
        processing_mode="Int 1D", live_mode=live, max_cores=1, bai_1d_args=args)


def _live(tmp_path, raw, request_value, **intent):
    executor = StandardRunExecutor(join_timeout=2.)
    return executor, _start(executor, _intent(tmp_path, raw, live=True, **intent), request_value=request_value)


def _ended(values):
    return any(event.kind in _TERMINAL for event in values)


def _messages(caplog):
    return [record.getMessage() for record in caplog.records]


def test_a_frame_whose_txt_never_comes_is_noted_then_processed_without_it_and_the_scan_goes_on(
        tmp_path, quick, caplog):
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    executor, identity = _live(tmp_path, raw, 8901)
    events = []
    try:
        _frame(raw, "grow", 1, "txt", 1.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "grow") or _ended(values))
        _frame(raw, "grow", 2, "txt", 2.0, sidecar=False)  # its .txt never comes
        _wait(executor, events, lambda values: _said(values, "waits for the metadata of grow_0002.tif") or _ended(
            values), timeout=15.)
        _wait(executor, events, lambda values: 2 in _labels(values, "grow") or _ended(values), timeout=15.)
        _frame(raw, "grow", 3, "txt", 3.0)
        _wait(executor, events, lambda values: 3 in _labels(values, "grow") or _ended(values), timeout=15.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert "waits for the metadata" not in terminal[0][1] and not _said(events, "stopped growing"), terminal
    messages = _messages(caplog)
    assert "grow_int1d.nexus is waiting for the metadata of grow_0002.tif [LIVE]" in messages
    assert ("grow_int1d.nexus: the metadata of grow_0002.tif did not arrive within 1 s; the frame is processed "
            "without it [LIVE]") in messages
    assert "Processed without their metadata (1): grow_0002.tif (grow_int1d.nexus) [LIVE]" in messages
    labels, i0, _epoch = _output(tmp_path, "grow")
    assert labels == (1, 2, 3) and (i0[0], i0[2]) == (1.0, 3.0) and math.isnan(i0[1]), (labels, i0)


def test_a_sidecar_that_never_settles_skips_its_scan_from_that_frame_and_the_run_goes_on(
        tmp_path, quick, monkeypatch, caplog):
    """Codex N-69301b5f NW-R1, the rare-case branch (declared before the fix): a .txt present but still changing at
    the limit is not read as the frame's metadata; the scan stops there, its earlier frames kept, no row for the frame,
    the warning names the scan and the frame, and a later scan runs. The rewriting hook is Codex's."""
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    real_ready = output_preflight._live_metadata_ready

    def ready(*args, **kwargs):
        sidecar = raw / "grow_0002.txt"
        if sidecar.exists():
            sidecar.write_text(sidecar.read_text() + "# still being written\n")
        return real_ready(*args, **kwargs)
    monkeypatch.setattr(output_preflight, "_live_metadata_ready", ready)
    executor, identity = _live(tmp_path, raw, 8907)
    events = []
    try:
        _frame(raw, "grow", 1, "txt", 1.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "grow") or _ended(values))
        _frame(raw, "grow", 2, "txt", 7.0)  # its .txt never holds still
        _wait(executor, events, lambda values: _said(values, "grow_int1d.nexus stopped growing") or _ended(values),
              timeout=15.)
        _frame(raw, "next", 1, "txt", 5.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "next") or _ended(values), timeout=15.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert "grow_int1d.nexus stopped growing" in terminal[0][1], terminal
    messages = _messages(caplog)
    assert any(message.startswith("grow_int1d.nexus stops growing: it could not be processed")
               and "the metadata of grow_0002.tif did not settle within 1 s" in message for message in messages)
    assert not any("processed without it" in message for message in messages)
    assert _output(tmp_path, "grow")[0] == (1,) and _output(tmp_path, "next")[0] == (1,)


def test_a_given_up_frame_whose_txt_lands_later_stops_its_scan_and_the_run_goes_on(tmp_path, quick, caplog):
    """A frame processed without its metadata keeps that: its .txt landing later is never read as its metadata, and
    its scan stops growing, named; a later scan runs."""
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    executor, identity = _live(tmp_path, raw, 8908)
    events = []
    try:
        _frame(raw, "grow", 1, "txt", 1.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "grow") or _ended(values))
        _frame(raw, "grow", 2, "txt", 2.0, sidecar=False)
        _wait(executor, events, lambda values: 2 in _labels(values, "grow") or _ended(values), timeout=15.)
        _frame(raw, "grow", 2, "txt", 7.0, image=False)  # its .txt, after the Run stopped waiting for it
        _settle(executor, events, 1.)
        _frame(raw, "grow", 3, "txt", 3.0)
        _wait(executor, events, lambda values: _said(values, "grow_int1d.nexus stopped growing") or 3 in _labels(
            values, "grow") or _ended(values), timeout=15.)
        _frame(raw, "next", 1, "txt", 5.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "next") or _ended(values), timeout=15.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert "grow_int1d.nexus stopped growing" in terminal[0][1], terminal
    assert any(message.startswith("grow_int1d.nexus stops growing: it could not be processed")
               and "the metadata of grow_0002.tif arrived after the Run stopped waiting for it" in message
               for message in _messages(caplog))
    labels, i0, _epoch = _output(tmp_path, "grow")
    assert labels == (1, 2) and i0[0] == 1.0 and math.isnan(i0[1]), (labels, i0)
    assert _output(tmp_path, "next")[0] == (1,)


def test_a_waited_frame_keeps_its_one_deadline_until_it_is_admitted(monkeypatch):
    """The wait owner (Codex N-0c4e4237): an attempt that neither admits nor names the waited frame (a retried read, a
    listing due, a READY admission a fresh listing cut before it) keeps its deadline and kind; admitting it, or a wait
    that ended (its scan covered, its files gone), clears it."""
    monkeypatch.setattr(run_executor, "monotonic", lambda: 100.5)
    watch, run = StandardRunExecutor(join_timeout=2.)._watch_metadata_wait, SimpleNamespace(live_waiting={})
    target, held = Path("grow_int1d.nexus"), ("grow_0002.tif", 100.0, "grow_int1d.nexus", "settling")
    waits, given_up = {"k": held}, {}
    assert not watch(run, target, "k", None, waits, given_up, retried=True) and waits["k"] == held
    assert not watch(run, target, "k", None, waits, given_up, admitted=frozenset({"grow_0001.tif"}))
    assert waits["k"] == held
    assert not watch(run, target, "k", None, waits, given_up, admitted=frozenset({"grow_0001.tif", "grow_0002.tif"}))
    assert "k" not in waits
    waits["k"] = held
    assert not watch(run, target, "k", None, waits, given_up) and "k" not in waits  # covered, or its files left


def test_a_spec_point_whose_row_never_comes_skips_its_scan_from_there(tmp_path, quick, caplog):
    """An aborted SPEC point: its scan stops there, its earlier points kept (its qualification reads SPEC rows itself,
    so the point is not processed without its row)."""
    caplog.set_level(logging.INFO)
    raw, _spec = spec_rows._acquisition(tmp_path)  # point 0, row then image
    executor, identity = spec_rows._run(tmp_path, raw, 8902)
    events = []
    try:
        _wait(executor, events, spec_rows._reduced(1), 20.)
        spec_rows._image(raw, 1, 1)  # its image; SPEC never writes its row
        _wait(executor, events, lambda values: _said(values, "stopped growing") or _ended(values), timeout=20.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    name = spec_rows.STEM.format(scan=1)
    assert any(message.startswith(f"{name}_int1d.nexus stops growing: it could not be processed")
               and f"the SPEC row of {name}_0001.tif did not arrive within 1 s" in message
               for message in _messages(caplog))
    assert spec_rows._output(tmp_path) == ((1,), (100.0,))


def test_a_run_normalizing_by_a_monitor_skips_the_scan_whose_metadata_never_comes(tmp_path, quick, caplog):
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    executor, identity = _live(tmp_path, raw, 8903, monitor="I0")
    events = []
    try:
        _frame(raw, "grow", 1, "txt", 1.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "grow") or _ended(values))
        _frame(raw, "grow", 2, "txt", 2.0, sidecar=False)  # its I0 never comes
        _wait(executor, events, lambda values: _said(values, "stopped growing") or _ended(values), timeout=15.)
        _frame(raw, "next", 1, "txt", 5.0)  # the Run goes on with the next scan
        _wait(executor, events, lambda values: 1 in _labels(values, "next") or _ended(values), timeout=15.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert "grow_int1d.nexus stopped growing" in terminal[0][1], terminal
    assert any("did not arrive within 1 s, and this Run normalizes by a monitor from it" in message
               for message in _messages(caplog))
    assert _output(tmp_path, "grow")[0] == (1,) and _output(tmp_path, "next")[0] == (1,)


def _refuse_admission_of(monkeypatch, stem):
    """A scan whose admission fails as a missing GI metadata motor does (the real check raises ValueError): Live's
    readiness check, and a directory Run's admission."""
    real, deferred = output_preflight._validate_exact_tiff_gi_motor, run_executor.materialize_deferred_output
    message = "GI metadata motor 'th' must have a finite value in every admitted TIFF (injected)"

    def validate(intent, items):
        if any(f"{stem}_int1d" in str(item.target) for item in items):
            raise ValueError(message)
        return real(intent, items)

    def materialize(receipt, session, entry, **kwargs):
        if f"{stem}_int1d" in str(entry.target):
            raise ValueError(message)
        return deferred(receipt, session, entry, **kwargs)
    monkeypatch.setattr(output_preflight, "_validate_exact_tiff_gi_motor", validate)
    monkeypatch.setattr(run_executor, "materialize_deferred_output", materialize)


def test_a_live_scan_that_cannot_be_admitted_is_skipped_and_the_run_goes_on(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _refuse_admission_of(monkeypatch, "bad")
    raw = tmp_path / "raw"
    raw.mkdir()
    executor, identity = _live(tmp_path, raw, 8904)
    events = []
    try:
        _frame(raw, "bad", 1, "txt", 1.0)
        _wait(executor, events, lambda values: _said(values, "bad_int1d.nexus stopped growing") or _ended(values))
        _frame(raw, "good", 1, "txt", 2.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "good") or _ended(values))
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert any(message.startswith("bad_int1d.nexus stops growing: it could not be processed") and
               "ValueError: GI metadata motor 'th'" in message for message in _messages(caplog))
    assert _output(tmp_path, "bad") is None and _output(tmp_path, "good")[0] == (1,)


def test_a_live_scan_whose_output_cannot_be_written_still_fails_the_run(tmp_path, monkeypatch):
    real = StandardRunExecutor._construct

    def construct(self, run, **kwargs):
        if "bad_int1d" in str(kwargs["item"].target):
            raise OSError(28, "No space left on device (injected)")
        return real(self, run, **kwargs)
    monkeypatch.setattr(StandardRunExecutor, "_construct", construct)
    raw = tmp_path / "raw"
    raw.mkdir()
    executor, identity = _live(tmp_path, raw, 8905)
    events = []
    try:
        _frame(raw, "bad", 1, "txt", 1.0)
        _wait(executor, events, _ended, timeout=20.)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["FAILED"], terminal
    assert "No space left on device (injected)" in terminal[0][1], terminal


def _deny(monkeypatch, image):
    """``image`` unreadable while the Run reads it (Codex N-69301b5f's technique): the real decoder meets a real
    PermissionError; nothing is fabricated."""
    actual = TiffSeriesSource._read_path

    def read(source, path):
        if Path(path) != image:
            return actual(source, path)
        mode = image.stat().st_mode & 0o777
        image.chmod(0)
        try:
            return actual(source, path)
        finally:
            image.chmod(mode)
    monkeypatch.setattr(TiffSeriesSource, "_read_path", read)


def test_a_live_frame_that_cannot_be_read_stops_its_scan_keeping_what_it_published(tmp_path, monkeypatch, caplog):
    """Its second frame, read in a later epoch than its first (Codex N-69301b5f NW-R2, later decoding)."""
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    _deny(monkeypatch, raw / "bad_0002.tif")
    executor, identity = _live(tmp_path, raw, 8909)
    events = []
    try:
        _frame(raw, "bad", 1, "txt", 1.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "bad") or _ended(values))
        _frame(raw, "bad", 2, "txt", 2.0)
        _wait(executor, events, lambda values: _said(values, "bad_int1d.nexus stopped growing") or _ended(values),
              timeout=15.)
        _frame(raw, "good", 1, "txt", 3.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "good") or _ended(values), timeout=15.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert any(message.startswith("bad_int1d.nexus stops growing: it could not be processed")
               and "SourceReadError" in message and "bad_0002.tif" in message for message in _messages(caplog))
    assert _output(tmp_path, "bad")[0] == (1,) and _output(tmp_path, "good")[0] == (1,)


def test_a_directory_scan_whose_later_frame_cannot_be_read_is_skipped_and_the_others_run(
        tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    for stem, value in (("bad", 1.0), ("good", 2.0)):
        for index in (1, 2):
            _frame(raw, stem, index, "txt", value + index)
    _deny(monkeypatch, raw / "bad_0002.tif")
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=8915)
    executor.close(identity)
    terminals = [event for event in events if event.kind in _TERMINAL]
    assert [event.kind.name for event in terminals] == ["FINISHED"], terminals
    assert any("bad_int1d.nexus" in warning and "SourceReadError" in warning for warning in terminals[0].warnings), \
        terminals[0].warnings
    assert any(message.startswith("bad_int1d.nexus skipped: it could not be processed; frames already saved are kept, "
                                            "and the other scans continue")
               for message in _messages(caplog))
    assert _output(tmp_path, "bad") is None and _output(tmp_path, "good")[0] == (1, 2)


def _stack(path, frames, corrupt=None):
    """A NeXus stack whose frame ``corrupt``'s compressed chunk is overwritten on disk: reading it really fails."""
    with h5py.File(path, "w") as handle:
        dataset = handle.create_group("entry/instrument/detector").create_dataset(
            "data", data=np.ones((frames, 195, 487), dtype=np.uint16), chunks=(1, 195, 487),
            maxshape=(None, 195, 487), compression="gzip")
        info = None if corrupt is None else dataset.id.get_chunk_info(corrupt)
    if info is not None:
        with open(path, "r+b") as raw:
            raw.seek(info.byte_offset)
            raw.write(b"\xff" * info.size)


@pytest.mark.parametrize("live", [True, False])
def test_a_container_frame_that_cannot_be_read_stops_its_scan_and_the_run_goes_on(tmp_path, caplog, live):
    """A container's frames are read by the executor's producer (Codex N-69301b5f NW-R2): its third frame cannot be
    decoded; the frames before it are kept, and the next container is processed."""
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    _stack(raw / "bad.nxs", 3, corrupt=2)
    _stack(raw / "good.nxs", 3)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    intent = RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".nxs",), metadata_format=None), poni_file=str(poni),
        project_root=str(raw), save_path=str(tmp_path / "processed"), output_mode="Append",
        processing_mode="Int 1D", live_mode=live, max_cores=1, bai_1d_args={"npt": 8, "method": "numpy"})
    if live:
        executor = StandardRunExecutor(join_timeout=2.)
        identity, events = _start(executor, intent, request_value=8918), []
        try:
            _wait(executor, events, lambda values: 2 in _labels(values, "good") or _ended(values), timeout=20.)
            _settle(executor, events, .5)
        finally:
            terminal = _stop(executor, identity, events)
        assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
        assert "bad_int1d.nexus stopped growing" in terminal[0][1], terminal
    else:
        executor, identity, events = _run_to_terminal(intent, request_value=8919)
        executor.close(identity)
        terminal = [(event.kind.name, event.detail) for event in events if event.kind in _TERMINAL]
        assert [kind for kind, _detail in terminal] == ["FINISHED"], terminal
    assert any("bad_int1d.nexus" in message and "could not be processed" in message and "inflate() failed" in message
               for message in _messages(caplog))
    assert _output(tmp_path, "bad")[0] == (0, 1) and _output(tmp_path, "good")[0] == (0, 1, 2)


@pytest.mark.parametrize("live", [True, False])
def test_an_output_write_failure_while_frames_are_reduced_still_fails_the_run(tmp_path, monkeypatch, live):
    """The sink-owner control (Codex N-69301b5f NW-R2): the NeXus record write of a real reduction fails."""
    real = NexusSink._write_records

    def write(self, writer, records):
        if "bad_int1d" in str(self.path):
            raise OSError(28, "No space left on device (injected)")
        return real(self, writer, records)
    monkeypatch.setattr(NexusSink, "_write_records", write)
    raw = tmp_path / "raw"
    raw.mkdir()
    for stem in ("bad", "good"):
        for index in (1, 2):
            _frame(raw, stem, index, "txt", float(index))
    if live:
        executor, identity = _live(tmp_path, raw, 8916)
        events = []
        try:
            _wait(executor, events, _ended, timeout=20.)
        finally:
            terminal = _stop(executor, identity, events)
    else:
        executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=8917)
        executor.close(identity)
        terminal = [(event.kind.name, event.detail) for event in events if event.kind in _TERMINAL]
    assert [kind for kind, _detail in terminal] == ["FAILED"], terminal
    assert "No space left on device (injected)" in terminal[0][1], terminal


def _gi_frame(raw, stem, index, i0, th=None):
    """One GI frame: its .txt (with the incidence motor ``th`` unless None), then its image."""
    motors = "energy = 16" if th is None else f"th = {th:.2f}, energy = 16"
    raw.joinpath(f"{stem}_{index:04d}.txt").write_text(
        f"# Counters\nI0 = {i0}\n# Motors\n{motors}\nUser: energy, time: Mon Jan 15 10:30:00 2024  # Temp\n")
    _frame(raw, stem, index, "txt", i0, sidecar=False)


def test_a_gi_directory_scan_missing_its_motor_is_skipped_and_the_other_scans_run(tmp_path, caplog):
    """A GI directory Run admits every scan before it starts: one whose frame lacks the incidence motor is skipped and
    named (it failed the whole Run before), the others run."""
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    _gi_frame(raw, "bad", 1, 1.0, th=0.1)
    _gi_frame(raw, "bad", 2, 2.0)  # no th
    _gi_frame(raw, "good", 1, 3.0, th=0.1)
    _gi_frame(raw, "good", 2, 4.0, th=0.2)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    intent = RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="txt"), poni_file=str(poni),
        project_root=str(raw), save_path=str(tmp_path / "processed"), output_mode="Append",
        processing_mode="Int 1D", live_mode=False, max_cores=1, bai_1d_args={"npt": 8, "method": "numpy"},
        gi=GIIntent(enabled=True, incidence_motor="th"))
    executor, identity, events = _run_to_terminal(intent, request_value=8921)
    executor.close(identity)
    terminals = [event for event in events if event.kind in _TERMINAL]
    assert [event.kind.name for event in terminals] == ["FINISHED"], terminals
    assert any("bad_gi_int1d.nexus" in warning and "GI metadata motor 'th'" in warning
               for warning in terminals[0].warnings), terminals[0].warnings
    assert any(message.startswith("bad_gi_int1d.nexus skipped: it could not be processed") for message in
               _messages(caplog))
    assert _output(tmp_path, "bad_gi") is None and _output(tmp_path, "good_gi")[0] == (1, 2)


@pytest.mark.parametrize("stage", ["admission", "construction"])
def test_a_directory_scan_that_fails_is_skipped_and_the_other_scans_run(tmp_path, monkeypatch, caplog, stage):
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    for stem, value in (("bad", 1.0), ("good", 2.0)):
        for index in (1, 2):
            _frame(raw, stem, index, "txt", value + index)
    if stage == "admission":
        _refuse_admission_of(monkeypatch, "bad")
    else:  # as an energy scan's refusal does, when its scan is bound
        real = StandardRunExecutor._construct

        def construct(self, run, **kwargs):
            if "bad_int1d" in str(kwargs["item"].target):
                raise ValueError("an energy scan cannot use the sensor parallax correction (injected)")
            return real(self, run, **kwargs)
        monkeypatch.setattr(StandardRunExecutor, "_construct", construct)
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=8906)
    executor.close(identity)
    terminals = [event for event in events if event.kind in _TERMINAL]
    assert [event.kind.name for event in terminals] == ["FINISHED"], terminals
    assert any("bad_int1d.nexus" in warning and "ValueError" in warning for warning in terminals[0].warnings), \
        terminals[0].warnings
    assert any(message.startswith("bad_int1d.nexus skipped: it could not be processed; frames already saved are kept, "
                                            "and the other scans continue")
               for message in _messages(caplog))
    assert _output(tmp_path, "bad") is None and _output(tmp_path, "good")[0] == (1, 2)
