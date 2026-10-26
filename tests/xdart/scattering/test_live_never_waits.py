"""A Live or directory Run never stops or waits forever on one scan (maintainer, 2026-10-24: "I don't want a live run
to stop or be waiting forever"; "if something fails, a live run or a directory run shouldn't stop. That scan/frame
should be logged and skipped and the remaining scan processing should continue").

A frame waiting for its own metadata is noted in the log and the Live status ("the log output should show it is
waiting"). A frame without its .txt is never processed without it (maintainer, 2026-10-26: "If I have chosen txt as
the metadata type, then any frame without the txt sidecar should be skipped. There is no need to wait. When the live
loop comes back to an image and there's a txt sidecar present it gets integrated"): it is processed when its .txt
lands, and left out once a later frame of its series has its own (his pick "Leave N out"). A sidecar that never holds
still, or a SPEC row that never comes (30 s), skips its scan from that frame on. A scan whose admission or
construction fails is skipped and listed; the other scans continue ("Skip that scan, keep the Run"); an output that
cannot be written still fails the Run ("Still fail the Run"). Real Runs over real TIFF + .txt or SPEC scans; the
limits are shortened here."""
from __future__ import annotations

import logging
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
    """The note after 0.2 s (a debug line), the regular log's after 0.6 s, and the limit after 1 s, for every kind."""
    monkeypatch.setattr(run_executor, "_METADATA_WAIT_NOTE_S", .2)
    monkeypatch.setattr(run_executor, "_METADATA_WAIT_INFO_S", .6)
    monkeypatch.setattr(run_executor, "_METADATA_WAIT_LIMIT_S", {"sidecar": 1., "spec": 1.})


def _intent(tmp_path, raw, *, live, monitor=None, fmt="txt"):
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    args = {"npt": 8, "method": "numpy", **({} if monitor is None else {"monitor": monitor})}
    return RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format=fmt), poni_file=str(poni),
        project_root=str(raw), save_path=str(tmp_path / "processed"), output_mode="Append",
        processing_mode="Int 1D", live_mode=live, max_cores=1, bai_1d_args=args)


def _live(tmp_path, raw, request_value, **intent):
    executor = StandardRunExecutor(join_timeout=2.)
    return executor, _start(executor, _intent(tmp_path, raw, live=True, **intent), request_value=request_value)


def _ended(values):
    return any(event.kind in _TERMINAL for event in values)


def _messages(caplog):
    return [record.getMessage() for record in caplog.records]


def test_a_frame_whose_txt_never_comes_waits_and_is_left_out_once_a_later_frame_has_its_own(
        tmp_path, quick, caplog):
    """The frame waits, noted, past the limit while nothing later has come, and is not processed without its .txt;
    once the next frame lands with its own, it is left out and the series goes on in its output, numbered in order."""
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
        _settle(executor, events, 1.5)  # past the 1 s limit
        held = 2 in _labels(events, "grow")
        _frame(raw, "grow", 3, "txt", 3.0)
        _wait(executor, events, lambda values: 2 in _labels(values, "grow") or _ended(values), timeout=15.)
        _settle(executor, events, .5)
        skipped = executor._exact_run(identity).files_skipped
    finally:
        terminal = _stop(executor, identity, events)
    assert not held, "a frame was processed without its .txt"
    assert skipped == 1, f"the left-out image is not counted as skipped: {skipped}"
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert not _said(events, "stopped growing"), terminal
    messages = _messages(caplog)
    assert ("grow_int1d.nexus has waited over 0.6 s for the metadata of grow_0002.tif; it is processed once its "
            "metadata arrives [LIVE]") in messages
    assert ("grow_int1d.nexus: grow_0002.tif is left out; its metadata did not arrive and a later frame's did "
            "[LIVE]") in messages
    assert "Left out without their metadata (1): grow_0002.tif (grow_int1d.nexus) [LIVE]" in messages
    assert not any("processed without" in message for message in messages), messages
    labels, i0, _epoch = _output(tmp_path, "grow")
    assert (labels, i0) == ((1, 2), (1.0, 3.0)), (labels, i0)


def test_a_frame_still_waiting_for_its_txt_when_the_run_stops_is_named_at_its_end(tmp_path, quick, caplog):
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    executor, identity = _live(tmp_path, raw, 8909)
    events = []
    try:
        _frame(raw, "grow", 1, "txt", 1.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "grow") or _ended(values))
        _frame(raw, "grow", 2, "txt", 2.0, sidecar=False)
        _wait(executor, events, lambda values: _said(values, "waits for the metadata of grow_0002.tif") or _ended(
            values), timeout=15.)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert ("Not processed, still waiting for their metadata (1): grow_int1d.nexus waits for the metadata of "
            "grow_0002.tif [LIVE]") in _messages(caplog)
    assert _output(tmp_path, "grow")[:2] == ((1,), (1.0,))


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


def test_a_frame_whose_txt_lands_late_is_processed_with_it_and_its_scan_goes_on(tmp_path, quick, caplog):
    """The frame waits past the limit with no later frame; its .txt lands, and the frame is processed with its own
    values, the scan growing on."""
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    executor, identity = _live(tmp_path, raw, 8908)
    events = []
    try:
        _frame(raw, "grow", 1, "txt", 1.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "grow") or _ended(values))
        _frame(raw, "grow", 2, "txt", 2.0, sidecar=False)
        _settle(executor, events, 1.5)  # past the 1 s limit
        held = 2 in _labels(events, "grow")
        _frame(raw, "grow", 2, "txt", 7.0, image=False)  # its .txt, late
        _wait(executor, events, lambda values: 2 in _labels(values, "grow") or _ended(values), timeout=15.)
        _frame(raw, "grow", 3, "txt", 3.0)
        _wait(executor, events, lambda values: 3 in _labels(values, "grow") or _ended(values), timeout=15.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert not held, "a frame was processed without its .txt"
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert not _said(events, "stopped growing"), terminal
    assert any(message.startswith("grow_int1d.nexus no longer waits for the metadata of grow_0002.tif, after ")
               and message.endswith(" s [LIVE]") for message in _messages(caplog)), _messages(caplog)
    assert _output(tmp_path, "grow")[:2] == ((1, 2, 3), (1.0, 7.0, 3.0))


def test_a_waited_frame_keeps_its_one_deadline_until_it_is_admitted(monkeypatch):
    """The wait owner (Codex N-0c4e4237): an attempt that neither admits nor names the waited frame (a retried read, a
    listing due, a READY admission a fresh listing cut before it) keeps its deadline, kind and its sidecar's first
    sighting; admitting it, or a wait that ended (its scan covered, its files gone), clears it."""
    monkeypatch.setattr(run_executor, "monotonic", lambda: 100.5)
    watch, run = StandardRunExecutor(join_timeout=2.)._watch_metadata_wait, SimpleNamespace(live_waiting={})
    target, held = Path("grow_int1d.nexus"), ("grow_0002.tif", 100.0, "grow_int1d.nexus", "settling", 100.0)
    waits = {"k": held}
    assert not watch(run, target, "k", None, waits, retried=True) and waits["k"] == held
    assert not watch(run, target, "k", None, waits, admitted=frozenset({"grow_0001.tif"}))
    assert waits["k"] == held
    assert not watch(run, target, "k", None, waits, admitted=frozenset({"grow_0001.tif", "grow_0002.tif"}))
    assert "k" not in waits
    waits["k"] = held
    assert not watch(run, target, "k", None, waits) and "k" not in waits  # covered, or its files left


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


def test_a_run_normalizing_by_a_monitor_never_processes_a_frame_without_its_metadata(tmp_path, quick, caplog):
    """Under Auto too: a Run whose monitor comes from the metadata expects a sidecar for every frame, so it asks
    nothing. A frame whose I0 has not come waits past the limit while the next scan goes on, and is left out once a
    later frame has its own."""
    caplog.set_level(logging.INFO)
    raw = tmp_path / "raw"
    raw.mkdir()
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, live=True, monitor="I0", fmt="auto"), request_value=8903)
    events, asked = [], []
    try:
        _frame(raw, "grow", 1, "txt", 1.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "grow") or _ended(values))
        _frame(raw, "grow", 2, "txt", 2.0, sidecar=False)  # its I0 has not come
        _frame(raw, "next", 1, "txt", 5.0)  # the Run goes on with the next scan
        _wait(executor, events, lambda values: 1 in _labels(values, "next") or _ended(values), timeout=15.)
        _settle(executor, events, 1.5)  # past the 1 s limit
        held = 2 in _labels(events, "grow")
        asked.append(executor.pending_metadata_question(identity))
        _frame(raw, "grow", 3, "txt", 3.0)
        _wait(executor, events, lambda values: 2 in _labels(values, "grow") or _ended(values), timeout=15.)
        _settle(executor, events, .5)
    finally:
        terminal = _stop(executor, identity, events)
    assert not held and asked == [None], (held, asked)
    assert [kind for kind, _detail in terminal] == ["STOPPED"], terminal
    assert not _said(events, "stopped growing"), terminal
    assert not any("asking whether" in message for message in _messages(caplog))
    assert _output(tmp_path, "grow")[:2] == ((1, 2), (1.0, 3.0)) and _output(tmp_path, "next")[0] == (1,)


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
    assert any("bad_int1d.nexus" in message and "could not be processed" in message  # HDF5's own words vary by OS
               and "SourceReadError: its next frames of bad.nxs could not be read: OSError" in message
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


def test_a_short_wait_for_a_txt_is_told_only_in_the_debug_log(tmp_path, quick, monkeypatch, caplog):
    """At the beamline every .txt lands 5-8 s after its image, so each frame waited, noted at 2 s, and two INFO lines
    told it (maintainer, 2026-10-26: "perhaps better to fold this in debug?", his pick "Normal log only after 30 s"):
    a wait shorter than _METADATA_WAIT_INFO_S writes debug lines only."""
    monkeypatch.setattr(run_executor, "_METADATA_WAIT_INFO_S", 30.)
    caplog.set_level(logging.DEBUG, logger=run_executor.__name__)
    raw = tmp_path / "raw"
    raw.mkdir()
    executor, identity = _live(tmp_path, raw, 8911)
    events = []
    try:
        _frame(raw, "grow", 1, "txt", 1.0)
        _wait(executor, events, lambda values: 1 in _labels(values, "grow") or _ended(values))
        _frame(raw, "grow", 2, "txt", 2.0, sidecar=False)
        _wait(executor, events, lambda values: _said(values, "waits for the metadata of grow_0002.tif") or _ended(
            values), timeout=15.)
        _frame(raw, "grow", 2, "txt", 7.0, image=False)  # its .txt, a moment later
        _wait(executor, events, lambda values: 2 in _labels(values, "grow") or _ended(values), timeout=15.)
        _settle(executor, events, .5)
    finally:
        _stop(executor, identity, events)
    told = [(record.levelno, record.getMessage()) for record in caplog.records if "metadata of grow_0002" in
            record.getMessage()]
    assert (logging.DEBUG, "grow_int1d.nexus is waiting for the metadata of grow_0002.tif [LIVE]") in told, told
    assert (logging.DEBUG, "grow_int1d.nexus no longer waits for the metadata of grow_0002.tif [LIVE]") in told, told
    assert all(level == logging.DEBUG for level, _message in told), told
