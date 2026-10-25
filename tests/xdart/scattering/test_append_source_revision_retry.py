"""A directory Append reads a scan once more when its files changed while it was read (Claude review of 03af89cc,
G-b3: ``SourceRevisionChanged`` is a retry in the Live loop but, caught as a ``ValueError`` by the finite directory
loop, skipped the scan for the Run with the Replace remedy, though nothing was left unchanged and Replace would only
repeat the read).

A real finite Append over real TIFF + .txt scans; the change is a real write of a sidecar, made inside the reader's
read of it on the Run's own thread (the shape of test_live_qualify_refusal_run's Q2)."""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import xrd_tools.io.metadata as metadata
from tests.xdart.scattering.test_live_already_processed import _intent
from tests.xdart.scattering.test_live_qualify_refusal_run import SIDECARS, _frame, _output
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal


def _worker():
    return threading.current_thread() is not threading.main_thread()


def _within(function):
    frame = sys._getframe(2)
    while frame is not None and frame.f_code.co_name != function:
        frame = frame.f_back
    return frame is not None


def _append(tmp_path, monkeypatch, caplog, *, changes, request_value, during=None):
    """The terminal event of a finite Append over two scans, where the second frame of ``grow``'s sidecar is
    rewritten (to I0 7.0) inside its first ``changes`` reads (those a call of ``during`` makes, when given); the
    names of the files it was rewritten in."""
    caplog.set_level(logging.INFO)
    raw = tmp_path / 'raw'
    raw.mkdir()
    for index in (1, 2, 3):
        _frame(raw, 'grow', index, 'txt', float(index))
    _frame(raw, 'other', 1, 'txt', 5.0)
    fired, real = [], metadata.read_txt_metadata

    def read(path, *args, **kwargs):
        if Path(path).name == 'grow_0002.txt' and len(fired) < changes and _worker() and (
                during is None or _within(during)):
            fired.append(Path(path).name)
            Path(path).write_text(SIDECARS['txt'](7.0))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(metadata, 'read_txt_metadata', read)
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False), request_value=request_value)
    executor.close(identity)
    terminals = [event for event in events if event.kind in _TERMINAL]
    assert [event.kind.name for event in terminals] == ['FINISHED'], terminals
    return terminals[0], fired


def test_a_scan_whose_files_changed_while_they_were_read_is_read_once_more_not_skipped(tmp_path, monkeypatch, caplog):
    terminal, fired = _append(tmp_path, monkeypatch, caplog, changes=1, request_value=9811)
    assert fired == ['grow_0002.txt'], 'the change landed inside the scan\'s first read'
    assert terminal.warnings == (), terminal.warnings
    assert not any('Replace' in record.getMessage() for record in caplog.records)
    labels, i0, _epoch = _output(tmp_path, 'grow')
    assert (labels, i0) == ((1, 2, 3), (1.0, 7.0, 3.0))  # the sidecar as the second read found it
    assert _output(tmp_path, 'other')[0] == (1,)


def test_a_scan_that_changes_again_while_it_is_read_once_more_is_skipped_with_its_own_remedy(
        tmp_path, monkeypatch, caplog):
    terminal, fired = _append(tmp_path, monkeypatch, caplog, changes=2, request_value=9812)
    assert fired == ['grow_0002.txt', 'grow_0002.txt']
    assert len(terminal.warnings) == 1 and 'grow_int1d.nexus' in terminal.warnings[0], terminal.warnings
    assert terminal.warnings[0].endswith('its files changed while they were read; run Append again'), terminal.warnings
    assert not any('Replace' in line for line in terminal.warnings)
    assert not any('Replace' in record.getMessage() for record in caplog.records)
    assert _output(tmp_path, 'grow') is None  # nothing was written for it, and the other scan ran
    assert _output(tmp_path, 'other')[0] == (1,)


def test_a_scan_whose_files_change_while_its_construction_reads_them_is_taken_once_more(tmp_path, monkeypatch, caplog):
    """The same change met by the reads of its construction, after its admission: the scan is admitted and
    constructed again, and its files are counted once."""
    terminal, fired = _append(tmp_path, monkeypatch, caplog, changes=1, request_value=9813, during='_construct')
    assert fired == ['grow_0002.txt'], 'the change landed inside the scan\'s construction'
    assert terminal.warnings == (), terminal.warnings
    assert not any('Replace' in record.getMessage() for record in caplog.records)
    labels, i0, _epoch = _output(tmp_path, 'grow')
    assert (labels, i0) == ((1, 2, 3), (1.0, 7.0, 3.0))
    assert _output(tmp_path, 'other')[0] == (1,)
    assert (terminal.files_processed, terminal.files_skipped, terminal.files_pending) == (4, 0, 0), terminal


def test_a_scan_that_changes_again_while_its_construction_reads_it_once_more_is_skipped(tmp_path, monkeypatch, caplog):
    terminal, fired = _append(tmp_path, monkeypatch, caplog, changes=2, request_value=9814, during='_construct')
    assert fired == ['grow_0002.txt', 'grow_0002.txt']
    assert len(terminal.warnings) == 1 and terminal.warnings[0].endswith(
        'grow_int1d.nexus: its files changed while they were read; run Append again'), terminal.warnings
    assert not any('Replace' in record.getMessage() for record in caplog.records)
    assert _output(tmp_path, 'grow') is None and _output(tmp_path, 'other')[0] == (1,)
    assert (terminal.files_processed, terminal.files_skipped, terminal.files_pending) == (1, 3, 0), terminal
