"""Under Auto a Live frame waits for its sidecar as under an explicit format (Claude review of 03af89cc, C1: Auto
admitted a frame after one 0.1 s look found no sidecar, so on a share where an image is readable before its sidecar
is listed the frame was reduced without metadata, and the sidecar landing after that stopped its whole scan).

A sidecar-less directory is not held for it: while the Run has seen no sidecar at all, a frame waits two seconds (the
note period), and the Run then knows its folder has none, so later frames do not wait. Real Live Runs over real
TIFF + .txt files under Auto; nothing about admission or readiness is replaced."""
import logging
import sys
import time

from tests.xdart.scattering.test_live_qualify_refusal_run import (
    _frame, _labels, _output, _run, _said, _settle, _stop, _terminal, _wait,
)
from xdart.gui.tabs.scattering.adapters import run_executor


def _frame_ready(executor, events, index, stem='scan', timeout=30.):
    """Seconds until frame ``index`` of ``stem`` is reduced."""
    started = time.monotonic()
    _wait(executor, events, lambda values: index in _labels(values, stem) or _said(values, 'stopped growing')
          or _terminal(values), timeout)
    assert index in _labels(events, stem), [event.detail for event in events[-3:]]
    return time.monotonic() - started


def _warm(executor, events, raw):
    """One finished scan with its sidecar: the Run's integrator is built before the frame under test lands."""
    _frame(raw, 'warm', 1, 'txt', 9.0)
    _frame_ready(executor, events, 1, 'warm')


def test_under_auto_an_image_whose_sidecar_lands_after_it_is_processed_with_it_and_its_scan_keeps_growing(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    executor, identity = _run(tmp_path, raw, 'auto', 3401)
    events = []
    try:
        _warm(executor, events, raw)
        _frame(raw, 'scan', 1, 'txt', 1.0, sidecar=False)  # its image first
        _settle(executor, events, .5)
        before = 1 in _labels(events)
        _frame(raw, 'scan', 1, 'txt', 1.0, image=False)  # its sidecar, half a second later
        _frame_ready(executor, events, 1)
        _frame(raw, 'scan', 2, 'txt', 2.0)  # the scan keeps growing
        _frame_ready(executor, events, 2)
        _settle(executor, events)
        blocked = dict(executor._exact_run(identity).live_blocked)
    finally:
        terminal = _stop(executor, identity, events)
    assert not before, 'the frame was reduced before its sidecar landed'
    assert blocked == {} and not _said(events, 'stopped growing'), blocked
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, i0, _epoch = _output(tmp_path)
    assert (labels, i0) == ((1, 2), (1.0, 2.0)), 'the first frame carries its own sidecar\'s values'


def test_under_auto_the_first_frame_of_a_fresh_folder_gets_its_sidecar_too(tmp_path):
    """No sidecar has been seen yet, and the Run waits two seconds for one: it lands within them."""
    raw = tmp_path / 'raw'
    raw.mkdir()
    executor, identity = _run(tmp_path, raw, 'auto', 3402)
    events = []
    try:
        _frame(raw, 'scan', 1, 'txt', 1.0, sidecar=False)
        _settle(executor, events, .5)
        before = 1 in _labels(events)
        _frame(raw, 'scan', 1, 'txt', 1.0, image=False)
        _frame_ready(executor, events, 1)
        _frame(raw, 'scan', 2, 'txt', 2.0)
        _frame_ready(executor, events, 2)
        _settle(executor, events)
        blocked = dict(executor._exact_run(identity).live_blocked)
    finally:
        terminal = _stop(executor, identity, events)
    assert not before and blocked == {}, (before, blocked)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _output(tmp_path)[:2] == ((1, 2), (1.0, 2.0))


def test_under_auto_a_folder_without_any_sidecar_holds_its_first_frame_two_seconds_and_no_later_one(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    raw = tmp_path / 'raw'
    raw.mkdir()
    executor, identity = _run(tmp_path, raw, 'auto', 3403)
    events, later = [], []
    try:
        _frame(raw, 'scan', 1, 'txt', 1.0, sidecar=False)
        first = _frame_ready(executor, events, 1)
        for index in (2, 3):
            _frame(raw, 'scan', index, 'txt', float(index), sidecar=False)
            later.append(_frame_ready(executor, events, index))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert 1.5 < first < 8., f'the first frame waited its two seconds, not the ten of an explicit format: {first:.2f} s'
    assert max(later) < 1.5, f'later frames of a folder known to have no sidecars wait: {later}'
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, i0, _epoch = _output(tmp_path)
    assert labels == (1, 2, 3) and i0 is None, (labels, i0)  # no frame has any metadata
    messages = [record.getMessage() for record in caplog.records]
    assert messages.count('scan_int1d.nexus: the metadata of scan_0001.tif did not arrive within 2 s; the frame is '
                          'processed without it [LIVE]') == 1, messages
    assert not [message for message in messages if 'did not arrive' in message and 'scan_0002' in message], messages
    assert 'Processed without their metadata (1): scan_0001.tif (scan_int1d.nexus) [LIVE]' in messages


def test_under_auto_a_folder_whose_earlier_scans_have_sidecars_is_known_to_have_them_before_its_first_admission(
        tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'old', 1, 'txt', 4.0)
    seen, real = [], run_executor.materialize_live_directory_group

    def attempt(*args, **kwargs):  # a spy: the attempt is the real one; what the Run knew when it was made
        seen.append(getattr(sys._getframe(1).f_locals['run'], 'sidecars_seen', None))
        return real(*args, **kwargs)
    monkeypatch.setattr(run_executor, 'materialize_live_directory_group', attempt)
    executor, identity = _run(tmp_path, raw, 'auto', 3404)
    events = []
    try:
        _frame_ready(executor, events, 1, 'old')
    finally:
        terminal = _stop(executor, identity, events)
    assert seen and seen[0] is True, seen
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _output(tmp_path, 'old')[:2] == ((1,), (4.0,))
