"""Under Auto a Live frame waits for its sidecar as under an explicit format (Claude review of 03af89cc, C1: Auto
admitted a frame after one 0.1 s look found no sidecar, so on a share where an image is readable before its sidecar
is listed the frame was reduced without metadata, and the sidecar landing after that stopped its whole scan).

The first sidecar the Run finds, beside any image in any folder, makes a sidecar its metadata type (maintainer,
2026-10-26: "in a live or directory run, it can be assumed that the moment a metadata sidecar is found, that type
applies universally for all data in all folders for that run"). While it has found none, a frame without one waits for
it (on the beamline share a .txt lands 5.1 s after its image), and the Run asks once whether it expects one (his pick
"Once per Run"): Yes, its frames wait for their own; No, or no answer within the sidecar limit (150 s), they are
processed without metadata. Real Live Runs over real TIFF + .txt files under Auto; nothing about admission or readiness
is replaced."""
import logging
import sys
import time

import pytest

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


def _asked(executor, identity, events, timeout=30.):
    """The Run's question, once it asks."""
    deadline = time.monotonic() + timeout
    while (question := executor.pending_metadata_question(identity)) is None:
        assert time.monotonic() < deadline, 'the Run never asked'
        events.extend(executor.drain_events())
        time.sleep(.01)
    return question


@pytest.fixture
def ask_soon(monkeypatch):
    """The question once a frame has waited 0.3 s for its sidecar (10 s by default)."""
    monkeypatch.setattr(run_executor, '_METADATA_QUESTION_AFTER_S', .3)


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
    """No sidecar has been found yet: the frame waits for one, and it lands before the Run asks."""
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


def test_under_auto_a_fresh_folder_whose_sidecars_land_seconds_after_their_images_keeps_their_metadata(tmp_path):
    """The beamline's shape: Live started on an empty folder, each .txt landing seconds after its image, longer than
    the two seconds a first frame once waited."""
    raw = tmp_path / 'raw'
    raw.mkdir()
    executor, identity = _run(tmp_path, raw, 'auto', 3406)
    events = []
    try:
        for index in (1, 2):
            _frame(raw, 'scan', index, 'txt', float(index), sidecar=False)
            _settle(executor, events, 3.)
            _frame(raw, 'scan', index, 'txt', float(index), image=False)
            _frame_ready(executor, events, index)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _output(tmp_path)[:2] == ((1, 2), (1.0, 2.0))  # each frame read its own .txt


def test_under_auto_a_folder_without_any_sidecar_asks_once_and_unanswered_goes_on_without_metadata(
        tmp_path, caplog, monkeypatch, ask_soon):
    caplog.set_level(logging.INFO)
    monkeypatch.setitem(run_executor._METADATA_WAIT_LIMIT_S, 'sidecar', 2.)  # 150 s by default
    raw = tmp_path / 'raw'
    raw.mkdir()
    executor, identity = _run(tmp_path, raw, 'auto', 3403)
    events, later = [], []
    try:
        _frame(raw, 'scan', 1, 'txt', 1.0, sidecar=False)
        started = time.monotonic()
        question = _asked(executor, identity, events)
        _frame_ready(executor, events, 1)
        first, after = time.monotonic() - started, executor.pending_metadata_question(identity)
        for index in (2, 3):
            _frame(raw, 'scan', index, 'txt', float(index), sidecar=False)
            later.append(_frame_ready(executor, events, index))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert question == ('scan_0001.tif', 'scan_int1d.nexus') and after is None, (question, after)
    assert 2. < first < 9., f'the first frame waited for the question and its limit: {first:.2f} s'
    assert max(later) < 1.5, f'later frames of a Run known to have no sidecars wait: {later}'
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, i0, _epoch = _output(tmp_path)
    assert labels == (1, 2, 3) and i0 is None, (labels, i0)  # no frame has any metadata
    messages = [record.getMessage() for record in caplog.records]
    assert messages.count('scan_int1d.nexus: scan_0001.tif has no metadata sidecar, and none has been found in this '
                          'Run; asking whether it expects one [LIVE]') == 1, messages
    assert messages.count('This Run has no metadata sidecars (no answer within 2 s): its frames are processed '
                          'without metadata [LIVE]') == 1, messages


def test_under_auto_answering_yes_makes_each_frame_wait_for_its_own_sidecar(tmp_path, monkeypatch, ask_soon):
    """Yes: the Run's type is a sidecar, so the frame waits past the limit, never processed without its own, and is
    processed with it when it lands."""
    monkeypatch.setitem(run_executor._METADATA_WAIT_LIMIT_S, 'sidecar', 1.)
    raw = tmp_path / 'raw'
    raw.mkdir()
    executor, identity = _run(tmp_path, raw, 'auto', 3407)
    events = []
    try:
        _frame(raw, 'scan', 1, 'txt', 1.0, sidecar=False)
        _asked(executor, identity, events)
        executor.answer_metadata_question(identity, True)
        answered = executor.pending_metadata_question(identity)
        _settle(executor, events, 2.)  # past the limit
        held, settled = 1 in _labels(events), executor._exact_run(identity).metadata_kind
        _frame(raw, 'scan', 1, 'txt', 1.0, image=False)
        _frame_ready(executor, events, 1)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert answered is None and not held and settled == 'sidecar', (answered, held, settled)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _output(tmp_path)[:2] == ((1,), (1.0,))


def test_under_auto_answering_no_goes_on_without_metadata_at_once(tmp_path, monkeypatch, ask_soon):
    """No: the frames go on without metadata at once, not after the limit."""
    monkeypatch.setitem(run_executor._METADATA_WAIT_LIMIT_S, 'sidecar', 60.)
    raw = tmp_path / 'raw'
    raw.mkdir()
    executor, identity = _run(tmp_path, raw, 'auto', 3408)
    events = []
    try:
        _frame(raw, 'scan', 1, 'txt', 1.0, sidecar=False)
        _asked(executor, identity, events)
        executor.answer_metadata_question(identity, False)
        answered = _frame_ready(executor, events, 1)
        _frame(raw, 'scan', 2, 'txt', 2.0, sidecar=False)
        later = _frame_ready(executor, events, 2)
        _settle(executor, events)
        settled = executor._exact_run(identity).metadata_kind
    finally:
        terminal = _stop(executor, identity, events)
    assert answered < 5. and later < 1.5 and settled == 'none', (answered, later, settled)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, i0, _epoch = _output(tmp_path)
    assert labels == (1, 2) and i0 is None, (labels, i0)


def test_under_auto_a_sidecar_that_lands_while_the_run_asks_ends_the_question(tmp_path, monkeypatch, ask_soon):
    """A sidecar found withdraws the question: the Run's type is a sidecar and the frame has its own; an answer that
    comes after (a late click) changes nothing."""
    monkeypatch.setitem(run_executor._METADATA_WAIT_LIMIT_S, 'sidecar', 60.)
    raw = tmp_path / 'raw'
    raw.mkdir()
    executor, identity = _run(tmp_path, raw, 'auto', 3409)
    events = []
    try:
        _frame(raw, 'scan', 1, 'txt', 1.0, sidecar=False)
        _asked(executor, identity, events)
        _frame(raw, 'scan', 1, 'txt', 1.0, image=False)
        _frame_ready(executor, events, 1)
        after = executor.pending_metadata_question(identity)
        executor.answer_metadata_question(identity, False)
        _settle(executor, events)
        settled = executor._exact_run(identity).metadata_kind
    finally:
        terminal = _stop(executor, identity, events)
    assert after is None and settled == 'sidecar', (after, settled)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _output(tmp_path)[:2] == ((1,), (1.0,))


def test_under_auto_a_question_whose_frame_left_the_folder_still_ends_at_its_limit(tmp_path, monkeypatch, ask_soon):
    """The image the Run asked about is removed: the question still ends, unanswered, at its limit."""
    monkeypatch.setitem(run_executor._METADATA_WAIT_LIMIT_S, 'sidecar', 1.5)
    raw = tmp_path / 'raw'
    raw.mkdir()
    executor, identity = _run(tmp_path, raw, 'auto', 3410)
    events = []
    try:
        image = _frame(raw, 'scan', 1, 'txt', 1.0, sidecar=False)
        _asked(executor, identity, events)
        image.unlink()
        _settle(executor, events, 2.5)
        after, settled = executor.pending_metadata_question(identity), executor._exact_run(identity).metadata_kind
    finally:
        terminal = _stop(executor, identity, events)
    assert after is None and settled == 'none', (after, settled)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal


def test_under_auto_a_folder_known_to_have_no_sidecars_admits_a_burst_of_frames_together(
        tmp_path, monkeypatch, ask_soon):
    """Once the Run has settled it has no sidecars, the frames that follow are admitted together in one look, as they
    were before the folder waited for sidecars at all (Claude review of 805a9f96: each frame without a sidecar was
    listed as waiting on one poll, given up on at the next and admitted at the third, so a burst of frames came through
    one frame per two polls, five a second at most)."""
    monkeypatch.setitem(run_executor._METADATA_WAIT_LIMIT_S, 'sidecar', 2.)  # 150 s by default
    raw = tmp_path / 'raw'
    raw.mkdir()
    executor, identity = _run(tmp_path, raw, 'auto', 3405)
    events = []
    try:
        _frame(raw, 'scan', 1, 'txt', 1.0, sidecar=False)
        _frame_ready(executor, events, 1)  # no answer within the limit: the Run has no sidecars
        for index in range(2, 14):
            _frame(raw, 'scan', index, 'txt', float(index), sidecar=False)
        burst = _frame_ready(executor, events, 13)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert burst < 1.5, f'twelve frames of a folder known to have no sidecars took {burst:.2f} s: one per two polls'
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, i0, _epoch = _output(tmp_path)
    assert labels == tuple(range(1, 14)) and i0 is None, (labels, i0)


def test_under_auto_a_folder_whose_earlier_scans_have_sidecars_is_known_to_have_them_before_its_first_admission(
        tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'old', 1, 'txt', 4.0)
    seen, real = [], run_executor.materialize_live_directory_group

    def attempt(*args, **kwargs):  # a spy: the attempt is the real one; what the Run knew when it was made
        seen.append(getattr(sys._getframe(1).f_locals['run'], 'metadata_kind', None))
        return real(*args, **kwargs)
    monkeypatch.setattr(run_executor, 'materialize_live_directory_group', attempt)
    executor, identity = _run(tmp_path, raw, 'auto', 3404)
    events = []
    try:
        _frame_ready(executor, events, 1, 'old')
    finally:
        terminal = _stop(executor, identity, events)
    assert seen and seen[0] == 'sidecar', seen
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert _output(tmp_path, 'old')[:2] == ((1,), (4.0,))
