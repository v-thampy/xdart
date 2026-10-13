"""LIVE-QUALIFY-REFUSAL-20261012 frozen live-Run oracle (Claude, October 12): Q1-Q5, Q7, Q9, P1-P2; Q3e, Q3f
and Q4c added before any production edit, after the maintainer chose that Auto waits one poll.

Real live Runs: a StandardRunExecutor watches a directory whose images carry metadata sidecars. Every row keeps
every drained event (no helper that discards events on a timeout), waits for its own predicate, ends with an
explicit Stop, and reads the durable output only after the Run closed: frame labels, recorded I0 and the Append
lineage's last epoch. Injected changes are real file writes, made inside a reader call on the Run's own thread or
between polls. Parent behaviour at 1cfccd74 is in the kickoff's freeze record.
"""
import json
import threading
import time
from pathlib import Path

import fabio
import h5py
import numpy as np
import pytest

import xrd_tools.io.metadata as metadata
import xrd_tools.sources.registry  # noqa: F401  # register built-in readers
from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _start
from xdart.gui.tabs.scattering.adapters import run_executor
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xrd_tools.io.append import science_fingerprint
from xrd_tools.io.nexus import read_scan_metadata
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.selection import DirectorySourceSpec

SIDECARS = {
    'txt': lambda i0: (f'# Counters\nI0 = {i0}\n# Motors\nenergy = 16\n'
                       'User: energy, time: Mon Jan 15 10:30:00 2024  # Temp\n'),
    'pdi': lambda i0: f'All Counters\nI0 = {i0}\n\n# All Motors\nenergy = 16\n#\n1706000000.0',
    'metadata': lambda i0: f'I0={i0}\nenergy=16\n',
}
FORMATS = tuple(SIDECARS)


def _frame(raw, stem, index, fmt, i0, *, sidecar=True, image=True):
    """One frame of an acquisition: its sidecar, then its image renamed into place whole."""
    target = raw / f'{stem}_{index:04d}.tif'
    if sidecar:
        target.with_suffix(f'.{fmt}').write_text(SIDECARS[fmt](i0))
    if image:
        partial = raw / f'{stem}_{index:04d}.part'
        fabio.tifimage.TifImage(data=np.full((195, 487), index, dtype=np.uint16)).write(str(partial))
        partial.replace(target)
    return target


def _run(tmp_path, raw, fmt, request_value):
    poni = tmp_path / 'cal.poni'
    write_poni(poni)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=('.tif',), metadata_format=fmt), poni_file=str(poni),
        project_root=str(raw), save_path=str(tmp_path / 'processed'), output_mode='Overwrite',
        processing_mode='Int 1D', live_mode=True, max_cores=1, bai_1d_args={'npt': 8, 'method': 'numpy'}),
        request_value=request_value)
    return executor, identity


def _terminal(events):
    return any(event.kind in _TERMINAL for event in events)


def _terminals(events):
    return [(event.kind.name, event.detail) for event in events if event.kind in _TERMINAL]


def _labels(events, stem='scan'):
    return [event.frame_key.local_frame_label for event in events
            if event.kind is StandardEventKind.FRAME_READY and f'{stem}_int1d' in str(event.artifact)]


def _said(events, text):
    return any(text in str(event.detail) for event in events)


def _wait(executor, events, predicate, timeout=60.):
    """Drain into ``events``, keeping every batch, until ``predicate(events)`` holds."""
    deadline = time.monotonic() + timeout
    while True:
        events.extend(executor.drain_events())
        if predicate(events):
            return
        if time.monotonic() > deadline:
            raise AssertionError(f'the live Run never reached its predicate; terminal: {_terminals(events)}')
        time.sleep(.01)


def _settle(executor, events, seconds=.5):
    """Keep draining for a few live polls (0.1 s each), so whatever a change triggers has happened."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        events.extend(executor.drain_events())
        time.sleep(.01)


def _stop(executor, identity, events):
    if not _terminal(events):
        executor.stop(identity)
        _wait(executor, events, _terminal)
    executor.close(identity)
    return _terminals(events)


def _published(executor, identity, timeout=30., members=1):
    """The published attempt's source digest and member count, as Append records them, once an attempt with
    ``members`` frames has settled (FRAME_READY arrives before the attempt settles)."""
    deadline = time.monotonic() + timeout
    while not ((attempts := executor.processed_live_revisions(identity))
               and len(attempts[0].decision.item.source_stamp.members) == members):
        assert time.monotonic() < deadline, 'the published attempt never settled'
        time.sleep(.01)
    assert len(attempts) == 1, attempts
    stamp = attempts[0].decision.item.source_stamp
    return science_fingerprint(stamp.execution_identity_v1.as_dict()), len(stamp.members)


def _output(tmp_path, stem='scan'):
    """(frame labels, recorded I0, last Append epoch's source) of one scan's output; None when none was written."""
    path = tmp_path / 'processed' / f'{stem}_int1d.nexus'
    if not path.exists():
        return None
    with h5py.File(path, 'r') as handle:
        labels = tuple(int(value) for value in handle['entry/integrated_1d/frame_index'][()])
        epoch = json.loads(handle['entry/reduction/config/append_lineage'][()])['epochs'][-1]['source']
    data = read_scan_metadata(path)
    i0 = tuple(float(value) for value in data['I0'].values) if 'I0' in data else None
    return labels, i0, epoch


def _worker():
    return threading.current_thread() is not threading.main_thread()


def test_q1_a_sidecar_rewritten_inside_the_settled_recheck_is_retried_then_its_output_stops_growing(
        tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    sidecar, armed, fired, real = raw / 'scan_0001.txt', threading.Event(), [], metadata.read_txt_metadata

    def read(path, *args, **kwargs):
        if armed.is_set() and not fired and _worker():
            fired.append(Path(path).name)
            sidecar.write_text(SIDECARS['txt'](2.0))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(metadata, 'read_txt_metadata', read)
    executor, identity = _run(tmp_path, raw, 'txt', 3101)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        digest, published = _published(executor, identity)
        armed.set()  # frame 1 is reduced and settled: the loop's next re-check reads its sidecar
        _wait(executor, events, lambda values: _said(values, 'scan_0001.txt') or _terminal(values))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert fired == ['scan_0001.txt'], 'the change landed inside a re-check read of the published frame'
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert 'scan_0001.txt' in terminal[0][1], 'the reason is still shown at Stop'
    labels, i0, epoch = _output(tmp_path)
    assert (labels, i0) == ((1,), (1.0,))
    assert (epoch['digest'], epoch['extent']) == (digest, published)


def test_q2_a_new_frames_sidecar_rewritten_inside_its_admission_read_is_admitted_later(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    sidecar, fired, real = raw / 'scan_0002.txt', [], metadata.read_txt_metadata

    def read(path, *args, **kwargs):
        if Path(path).name == 'scan_0002.txt' and not fired and _worker():
            fired.append(Path(path).name)
            sidecar.write_text(SIDECARS['txt'](3.0))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(metadata, 'read_txt_metadata', read)
    executor, identity = _run(tmp_path, raw, 'txt', 3102)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        _frame(raw, 'scan', 2, 'txt', 2.0)
        _wait(executor, events, lambda values: 2 in _labels(values) or _terminal(values))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert fired == ['scan_0002.txt'], 'the change landed inside the new frame\'s first read'
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, i0, _epoch = _output(tmp_path)
    assert (labels, i0) == ((1, 2), (1.0, 3.0))


@pytest.mark.parametrize('fmt', FORMATS)
def test_q3a_a_sidecar_landing_after_its_image_is_waited_for(tmp_path, fmt):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, fmt, 1.0)
    executor, identity = _run(tmp_path, raw, fmt, 3110 + FORMATS.index(fmt))
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        _frame(raw, 'scan', 2, fmt, 2.0, sidecar=False)
        time.sleep(.3)
        _frame(raw, 'scan', 2, fmt, 2.0, image=False)  # its sidecar, 0.3 s after the image
        _wait(executor, events, lambda values: 2 in _labels(values) or _terminal(values))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, i0, _epoch = _output(tmp_path)
    assert (labels, i0) == ((1, 2), (1.0, 2.0))


@pytest.mark.parametrize('fmt', FORMATS)
def test_q3b_a_frame_whose_sidecar_is_missing_when_the_run_starts_waits_and_is_reported_pending(tmp_path, fmt):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, fmt, 1.0, sidecar=False)
    executor, identity = _run(tmp_path, raw, fmt, 3120 + FORMATS.index(fmt))
    events = []
    try:
        _settle(executor, events, 1.)  # ten polls with the sidecar missing
        waiting = [event.detail for event in events if event.kind is StandardEventKind.DISCOVERY]
        reduced_early = list(_labels(events))
        _frame(raw, 'scan', 1, fmt, 1.0, image=False)  # the sidecar lands
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert reduced_early == [], 'nothing was reduced before its sidecar landed'
    assert waiting and '1 pending' in waiting[-1], waiting
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, i0, _epoch = _output(tmp_path)
    assert (labels, i0) == ((1,), (1.0,))


@pytest.mark.parametrize('fmt', FORMATS)
def test_q3c_a_sidecar_still_growing_is_not_read_until_it_holds_still(tmp_path, fmt):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, fmt, 1.0)
    executor, identity = _run(tmp_path, raw, fmt, 3130 + FORMATS.index(fmt))
    events, sidecar, text = [], raw / f'scan_0002.{fmt}', SIDECARS[fmt](2.0)
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        _frame(raw, 'scan', 2, fmt, 2.0, sidecar=False)
        for end in range(1, len(text), max(1, len(text) // 20)):  # pieces 30 ms apart: faster than a poll
            sidecar.write_text(text[:end])
            time.sleep(.03)
        sidecar.write_text(text)
        _wait(executor, events, lambda values: 2 in _labels(values) or _terminal(values))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, i0, _epoch = _output(tmp_path)
    assert (labels, i0) == ((1, 2), (1.0, 2.0))


def test_q3d_a_pending_frame_holds_up_neither_another_scan_nor_unrelated_directory_churn(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'alpha', 1, 'txt', 1.0, sidecar=False)  # alpha's sidecar never lands
    executor, identity = _run(tmp_path, raw, 'txt', 3140)
    events, quiet = [], threading.Event()

    def churn():
        index = 0
        while not quiet.is_set():
            (raw / f'noise_{index:05d}.dat').write_bytes(b'x')
            index += 1
            time.sleep(.03)
    noise = threading.Thread(target=churn)
    noise.start()
    try:
        _settle(executor, events, .5)
        _frame(raw, 'beta', 1, 'txt', 5.0)
        _wait(executor, events, lambda values: 1 in _labels(values, 'beta') or _terminal(values))
        _settle(executor, events)
    finally:
        quiet.set()
        noise.join()
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, i0, _epoch = _output(tmp_path, 'beta')
    assert (labels, i0) == ((1,), (5.0,))
    assert _output(tmp_path, 'alpha') is None, 'alpha stayed pending: nothing was reduced without its sidecar'
    assert '1 pending' in terminal[0][1], terminal


def test_q3e_under_auto_a_sidecar_landing_just_after_its_image_is_picked_up(tmp_path, monkeypatch):
    """Maintainer, October 12: under Auto a frame waits one poll, so a near-simultaneous sidecar is read."""
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    armed, wrote, real = threading.Event(), [], run_executor.materialize_live_directory_group

    def materialize(*args, **kwargs):
        attempt = real(*args, **kwargs)
        if armed.is_set() and not wrote and any(Path(path).name == 'scan_0002.tif' for path in args[3].physical_paths):
            # The sidecar lands 30 ms after the first admission attempt saw the image alone: after an immediate
            # construction has checked the frame, well before the next poll.
            wrote.append(threading.Timer(.03, (raw / 'scan_0002.txt').write_text, (SIDECARS['txt'](2.0),)))
            wrote[0].start()
        return attempt
    monkeypatch.setattr(run_executor, 'materialize_live_directory_group', materialize)
    executor, identity = _run(tmp_path, raw, 'auto', 3220)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        armed.set()
        _frame(raw, 'scan', 2, 'txt', 2.0, sidecar=False)
        _wait(executor, events, lambda values: 2 in _labels(values) or _terminal(values))
        _settle(executor, events)
    finally:
        for timer in wrote:
            timer.join()
        terminal = _stop(executor, identity, events)
    assert len(wrote) == 1
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert not _said(events, 'scan_0002.txt'), 'no output was stopped'
    labels, i0, _epoch = _output(tmp_path)
    assert (labels, i0) == ((1, 2), (1.0, 2.0))


def test_q3f_under_auto_a_frame_without_any_sidecar_is_still_reduced_without_metadata(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    executor, identity = _run(tmp_path, raw, 'auto', 3230)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        _frame(raw, 'scan', 2, 'txt', 2.0, sidecar=False)  # no sidecar, ever
        _wait(executor, events, lambda values: 2 in _labels(values) or _terminal(values))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, i0, _epoch = _output(tmp_path)
    assert labels == (1, 2) and i0[0] == 1.0 and np.isnan(i0[1]), (labels, i0)


def test_q4a_a_published_sidecar_changed_at_the_same_extent_stops_its_output_growing(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    executor, identity = _run(tmp_path, raw, 'txt', 3150)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        digest, published = _published(executor, identity)
        (raw / 'scan_0001.txt').write_text(SIDECARS['txt'](2.0))  # between reads, after frame 1 was published
        _wait(executor, events, lambda values: _said(values, 'scan_0001.txt') or _terminal(values))
        _frame(raw, 'scan', 2, 'txt', 2.0)  # a new member of the output that stopped growing
        _wait(executor, events, lambda values: _said(values, '1 skipped') or _terminal(values))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert 'scan_0001.txt' in terminal[0][1] and '1 skipped' in terminal[0][1], terminal
    assert 2 not in _labels(events)
    labels, i0, epoch = _output(tmp_path)
    assert (labels, i0) == ((1,), (1.0,))
    assert (epoch['digest'], epoch['extent']) == (digest, published)


def test_q4b_a_published_sidecar_change_arriving_with_the_next_image_is_still_fenced(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    armed, inside, released, real = threading.Event(), threading.Event(), threading.Event(), DirectoryIndexSession.observe

    def observe(self, *, refresh=True):
        if armed.is_set() and not inside.is_set() and _worker():
            inside.set()
            released.wait(30.)
        return real(self, refresh=refresh)
    monkeypatch.setattr(DirectoryIndexSession, 'observe', observe)
    executor, identity = _run(tmp_path, raw, 'txt', 3160)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        digest, published = _published(executor, identity)
        armed.set()
        assert inside.wait(30.), 'the live loop is held before its next observation'
        (raw / 'scan_0001.txt').write_text(SIDECARS['txt'](2.0))
        _frame(raw, 'scan', 2, 'txt', 2.0)
        released.set()  # one observation sees the changed sidecar and the new image together
        _frame(raw, 'other', 1, 'txt', 7.0)
        _wait(executor, events, lambda values: 1 in _labels(values, 'other') or _terminal(values))
        _settle(executor, events)
    finally:
        released.set()
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert 'scan_0001.txt' in terminal[0][1], terminal
    labels, i0, epoch = _output(tmp_path)
    assert (labels, i0) == ((1,), (1.0,)), 'the output did not grow past the changed published frame'
    assert (epoch['digest'], epoch['extent']) == (digest, published)
    labels, i0, _epoch = _output(tmp_path, 'other')
    assert (labels, i0) == ((1,), (7.0,))


def test_q4c_under_auto_a_sidecar_appearing_after_its_frame_was_published_stops_that_output(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    executor, identity = _run(tmp_path, raw, 'auto', 3240)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        _frame(raw, 'scan', 2, 'txt', 2.0, sidecar=False)
        _wait(executor, events, lambda values: 2 in _labels(values) or _terminal(values))
        digest, published = _published(executor, identity, members=2)
        (raw / 'scan_0002.txt').write_text(SIDECARS['txt'](2.0))  # appears after frame 2 was published
        _wait(executor, events, lambda values: _said(values, 'scan_0002.txt') or _terminal(values))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert 'scan_0002.txt' in terminal[0][1], terminal
    labels, i0, epoch = _output(tmp_path)
    assert labels == (1, 2) and i0[0] == 1.0 and np.isnan(i0[1]), (labels, i0)
    assert (epoch['digest'], epoch['extent']) == (digest, published)


def test_q5_an_explicit_sidecar_that_never_arrives_keeps_its_frame_pending_until_stop(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    executor, identity = _run(tmp_path, raw, 'txt', 3170)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        _frame(raw, 'scan', 2, 'txt', 2.0, sidecar=False)  # its sidecar never lands
        _settle(executor, events, 1.)
        waiting = [event.detail for event in events if event.kind is StandardEventKind.DISCOVERY]
    finally:
        terminal = _stop(executor, identity, events)
    assert 2 not in _labels(events), 'frame 2 was not reduced without its sidecar'
    assert waiting and '1 pending' in waiting[-1], waiting
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert '1 pending' in terminal[0][1], terminal
    labels, i0, _epoch = _output(tmp_path)
    assert (labels, i0) == ((1,), (1.0,))


def test_q7_an_unrelated_read_error_in_a_live_recheck_still_fails_the_run(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    armed, fired, real = threading.Event(), [], metadata.read_txt_metadata

    def read(path, *args, **kwargs):
        if armed.is_set() and not fired and _worker():
            fired.append(Path(path).name)
            raise PermissionError(f'permission denied (injected): {path}')
        return real(path, *args, **kwargs)
    monkeypatch.setattr(metadata, 'read_txt_metadata', read)
    executor, identity = _run(tmp_path, raw, 'txt', 3180)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        armed.set()
        _wait(executor, events, _terminal)
    finally:
        terminal = _stop(executor, identity, events)
    assert fired == ['scan_0001.txt']
    assert [kind for kind, _detail in terminal] == ['FAILED'], terminal
    assert 'permission denied (injected)' in terminal[0][1], terminal


def test_q9_the_reason_arrives_with_unchanged_counts_and_survives_other_progress_and_stop(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    executor, identity = _run(tmp_path, raw, 'txt', 3190)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        (raw / 'scan_0001.txt').write_text(SIDECARS['txt'](2.0))
        _wait(executor, events, lambda values: _said(values, 'scan_0001.txt') or _terminal(values))
        _frame(raw, 'other', 1, 'txt', 7.0)
        _wait(executor, events, lambda values: 1 in _labels(values, 'other') or _terminal(values))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    warned = [event for event in events if 'scan_0001.txt' in str(event.detail)]
    assert warned, _terminals(events)
    first = warned[0]
    assert first.kind is StandardEventKind.DISCOVERY
    assert (first.files_processed, first.files_skipped, first.files_pending, first.files_discovered) == (1, 0, 0, 1)
    other = [event for event in events
             if event.kind is StandardEventKind.FRAME_READY and 'other_int1d' in str(event.artifact)]
    assert other and 'scan_0001.txt' in str(other[-1].detail), 'later progress keeps the reason'
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert 'scan_0001.txt' in terminal[0][1], terminal


def test_p1_a_change_inside_the_construct_recheck_is_still_rescued(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    sidecar, armed, fired, state, calls = raw / 'scan_0002.txt', threading.Event(), [], threading.local(), []
    real_validate, real_read = run_executor.validate_planned_source, metadata.read_txt_metadata

    def validate(item, **kwargs):
        pair = len(item.source_stamp.members) == 2
        state.construct = armed.is_set() and pair and not calls
        calls.extend([1] if pair else [])
        try:
            return real_validate(item, **kwargs)
        finally:
            state.construct = False

    def read(path, *args, **kwargs):
        if getattr(state, 'construct', False) and not fired and Path(path).name == 'scan_0002.txt':
            fired.append(Path(path).name)
            with open(sidecar, 'a') as stream:
                stream.write('late = 1\n')
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(run_executor, 'validate_planned_source', validate)
    monkeypatch.setattr(metadata, 'read_txt_metadata', read)
    executor, identity = _run(tmp_path, raw, 'txt', 3200)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        armed.set()
        _frame(raw, 'scan', 2, 'txt', 2.0)
        _wait(executor, events, lambda values: 2 in _labels(values) or _terminal(values))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert fired == ['scan_0002.txt'], 'the change landed inside _construct\'s first re-check'
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, i0, _epoch = _output(tmp_path)
    assert (labels, i0) == ((1, 2), (1.0, 2.0))


def test_p2_a_sidecar_first_acquisition_runs_to_stop_with_every_frame_and_its_metadata(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    executor, identity = _run(tmp_path, raw, 'txt', 3210)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        for index in (2, 3, 4):
            _frame(raw, 'scan', index, 'txt', float(index))
            time.sleep(.3)
        _wait(executor, events, lambda values: {1, 2, 3, 4} <= set(_labels(values)) or _terminal(values))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, i0, _epoch = _output(tmp_path)
    assert (labels, i0) == ((1, 2, 3, 4), (1.0, 2.0, 3.0, 4.0))
