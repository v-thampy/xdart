"""A Live directory Append keeps each output's writer, display and hydration in one generation (Codex review
N-0243a1bb of dev 4c28033b, F1/F2; maintainer 2026-10-23: "the biggest issues are live directory mode in append mode
regularly crashing, ...").

F1: qualifying another scan settled the scan still growing before it knew whether that scan could run, and the
grown scan's next frame then failed on its own display store, "cannot bind an allocation to a populated store". A scan
that stops growing while it is the current output is retired at once. F2: a scan shown from its saved record, or
settled to admit another, that grows again starts a new display generation under its writer's checkpoint authority;
a read still in flight from the closed generation never lands in the new one."""
from __future__ import annotations

import os
import threading
import time

import pytest

from tests.xdart.scattering.test_live_qualify_refusal_run import (
    SIDECARS, _frame, _labels, _output, _published, _said, _settle, _stop, _terminal, _terminals, _wait,
)
from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal, _start
from xdart.gui.tabs.scattering import hydration_transport
from xdart.gui.tabs.scattering.adapters import dynamic_output, run_executor
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_runtime import RunDisplayState
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xrd_tools.session.hydration import HydrationOutcome
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def _intent(tmp_path, raw, *, live, mode='Int 1D', output='Append'):
    poni = tmp_path / 'cal.poni'
    write_poni(poni)
    return RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=('.tif',), metadata_format='txt'), poni_file=str(poni),
        project_root=str(raw), save_path=str(tmp_path / 'processed'), output_mode=output,
        processing_mode=mode, live_mode=live, max_cores=1, bai_1d_args={'npt': 8, 'method': 'numpy'},
        bai_2d_args={'npt_rad': 8, 'npt_azim': 4, 'method': 'numpy'})


def _saved(tmp_path, raw, stem, count, request_value, *, mode='Int 1D'):
    """``count`` frames of ``stem`` reduced by a finished directory Run: the output a later Live Run appends to."""
    for index in range(1, count + 1):
        _frame(raw, stem, index, 'txt', float(index))
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False, mode=mode),
                                                  request_value=request_value)
    executor.close(identity)
    assert [event.kind.name for event in events if event.kind in _TERMINAL] == ['FINISHED']


def _key(executor, identity, stem, label):
    catalog = executor.frame_catalog(identity)
    return next((key for key in (() if catalog is None else catalog.entries)
                 if key.source_scan == stem and key.local_frame_label == label), None)


def _owner(executor, identity, stem):
    run = executor._exact_run(identity)
    return next(owner for owner in run.display.artifacts.values() if owner.source_scan == stem)


def _ready(events, stem):
    return [event.frame_key.local_frame_label for event in events
            if event.kind is StandardEventKind.FRAME_READY and os.path.basename(str(event.artifact)).startswith(
                f'{stem}_int')]


def _show(executor, identity, events, key, timeout=30., *, running=True):
    """What the page shows for one frame through the Run's own display owner: the payload once its publication is
    resident, read from the saved file or the writer's checkpoint first when it is not."""
    assert key is not None, f'no such frame; terminal: {_terminals(events)}'
    run, context = executor._exact_run(identity), executor.acquisition_context(identity)
    deadline = time.monotonic() + timeout
    while (payload := run.display.project(key, 1, closed=run.closed, owner=context.hydration_owner,
                                          commit_gate=context.commit_gate)) is None:
        assert time.monotonic() < deadline and not (running and _terminal(events)), (key, _terminals(events))
        _settle(executor, events, .05)
    return payload


def _full_raw(executor, identity, events, key, timeout=30., *, running=True):
    run, context = executor._exact_run(identity), executor.acquisition_context(identity)
    assert run.display.request_full(key, 1, owner=context.hydration_owner,
                                    commit_gate=context.commit_gate) is not None
    deadline = time.monotonic() + timeout
    while not (status := run.display.full_raw_status(key))[0] and status[2] is None:
        assert time.monotonic() < deadline and not (running and _terminal(events)), (key, _terminals(events))
        _settle(executor, events, .05)
    return status


@pytest.mark.parametrize('neighbour', ['waits', 'refuses'])
def test_f1_a_neighbour_that_waits_or_is_refused_leaves_the_growing_output_current(tmp_path, monkeypatch, neighbour):
    raw = tmp_path / 'raw'
    raw.mkdir()
    if neighbour == 'refuses':
        _saved(tmp_path, raw, 'old', 1, 8811)
    attempts, real = [], run_executor.materialize_live_directory_group

    def materialize(receipt, configuration, session, group, **kwargs):
        attempts.append(group.target.name)
        return real(receipt, configuration, session, group, **kwargs)
    monkeypatch.setattr(run_executor, 'materialize_live_directory_group', materialize)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, live=True), request_value=8812)
    events = []
    try:
        _frame(raw, 'grow', 1, 'txt', 1.0)
        _wait(executor, events, lambda values: 1 in _labels(values, 'grow') or _terminal(values))
        if neighbour == 'waits':  # its image lands before its sidecar
            _frame(raw, 'wait', 1, 'txt', 1.0, sidecar=False)
            _wait(executor, events, lambda values: 'wait_int1d.nexus' in attempts or _terminal(values))
        else:  # the beamline touches an image it reduced earlier: that scan's Append is refused
            image = raw / 'old_0001.tif'
            stat = image.stat()
            os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
            _wait(executor, events, lambda values: _said(values, 'its Append was refused') or _terminal(values))
        # The page goes on showing the growing scan's frame meanwhile.
        shown = _show(executor, identity, events, _key(executor, identity, 'grow', 1))
        _frame(raw, 'grow', 2, 'txt', 2.0)
        _wait(executor, events, lambda values: 2 in _labels(values, 'grow') or _terminal(values))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert shown.view.intensity_1d is not None
    assert sorted(set(_labels(events, 'grow'))) == [1, 2]
    assert _output(tmp_path, 'grow')[0] == (1, 2)
    if neighbour == 'waits':
        assert '1 pending' in terminal[0][1], terminal
    else:
        assert _output(tmp_path, 'old')[0] == (1,)
        assert 'old_int1d.nexus' in str(next(event for event in events if event.kind is StandardEventKind.STOPPED)
                                        .warnings)


@pytest.mark.parametrize('change', ['refused', 'drift', 'writer-fails'])
def test_f1_a_current_output_that_stops_growing_is_retired_at_once(tmp_path, monkeypatch, change):
    raw = tmp_path / 'raw'
    raw.mkdir()
    armed, real_finish = threading.Event(), dynamic_output.NexusSink.finish

    def finish(self, *args, **kwargs):  # the writer's own finish, which retiring a committed output reaches
        if armed.is_set():
            raise OSError('the share went away (injected)')
        return real_finish(self, *args, **kwargs)
    monkeypatch.setattr(dynamic_output.NexusSink, 'finish', finish)
    executor = StandardRunExecutor(join_timeout=2.)
    # Overwrite reaches the published-metadata drift check (Append's own qualification refuses that change first).
    identity = _start(executor, _intent(tmp_path, raw, live=True, output='Overwrite' if change == 'drift' else 'Append'),
                      request_value=8821)
    events, retired = [], None
    try:
        _frame(raw, 'grow', 1, 'txt', 1.0)
        _wait(executor, events, lambda values: 1 in _labels(values, 'grow') or _terminal(values))
        _published(executor, identity)  # its attempt has settled: a later change is news
        if change == 'drift':  # its sidecar changes after its frame was reduced
            (raw / 'grow_0001.txt').write_text(SIDECARS['txt'](2.0))
            reason = 'grow_0001.txt'
        else:  # an image it already reduced is touched: its Append is refused
            image = raw / 'grow_0001.tif'
            stat = image.stat()
            if change == 'writer-fails':
                armed.set()  # and its writer cannot close
            os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
            reason = 'its Append was refused'
        _wait(executor, events, lambda values: _said(values, reason) or _terminal(values))
        if change != 'writer-fails':
            run, owner = executor._exact_run(identity), _owner(executor, identity, 'grow')
            retired = (run.output.owns_target(owner.artifact), owner.hydration_closed, owner.light_lease)
            _frame(raw, 'next', 1, 'txt', 3.0)
            _wait(executor, events, lambda values: 1 in _labels(values, 'next') or _terminal(values))
            _settle(executor, events)
    finally:
        armed.clear()
        terminal = _stop(executor, identity, events)
    if change == 'writer-fails':  # a writer failure still fails the Run; it is not an Append skip
        assert [kind for kind, _detail in terminal] == ['FAILED'], terminal
        assert 'the share went away (injected)' in terminal[0][1], terminal
        return
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert retired == (False, True, None), 'the stopped output kept its writer and display open'
    assert reason in terminal[0][1], terminal
    assert _output(tmp_path, 'grow')[0] == (1,) and _output(tmp_path, 'next')[0] == (1,)


def test_f2_a_saved_scan_that_grows_reopens_under_its_writers_checkpoint_authority(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _saved(tmp_path, raw, 'saved', 2, 8831, mode='Int 2D')
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, live=True, mode='Int 2D'), request_value=8832)
    events = []
    try:
        _wait(executor, events, lambda values: _key(executor, identity, 'saved', 2) is not None or _terminal(values))
        first = _key(executor, identity, 'saved', 1)
        closed = _show(executor, identity, events, first)  # read from the saved file
        closed_raw = _full_raw(executor, identity, events, first)
        _frame(raw, 'saved', 3, 'txt', 3.0)  # the same scan grows
        _wait(executor, events, lambda values: 3 in _ready(values, 'saved') or _terminal(values))
        _settle(executor, events)
        owner = _owner(executor, identity, 'saved')
        active = (owner.hydration_closed, owner.publications.allocation is not None)
        grown = _show(executor, identity, events, _key(executor, identity, 'saved', 3))
        # The saved frames now answer to the writer's checkpoint authority, as any Live Append's earlier frames do:
        # Full Raw needs its checkpoint token (none yet after one epoch), and no closed read is made meanwhile.
        run, context = executor._exact_run(identity), executor.acquisition_context(identity)
        held = (owner.records._checkpoint_hydration_authority()[0],
                run.display.request_full(first, 1, owner=context.hydration_owner, commit_gate=context.commit_gate))
        catalog = [(key.source_scan, key.local_frame_label) for key in executor.frame_catalog(identity).entries]
        executor.stop(identity)
        _wait(executor, events, _terminal)
        again = _show(executor, identity, events, first, running=False)  # the Run's end closes the output again
        closed_again_raw = _full_raw(executor, identity, events, first, running=False)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert active == (False, True), 'the grown scan kept its closed display generation'
    assert catalog == [('saved', 1), ('saved', 2), ('saved', 3)], catalog
    assert held[1] is None or held[0] is not None, held
    for payload in (closed, grown, again):
        assert payload.view.intensity_1d is not None and payload.view.has_2d, payload
    # Shown only from its saved record, the scan has no saturation ceiling yet (a live frame stamps it), so Full Raw
    # explains itself; once its writer has run, the closed output's Full Raw is resident.
    assert closed_raw == (False, False, 'detector mask/value projection unavailable'), closed_raw
    assert closed_again_raw[:2] == (True, False), closed_again_raw


def test_f2_a_read_from_the_closed_generation_never_lands_in_the_reopened_one(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _saved(tmp_path, raw, 'saved', 2, 8841)
    entered, release, real_read = threading.Event(), threading.Event(), hydration_transport.read_frame_preview

    def read(read_key, **kwargs):
        if not entered.is_set() and read_key.frame_identity == 1:
            entered.set()
            release.wait(30.)
        return real_read(read_key, **kwargs)
    monkeypatch.setattr(hydration_transport, 'read_frame_preview', read)
    reopening, real_begin = threading.Event(), getattr(RunDisplayState, 'begin_active_generation', None)

    def begin(self, *args, **kwargs):
        reopening.set()
        return real_begin(self, *args, **kwargs)
    monkeypatch.setattr(RunDisplayState, 'begin_active_generation', begin, raising=False)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, live=True), request_value=8842)
    events = []
    try:
        _wait(executor, events, lambda values: _key(executor, identity, 'saved', 2) is not None or _terminal(values))
        run, context = executor._exact_run(identity), executor.acquisition_context(identity)
        first = _key(executor, identity, 'saved', 1)
        assert run.display.project(first, 1, closed=run.closed, owner=context.hydration_owner,
                                   commit_gate=context.commit_gate) is None
        assert entered.wait(10.), 'the saved frame was never read'
        mismatched = run.display.transport.counters().get(HydrationOutcome.OWNER_MISMATCH, 0)
        _frame(raw, 'saved', 3, 'txt', 3.0)  # the scan grows while that read is in flight
        # The read finishes once the scan's writer waits for it (4c28033b, which does not wait: once it has written).
        _wait(executor, events, lambda values: reopening.is_set() or 3 in _labels(values, 'saved')
              or _terminal(values))
        release.set()
        _wait(executor, events, lambda values: 3 in _labels(values, 'saved') or _terminal(values))
        deadline = time.monotonic() + 10.
        while run.display.transport.counters().get(HydrationOutcome.OWNER_MISMATCH, 0) == mismatched \
                and time.monotonic() < deadline and not _terminal(events):
            _settle(executor, events, .05)
        landed = _owner(executor, identity, 'saved').publications.get(1)
        refused = run.display.transport.counters().get(HydrationOutcome.OWNER_MISMATCH, 0) - mismatched
    finally:
        release.set()
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert landed is None and refused == 1, (landed, refused)
    assert _output(tmp_path, 'saved')[0] == (1, 2, 3)


def test_f2_a_reopened_scan_takes_no_read_before_its_writer_binds_it(tmp_path, monkeypatch):
    """Codex N-56ea524a (its discriminator test_review_generation_admission.py, 75ca5f2a...): between the new
    generation's stores and the writer's allocation a page read hydrated the fresh store, and the allocation was
    refused. The new generation answers to the writer's checkpoint authority from the start."""
    raw = tmp_path / 'raw'
    raw.mkdir()
    _saved(tmp_path, raw, 'saved', 2, 8851)
    rotated, resume, real_begin = threading.Event(), threading.Event(), RunDisplayState.begin_active_generation

    def begin(self, owner, **kwargs):  # the window between the new stores and the writer's allocation, held open
        real_begin(self, owner, **kwargs)
        rotated.set()
        assert resume.wait(30.), 'the window was never closed'
    monkeypatch.setattr(RunDisplayState, 'begin_active_generation', begin)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, live=True), request_value=8852)
    events = []
    try:
        _wait(executor, events, lambda values: _key(executor, identity, 'saved', 2) is not None or _terminal(values))
        first, retired = _key(executor, identity, 'saved', 1), _owner(executor, identity, 'saved').records
        _frame(raw, 'saved', 3, 'txt', 3.0)
        _wait(executor, events, lambda values: rotated.is_set() or _terminal(values))
        run, context, owner = executor._exact_run(identity), executor.acquisition_context(identity), \
            _owner(executor, identity, 'saved')
        before, deadline = run.display.transport.counters(), time.monotonic() + 1.
        while time.monotonic() < deadline:  # the page keeps asking for that frame meanwhile
            assert run.display.project(first, 1, closed=run.closed, owner=context.hydration_owner,
                                       commit_gate=context.commit_gate) is None
            _settle(executor, events, .05)
        window = (owner.records is not retired, owner.records._checkpoint_hydration_required(),
                  owner.publications.get(1), run.display.transport.counters() == before)
        resume.set()
        _wait(executor, events, lambda values: 3 in _labels(values, 'saved') or _terminal(values))
        _settle(executor, events)
    finally:
        resume.set()
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert window == (True, True, None, True), window
    assert _output(tmp_path, 'saved')[0] == (1, 2, 3)


def test_f2_a_read_asked_for_across_the_reopen_follows_the_new_generation(tmp_path, monkeypatch):
    """Codex N-56ea524a window 1: a page read that saw the closed generation but took its stores after the reopen
    carried the closed authority into the new stores and committed there unchecked."""
    raw = tmp_path / 'raw'
    raw.mkdir()
    _saved(tmp_path, raw, 'saved', 2, 8861)
    asked, held, real_request = threading.Event(), {}, RunDisplayState._request_preview

    def request(self, art, *args, **kwargs):  # the page's request, built only once the stores have changed
        if held.pop('retired', None) is not None and art.source_scan == 'saved':
            asked.set()
            deadline = time.monotonic() + 30.
            while art.records is retired and time.monotonic() < deadline:
                time.sleep(.005)
        return real_request(self, art, *args, **kwargs)
    monkeypatch.setattr(RunDisplayState, '_request_preview', request)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, live=True), request_value=8862)
    events = []
    try:
        _wait(executor, events, lambda values: _key(executor, identity, 'saved', 2) is not None or _terminal(values))
        run, context = executor._exact_run(identity), executor.acquisition_context(identity)
        first, owner = _key(executor, identity, 'saved', 1), _owner(executor, identity, 'saved')
        retired = held['retired'] = owner.records
        before = run.display.transport.counters()
        page = threading.Thread(target=run.display.project, args=(first, 1), kwargs=dict(
            closed=run.closed, owner=context.hydration_owner, commit_gate=context.commit_gate))
        page.start()
        assert asked.wait(10.), 'the page never asked'
        _frame(raw, 'saved', 3, 'txt', 3.0)
        page.join(30.)
        _wait(executor, events, lambda values: 3 in _labels(values, 'saved') or _terminal(values))
        _settle(executor, events)
        landed = (owner.records is not retired, owner.publications.get(1), run.display.transport.counters() == before)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert landed == (True, None, True), landed
    assert _output(tmp_path, 'saved')[0] == (1, 2, 3)


def test_f2_a_read_admitted_behind_the_reopen_never_reads_the_new_writers_file(tmp_path, monkeypatch):
    """Codex N-56ea524a window 2: with one read active and one queued, a page read of the closing scan was admitted
    as a replacement, installed only after the reopen, missed by its settlement, and read the file the new writer
    had opened."""
    raw = tmp_path / 'raw'
    raw.mkdir()
    for stem in ('ascan', 'bscan'):
        _frame(raw, stem, 1, 'txt', 1.0)
    _saved(tmp_path, raw, 'saved', 2, 8871)
    release, holding, reads, retired = threading.Event(), threading.Event(), [], {}
    real_read, real_dispatch = hydration_transport.read_frame_preview, hydration_transport.HydrationTransport.dispatch_detached

    def read(read_key, **kwargs):
        name = os.path.basename(str(read_key.artifact_identity))
        owner = retired.get('owner')
        reads.append((name, read_key.frame_identity, owner is not None and owner.records is not retired['records']))
        if name == 'ascan_int1d.nexus':  # the read already active
            release.wait(30.)
        return real_read(read_key, **kwargs)

    def dispatch(self, mutation):  # the replacement is installed only after the reopen
        entry = getattr(mutation, 'replacement', None)
        owner = retired.get('owner')
        if entry is not None and owner is not None and entry.request.stores[1] is retired['publications']:
            holding.set()
            deadline = time.monotonic() + 30.
            while owner.records is retired['records'] and time.monotonic() < deadline:
                time.sleep(.005)
        return real_dispatch(self, mutation)
    monkeypatch.setattr(hydration_transport, 'read_frame_preview', read)
    monkeypatch.setattr(hydration_transport.HydrationTransport, 'dispatch_detached', dispatch)
    executor = StandardRunExecutor(join_timeout=2.)
    identity = _start(executor, _intent(tmp_path, raw, live=True), request_value=8872)
    events, pages = [], []
    try:
        _wait(executor, events, lambda values: _key(executor, identity, 'saved', 2) is not None or _terminal(values))
        run, context = executor._exact_run(identity), executor.acquisition_context(identity)
        owner = _owner(executor, identity, 'saved')
        retired.update(owner=owner, records=owner.records, publications=owner.publications)
        for stem in ('ascan', 'bscan', 'saved'):  # active, queued, then the replacement
            page = threading.Thread(target=run.display.project, args=(_key(executor, identity, stem, 1), 1),
                                    kwargs=dict(closed=run.closed, owner=context.hydration_owner,
                                                commit_gate=context.commit_gate))
            page.start()
            pages.append(page)
            if stem == 'ascan':  # its read is the active one
                assert _wait_for(lambda: any(row[0] == 'ascan_int1d.nexus' for row in reads))
            elif stem == 'bscan':  # queued behind it
                page.join(5.)
        assert holding.wait(10.), 'the replacement was never admitted'
        _frame(raw, 'saved', 3, 'txt', 3.0)
        _wait(executor, events, lambda values: 3 in _labels(values, 'saved') or _terminal(values))
        release.set()
        for page in pages:
            page.join(30.)
        _settle(executor, events, 1.)
    finally:
        release.set()
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert not [row for row in reads if row[0] == 'saved_int1d.nexus' and row[2]], reads
    assert _output(tmp_path, 'saved')[0] == (1, 2, 3)


def _wait_for(predicate, timeout=10.):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(.005)
    return True
