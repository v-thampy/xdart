"""LIVE-SPEC-METADATA-20261012 oracle completion (Claude, October 13, from Codex's pre-writer review): R1-R10.

Real live Runs over a SPEC acquisition, as in L1-L8: own-row custody between readiness, qualification and the
reduction read (CR1); published-metadata removal, rewrite and raw drift while a frame waits, and Stop inside the new
currency check and inside JIT revalidation (CR2); progressive publication (CR4); and a saved live output's boundary:
unchanged inputs replay and Append, a grown SPEC file refuses both (CR3). Every row keeps every drained event, waits
for its own predicate, ends with an explicit Stop (or the Run's own terminal) and reads durable output after close.
Injected changes are real file writes, made inside a reader call on the Run's own thread or between polls. Parent
behaviour is in the kickoff's freeze record.
"""
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import silx.io.specfile as specfile

import xrd_tools.io.metadata as metadata
import xrd_tools.sources.image as image_source
from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_live_metadata_race_run import _reregister_image_adapter
from tests.xdart.scattering.test_live_qualify_refusal_run import (
    _labels, _said, _settle, _stop, _terminal, _wait, _worker,
)
from tests.xdart.scattering.test_live_spec_metadata_run import (
    STEM, _acquisition, _image, _output, _reduced, _row, _run,
)
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _run_to_terminal
from xdart.gui.tabs.scattering import output_preflight
from xdart.gui.tabs.scattering.adapters import run_executor
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.selection import DirectorySourceSpec

SCAN1 = STEM.format(scan=1)


def _replace(spec, old, new):
    text = spec.read_text()
    assert text.count(old) == 1
    spec.write_text(text.replace(old, new))  # in place: truncate, then write


def _stopped_growing(values):
    return _said(values, 'stopped growing') or _terminal(values)


def _kinds(terminal):
    return [kind for kind, _detail in terminal]


def test_r1_a_row_removed_after_readiness_waits_then_publishes_its_own_values(tmp_path, monkeypatch):
    raw, spec = _acquisition(tmp_path)
    real = getattr(metadata, 'spec_rows_ready', None)
    removed = threading.Event()

    def ready(paths, *args, **kwargs):
        count = real(paths, *args, **kwargs)
        if (not removed.is_set() and _worker()
                and any(Path(path).name == f'{SCAN1}_0001.tif' for path in tuple(paths)[:count])):
            _replace(spec, '0.1 101\n', '')  # the row readiness just saw is gone before qualification reads it
            removed.set()
        return count

    monkeypatch.setattr(metadata, 'spec_rows_ready', ready, raising=False)
    executor, identity = _run(tmp_path, raw, 5201)
    events = []
    try:
        _wait(executor, events, _reduced(1), 20.)
        _row(spec, 1, 1)
        _image(raw, 1, 1)
        _wait(executor, events,
              lambda values: removed.is_set() or 2 in _labels(values, SCAN1) or _terminal(values), 20.)
        _settle(executor, events, 1.)  # ten polls without the row
        early = list(_labels(events, SCAN1))
        _row(spec, 1, 1)  # valid bytes arrive again
        _wait(executor, events, _reduced(2), 20.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert removed.is_set(), 'readiness saw the row before it was removed'
    assert early == [1], 'the frame waited while its row was gone'
    assert _kinds(terminal) == ['STOPPED'] and not _said(events, 'stopped growing'), terminal
    assert _output(tmp_path) == ((1, 2), (100.0, 101.0))


def test_r2_a_complete_line_that_is_not_a_row_fails_the_run_naming_it(tmp_path):
    raw, spec = _acquisition(tmp_path)
    executor, identity = _run(tmp_path, raw, 5202)
    events = []
    try:
        _wait(executor, events, _reduced(1), 20.)
        with spec.open('a') as stream:
            stream.write('0.1\n')  # point 2's row, one number short, newline-terminated
        _image(raw, 1, 1)
        _wait(executor, events, lambda values: 2 in _labels(values, SCAN1) or _terminal(values), 20.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert _kinds(terminal) == ['FAILED'], terminal
    detail = terminal[0][1]
    assert 'calibrant' in detail and 'scan 1' in detail and 'row 2' in detail, terminal
    assert _output(tmp_path) == ((1,), (100.0,))


def test_r3_a_row_rewritten_before_the_reduction_read_is_read_again_not_mixed(tmp_path, monkeypatch):
    raw, spec = _acquisition(tmp_path)
    real = image_source.read_image_metadata_observed
    rewritten = threading.Event()

    def read(path, *args, **kwargs):
        if not rewritten.is_set() and _worker() and Path(path).name == f'{SCAN1}_0001.tif':
            _replace(spec, '0.1 101\n', '0.1 109\n')  # after admission, inside the reduction-time read
            rewritten.set()
        return real(path, *args, **kwargs)

    monkeypatch.setattr(image_source, 'read_image_metadata_observed', read)
    executor, identity = _run(tmp_path, raw, 5203)
    events = []
    try:
        _wait(executor, events, _reduced(1), 20.)
        _row(spec, 1, 1)
        _image(raw, 1, 1)
        _wait(executor, events, _reduced(2), 20.)
        _settle(executor, events, 1.)
    finally:
        terminal = _stop(executor, identity, events)
    assert rewritten.is_set(), 'the row changed inside the reduction-time read'
    assert _kinds(terminal) == ['STOPPED'] and not _said(events, 'stopped growing'), terminal
    assert _output(tmp_path) == ((1, 2), (100.0, 109.0))


def _waiting_third_point(tmp_path, request_value):
    """Points 1 and 2 published; point 3's image has landed and waits for its row."""
    raw, spec = _acquisition(tmp_path, points=2)
    executor, identity = _run(tmp_path, raw, request_value)
    events = []
    _wait(executor, events, _reduced(2), 20.)
    _settle(executor, events)
    _image(raw, 1, 2)
    _settle(executor, events)
    return raw, spec, executor, identity, events


def test_r4_removing_the_spec_file_while_a_frame_waits_stops_that_output(tmp_path):
    raw, spec, executor, identity, events = _waiting_third_point(tmp_path, 5204)
    try:
        spec.unlink()
        _wait(executor, events, _stopped_growing, 20.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert _kinds(terminal) == ['STOPPED'], terminal
    assert f'{SCAN1}_int1d.nexus stopped growing' in terminal[0][1] and 'calibrant' in terminal[0][1], terminal
    assert _output(tmp_path) == ((1, 2), (100.0, 101.0))


@pytest.mark.parametrize('old, new', [('0.0 100\n', '0.0 105\n'), ('#O0 th  chi\n', '#O0 tth chi\n')],
                         ids=['covered-row', 'header'])
def test_r5_a_covered_row_or_header_rewritten_while_a_frame_waits_stops_that_output(tmp_path, old, new):
    raw, spec, executor, identity, events = _waiting_third_point(tmp_path, 5205)
    try:
        _replace(spec, old, new)
        _wait(executor, events, _stopped_growing, 20.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert _kinds(terminal) == ['STOPPED'], terminal
    assert f'{SCAN1}_int1d.nexus stopped growing' in terminal[0][1], terminal
    assert _output(tmp_path) == ((1, 2), (100.0, 101.0))


def test_r6_a_raw_rewrite_with_its_row_rewritten_while_a_frame_waits_fails_on_raw_history(tmp_path, monkeypatch):
    armed, inside, released = threading.Event(), threading.Event(), threading.Event()
    real_observe = DirectoryIndexSession.observe

    def observe(self, *, refresh=True):
        if armed.is_set() and not inside.is_set() and _worker():
            inside.set()
            released.wait(30.)
        return real_observe(self, refresh=refresh)

    monkeypatch.setattr(DirectoryIndexSession, 'observe', observe)
    raw, spec, executor, identity, events = _waiting_third_point(tmp_path, 5206)
    try:
        armed.set()
        assert inside.wait(30.), 'the live loop is held before its next observation'
        _image(raw, 1, 0)  # point 1's image replaced
        _replace(spec, '0.0 100\n', '0.0 105\n')  # and its row rewritten
        released.set()
        _wait(executor, events, _terminal, 20.)
    finally:
        released.set()
        terminal = _stop(executor, identity, events)
    assert _kinds(terminal) == ['FAILED'], terminal
    assert 'image-member history' in terminal[0][1], terminal


def _hold_inside_validation(monkeypatch, paths):
    """Hold the Run inside a SPEC parse made by live validation, only within an attempt for ``paths`` images."""
    armed, inside, released = threading.Event(), threading.Event(), threading.Event()
    state = threading.local()
    real_materialize = run_executor.materialize_live_directory_group
    real_validate = output_preflight.validate_source_execution_graph
    real_spec_file = specfile.SpecFile

    def materialize(*args, **kwargs):
        state.selected = armed.is_set() and len(args[3].physical_paths) == paths
        try:
            return real_materialize(*args, **kwargs)
        finally:
            state.selected = False

    def validate(*args, **kwargs):
        state.validating = getattr(state, 'selected', False)
        try:
            return real_validate(*args, **kwargs)
        finally:
            state.validating = False

    def spec_file(*args, **kwargs):
        if getattr(state, 'validating', False) and not inside.is_set() and _worker():
            inside.set()
            released.wait(30.)
        return real_spec_file(*args, **kwargs)

    real_qualify = output_preflight.qualify_source_execution_graph

    def qualify(*args, **kwargs):  # a Live admission's own reads are its JIT validation of a new frame
        state.validating = getattr(state, 'selected', False)
        try:
            return real_qualify(*args, **kwargs)
        finally:
            state.validating = False

    monkeypatch.setattr(output_preflight, 'qualify_source_execution_graph', qualify)
    monkeypatch.setattr(run_executor, 'materialize_live_directory_group', materialize)
    monkeypatch.setattr(output_preflight, 'validate_source_execution_graph', validate)
    monkeypatch.setattr(specfile, 'SpecFile', spec_file)
    return armed, inside, released


@pytest.mark.parametrize('ready', [False, True], ids=['currency-check-while-a-frame-waits', 'jit-with-a-new-frame'])
def test_r7_r8_stop_inside_a_live_spec_validation_read_ends_stopped(tmp_path, monkeypatch, ready):
    armed, inside, released = _hold_inside_validation(monkeypatch, paths=3)
    raw, spec = _acquisition(tmp_path, points=2)
    executor, identity = _run(tmp_path, raw, 5207 + ready)
    events = []
    try:
        _wait(executor, events, _reduced(2), 20.)
        _settle(executor, events)
        armed.set()
        if ready:
            _row(spec, 1, 2)
        _image(raw, 1, 2)
        # A Live Run validates a published prefix strictly only after a change; a registration is one.
        _reregister_image_adapter(monkeypatch, raw / f'{SCAN1}_0001.tif')
        assert inside.wait(30.), 'the Run is inside a SPEC read of its live validation'
        executor.stop(identity)
        released.set()
        _wait(executor, events, _terminal, 20.)
    finally:
        released.set()
        terminal = _stop(executor, identity, events)
    assert _kinds(terminal) == ['STOPPED'], terminal


def test_r9_a_fast_acquisition_publishes_frames_while_rows_are_still_arriving(tmp_path):
    raw, spec = _acquisition(tmp_path)
    executor, identity = _run(tmp_path, raw, 5209)
    events = []
    try:
        _wait(executor, events, _reduced(1), 20.)
        for k in range(1, 11):  # a point every 30 ms: its row, then its image
            _row(spec, 1, k)
            _image(raw, 1, k)
            time.sleep(.03)
        # The producer holds at point 11 until a published prefix beyond point 1 is observed.
        _wait(executor, events, lambda values: len(set(_labels(values, SCAN1))) >= 6 or _terminal(values), 30.)
        held = sorted(set(_labels(events, SCAN1)))
        for k in range(11, 21):
            _row(spec, 1, k)
            _image(raw, 1, k)
            time.sleep(.03)
        _wait(executor, events, _reduced(21), 30.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert held[:6] == [1, 2, 3, 4, 5, 6] and len(held) <= 11, held
    assert _kinds(terminal) == ['STOPPED'] and not _said(events, 'stopped growing'), terminal
    assert _output(tmp_path) == (tuple(range(1, 22)), tuple(float(100 + k) for k in range(21)))


def _intent(tmp_path, raw, *, live, mode):
    poni = tmp_path / 'cal.poni'
    write_poni(poni)
    return RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=('.tif',), metadata_format='spec'), poni_file=str(poni),
        project_root=str(raw), save_path=str(tmp_path / 'processed'), output_mode=mode,
        processing_mode='Int 1D', live_mode=live, max_cores=1, bai_1d_args={'npt': 8, 'method': 'numpy'})


def _append(tmp_path, raw, request_value):
    executor, identity, events = _run_to_terminal(_intent(tmp_path, raw, live=False, mode='Append'),
                                                  request_value=request_value)
    executor.close(identity)
    return [(event.kind.name, event.detail) for event in events if event.kind in _TERMINAL]


def _reintegrate(tmp_path, raw, artifact):
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xrd_tools.io import open_scan
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import ReintegrateSuccessorPlan, run_reintegrate_successor
    labels = tuple(int(value) for value in open_scan(artifact).frame_indices)
    plan = ReintegrateSuccessorPlan.from_artifact(
        artifact, entry='entry', dimension='1d', source_root=str(raw),
        expected_target_snapshot=capture_target_snapshot(artifact), expected_labels=labels,
        preparation=ScatteringWorkspace._reintegrate_preparation(_intent(tmp_path, raw, live=True, mode='Overwrite'),
                                                                 '1d'))
    result = run_reintegrate_successor(plan)
    assert result.disposition == 'COMMITTED', result
    return Path(result.output_artifact)


def test_r10_a_saved_live_output_replays_unchanged_inputs_and_refuses_after_growth(tmp_path):
    from xrd_tools.io import open_scan
    raw, spec = _acquisition(tmp_path, points=3)
    executor, identity = _run(tmp_path, raw, 5210)
    events = []
    try:
        _wait(executor, events, _reduced(3), 20.)
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert _kinds(terminal) == ['STOPPED'], terminal
    artifact = tmp_path / 'processed' / f'{SCAN1}_int1d.nexus'
    assert _output(tmp_path) == ((1, 2, 3), (100.0, 101.0, 102.0))
    saved = open_scan(artifact)
    replay = open_scan(_reintegrate(tmp_path, raw, artifact))
    for label in saved.frame_indices:  # unchanged inputs: the replay is bit-exact
        np.testing.assert_array_equal(saved.get_1d(label).intensity, replay.get_1d(label).intensity)
    assert _kinds(_append(tmp_path, raw, 5211)) == ['FINISHED']  # and Append accepts the unchanged inputs
    assert _output(tmp_path) == ((1, 2, 3), (100.0, 101.0, 102.0))
    _row(spec, 1, 3)  # the acquisition goes on: the SPEC file grows past the saved rows
    before = artifact.read_bytes()
    with pytest.raises(ValueError, match='REVISION'):
        _reintegrate(tmp_path, raw, artifact)
    refused = _append(tmp_path, raw, 5212)
    assert _kinds(refused) == ['FAILED'] and 'same-extent source identity changed' in refused[0][1], refused
    assert artifact.read_bytes() == before
