"""Terminal-review discriminators for published metadata removal and mixed raw drift."""
import threading

import pytest
import xrd_tools.io.metadata as metadata

from tests.xdart.scattering.test_live_qualify_refusal_run import (
    SIDECARS, _frame, _labels, _output, _published, _run, _said, _settle,
    _stop, _terminal, _wait, _worker,
)
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xdart.gui.tabs.scattering.adapters import run_executor
from xdart.gui.tabs.scattering import output_preflight


def test_a_removed_published_explicit_sidecar_blocks_its_target_and_reports_it(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    executor, identity = _run(tmp_path, raw, 'txt', 3301)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        digest, published = _published(executor, identity)
        (raw / 'scan_0001.txt').unlink()
        _frame(raw, 'scan', 2, 'txt', 2.0)
        _frame(raw, 'other', 1, 'txt', 7.0)
        _wait(executor, events, lambda values: 1 in _labels(values, 'other') or _terminal(values))
        _settle(executor, events, 1.)
    finally:
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    assert 'scan_0001.txt' in terminal[0][1] and '1 skipped' in terminal[0][1], terminal
    labels, i0, epoch = _output(tmp_path)
    assert (labels, i0) == ((1,), (1.0,))
    assert (epoch['digest'], epoch['extent']) == (digest, published)
    labels, i0, _epoch = _output(tmp_path, 'other')
    assert (labels, i0) == ((1,), (7.0,))


@pytest.mark.parametrize('next_ready', (True, False))
def test_raw_member_rewrite_with_metadata_drift_keeps_the_raw_validator_terminal(tmp_path, monkeypatch, next_ready):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    armed, inside, released = threading.Event(), threading.Event(), threading.Event()
    real_observe = DirectoryIndexSession.observe

    def observe(self, *, refresh=True):
        if armed.is_set() and not inside.is_set() and _worker():
            inside.set()
            released.wait(30.)
        return real_observe(self, refresh=refresh)

    monkeypatch.setattr(DirectoryIndexSession, 'observe', observe)
    executor, identity = _run(tmp_path, raw, 'txt', 3302)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        _published(executor, identity)
        armed.set()
        assert inside.wait(30.), 'the live loop is held before its next observation'
        _frame(raw, 'scan', 1, 'txt', 9.0)
        _frame(raw, 'scan', 2, 'txt', 2.0, sidecar=next_ready)
        released.set()
        if next_ready:
            _wait(executor, events, lambda values: _said(values, 'stopped growing') or _terminal(values))
        else:
            _settle(executor, events, 1.)
    finally:
        released.set()
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['FAILED'], terminal
    assert 'image-member history' in terminal[0][1], terminal


def test_stop_inside_a_live_admission_metadata_read_ends_stopped(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    armed, inside, released = threading.Event(), threading.Event(), threading.Event()
    state = threading.local()
    real_materialize = run_executor.materialize_live_directory_group
    real_read = metadata.read_txt_metadata
    real_validate = output_preflight.validate_source_execution_graph

    def materialize(*args, **kwargs):
        state.selected = armed.is_set() and len(args[3].physical_paths) == 2
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

    def read(path, *args, **kwargs):
        if getattr(state, 'validating', False) and not inside.is_set() and _worker():
            inside.set()
            released.wait(30.)
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(run_executor, 'materialize_live_directory_group', materialize)
    monkeypatch.setattr(output_preflight, 'validate_source_execution_graph', validate)
    monkeypatch.setattr(metadata, 'read_txt_metadata', read)
    executor, identity = _run(tmp_path, raw, 'txt', 3303)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _terminal(values))
        _published(executor, identity)
        armed.set()
        _frame(raw, 'scan', 2, 'txt', 2.0, sidecar=False)
        assert inside.wait(30.), 'admission is inside a real metadata read'
        executor.stop(identity)
        released.set()
        _wait(executor, events, _terminal)
    finally:
        released.set()
        terminal = _stop(executor, identity, events)
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
