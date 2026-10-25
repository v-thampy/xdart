"""A share hiccup while a Live frame is qualified is retried at the next poll, not a block on its scan (Claude review of
03af89cc, C2: qualification captured each member's file state with a bare ``OSError`` that no loop retries, so one
``PermissionError``/``WinError 64`` on one stat left the scan stopped for the whole Run).

A real Live Run over a real TIFF + .txt scan; the one double is a ``PermissionError`` raised once, from the first stat
of the image that qualification makes."""
import sys
import threading
from pathlib import Path

import xrd_tools.sources.registry  # noqa: F401  # register built-in readers
from tests.xdart.scattering.test_live_qualify_refusal_run import (
    _frame, _labels, _output, _run, _said, _settle, _stop, _terminal, _wait,
)
from xdart.gui.tabs.scattering.adapters import run_executor


def _worker():
    return threading.current_thread() is not threading.main_thread()


def _qualifying():
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_name == '_qualify_tiff':
            return True
        frame = frame.f_back
    return False


def test_a_transient_oserror_while_a_live_frame_is_qualified_is_retried_at_the_next_poll(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    _frame(raw, 'scan', 1, 'txt', 1.0)
    fired, real_stat = [], Path.stat

    def stat(self, *args, **kwargs):
        if not fired and self.name == 'scan_0001.tif' and _worker() and _qualifying():
            fired.append(str(self))
            raise PermissionError(13, 'permission denied (injected)', str(self))
        return real_stat(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', stat)
    attempts, real_attempt = [], run_executor.materialize_live_directory_group

    def attempt(*args, **kwargs):  # a spy: the attempt is the real one
        value = real_attempt(*args, **kwargs)
        attempts.append((value.state.name, value.revision_changed))
        return value
    monkeypatch.setattr(run_executor, 'materialize_live_directory_group', attempt)
    executor, identity = _run(tmp_path, raw, 'txt', 3301)
    events = []
    try:
        _wait(executor, events, lambda values: 1 in _labels(values) or _said(values, 'stopped growing')
              or _terminal(values))
        run = executor._exact_run(identity)
        blocked = dict(run.live_blocked)
        _frame(raw, 'scan', 2, 'txt', 2.0)  # the scan still grows
        _wait(executor, events, lambda values: 2 in _labels(values) or _said(values, 'stopped growing')
              or _terminal(values))
        _settle(executor, events)
    finally:
        terminal = _stop(executor, identity, events)
    assert fired == [str(raw / 'scan_0001.tif')], 'the injected error landed inside the frame\'s qualification'
    assert ('IN_PROGRESS', True) in attempts, attempts  # a revision that changed while it was read: again next poll
    assert blocked == {} and not _said(events, 'stopped growing'), blocked
    assert [kind for kind, _detail in terminal] == ['STOPPED'], terminal
    labels, i0, _epoch = _output(tmp_path)
    assert (labels, i0) == ((1, 2), (1.0, 2.0))

