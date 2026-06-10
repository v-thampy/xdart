"""Pause/Resume (Phase B) — the worker-thread freeze primitive + the GUI
freeze-guard lift.

Pause is a THIRD command state between the run state ('start') and 'stop': the
worker's processing loops call ``_wait_if_paused`` at their top, which on entry
quiesces the writer at a frame boundary (``_enter_pause``: streaming ->
session.drain() + sink flush; serial -> flush the unsaved tail) and emits
``sigPaused`` so the GUI lifts the disk-read freeze guard for browsing, then
spins until ``command`` leaves 'pause' ('start' = resume, 'stop' = terminal).

These are headless unit tests of the pure logic (offscreen Qt not even needed
for most); the live drain/flush/browse loop is verified in the GUI.
"""
import threading
import time
from types import SimpleNamespace, MethodType

import pytest

import xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread as itmod

imageThread = itmod.imageThread


# ── _enter_pause: drain/flush ordering + signal ─────────────────────────────

def test_enter_pause_streaming_drains_then_flushes_then_signals():
    """Streaming pause: drain() (non-terminal) MUST run before the sink flush,
    and sigPaused fires only AFTER both (so the writer is provably idle before
    the GUI reads disk)."""
    calls = []
    session = SimpleNamespace(
        drain=lambda timeout=None: (calls.append('drain'), True)[1])
    sink = SimpleNamespace(_flush=lambda *, force=False: calls.append(('flush', force)))
    emitted = []
    w = SimpleNamespace(
        _streaming_session=session, _streaming_sink=sink,
        _active_scan=None, xye_only=False, _frames_since_save=0,
        sigPaused=SimpleNamespace(emit=lambda: emitted.append('paused')),
    )
    w._enter_pause = MethodType(imageThread._enter_pause, w)

    w._enter_pause()

    assert calls == ['drain', ('flush', True)]      # drain strictly before flush
    assert emitted == ['paused']                    # signalled after drain+flush


def test_enter_pause_serial_tail_wins_over_open_streaming_session(monkeypatch):
    """Adversarial-review fix: in the true-live WATCH loop the Phase-2 streaming
    session is still open but DORMANT, while the serial path accumulates the
    watch tail (_frames_since_save).  _enter_pause must route on the serial tail
    FIRST -- flush the .nxs + reset the counter -- not take the streaming branch
    (which would leave _frames_since_save leaked).  It still drains the dormant
    session first (a no-op) to keep the writer idle."""
    monkeypatch.setattr(itmod, '_get_h5pool',
                        lambda: SimpleNamespace(pause=lambda f: None,
                                                resume=lambda f: None))
    order = []
    session = SimpleNamespace(
        drain=lambda timeout=None: order.append('drain') or True)
    sink = SimpleNamespace(_flush=lambda *, force=False: order.append('sink_flush'))
    scan = SimpleNamespace(data_file='x.nxs',
                           _save_to_nexus=lambda: order.append('save'))
    w = SimpleNamespace(
        _streaming_session=session, _streaming_sink=sink,   # open but dormant
        _active_scan=scan, xye_only=False, _frames_since_save=4,
        file_lock=threading.RLock(),
        _flush_xye_buffer=lambda s: order.append('xye'),
        sigPaused=SimpleNamespace(emit=lambda: order.append('paused')),
    )
    w._enter_pause = MethodType(imageThread._enter_pause, w)

    w._enter_pause()

    # Serial branch ran (drain the dormant session, then SERIAL save+xye), and
    # the sink streaming flush did NOT (would re-route the watch tail wrongly).
    assert 'save' in order and 'xye' in order
    assert 'sink_flush' not in order
    assert order.index('drain') < order.index('save')   # idle the writer first
    assert w._frames_since_save == 0                     # counter reset, not leaked
    assert order[-1] == 'paused'


def test_enter_pause_serial_flushes_unsaved_tail(monkeypatch):
    """Serial true-live pause: flush the unsaved _frames_since_save tail to .nxs
    (so the file is at a frame boundary) then signal."""
    monkeypatch.setattr(itmod, '_get_h5pool',
                        lambda: SimpleNamespace(pause=lambda f: None,
                                                resume=lambda f: None))
    saved, flushed, emitted = [], [], []
    scan = SimpleNamespace(data_file='x.nxs',
                           _save_to_nexus=lambda: saved.append('save'))
    w = SimpleNamespace(
        _streaming_session=None, _streaming_sink=None,
        _active_scan=scan, xye_only=False, _frames_since_save=3,
        file_lock=threading.RLock(),
        _flush_xye_buffer=lambda s: flushed.append(s),
        sigPaused=SimpleNamespace(emit=lambda: emitted.append('paused')),
    )
    w._enter_pause = MethodType(imageThread._enter_pause, w)

    w._enter_pause()

    assert saved == ['save'] and flushed == [scan]
    assert w._frames_since_save == 0                # tail flushed -> counter reset
    assert emitted == ['paused']


def test_enter_pause_serial_nothing_to_flush_still_signals(monkeypatch):
    """Pause before any frame / right after a save (_frames_since_save==0):
    nothing to flush, but sigPaused still fires so the guard lifts."""
    monkeypatch.setattr(itmod, '_get_h5pool',
                        lambda: SimpleNamespace(pause=lambda f: None,
                                                resume=lambda f: None))
    emitted = []
    w = SimpleNamespace(
        _streaming_session=None, _streaming_sink=None,
        _active_scan=None, xye_only=True, _frames_since_save=0,
        sigPaused=SimpleNamespace(emit=lambda: emitted.append('paused')),
    )
    w._enter_pause = MethodType(imageThread._enter_pause, w)
    w._enter_pause()
    assert emitted == ['paused']


def test_enter_pause_signals_even_if_drain_raises():
    """A drain/flush failure must not strand the run: log + still emit sigPaused
    (we've stopped submitting, so the writer is idle and reads are safe)."""
    def _boom(timeout=None):
        raise RuntimeError("writer exploded")
    emitted = []
    w = SimpleNamespace(
        _streaming_session=SimpleNamespace(drain=_boom),
        _streaming_sink=SimpleNamespace(_flush=lambda *, force=False: None),
        _active_scan=None, xye_only=False, _frames_since_save=0,
        sigPaused=SimpleNamespace(emit=lambda: emitted.append('paused')),
    )
    w._enter_pause = MethodType(imageThread._enter_pause, w)
    w._enter_pause()
    assert emitted == ['paused']


# ── _wait_if_paused: block until resume/stop, no-op otherwise ────────────────

def test_wait_if_paused_noop_when_not_paused():
    entered = []
    w = SimpleNamespace(command='start')
    w._enter_pause = lambda: entered.append('enter')
    w._wait_if_paused = MethodType(imageThread._wait_if_paused, w)
    w._wait_if_paused()
    assert entered == []          # never enters pause when command != 'pause'


def test_wait_if_paused_blocks_until_resume_and_enters_once():
    entered = []
    w = SimpleNamespace(command='pause')
    w._enter_pause = lambda: entered.append('enter')
    w._wait_if_paused = MethodType(imageThread._wait_if_paused, w)

    done = []
    t = threading.Thread(target=lambda: (w._wait_if_paused(), done.append(True)))
    t.start()
    time.sleep(0.15)
    assert not done               # still blocked while command == 'pause'
    assert entered == ['enter']   # _enter_pause ran exactly once on entry
    w.command = 'start'           # resume
    t.join(timeout=2)
    assert done == [True]


def test_wait_if_paused_exits_on_stop():
    """Shutdown-safe: setting command='stop' (close/Stop) breaks the pause wait
    just like resume, so the loop returns and run() can finalize."""
    w = SimpleNamespace(command='pause')
    w._enter_pause = lambda: None
    w._wait_if_paused = MethodType(imageThread._wait_if_paused, w)
    done = []
    t = threading.Thread(target=lambda: (w._wait_if_paused(), done.append(True)))
    t.start()
    time.sleep(0.1)
    assert not done
    w.command = 'stop'
    t.join(timeout=2)
    assert done == [True]


# ── GUI freeze-guard lift / re-engage (static_scan_widget) ──────────────────

def test_on_run_paused_lifts_guard_but_keeps_run_active():
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    calls = []
    w = SimpleNamespace(
        _run_active=True,
        displayframe=SimpleNamespace(
            set_processing_active=lambda v: calls.append(('proc', v))),
        h5viewer=SimpleNamespace(
            set_run_writing=lambda v: calls.append(('write', v))),
    )
    staticWidget._on_run_paused(w)
    assert ('proc', False) in calls and ('write', False) in calls   # guard LIFTED
    assert w._run_active is True            # run still active, just frozen


def test_on_run_resuming_reengages_guard_before_resume():
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    calls = []
    w = SimpleNamespace(
        _run_active=True,
        displayframe=SimpleNamespace(
            set_processing_active=lambda v: calls.append(('proc', v))),
        h5viewer=SimpleNamespace(
            set_run_writing=lambda v: calls.append(('write', v))),
    )
    staticWidget._on_run_resuming(w)
    assert ('write', True) in calls and ('proc', True) in calls     # guard RE-ENGAGED


def test_guard_lift_noop_when_not_in_run():
    """Defensive: a stray pause/resume signal when no run is active does nothing."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    calls = []
    w = SimpleNamespace(
        _run_active=False,
        displayframe=SimpleNamespace(
            set_processing_active=lambda v: calls.append(v)),
        h5viewer=SimpleNamespace(set_run_writing=lambda v: calls.append(v)),
    )
    staticWidget._on_run_paused(w)
    staticWidget._on_run_resuming(w)
    assert calls == []


def test_enter_pause_drain_timeout_skips_flush_but_signals():
    """RS-1: a drain() timeout means the writer is provably NOT idle — a
    save/flush from the wrangler thread would violate the single-writer
    invariant, and resetting the save counter without saving would break
    persist-before-evict.  Pause still proceeds (sigPaused fires) without
    touching the file; the tail flushes on resume/finish."""
    calls, emitted = [], []
    session = SimpleNamespace(
        drain=lambda timeout=None: (calls.append('drain'), False)[1])
    sink = SimpleNamespace(
        _flush=lambda *, force=False: calls.append(('flush', force)))
    scan = SimpleNamespace(data_file='x.nxs',
                           _save_to_nexus=lambda: calls.append('save'))
    w = SimpleNamespace(
        _streaming_session=session, _streaming_sink=sink,
        _active_scan=scan, xye_only=False, _frames_since_save=4,
        file_lock=threading.RLock(),
        _flush_xye_buffer=lambda s: calls.append('xye'),
        sigPaused=SimpleNamespace(emit=lambda: emitted.append('paused')),
    )
    w._enter_pause = MethodType(imageThread._enter_pause, w)

    w._enter_pause()

    assert calls == ['drain']            # no save, no sink flush, no xye
    assert w._frames_since_save == 4     # counter preserved for the next save
    assert emitted == ['paused']         # the freeze guard still lifts


def _rs2_wrangler(command, thread_command):
    """Light holder driving the real pause()/_on_resume() command logic."""
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler import imageWrangler
    calls = []
    w = SimpleNamespace(
        command=command,
        thread=SimpleNamespace(command=thread_command),
        _set_action_button=lambda phase: calls.append(('button', phase)),
        sigResuming=SimpleNamespace(emit=lambda: calls.append('resuming')),
    )
    w.pause = MethodType(imageWrangler.pause, w)
    w._on_resume = MethodType(imageWrangler._on_resume, w)
    return w, calls


def test_pause_does_not_overwrite_worker_stop():
    """RS-2: the worker self-stops by writing thread.command='stop' directly
    (write-failure stop, GI freeze abort).  A Pause click landing just after
    must NOT overwrite it — that silently revived a run that had declared
    itself dead."""
    w, calls = _rs2_wrangler(command='start', thread_command='stop')

    w.pause()

    assert w.thread.command == 'stop'    # stop preserved
    assert w.command == 'start'          # GUI mirror untouched (no 'pause')
    assert calls == []                   # no 'pausing' morph for a dead run


def test_resume_does_not_revive_stop():
    """RS-2: a stop that landed while paused must stay a stop — _on_resume
    must not re-engage the freeze guard or flip the command back to start."""
    w, calls = _rs2_wrangler(command='start', thread_command='stop')

    w._on_resume()

    assert w.thread.command == 'stop'
    assert 'resuming' not in calls       # guard NOT re-engaged for a dead run
    assert calls == []


def test_pause_and_resume_still_work_when_running():
    """RS-2 control: the normal path is unchanged."""
    w, calls = _rs2_wrangler(command='start', thread_command='start')
    w.pause()
    assert w.thread.command == 'pause' and w.command == 'pause'
    assert ('button', 'pausing') in calls

    w2, calls2 = _rs2_wrangler(command='pause', thread_command='pause')
    w2._on_resume()
    assert w2.thread.command == 'start' and w2.command == 'start'
    assert calls2[0] == 'resuming'       # guard re-engaged FIRST
    assert ('button', 'running') in calls2
