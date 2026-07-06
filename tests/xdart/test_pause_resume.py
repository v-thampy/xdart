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
import xdart.gui.tabs.static_scan.wranglers.wrangler_widget as wwmod

imageThread = itmod.imageThread


def _bind_serial_tail(w):
    """Bind the DRYed serial-save methods (flush_serial_tail + _h5pool_bracket)
    onto a stand-in so _enter_pause / _dispatch_batch_serial reach them.  The
    caller supplies the ``_save_due`` gate (a stub) for the desired outcome."""
    from types import MethodType as _MT
    w.flush_serial_tail = _MT(imageThread.flush_serial_tail, w)
    w._h5pool_bracket = _MT(imageThread._h5pool_bracket, w)


# ── _enter_pause: drain/flush ordering + signal ─────────────────────────────

def test_enter_pause_streaming_drains_then_flushes_then_signals():
    """Streaming pause: drain() (non-terminal) MUST run before the sink flush,
    and sigPaused fires only AFTER both (so the writer is provably idle before
    the GUI reads disk)."""
    calls = []
    session = SimpleNamespace(
        drain=lambda timeout=None: (calls.append('drain'), True)[1])
    sink = SimpleNamespace(flush=lambda *, force=False: calls.append(('flush', force)))
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
    sink = SimpleNamespace(flush=lambda *, force=False: order.append('sink_flush'))
    scan = SimpleNamespace(data_file='x.nxs',
                           _save_to_nexus=lambda: order.append('save'))
    w = SimpleNamespace(
        _streaming_session=session, _streaming_sink=sink,   # open but dormant
        _active_scan=scan, xye_only=False, _frames_since_save=4,
        file_lock=threading.RLock(),
        _save_due=lambda scan, force=False: True,
        _flush_xye_buffer=lambda s: order.append('xye'),
        sigPaused=SimpleNamespace(emit=lambda: order.append('paused')),
    )
    w._enter_pause = MethodType(imageThread._enter_pause, w)
    _bind_serial_tail(w)

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
        _save_due=lambda scan, force=False: True,
        _flush_xye_buffer=lambda s: flushed.append(s),
        sigPaused=SimpleNamespace(emit=lambda: emitted.append('paused')),
    )
    w._enter_pause = MethodType(imageThread._enter_pause, w)
    _bind_serial_tail(w)

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
        _streaming_sink=SimpleNamespace(flush=lambda *, force=False: None),
        _active_scan=None, xye_only=False, _frames_since_save=0,
        sigPaused=SimpleNamespace(emit=lambda: emitted.append('paused')),
    )
    w._enter_pause = MethodType(imageThread._enter_pause, w)
    w._enter_pause()
    assert emitted == ['paused']


# ── D₂: the DRYed serial-save helpers (_h5pool_bracket / flush_serial_tail) ──

def test_h5pool_bracket_resumes_even_when_body_raises(monkeypatch):
    """The symmetric bracket resumes the h5 pool even if the wrapped body raises
    — a save failure must never strand the pool paused (which would deadlock
    every later write to that file)."""
    events = []
    monkeypatch.setattr(itmod, '_get_h5pool',
                        lambda: SimpleNamespace(
                            pause=lambda f: events.append(('pause', f)),
                            resume=lambda f: events.append(('resume', f))))
    scan = SimpleNamespace(data_file='x.nxs')
    w = SimpleNamespace()
    w._h5pool_bracket = MethodType(imageThread._h5pool_bracket, w)

    with pytest.raises(RuntimeError, match="boom"):
        with w._h5pool_bracket(scan):
            raise RuntimeError("boom")

    assert events == [('pause', 'x.nxs'), ('resume', 'x.nxs')]   # balanced


def test_flush_serial_tail_locks_before_pausing_h5pool(monkeypatch):
    """Writer must own file_lock before closing pooled read handles.

    A load worker borrows from h5pool while holding this same lock.  Pausing the
    pool before the lock can close a read handle that is still active, leaving
    HDF5 to reject the following r+ writer open as "already open read-only".
    """
    events = []

    class _TrackingLock:
        held = False

        def __enter__(self):
            events.append("lock-enter")
            self.held = True
            return self

        def __exit__(self, *_exc):
            events.append("lock-exit")
            self.held = False
            return False

    lock = _TrackingLock()

    class _Pool:
        def pause(self, path):
            events.append(("pause", path, lock.held))
            assert lock.held is True

        def resume(self, path):
            events.append(("resume", path, lock.held))
            assert lock.held is True

    monkeypatch.setattr(itmod, '_get_h5pool', lambda: _Pool())
    scan = SimpleNamespace(
        data_file='x.nxs',
        _save_to_nexus=lambda: events.append(("save", lock.held)),
    )
    w = SimpleNamespace(
        xye_only=False, _frames_since_save=1, file_lock=lock,
        _save_due=lambda scan, force=False: True,
        _flush_xye_buffer=lambda s: events.append(("xye", lock.held)),
    )
    _bind_serial_tail(w)

    assert w.flush_serial_tail(scan, force=True) is True
    assert events == [
        "lock-enter",
        ("pause", "x.nxs", True),
        ("save", True),
        ("resume", "x.nxs", True),
        "lock-exit",
        ("xye", False),
    ]


def test_base_save_to_disk_locks_before_pausing_h5pool(monkeypatch):
    events = []

    class _TrackingLock:
        held = False

        def __enter__(self):
            events.append("lock-enter")
            self.held = True
            return self

        def __exit__(self, *_exc):
            events.append("lock-exit")
            self.held = False
            return False

    lock = _TrackingLock()

    class _Pool:
        def pause(self, path):
            events.append(("pause", path, lock.held))
            assert lock.held is True

        def resume(self, path):
            events.append(("resume", path, lock.held))
            assert lock.held is True

    monkeypatch.setattr(wwmod, '_get_h5pool', lambda: _Pool())
    scan = SimpleNamespace(
        data_file='x.nxs',
        _save_to_nexus=lambda: events.append(("save", lock.held)),
    )
    w = SimpleNamespace(xye_only=False, file_lock=lock)
    w._save_to_disk = MethodType(wwmod.wranglerThread._save_to_disk, w)

    w._save_to_disk(scan)

    assert events == [
        "lock-enter",
        ("pause", "x.nxs", True),
        ("save", True),
        ("resume", "x.nxs", True),
        "lock-exit",
    ]


def test_nexus_final_save_locks_before_pausing_h5pool(monkeypatch):
    import xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread as nxmod
    from xdart.gui.tabs.static_scan.wranglers.nexus_wrangler_thread import (
        nexusThread)

    events = []

    class _TrackingLock:
        held = False

        def __enter__(self):
            events.append("lock-enter")
            self.held = True
            return self

        def __exit__(self, *_exc):
            events.append("lock-exit")
            self.held = False
            return False

    lock = _TrackingLock()

    class _Pool:
        def pause(self, path):
            events.append(("pause", path, lock.held))
            assert lock.held is True

        def resume(self, path):
            events.append(("resume", path, lock.held))
            assert lock.held is True

    monkeypatch.setattr(nxmod, '_get_h5pool', lambda: _Pool())
    scan = SimpleNamespace(
        data_file='x.nxs',
        default_geometry=lambda: events.append(("geometry", lock.held)),
        save_to_nexus=lambda replace=False, finalize=True: events.append(
            ("save", replace, finalize, lock.held)),
    )
    w = SimpleNamespace(xye_only=False, command='stop', file_lock=lock)
    w._final_save_to_nexus = MethodType(nexusThread._final_save_to_nexus, w)

    w._final_save_to_nexus(scan, 3)

    assert events == [
        "lock-enter",
        ("pause", "x.nxs", True),
        ("geometry", True),
        ("save", False, False, True),
        ("resume", "x.nxs", True),
        "lock-exit",
    ]


def test_flush_serial_tail_persists_before_resetting_counter(monkeypatch):
    """persist-before-evict: ``_save_to_nexus`` (which marks frames persisted)
    completes BEFORE ``_frames_since_save`` is reset — the save sees the
    pre-reset counter, so an unsaved frame can never be evicted out from under
    the writer."""
    monkeypatch.setattr(itmod, '_get_h5pool',
                        lambda: SimpleNamespace(pause=lambda f: None,
                                                resume=lambda f: None))
    saw_counter = []
    scan = SimpleNamespace(
        data_file='x.nxs',
        _save_to_nexus=lambda: saw_counter.append(w._frames_since_save))
    w = SimpleNamespace(
        xye_only=False, _frames_since_save=5, file_lock=threading.RLock(),
        _save_due=lambda scan, force=False: True,
        _flush_xye_buffer=lambda s: None,
    )
    _bind_serial_tail(w)

    assert w.flush_serial_tail(scan, force=True) is True
    assert saw_counter == [5]              # counter NOT yet reset when saving
    assert w._frames_since_save == 0       # reset only AFTER the persist

    # not-due (the _save_due gate says no) and scan-None are no-ops -> False.
    w._save_due = lambda scan, force=False: False
    assert w.flush_serial_tail(scan, force=True) is False
    assert w.flush_serial_tail(None, force=True) is False


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
    w._set_scan_integrated_reads_transient = MethodType(
        staticWidget._set_scan_integrated_reads_transient, w)  # no-op: host has no scan
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
    w._set_scan_integrated_reads_transient = MethodType(
        staticWidget._set_scan_integrated_reads_transient, w)  # no-op: host has no scan
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
        flush=lambda *, force=False: calls.append(('flush', force)))
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
        thread=SimpleNamespace(command=thread_command,
                               command_lock=threading.Lock()),
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


def test_serial_dispatch_drains_xye_buffer_without_nxs_save():
    """Int 1D (XYE) on the serial fallback (XDART_LIVE_EXECUTION=serial) has
    no .nxs save to ride on (_save_due is always False with xye_only), so the
    dispatch itself must drain the XYE buffer -- it previously never did,
    and the documented fallback path silently wrote ZERO output."""
    from types import MethodType, SimpleNamespace
    from xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread import (
        imageThread)

    flushed = []
    w = SimpleNamespace(
        xye_only=True,
        _frames_since_save=0,
        _wait_if_paused=lambda: None,
        _save_due=lambda scan, force=False: False,
        _flush_xye_buffer=lambda scan, **k: flushed.append(scan),
    )
    w._dispatch_batch_serial = MethodType(
        imageThread._dispatch_batch_serial, w)
    _bind_serial_tail(w)
    scan = object()
    w._dispatch_batch_serial(scan, [])
    assert flushed == [scan]

    # Non-XYE mode with no save due: no flush (unchanged behavior).
    flushed.clear()
    w.xye_only = False
    w._dispatch_batch_serial(scan, [])
    assert flushed == []


# ── 4c-1: _enter_pause / _wait_if_paused route through ScanSessionAdapter ────

class _SpyAdapter:
    """Records quiesce/flush/resume; the production pause path (4c-1)."""
    def __init__(self, drained=True):
        self._drained = drained
        self.calls = []
        self._paused = False

    def quiesce(self, timeout=None):
        self.calls.append(('quiesce', timeout))
        self._paused = self._drained
        return self._drained

    def flush(self):
        self.calls.append('flush')

    def resume(self):
        self.calls.append('resume')
        self._paused = False

    @property
    def is_paused(self):
        return self._paused


def test_enter_pause_streaming_routes_through_adapter():
    """With an adapter present (production streaming), _enter_pause quiesces
    via the adapter (session.pause: flag+drain) then flushes via the adapter,
    BEFORE sigPaused — the legacy bare session.drain/sink.flush are bypassed."""
    adapter = _SpyAdapter()
    emitted = []
    # legacy session/sink present too, but the adapter must win and they must
    # NOT be touched.
    session = SimpleNamespace(drain=lambda timeout=None: emitted.append('LEGACY_drain') or True)
    sink = SimpleNamespace(flush=lambda *, force=False: emitted.append('LEGACY_flush'))
    w = SimpleNamespace(
        _scan_session_adapter=adapter,
        _streaming_session=session, _streaming_sink=sink,
        _active_scan=None, xye_only=False, _frames_since_save=0,
        sigPaused=SimpleNamespace(emit=lambda: emitted.append('paused')),
    )
    w._enter_pause = MethodType(imageThread._enter_pause, w)
    w._enter_pause()

    assert adapter.calls == [('quiesce', 30.0), 'flush']   # quiesce before flush
    assert 'LEGACY_drain' not in emitted and 'LEGACY_flush' not in emitted
    assert emitted == ['paused']


def test_enter_pause_serial_tail_wins_with_adapter_present(monkeypatch):
    """The serial-tail routing survives the adapter: in live-watch the adapter
    (dormant streaming session) is quiesced, but the SERIAL flush wins and the
    adapter's sink flush is NOT called (would mis-route the watch tail)."""
    monkeypatch.setattr(itmod, '_get_h5pool',
                        lambda: SimpleNamespace(pause=lambda f: None,
                                                resume=lambda f: None))
    adapter = _SpyAdapter()
    order = []
    scan = SimpleNamespace(data_file='x.nxs',
                           _save_to_nexus=lambda: order.append('save'))
    w = SimpleNamespace(
        _scan_session_adapter=adapter,
        _streaming_session=SimpleNamespace(), _streaming_sink=SimpleNamespace(),
        _active_scan=scan, xye_only=False, _frames_since_save=4,
        file_lock=threading.RLock(),
        _save_due=lambda scan, force=False: True,
        _flush_xye_buffer=lambda s: order.append('xye'),
        sigPaused=SimpleNamespace(emit=lambda: order.append('paused')),
    )
    w._enter_pause = MethodType(imageThread._enter_pause, w)
    _bind_serial_tail(w)
    w._enter_pause()

    assert adapter.calls == [('quiesce', 30.0)]      # quiesced, but NOT flushed
    assert 'flush' not in adapter.calls
    assert 'save' in order and 'xye' in order
    assert w._frames_since_save == 0
    assert order[-1] == 'paused'


def test_wait_if_paused_resumes_adapter_on_exit():
    """On leaving the pause spin (resume), _wait_if_paused clears the session
    pause flag via adapter.resume() so the next submit isn't rejected (4a)."""
    adapter = _SpyAdapter()
    w = SimpleNamespace(command='pause', _scan_session_adapter=adapter)
    w._enter_pause = lambda: adapter.calls.append('enter')
    w._wait_if_paused = MethodType(imageThread._wait_if_paused, w)

    done = []
    t = threading.Thread(target=lambda: (w._wait_if_paused(), done.append(True)))
    t.start()
    time.sleep(0.1)
    assert 'resume' not in adapter.calls      # not resumed while still paused
    w.command = 'start'
    t.join(timeout=2)
    assert done == [True]
    assert adapter.calls[-1] == 'resume'      # resumed on exit


# ── 4d: run-state reads the session seam; pause must not bump generation ─────

def test_scan_session_property_exposes_adapter():
    """4d: `wrangler.scan_session` is the single read-only accessor for the
    streaming adapter — None before a session opens, the adapter once set.  The
    GUI reads run-state through this, never the private slot."""
    from xdart.gui.tabs.static_scan.wranglers.wrangler_widget import wranglerThread
    w = SimpleNamespace(_scan_session_adapter=None)
    assert wranglerThread.scan_session.fget(w) is None
    sentinel = object()
    w._scan_session_adapter = sentinel
    assert wranglerThread.scan_session.fget(w) is sentinel


def test_session_run_active_or_logic():
    """4d: `_session_run_active` reads `wrangler.scan_session.is_running` when a
    session is open, returns False otherwise (so the OR with `_run_active` falls
    through to the cache), and never raises on a partial/duck wrangler."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    f = staticWidget._session_run_active

    # no wrangler / no session -> False (cache governs)
    assert f(SimpleNamespace(wrangler=None)) is False
    assert f(SimpleNamespace(wrangler=SimpleNamespace(scan_session=None))) is False
    # open session reports its state through
    assert f(SimpleNamespace(wrangler=SimpleNamespace(
        scan_session=SimpleNamespace(is_running=True)))) is True
    assert f(SimpleNamespace(wrangler=SimpleNamespace(
        scan_session=SimpleNamespace(is_running=False)))) is False

    # a raising `is_running` is swallowed (never crashes the control-state apply)
    class _Boom:
        @property
        def is_running(self):
            raise RuntimeError("boom")
    assert f(SimpleNamespace(wrangler=SimpleNamespace(scan_session=_Boom()))) is False


def test_pause_resume_does_not_bump_display_generation():
    """4d (R-generation): a pause/resume cycle changes neither the selection nor
    the mode, so it MUST NOT bump `display_generation` (which gates stale-render
    drops).  The pause display side-effect is `set_processing_active` (a pure
    bool flip) + a same-selection re-fire (sig unchanged -> no bump).  A real
    selection change still bumps — proving the guard is sensitive, not inert."""
    from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget
    host = SimpleNamespace(display_generation=7, _last_selection_sig=None,
                           idxs=[3], overall=False)
    host._bump_display_generation = MethodType(
        displayFrameWidget._bump_display_generation, host)
    host._note_selection_generation = MethodType(
        displayFrameWidget._note_selection_generation, host)
    host.set_processing_active = MethodType(
        displayFrameWidget.set_processing_active, host)

    host._note_selection_generation()            # records baseline sig, no bump
    g0 = host.display_generation
    assert g0 == 7

    # pause: freeze the display, lift later — neither toggles generation
    host.set_processing_active(False)
    host.set_processing_active(True)
    assert host.display_generation == g0

    # resume re-fires the SAME standing selection -> sig unchanged -> no bump
    host._note_selection_generation()
    assert host.display_generation == g0

    # sensitivity: a genuine selection change DOES bump
    host.idxs = [4]
    host._note_selection_generation()
    assert host.display_generation == g0 + 1
