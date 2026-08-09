"""Pause/Resume tests for the worker pause primitive and GUI freeze guard.

The dynamic worker quiesces and commits through its mounted
``ScanSessionAdapter`` before publishing ``sigPaused``.  Resume keeps the same
session alive and does not change the display generation.
"""
import threading
import time
from types import SimpleNamespace, MethodType

from tests.xdart._accepted_run import accepted_run  # noqa: E402
import xdart.gui.tabs.static_scan.wranglers.image_wrangler_thread as itmod

imageThread = itmod.imageThread

# ── _wait_if_paused: block until resume/stop, no-op otherwise ────────────────

def test_wait_if_paused_noop_when_not_paused():
    entered = []
    w = SimpleNamespace(command='start', run_configuration=accepted_run())
    w._enter_pause = lambda _frozen: entered.append('enter')
    w._wait_if_paused = MethodType(imageThread._wait_if_paused, w)
    w._wait_if_paused(w.run_configuration)
    assert entered == []          # never enters pause when command != 'pause'


def test_wait_if_paused_blocks_until_resume_and_enters_once():
    entered = []
    w = SimpleNamespace(command='pause', run_configuration=accepted_run())
    w._enter_pause = lambda _frozen: entered.append('enter')
    w._wait_if_paused = MethodType(imageThread._wait_if_paused, w)

    done = []
    t = threading.Thread(target=lambda: (w._wait_if_paused(w.run_configuration), done.append(True)))
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
    w = SimpleNamespace(command='pause', run_configuration=accepted_run())
    w._enter_pause = lambda _frozen: None
    w._wait_if_paused = MethodType(imageThread._wait_if_paused, w)
    done = []
    t = threading.Thread(target=lambda: (w._wait_if_paused(w.run_configuration), done.append(True)))
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
            display_generation=9,
            set_processing_active=lambda v: calls.append(('proc', v)),
            invalidate_image_level_caches=lambda: calls.append(('levels', None)),
            request_current_selection_repaint=lambda **kwargs:
                calls.append(('repaint', kwargs))),
        h5viewer=SimpleNamespace(
            file_thread=SimpleNamespace(live_run=True),
            set_run_writing=lambda v: calls.append(('write', v))),
    )
    w._set_scan_integrated_reads_transient = MethodType(
        staticWidget._set_scan_integrated_reads_transient, w)  # no-op: host has no scan
    staticWidget._on_run_paused(w)
    assert ('proc', False) in calls and ('write', False) in calls   # guard LIFTED
    assert ('levels', None) in calls
    assert ('repaint', {'generation': 9, 'reason': 'pause'}) in calls
    assert w.h5viewer.file_thread.live_run is False
    assert w._run_active is True            # run still active, just frozen


def test_on_run_resuming_reengages_guard_before_resume():
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    calls = []
    class _FileThread:
        live_run = False

    file_thread = _FileThread()
    w = SimpleNamespace(
        _run_active=True,
        displayframe=SimpleNamespace(
            set_processing_active=lambda v: calls.append(('proc', v))),
        h5viewer=SimpleNamespace(
            live_run_active=True,
            file_thread=file_thread,
            set_run_writing=lambda v: calls.append(
                ('write', v, file_thread.live_run))),
    )
    w._set_scan_integrated_reads_transient = MethodType(
        staticWidget._set_scan_integrated_reads_transient, w)  # no-op: host has no scan
    staticWidget._on_run_resuming(w)
    assert ('write', True, True) in calls and ('proc', True) in calls
    assert file_thread.live_run is True


def test_active_paused_browse_does_not_hydrate_integrator_settings():
    """A paused run may load display data, never a browsed scan's run config."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    calls = []
    w = SimpleNamespace(
        _run_active=True,
        _controls_v2_enabled=lambda: True,
        _controls_v2_ensure_native_int_defaults=lambda: calls.append("defaults"),
        _controls_v2_hydrate_advanced_from_scan=lambda: calls.append("hydrate"),
        _refresh_controls_v2_profile=lambda **kwargs: calls.append("refresh"),
        integratorTree=SimpleNamespace(
            hydrate_from_scan=lambda: calls.append("legacy")),
    )

    staticWidget._hydrate_integrator_on_load(w, "browsed_scan.nxs")

    assert calls == []


def test_paused_transition_enables_full_browser_frame_index_load():
    """Pause switches the file thread from live repoint to normal scan load."""
    from xdart.gui.tabs.static_scan.scan_threads import fileHandlerThread
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    loaded = []
    scan = SimpleNamespace(
        skip_2d=False,
        frames=SimpleNamespace(data_file="live_output.nxs"),
        # X1 O-3 (c2): the full browser load supplies the canonical name.
        set_datafile=lambda path, name=None: loaded.append(path),
    )
    file_thread = SimpleNamespace(
        scan=scan,
        fname="browsed_scan.nxs",
        file_lock=threading.RLock(),
        live_run=True,
        no_nxs=False,
        sigNewFile=SimpleNamespace(emit=lambda *_: None),
        sigUpdate=SimpleNamespace(emit=lambda *_: None),
    )
    w = SimpleNamespace(
        _run_active=True,
        displayframe=SimpleNamespace(
            display_generation=1,
            set_processing_active=lambda _value: None,
            invalidate_image_level_caches=lambda: None,
            request_current_selection_repaint=lambda **_kwargs: None,
        ),
        h5viewer=SimpleNamespace(
            file_thread=file_thread,
            set_run_writing=lambda _value: None,
        ),
    )
    w._set_scan_integrated_reads_transient = MethodType(
        staticWidget._set_scan_integrated_reads_transient, w)

    staticWidget._on_run_paused(w)
    MethodType(fileHandlerThread.set_datafile, file_thread)()

    assert loaded == ["browsed_scan.nxs"]


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

# ── 4c-1: _enter_pause / _wait_if_paused route through ScanSessionAdapter ────

class _SpyAdapter:
    """Record the accepted adapter cadence used by the pause path."""

    def __init__(self, drained=True, flush_due=True, commit_error=None):
        self._drained = drained
        self._flush_due = flush_due
        self._commit_error = commit_error
        self.calls = []

    def quiesce(self, timeout=None):
        self.calls.append(('quiesce', timeout))
        return self._drained

    def should_flush(self, frames_since_flush, *, unsaved_in_memory=None,
                     force=False):
        self.calls.append(
            ('should_flush', frames_since_flush, unsaved_in_memory, force))
        return self._flush_due

    def commit_epoch(self):
        self.calls.append('commit_epoch')
        if self._commit_error is not None:
            raise self._commit_error

    def resume(self):
        self.calls.append('resume')


def test_enter_pause_streaming_routes_through_adapter():
    """Pause commits in cadence order and resets only after commit succeeds."""
    adapter = _SpyAdapter()
    w = SimpleNamespace(
        PAUSE_DRAIN_TIMEOUT=imageThread.PAUSE_DRAIN_TIMEOUT,
        _scan_session_adapter=adapter,
        _frames_since_save=4,
        sigPaused=SimpleNamespace(emit=lambda: adapter.calls.append('sigPaused')),
        _retain_dynamic_failure=lambda *_args: None,
    )
    w._enter_pause = MethodType(imageThread._enter_pause, w)
    w._enter_pause(accepted_run())

    assert adapter.calls == [
        ('quiesce', 30.0),
        ('should_flush', 4, None, True),
        'commit_epoch',
        'sigPaused',
    ]
    assert w._frames_since_save == 0

    error = RuntimeError("commit failed")
    failed = _SpyAdapter(commit_error=error)
    retained = []
    w = SimpleNamespace(
        PAUSE_DRAIN_TIMEOUT=imageThread.PAUSE_DRAIN_TIMEOUT,
        _scan_session_adapter=failed,
        _frames_since_save=4,
        sigPaused=SimpleNamespace(emit=lambda: failed.calls.append('sigPaused')),
        _retain_dynamic_failure=lambda exc, message: retained.append(
            (exc, message)),
    )
    w._enter_pause = MethodType(imageThread._enter_pause, w)
    w._enter_pause(accepted_run())

    assert failed.calls == [
        ('quiesce', 30.0),
        ('should_flush', 4, None, True),
        'commit_epoch',
    ]
    assert w._frames_since_save == 4
    assert retained == [(error, "Pause failed; run stopped")]




def test_wait_if_paused_resumes_adapter_on_exit():
    """On leaving the pause spin (resume), _wait_if_paused clears the session
    pause flag via adapter.resume() so the next submit isn't rejected (4a)."""
    adapter = _SpyAdapter()
    w = SimpleNamespace(command='pause', _scan_session_adapter=adapter,
                        run_configuration=accepted_run())
    w._enter_pause = lambda _frozen: adapter.calls.append('enter')
    w._wait_if_paused = MethodType(imageThread._wait_if_paused, w)

    done = []
    t = threading.Thread(target=lambda: (w._wait_if_paused(w.run_configuration), done.append(True)))
    t.start()
    time.sleep(0.1)
    assert 'resume' not in adapter.calls      # not resumed while still paused
    w.command = 'start'
    t.join(timeout=2)
    assert done == [True]
    assert adapter.calls[-1] == 'resume'      # resumed on exit


# ── 4d: run-state reads the public probe; pause must not bump generation ─────

def test_session_run_active_or_logic():
    """The scalar public probe is total over absent/false/true/raising states."""
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget
    f = staticWidget._session_run_active

    # No wrangler or no public activity probe leaves the cache in control.
    assert f(SimpleNamespace(wrangler=None)) is False
    assert f(SimpleNamespace(wrangler=SimpleNamespace())) is False

    # The public scalar probe reports the healthy dynamic run state.
    assert f(SimpleNamespace(wrangler=SimpleNamespace(
        dynamic_session_running=lambda: False))) is False
    assert f(SimpleNamespace(wrangler=SimpleNamespace(
        dynamic_session_running=lambda: True))) is True

    def raising_probe():
        raise RuntimeError("boom")

    assert f(SimpleNamespace(wrangler=SimpleNamespace(
        dynamic_session_running=raising_probe))) is False


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
