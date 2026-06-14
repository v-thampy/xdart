# -*- coding: utf-8 -*-
"""Regression: the GUI must reload + auto-select the LAST frame when a run
finishes.  The live run already pointed file_thread.fname at the output file
(new_scan wires it internal=True), so the end-of-run auto-load MUST be
internal=True to get past set_file's same-file dedupe — otherwise the reload and
the select-last silently no-op and the last frame never appears."""
from types import SimpleNamespace, MethodType

from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget


def _finish_host(tmp_path, *, batch, saw_frame, xye_only=False):
    nxs = tmp_path / "scan.nxs"
    nxs.write_bytes(b"")                         # os.path.exists -> True
    calls = []
    file_thread = SimpleNamespace(live_run=True, no_nxs=True, fname=str(nxs))
    h5viewer = SimpleNamespace(
        live_run_active=True, file_thread=file_thread, dirname=str(tmp_path),
        _auto_select_last_on_finish=False,
        update_scans=lambda: None,
        set_file=lambda fname, *, internal=False: calls.append((fname, internal)),
    )
    wrangler = SimpleNamespace(
        thread=SimpleNamespace(batch_mode=batch, xye_only=xye_only, fname=str(nxs)),
        fname=str(nxs), stop=lambda: None,
        # scan_name differs from host.scan.name -> the post-load branch takes the
        # `else: wrangler.enabled(True)` path (no integrator_thread_finished).
        scan_name="other", enabled=lambda e: None,
    )
    host = SimpleNamespace(
        integratorTree=SimpleNamespace(
            integrator_thread=SimpleNamespace(isRunning=lambda: False)),
        _exit_run_state=lambda: None,
        _update_timer=SimpleNamespace(stop=lambda: None),
        _flush_pending_update=lambda: None,
        thread_state_changed=lambda: None,
        h5viewer=h5viewer, wrangler=wrangler,
        scan=SimpleNamespace(name="scanA", data_file=str(nxs)),
        _run_saw_frame=saw_frame,
    )
    host.wrangler_finished = MethodType(staticWidget.wrangler_finished, host)
    return host, h5viewer, str(nxs), calls


def test_batch_finish_forces_internal_reload_and_select_last(tmp_path):
    host, h5viewer, nxs, calls = _finish_host(tmp_path, batch=True, saw_frame=True)
    host.wrangler_finished()
    # exactly one reload, forced internal (past the same-file dedupe)
    assert calls == [(nxs, True)], f"expected one internal reload; got {calls}"
    assert h5viewer._auto_select_last_on_finish is True   # select-last armed


def test_append_zero_frame_finish_forces_internal_reload(tmp_path):
    # non-batch Append that processed 0 new frames -> also reload + select last
    host, h5viewer, nxs, calls = _finish_host(tmp_path, batch=False, saw_frame=False)
    host.wrangler_finished()
    assert calls == [(nxs, True)], f"expected one internal reload; got {calls}"
    assert h5viewer._auto_select_last_on_finish is True


def test_nonbatch_run_that_saw_frames_does_not_reload(tmp_path):
    # a normal live run that already displayed frames must NOT reload at finish
    host, h5viewer, nxs, calls = _finish_host(tmp_path, batch=False, saw_frame=True)
    host.wrangler_finished()
    assert calls == [], f"a frames-seen run must not reload; got {calls}"
