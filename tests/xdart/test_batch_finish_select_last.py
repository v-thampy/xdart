# -*- coding: utf-8 -*-
"""Regression: the GUI must reload + auto-select the LAST frame when a run
finishes.  The live run already pointed file_thread.fname at the output file
(new_scan wires it internal=True), so the end-of-run auto-load MUST be
internal=True to get past set_file's same-file dedupe — otherwise the reload and
the select-last silently no-op and the last frame never appears."""
from types import SimpleNamespace, MethodType
import logging

from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget


class _StopTimer:
    def __init__(self):
        self.stopped = 0

    def stop(self):
        self.stopped += 1

    def trigger(self):
        pass


def _finish_host(tmp_path, *, batch, saw_frame, xye_only=False,
                 reintegrate_running=False):
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
    loaded = []          # records post-live scan.frames populate (load_frame_index_only)
    host = SimpleNamespace(
        integratorTree=SimpleNamespace(
            integrator_thread=SimpleNamespace(
                isRunning=lambda: reintegrate_running)),
        _exit_run_state=lambda: None,
        _update_timer=_StopTimer(),
        _list_timer=_StopTimer(),
        _reint_update_timer=_StopTimer(),
        _flush_pending_update=lambda: None,
        thread_state_changed=lambda: None,
        _apply_integration_control_state=lambda: None,
        h5viewer=h5viewer, wrangler=wrangler,
        # frames.index empty == the post-live state (streaming wrote the .nxs but
        # never populated scan.frames); load_frame_index_only is the populate hook.
        scan=SimpleNamespace(
            name="scanA", data_file=str(nxs),
            frames=SimpleNamespace(index=[]),
            load_frame_index_only=lambda f: (loaded.append(f), 0)[1],
        ),
        _run_saw_frame=saw_frame,
    )
    host._loaded_paths = loaded
    host.wrangler_finished = MethodType(staticWidget.wrangler_finished, host)
    # wrangler_finished now delegates the post-live frame-index populate to the
    # real _reconcile_h5viewer_frame_list_after_run (it calls
    # scan.load_frame_index_only and returns the indexed count).  Bind the real
    # method so the mock exercises that contract: load_frame_index_only fires
    # (populating _loaded_paths) and the reconcile returns 0 for these mocks (no
    # frame index) so the reload branch is correctly skipped.
    host._reconcile_h5viewer_frame_list_after_run = MethodType(
        staticWidget._reconcile_h5viewer_frame_list_after_run, host)
    return host, h5viewer, str(nxs), calls


def test_batch_finish_forces_internal_reload_and_select_last(tmp_path):
    host, h5viewer, nxs, calls = _finish_host(tmp_path, batch=True, saw_frame=True)
    host.wrangler_finished()
    # exactly one reload, forced internal (past the same-file dedupe)
    assert calls == [(nxs, True)], f"expected one internal reload; got {calls}"
    assert h5viewer._auto_select_last_on_finish is True   # select-last armed
    assert host._reint_update_timer.stopped == 1


def test_append_zero_frame_finish_forces_internal_reload(tmp_path):
    # non-batch Append that processed 0 new frames -> also reload + select last
    host, h5viewer, nxs, calls = _finish_host(tmp_path, batch=False, saw_frame=False)
    host.wrangler_finished()
    assert calls == [(nxs, True)], f"expected one internal reload; got {calls}"
    assert h5viewer._auto_select_last_on_finish is True


def test_nonbatch_run_that_saw_frames_does_not_reload(tmp_path):
    # a normal live run that already displayed frames must NOT do a display reload
    # (set_file) at finish — but it DOES do the lightweight scan.frames populate so
    # the Reintegrate buttons work post-live.
    host, h5viewer, nxs, calls = _finish_host(tmp_path, batch=False, saw_frame=True)
    host.wrangler_finished()
    assert calls == [], f"a frames-seen run must not display-reload; got {calls}"
    assert host._loaded_paths == [nxs], \
        f"post-live should populate scan.frames once from the .nxs; got {host._loaded_paths}"


def test_post_live_warns_when_indexed_less_than_processed(tmp_path, caplog):
    host, _h5viewer, _nxs, _calls = _finish_host(
        tmp_path, batch=False, saw_frame=True)
    host.wrangler.thread.files_processed = 5

    with caplog.at_level(logging.WARNING):
        host.wrangler_finished()

    assert "indexed fewer frames than processed" in caplog.text


def test_batch_finish_skips_reload_while_reintegrate_running(tmp_path):
    """Overlap guard (review finding): if a reintegrate-all is still WRITING at
    batch finish, the forced internal=True reload must NOT fire — it could read a
    half-written .nxs.  The reintegrate's own completion refreshes the display."""
    host, h5viewer, nxs, calls = _finish_host(
        tmp_path, batch=True, saw_frame=True, reintegrate_running=True)
    host.wrangler_finished()
    assert calls == [], f"must not reload while a reintegrate writes; got {calls}"
