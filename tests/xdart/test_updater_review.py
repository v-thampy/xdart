"""UPDATER-127 review oracles through real state files, OS locks and child processes."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.xdart.test_updater import _job, _script, _await, hold_update_lock
from tests.xdart.test_updater_gui import window, qapp, _meta, _pump
from xdart import _gui_main, _updater
from xdart.modules import updater, update_launch


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX helper contract")
def test_abandoned_attempt_cannot_run_a_late_helper(tmp_path):
    state, marker, ran = tmp_path / "state", tmp_path / "relaunch", tmp_path / "updated"
    previous = updater.begin_attempt(state, install="test-install", current="1.2.6", target="1.2.7", log="l")
    job_path = _job(state, _script(f"open({str(ran)!r}, 'w').write('ran')"), marker=marker)
    # Capture the exact already-launched job, then revoke that attempt.
    updater.abandon_attempt(state, previous)
    _updater.main([str(job_path)])
    assert not ran.exists() and not marker.exists()


def test_plain_manifest_excludes_a_dependency_channel_constraint(tmp_path, monkeypatch):
    from tests.xdart.test_updater import _pixi_home, _REPODATA
    _home, env = _pixi_home(tmp_path, spec='{ version = "*", channel = "https://example.invalid/private" }')
    monkeypatch.setattr(updater, "current_version", lambda: "1.2.5")
    result = updater.check_for_update("pixi-global", prefix=str(env), opener=lambda *_: _REPODATA.read_bytes())
    assert not result.automatic


def test_unreachable_priority_channel_does_not_authorize_a_lower_channel(tmp_path):
    from tests.xdart.test_updater import _pixi_home, _REPODATA
    _home, env = _pixi_home(tmp_path, channels='["https://first.invalid", "https://second.invalid"]')
    def fetch(url, _timeout):
        if url.startswith("https://first.invalid"):
            raise OSError("first channel is unavailable")
        return _REPODATA.read_bytes()
    assert updater.check_for_update("pixi-global", prefix=str(env), opener=fetch).status == "unreachable"


def test_help_keeps_a_constrained_install_on_the_manual_command(window, monkeypatch):
    shown = []
    window._update_kind, window._update_source = "pixi-global", "help"
    monkeypatch.setattr(updater, "current_version", lambda: "1.2.6")
    monkeypatch.setattr(window, "_show_copyable_update", lambda *args: shown.append(args))
    window._on_update_check(updater.UpdateCheck("ok", "1.2.7", False, "pinned manifest"), window._update_generation)
    assert shown == [("pixi-global", "1.2.7", "1.2.6", "pinned manifest")]


def test_disabling_startup_updates_cancels_the_live_offer(window, qapp, monkeypatch, tmp_path):
    from tests.xdart.test_updater_gui import _startup_result
    admitted = _startup_result(window, monkeypatch, tmp_path)
    assert window._update_dialog is not None
    window.actionCheckUpdatesAtStartup.setChecked(False)
    assert window._update_dialog is None
    assert not _pump(qapp, lambda: bool(admitted), 2.5)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX helper lock")
def test_another_live_helper_cannot_have_its_attempt_overwritten(window, monkeypatch, tmp_path):
    state = updater.update_state_dir(_meta(tmp_path))
    updater.begin_attempt(state, install="test-install", current="1.2.6", target="1.2.7", log="original")
    before = (state / updater.STATE_FILE).read_bytes()
    holder = hold_update_lock(state)
    monkeypatch.setattr(updater, "install_kind", lambda: "pixi-global")
    monkeypatch.setattr(updater, "install_identity", lambda: "test-install")
    monkeypatch.setattr(window, "_show_copyable_update", lambda *a: None)
    real_launch = update_launch.launch_helper
    # Exercise the real helper without opening a native Terminal during an offscreen test.
    monkeypatch.setattr(update_launch, "launch_helper", lambda root, job: real_launch(root, job, platform="linux"))
    try:
        window._admit_update("pixi-global", _meta(tmp_path), "1.2.8")
        assert (state / updater.STATE_FILE).read_bytes() == before
    finally:
        holder.kill()
        holder.wait()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX process group")
def test_output_failure_reaps_the_real_update_before_returning(tmp_path):
    class BrokenOutput:
        def line(self, _text):
            raise OSError("log volume is full")

    result = {}
    job = {"update_cmd": _script("print('started', flush=True); import time; time.sleep(2)"),
           "time_limit_s": 5, "grace_s": .2, "tree_wait_s": 1}
    try:
        with pytest.raises(OSError, match="volume is full"):
            _updater.run_update(job, BrokenOutput(), result)
        assert not _updater._group_alive(result["update_pid"])
    finally:
        # A failed parent oracle may leave its own two-second stand-in; reap naturally.
        time.sleep(2.1)
        try:
            os.waitpid(result["update_pid"], 0)
        except ChildProcessError:
            pass


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX helper completion ordering")
def test_relaunch_observes_a_quiescent_completed_attempt(tmp_path, monkeypatch):
    state, marker = tmp_path / "state", tmp_path / "observed"
    job_path = _job(state, _script("pass"), marker=tmp_path / "unused")
    job = json.loads(job_path.read_text())
    code = ("import json; from pathlib import Path; from xdart.modules import updater; "
            f"state=Path({str(state)!r}); "
            "outcome,_=updater.settle_previous_attempt(state, install='test-install', current='1.2.7', alive=lambda p: True); "
            f"Path({str(marker)!r}).write_text(outcome)")
    job["relaunch_cmd"] = _script(code)
    updater.write_json_atomic(job_path, job)
    real_relaunch = _updater.relaunch
    def relaunch_and_let_the_child_run(cmd):
        real_relaunch(cmd)
        assert _await(marker)
    monkeypatch.setattr(_updater, "relaunch", relaunch_and_let_the_child_run)
    assert _updater.main([str(job_path)]) == 0
    assert _await(marker)
    assert marker.read_text() == "updated"


def test_second_start_respects_the_app_reservation_before_helper_handshake(tmp_path):
    reservation = updater.reserve_update(tmp_path)
    try:
        updater.begin_attempt(tmp_path, install="test-install", current="1.2.6", target="1.2.7", log="l")
        before = (tmp_path / updater.STATE_FILE).read_bytes()
        assert updater.settle_previous_attempt(tmp_path, install="test-install", current="1.2.6",
                                                alive=lambda _: False)[0] == "active"
        assert (tmp_path / updater.STATE_FILE).read_bytes() == before
        with pytest.raises(updater.UpdateInProgress):
            updater.reserve_update(tmp_path)
    finally:
        reservation.close()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="real POSIX helper handshake")
def test_run_during_helper_launch_revokes_the_attempt_without_closing(window, monkeypatch, tmp_path):
    processes, exited = [], []
    monkeypatch.setattr(updater, "install_kind", lambda: "pixi-global")
    monkeypatch.setattr(window, "exit", lambda: exited.append(True))
    def launch(state, job):
        path = update_launch.write_job(state, job)
        processes.append(subprocess.Popen([sys.executable, "-m", "xdart._updater", str(path)]))
        window.activity[0] = True
    monkeypatch.setattr(update_launch, "launch_helper", launch)
    try:
        window._admit_update("pixi-global", _meta(tmp_path), "1.2.7")
        assert not exited and not window._update_busy
        state = updater.update_state_dir(_meta(tmp_path))
        assert updater.load_update_state(state, updater.install_identity())["attempt"] is None
        assert processes[0].wait(timeout=5) in (1, 2)
    finally:
        for proc in processes:
            proc.wait(timeout=5)


def test_each_attempt_keeps_its_own_immutable_job(tmp_path):
    first = _job(tmp_path, _script("pass"), marker=tmp_path / "one")
    before = first.read_bytes()
    second = _job(tmp_path, _script("pass"), marker=tmp_path / "two")
    assert first != second and first.read_bytes() == before


def test_entrypoint_refuses_an_active_update_before_importing_qt(tmp_path):
    reservation = updater.reserve_update(tmp_path)
    # The notice's system program is stubbed: a real one leaves a dialog on the screen of whoever runs the tests.
    code = ("import sys; from pathlib import Path; from xdart.modules import updater, update_launch; "
            "updater.install_kind=lambda:'pixi-global'; "
            f"updater.state_dir_for=lambda kind:Path({str(tmp_path)!r}); "
            "show=update_launch.show_native_notice; "
            "update_launch.show_native_notice=lambda title, text: show(title, text, "
            "popen=lambda argv, **kwargs: print('notice program', argv[0])); "
            "from xdart.xdart_main import main; rc=main(); "
            "assert not any(k.startswith(('PySide6','pyqtgraph')) for k in sys.modules); sys.exit(rc)")
    try:
        done = subprocess.run(_script(code), capture_output=True, text=True, timeout=10)
        assert done.returncode == 0 and "update" in done.stderr, done.stdout + done.stderr
        assert ("notice program" in done.stdout) == (sys.platform == "darwin" or sys.platform.startswith("win"))
    finally:
        reservation.close()


def test_editable_bootstrap_remains_available(monkeypatch):
    monkeypatch.setattr(updater, "is_editable_install", lambda: True)
    def forbidden(*args, **kwargs):
        pytest.fail("editable bootstrap read install metadata")
    monkeypatch.setattr(updater, "find_install_meta", forbidden)
    assert updater.startup_update_blocked() is False


def test_startup_probe_accepts_an_unheld_readonly_lock_file(tmp_path):
    path = tmp_path / updater.LOCK_FILE
    path.write_bytes(b"existing lock")
    path.chmod(0o444)
    try:
        assert updater.helper_lock_held(tmp_path) is False
        assert path.read_bytes() == b"existing lock"
    finally:
        path.chmod(0o644)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX helper race")
def test_closing_during_handshake_revokes_a_late_helper(window, monkeypatch, tmp_path):
    jobs = []
    monkeypatch.setattr(updater, "install_kind", lambda: "pixi-global")
    monkeypatch.setattr(update_launch, "launch_helper", lambda root, job: jobs.append(update_launch.write_job(root, job)))
    window._admit_update("pixi-global", _meta(tmp_path), "1.2.7")
    assert window._update_busy
    window.close()
    assert not window._update_busy
    assert _updater.main([str(jobs[0])]) == 2


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="native Windows Job Object and argv")
def test_windows_job_tracks_a_late_child_after_its_parent_exits(tmp_path):
    from tests.xdart.test_updater import _windows_job, _run_powershell
    child = tmp_path / "child.pid"
    command = ("import subprocess,sys,time; time.sleep(.7); "
               "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
               f"open({str(child)!r},'w').write(str(p.pid))")
    job, state, marker = _windows_job(tmp_path, _script(command))
    done = _run_powershell(job)
    assert done.returncode in (0, 1), done.stdout + done.stderr
    assert _await(marker, 30)
    assert not _updater._pid_alive(int(child.read_text()))


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="native PowerShell5.1 argv")
def test_windows_job_preserves_empty_quoted_and_trailing_slash_arguments(tmp_path):
    from tests.xdart.test_updater import _windows_job, _run_powershell
    received = tmp_path / "argv.json"
    arguments = ["", 'a"quoted value', "space with trailing\\", "Jane O'Neil ü"]
    command = _script(f"import json,sys; open({str(received)!r},'w',encoding='utf-8').write(json.dumps(sys.argv[1:]))")
    job, _state, marker = _windows_job(tmp_path, command + arguments)
    done = _run_powershell(job)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(received.read_text(encoding="utf-8")) == arguments
    assert _await(marker, 30)
