# -*- coding: utf-8 -*-
"""Update-on-exit helper (POSIX) — ``python -m xdart._updater <job.json>``.

The app writes the job (``xdart.modules.updater``); on macOS its progress window (``update_progress_macos``) starts
this helper once it shows. Windows uses the PowerShell helper from ``xdart.modules.update_launch`` outside the env,
because the env's own python.exe is locked while it runs. Steps: record the handshake (this PID), wait for xdart to
close, refuse unless the progress window still runs, run the update with its output logged as it arrives, stop it at
the time limit (child processes first, then pixi after a grace period, then the whole process group), and record the
outcome; the user restarts xdart from the progress window. A job without ``manual_restart`` (the older launchers)
instead relaunches xdart detached once nothing of the update is left running. Stdlib only. See
``docs/design/design_install_and_update_jul2026.md`` section 4.
"""
from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from xdart.modules import updater as update_state

_HEARTBEAT_S = 30.0


def _pid_alive(pid):
    """True while process ``pid`` exists.  POSIX uses a signal-0 probe; Windows
    uses an OpenProcess/WaitForSingleObject handle probe -- ``os.kill(pid, 0)`` on
    Windows calls ``TerminateProcess``, NOT a probe (B2), so it must never run
    there."""
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):             # pragma: no cover - windows
        import ctypes
        SYNCHRONIZE = 0x00100000
        WAIT_TIMEOUT = 0x00000102
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by someone else
    except OSError:
        return False
    return True


def wait_for_exit(pid, *, timeout=60.0, interval=0.5, sleep=time.sleep,
                  alive=_pid_alive):
    """Poll until the parent ``pid`` exits or ``timeout`` elapses.  Returns True if
    it is gone.  ``sleep``/``alive`` are injectable for tests."""
    waited = 0.0
    while waited < timeout:
        if not alive(pid):
            return True
        sleep(interval)
        waited += interval
    return not alive(pid)


def _write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class _Output:
    """Echo to the window (stdout, when there is one) and append to the log as each line arrives."""

    def __init__(self, log_path):
        self._log = open(log_path, "a", encoding="utf-8")

    def line(self, text):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for part in str(text).splitlines() or [""]:
            self._log.write(f"{stamp} {part}\n")
            if sys.stdout is not None:
                try:
                    sys.stdout.write(part + "\n")
                    sys.stdout.flush()
                except (OSError, ValueError):
                    pass
        self._log.flush()

    def close(self):
        self._log.close()


def _group_members(pgid):
    """PIDs in process group ``pgid`` (POSIX), from ``ps``; empty when it cannot be read."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,pgid="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    members = []
    for row in out.splitlines():
        fields = row.split()
        if len(fields) == 2 and fields[1] == str(pgid) and fields[0].isdigit():
            members.append(int(fields[0]))
    return members


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _wait_until(predicate, timeout, drain):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        drain()
        if predicate():
            return True
        time.sleep(0.2)
    drain()
    return predicate()


def run_update(job, out, result=None):
    """Run the update command; returns ``(exit_code_or_None, timed_out, tree_gone)``."""
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (job.get("update_env") or {}).items()})
    proc = subprocess.Popen(list(job["update_cmd"]), env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    pgid = proc.pid
    if result is not None:
        result["update_pid"] = proc.pid
    try:
        return _monitor_update(job, out, proc)
    except BaseException:
        # Cleanup cannot depend on the output sink that may have just failed.
        _signal_group(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            pass
        if _group_alive(proc.pid):
            _signal_group(proc.pid, signal.SIGKILL)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        _wait_until(lambda: not _group_alive(proc.pid), 2, lambda: None)
        raise


def _monitor_update(job, out, proc):
    pgid = proc.pid
    lines = queue.Queue()

    def read():
        for raw in iter(proc.stdout.readline, b""):
            lines.put(raw.decode("utf-8", "replace").rstrip("\r\n"))
        proc.stdout.close()

    reader = threading.Thread(target=read, name="xdart-update-output", daemon=True)
    reader.start()

    def drain():
        while True:
            try:
                out.line(lines.get_nowait())
            except queue.Empty:
                return

    started = time.monotonic()
    limit = float(job.get("time_limit_s") or 1200)
    beat = started + _HEARTBEAT_S
    timed_out = False
    while proc.poll() is None:
        drain()
        now = time.monotonic()
        if now >= beat:
            out.line(f"Still updating… {int((now - started) // 60)} min {int((now - started) % 60)} s")
            beat = now + _HEARTBEAT_S
        if now - started >= limit:
            timed_out = True
            out.line(f"The update is still running after {int(limit // 60)} min; stopping it.")
            children = [pid for pid in _group_members(pgid) if pid != proc.pid]
            for pid in children:
                try:
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            out.line(f"Stopped {len(children)} child process(es) of the update; waiting for pixi to finish.")
            if not _wait_until(lambda: proc.poll() is not None, float(job.get("grace_s") or 30), drain):
                out.line("pixi did not finish; stopping the whole update.")
                _signal_group(pgid, signal.SIGTERM)
                if not _wait_until(lambda: proc.poll() is not None, 5.0, drain):
                    _signal_group(pgid, signal.SIGKILL)
            break
        time.sleep(0.2)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    reader.join(timeout=5)
    drain()
    tree_gone = _wait_until(lambda: not _group_alive(pgid), float(job.get("tree_wait_s") or 10), drain)
    if not tree_gone and timed_out:
        _signal_group(pgid, signal.SIGKILL)
        tree_gone = _wait_until(lambda: not _group_alive(pgid), 5.0, drain)
    return proc.returncode, timed_out, tree_gone


def relaunch(cmd):
    """Start xdart detached: its own session and no terminal, so closing the update window cannot end it."""
    subprocess.Popen(list(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True, close_fds=True)


def _schedule_window_close(job):
    """macOS: the app opened this helper in its own Terminal window (``close_window_s``); close it after success."""
    delay = job.get("close_window_s")
    if not delay or sys.platform != "darwin" or sys.stdout is None or not sys.stdout.isatty():
        return False
    from xdart.modules import update_launch
    return update_launch.schedule_terminal_close(os.ttyname(sys.stdout.fileno()), float(delay))


def run_job(job):
    state_dir = Path(job["state_dir"])
    helper_file = state_dir / job.get("helper_file", "xdart-update-helper.json")
    lock = update_state.take_file_lock(state_dir / job.get("lock_file", "xdart-update.lock"))
    if lock is None:
        if sys.stdout is not None:
            print("Another xdart update is running for this install, so nothing was changed.", flush=True)
        return 2
    if update_state.job_attempt(job) is None:
        lock.close()
        return 2
    started_at = time.time()
    _write_json(helper_file, {"pid": os.getpid(), "state": "running", "started_at": started_at,
                              "attempt_id": job["attempt_id"]})
    out = _Output(job["log"])
    result = {"pid": os.getpid(), "state": "completed", "started_at": started_at, "exit": None,
              "timed_out": False, "relaunched": False, "attempt_id": job["attempt_id"]}
    old_signals = {}
    if threading.current_thread() is threading.main_thread():
        def interrupted(sig, _frame):
            raise SystemExit(128 + sig)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            old_signals[sig] = signal.signal(sig, interrupted)
    try:
        out.line(f"Updating xdart from {job.get('from')} to {job.get('target')}. "
                 + ("Choose Restart xdart when the update finishes." if job.get("manual_restart")
                    else "Keep this window open; xdart reopens when the update has finished."))
        out.line("Command: " + " ".join(str(part) for part in job["update_cmd"]))
        parent = int(job.get("parent_pid") or 0)
        deadline = time.monotonic() + float(job.get("parent_wait_s") or 60)
        while parent > 0 and _pid_alive(parent) and time.monotonic() < deadline:
            if update_state.job_attempt(job) is None:
                out.line("The update was cancelled in xdart, so nothing was changed.")
                result["refused"] = "attempt cancelled"
                return 1
            time.sleep(.1)
        attempt = update_state.job_attempt(job)
        if attempt is None or not attempt.get("authorized"):
            out.line("xdart did not confirm this update, so nothing was changed.")
            result["refused"] = "attempt not authorized"
            return 1
        if parent > 0 and _pid_alive(parent):
            out.line("xdart did not close, so nothing was changed. You can close this window.")
            result["refused"] = "xdart did not close"
            return 1
        if not update_state.progress_host_ready(job):
            out.line("The update progress window closed before installation, so nothing was changed.")
            result["refused"] = "the update progress window closed before installation"
            return 1
        _write_json(helper_file, {**result, "state": "running", "phase": "updating"})
        code, timed_out, tree_gone = run_update(job, out, result)
        result.update(exit=code, timed_out=timed_out)
        if not tree_gone:
            out.line("Part of the update is still running, so xdart was not reopened (it could find the environment "
                     f"half-changed). Recovery is required after those processes stop. Log: {job['log']}")
            result["refused"] = "update still running"
            result["state"] = "blocked"
            return 1
        if job.get("manual_restart"):
            out.line("Update complete. Choose Restart xdart in the progress window." if code == 0 and not timed_out
                     else f"Update failed. Read the details in the progress window or {job['log']}.")
            return 0 if code == 0 and not timed_out else 1
        if code == 0 and not timed_out:
            out.line("pixi finished. Reopening xdart; it confirms the new version when it starts.")
        else:
            out.line(f"The update did not finish (exit {code}). Reopening xdart; what it opens depends on how far the "
                     f"update got. Log: {job['log']}")
        # A fast relaunch must observe quiescent completion rather than the running handshake.
        result["finished_at"] = time.time()
        _write_json(helper_file, result)
        try:
            relaunch(job["relaunch_cmd"])
            result["relaunched"] = True
        except Exception as exc:                         # noqa: BLE001 - recorded for the user
            out.line(f"Could not reopen xdart ({exc!r}); open it yourself.")
        if code == 0 and not timed_out and result["relaunched"] and _schedule_window_close(job):
            out.line(f"This window closes in {float(job['close_window_s']):g} s.")
        else:
            out.line("You can close this window.")
        return 0 if code == 0 and not timed_out else 1
    finally:
        if "update_pid" in result and _group_alive(result["update_pid"]):
            result["state"] = "blocked"
            result["refused"] = "update still running"
        result["finished_at"] = time.time()
        try:
            _write_json(helper_file, result)
        finally:
            try:
                out.close()
            finally:
                lock.close()
                for sig, handler in old_signals.items():
                    signal.signal(sig, handler)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        if sys.stderr is not None:   # None under pythonw (detached from the GUI)
            sys.stderr.write("usage: python -m xdart._updater JOB_JSON\n")
        return 2
    try:
        with open(argv[0], "r", encoding="utf-8") as fh:
            job = json.load(fh)
    except (OSError, ValueError) as exc:
        if sys.stderr is not None:
            sys.stderr.write(f"xdart updater: cannot read the job file: {exc}\n")
        return 2
    return run_job(job)


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main())
