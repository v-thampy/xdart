# -*- coding: utf-8 -*-
"""Update-on-exit helper — launched DETACHED by the GUI as
``python -m xdart._updater <parent_pid> <app_root> <update_cmd_json>
<relaunch_cmd_json> <log_path>``.

It waits for the running xdart to exit (its env files are then unlocked — this is
the one cross-platform-safe moment to upgrade), runs the installer's update
command, writes a log, and relaunches xdart (the new version on success, the old
one on failure).  Stdlib only, so it keeps working inside the freshly-updated env.
See ``docs/design/design_install_and_update_jul2026.md`` section 4.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _pid_alive(pid):
    """True while process ``pid`` exists (signal 0 probe; no signal sent)."""
    if pid <= 0:
        return False
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


def run_update(update_cmd, relaunch_cmd, log_path, *, runner=None, launcher=None):
    """Run ``update_cmd``; relaunch via ``relaunch_cmd`` (new version on success,
    OLD version on failure — the shim launches whatever is installed either way).
    Always writes a log to ``log_path``.  Returns True iff the update succeeded.
    ``runner``/``launcher`` are injectable for tests (no real pixi, no real spawn).
    """
    runner = runner or (lambda cmd: subprocess.run(  # noqa: E731
        cmd, capture_output=True, text=True))
    launcher = launcher or (lambda cmd: subprocess.Popen(cmd))  # noqa: E731
    lines = ["xdart update-on-exit", f"update_cmd: {update_cmd}"]
    ok = False
    if update_cmd:
        try:
            proc = runner(list(update_cmd))
            ok = getattr(proc, "returncode", 1) == 0
            lines += [f"returncode: {getattr(proc, 'returncode', '?')}",
                      "--- stdout ---", getattr(proc, "stdout", "") or "",
                      "--- stderr ---", getattr(proc, "stderr", "") or ""]
        except Exception as exc:                       # pragma: no cover - defensive
            lines.append(f"update FAILED to run: {exc!r}")
    else:
        lines.append("no update_cmd — nothing to do")
    lines.append("RESULT: " + ("success" if ok else "FAILED — relaunching old version"))
    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("\n".join(str(ln) for ln in lines))
    except OSError:
        pass
    if relaunch_cmd:
        try:
            launcher(list(relaunch_cmd))
        except Exception:                              # pragma: no cover - defensive
            pass
    return ok


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 4:
        sys.stderr.write(
            "usage: python -m xdart._updater PARENT_PID APP_ROOT "
            "UPDATE_CMD_JSON RELAUNCH_CMD_JSON [LOG_PATH]\n")
        return 2
    try:
        parent_pid = int(argv[0])
    except ValueError:
        parent_pid = -1
    app_root = argv[1]
    update_cmd = json.loads(argv[2]) if argv[2] else []
    relaunch_cmd = json.loads(argv[3]) if argv[3] else []
    log_path = argv[4] if len(argv) > 4 and argv[4] else str(
        Path(app_root or ".") / "update.log")
    if parent_pid > 0:
        wait_for_exit(parent_pid)
    run_update(update_cmd, relaunch_cmd, log_path)
    return 0


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main())
