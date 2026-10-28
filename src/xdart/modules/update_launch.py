# -*- coding: utf-8 -*-
"""Start the update helper in a visible window (maintainer's pick 2026-10-27, "Visible window").

Qt-free. The app writes one JSON job file and passes only its path to the helper, so the only value quoted onto a
command line is that path: macOS runs ``xdart._updater`` in a Terminal window through a ``.command`` wrapper, Linux
runs it hidden, and Windows runs a PowerShell helper outside the env (the env's python.exe is locked while xdart
runs), started with CREATE_BREAKAWAY_FROM_JOB or not at all.
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

from xdart.modules.updater import HELPER_FILE, JOB_FILE, LOCK_FILE, write_json_atomic


class UpdateLaunchRefused(RuntimeError):
    """The helper cannot be started safely; the caller keeps xdart open and shows the copyable command."""


def write_job(state_dir, job):
    # A delayed Terminal/PowerShell launch must never read a replacement attempt's job.
    suffix = job.get("attempt_id")
    path = Path(state_dir) / (f"xdart-update-job-{suffix}.json" if suffix else JOB_FILE)
    write_json_atomic(path, {**job, "state_dir": str(state_dir), "helper_file": HELPER_FILE, "lock_file": LOCK_FILE,
                            "update_command_line": subprocess.list2cmdline(job["update_cmd"]),
                            "relaunch_arguments": subprocess.list2cmdline(job["relaunch_cmd"][1:])})
    return path


def sh_quote(value):
    """POSIX shell single-quoting: the value cannot end the quotes (``'`` becomes ``'"'"'``)."""
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def command_wrapper_text(python, job_path):
    return ("#!/bin/sh\n"
            "# xdart update window: runs the update helper; the window can be closed once it says so.\n"
            f"exec {sh_quote(python)} -m xdart._updater {sh_quote(job_path)}\n")


def write_command_wrapper(state_dir, python, job_path):
    path = Path(state_dir) / (Path(job_path).stem + ".command")
    path.write_text(command_wrapper_text(python, job_path), encoding="utf-8")
    os.chmod(path, 0o755)
    return path


def launch_macos(state_dir, python, job_path, *, runner=subprocess.run):
    """Open the wrapper in Terminal through LaunchServices (no Automation permission is needed)."""
    wrapper = write_command_wrapper(state_dir, python, job_path)
    done = runner(["/usr/bin/open", "-a", "Terminal", str(wrapper)], capture_output=True, text=True, timeout=30)
    if getattr(done, "returncode", 1) != 0:
        raise UpdateLaunchRefused(f"Terminal could not be opened ({(getattr(done, 'stderr', '') or '').strip()})")
    return wrapper


#: Seconds after a successful update before the macOS update window closes (the Windows console says the same).
CLOSE_WINDOW_S = 10

_TERMINAL_WINDOW_OF_TTY = """on run argv
  tell application "Terminal"
    repeat with w in windows
      try
        if (count of tabs of w) is 1 and tty of tab 1 of w is item 1 of argv then return id of w
      end try
    end repeat
  end tell
  return ""
end run"""

_CLOSE_TERMINAL_WINDOW = """on run argv
  set wid to (item 1 of argv) as integer
  delay (item 2 of argv) as number
  repeat 20 times
    try
      tell application "Terminal"
        if not (exists window id wid) then return
        if (count of tabs of window id wid) is not 1 then return
        if tty of tab 1 of window id wid is not item 3 of argv then return
        if not (busy of tab 1 of window id wid) then
          close window id wid
          return
        end if
      end tell
    end try
    delay 0.5
  end repeat
end run"""


def schedule_terminal_close(tty, delay_s, *, runner=subprocess.run, popen=subprocess.Popen):
    """Close the Terminal window showing this helper ``delay_s`` after it ends, when the window holds this tab alone
    (Terminal's AppleScript cannot close one tab of several). The window is found while this process still owns the
    tty; the closer runs detached and closes the window only once its tab runs nothing, so Terminal never asks to
    stop a process. Best effort: False leaves the window open."""
    try:
        found = runner(["/usr/bin/osascript", "-e", _TERMINAL_WINDOW_OF_TTY, str(tty)], capture_output=True,
                       text=True, timeout=10)
        window_id = (getattr(found, "stdout", "") or "").strip()
        if getattr(found, "returncode", 1) != 0 or not window_id.isdigit():
            return False
        popen(["/usr/bin/osascript", "-e", _CLOSE_TERMINAL_WINDOW, window_id, f"{delay_s:g}", str(tty)],
              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
              start_new_session=True, close_fds=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def launch_linux(python, job_path, *, popen=subprocess.Popen):
    popen([str(python), "-m", "xdart._updater", str(job_path)], stdin=subprocess.DEVNULL,
          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)


def ps_literal(value):
    """A PowerShell single-quoted literal: nothing inside is expanded; ``'`` is doubled."""
    return "'" + str(value).replace("'", "''") + "'"


_POWERSHELL = r"""
$ErrorActionPreference = 'Stop'
$job = Get-Content -LiteralPath __JOB__ -Raw -Encoding UTF8 | ConvertFrom-Json
Add-Type -TypeDefinition @'
__NATIVE_JOB__
'@
$helperFile = Join-Path ([string]$job.state_dir) ([string]$job.helper_file)
$started = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
$result = @{ pid = $PID; state = 'completed'; started_at = $started; exit = $null; timed_out = $false; relaunched = $false }
function Save-Helper($data) {
  $data['attempt_id'] = [string]$job.attempt_id
  $tmp = "$helperFile.tmp"
  ($data | ConvertTo-Json -Depth 4) | Set-Content -LiteralPath $tmp -Encoding UTF8
  Move-Item -LiteralPath $tmp -Destination $helperFile -Force
}
function Say([string]$text) {
  Write-Host $text
  Add-Content -LiteralPath ([string]$job.log) -Encoding UTF8 -Value ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $text)
}
function Read-Attempt {
  if ([string]::IsNullOrEmpty([string]$job.attempt_id)) { return $null }
  try {
    $s = Get-Content -LiteralPath (Join-Path ([string]$job.state_dir) 'xdart-update-state.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($s.install -eq $job.install -and $s.attempt.id -eq $job.attempt_id) { return $s.attempt }
  } catch { }
  return $null
}
function Start-Xdart {
  $result.finished_at = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
  Save-Helper $result  # quiescent completion precedes a fast restarted app's state check
  if ($job.manual_restart -eq $true) { return }
  $exe = [string]$job.relaunch_cmd[0]
  $rest = [string]$job.relaunch_arguments
  # Its own console without a window and no job: closing this window cannot end xdart, and no console window opens
  # beside it (pixi's xdart.exe is a console program). Not -WindowStyle Hidden: that passes SW_HIDE on to xdart.
  $line = '"' + $exe + '"'
  if ($rest.Length -gt 0) { $line += ' ' + $rest }
  try { [void][XdartUpdateJob]::Launch($exe, $line) }
  catch {
    Say ("Reopening xdart with a console window instead ({0})." -f $_.Exception.Message)
    if ($rest.Length -gt 0) { Start-Process -FilePath $exe -ArgumentList $rest | Out-Null }
    else { Start-Process -FilePath $exe | Out-Null }
  }
  $result.relaunched = $true
}
function Invoke-Update {
  Say ("Updating xdart from {0} to {1}. Keep this window open; xdart reopens when the update has finished." -f $job.from, $job.target)
  Say ("Command: " + (@($job.update_cmd) -join ' '))
  $deadline = (Get-Date).AddSeconds([double]$job.parent_wait_s)
  $parent = [int]$job.parent_pid
  while ($parent -gt 0 -and (Get-Process -Id $parent -ErrorAction SilentlyContinue) -and ((Get-Date) -lt $deadline)) {
    if ($null -eq (Read-Attempt)) { $result.refused = 'attempt cancelled'; return 1 }
    Start-Sleep -Milliseconds 100
  }
  $attempt = Read-Attempt
  if ($null -eq $attempt -or -not $attempt.authorized) { $result.refused = 'attempt not authorized'; return 1 }
  if ($parent -gt 0 -and (Get-Process -Id $parent -ErrorAction SilentlyContinue)) {
    Say 'xdart did not close, so nothing was changed.'; $result.refused = 'xdart did not close'; return 1
  }
  $prefix = ([string]$job.env_prefix).TrimEnd('\') + '\'
  $tramp = [string]$job.relaunch_cmd[0]
  # pixi's xdart.exe waits for xdart and exits just after it, so a program here gets env_wait_s to close, timed from
  # when it is first seen (the first process query in a fresh PowerShell can take seconds).
  $envDeadline = $null
  while ($true) {
    try {
      $holders = @(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.ExecutablePath -and ($_.ExecutablePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or $_.ExecutablePath -ieq $tramp) })
    } catch {
      Say ("Could not list the programs using xdart's environment ({0}), so nothing was changed." -f $_.Exception.Message)
      $result.refused = 'environment check failed'; Start-Xdart; return 1
    }
    if ($holders.Count -eq 0) { break }
    $names = ($holders | ForEach-Object { "{0} (PID {1})" -f $_.Name, $_.ProcessId }) -join ', '
    if ($null -eq $envDeadline) {
      Say ("Waiting up to {0} s for {1} to close." -f [double]$job.env_wait_s, $names)
      $envDeadline = (Get-Date).AddSeconds([double]$job.env_wait_s)
    } elseif ((Get-Date) -ge $envDeadline) {
      Say ("Not updating: these programs use xdart's environment: {0}. Close them and update again." -f $names)
      $result.refused = 'environment in use'; Start-Xdart; return 1
    }
    Start-Sleep -Milliseconds 500
  }
  if ($job.require_progress_host -eq $true) {
    $hostState = Get-Content -LiteralPath ([string]$job.host_file) -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($hostState.attempt_id -ne $job.attempt_id -or $hostState.ready -ne $true) {
      $result.refused = 'progress window is not ready'; return 1
    }
    try {
      $probe = [IO.File]::Open([string]$job.host_lock, 'Open', 'Read', 'None')
      $probe.Dispose(); $result.refused = 'progress window closed before installation'; return 1
    } catch [IO.IOException] {
      if (($_.Exception.HResult -band 65535) -notin @(32, 33)) { throw }
    } # only a sharing/lock violation proves a live host
  }
  Save-Helper @{ pid = $PID; state = 'running'; started_at = $started; phase = 'updating' }
  foreach ($k in @($job.update_env.PSObject.Properties)) { Set-Item -LiteralPath ("Env:" + $k.Name) -Value ([string]$k.Value) }
  $out = [string]$job.log + '.stdout'; $err = [string]$job.log + '.stderr'
  Remove-Item -LiteralPath $out, $err -ErrorAction SilentlyContinue
  $exe = [string]$job.update_cmd[0]
  $p = New-Object XdartUpdateJob -ArgumentList $exe, ([string]$job.update_command_line), $out, $err
  $script:updateProcess = $p
  $offsets = @{ $out = [long]0; $err = [long]0 }
  # Resume where the read stopped (Position): output written after it is read next time.
  $pump = {
    foreach ($f in @($out, $err)) {
      if (-not (Test-Path -LiteralPath $f)) { continue }
      $fs = [IO.File]::Open($f, 'Open', 'Read', 'ReadWrite')
      try { $null = $fs.Seek($offsets[$f], 'Begin'); $text = (New-Object IO.StreamReader($fs)).ReadToEnd(); $offsets[$f] = $fs.Position } finally { $fs.Dispose() }
      foreach ($line in ($text -split "`r?`n")) { if ($line) { Say $line } }
    }
  }
  $t0 = Get-Date; $beat = $t0.AddSeconds(30); $timedOut = $false
  while (-not $p.HasExited) {
    & $pump
    $now = Get-Date
    if ($now -ge $beat) { $el = $now - $t0; Say ("Still updating... {0} min {1} s" -f [int][math]::Floor($el.TotalMinutes), $el.Seconds); $beat = $now.AddSeconds(30) }
    if (($now - $t0).TotalSeconds -ge [double]$job.time_limit_s) {
      $timedOut = $true; $result.timed_out = $true
      Say ("The update is still running after {0} min; stopping it." -f [int]([double]$job.time_limit_s / 60))
      $kidCount = $p.StopChildren()
      Say ("Stopped {0} child process(es) of the update; waiting for pixi to finish." -f $kidCount)
      if (-not $p.WaitForExit([int]([double]$job.grace_s * 1000))) {
        Say 'pixi did not finish; stopping the whole update.'
        $p.Stop()
      }
      break
    }
    Start-Sleep -Milliseconds 500
  }
  $null = $p.WaitForExit(10000); & $pump
  $result.exit = $p.ExitCode
  $until = (Get-Date).AddSeconds(10)
  while ($p.ActiveCount -gt 0 -and (Get-Date) -lt $until) { Start-Sleep -Milliseconds 100 }
  if ($p.ActiveCount -gt 0) {
    $p.Stop()
    $until = (Get-Date).AddSeconds(5)
    while ($p.ActiveCount -gt 0 -and (Get-Date) -lt $until) { Start-Sleep -Milliseconds 100 }
  }
  if ($p.ActiveCount -gt 0) {
    Say ("Part of the update is still running, so xdart was not reopened (it could find the environment half-changed). Log: {0}" -f $job.log)
    $result.refused = 'update still running'; $result.state = 'blocked'; return 1
  }
  if ($p.ExitCode -eq 0 -and -not $timedOut) {
    if ($job.manual_restart -eq $true) { Say 'Update complete. Choose Restart xdart in the progress window.' }
    else { Say 'pixi finished. Reopening xdart; it confirms the new version when it starts.' }
    Start-Xdart; return 0
  }
  Say ("The update did not finish (exit {0}). See the update details and log: {1}" -f $p.ExitCode, $job.log)
  Start-Xdart; return 1
}
$lockPath = Join-Path ([string]$job.state_dir) ([string]$job.lock_file)
# Held unshared for this helper's life; xdart treats an open lock as an update still running.
try { $lockStream = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None) }
catch { Write-Host 'Another xdart update is running for this install, so nothing was changed.'; exit 2 }
if ($null -eq (Read-Attempt)) { $lockStream.Dispose(); exit 2 }
Save-Helper @{ pid = $PID; state = 'running'; started_at = $started }
$code = 1
$script:updateProcess = $null
try { $code = Invoke-Update } catch { Say ("The updater failed: {0}" -f $_.Exception.Message) }
finally {
  try {
    if ($null -ne $script:updateProcess) {
      if ($script:updateProcess.ActiveCount -gt 0) {
        $script:updateProcess.Stop()
        $until = (Get-Date).AddSeconds(5)
        while ($script:updateProcess.ActiveCount -gt 0 -and (Get-Date) -lt $until) { Start-Sleep -Milliseconds 100 }
      }
      if ($script:updateProcess.ActiveCount -gt 0) { $result.state = 'blocked' }
    }
  } catch { $result.state = 'blocked' }
  finally {
    try {
      $result.finished_at = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
      Save-Helper $result
    } finally {
      if ($null -ne $script:updateProcess) { $script:updateProcess.Dispose() }
      $lockStream.Dispose()
    }
  }
}
if ($job.pause_at_end -ne $false) {
  if ($code -eq 0) { Write-Host 'This window closes in 10 s.'; Start-Sleep -Seconds 10 } else { Read-Host 'Press Enter to close this window' | Out-Null }
}
exit $code
"""


def powershell_script(job_path):
    """The Windows helper; the job path is its only embedded value, as a single-quoted literal."""
    from .windows_update_job import SOURCE
    return _POWERSHELL.replace("__JOB__", ps_literal(job_path)).replace("__NATIVE_JOB__", SOURCE)


def encoded_command(script):
    """``-EncodedCommand`` takes the script as base64 of its UTF-16LE bytes, so no part of it is re-quoted."""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


CREATE_NEW_CONSOLE = 0x00000010
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def powershell_argv(job_path):
    root = os.environ.get("SystemRoot") or r"C:\Windows"
    exe = os.path.join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    script_path = Path(job_path).with_suffix(".ps1")
    # Keep CreateProcess below its command-line limit; Windows PowerShell 5.1 needs a BOM for Unicode source.
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(powershell_script(job_path), encoding="utf-8-sig")
    return [exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand",
            encoded_command("& " + ps_literal(script_path))]


def launch_windows(job_path, *, popen=subprocess.Popen):
    """A visible console outside xdart's job. If the job refuses breakaway, the helper would die with xdart, so the
    update is refused (DETACHED_PROCESS does not leave a job)."""
    try:
        popen(powershell_argv(job_path), creationflags=CREATE_NEW_CONSOLE | CREATE_BREAKAWAY_FROM_JOB,
              close_fds=True)
    except OSError as exc:
        raise UpdateLaunchRefused(f"the update window could not be started outside xdart ({exc})") from exc


def launch_helper(state_dir, job, *, platform=None, python=None):
    """Prepare one immutable job and start its independent progress host (legacy jobs retain their old launcher)."""
    platform = sys.platform if platform is None else platform
    python = python or sys.executable
    if job.get("manual_restart"):
        if platform != "darwin" and not platform.startswith("win"):
            raise UpdateLaunchRefused("This platform has no independent update progress window. Use the manual command.")
        suffix = job["attempt_id"]
        base = Path(state_dir) / f"xdart-update-job-{suffix}"
        base.parent.mkdir(parents=True, exist_ok=True)
        job.update(host_file=str(base.with_suffix(".host.json")), host_lock=str(base.with_suffix(".host.lock")))
        Path(job["host_lock"]).touch(exist_ok=True)
        job_path = write_job(state_dir, job)
        if platform.startswith("win"):
            from . import update_progress_windows
            # Engine and host live in separate processes, both outside the mutable environment.
            engine = Path(job_path).with_suffix(".ps1")
            engine.write_text(powershell_script(job_path), encoding="utf-8-sig")
            host = Path(job_path).with_suffix(".host.ps1")
            host.write_text(update_progress_windows.script(job_path, engine), encoding="utf-8-sig")
            subprocess.Popen(update_progress_windows.argv(host),
                             creationflags=CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB, close_fds=True)
        else:
            from . import update_progress_macos
            host = Path(job_path).with_suffix(".js")
            host.write_text(update_progress_macos.script(job_path, python), encoding="utf-8")
            subprocess.Popen(["/usr/bin/osascript", "-l", "JavaScript", str(host)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True, close_fds=True)
        return job_path
    if platform == "darwin":
        job = {**job, "close_window_s": CLOSE_WINDOW_S}
    job_path = write_job(state_dir, job)
    if platform.startswith("win"):
        launch_windows(job_path)
    elif platform == "darwin":
        launch_macos(state_dir, python, job_path)
    else:
        launch_linux(python, job_path)
    return job_path


CREATE_NO_WINDOW = 0x08000000


def applescript_string(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def show_native_notice(title, text, *, popen=subprocess.Popen, platform=None):
    """Tell the user why xdart did not start, through a system program started detached (osascript; Windows
    PowerShell), so this process exits at once and keeps no file of an environment an update may be changing open.
    A launch from the app or Start menu has no terminal; stderr alone would show nothing."""
    platform = sys.platform if platform is None else platform
    if sys.stderr is not None:
        try:
            print(text, file=sys.stderr)
        except (OSError, ValueError):
            pass
    try:
        if platform == "darwin":
            script = (f"display dialog {applescript_string(text)} with title {applescript_string(title)} "
                      'buttons {"OK"} default button "OK" with icon caution')
            popen(["/usr/bin/osascript", "-e", script], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                  stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        elif platform.startswith("win"):
            root = os.environ.get("SystemRoot") or r"C:\Windows"
            command = ("Add-Type -AssemblyName PresentationFramework; "
                       f"[void][System.Windows.MessageBox]::Show({ps_literal(text)}, {ps_literal(title)})")
            argv = [os.path.join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"), "-NoProfile",
                    "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded_command(command)]
            try:
                popen(argv, creationflags=CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB, close_fds=True)
            except OSError:
                popen(argv, creationflags=CREATE_NO_WINDOW, close_fds=True)
    except Exception:
        pass   # the stderr line stays; never let the notice itself fail startup handling
