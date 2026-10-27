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

from xdart.modules.updater import HELPER_FILE, JOB_FILE, write_json_atomic


class UpdateLaunchRefused(RuntimeError):
    """The helper cannot be started safely; the caller keeps xdart open and shows the copyable command."""


def write_job(state_dir, job):
    path = Path(state_dir) / JOB_FILE
    write_json_atomic(path, {**job, "state_dir": str(state_dir), "helper_file": HELPER_FILE})
    return path


def sh_quote(value):
    """POSIX shell single-quoting: the value cannot end the quotes (``'`` becomes ``'"'"'``)."""
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def command_wrapper_text(python, job_path):
    return ("#!/bin/sh\n"
            "# xdart update window: runs the update helper; the window can be closed once it says so.\n"
            f"exec {sh_quote(python)} -m xdart._updater {sh_quote(job_path)}\n")


def write_command_wrapper(state_dir, python, job_path):
    path = Path(state_dir) / "xdart-update.command"
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


def launch_linux(python, job_path, *, popen=subprocess.Popen):
    popen([str(python), "-m", "xdart._updater", str(job_path)], stdin=subprocess.DEVNULL,
          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)


def ps_literal(value):
    """A PowerShell single-quoted literal: nothing inside is expanded; ``'`` is doubled."""
    return "'" + str(value).replace("'", "''") + "'"


_POWERSHELL = r"""
$ErrorActionPreference = 'Stop'
$job = Get-Content -LiteralPath __JOB__ -Raw -Encoding UTF8 | ConvertFrom-Json
$helperFile = Join-Path ([string]$job.state_dir) ([string]$job.helper_file)
$started = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
$result = @{ pid = $PID; state = 'completed'; started_at = $started; exit = $null; timed_out = $false; relaunched = $false }
function Save-Helper($data) {
  $tmp = "$helperFile.tmp"
  ($data | ConvertTo-Json -Depth 4) | Set-Content -LiteralPath $tmp -Encoding UTF8
  Move-Item -LiteralPath $tmp -Destination $helperFile -Force
}
function Say([string]$text) {
  Write-Host $text
  Add-Content -LiteralPath ([string]$job.log) -Encoding UTF8 -Value ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $text)
}
function Descendants([int]$root) {
  try { $all = @(Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId) } catch { return @() }
  $found = @(); $frontier = @($root)
  while ($frontier.Count -gt 0) {
    $next = @()
    foreach ($id in $frontier) { foreach ($q in $all) {
      if ($q.ParentProcessId -eq $id -and $q.ProcessId -ne $id -and $found -notcontains $q.ProcessId) { $found += $q.ProcessId; $next += $q.ProcessId }
    } }
    $frontier = $next
  }
  return $found
}
function Start-Xdart {
  $exe = [string]$job.relaunch_cmd[0]
  $rest = @($job.relaunch_cmd | Select-Object -Skip 1)
  # Its own console (not this window's) and no job: closing this window cannot end xdart. Not hidden: a hidden start
  # can pass SW_HIDE on to xdart's first window.
  if ($rest.Count -gt 0) { Start-Process -FilePath $exe -ArgumentList $rest | Out-Null }
  else { Start-Process -FilePath $exe | Out-Null }
  $result.relaunched = $true
}
function Invoke-Update {
  Say ("Updating xdart from {0} to {1}. Keep this window open; xdart reopens when the update has finished." -f $job.from, $job.target)
  Say ("Command: " + (@($job.update_cmd) -join ' '))
  $deadline = (Get-Date).AddSeconds([double]$job.parent_wait_s)
  while ((Get-Process -Id ([int]$job.parent_pid) -ErrorAction SilentlyContinue) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Milliseconds 500 }
  if (Get-Process -Id ([int]$job.parent_pid) -ErrorAction SilentlyContinue) {
    Say 'xdart did not close, so nothing was changed.'; $result.refused = 'xdart did not close'; return 1
  }
  $prefix = ([string]$job.env_prefix).TrimEnd('\') + '\'
  $tramp = [string]$job.relaunch_cmd[0]
  try {
    $holders = @(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.ExecutablePath -and ($_.ExecutablePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or $_.ExecutablePath -ieq $tramp) })
  } catch {
    Say ("Could not list the programs using xdart's environment ({0}), so nothing was changed." -f $_.Exception.Message)
    $result.refused = 'environment check failed'; Start-Xdart; return 1
  }
  if ($holders.Count -gt 0) {
    $names = ($holders | ForEach-Object { "{0} (PID {1})" -f $_.Name, $_.ProcessId }) -join ', '
    Say ("Not updating: these programs use xdart's environment: {0}. Close them and update again." -f $names)
    $result.refused = 'environment in use'; Start-Xdart; return 1
  }
  foreach ($k in @($job.update_env.PSObject.Properties)) { Set-Item -LiteralPath ("Env:" + $k.Name) -Value ([string]$k.Value) }
  $out = [string]$job.log + '.stdout'; $err = [string]$job.log + '.stderr'
  Remove-Item -LiteralPath $out, $err -ErrorAction SilentlyContinue
  $exe = [string]$job.update_cmd[0]
  $argv = @($job.update_cmd | Select-Object -Skip 1 | ForEach-Object { $a = [string]$_; if ($a -match '\s') { '"' + $a + '"' } else { $a } })
  $p = Start-Process -FilePath $exe -ArgumentList $argv -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  $null = $p.Handle   # keeps ExitCode readable after the exit (Windows PowerShell 5.1)
  $offsets = @{ $out = [long]0; $err = [long]0 }
  $pump = {
    foreach ($f in @($out, $err)) {
      if (-not (Test-Path -LiteralPath $f)) { continue }
      $fs = [IO.File]::Open($f, 'Open', 'Read', 'ReadWrite')
      try { $null = $fs.Seek($offsets[$f], 'Begin'); $text = (New-Object IO.StreamReader($fs)).ReadToEnd(); $offsets[$f] = $fs.Length } finally { $fs.Dispose() }
      foreach ($line in ($text -split "`r?`n")) { if ($line) { Say $line } }
    }
  }
  $t0 = Get-Date; $beat = $t0.AddSeconds(30); $snap = $t0; $seen = @(); $timedOut = $false
  while (-not $p.HasExited) {
    & $pump
    $now = Get-Date
    if ($now -ge $snap) { $seen = @($seen + (Descendants $p.Id) | Select-Object -Unique); $snap = $now.AddSeconds(5) }
    if ($now -ge $beat) { $el = $now - $t0; Say ("Still updating... {0} min {1} s" -f [int][math]::Floor($el.TotalMinutes), $el.Seconds); $beat = $now.AddSeconds(30) }
    if (($now - $t0).TotalSeconds -ge [double]$job.time_limit_s) {
      $timedOut = $true; $result.timed_out = $true
      Say ("The update is still running after {0} min; stopping it." -f [int]([double]$job.time_limit_s / 60))
      $kids = @(Get-CimInstance Win32_Process -Filter ("ParentProcessId={0}" -f $p.Id))
      foreach ($c in $kids) { Stop-Process -Id $c.ProcessId -Force -ErrorAction SilentlyContinue }
      Say ("Stopped {0} child process(es) of the update; waiting for pixi to finish." -f $kids.Count)
      if (-not $p.WaitForExit([int]([double]$job.grace_s * 1000))) {
        Say 'pixi did not finish; stopping the whole update.'
        & taskkill.exe /T /F /PID $p.Id | Out-Null
      }
      break
    }
    Start-Sleep -Milliseconds 500
  }
  $null = $p.WaitForExit(10000); & $pump
  $result.exit = $p.ExitCode
  $left = @($seen | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
  $until = (Get-Date).AddSeconds(10)
  while ($left.Count -gt 0 -and (Get-Date) -lt $until) { Start-Sleep -Milliseconds 500; $left = @($left | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }) }
  if ($left.Count -gt 0) {
    Say ("Part of the update is still running, so xdart was not reopened (it could find the environment half-changed). Log: {0}" -f $job.log)
    $result.refused = 'update still running'; return 1
  }
  if ($p.ExitCode -eq 0 -and -not $timedOut) { Say 'pixi finished. Reopening xdart; it confirms the new version when it starts.'; Start-Xdart; return 0 }
  Say ("The update did not finish (exit {0}). Reopening xdart; what it opens depends on how far the update got. Log: {1}" -f $p.ExitCode, $job.log)
  Start-Xdart; return 1
}
Save-Helper @{ pid = $PID; state = 'running'; started_at = $started }
$code = 1
try { $code = Invoke-Update } catch { Say ("The updater failed: {0}" -f $_.Exception.Message) }
finally { $result.finished_at = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0; Save-Helper $result }
if ($code -eq 0) { Write-Host 'This window closes in 10 s.'; Start-Sleep -Seconds 10 } else { Read-Host 'Press Enter to close this window' | Out-Null }
exit $code
"""


def powershell_script(job_path):
    """The Windows helper; the job path is its only embedded value, as a single-quoted literal."""
    return _POWERSHELL.replace("__JOB__", ps_literal(job_path))


def encoded_command(script):
    """``-EncodedCommand`` takes the script as base64 of its UTF-16LE bytes, so no part of it is re-quoted."""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


CREATE_NEW_CONSOLE = 0x00000010
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def powershell_argv(job_path):
    root = os.environ.get("SystemRoot") or r"C:\Windows"
    exe = os.path.join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    return [exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand",
            encoded_command(powershell_script(job_path))]


def launch_windows(job_path, *, popen=subprocess.Popen):
    """A visible console outside xdart's job. If the job refuses breakaway, the helper would die with xdart, so the
    update is refused (DETACHED_PROCESS does not leave a job)."""
    try:
        popen(powershell_argv(job_path), creationflags=CREATE_NEW_CONSOLE | CREATE_BREAKAWAY_FROM_JOB,
              close_fds=True)
    except OSError as exc:
        raise UpdateLaunchRefused(f"the update window could not be started outside xdart ({exc})") from exc


def launch_helper(state_dir, job, *, platform=None, python=None):
    """Write the job and start the helper for this platform; raises :class:`UpdateLaunchRefused`."""
    platform = sys.platform if platform is None else platform
    python = python or sys.executable
    job_path = write_job(state_dir, job)
    if platform.startswith("win"):
        launch_windows(job_path)
    elif platform == "darwin":
        launch_macos(state_dir, python, job_path)
    else:
        launch_linux(python, job_path)
    return job_path
