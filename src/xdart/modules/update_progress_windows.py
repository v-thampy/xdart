"""External Windows progress window; the existing PowerShell helper owns updates.

The host uses system PowerShell/WPF, so updating the application's environment
does not unload its UI. It starts the unchanged engine only after showing itself.
"""
from __future__ import annotations

import os


def script(job_path, engine_path):
    """Return the STA host source, embedding paths as literals and the existing native launcher."""
    from .update_launch import encoded_command, ps_literal
    from .windows_update_job import SOURCE
    return (_SCRIPT.replace("__NATIVE_JOB__", SOURCE)
            .replace("__ENGINE__", encoded_command("& " + ps_literal(engine_path)))
            .replace("__JOB__", ps_literal(job_path)))


def argv(host_path):
    """System PowerShell invocation; the caller uses NO_WINDOW and BREAKAWAY_FROM_JOB."""
    from .update_launch import encoded_command, ps_literal
    root = os.environ.get("SystemRoot") or r"C:\Windows"
    exe = os.path.join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    return [exe, "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-EncodedCommand",
            encoded_command("& " + ps_literal(host_path))]


_SCRIPT = r'''
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName PresentationFramework
$job = Get-Content -LiteralPath __JOB__ -Raw -Encoding UTF8 | ConvertFrom-Json
# Compiled only when Restart needs it, so the window and the helper start without waiting for a compile.
$nativeJob = @'
__NATIVE_JOB__
'@
$helperPath = Join-Path ([string]$job.state_dir) ([string]$job.helper_file)
$statePath = Join-Path ([string]$job.state_dir) 'xdart-update-state.json'
$lockPath = Join-Path ([string]$job.state_dir) ([string]$job.lock_file)
$admissionPath = Join-Path ([string]$job.state_dir) 'xdart-update-admission.lock'
function Open-Lock([string]$path) {
  return [IO.File]::Open($path, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
}
function Read-Json([string]$path) {
  try { return (Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json) } catch { return $null }
}
function Current-Attempt {
  $s = Read-Json $statePath
  return ($null -ne $s -and $s.install -ceq $job.install -and
          $s.attempt.id -ceq $job.attempt_id -and $s.attempt.authorized -eq $true)
}
function Current-Helper {
  $h = Read-Json $helperPath
  if ($null -ne $h -and $h.attempt_id -ceq $job.attempt_id -and
      $null -ne $script:worker -and $h.pid -eq $script:worker.Id) { return $h }
  return $null
}
function Successful($h) {
  return ($null -ne $h -and $h.state -eq 'completed' -and $null -ne $h.exit -and
          $h.exit -eq 0 -and -not $h.timed_out -and -not $h.refused)
}
function Read-Tail {
  $fs = $null
  try {
    $fs = [IO.File]::Open([string]$job.log, 'Open', 'Read', ([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
    # This update's output only: from where the log ended when the window opened, at most its last 128 KiB.
    $end = $fs.Length  # one snapshot: appends between offset and count must not exceed the cap
    $from = [Math]::Min($end, [Math]::Max([long]$script:logStart, $end - 131072))
    $count = [int]($end - $from)
    $null = $fs.Seek($from, [IO.SeekOrigin]::Begin)
    $bytes = New-Object byte[] $count
    $read = $fs.Read($bytes, 0, $count)
    return [Text.Encoding]::UTF8.GetString($bytes, 0, $read)
  } catch { return $null } finally { if ($null -ne $fs) { $fs.Dispose() } }
}
[xml]$xaml = @'
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation" Title="Updating xdart"
        Width="760" Height="520" MinWidth="600" MinHeight="360" WindowStartupLocation="CenterScreen">
  <Grid Margin="20">
    <Grid.RowDefinitions><RowDefinition Height="Auto"/><RowDefinition Height="Auto"/>
      <RowDefinition Height="Auto"/><RowDefinition Height="*"/><RowDefinition Height="Auto"/></Grid.RowDefinitions>
    <TextBlock Name="Version" FontSize="19" Margin="0,0,0,12" TextWrapping="Wrap"
               ToolTip="The installed version and the version requested by this update."/>
    <TextBlock Name="Stage" Grid.Row="1" Margin="0,0,0,10" TextWrapping="Wrap"
               ToolTip="Current update stage. Keep this window open while the updater is active."/>
    <ProgressBar Name="Progress" Grid.Row="2" Height="6" Margin="0,0,0,14" IsIndeterminate="True"
                 ToolTip="The update is active; the package manager does not report a completion percentage."/>
    <TextBox Name="Details" Grid.Row="3" IsReadOnly="True" TextWrapping="Wrap"
             VerticalScrollBarVisibility="Auto" HorizontalScrollBarVisibility="Disabled"
             FontFamily="Consolas" Padding="8" ToolTip="Recent update output. Select and copy text for troubleshooting."/>
    <StackPanel Grid.Row="4" Orientation="Horizontal" HorizontalAlignment="Right" Margin="0,16,0,0">
      <Button Name="Restart" Content="Restart xdart" IsEnabled="False" Padding="15,7" Margin="0,0,10,0"
              ToolTip="Open xdart after this update has finished and all update processes have exited."/>
      <Button Name="Close" Content="Close" IsEnabled="False" Padding="15,7"
              ToolTip="Close this window after the updater has exited. The update log remains on disk."/>
    </StackPanel>
  </Grid>
</Window>
'@
$window = [Windows.Markup.XamlReader]::Load((New-Object Xml.XmlNodeReader $xaml))
$version = $window.FindName('Version'); $stage = $window.FindName('Stage')
$bar = $window.FindName('Progress'); $details = $window.FindName('Details')
$restart = $window.FindName('Restart'); $close = $window.FindName('Close')
$version.Text = "xdart $($job.from) → $($job.target)"
$stage.Text = 'Preparing the update window…'
$details.ToolTip = "Recent output (up to 128 KiB). Full log: $($job.log)"
$script:worker = $null; $script:started = $false; $script:terminal = $false
$script:failure = ''; $script:hostLock = $null; $script:logStart = [long]0
try { $script:logStart = (Get-Item -LiteralPath ([string]$job.log)).Length } catch { }  # it holds earlier updates too
function Refresh-Progress {
  $exitedBefore = ($null -ne $script:worker -and $script:worker.HasExited)
  $tail = Read-Tail
  if ($null -ne $tail -and $details.Text -cne $tail) { $details.Text = $tail; $details.ScrollToEnd() }
  if ($null -eq $script:worker) { return }
  $h = Current-Helper
  $dead = $script:worker.HasExited
  $script:terminal = $dead
  $close.IsEnabled = $dead; $restart.IsEnabled = $false; $bar.IsIndeterminate = -not $dead
  if (-not $dead) {
    if ($null -ne $h -and $h.state -eq 'running') {
      $latest = if ($tail) { ($tail.TrimEnd() -split "`r?`n")[-1] } else { '' }
      $stage.Text = 'Updating xdart. Keep this window open.'
      if ($latest) { $stage.Text = $latest.Substring(0, [Math]::Min(240, $latest.Length)) }
    }
    elseif ($null -ne $h -and $h.state -eq 'completed') { $stage.Text = 'Finishing update cleanup…' }
    elseif ($null -ne $h -and $h.state -eq 'blocked') { $stage.Text = 'Cleanup could not be confirmed. Waiting for the updater to exit; restart is unavailable.' }
    elseif (((Get-Date) - $script:launchedAt).TotalSeconds -gt 60) {
      $stage.Text = 'The updater did not report its startup. Keep this window open while it remains active; see the log.'
    } else { $stage.Text = 'Waiting for the updater to start…' }
    return
  }
  $free = $false; $probe = $null
  try { $probe = Open-Lock $lockPath; $free = $true } catch { } finally { if ($probe) { $probe.Dispose() } }
  if ((Successful $h) -and $script:worker.ExitCode -eq 0 -and $free -and (Current-Attempt)) {
    $stage.Text = 'The update finished. Restart xdart to confirm the installed version.'
    $restart.IsEnabled = $true
  } elseif (-not $free -or ($null -ne $h -and $h.state -eq 'blocked')) {
    $stage.Text = 'Update cleanup could not be confirmed. Restart is unavailable. See the log.'
  } elseif ($null -eq $h) {
    $stage.Text = 'The updater exited without a matching completion report. See the log.'
  } else { $stage.Text = "The update did not finish successfully. Restart is unavailable. Log: $($job.log)" }
  if ($script:failure) { $stage.Text = $script:failure }
  if ($exitedBefore) { $timer.Stop() }  # this read followed the helper's exit, so the output and outcome are final
}
$timer = New-Object Windows.Threading.DispatcherTimer
$timer.Interval = [TimeSpan]::FromMilliseconds(500)
$timer.Add_Tick({
  try { Refresh-Progress } catch {
    $restart.IsEnabled = $false
    if ($null -ne $script:worker -and $script:worker.HasExited) { $script:terminal = $true; $close.IsEnabled = $true }
    $stage.Text = "Progress could not be read: $($_.Exception.Message). The updater retains control of cleanup."
  }
})
$restart.Add_Click({
  $admission = $null; $probe = $null
  try {
    # Compile before taking admission: a manual startup during compilation must be seen by the fresh checks below.
    # Looked up by name: a type literal here could fail when this handler compiles, before Add-Type has run.
    $native = 'XdartUpdateJob' -as [type]
    if ($null -eq $native) { Add-Type -TypeDefinition $nativeJob; $native = 'XdartUpdateJob' -as [type] }
    $admission = Open-Lock $admissionPath
    if ($null -eq $script:worker -or -not $script:worker.HasExited -or $script:worker.ExitCode -ne 0 -or
        -not (Current-Attempt) -or -not (Successful (Current-Helper))) { throw 'This update is no longer ready to restart.' }
    $probe = Open-Lock $lockPath
    # Keep the helper lock through launch to exclude another update; release admission so the new app can start.
    $admission.Dispose(); $admission = $null
    $exe = [string]$job.relaunch_cmd[0]
    $line = '"' + $exe + '"'
    if ([string]$job.relaunch_arguments) { $line += ' ' + [string]$job.relaunch_arguments }
    [void]$native::Launch($exe, $line)
    $script:failure = ''
    $window.Close()
  } catch {
    $script:failure = "xdart could not be restarted: $($_.Exception.Message). Try Restart again or close this window."
    $stage.Text = $script:failure
  } finally {
    if ($probe) { $probe.Dispose() }
    if ($admission) { $admission.Dispose() }
  }
})
$close.Add_Click({ $window.Close() })
$window.Add_Closing({
  param($sender, $event)
  if (-not $script:terminal) { $event.Cancel = $true; $stage.Text = 'The updater is active. Keep this window open until it finishes.' }
})
$window.Add_ContentRendered({
  if ($script:started) { return }; $script:started = $true
  [void]$window.Dispatcher.BeginInvoke([Windows.Threading.DispatcherPriority]::Background, [Action]{
    try {
      $ready = @{ attempt_id = [string]$job.attempt_id; pid = $PID; ready = $true }
      $tmp = [string]$job.host_file + '.tmp'
      ($ready | ConvertTo-Json) | Set-Content -LiteralPath $tmp -Encoding UTF8
      Move-Item -LiteralPath $tmp -Destination ([string]$job.host_file) -Force
      $script:launchedAt = Get-Date
      $script:worker = Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -WindowStyle Hidden -PassThru -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', '__ENGINE__')
      $timer.Start()
      $null = $script:worker.Handle
    } catch {
      $script:terminal = ($null -eq $script:worker -or $script:worker.HasExited)
      $bar.IsIndeterminate = -not $script:terminal; $close.IsEnabled = $script:terminal
      $stage.Text = "The updater could not report progress: $($_.Exception.Message). Log: $($job.log)"
    }
  })
})
$window.Add_Closed({ $timer.Stop() })
$window.Dispatcher.Add_UnhandledException({
  param($sender, $event)
  $event.Handled = $true; $restart.IsEnabled = $false
  $stage.Text = "The progress window encountered an error. The updater retains control of cleanup. Log: $($job.log)"
})
try {
  $script:hostLock = Open-Lock ([string]$job.host_lock)
  [void]$window.ShowDialog()
} catch {
  [void][Windows.MessageBox]::Show("The update window failed: $($_.Exception.Message). Log: $($job.log)", 'xdart update')
} finally {
  $timer.Stop()
  if ($script:hostLock) { $script:hostLock.Dispose() }
  if ($null -ne $script:worker) { $script:worker.Dispose() }
}
'''
