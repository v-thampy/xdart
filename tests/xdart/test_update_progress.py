"""Native-host custody and the real update engine; no OS GUI/Terminal is launched here."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.xdart.test_updater import _job, _script, _await
from xdart import _updater
from xdart.modules import update_launch, updater


@pytest.mark.skipif(sys.platform.startswith('win'), reason='POSIX engine')
def test_manual_restart_runs_update_but_never_launches_xdart(tmp_path):
    ran, reopened = tmp_path/'updated', tmp_path/'reopened'
    path = _job(tmp_path/'state', _script(f"open({str(ran)!r}, 'w').write('updated')"), marker=reopened)
    job = json.loads(path.read_text()); job['manual_restart'] = True
    updater.write_json_atomic(path, job)
    assert _updater.main([str(path)]) == 0
    assert ran.exists() and not _await(reopened, seconds=.4)
    helper = updater.read_json(path.parent/updater.HELPER_FILE)
    assert helper['state'] == 'completed' and helper['exit'] == 0 and not helper['relaunched']
    assert updater.helper_lock_held(path.parent) is False


@pytest.mark.skipif(sys.platform.startswith('win'), reason='POSIX engine')
def test_missing_progress_host_refuses_before_environment_mutation(tmp_path):
    ran = tmp_path/'updated'
    path = _job(tmp_path/'state', _script(f"open({str(ran)!r}, 'w').write('updated')"), marker=tmp_path/'reopened')
    job = json.loads(path.read_text()); job.update(manual_restart=True, require_progress_host=True)
    updater.write_json_atomic(path, job)
    assert _updater.main([str(path)]) == 1 and not ran.exists()
    assert 'progress window' in updater.read_json(path.parent/updater.HELPER_FILE)['refused']


def test_host_readiness_requires_exact_attempt_and_a_live_os_lock(tmp_path):
    job = dict(require_progress_host=True, attempt_id='one', host_file=str(tmp_path/'host.json'), host_lock=str(tmp_path/'host.lock'))
    updater.write_json_atomic(job['host_file'], {'attempt_id':'one','ready':True})
    assert not updater.progress_host_ready(job)
    lock = updater.take_file_lock(job['host_lock'])
    try:
        assert updater.progress_host_ready(job)
        updater.write_json_atomic(job['host_file'], {'attempt_id':'other','ready':True})
        assert not updater.progress_host_ready(job)
    finally:
        lock.close()
    assert not updater.progress_host_ready(job)


def test_native_progress_launch_uses_immutable_files_and_no_terminal(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(update_launch.subprocess, 'Popen', lambda *a, **k: calls.append((a,k)))
    path = _job(tmp_path/"O'Neil ü", ['pixi','global','update','xdart'], marker=tmp_path/'reopened')
    job = json.loads(path.read_text()); job.update(manual_restart=True, require_progress_host=True)
    result = update_launch.launch_helper(path.parent, job, platform='darwin', python='/test/python')
    assert calls[0][0][0][:3] == ['/usr/bin/osascript','-l','JavaScript']
    assert job['host_file'] and Path(job['host_lock']).exists() and result == path
    source = path.with_suffix('.js').read_text()
    assert 'xdart._updater' in source and 'app.run' in source and 'Restart xdart' in source
    calls.clear()
    update_launch.launch_helper(path.parent, job, platform='win32')
    assert calls[0][1]['creationflags'] == update_launch.CREATE_NO_WINDOW | update_launch.CREATE_BREAKAWAY_FROM_JOB
    assert '-STA' in calls[0][0][0]
    assert path.with_suffix('.host.ps1').read_bytes().startswith(b'\xef\xbb\xbf')


@pytest.mark.skipif(sys.platform.startswith('win'), reason='POSIX engine')
def test_live_progress_host_allows_manual_update_and_released_host_refuses(tmp_path):
    ran = tmp_path/'updated'
    path = _job(tmp_path/'state', _script(f"open({str(ran)!r}, 'w').write('updated')"), marker=tmp_path/'reopened')
    job = json.loads(path.read_text())
    job.update(manual_restart=True, require_progress_host=True,
               host_file=str(tmp_path/'host.json'), host_lock=str(tmp_path/'host.lock'))
    updater.write_json_atomic(job['host_file'], {'attempt_id':job['attempt_id'],'ready':True})
    updater.write_json_atomic(path, job)
    lock = updater.take_file_lock(job['host_lock'])
    try:
        assert _updater.main([str(path)]) == 0 and ran.exists()
    finally:
        lock.close()
    ran.unlink()
    assert _updater.main([str(path)]) == 1 and not ran.exists()


def test_native_launch_refuses_unsupported_host_before_closing_xdart(tmp_path):
    job = {'attempt_id':'one','manual_restart':True}
    with pytest.raises(update_launch.UpdateLaunchRefused, match='manual command'):
        update_launch.launch_helper(tmp_path, job, platform='linux')


@pytest.mark.skipif(sys.platform != 'darwin', reason='headless system Foundation bridge on macOS')
def test_macos_foundation_tail_handles_utf8_and_bounds_bytes(tmp_path):
    # Executes the product's file-reading functions, with Foundation only: no AppKit, NSApplication or window.
    from xdart.modules.update_progress_macos import FOUNDATION
    path, missing = tmp_path/'unicode.log', tmp_path/'missing'
    path.write_text('x' + '€' * 60000, encoding='utf-8')
    code = FOUNDATION + '\nJSON.stringify({tail:logTail(' + json.dumps(str(path)) + '),missing:logTail(' + json.dumps(str(missing)) + '),text:text(' + json.dumps(str(missing)) + ')})'
    done = subprocess.run(['/usr/bin/osascript','-l','JavaScript','-e',code], capture_output=True, text=True, timeout=10)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout)
    assert result['missing'] is None and result['text'] == ''
    assert result['tail'] and set(result['tail']) == {'€'}
    assert len(result['tail'].encode('utf-8')) <= 131072


@pytest.mark.skipif(sys.platform != 'darwin', reason='osacompile is macOS')
def test_macos_host_script_compiles(tmp_path):
    # Parses the generated host without running it: no window.
    from xdart.modules import update_progress_macos
    host = tmp_path / 'host.js'
    host.write_text(update_progress_macos.script(tmp_path / "job O'Neil ü.json", '/env/bin/python'), encoding='utf-8')
    done = subprocess.run(['/usr/bin/osacompile', '-l', 'JavaScript', '-o', str(tmp_path / 'host.scpt'), str(host)],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS system frameworks')
def test_macos_host_names_resolve_without_an_application():
    # Reads the AppKit constants and the selector the host relies on; creates no NSApplication or window.
    code = ("ObjC.import('Cocoa'); JSON.stringify([$.NSProgressIndicatorStyleBar, $.NSBezelStyleRounded, "
            "$.NSApplicationActivationPolicyRegular, $.NSBackingStoreBuffered, $.NSUTF8StringEncoding, "
            "ObjC.unwrap($.NSRunLoopCommonModes), $.NSApplication.instancesRespondToSelector('activate')])")
    done = subprocess.run(['/usr/bin/osascript', '-l', 'JavaScript', '-e', code], capture_output=True, text=True,
                          timeout=30)
    assert done.returncode == 0, done.stderr
    values = json.loads(done.stdout)
    assert all(str(value).isdigit() for value in values[:5]), values      # the bridge gives enums as "0", "1", …
    assert isinstance(values[5], str) and values[5] and isinstance(values[6], bool), values


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS host lock through the system bridge')
def test_macos_host_lock_is_what_the_app_waits_for(tmp_path):
    # The host's own lock() (Foundation only, no window) holds the OS lock the app and the helper check.
    from xdart.modules.update_progress_macos import FOUNDATION
    lock_path, held = tmp_path / 'host.lock', tmp_path / 'held.json'
    lock_path.touch()
    job = dict(require_progress_host=True, attempt_id='one', host_file=str(tmp_path / 'host.json'),
               host_lock=str(lock_path))
    updater.write_json_atomic(job['host_file'], {'attempt_id': 'one', 'ready': True})
    path = json.dumps(str(lock_path))
    code = (FOUNDATION + f'\nvar fd = lock({path});\nsave({json.dumps(str(held))}, {{fd: fd, again: lock({path})}});\n'
            '$.NSThread.sleepForTimeInterval(20);')
    proc = subprocess.Popen(['/usr/bin/osascript', '-l', 'JavaScript', '-e', code],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _await(held, seconds=15)
        state = updater.read_json(held)
        assert state['fd'] >= 0 and state['again'] == -1          # a second window cannot take it
        assert updater.progress_host_ready(job)
    finally:
        proc.kill()      # always: a sleeping host would keep host.lock for 20 s into the next test
        proc.wait(10)
    assert not updater.progress_host_ready(job)


@pytest.mark.skipif(not sys.platform.startswith('win'), reason='Windows PowerShell 5.1 and WPF')
def test_windows_progress_window_script_parses_and_its_layout_loads(tmp_path):
    # Parses the generated host and builds its WPF window without showing it; nothing starts an update.
    from xdart.modules import update_progress_windows
    host = tmp_path / "host O'Neil ü.ps1"
    host.write_text(update_progress_windows.script(tmp_path / 'job.json', tmp_path / 'engine.ps1'),
                    encoding='utf-8-sig')
    check = tmp_path / 'check.ps1'
    check.write_text(r'''param([string]$path)
$errors = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$null, [ref]$errors)
if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }
Add-Type -AssemblyName PresentationFramework
$source = Get-Content -LiteralPath $path -Raw -Encoding UTF8
$match = [regex]::Match($source, "(?s)\[xml\]\`$xaml = @'\r?\n(.*?)\r?\n'@")
if (-not $match.Success) { 'no XAML'; exit 2 }
$window = [Windows.Markup.XamlReader]::Load((New-Object Xml.XmlNodeReader ([xml]$match.Groups[1].Value)))
foreach ($name in 'Version', 'Stage', 'Progress', 'Details', 'Restart', 'Close') {
  if ($null -eq $window.FindName($name)) { "missing $name"; exit 3 }
}
'ok'
''', encoding='utf-8-sig')
    exe = update_progress_windows.argv(host)[0]
    done = subprocess.run([exe, '-NoProfile', '-STA', '-ExecutionPolicy', 'Bypass', '-File', str(check), str(host)],
                          capture_output=True, text=True, timeout=120, creationflags=update_launch.CREATE_NO_WINDOW)
    assert done.returncode == 0 and done.stdout.strip().endswith('ok'), done.stdout + done.stderr


@pytest.mark.skipif(not sys.platform.startswith('win'), reason='Windows PowerShell helper')
def test_windows_helper_needs_its_progress_window_and_never_reopens_xdart(tmp_path):
    from tests.xdart.test_updater import _run_powershell, _windows_job
    ran = tmp_path / 'updated'
    job_path, state, marker = _windows_job(tmp_path, _script(f"open({str(ran)!r}, 'w').write('updated')"))
    job = json.loads(job_path.read_text(encoding='utf-8'))
    job.update(manual_restart=True, require_progress_host=True,
               host_file=str(tmp_path / 'host.json'), host_lock=str(tmp_path / 'host.lock'))
    updater.write_json_atomic(job['host_file'], {'attempt_id': job['attempt_id'], 'ready': True})
    job_path.write_text(json.dumps(job), encoding='utf-8')
    lock = updater.take_file_lock(job['host_lock'])               # held unshared, as the WPF window holds it
    try:
        done = _run_powershell(job_path)
    finally:
        lock.close()
    assert done.returncode == 0 and ran.exists(), done.stdout + done.stderr
    result = json.loads((state / updater.HELPER_FILE).read_text(encoding='utf-8-sig'))
    assert (result['state'], result['exit'], result['relaunched']) == ('completed', 0, False)
    assert not _await(marker, seconds=2)
    ran.unlink()
    done = _run_powershell(job_path)                              # the window has gone: nothing changes
    assert done.returncode == 1 and not ran.exists(), done.stdout + done.stderr
    log = (state / 'update.log').read_text(encoding='utf-8-sig')
    assert 'progress window closed before installation, so nothing was changed' in log
