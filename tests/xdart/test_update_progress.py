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
