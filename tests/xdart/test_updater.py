# -*- coding: utf-8 -*-
"""In-app updater (Help -> Check for Updates...) — pure logic + the update-on-exit
helper.  No Qt, no network, no real pixi: every external edge is injected.
Spec: docs/design/design_install_and_update_jul2026.md section 4.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from xdart.modules import updater
from xdart import _updater as helper
from xdart.modules import update_launch


# ── install_meta discovery ────────────────────────────────────────────────────

def _write_meta(path, **extra):
    path.write_text(json.dumps({"flavor": "pixi-workspace", "app_root": str(path.parent),
                                "update_cmd": ["pixi", "update"],
                                "relaunch_cmd": ["xdart"], **extra}))


def test_find_install_meta_walks_up_from_prefix(tmp_path):
    # meta at APP_ROOT; sys.prefix is the nested pixi env dir.
    _write_meta(tmp_path / "install_meta.json")
    env = tmp_path / ".pixi" / "envs" / "default"
    env.mkdir(parents=True)
    meta = updater.find_install_meta(prefix=str(env))
    assert meta is not None and meta["flavor"] == "pixi-workspace"


def test_find_install_meta_absent(tmp_path):
    assert updater.find_install_meta(prefix=str(tmp_path)) is None


def test_find_install_meta_env_override_dir_and_file(tmp_path):
    _write_meta(tmp_path / "install_meta.json")
    # override as a directory
    assert updater.find_install_meta(prefix="/nonexistent",
                                     env_override=str(tmp_path)) is not None
    # override as the file itself
    assert updater.find_install_meta(prefix="/nonexistent",
                                     env_override=str(tmp_path / "install_meta.json")) is not None


def test_find_install_meta_malformed_is_none(tmp_path):
    (tmp_path / "install_meta.json").write_text("{ not json")
    assert updater.find_install_meta(prefix=str(tmp_path)) is None


# ── install_kind classification ───────────────────────────────────────────────

def test_install_kind_editable(monkeypatch):
    monkeypatch.setattr(updater, "is_editable_install", lambda: True)
    assert updater.install_kind(meta={"app_root": "x"}) == "editable"


def test_install_kind_pixi_when_meta_present(monkeypatch):
    monkeypatch.setattr(updater, "is_editable_install", lambda: False)
    monkeypatch.setattr(updater, "detect_pixi_global_env", lambda: False)
    assert updater.install_kind(meta={"app_root": "x"}) == "pixi"


def test_install_kind_managed_when_no_meta(monkeypatch):
    monkeypatch.setattr(updater, "is_editable_install", lambda: False)
    monkeypatch.setattr(updater, "find_install_meta", lambda: None)
    monkeypatch.setattr(updater, "detect_pixi_global_env", lambda: False)
    assert updater.install_kind() == "managed"


# ── pixi-global flavor (v1.1 conda package) ───────────────────────────────────

def test_detect_pixi_global_env_by_path(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXI_HOME", raising=False)
    # <root>/.pixi/envs/xrd-tools  -> pixi-global
    good = tmp_path / ".pixi" / "envs" / "xrd-tools"
    good.mkdir(parents=True)
    assert updater.detect_pixi_global_env(prefix=str(good)) is True
    # a normal venv/conda prefix -> not pixi-global
    other = tmp_path / "envs2" / "myenv"
    other.mkdir(parents=True)
    assert updater.detect_pixi_global_env(prefix=str(other)) is False


def test_detect_pixi_global_env_honors_pixi_home(tmp_path, monkeypatch):
    home = tmp_path / "custom-pixi"
    env = home / "envs" / "xrd-tools"
    env.mkdir(parents=True)
    monkeypatch.setenv("PIXI_HOME", str(home))
    assert updater.detect_pixi_global_env(prefix=str(env)) is True


def test_install_kind_pixi_global(monkeypatch):
    monkeypatch.setattr(updater, "is_editable_install", lambda: False)
    monkeypatch.setattr(updater, "find_install_meta", lambda: None)
    monkeypatch.setattr(updater, "detect_pixi_global_env", lambda: True)
    assert updater.install_kind() == "pixi-global"


def test_install_kind_prefers_install_meta_over_the_path_probe(monkeypatch):
    # The script installer's workspace env lives under <APP_ROOT>/.pixi/envs/default, which the pixi-global path
    # probe also accepts; its install_meta.json decides.
    monkeypatch.setattr(updater, "is_editable_install", lambda: False)
    monkeypatch.setattr(updater, "detect_pixi_global_env", lambda: True)
    assert updater.install_kind(meta={"app_root": "x"}) == "pixi"


def _pixi_home(tmp_path, *, env="xdart-test", exposed='{ xdart = "xdart" }', spec='"*"',
               channels='["https://prefix.dev/xrd-tools", "conda-forge"]', exe=""):
    """A pixi home with one global env, its manifest entry, and the pixi and trampoline executables."""
    home = tmp_path / ".pixi"
    env_dir = home / "envs" / env
    env_dir.mkdir(parents=True)
    (home / "manifests").mkdir()
    (home / "bin").mkdir()
    (home / "manifests" / "pixi-global.toml").write_text(
        f'version = 1\n\n[envs.{env}]\nchannels = {channels}\ndependencies = {{ xdart = {spec} }}\n'
        f'exposed = {exposed}\n', encoding="utf-8")
    for name in ("pixi", "xdart"):
        (home / "bin" / f"{name}{exe}").write_bytes(b"")
    return home.resolve(), env_dir


def test_pixi_global_meta_builds_update_command(monkeypatch, tmp_path):
    # The home comes from the running env, not $PIXI_HOME: an install under another home updates that home.
    monkeypatch.setenv("PIXI_HOME", str(tmp_path / "elsewhere"))
    home, env = _pixi_home(tmp_path)
    meta = updater.pixi_global_meta(prefix=str(env), platform="darwin")
    assert meta["flavor"] == "pixi-global"
    assert meta["update_cmd"] == [str(home / "bin" / "pixi"), "global", "update", "xdart-test"]
    assert meta["update_env"] == {"PIXI_HOME": str(home)}
    assert meta["relaunch_cmd"] == [str(home / "bin" / "xdart")]  # the trampoline, not the env's script
    assert meta["env_prefix"] == str(home / "envs" / "xdart-test")
    assert updater.update_state_dir(meta) == home / "xdart-update" / "xdart-test"


def test_pixi_global_meta_relaunches_the_exposed_command_or_refuses(tmp_path):
    home, env = _pixi_home(tmp_path, exposed='{ xd = "xdart", py = "python" }')
    (home / "bin" / "xd").write_bytes(b"")
    assert updater.pixi_global_meta(prefix=str(env), platform="darwin")["relaunch_cmd"] == [str(home / "bin" / "xd")]
    (home / "manifests" / "pixi-global.toml").write_text(
        'version = 1\n\n[envs.xdart-test]\nchannels = ["https://prefix.dev/xrd-tools"]\nexposed = { py = "python" }\n')
    assert updater.pixi_global_meta(prefix=str(env), platform="darwin") is None   # nothing exposed runs xdart
    (home / "manifests" / "pixi-global.toml").write_text("not toml [")
    assert updater.pixi_global_meta(prefix=str(env), platform="darwin") is None   # unreadable: unknown state


def test_pixi_global_meta_on_windows_needs_both_executables(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setenv("PIXI_HOME", str(tmp_path / "no-default-home"))
    home, env = _pixi_home(tmp_path, env="xdart", exe=".exe")
    meta = updater.pixi_global_meta(prefix=str(env), platform="win32")
    assert meta["update_cmd"] == [str(home / "bin" / "pixi.exe"), "global", "update", "xdart"]
    assert meta["relaunch_cmd"] == [str(home / "bin" / "xdart.exe")]
    (home / "bin" / "pixi.exe").unlink()
    assert updater.pixi_global_meta(prefix=str(env), platform="win32") is None   # no pixi.exe anywhere
    (home / "bin" / "pixi.exe").write_bytes(b"")
    (home / "bin" / "xdart.exe").unlink()
    assert updater.pixi_global_meta(prefix=str(env), platform="win32") is None   # no trampoline


def test_copyable_command_never_offers_pip_for_a_pixi_install(tmp_path, monkeypatch):
    home, env = _pixi_home(tmp_path, env="xdart")
    monkeypatch.setenv("PIXI_HOME", str(home))
    assert updater.copyable_update_command("pixi-global", prefix=str(env)) == "pixi global update xdart"
    monkeypatch.setenv("PIXI_HOME", str(tmp_path / "default-home"))
    other_home = (f"$env:PIXI_HOME = '{home}'; pixi global" if sys.platform.startswith("win")   # PowerShell
                  else f'PIXI_HOME="{home}" pixi global')
    assert updater.copyable_update_command("pixi-global", prefix=str(env)) == f"{other_home} update xdart"
    assert updater.pixi_global_command("sync", prefix=str(env)) == f"{other_home} sync"
    assert updater.copyable_update_command("managed") == 'pip install -U "xdart[gui]"'


def test_check_for_update_reads_the_install_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, "current_version", lambda: "1.2.5")
    payload = _REPODATA.read_bytes()

    def check(**manifest):
        root = tmp_path / str(len(list(tmp_path.iterdir())))
        root.mkdir()
        _home, env = _pixi_home(root, **manifest)
        return updater.check_for_update("pixi-global", prefix=str(env), opener=lambda url, _t: payload)

    plain = check()
    assert (plain.status, plain.latest, plain.automatic) == ("ok", "1.2.6", True)
    assert check(spec='">=1.2,<1.3"').automatic is False                       # pinned: Help only
    assert check(channels='["https://prefix.dev/xrd-tools", "my-mirror"]').automatic is False  # unknown authority
    assert check(exposed='{ py = "python" }').automatic is False                # no trampoline for xdart
    assert check(channels='["conda-forge"]').status == "unsupported"            # no channel URL for xdart


def test_check_for_update_tells_unreachable_from_absent(tmp_path):
    _home, env = _pixi_home(tmp_path)

    def down(_url, _timeout):
        raise OSError("offline")

    assert updater.check_for_update("pixi-global", prefix=str(env), opener=down).status == "unreachable"
    empty = updater.check_for_update("pixi-global", prefix=str(env), opener=lambda _u, _t: b'{"packages": {}}')
    assert (empty.status, empty.latest) == ("absent", None)
    (tmp_path / ".pixi" / "manifests" / "pixi-global.toml").unlink()
    assert updater.check_for_update("pixi-global", prefix=str(env), opener=down).status == "unsupported"


def test_check_for_update_workspace_is_automatic_within_its_major(monkeypatch):
    # The script installer's workspace asks for xdart ">=1,<2" from PyPI.
    monkeypatch.setattr(updater, "current_version", lambda: "1.2.6")
    meta = {"app_root": "x", "update_cmd": ["pixi", "update"], "relaunch_cmd": ["xdart"]}

    def pypi(version):
        return lambda _url, _t: json.dumps({"info": {"version": version}}).encode()

    assert updater.check_for_update("pixi", meta=meta, opener=pypi("1.2.7")).automatic is True
    assert updater.check_for_update("pixi", meta=meta, opener=pypi("2.0.0")).automatic is False
    assert updater.check_for_update("managed", opener=pypi("1.2.7")).automatic is False


def test_resolve_update_meta_by_kind(monkeypatch):
    monkeypatch.setattr(updater, "pixi_global_meta", lambda: {"flavor": "pixi-global"})
    monkeypatch.setattr(updater, "find_install_meta", lambda: {"flavor": "pixi-workspace"})
    assert updater.resolve_update_meta("pixi-global")["flavor"] == "pixi-global"
    assert updater.resolve_update_meta("pixi")["flavor"] == "pixi-workspace"
    assert updater.resolve_update_meta("managed") is None
    assert updater.resolve_update_meta("editable") is None


# ── PyPI fetch (injected opener; never touches the network) ────────────────────

def test_fetch_latest_pypi_ok():
    payload = json.dumps({"info": {"version": "1.2.3"}}).encode()
    assert updater.fetch_latest_pypi(opener=lambda url, t: payload) == "1.2.3"


def test_fetch_latest_pypi_offline_returns_none():
    def boom(url, t):
        raise OSError("offline")
    assert updater.fetch_latest_pypi(opener=boom) is None


def test_fetch_latest_pypi_malformed_returns_none():
    assert updater.fetch_latest_pypi(opener=lambda url, t: b"<html>nope") is None


# ── conda channel (pixi-global): its own channel, a named User-Agent ─────────────

_REPODATA = Path(__file__).with_name("prefix_dev_repodata_sample.json")  # trimmed prefix.dev index, 2026-10-27


def test_latest_from_repodata_reads_the_saved_prefix_dev_index():
    assert updater.latest_from_repodata(json.loads(_REPODATA.read_text())) == "1.2.6"


def test_latest_from_repodata_skips_other_names_prereleases_and_bad_versions():
    data = json.loads(_REPODATA.read_text())
    data["packages.conda"]["xdart-1.3.0rc1-0.conda"] = {"name": "xdart", "version": "1.3.0rc1"}
    data["packages.conda"]["other-9.0-0.conda"] = {"name": "other", "version": "9.0"}
    data["packages.conda"]["xdart-bad-0.conda"] = {"name": "xdart", "version": "not a version"}
    data["packages"]["xdart-1.2.7-0.tar.bz2"] = {"name": "xdart", "version": "1.2.7"}  # the legacy key counts
    assert updater.latest_from_repodata(data) == "1.2.7"
    assert updater.latest_from_repodata(data, include_prereleases=True) == "1.3.0rc1"
    assert updater.latest_from_repodata({"packages": {}}) is None
    assert updater.latest_from_repodata("not a dict") is None


def test_fetch_latest_conda_asks_channels_in_order_and_never_raises():
    seen, payload = [], _REPODATA.read_bytes()

    def opener(url, _timeout):
        seen.append(url)
        if "down" in url:
            raise OSError("offline")
        return json.dumps({"packages": {}}).encode() if "empty" in url else payload

    channels = ["https://down.example/a", "https://empty.example/b/", "https://prefix.dev/xrd-tools"]
    assert updater.fetch_latest_conda(channels, opener=opener) == "1.2.6"
    assert seen == [f"{channel.rstrip('/')}/noarch/repodata.json" for channel in channels]
    assert updater.fetch_latest_conda(["https://down.example/a"], opener=opener) is None


def test_pixi_global_channels_come_from_the_manifest(tmp_path):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "pixi-global.toml").write_text(
        'version = 1\n\n[envs.xdart]\nchannels = ["conda-forge", "https://prefix.dev/xrd-tools/", "file:///tmp/ch"]\n'
        'dependencies = { xdart = "*" }\n')
    assert updater.pixi_global_channels("xdart", tmp_path) == ["https://prefix.dev/xrd-tools", "file:///tmp/ch"]
    assert updater.pixi_global_channels("other", tmp_path) == [updater.DEFAULT_CONDA_CHANNEL]
    assert updater.pixi_global_channels("xdart", tmp_path / "missing") == [updater.DEFAULT_CONDA_CHANNEL]


def test_fetch_latest_for_routes_pixi_global_to_its_own_channel(tmp_path, monkeypatch):
    home = tmp_path / ".pixi"
    env = home / "envs" / "xdart"
    env.mkdir(parents=True)
    (home / "manifests").mkdir()
    (home / "manifests" / "pixi-global.toml").write_text(
        'version = 1\n\n[envs.xdart]\nchannels = ["file:///chan", "conda-forge"]\n')
    monkeypatch.setattr(updater, "current_version", lambda: "1.2.5")
    urls = []

    def opener(url, _timeout):
        urls.append(url)
        return _REPODATA.read_bytes() if url.endswith("repodata.json") else b'{"info": {"version": "1.2.9"}}'

    assert updater.fetch_latest_for("pixi-global", opener=opener, prefix=str(env)) == "1.2.6"
    assert updater.fetch_latest_for("pixi", opener=opener) == "1.2.9"
    assert updater.fetch_latest_for("managed", opener=opener) == "1.2.9"
    assert urls == ["file:///chan/noarch/repodata.json", updater.PYPI_JSON_URL, updater.PYPI_JSON_URL]


def test_conda_fetch_follows_the_signed_redirect_with_a_named_agent():
    # prefix.dev answers 303 to a signed packages.prefix.dev URL that refuses Python's default agent (HTTP 403,
    # measured 2026-10-27). A local server reproduces both; the real urllib path runs.
    import http.server
    import threading
    import urllib.error
    import urllib.request
    payload, agents = _REPODATA.read_bytes(), []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            agent = self.headers.get("User-Agent", "")
            agents.append(agent)
            if self.path == "/xrd-tools/noarch/repodata.json":
                self.send_response(303)
                self.send_header("Location", "/signed/repodata.json?verify=token")
                self.end_headers()
            elif agent.startswith("Python-urllib"):
                self.send_response(403)
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        def log_message(self, format, *args):  # noqa: A002 - the base signature
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/xrd-tools"
    try:
        with pytest.raises(urllib.error.HTTPError, match="403"):
            urllib.request.urlopen(f"{base}/noarch/repodata.json", timeout=3)  # Python's default agent
        agents.clear()
        assert updater.fetch_latest_conda([base]) == "1.2.6"
        assert len(agents) == 2 and all(agent.startswith("xdart/") for agent in agents)
    finally:
        server.shutdown()
        server.server_close()


# ── version comparison ────────────────────────────────────────────────────────

@pytest.mark.parametrize("cur,latest,expected", [
    ("1.0.0", "1.0.1", True),
    ("1.0.0", "1.0.0", False),
    ("1.0.1", "1.0.0", False),
    ("1.0.0", "2.0.0", True),
    ("1.0.0", "1.1.0rc1", False),     # stable user not nudged to a pre-release
    ("1.1.0rc1", "1.1.0rc2", True),   # pre-release user may move forward
    ("1.0.0", None, False),
    (None, "1.0.1", False),
    ("1.0.0", "garbage", False),
])
def test_update_available(cur, latest, expected):
    assert updater.update_available(cur, latest) is expected


# ── update-on-exit helper ─────────────────────────────────────────────────────

def test_wait_for_exit_returns_true_when_pid_gone():
    calls = {"n": 0}

    def alive(pid):
        calls["n"] += 1
        return calls["n"] < 3          # alive for 2 polls, then gone

    assert helper.wait_for_exit(1234, alive=alive, sleep=lambda s: None) is True


def test_wait_for_exit_times_out():
    assert helper.wait_for_exit(
        1234, timeout=1.0, interval=0.5,
        alive=lambda pid: True, sleep=lambda s: None) is False


# ── the update helper (real processes: stand-in update commands, a real relaunch) ──────────────────────────────

posix_only = pytest.mark.skipif(sys.platform.startswith("win"), reason="the POSIX helper; Windows runs PowerShell")


def _script(body):
    return [sys.executable, "-c", body]


def _job(state, update_cmd, *, marker, parent_pid=0, limit=60, grace=2, tree_wait=10, parent_wait=5):
    state.mkdir(parents=True, exist_ok=True)
    updater.begin_attempt(state, install="test-install", current="1.2.6", target="1.2.7", log=state / "update.log")
    attempt = updater.load_update_state(state, "test-install")["attempt"]
    updater.authorize_attempt(state, "test-install", attempt["id"])
    job = {"parent_pid": parent_pid, "install": "test-install", "from": "1.2.6", "target": "1.2.7",
           "attempt_id": attempt["id"],
           "update_cmd": update_cmd, "update_env": {"XDART_TEST_UPDATE_ENV": "set"},
           "relaunch_cmd": _script(f"open({str(marker)!r}, 'w').write('relaunched')"),
           "env_prefix": str(state), "log": str(state / "update.log"), "time_limit_s": limit, "grace_s": grace,
           "tree_wait_s": tree_wait, "parent_wait_s": parent_wait}
    return update_launch.write_job(state, job)


def _await(path, seconds=10.0):
    import time
    end = time.monotonic() + seconds
    while time.monotonic() < end and not path.exists():
        time.sleep(0.05)
    return path.exists()


def _helper_result(state):
    return json.loads((state / updater.HELPER_FILE).read_text())


@posix_only
def test_helper_streams_the_update_and_relaunches_after_success(tmp_path):
    state, marker = tmp_path / "state", tmp_path / "relaunched"
    job = _job(state, _script("import os, time; print('step 1', flush=True); time.sleep(0.3); "
                              "print('env', os.environ['XDART_TEST_UPDATE_ENV'], flush=True)"), marker=marker)
    assert helper.main([str(job)]) == 0
    log = (state / "update.log").read_text()
    assert "Updating xdart from 1.2.6 to 1.2.7" in log and " step 1\n" in log and " env set\n" in log
    result = _helper_result(state)
    assert (result["state"], result["exit"], result["timed_out"], result["relaunched"]) == ("completed", 0, False, True)
    assert _await(marker)                                   # the detached relaunch ran


@posix_only
def test_helper_time_limit_stops_children_first_then_relaunches(tmp_path):
    state, marker = tmp_path / "state", tmp_path / "relaunched"
    pixi = ("import subprocess, sys, time; "
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "print('child started', flush=True); rc = child.wait(); "
            "print(f'child ended rc={rc}', flush=True); sys.exit(3)")
    job = _job(state, _script(pixi), marker=marker, limit=1, grace=10)
    assert helper.main([str(job)]) == 1
    log = (state / "update.log").read_text()
    lines = [line.split(" ", 2)[2] for line in log.splitlines()]   # drop the timestamp; the command echo is one line

    def at(prefix):
        return next(i for i, line in enumerate(lines) if line.startswith(prefix))

    order = [at("child started"), at("The update is still running"), at("Stopped 1 child process"),
             at("child ended rc=")]
    assert order == sorted(order) and "pixi did not finish" not in log
    result = _helper_result(state)
    assert (result["exit"], result["timed_out"], result["relaunched"]) == (3, True, True)
    assert _await(marker)


@posix_only
def test_helper_escalates_when_pixi_ignores_the_stop(tmp_path):
    state, marker = tmp_path / "state", tmp_path / "relaunched"
    pixi = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('busy', flush=True); time.sleep(60)"
    job = _job(state, _script(pixi), marker=marker, limit=1, grace=1)
    assert helper.main([str(job)]) == 1
    log = (state / "update.log").read_text()
    assert "pixi did not finish; stopping the whole update." in log
    result = _helper_result(state)
    assert result["timed_out"] is True and result["relaunched"] is True and result["exit"] == -9
    assert _await(marker)


@posix_only
def test_helper_leaves_xdart_closed_while_part_of_the_update_still_runs(tmp_path):
    import os
    import signal
    state, marker = tmp_path / "state", tmp_path / "relaunched"
    pixi = ("import subprocess, sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
            "print('done, leaving a child', flush=True)")
    job = _job(state, _script(pixi), marker=marker, tree_wait=1)
    try:
        assert helper.main([str(job)]) == 1
        result = _helper_result(state)
        assert result["exit"] == 0 and result["relaunched"] is False and result["refused"] == "update still running"
        assert "xdart was not reopened" in (state / "update.log").read_text()
        assert not marker.exists()
    finally:
        try:
            os.killpg(_helper_result(state)["update_pid"], signal.SIGKILL)
        except (ProcessLookupError, KeyError, PermissionError):
            pass


@posix_only
def test_helper_changes_nothing_when_xdart_does_not_close(tmp_path):
    import subprocess
    state, marker, ran = tmp_path / "state", tmp_path / "relaunched", tmp_path / "update-ran"
    xdart = subprocess.Popen(_script("import time; time.sleep(30)"))
    try:
        job = _job(state, _script(f"open({str(ran)!r}, 'w').write('ran')"), marker=marker,
                   parent_pid=xdart.pid, parent_wait=1)
        assert helper.main([str(job)]) == 1
        assert _helper_result(state)["refused"] == "xdart did not close"
        assert not ran.exists() and not marker.exists()
    finally:
        xdart.kill()
        xdart.wait()


@posix_only
def test_command_wrapper_passes_a_hard_path_to_the_helper(tmp_path):
    # The Terminal wrapper's only argument is the job path: spaces, an apostrophe and non-ASCII must survive a real sh.
    import subprocess
    state, marker = tmp_path / "Jane O'Neil ü" / "xdart-update", tmp_path / "relaunched"
    job = _job(state, _script("print('ok', flush=True)"), marker=marker)
    wrapper = update_launch.write_command_wrapper(state, sys.executable, job)
    assert os.access(wrapper, os.X_OK)
    env = {**os.environ, "PYTHONPATH": str(Path(updater.__file__).resolve().parents[2])}
    done = subprocess.run(["/bin/sh", str(wrapper)], capture_output=True, text=True, timeout=60, env=env)
    assert done.returncode == 0, done.stderr
    assert _helper_result(state)["exit"] == 0 and _await(marker)


def test_launch_macos_opens_the_wrapper_in_terminal(tmp_path):
    seen = []

    class _Done:
        returncode, stderr = 0, ""

    job = tmp_path / "job.json"
    wrapper = update_launch.launch_macos(tmp_path, "/env/bin/python", job, runner=lambda argv, **_k: seen.append(argv) or _Done())
    assert seen == [["/usr/bin/open", "-a", "Terminal", str(wrapper)]]
    assert wrapper.read_text().endswith(f"exec '/env/bin/python' -m xdart._updater '{job}'\n")
    _Done.returncode, _Done.stderr = 1, "no Terminal"
    with pytest.raises(update_launch.UpdateLaunchRefused, match="no Terminal"):
        update_launch.launch_macos(tmp_path, "/env/bin/python", job, runner=lambda argv, **_k: _Done())


def test_windows_launch_breaks_away_from_the_job_or_refuses(tmp_path):
    seen = []
    job = tmp_path / "Jane O'Neil ü" / "job.json"
    update_launch.launch_windows(job, popen=lambda argv, **kwargs: seen.append((argv, kwargs)))
    (argv, kwargs), = seen
    assert argv[1:5] == ["-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand"]
    assert kwargs["creationflags"] & update_launch.CREATE_BREAKAWAY_FROM_JOB
    assert kwargs["creationflags"] & update_launch.CREATE_NEW_CONSOLE
    import base64
    bootstrap = base64.b64decode(argv[5]).decode("utf-16-le")
    assert bootstrap == "& " + update_launch.ps_literal(job.with_suffix(".ps1"))
    script = job.with_suffix(".ps1").read_text(encoding="utf-8-sig")
    assert script == update_launch.powershell_script(job)
    assert script.count(update_launch.ps_literal(job)) == 1 and "__JOB__" not in script
    assert update_launch.ps_literal("C:\\Users\\Jane O'Neil\\ü\\job.json") == "'C:\\Users\\Jane O''Neil\\ü\\job.json'"

    def refuse(_argv, **_kwargs):
        raise PermissionError(5, "Access is denied")

    with pytest.raises(update_launch.UpdateLaunchRefused, match="outside xdart"):
        update_launch.launch_windows(job, popen=refuse)


# ── per-install update state ──────────────────────────────────────────────────────────────────────────────────

def test_an_attempt_settles_by_version(tmp_path):
    def attempt(current_at_settle, helper=None):
        state = tmp_path / current_at_settle
        updater.begin_attempt(state, install="I", current="1.2.6", target="1.2.7", log=state / "update.log")
        if helper is not None:
            updater.write_json_atomic(state / updater.HELPER_FILE, helper)
        return state, updater.settle_previous_attempt(state, install="I", current=current_at_settle,
                                                     alive=lambda _pid: False)

    state, (outcome, record) = attempt("1.2.7", {"pid": 1, "state": "completed", "exit": 0})
    assert outcome == "updated" and record["to"] == "1.2.7"
    assert updater.load_update_state(state, "I")["attempt"] is None and not (state / updater.HELPER_FILE).exists()
    assert attempt("1.2.8")[1][0] == "updated"              # a newer install is not a failure of the older target
    state, (outcome, _record) = attempt("1.2.6", {"pid": 1, "state": "completed", "exit": 0})
    assert outcome == "incomplete" and updater.target_failed(state, "I", "1.2.7")  # exit 0, version unchanged
    assert attempt("1.2.6-dead", {"pid": 99999999, "state": "running"})[1][0] == "incomplete"
    assert updater.settle_previous_attempt(tmp_path / "none", install="I", current="1", alive=bool) == ("none", None)


def hold_update_lock(state_dir):
    """A stand-in helper: another process holding this install's update lock (POSIX flock), as the helper does."""
    import subprocess
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    holder = subprocess.Popen(
        [sys.executable, "-c", "import fcntl, sys, time; f = open(sys.argv[1], 'a+b'); fcntl.flock(f, fcntl.LOCK_EX); "
         "print('locked', flush=True); time.sleep(60)", str(Path(state_dir) / updater.LOCK_FILE)],
        stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "locked"
    return holder


@posix_only
def test_a_running_helper_keeps_the_attempt(tmp_path):
    updater.begin_attempt(tmp_path, install="I", current="1.2.6", target="1.2.7", log=tmp_path / "update.log")
    holder = hold_update_lock(tmp_path)
    try:
        updater.write_json_atomic(tmp_path / updater.HELPER_FILE, {"pid": holder.pid, "state": "running"})
        assert updater.helper_lock_held(tmp_path) is True
        outcome, record = updater.settle_previous_attempt(tmp_path, install="I", current="1.2.6", alive=lambda _p: True)
        assert outcome == "active" and record["to"] == "1.2.7"
        assert updater.load_update_state(tmp_path, "I")["attempt"]["to"] == "1.2.7"   # untouched
    finally:
        holder.kill()
        holder.wait()
    assert updater.helper_lock_held(tmp_path) is False                              # the lock dies with it


@posix_only
def test_a_reused_pid_does_not_fake_a_running_helper(tmp_path):
    # The helper died without a record; its PID now belongs to a live process. Only the lock decides.
    updater.begin_attempt(tmp_path, install="I", current="1.2.6", target="1.2.7", log=tmp_path / "update.log")
    (tmp_path / updater.LOCK_FILE).write_bytes(b"")
    updater.write_json_atomic(tmp_path / updater.HELPER_FILE, {"pid": os.getpid(), "state": "running"})
    outcome, _record = updater.settle_previous_attempt(tmp_path, install="I", current="1.2.6", alive=lambda _p: True)
    assert outcome == "incomplete"


def test_a_late_helper_after_an_abandoned_attempt_is_ignored(tmp_path):
    # The handshake timed out and the attempt was undone; the helper started late and wrote its record anyway.
    previous = updater.begin_attempt(tmp_path, install="I", current="1.2.6", target="1.2.7", log="l")
    updater.abandon_attempt(tmp_path, previous)
    updater.write_json_atomic(tmp_path / updater.HELPER_FILE, {"pid": os.getpid(), "state": "running"})
    assert updater.settle_previous_attempt(tmp_path, install="I", current="1.2.6", alive=lambda _p: True) == (
        "none", None)


@posix_only
def test_helper_changes_nothing_while_another_helper_holds_the_lock(tmp_path):
    state, marker, ran = tmp_path / "state", tmp_path / "relaunched", tmp_path / "update-ran"
    job = _job(state, _script(f"open({str(ran)!r}, 'w').write('ran')"), marker=marker)
    holder = hold_update_lock(state)
    try:
        assert helper.main([str(job)]) == 2
        assert not (state / updater.HELPER_FILE).exists() and not ran.exists() and not marker.exists()
    finally:
        holder.kill()
        holder.wait()


def test_abandon_restores_the_state_and_installs_do_not_share_it(tmp_path):
    previous = updater.begin_attempt(tmp_path, install="I", current="1.2.6", target="1.2.7", log="l")
    updater.abandon_attempt(tmp_path, previous)
    assert updater.load_update_state(tmp_path, "I")["attempt"] is None
    updater.begin_attempt(tmp_path, install="I", current="1.2.6", target="1.2.7", log="l")
    assert updater.load_update_state(tmp_path, "another install")["attempt"] is None


def test_update_check_runs_off_the_gui_thread():
    # The network check never runs on the GUI thread; its result crosses back through a queued signal.
    import threading
    from pyqtgraph.Qt import QtCore, QtWidgets
    from xdart import _gui_main
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    threads, delivered = [], []
    relay = _gui_main._UpdateCheckRelay()
    relay.result_ready.connect(lambda result, generation: delivered.append((result.status, generation,
                                                                              threading.current_thread())))
    original = updater.check_for_update
    updater.check_for_update = lambda kind: threads.append(threading.current_thread()) or updater.UpdateCheck("ok", "9")
    try:
        worker = threading.Thread(target=_gui_main._run_update_check, args=(relay, "pixi-global", 7), daemon=True)
        worker.start()
        worker.join(5)
        for _ in range(100):
            app.processEvents()
            if delivered:
                break
    finally:
        updater.check_for_update = original
    assert threads and threads[0] is not threading.main_thread()
    assert delivered == [("ok", 7, threading.main_thread())]
    assert not isinstance(relay, QtCore.QThread)


def test_updater_helper_imports_no_qt():
    # S2: `python -m xdart._updater` must pull ZERO Qt (importing it runs
    # xdart/__init__ + the stdlib-only helper) -- otherwise the update locks
    # .pyd/.dll on Windows.  Checked in a fresh subprocess (this process already
    # has Qt loaded from other tests).
    import subprocess
    code = ("import sys, xdart._updater; "
            "print([m for m in sys.modules "
            "if 'PySide' in m or 'shiboken' in m or 'pyqtgraph' in m])")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", "helper pulled Qt: " + out.stdout


# ── the Windows helper, run for real where PowerShell exists (the windows-pixi-native job) ─────────────────────

windows_only = pytest.mark.skipif(not sys.platform.startswith("win"), reason="runs the real Windows PowerShell helper")
#: Stands in for pixi's xdart.exe launcher, which the helper waits for; never the interpreter running these tests.
_PYTHONW = str(Path(sys.executable).with_name("pythonw.exe"))


def _windows_job(tmp_path, update_cmd, *, env_prefix=None, limit=60, grace=10, env_wait=10):
    state, marker = tmp_path / "Jane O'Neil ü" / "xdart-update", tmp_path / "relaunched"
    job = _job(state, update_cmd, marker=marker, limit=limit, grace=grace)
    data = json.loads(job.read_text(encoding="utf-8"))
    assert Path(_PYTHONW).is_file(), _PYTHONW
    data["relaunch_cmd"][0] = _PYTHONW
    data.update(pause_at_end=False, env_prefix=env_prefix or str(tmp_path / "no-env"), env_wait_s=env_wait)
    job.write_text(json.dumps(data), encoding="utf-8")
    return job, state, marker


def _run_powershell(job):
    import subprocess
    return subprocess.run(update_launch.powershell_argv(job), capture_output=True, text=True, timeout=180)


@windows_only
def test_powershell_helper_runs_the_update_and_relaunches(tmp_path):
    job, state, marker = _windows_job(tmp_path, _script(
        "import os; print('step 1', flush=True); print('env', os.environ['XDART_TEST_UPDATE_ENV'], flush=True)"))
    done = _run_powershell(job)
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads((state / updater.HELPER_FILE).read_text(encoding="utf-8-sig"))
    assert (result["state"], result["exit"], result["relaunched"]) == ("completed", 0, True)
    log = (state / "update.log").read_text(encoding="utf-8-sig")
    assert " step 1\n" in log and " env set\n" in log          # whole lines: the logged command holds "step 1"
    assert "with a console window instead" not in log           # reopened without a console window
    assert _await(marker, 30)


@windows_only
def test_powershell_helper_time_limit_stops_the_child_then_relaunches(tmp_path):
    pixi = ("import subprocess, sys; child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "print('child started', flush=True); rc = child.wait(); print('child ended', flush=True); sys.exit(3)")
    job, state, marker = _windows_job(tmp_path, _script(pixi), limit=3, grace=20)
    done = _run_powershell(job)
    assert done.returncode == 1, done.stdout + done.stderr
    result = json.loads((state / updater.HELPER_FILE).read_text(encoding="utf-8-sig"))
    assert (result["exit"], result["timed_out"], result["relaunched"]) == (3, True, True)
    log = (state / "update.log").read_text(encoding="utf-8-sig")
    assert "Stopped 1 child process" in log and " child ended\n" in log and "pixi did not finish" not in log
    assert _await(marker, 30)


@windows_only
def test_powershell_helper_refuses_while_the_environment_is_in_use(tmp_path):
    ran = tmp_path / "update-ran"
    job, state, marker = _windows_job(tmp_path, _script(f"open({str(ran)!r}, 'w').write('ran')"),
                                      env_prefix=sys.prefix, env_wait=1)   # this pytest process runs from that env
    done = _run_powershell(job)
    assert done.returncode == 1, done.stdout + done.stderr
    result = json.loads((state / updater.HELPER_FILE).read_text(encoding="utf-8-sig"))
    assert result["refused"] == "environment in use" and not ran.exists()
    log = (state / "update.log").read_text(encoding="utf-8-sig")
    assert "Waiting up to 1 s for python" in log and "use xdart's environment" in log


@windows_only
def test_powershell_helper_waits_for_a_launcher_that_outlives_xdart(tmp_path):
    import subprocess
    job, state, marker = _windows_job(tmp_path, _script("print('updated', flush=True)"), env_wait=60)
    log, launcher_py = state / "update.log", tmp_path / "launcher.py"
    # Exits once the helper has seen it running, as pixi's xdart.exe does just after xdart.
    launcher_py.write_text(
        "import pathlib, sys, time\n"
        "log, end = pathlib.Path(sys.argv[1]), time.monotonic() + 60\n"
        "def seen():\n"
        "    try:\n"
        "        return 'Waiting up to' in log.read_text('utf-8', 'replace')\n"
        "    except OSError:\n"
        "        return False\n"
        "while time.monotonic() < end and not seen():\n"
        "    time.sleep(0.2)\n", encoding="utf-8")
    launcher = subprocess.Popen([_PYTHONW, str(launcher_py), str(log)])
    try:
        done = _run_powershell(job)
    finally:
        launcher.kill()
        launcher.wait(10)
    assert done.returncode == 0, done.stdout + done.stderr
    text = log.read_text(encoding="utf-8-sig")
    assert "Waiting up to 60 s for pythonw.exe" in text and " updated\n" in text and "use xdart's environment" not in text
    assert _await(marker, 30)


# ── a start refused during an update says why (an app or Start-menu launch has no terminal) ─────────────────────

def _notice_install(monkeypatch, state_dir):
    monkeypatch.setattr(updater, "install_kind", lambda meta=None: "pixi-global")
    monkeypatch.setattr(updater, "state_dir_for", lambda kind, prefix=None: state_dir)
    monkeypatch.setattr(updater, "install_identity", lambda prefix=None: "I")


def test_startup_notice_names_the_recovery_step_when_cleanup_is_unproven(tmp_path, monkeypatch):
    _notice_install(monkeypatch, tmp_path)
    assert updater.startup_update_notice() is None and updater.startup_update_blocked() is False
    updater.begin_attempt(tmp_path, install="I", current="1.2.6", target="1.2.7", log=tmp_path / "update.log")
    updater.write_json_atomic(tmp_path / updater.HELPER_FILE, {"pid": 1, "state": "blocked"})
    title, text = updater.startup_update_notice()
    assert title == "xdart update needs recovery" and f"rename the folder {tmp_path}" in text
    assert str(tmp_path / "update.log") in text and updater.startup_update_blocked() is True


@posix_only
def test_startup_notice_while_this_installs_helper_runs(tmp_path, monkeypatch):
    _notice_install(monkeypatch, tmp_path)
    holder = hold_update_lock(tmp_path)
    try:
        updater.write_json_atomic(tmp_path / updater.HELPER_FILE, {"pid": holder.pid, "state": "running"})
        assert updater.startup_update_notice()[0] == "xdart is updating"
    finally:
        holder.kill()
        holder.wait()
    assert updater.startup_update_notice() is None                 # the lock died with the helper


def test_native_notice_comes_from_a_detached_system_program():
    import base64
    seen = []
    update_launch.show_native_notice("T", 'He said "hi" \\ there', platform="darwin",
                                     popen=lambda argv, **kwargs: seen.append((argv, kwargs)))
    (argv, kwargs), = seen
    assert argv[:2] == ["/usr/bin/osascript", "-e"] and kwargs["start_new_session"] is True
    assert 'display dialog "He said \\"hi\\" \\\\ there" with title "T"' in argv[2]
    seen.clear()
    update_launch.show_native_notice("T", "it's busy", platform="win32",
                                     popen=lambda argv, **kwargs: seen.append((argv, kwargs)))
    (argv, kwargs), = seen
    assert argv[0].endswith("powershell.exe") and kwargs["creationflags"] & update_launch.CREATE_NO_WINDOW
    assert "[System.Windows.MessageBox]::Show('it''s busy', 'T')" in base64.b64decode(argv[-1]).decode("utf-16-le")


def test_main_shows_the_notice_instead_of_starting(monkeypatch):
    from xdart import xdart_main
    shown = []
    monkeypatch.setattr(updater, "startup_update_notice", lambda: ("xdart is updating", "busy"))
    monkeypatch.setattr(update_launch, "show_native_notice", lambda title, text: shown.append((title, text)))
    assert xdart_main.main() == 0 and shown == [("xdart is updating", "busy")]


# ── the macOS update window closes after a successful update (osascript is never run here) ──────────────────────

def test_terminal_close_finds_the_window_of_this_tab_then_closes_it_detached():
    from types import SimpleNamespace
    calls = []
    runner = lambda argv, **kwargs: calls.append((argv, kwargs)) or SimpleNamespace(returncode=0, stdout="4242\n")
    popen = lambda argv, **kwargs: calls.append((argv, kwargs))
    assert update_launch.schedule_terminal_close("/dev/ttys009", 10.0, runner=runner, popen=popen) is True
    (lookup, lookup_kwargs), (closer, closer_kwargs) = calls
    assert lookup[:2] == ["/usr/bin/osascript", "-e"] and lookup[3:] == ["/dev/ttys009"] and lookup_kwargs["timeout"]
    assert "(count of tabs of w) is 1 and tty of tab 1 of w" in lookup[2]
    assert closer[:2] == ["/usr/bin/osascript", "-e"] and closer[3:] == ["4242", "10"]
    assert "not (busy of tab 1 of window id wid)" in closer[2] and "close window id wid" in closer[2]
    assert closer_kwargs["start_new_session"] is True and closer_kwargs["stdout"] is update_launch.subprocess.DEVNULL


@pytest.mark.parametrize("found", [(0, ""), (1, "4242"), "timeout", "missing"])
def test_terminal_close_leaves_the_window_open_when_it_cannot_find_it(found):
    from types import SimpleNamespace

    def runner(argv, **kwargs):
        if found == "timeout":
            raise update_launch.subprocess.TimeoutExpired(argv, 10)
        if found == "missing":
            raise FileNotFoundError(argv[0])
        return SimpleNamespace(returncode=found[0], stdout=found[1])
    started = []
    assert update_launch.schedule_terminal_close("/dev/ttys009", 10, runner=runner,
                                                 popen=lambda *args, **kwargs: started.append(args)) is False
    assert started == []


def test_window_close_needs_the_macos_terminal_the_app_opened(monkeypatch):
    import io
    called = []
    monkeypatch.setattr(update_launch, "schedule_terminal_close", lambda tty, s: called.append((tty, s)) or True)
    assert helper._schedule_window_close({"close_window_s": 10}) is False     # pytest's stdout is no terminal

    class Terminal(io.StringIO):
        def isatty(self):
            return True

        def fileno(self):
            return 1
    monkeypatch.setattr(sys, "stdout", Terminal())
    monkeypatch.setattr(os, "ttyname", lambda fd: "/dev/ttys009", raising=False)   # Windows has no ttyname
    monkeypatch.setattr(sys, "platform", "linux")
    assert helper._schedule_window_close({"close_window_s": 10}) is False
    monkeypatch.setattr(sys, "platform", "darwin")
    assert helper._schedule_window_close({}) is False                          # not opened by the app's Terminal launch
    assert helper._schedule_window_close({"close_window_s": 10}) is True and called == [("/dev/ttys009", 10.0)]


@posix_only
@pytest.mark.parametrize("closes", [True, False])
def test_helper_says_whether_its_window_closes(tmp_path, monkeypatch, closes):
    monkeypatch.setattr(helper, "_schedule_window_close", lambda job: closes)
    state, marker = tmp_path / "state", tmp_path / "relaunched"
    job = json.loads(_job(state, _script("pass"), marker=marker).read_text())
    path = update_launch.write_job(state, {**job, "close_window_s": 10})
    assert helper.main([str(path)]) == 0 and _await(marker)
    log = (state / "update.log").read_text()
    assert (" This window closes in 10 s.\n" in log) is closes
    assert (" You can close this window.\n" in log) is not closes


def test_only_the_macos_launch_asks_the_helper_to_close_its_window(tmp_path, monkeypatch):
    monkeypatch.setattr(update_launch, "launch_macos", lambda *args, **kwargs: None)
    monkeypatch.setattr(update_launch, "launch_linux", lambda *args, **kwargs: None)
    job = {"update_cmd": ["pixi"], "relaunch_cmd": ["xdart"]}
    mac = update_launch.launch_helper(tmp_path, {**job, "attempt_id": "mac"}, platform="darwin", python="py")
    linux = update_launch.launch_helper(tmp_path, {**job, "attempt_id": "linux"}, platform="linux", python="py")
    assert json.loads(mac.read_text())["close_window_s"] == update_launch.CLOSE_WINDOW_S
    assert "close_window_s" not in json.loads(linux.read_text())
