# -*- coding: utf-8 -*-
"""In-app updater (Help -> Check for Updates...) — pure logic + the update-on-exit
helper.  No Qt, no network, no real pixi: every external edge is injected.
Spec: docs/design/design_install_and_update_jul2026.md section 4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from xdart.modules import updater
from xdart import _updater as helper


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


def test_pixi_global_meta_builds_update_command(monkeypatch, tmp_path):
    # The home comes from the running env, not $PIXI_HOME: an install under another home updates that home.
    monkeypatch.setenv("PIXI_HOME", str(tmp_path / "elsewhere"))
    home = tmp_path / ".pixi"
    env = home / "envs" / "xdart-test"
    env.mkdir(parents=True)
    meta = updater.pixi_global_meta(prefix=str(env), platform="darwin")
    assert meta["flavor"] == "pixi-global"
    assert meta["update_cmd"][-3:] == ["global", "update", "xdart-test"]
    assert meta["update_env"] == {"PIXI_HOME": str(home.resolve())}
    assert meta["relaunch_cmd"] == [str(home.resolve() / "bin" / "xdart")]  # the trampoline, not the env's script


def test_pixi_global_meta_on_windows_needs_both_executables(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setenv("PIXI_HOME", str(tmp_path / "no-default-home"))
    home = tmp_path / ".pixi"
    env = home / "envs" / "xdart"
    env.mkdir(parents=True)
    (home / "bin").mkdir()
    (home / "bin" / "xdart.exe").write_bytes(b"")
    assert updater.pixi_global_meta(prefix=str(env), platform="win32") is None  # no pixi.exe anywhere
    pixi = home / "bin" / "pixi.exe"
    pixi.write_bytes(b"")
    meta = updater.pixi_global_meta(prefix=str(env), platform="win32")
    assert meta["update_cmd"] == [str(pixi.resolve()), "global", "update", "xdart"]
    assert meta["relaunch_cmd"] == [str(home.resolve() / "bin" / "xdart.exe")]
    (home / "bin" / "xdart.exe").unlink()
    assert updater.pixi_global_meta(prefix=str(env), platform="win32") is None


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


def test_run_update_success_relaunches_new(tmp_path):
    launched = []
    log = tmp_path / "update.log"

    class _Ok:
        returncode = 0
        stdout = "updated"
        stderr = ""

    ok = helper.run_update(
        ["pixi", "update"], ["xdart"], str(log),
        runner=lambda cmd: _Ok(), launcher=launched.append)
    assert ok is True
    assert launched == [["xdart"]]                     # relaunched
    assert "success" in log.read_text()


def test_run_update_failure_still_relaunches_old(tmp_path):
    launched = []
    log = tmp_path / "update.log"

    class _Fail:
        returncode = 1
        stdout = ""
        stderr = "boom"

    ok = helper.run_update(
        ["pixi", "update"], ["xdart"], str(log),
        runner=lambda cmd: _Fail(), launcher=launched.append)
    assert ok is False
    assert launched == [["xdart"]]                     # OLD version relaunched anyway
    assert "FAILED" in log.read_text()


def test_update_check_runs_off_the_gui_thread():
    # The PyPI fetch must never run on the GUI thread: the check worker is a
    # QThread subclass carrying its result back over a signal (spec section 4).
    from pyqtgraph.Qt import QtCore
    from xdart._gui_main import _UpdateCheckThread
    assert issubclass(_UpdateCheckThread, QtCore.QThread)
    assert hasattr(_UpdateCheckThread, "result_ready")


def test_helper_main_parses_args_and_runs(tmp_path, monkeypatch):
    # end-to-end arg parsing with the wait + update injected out.
    seen = {}

    def fake_wait(pid, **k):
        seen["pid"] = pid
        return True                                    # parent exited

    monkeypatch.setattr(helper, "wait_for_exit", fake_wait)
    monkeypatch.setattr(helper, "run_update",
                        lambda uc, rc, lp, **k: seen.update(uc=uc, rc=rc, lp=lp))
    rc = helper.main([str(4321), str(tmp_path),
                      json.dumps(["pixi", "update"]), json.dumps(["xdart"]),
                      str(tmp_path / "u.log")])
    assert rc == 0
    assert seen["pid"] == 4321
    assert seen["uc"] == ["pixi", "update"] and seen["rc"] == ["xdart"]


def test_helper_main_aborts_update_if_parent_survives(tmp_path, monkeypatch):
    # S1: if the app does NOT exit within the timeout, the helper must NOT update
    # (its env files are in use) -- log + bail, leave the app running.
    ran = {"update": False}
    monkeypatch.setattr(helper, "wait_for_exit", lambda pid, **k: False)  # timed out
    monkeypatch.setattr(helper, "run_update",
                        lambda *a, **k: ran.__setitem__("update", True))
    log = tmp_path / "u.log"
    rc = helper.main([str(999999), str(tmp_path), "[]", "[]", str(log)])
    assert rc == 1
    assert ran["update"] is False                      # update was NOT run
    assert "ABORTED" in log.read_text()


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
