# -*- coding: utf-8 -*-
"""In-app updater (Help -> Check for Updates...) — pure logic + the update-on-exit
helper.  No Qt, no network, no real pixi: every external edge is injected.
Spec: docs/design/design_install_and_update_jul2026.md section 4.
"""
from __future__ import annotations

import json

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
    monkeypatch.setattr(updater, "detect_pixi_global_env", lambda: True)
    assert updater.install_kind() == "pixi-global"


def test_pixi_global_meta_builds_update_command(monkeypatch, tmp_path):
    monkeypatch.setenv("PIXI_HOME", str(tmp_path))
    meta = updater.pixi_global_meta()
    assert meta["flavor"] == "pixi-global"
    assert meta["update_cmd"][-3:] == ["global", "update", "xrd-tools"]
    assert meta["relaunch_cmd"]                      # a launch command exists


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
    monkeypatch.setattr(helper, "wait_for_exit", lambda pid, **k: seen.setdefault("pid", pid))
    monkeypatch.setattr(helper, "run_update",
                        lambda uc, rc, lp, **k: seen.update(uc=uc, rc=rc, lp=lp))
    rc = helper.main([str(4321), str(tmp_path),
                      json.dumps(["pixi", "update"]), json.dumps(["xdart"]),
                      str(tmp_path / "u.log")])
    assert rc == 0
    assert seen["pid"] == 4321
    assert seen["uc"] == ["pixi", "update"] and seen["rc"] == ["xdart"]
