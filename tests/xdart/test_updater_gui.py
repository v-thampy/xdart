# -*- coding: utf-8 -*-
"""The in-app updater's GUI flow on a real main window (synthetic page): the startup notice and its countdown, Not
now, a Run starting during the countdown, Quit during a check, the one admission with the helper's handshake, and the
start of a second xdart while an update runs. Settings and update state are isolated (rule 10)."""
from __future__ import annotations

import sys
import threading
import time

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.pages.test_synthetic_page_host import _descriptor, _services
from tests.xdart.test_updater import hold_update_lock
from xdart import _gui_main
from xdart.gui.pages.values import PageCapability
from xdart.modules import update_launch, updater


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(qapp, monkeypatch):
    activity = [False]
    page = _descriptor("updater-page", [], capabilities=frozenset({PageCapability.RUN_ACTIVITY}), activity=activity)
    main = _gui_main.Main(page_descriptors=(page,), host_services=_services([]), selected_page_key=page.key)
    main.activity = activity
    monkeypatch.setattr(_gui_main, "_UPDATE_NOTICE_S", 2)
    yield main
    main._update_generation += 1
    main.close()
    main.deleteLater()
    qapp.processEvents()


def _pump(qapp, predicate, seconds=10.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _meta(tmp_path):
    return {"flavor": "pixi-global", "app_root": str(tmp_path), "env_name": "xdart", "env_prefix": str(tmp_path),
            "update_cmd": ["pixi", "global", "update", "xdart"], "update_env": {"PIXI_HOME": str(tmp_path)},
            "relaunch_cmd": [str(tmp_path / "bin" / "xdart")]}


def _startup_result(window, monkeypatch, tmp_path, *, automatic=True, latest="9.9.9"):
    monkeypatch.setattr(updater, "current_version", lambda: "1.2.6")
    monkeypatch.setattr(updater, "resolve_update_meta", lambda kind: _meta(tmp_path))
    monkeypatch.setattr(updater, "install_identity", lambda prefix=None: "test-install")
    admitted = []
    monkeypatch.setattr(window, "_admit_update", lambda kind, meta, target: admitted.append(target))
    window._update_kind, window._update_source = "pixi-global", "startup"
    window._update_generation += 1
    window._on_update_check(updater.UpdateCheck("ok", latest, automatic), window._update_generation)
    return admitted


def test_startup_notice_counts_down_then_admits_the_update(window, qapp, monkeypatch, tmp_path):
    admitted = _startup_result(window, monkeypatch, tmp_path)
    box = window._update_countdown
    assert box is not None and box.isVisible()
    assert box.text() == "Updating xdart to 9.9.9: it will close, update and reopen."
    not_now, = box.buttons()
    assert not_now.text() == "Not now" and not_now.toolTip() and box.toolTip()
    assert _pump(qapp, lambda: admitted == ["9.9.9"], 6)
    assert window._update_countdown is None


def test_not_now_skips_the_version_until_the_next_start(window, qapp, monkeypatch, tmp_path):
    admitted = _startup_result(window, monkeypatch, tmp_path)
    window._update_countdown.buttons()[0].click()
    assert _pump(qapp, lambda: window._update_countdown is None, 2)
    qapp.processEvents()
    time.sleep(2.5)
    qapp.processEvents()
    assert admitted == [] and "9.9.9" in window._update_skipped
    _startup_result(window, monkeypatch, tmp_path)
    assert window._update_countdown is None                 # the same version is not offered again this session


def test_a_run_starting_during_the_countdown_postpones_the_update(window, qapp, monkeypatch, tmp_path):
    admitted = _startup_result(window, monkeypatch, tmp_path)
    window.activity[0] = True
    assert _pump(qapp, lambda: window._update_countdown is None, 4)
    assert admitted == [] and "postponed" in window.statusBar().currentMessage()


def test_the_startup_update_never_acts_on_an_unvalidated_install_or_a_failed_target(window, monkeypatch, tmp_path):
    _startup_result(window, monkeypatch, tmp_path, automatic=False)
    assert window._update_countdown is None
    updater.begin_attempt(updater.update_state_dir(_meta(tmp_path)), install="test-install", current="1.2.6",
                          target="9.9.9", log="l")
    updater.settle_previous_attempt(updater.update_state_dir(_meta(tmp_path)), install="test-install",
                                    current="1.2.6", alive=lambda _pid: False)    # 9.9.9 failed before
    _startup_result(window, monkeypatch, tmp_path)
    assert window._update_countdown is None


def test_quit_during_a_check_ignores_its_late_result(window, qapp, monkeypatch):
    release, finished = threading.Event(), threading.Event()

    def slow_check(kind):
        release.wait(5)
        finished.set()
        return updater.UpdateCheck("ok", "9.9.9", True)

    monkeypatch.setattr(updater, "check_for_update", slow_check)
    seen = []
    monkeypatch.setattr(window, "_show_update_countdown", lambda *args: seen.append(args))
    window._begin_update_check("pixi-global", "startup")
    window.close()                                           # closing bumps the generation
    release.set()
    assert finished.wait(5)
    qapp.processEvents()
    assert seen == []


def test_the_startup_toggle_is_remembered(window):
    action = window.actionCheckUpdatesAtStartup
    assert action.isCheckable() and action.toolTip() and window.actionCheckForUpdates.toolTip()
    action.setChecked(False)
    assert _gui_main._app_settings().value(_gui_main._UPDATES_AT_STARTUP_KEY, True, type=bool) is False
    action.setChecked(True)
    assert _gui_main._app_settings().value(_gui_main._UPDATES_AT_STARTUP_KEY, False, type=bool) is True


def _admission(window, monkeypatch, tmp_path, launch):
    monkeypatch.setattr(updater, "install_kind", lambda meta=None: "pixi-global")
    monkeypatch.setattr(updater, "current_version", lambda: "1.2.6")
    monkeypatch.setattr(updater, "install_identity", lambda prefix=None: "test-install")
    monkeypatch.setattr(update_launch, "launch_helper", launch)
    exits, copyable = [], []
    monkeypatch.setattr(window, "exit", lambda: exits.append(True))
    monkeypatch.setattr(window, "_show_copyable_update", lambda *args: copyable.append(args))
    monkeypatch.setattr(_gui_main, "_UPDATE_HANDSHAKE_S", 1.0)
    window._admit_update("pixi-global", _meta(tmp_path), "9.9.9")
    return exits, copyable


def test_admission_records_the_attempt_and_closes_once_the_helper_reports(window, qapp, monkeypatch, tmp_path):
    jobs = []

    def launch(state_dir, job):
        jobs.append(job)
        updater.write_json_atomic(state_dir / updater.HELPER_FILE, {"pid": 4242, "state": "running"})

    exits, copyable = _admission(window, monkeypatch, tmp_path, launch)
    assert _pump(qapp, lambda: exits == [True], 3) and copyable == []
    job, = jobs
    assert job["target"] == "9.9.9" and job["update_env"] == {"PIXI_HOME": str(tmp_path)}
    assert job["time_limit_s"] == updater.UPDATE_TIME_LIMIT_S and job["parent_wait_s"] == updater.PARENT_WAIT_S
    attempt = updater.load_update_state(updater.update_state_dir(_meta(tmp_path)), "test-install")["attempt"]
    assert attempt["from"] == "1.2.6" and attempt["to"] == "9.9.9"


def test_a_refused_launch_keeps_xdart_open_with_the_copyable_command(window, qapp, monkeypatch, tmp_path):
    def refuse(state_dir, job):
        raise update_launch.UpdateLaunchRefused("the job refuses breakaway")

    exits, copyable = _admission(window, monkeypatch, tmp_path, refuse)
    assert exits == [] and "breakaway" in copyable[0][3]
    assert updater.load_update_state(updater.update_state_dir(_meta(tmp_path)), "test-install")["attempt"] is None


def test_a_helper_that_never_reports_is_abandoned(window, qapp, monkeypatch, tmp_path):
    exits, copyable = _admission(window, monkeypatch, tmp_path, lambda state_dir, job: None)
    assert _pump(qapp, lambda: bool(copyable), 4) and exits == []
    assert "did not start" in copyable[0][3] and window._update_busy is False
    assert updater.load_update_state(updater.update_state_dir(_meta(tmp_path)), "test-install")["attempt"] is None


@pytest.mark.skipif(sys.platform.startswith("win"), reason="the stand-in lock holder uses POSIX flock")
def test_a_second_start_during_an_update_quits(monkeypatch, tmp_path, qapp):
    shown = []
    monkeypatch.setattr(updater, "install_kind", lambda meta=None: "pixi-global")
    monkeypatch.setattr(updater, "state_dir_for", lambda kind, prefix=None: tmp_path)
    monkeypatch.setattr(updater, "install_identity", lambda prefix=None: "test-install")
    monkeypatch.setattr(updater, "current_version", lambda: "1.2.6")
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: shown.append(args[2]))
    updater.begin_attempt(tmp_path, install="test-install", current="1.2.6", target="1.2.7", log="l")
    holder = hold_update_lock(tmp_path)               # a helper still running for this install
    try:
        updater.write_json_atomic(tmp_path / updater.HELPER_FILE, {"pid": holder.pid, "state": "running"})
        assert _gui_main._settle_update_at_startup() == "active"
    finally:
        holder.kill()
        holder.wait()
    assert shown and "being updated" in shown[0]
    assert updater.load_update_state(tmp_path, "test-install")["attempt"]["to"] == "1.2.7"   # untouched


def test_the_next_start_reports_how_the_update_went(window, qapp, monkeypatch):
    monkeypatch.setattr(updater, "current_version", lambda: "1.2.7")
    monkeypatch.setattr(updater, "install_kind", lambda meta=None: "managed")   # no startup check afterwards
    window.start_startup_updates(("updated", {"to": "1.2.7"}, "pixi-global"))
    assert window.statusBar().currentMessage() == "Updated to xdart 1.2.7."
    boxes = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "open", lambda self: boxes.append(self.text()))
    window.start_startup_updates(("incomplete", {"to": "1.2.8", "log": "/x/update.log"}, "pixi-global"))
    assert boxes and boxes[0].startswith("The update to xdart 1.2.8 did not complete; see /x/update.log.")
