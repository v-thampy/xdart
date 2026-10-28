# -*- coding: utf-8 -*-
"""The in-app updater's GUI flow on a real main window (synthetic page): the startup notice and its offer, Cancel, a Run starting during the offer, Quit during a check, the one admission with the helper's handshake, and the
start of a second xdart while an update runs. Settings and update state are isolated (rule 10)."""
from __future__ import annotations

import json
import sys
import threading
import time

import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

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
    from xdart.modules import update_notes
    monkeypatch.setattr(update_notes, "fetch_release_notes", lambda *a: update_notes.ReleaseNotes(
        "Version 9.9.9\nNew\n• Clearer update controls", update_notes.RELEASES_URL))
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


def test_startup_offer_waits_for_explicit_update(window, qapp, monkeypatch, tmp_path):
    admitted = _startup_result(window, monkeypatch, tmp_path)
    box = window._update_dialog
    assert box is not None and box.isVisible()
    assert "Installed: 1.2.6" in box.versions.text() and "Available: 9.9.9" in box.versions.text()
    assert box.update_button.text() == "Update" and box.cancel_button.text() == "Cancel"
    assert box.update_button.toolTip() and box.notes.toolTip()
    assert _pump(qapp, lambda: "Clearer update controls" in box.notes.toPlainText(), 2)
    assert not _pump(qapp, lambda: bool(admitted), 2.5)
    box.update_button.click()
    assert _pump(qapp, lambda: admitted == ["9.9.9"], 2)
    assert window._update_dialog is None


def test_cancel_skips_the_version_until_the_next_start(window, qapp, monkeypatch, tmp_path):
    admitted = _startup_result(window, monkeypatch, tmp_path)
    window._update_dialog.cancel_button.click()
    assert _pump(qapp, lambda: window._update_dialog is None, 2)
    qapp.processEvents()
    time.sleep(2.5)
    qapp.processEvents()
    assert admitted == [] and "9.9.9" in window._update_skipped
    _startup_result(window, monkeypatch, tmp_path)
    assert window._update_dialog is None                 # the same version is not offered again this session


def test_a_run_starting_during_the_offer_postpones_the_update(window, qapp, monkeypatch, tmp_path):
    admitted = _startup_result(window, monkeypatch, tmp_path)
    window.activity[0] = True
    assert _pump(qapp, lambda: window._update_dialog is None, 4)
    assert admitted == [] and "postponed" in window.statusBar().currentMessage()


def test_the_startup_offer_skips_an_unvalidated_install_and_offers_a_failed_version_from_the_next_start(
        window, monkeypatch, tmp_path):
    _startup_result(window, monkeypatch, tmp_path, automatic=False)
    assert window._update_dialog is None
    state = updater.update_state_dir(_meta(tmp_path))
    updater.begin_attempt(state, install="test-install", current="1.2.6", target="9.9.9", log="l")
    note = updater.settle_previous_attempt(state, install="test-install", current="1.2.6",
                                           alive=lambda _pid: False)              # 9.9.9 did not install
    assert note[0] == "incomplete" and "failed" not in json.loads((state / updater.STATE_FILE).read_text())
    monkeypatch.setattr(updater, "install_kind", lambda meta=None: "managed")   # this test runs the check itself
    monkeypatch.setattr(QtWidgets.QMessageBox, "open", lambda self: None)
    window.start_startup_updates((*note, "pixi-global"))
    _startup_result(window, monkeypatch, tmp_path)
    assert window._update_dialog is None                      # not on the start that reports the failure
    window._update_skipped.clear()                            # the next start: nothing on disk holds it back
    _startup_result(window, monkeypatch, tmp_path)
    assert window._update_dialog is not None


def test_quit_during_a_check_ignores_its_late_result(window, qapp, monkeypatch):
    release, finished = threading.Event(), threading.Event()

    def slow_check(kind):
        release.wait(5)
        finished.set()
        return updater.UpdateCheck("ok", "9.9.9", True)

    monkeypatch.setattr(updater, "check_for_update", slow_check)
    seen = []
    monkeypatch.setattr(window, "_show_update_dialog", lambda *args: seen.append(args))
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
    holders = []

    def launch(state_dir, job):
        jobs.append(job)
        job.update(manual_restart=True, require_progress_host=True)   # as launch_helper sets on macOS and Windows
        holders.append(updater.take_file_lock(state_dir / updater.LOCK_FILE))
        job.update(host_file=str(state_dir / "test.host.json"), host_lock=str(state_dir / "test.host.lock"))
        holders.append(updater.take_file_lock(job["host_lock"]))
        updater.write_json_atomic(job["host_file"], {"attempt_id": job["attempt_id"], "ready": True})
        updater.write_json_atomic(state_dir / updater.HELPER_FILE,
                                  {"pid": 4242, "state": "running", "attempt_id": job["attempt_id"]})

    try:
        exits, copyable = _admission(window, monkeypatch, tmp_path, launch)
        assert _pump(qapp, lambda: exits == [True], 3) and copyable == []
    finally:
        for holder in holders:
            holder.close()
    job, = jobs
    assert job["target"] == "9.9.9" and job["update_env"] == {"PIXI_HOME": str(tmp_path)}
    assert job["time_limit_s"] == updater.UPDATE_TIME_LIMIT_S and job["parent_wait_s"] == updater.PARENT_WAIT_S
    assert job["env_wait_s"] == updater.ENV_WAIT_S
    attempt = updater.load_update_state(updater.update_state_dir(_meta(tmp_path)), "test-install")["attempt"]
    assert attempt["from"] == "1.2.6" and attempt["to"] == "9.9.9"


def test_a_refused_launch_keeps_xdart_open_with_the_copyable_command(window, qapp, monkeypatch, tmp_path):
    def refuse(state_dir, job):
        raise PermissionError(5, "the job refuses breakaway")

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
    boxes = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "open", lambda self: boxes.append(self.text()))   # macOS drops titles
    window.start_startup_updates(("updated", {"from": "1.2.6", "to": "1.2.7"}, "pixi-global"))
    assert boxes == ["xdart was updated from 1.2.6 to 1.2.7."]
    window.start_startup_updates(("incomplete", {"to": "1.2.8", "log": "/x/update.log"}, "pixi-global"))
    assert boxes[1].startswith("The update to xdart 1.2.8 did not complete; see /x/update.log.")


def test_without_a_progress_window_the_offer_updates_in_the_background(window, qapp, monkeypatch, tmp_path):
    from xdart.gui.widgets.read_only_text import ReadOnlyTextWindow
    monkeypatch.setattr(update_launch, "has_progress_window", lambda platform=None: False)
    admitted = _startup_result(window, monkeypatch, tmp_path)
    box = window._update_dialog                                   # Linux: offered at startup as elsewhere
    assert isinstance(box, ReadOnlyTextWindow) and box.notes is box.pane      # the shared read-only window (rule 14)
    assert box.versions.textFormat() == QtCore.Qt.TextFormat.PlainText
    assert box.update_button.text() == "Update" and "reopen" in box.update_button.toolTip()
    assert "reopens it when the update finishes" in box.explanation.text() and box.explanation.toolTip()
    box.update_button.click()
    assert _pump(qapp, lambda: admitted == ["9.9.9"], 2) and window._update_dialog is None
