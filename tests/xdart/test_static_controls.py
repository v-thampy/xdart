"""Unit tests for the shared StaticControls run-controls widget (Stage 2a).

The widget owns the run-lifecycle controls and the Phase-B action-button morph;
it emits intent and the active wrangler owns the logic.  These exercise the
morph + signals + profile in isolation (no wrangler), before any app wiring.
"""
import os
import logging
from types import MethodType, SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtCore, QtGui, QtWidgets

from xdart.gui.tabs.static_scan.ui.static_controls import StaticControls


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_action_phase_morph_cycles(qapp):
    c = StaticControls()
    c.set_action_phase('idle')
    assert c.startButton.text() == '▶ Run'
    assert c.startButton.property('runPhase') == 'idle'
    assert c.startButton.isEnabled()

    c.set_action_phase('running')
    assert c.startButton.text() == '❚❚ Pause'
    assert c.startButton.property('runPhase') == 'active'

    c.set_action_phase('pausing')
    assert 'Pausing' in c.startButton.text()
    assert not c.startButton.isEnabled()           # transient, disabled

    c.set_action_phase('paused')
    assert c.startButton.text() == '▶ Resume'
    assert c.startButton.property('runPhase') == 'active'
    assert c.action_phase() == 'paused'

    c.set_action_phase('idle')                      # morph back to green
    assert c.startButton.text() == '▶ Run'
    assert c.startButton.property('runPhase') == 'idle'


def test_signals_emit_on_user_actions(qapp):
    c = StaticControls()
    c.apply_profile(modes=['Int 1D', 'Int 2D'])
    got = {'action': 0, 'stop': 0, 'mode': [], 'batch': [], 'live': []}
    c.actionClicked.connect(lambda: got.__setitem__('action', got['action'] + 1))
    c.stopClicked.connect(lambda: got.__setitem__('stop', got['stop'] + 1))
    c.modeChanged.connect(lambda s: got['mode'].append(s))
    c.batchToggled.connect(lambda b: got['batch'].append(b))
    c.liveToggled.connect(lambda b: got['live'].append(b))

    c.startButton.click()
    c.set_stop_enabled(True)        # Stop starts disabled (only live during a run)
    c.stopButton.click()
    c.modeCombo.setCurrentText('Int 2D')
    c.batchButton.setChecked(not c.batchButton.isChecked())
    c.liveButton.setChecked(not c.liveButton.isChecked())

    assert got['action'] == 1
    assert got['stop'] == 1
    assert got['mode'][-1] == 'Int 2D'
    assert got['batch'] and got['live']


def test_profile_hides_live_and_batch(qapp):
    c = StaticControls()
    c.apply_profile(modes=['Int 1D + 2D', 'Int 1D'], live=False, batch=False)
    assert c.liveButton.isHidden()
    assert c.batchButton.isHidden()
    assert [c.modeCombo.itemText(i) for i in range(c.modeCombo.count())] == \
        ['Int 1D + 2D', 'Int 1D']


def test_run_active_locks_controls(qapp):
    c = StaticControls()
    c.set_run_active(True)
    assert not c.modeCombo.isEnabled()
    assert not c.batchButton.isEnabled()
    assert not c.coresSpin.isEnabled()
    assert not c.liveButton.isEnabled()
    assert c.stopButton.isEnabled()                # Stop stays live during a run
    c.set_run_active(False)
    assert c.modeCombo.isEnabled()
    assert c.liveButton.isEnabled()


def test_mode_row_enabled_locks_mode_batch_cores(qapp):
    """The mode row (mode combo + Batch + Cores) locks during a run while the
    action row stays usable.  Owned by _enter/_exit_run_state so a reintegrate
    (which skips wrangler.enabled()) also locks it."""
    c = StaticControls()
    c.set_mode_row_enabled(False)
    assert not c.modeCombo.isEnabled()
    assert not c.batchButton.isEnabled()
    assert not c.coresSpin.isEnabled()
    # The Append/Replace write-mode toggle is a run-config control too: it must
    # lock during a run (you can't switch Append<->Replace mid-scan).
    assert not c.writeModeButton.isEnabled()
    assert c.actionRow.isEnabled()                 # action row left alone
    c.set_mode_row_enabled(True)
    assert c.modeCombo.isEnabled()
    assert c.batchButton.isEnabled()
    assert c.writeModeButton.isEnabled()


def _shortcut_host(**attrs):
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    host = SimpleNamespace(**attrs)
    for name in (
        "_commit_shortcut_focus",
        "shortcut_run_pause",
        "shortcut_stop",
        "shortcut_toggle_write_mode",
        "shortcut_load_settings",
        "shortcut_save_settings",
    ):
        setattr(host, name, MethodType(getattr(staticWidget, name), host))
    return host


def test_run_shortcut_commits_line_edit_before_click(qapp):
    c = StaticControls()
    host = _shortcut_host(controls=c)
    order = []
    c.startButton.clicked.connect(lambda: order.append("run"))

    editor = QtWidgets.QLineEdit()
    editor.show()
    editor.setText("5")
    editor.setModified(True)
    editor.editingFinished.connect(lambda: order.append("commit"))
    editor.activateWindow()
    editor.setFocus(QtCore.Qt.OtherFocusReason)
    qapp.processEvents()
    assert qapp.focusWidget() is editor

    host.shortcut_run_pause()

    assert order == ["commit", "run"]


def test_run_shortcut_noops_when_start_disabled(qapp):
    c = StaticControls()
    host = _shortcut_host(controls=c)
    clicks = []
    c.startButton.clicked.connect(lambda: clicks.append("run"))
    c.startButton.setEnabled(False)

    host.shortcut_run_pause()

    assert clicks == []


def test_stop_shortcut_stays_available_when_mode_row_locked(qapp):
    c = StaticControls()
    host = _shortcut_host(controls=c)
    clicks = []
    c.stopButton.clicked.connect(lambda: clicks.append("stop"))
    c.set_mode_row_enabled(False)
    c.set_stop_enabled(True)

    host.shortcut_stop()

    assert clicks == ["stop"]


def test_toggle_write_mode_shortcut_respects_run_lock(qapp):
    c = StaticControls()
    host = _shortcut_host(controls=c)
    changes = []
    c.writeModeChanged.connect(changes.append)

    c.set_mode_row_enabled(False)
    host.shortcut_toggle_write_mode()
    assert c.write_mode() == "Append"
    assert changes == []

    c.set_mode_row_enabled(True)
    host.shortcut_toggle_write_mode()
    assert c.write_mode() == "Overwrite"
    assert changes == ["Overwrite"]


def test_load_save_shortcuts_route_through_existing_actions(qapp):
    load = QtGui.QAction()
    save = QtGui.QAction()
    calls = []
    load.triggered.connect(lambda: calls.append("load"))
    save.triggered.connect(lambda: calls.append("save"))
    host = _shortcut_host(
        h5viewer=SimpleNamespace(
            actionLoadParams=load,
            actionSaveParams=save,
        )
    )

    host.shortcut_load_settings()
    host.shortcut_save_settings()

    assert calls == ["load", "save"]


def test_main_window_shortcuts_are_menu_backed(qapp, monkeypatch, caplog):
    from xdart import _gui_main

    class FakeStaticWidget(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.h5viewer = SimpleNamespace(paramMenu=QtWidgets.QMenu(self))
            self.ui = SimpleNamespace(
                leftFrame=QtWidgets.QFrame(self),
                middleFrame=QtWidgets.QFrame(self),
                rightFrame=QtWidgets.QFrame(self),
            )
            self.ui.leftFrame.setObjectName("leftFrame")
            self.ui.middleFrame.setObjectName("middleFrame")
            self.ui.rightFrame.setObjectName("rightFrame")

        def enable_async_hydration(self):
            pass

        def open_file(self):
            self.calls.append("open_file")

        def shortcut_run_pause(self):
            self.calls.append("run")

        def shortcut_stop(self):
            self.calls.append("stop")

        def shortcut_toggle_write_mode(self):
            self.calls.append("toggle")

        def shortcut_load_settings(self):
            self.calls.append("load")

        def shortcut_save_settings(self):
            self.calls.append("save")

    monkeypatch.setattr(
        _gui_main.tabs.static_scan, "staticWidget", FakeStaticWidget)
    window = _gui_main.Main()
    try:
        file_actions = [a.text() for a in window.ui.menuFile.actions()]
        run_actions = [a.text() for a in window.ui.menuRun.actions()]
        config_actions = window.main_widget.h5viewer.paramMenu.actions()

        assert "Load Settings" in file_actions
        assert "Save Settings" in file_actions
        assert run_actions == ["Run / Pause", "Stop", "Toggle Append / Replace"]
        assert window.debugMenu.menuAction() in config_actions
        assert window.debugMenu.title() == "Debug"
        assert [a.text() for a in window.debugMenu.actions()] == [
            "Window State"]
        assert window.actionRunPause.shortcut().toString(
            QtGui.QKeySequence.PortableText) == "Ctrl+R"
        assert window.actionStopRun.shortcut().toString(
            QtGui.QKeySequence.PortableText) == "Ctrl+Shift+C"
        assert window.actionToggleWriteMode.shortcut().toString(
            QtGui.QKeySequence.PortableText) == "Ctrl+Shift+A"
        assert window.actionLoadSettings.shortcut().toString(
            QtGui.QKeySequence.PortableText) == "Ctrl+O"
        assert window.actionSaveSettings.shortcut().toString(
            QtGui.QKeySequence.PortableText) == "Ctrl+S"

        window.actionRunPause.trigger()
        window.actionStopRun.trigger()
        window.actionToggleWriteMode.trigger()
        window.actionLoadSettings.trigger()
        window.actionSaveSettings.trigger()

        assert window.main_widget.calls == [
            "run", "stop", "toggle", "load", "save"]

        with caplog.at_level(logging.WARNING, logger=_gui_main.__name__):
            window.actionDebugWindowState.trigger()
        assert "Window State main:" in caplog.text
        assert "left browser" in caplog.text
        assert "middle display" in caplog.text
        assert "right controls" in caplog.text
        assert "overrideCursor=" in caplog.text
        assert "mouseGrabber=" in caplog.text
        assert "top-level widgets: total=" in caplog.text
    finally:
        window.close()


def test_run_row_visibility_can_collapse_to_mode_only(qapp):
    """File viewers can collapse the controls bar to the mode row only."""
    c = StaticControls()
    c.set_run_row_visible(False)
    assert c.actionRow.isHidden()
    assert c._divider.isHidden()
    c.set_run_row_visible(True)
    assert not c.actionRow.isHidden()
    assert not c._divider.isHidden()


def test_getters(qapp):
    c = StaticControls()
    c.apply_profile(modes=['Int 1D', 'Int 2D'])
    c.modeCombo.setCurrentText('Int 2D')
    c.batchButton.setChecked(True)
    c.liveButton.setChecked(False)
    c.coresSpin.setValue(2)
    assert c.current_mode() == 'Int 2D'
    assert c.is_batch() is True
    assert c.is_live() is False
    assert c.get_cores() == 2


def test_readiness_summary_toggles_compact_status_row(qapp):
    c = StaticControls()
    assert c.readinessRow.isHidden()

    changed = c.set_readiness_summary(
        "Ready · Int 2D · 50 frames",
        ready=True,
        tooltip="Everything is ready.",
    )

    assert changed is True
    assert not c.readinessRow.isHidden()
    assert c.readinessLabel.text() == "Ready · Int 2D · 50 frames"
    assert c.readinessLabel.toolTip() == "Everything is ready."
    assert c.readinessDot.property("ready") is True

    assert c.set_readiness_summary("", ready=False) is True
    assert c.readinessRow.isHidden()
