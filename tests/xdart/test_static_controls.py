"""Unit tests for the shared StaticControls run-controls widget (Stage 2a).

The widget owns the run-lifecycle controls and the Phase-B action-button morph;
it emits intent and the active wrangler owns the logic.  These exercise the
morph + signals + profile in isolation (no wrangler), before any app wiring.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pyqtgraph")
from pyqtgraph import QtWidgets

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
    assert c.actionRow.isEnabled()                 # action row left alone
    c.set_mode_row_enabled(True)
    assert c.modeCombo.isEnabled()
    assert c.batchButton.isEnabled()


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
