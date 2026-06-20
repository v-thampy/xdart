"""StaticControls — the shared run-controls bar (the CONTROLS section).

Hand-authored Qt widget (NOT a regenerated .ui).  staticWidget owns ONE
instance and routes its signals to the ACTIVE wrangler, instead of every
wrangler carrying its own copy of these run-lifecycle controls in specUI.  This
is the mode-agnostic seam for future Stitch/RSM modes: the controls stay put,
only the active wrangler behind them changes.

The widget is STATELESS about run logic — it emits *intent*
(``actionClicked`` / ``stopClicked`` / ``modeChanged`` / ``batchToggled`` /
``liveToggled``); the active wrangler owns the state machine and drives the
single Start/Pause/Resume action button's look via :meth:`set_action_phase`
(the Phase-B morph, lifted from the per-wrangler ``_set_action_button``).
"""
import os

from pyqtgraph.Qt import QtCore, QtWidgets

_CPU = os.cpu_count() or 1


class StaticControls(QtWidgets.QWidget):
    """Mode selector + Batch + cores + Live + Start/Pause/Resume + Stop."""

    actionClicked = QtCore.Signal()          # the single 3-state action button
    stopClicked = QtCore.Signal()
    modeChanged = QtCore.Signal(str)
    batchToggled = QtCore.Signal(bool)
    liveToggled = QtCore.Signal(bool)

    # Phase-B action-button morph table (text, runPhase property, enabled),
    # lifted verbatim from imageWrangler._set_action_button.
    _PHASES = {
        'idle':    ('Start',    'idle',   True),
        'running': ('Pause',    'active', True),
        'pausing': ('Pausing…', 'active', False),
        'paused':  ('Resume',   'active', True),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(5, 0, 5, 0)
        row.setSpacing(6)

        self.modeCombo = QtWidgets.QComboBox()
        self.modeCombo.setObjectName('processingModeCombo')
        row.addWidget(self.modeCombo)

        self.batchButton = QtWidgets.QPushButton('Batch')
        self.batchButton.setObjectName('batchCheckBox')
        self.batchButton.setCheckable(True)
        self.batchButton.setChecked(True)
        self.batchButton.setMaximumWidth(70)
        row.addWidget(self.batchButton)

        self.coresLabel = QtWidgets.QLabel('Cores')
        self.coresLabel.setObjectName('coresLabel')
        row.addWidget(self.coresLabel)
        self.coresSpin = QtWidgets.QSpinBox()
        self.coresSpin.setObjectName('maxCoresSpinBox')
        self.coresSpin.setMinimum(1)
        self.coresSpin.setMaximum(_CPU)
        self.coresSpin.setValue(min(_CPU - 1, 4) or 1)
        self.coresSpin.setMaximumWidth(55)
        row.addWidget(self.coresSpin)

        self.liveButton = QtWidgets.QPushButton('Live')
        self.liveButton.setObjectName('liveCheckBox')
        self.liveButton.setCheckable(True)
        self.liveButton.setMaximumWidth(140)
        row.addWidget(self.liveButton)

        self.startButton = QtWidgets.QPushButton('Start')
        # objectName 'startButton' + the runPhase property are what the dark
        # theme keys the green/orange Start/Pause styling on.
        self.startButton.setObjectName('startButton')
        self.startButton.setProperty('runPhase', 'idle')
        row.addWidget(self.startButton)

        self.stopButton = QtWidgets.QPushButton('Stop')
        self.stopButton.setObjectName('stopButton')
        self.stopButton.setEnabled(False)
        row.addWidget(self.stopButton)

        self._run_phase = 'idle'

        # Emit intent only — the active wrangler owns the logic.
        self.startButton.clicked.connect(self.actionClicked)
        self.stopButton.clicked.connect(self.stopClicked)
        self.modeCombo.currentTextChanged.connect(self.modeChanged)
        self.batchButton.toggled.connect(self.batchToggled)
        self.liveButton.toggled.connect(self.liveToggled)

    # ── action-button morph (the wrangler drives this via its back-ref) ──
    def set_action_phase(self, phase):
        """Morph the single action button: text + the green/orange runPhase
        property.  ``pausing`` is the transient disabled state.  Mirror of the
        old per-wrangler ``_set_action_button`` so the visual contract is
        identical."""
        label, prop, enabled = self._PHASES.get(phase, self._PHASES['idle'])
        self._run_phase = phase if phase in self._PHASES else 'idle'
        self.startButton.setText(label)
        self.startButton.setEnabled(enabled)
        if self.startButton.property('runPhase') != prop:
            self.startButton.setProperty('runPhase', prop)
            self.startButton.style().unpolish(self.startButton)
            self.startButton.style().polish(self.startButton)

    def action_phase(self):
        return self._run_phase

    def set_stop_enabled(self, enabled):
        self.stopButton.setEnabled(bool(enabled))

    # ── run-state gating (driven by staticWidget._enter/_exit_run_state) ──
    def set_run_active(self, active):
        """During a run, lock mode/Batch/cores/Live; keep Stop and the action
        button (now 'Pause') usable.  On exit, the active wrangler's
        ``_on_mode_changed`` restores the per-mode widget state."""
        active = bool(active)
        for w in (self.modeCombo, self.batchButton, self.coresSpin,
                  self.liveButton):
            w.setEnabled(not active)
        if active:
            self.stopButton.setEnabled(True)

    # ── per-wrangler capability profile ──
    def apply_profile(self, *, modes=None, live=True, batch=True, cores=True):
        """Repopulate the mode items + show/hide Live/Batch/cores for the active
        wrangler.  The caller blocks signals around this (swap restore race)."""
        if modes is not None:
            self.modeCombo.clear()
            self.modeCombo.addItems(list(modes))
        self.liveButton.setVisible(bool(live))
        self.batchButton.setVisible(bool(batch))
        self.coresLabel.setVisible(bool(cores))
        self.coresSpin.setVisible(bool(cores))

    # ── value getters (read at run-setup) ──
    def get_cores(self):
        return int(self.coresSpin.value())

    def is_batch(self):
        return bool(self.batchButton.isChecked())

    def is_live(self):
        return bool(self.liveButton.isChecked())

    def current_mode(self):
        return self.modeCombo.currentText()
