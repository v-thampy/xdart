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
        # Two rows separated by a divider: the SELECTION/OPTIONS row (mode +
        # Batch + Cores) on top, the ACTION row (Live + Start + Stop) below.  The
        # action row lives in its own container so it (and the divider) can be
        # hidden as a unit in file-viewer modes where there is no run.  Uniform
        # margins == spacing == _PAD so the gap above row 1, between each row and
        # the divider, and below row 2 are all equal; the host sizes the frame to
        # hug this content (no centring slack).
        _PAD = 6
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(5, _PAD, 5, _PAD)
        outer.setSpacing(_PAD)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Fixed)

        # Status / message bar — the control-layer home for run + browse messages
        # (relocated here from the wrangler's orphaned specLabel).  Sits at the
        # top of the control stack, above the mode/Batch row; the wranglers route
        # status to it via wranglerWidget._status_label when controls are attached.
        self.statusLabel = QtWidgets.QLabel('')
        self.statusLabel.setObjectName('statusLabel')
        self.statusLabel.setMinimumHeight(21)
        # Ignored horizontal policy: overlong status text clips/elides instead of
        # forcing the right panel wider (mirrors wranglerWidget._guard_status_label).
        _sp = self.statusLabel.sizePolicy()
        _sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Ignored)
        self.statusLabel.setSizePolicy(_sp)
        outer.addWidget(self.statusLabel)

        row1 = QtWidgets.QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(6)
        outer.addLayout(row1)

        self.modeCombo = QtWidgets.QComboBox()
        self.modeCombo.setObjectName('processingModeCombo')
        # The mode combo absorbs the row's slack (stretch=1) so Cores stays
        # compact instead of ballooning to fill the bar.
        row1.addWidget(self.modeCombo, 1)

        self.batchButton = QtWidgets.QPushButton('Batch')
        self.batchButton.setObjectName('batchCheckBox')
        self.batchButton.setCheckable(True)
        # Live is the default everywhere (faster per the perf baselines, and the
        # interactive default for Reintegrate); Batch is opt-in for the fast
        # multicore path.  Drives BOTH wrangler runs and Reintegrate.
        self.batchButton.setChecked(False)
        self.batchButton.setMaximumWidth(70)
        row1.addWidget(self.batchButton)

        self.coresLabel = QtWidgets.QLabel('Cores')
        self.coresLabel.setObjectName('coresLabel')
        row1.addWidget(self.coresLabel)
        self.coresSpin = QtWidgets.QSpinBox()
        self.coresSpin.setObjectName('maxCoresSpinBox')
        self.coresSpin.setMinimum(1)
        self.coresSpin.setMaximum(_CPU)
        self.coresSpin.setValue(min(_CPU - 1, 4) or 1)
        self.coresSpin.setMaximumWidth(55)
        row1.addWidget(self.coresSpin)

        self._divider = QtWidgets.QFrame()
        self._divider.setObjectName('controlsDivider')
        self._divider.setFrameShape(QtWidgets.QFrame.HLine)
        self._divider.setFrameShadow(QtWidgets.QFrame.Sunken)
        self._divider.setFixedHeight(2)
        outer.addWidget(self._divider)

        self.actionRow = QtWidgets.QWidget()
        row2 = QtWidgets.QHBoxLayout(self.actionRow)
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(6)
        outer.addWidget(self.actionRow)

        self.liveButton = QtWidgets.QPushButton('Live')
        self.liveButton.setObjectName('liveCheckBox')
        self.liveButton.setCheckable(True)
        self.liveButton.setMaximumWidth(140)
        row2.addWidget(self.liveButton)

        self.startButton = QtWidgets.QPushButton('Start')
        # objectName 'startButton' + the runPhase property are what the dark
        # theme keys the green/orange Start/Pause styling on.
        self.startButton.setObjectName('startButton')
        self.startButton.setProperty('runPhase', 'idle')
        row2.addWidget(self.startButton)

        self.stopButton = QtWidgets.QPushButton('Stop')
        self.stopButton.setObjectName('stopButton')
        self.stopButton.setEnabled(False)
        row2.addWidget(self.stopButton)

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

    # ── run-state gating (self-contained helper) ──
    def set_run_active(self, active):
        """During a run, lock mode/Batch/cores/Live; keep Stop and the action
        button (now 'Pause') usable.  On exit, the active wrangler's
        ``_on_mode_changed`` restores the per-mode widget state.

        NOTE: currently NOT wired in production — run-locking is owned by
        ``staticWidget._enter_run_state`` (which gates these controls directly,
        plus Start/Stop/Advanced) and the wrangler's control alias.  Kept as a
        self-contained helper (and unit-tested); do not also call it from the
        run-state path or the controls would be double-gated."""
        active = bool(active)
        for w in (self.modeCombo, self.batchButton, self.coresSpin,
                  self.liveButton):
            w.setEnabled(not active)
        if active:
            self.stopButton.setEnabled(True)

    def set_run_row_visible(self, visible):
        """Show/hide the ACTION row (Live/Start/Stop) + its divider.  Hidden in
        file-viewer modes (no run), shown in processing modes.  Per-button
        visibility within the row (e.g. Live hidden for NeXus) is owned by
        apply_profile and preserved across this toggle."""
        visible = bool(visible)
        self._divider.setVisible(visible)
        self.actionRow.setVisible(visible)

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
        # A freshly-attached wrangler starts with its action row shown; the
        # wrangler's _on_mode_changed re-hides it for file-viewer modes.  This
        # also un-hides it after swapping away from an image-viewer mode that
        # had hidden it.
        self.set_run_row_visible(True)

    # ── value getters (read at run-setup) ──
    def get_cores(self):
        return int(self.coresSpin.value())

    def is_batch(self):
        return bool(self.batchButton.isChecked())

    def is_live(self):
        return bool(self.liveButton.isChecked())

    def current_mode(self):
        return self.modeCombo.currentText()
