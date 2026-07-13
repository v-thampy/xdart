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


class _ElidingLabel(QtWidgets.QLabel):
    """Single-line label whose text never contributes horizontal size pressure."""

    def __init__(self, text='', parent=None):
        super().__init__(text, parent)
        self._full_text = str(text or '')
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )

    def set_full_text(self, text):
        self._full_text = str(text or '')
        self._refresh_elide()

    def full_text(self):
        return self._full_text

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_elide()

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(0)
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def _refresh_elide(self):
        width = max(0, self.contentsRect().width() - 2)
        metrics = self.fontMetrics()
        text = metrics.elidedText(
            self._full_text,
            QtCore.Qt.TextElideMode.ElideRight,
            width,
        )
        if self.text() != text:
            super().setText(text)


class StaticControls(QtWidgets.QWidget):
    """Mode selector + Batch + cores + Live + Start/Pause/Resume + Stop."""

    actionClicked = QtCore.Signal()          # the single 3-state action button
    stopClicked = QtCore.Signal()
    modeChanged = QtCore.Signal(str)
    batchToggled = QtCore.Signal(bool)
    liveToggled = QtCore.Signal(bool)
    writeModeChanged = QtCore.Signal(str)    # 'Append' / 'Overwrite' (output mode)

    # Phase-B action-button morph table (text, runPhase property, enabled).
    # Idle reads "Run" (the run trigger); the running/paused phases keep the
    # Pause/Resume morph.
    _PHASES = {
        'idle':    ('▶ Run',      'idle',   True),
        'running': ('❚❚ Pause',    'active', True),
        'pausing': ('❚❚ Pausing…', 'active', False),
        'paused':  ('▶ Resume',   'active', True),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('staticRunControls')
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

        # Run/browse status now shows in the main window's BOTTOM status bar
        # (wranglerWidget._set_status_text routes there), not in a strip above the
        # mode row.  Keep this label — parented + hidden, NOT in the layout — only
        # for back-compat references and the standalone fallback; dropping it from
        # the layout frees the vertical space it used to occupy at the top.
        self.statusLabel = QtWidgets.QLabel('', self)
        self.statusLabel.setObjectName('statusLabel')
        self.statusLabel.hide()

        self.readinessRow = QtWidgets.QWidget()
        self.readinessRow.setObjectName('runReadinessRow')
        readiness = QtWidgets.QHBoxLayout(self.readinessRow)
        readiness.setContentsMargins(2, 0, 2, 0)
        readiness.setSpacing(6)
        self.readinessDot = QtWidgets.QLabel('●')
        self.readinessDot.setObjectName('runReadinessDot')
        self.readinessDot.setAlignment(QtCore.Qt.AlignCenter)
        readiness.addWidget(self.readinessDot)
        # "Live" chip (maintainer, 2026-07-13): shown while the Live toggle is
        # on so the readiness line reads "● Live Ready · Int 2D · N frames".
        # A separate label because the summary label ELIDES plain text (UI-4)
        # and cannot carry rich-text styling mid-string.
        self.readinessLive = QtWidgets.QLabel('Live')
        self.readinessLive.setObjectName('runReadinessLive')
        self.readinessLive.setStyleSheet(
            'QLabel#runReadinessLive { color: #ffb86c; font-weight: 700; }')
        self.readinessLive.hide()
        readiness.addWidget(self.readinessLive)
        self.readinessLabel = _ElidingLabel('')
        self.readinessLabel.setObjectName('runReadinessLabel')
        readiness.addWidget(self.readinessLabel, 1)
        self.readinessRow.hide()
        outer.addWidget(self.readinessRow)

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

        # Row order: Live · Run · Stop · Append (Vivek).  Live + Stop are compact
        # icon buttons (◉ / ■, reduced width); Run (▶) is the wide primary action;
        # Append/Replace (⇄) is a touch wider for its label.  Symbols carry
        # tooltips so the icon-only buttons stay discoverable.
        self.liveButton = QtWidgets.QPushButton('◉')
        self.liveButton.setObjectName('liveCheckBox')
        self.liveButton.setCheckable(True)
        self.liveButton.setToolTip('Live (per-frame display)')
        self.liveButton.setMaximumWidth(50)

        # objectName 'startButton' + the runPhase property are what the dark
        # theme keys the green/orange Run/Pause styling on.  Text comes from
        # _PHASES (▶ Run / ❚❚ Pause / ▶ Resume) via set_action_phase.
        self.startButton = QtWidgets.QPushButton('▶ Run')
        self.startButton.setObjectName('startButton')
        self.startButton.setProperty('runPhase', 'idle')

        self.stopButton = QtWidgets.QPushButton('■')
        self.stopButton.setObjectName('stopButton')
        self.stopButton.setToolTip('Stop')
        self.stopButton.setEnabled(False)
        self.stopButton.setMaximumWidth(50)

        # Output mode (Append vs Replace): a run/output property.  Checkable;
        # Replace (== writer 'Overwrite') is destructive.  The displayed label is
        # 'Append'/'Replace' (+ a ⇄ glyph); the writer value stays
        # 'Append'/'Overwrite' (see write_mode).
        self.writeModeButton = QtWidgets.QPushButton('Append ⇄')
        self.writeModeButton.setObjectName('writeModeButton')
        self.writeModeButton.setCheckable(True)
        self.writeModeButton.setChecked(False)            # Append
        self.writeModeButton.setMinimumWidth(96)

        row2.addWidget(self.liveButton)          # compact icon, stretch 0
        row2.addWidget(self.startButton, 6)       # wide primary action
        row2.addWidget(self.stopButton)          # compact icon, stretch 0
        row2.addWidget(self.writeModeButton, 2)   # ~20% wider than the icons' share

        self._run_phase = 'idle'

        # Emit intent only — the active wrangler owns the logic.
        self.startButton.clicked.connect(self.actionClicked)
        self.stopButton.clicked.connect(self.stopClicked)
        self.modeCombo.currentTextChanged.connect(self.modeChanged)
        self.batchButton.toggled.connect(self.batchToggled)
        self.liveButton.toggled.connect(self.liveToggled)
        self.writeModeButton.toggled.connect(self._on_write_mode_toggled)

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

    def set_readiness_summary(self, text='', *, ready=True, tooltip='',
                              live=False):
        text = str(text or '').strip()
        was_hidden = self.readinessRow.isHidden()
        self.readinessRow.setVisible(bool(text))
        self.readinessLive.setVisible(bool(text) and bool(live))
        self.readinessDot.setProperty('ready', bool(ready))
        setter = getattr(self.readinessLabel, 'set_full_text', None)
        if callable(setter):
            setter(text)
        else:
            self.readinessLabel.setText(text)
        self.readinessLabel.setToolTip(tooltip or text)
        self.readinessDot.style().unpolish(self.readinessDot)
        self.readinessDot.style().polish(self.readinessDot)
        return was_hidden != self.readinessRow.isHidden()

    def set_stop_enabled(self, enabled):
        self.stopButton.setEnabled(bool(enabled))

    # ── output mode (Append / Overwrite) ──
    # The button DISPLAYS 'Append'/'Replace'; the VALUE the writer compares
    # against stays 'Append'/'Overwrite' (write_mode).  Keep the two apart.
    def write_mode(self):
        """The current output mode string the writer expects."""
        return 'Overwrite' if self.writeModeButton.isChecked() else 'Append'

    def set_write_mode(self, mode):
        """Set the toggle from a mode string (session restore / external sync).
        Blocks signals so a programmatic set doesn't re-emit writeModeChanged."""
        checked = str(mode) == 'Overwrite'
        if self.writeModeButton.isChecked() != checked:
            self.writeModeButton.blockSignals(True)
            self.writeModeButton.setChecked(checked)
            self.writeModeButton.blockSignals(False)
        self.writeModeButton.setText('Replace ⇄' if checked else 'Append ⇄')

    def _on_write_mode_toggled(self, checked):
        self.writeModeButton.setText('Replace ⇄' if checked else 'Append ⇄')
        self.writeModeChanged.emit(self.write_mode())

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
                  self.liveButton, self.writeModeButton):
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

    def set_run_row_enabled(self, enabled):
        """Enable/disable the whole ACTION row (Live/Start/Stop) while keeping it
        visible.  Used for not-ready processing states; file-viewer modes hide
        the row altogether.  Within an enabled row the per-button states (Stop
        disabled at idle, etc.) still apply."""
        self.actionRow.setEnabled(bool(enabled))

    def set_mode_row_enabled(self, enabled):
        """Enable/disable the run-config controls: mode combo + Batch + Cores + the
        Append/Replace write-mode toggle.  Locked while a run is in progress --
        including a reintegrate, which (unlike a wrangler run) does not go through
        wrangler.enabled().  The ACTION row (Pause/Resume/Stop) is left alone so
        those stay usable.  writeModeButton physically sits in the action row but is
        a run-config control, so it locks here: you can't flip Append<->Replace
        mid-scan."""
        enabled = bool(enabled)
        for w in (self.modeCombo, self.batchButton, self.coresSpin,
                  self.coresLabel, self.writeModeButton):
            w.setEnabled(enabled)

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
