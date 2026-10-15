"""Read-only calibration consumers of Int's full controls and display owners."""
from __future__ import annotations

import math

from pyqtgraph.Qt import QtCore

from xdart.gui.widgets.controls_panel import ActionButton, FormRow, SectionCard
from xdart.gui.widgets.read_only_text import ReadOnlyTextPane, ReadOnlyTextWindow
from xdart.gui.tabs.scattering.results_table import ResultsTable
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.shell_values import KeyedFigureProjection, KeyedPanelsProjection
from xrd_tools.session.display_logic import PanelKey, PanelRole
from xrd_tools.session.readiness import ControlAction, ControlActionSpec, SectionId
from xrd_tools.core.geometry.camera import mounting_description


def show_figures(view, figures):
    keys = tuple(PanelKey(PanelRole.FIGURE, f'geometry-{i}') for i in range(len(figures)))
    view.reconcile_keyed(KeyedPanelsProjection(tuple(tuple(keys[i:i+2]) for i in range(0, len(keys), 2)),
        tuple(KeyedFigureProjection(k, f.title, f) for k, f in zip(keys, figures))))


def action(label, tip, parent):
    button = ActionButton(ControlActionSpec(ControlAction.PREVIEW, label, SectionId.EXPERIMENT), parent)
    button.set_purpose_tooltip(tip)
    button.setAutoDefault(False)
    return button


class OrientationPanel(SectionCard):
    """One advisory panel; the dialog owns execution, identity and editor focus."""

    def __init__(self, parent, *, run, cancel, camera_focus, sign_focus, feature):
        super().__init__('Check detector orientation', parent, accent='experiment', collapsible=False)
        self.set_help_topic('geometry.orientation')
        self.setToolTip('Read-only orientation checks with fixed geometry; edit or refit only in the existing controls.')
        self.status.setToolTip('Eligibility and lifecycle of this read-only orientation check.')
        self.finding = None
        #: the selected hypothesis's (camera, senses), kept across a re-run so a reordered finding keeps it
        self._selected = None
        #: whether a check can start, and why not (the dialog's readiness fact, LT-008), and whether one runs
        self._ready, self._reason, self._busy = True, '', False
        self.frame_row = FormRow(label='Frame', path=('orientation', 'frame'), value=None, kind='combo')
        self.frame_row.set_purpose_tooltip('Choose a staged calibration frame by its identity and motor pose.')
        self.step_row = FormRow(label='Step', path=('orientation', 'step'), value='Beam and rings', kind='combo',
            choices=('Beam and rings', 'Motion and signs', 'Known feature (optional)', 'Finding'))
        self.step_row.set_purpose_tooltip('Select the check to run; a known feature requires its declared lab direction.')
        self.candidate_row = FormRow(label='Candidate', path=('orientation', 'candidate'), value=None, kind='combo')
        self.candidate_row.set_purpose_tooltip('Inspect a camera and motor-sense hypothesis; this never applies it.')
        self.feature_row = FormRow(label='Known feature direction', path=('orientation', 'feature'),
            value='Undeclared', kind='combo', choices=('Undeclared', 'x+', 'x-', 'z+', 'z-'))
        self.feature_row.set_purpose_tooltip('Declare an independently known transverse lab direction, then click that feature in the image; binned clicks map to native pixels and clamp at partial edges. Powder rings do not establish this direction.')
        self.figure_view = ScientificView(self)
        self.figure_view.setToolTip('Observed frame and rings use display binning for large detectors; masked/gap pixels are excluded where a detector mask exists and blank blocks are unknown. Geometry and scoring remain native; right-click for Export.')
        tips = {0: 'Camera hypothesis, looking downstream; not an applied setting.',
                1: 'Candidate del motor sense.', 2: 'Candidate nu motor sense.',
                3: 'Unitless ring-versus-background AUC; not a probability.',
                4: 'Conservative segment standard error; excludes geometry uncertainty.',
                5: 'Mean eligible per-ring angular support; unknown regions are excluded.'}
        self.table = ResultsTable(('Mounting', 'del', 'nu', 'AUC', 'SE', 'Support'), self,
                                   header_tooltips=tips, max_height=180)
        self.table.setToolTip('All hypotheses, support and conditional scores; a tie never chooses an arbitrary winner.')
        self.finding_pane = ReadOnlyTextPane(self)
        self.finding_pane.setToolTip('Read-only finding, named ambiguities, refusals and the physical check needed.')
        self.finding_pane.setMaximumHeight(110)
        self.run_button = action('Run orientation check', 'Use staged frames or the existing direct-beam fit off-thread; no calibration values change.', self)
        self.cancel_button = action('Cancel check', 'Cancel through the calibration worker and retain previously completed results.', self)
        self.show_camera_button = action('Show in Camera editor', 'Resolve the held help draft and focus the existing Camera editor; apply nothing.', self)
        self.show_sign_button = action('Show motor sign', 'Resolve the held help draft and focus the existing scanned circle direction; apply nothing.', self)
        for widget in (self.frame_row, self.step_row, self.candidate_row, self.feature_row, self.figure_view,
                       self.table, self.finding_pane, self.run_button, self.cancel_button,
                       self.show_camera_button, self.show_sign_button):
            self.body_layout.addWidget(widget)
        self.run_button.clicked.connect(run)
        self.cancel_button.clicked.connect(cancel)
        self.show_camera_button.clicked.connect(camera_focus)
        self.show_sign_button.clicked.connect(sign_focus)
        self.figure_view.figureImageClicked.connect(feature)
        self.feature_row.editor.currentIndexChanged.connect(lambda: self.finding_pane.set_text('Declare the known feature’s lab direction, then click it in the displayed frame.'))
        self.candidate_row.editor.currentIndexChanged.connect(self._remember_selection)
        self.cancel_button.setEnabled(False)
        self.figure_view.hide()
        self.invalidate()

    def set_frames(self, frames):
        current = self.frame_row.editor.currentData()
        blocked = QtCore.QSignalBlocker(self.frame_row.editor)
        self.frame_row.editor.clear()
        for frame in frames:
            label = f'Frame {frame.index} · '+', '.join(f'{m} {v:g}°' for m, v in frame.motors.items())
            self.frame_row.editor.addItem(label, frame.identity)
        index = self.frame_row.editor.findData(current)
        self.frame_row.editor.setCurrentIndex(max(0, index))
        del blocked

    def busy(self, value):
        self._busy = bool(value)
        self.cancel_button.setEnabled(value)
        for row in (self.frame_row, self.step_row, self.candidate_row, self.feature_row):
            row.setEnabled(not value)
        self.show_camera_button.setEnabled(not value)
        self.show_sign_button.setEnabled(not value)
        self.set_status_text('Checking…' if value else '')
        self._sync_run()

    def set_readiness(self, ready, reason=''):
        """Whether a check can start with the dialog's current inputs and, when not, why (LT-008): Run follows this
        and ``busy`` together, never ``busy`` alone."""
        self._ready, self._reason = bool(ready), str(reason)
        self._sync_run()

    def _sync_run(self):
        from dataclasses import replace
        enabled = self._ready and not self._busy
        reason = '' if enabled else self._reason or 'Wait for the running operation to finish, or cancel it.'
        self.run_button.apply_spec(replace(self.run_button.spec, enabled=enabled, reason=reason))

    def invalidate(self, *_args):
        self.finding = None
        self.table.clear_rows()
        self.finding_pane.set_text('Run a new check after changing geometry, frames or extraction inputs.')
        self.figure_view.hide()

    def _remember_selection(self, *_args):
        index = self.candidate_row.editor.currentData()
        if self.finding is not None and index is not None and 0 <= index < len(self.finding.candidates):
            candidate = self.finding.candidates[index]
            self._selected = (candidate.camera, candidate.senses)

    def show_finding(self, finding, figure=None, *, configured_senses=None):
        """Show ``finding``, keeping the selected hypothesis by its camera and senses (a re-run may order the
        candidates differently); the first candidate when it is not among them."""
        self.finding = finding
        self.finding_pane.set_text(finding.reason)
        rows, tips = [], {}
        blocked = QtCore.QSignalBlocker(self.candidate_row.editor)
        self.candidate_row.editor.clear()
        for i, c in enumerate(finding.candidates):
            support = sum(r.support for r in c.ring_scores)/len(c.ring_scores) if c.ring_scores else float('nan')
            label = mounting_description(c.camera)
            if finding.unmeasured_motor and configured_senses and all(
                    configured_senses.get(role, sense) == sense for role, sense in zip(('del', 'nu'), c.senses)):
                label += ' (keeps configured senses)'
            senses = tuple(f'unmeasured ({sense:+d} hypothesis)' if role == finding.unmeasured_motor
                           else f'{sense:+d}' for role, sense in zip(('del', 'nu'), c.senses))
            self.candidate_row.editor.addItem(f'{label} · del {senses[0]}, nu {senses[1]}', i)
            self.candidate_row.editor.setItemData(i,
                'Configured matches are a reference, not verified senses. An unmeasured circle has '
                'both sense hypotheses listed; this scan cannot choose between them.',
                QtCore.Qt.ItemDataRole.ToolTipRole)
            rows.append((label, *senses, *(f'{v:.4g}' if math.isfinite(v) else '—'
                                           for v in (c.score, c.standard_error, support))))
            for column in range(6):
                tips[i, column] = ('Hypothesis only. Explicit geometric exclusions '
                    f'{finding.exclusions.geometric}; value exclusions {finding.exclusions.value}. '
                    'Detector gaps are excluded from ring support. Fixed inputs are not refitted. '
                    'Configured matches are a reference, not verified senses; unmeasured senses are hypotheses.')
        index = next((i for i, c in enumerate(finding.candidates) if (c.camera, c.senses) == self._selected), 0)
        self.candidate_row.editor.setCurrentIndex(index)
        del blocked
        self._remember_selection()
        self.table.show_rows(('Mounting', 'del', 'nu', 'AUC', 'SE', 'Support'), rows, cell_tooltips=tips)
        if figure is not None:
            show_figures(self.figure_view, (figure,)); self.figure_view.show()


class InstrumentSchematicWindow(ReadOnlyTextWindow):
    def __init__(self, parent):
        # the views above an adjustable split, the messages below opening five lines tall (LT-007)
        super().__init__('Instrument geometry', (1000, 820), parent, message_lines=5)
        self.legend = self.pane
        self.legend.setToolTip("The pose drawn, then each circle's motor readback, zero and resulting angle and its "
                               'stack; configured conventions, not a physical validation.')
        self.figure_view = ScientificView(self)
        self.figure_view.setToolTip('Computed side, top and downstream views from the current instrument; right-click for native Export.')
        self.insert_content(self.figure_view)
        self.projection = None

    def show_geometry(self, geometry, readings, pose='zero-readback example'):
        from xrd_tools.geometry_figures import instrument_figure
        self.projection = instrument_figure(geometry, readings, pose=pose)
        if self.projection.reason:
            self.legend.set_text(self.projection.reason); self.figure_view.hide()
        else:
            lines = ['Explicit zero-readback example: every motor reads 0°; not a measured pose.'
                     if pose == 'zero-readback example' else f'Drawn at the {pose}.']
            lines += [f'{c.name} ({c.role}) · motor {c.motor}: readback {c.readback:g}° − zero {c.zero:g}° = '
                      f'{c.angle_deg:g}° about {c.axis}' for c in self.projection.circles]
            self.legend.set_text('\n'.join(lines))
            show_figures(self.figure_view, self.projection.panels); self.figure_view.show()
