"""Offline help routing and one modeless shared-text window per participating root.

Tags are passive; buttons resolve their top-level owner at click time so retained
pages can be mounted in Main without creating a second shortcut or window.
"""
from __future__ import annotations

import logging
import os

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xrd_tools.help_topics import TOPICS, get_topic, search_topics
from .read_only_text import ReadOnlyTextWindow


def bind_topic(widget, topic_id: str, *, details=None) -> None:
    """Tag a shared control; optional details reads only already displayed strings."""
    widget.setProperty("helpTopic", str(topic_id))
    if details is not None:
        widget._help_details = details


def _context(widget, root):
    topic = None
    details = ""
    while widget is not None:
        if not details and hasattr(widget, "_help_details"):
            details = str(widget._help_details())
        if topic is None and widget.property("helpTopic"):
            topic = str(widget.property("helpTopic"))
        if widget is root:
            break
        widget = widget.parentWidget()
    return topic, details


def preserve_focused_draft(root):
    """Help focus transfer is presentation, not a FormRow editingFinished commit."""
    widget = QtWidgets.QApplication.focusWidget()
    while widget is not None and widget.window() is root:
        if hasattr(widget, "preserve_help_draft"):
            widget.preserve_help_draft()
            break
        widget = widget.parentWidget()


def _help_action(widget):
    from .controls_panel import ActionButton
    from xrd_tools.session.readiness import ControlAction
    return isinstance(widget, ActionButton) and widget.spec.action is ControlAction.HELP


class HelpWindow(ReadOnlyTextWindow):
    """Compose Int's text, keyed browser and form row; no alternative editors."""

    def __init__(self, parent):
        super().__init__("Scientific help", (920, 700), parent)
        # Lazy composition avoids a module-level widgets → tabs/scattering edge.
        from .controls_panel import FormRow
        from xdart.gui.tabs.scattering.browser_view import BrowserResult, BrowserView, keyed_list_profile
        from xdart.gui.tabs.scattering.shell_values import ShellCommandKind
        self._entry_type = BrowserResult
        self.topic_id = None
        self.figure_view = None
        self._walkthrough = False
        self._synthetic_findings = {}
        self._details = ''
        self.search_row = FormRow(label="Search help", path=("help", "search"), value="")
        self.search_row.set_purpose_tooltip("Search topic IDs, titles and text; every entered word must match.")
        self.browser = BrowserView(self, profile=keyed_list_profile("Help topics"))
        self.browser.scans_label.setToolTip("Select an offline topic to read its instructions.")
        self.browser.scans.setToolTip("Matching topics in catalogue order; selecting one changes only help text.")
        self.browser.setMaximumHeight(160)
        self.pane.setToolTip("Read, select and copy these instructions and literal diagnostics; text cannot be edited.")
        self.insert_content(self.search_row)
        self.insert_content(self.browser, 1)
        self.search_row.editor.textChanged.connect(self._search)
        self.browser.commandRequested.connect(lambda command:
            self.show_topic(command.value) if command.kind is ShellCommandKind.SELECT_RESULT else None)
        self._f1 = QtGui.QShortcut(QtGui.QKeySequence("F1"), self)
        self._f1.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._f1.activated.connect(self.show_index)
        self.pane.installEventFilter(self)
        self.show_index()

    def eventFilter(self, watched, event):
        # Read-only help keeps F1 local, including a directly delivered key
        # while another native popup/window is settling its activation.
        if watched is self.pane and event.type() in (QtCore.QEvent.Type.ShortcutOverride,
                                                     QtCore.QEvent.Type.KeyPress):
            if event.key() == QtCore.Qt.Key.Key_F1:
                event.accept()
                if event.type() == QtCore.QEvent.Type.KeyPress:
                    self.show_index()
                return True
        return super().eventFilter(watched, event)

    def _search(self, query: str) -> None:
        topics = search_topics(query)
        self.browser.reconcile_results(tuple(self._entry_type(t.identifier, t.title, t.identifier)
                                             for t in topics), self.topic_id)

    def show_index(self, requested: str | None = None, details: str = "") -> None:
        self.topic_id = None
        if self.figure_view is not None:
            for widget in self._figure_widgets:
                widget.hide()
        self.search_row.editor.clear()
        self._search("")
        text = "Scientific help\n\nSearch or select a topic. F1 opens help for the focused control.\n\n"
        if requested:
            text += f"Unknown topic: {requested}\n\n"
        text += "\n".join(f"{t.title} — {t.identifier}" for t in TOPICS)
        self.set_text(text + ("\n\nCurrent context\n" + details if details else ""))
        self.search_row.editor.setFocus()

    def show_topic(self, identifier: str, details: str = "") -> None:
        topic = get_topic(identifier)
        if topic is None:
            self.show_index(identifier, details)
            return
        self.topic_id = identifier
        self._details = details
        self._walkthrough = identifier == 'geometry.orientation'
        self._search(self.search_row.editor.text())
        self.set_text(topic.title + "\n" + topic.identifier + "\n\n" + topic.text +
                      ("\n\nCurrent context\n" + details if details else ""))
        if topic.figures:
            self._ensure_figures()
            from xrd_tools.geometry_figures import HELP_FIGURE_TITLES
            labels = dict(HELP_FIGURE_TITLES)
            with QtCore.QSignalBlocker(self.figure_row.editor):
                self.figure_row.editor.clear()
                self.figure_row.editor.addItems([labels[k] for k in topic.figures])
            for widget in self._figure_widgets:
                widget.show()
            self.step_row.setVisible(self._walkthrough)
            self.feature_row.setVisible(False)
            self.score_table.setVisible(False)
            self.previous_button.setVisible(self._walkthrough)
            self.next_button.setVisible(self._walkthrough)
            self.check_button.setVisible(identifier == 'geometry.orientation')
            self.run_button.setVisible(bool(topic.run_key))
            capability = getattr(self.parentWidget(), 'open_orientation_check', None)
            self.run_button.setEnabled(bool(topic.run_key and callable(capability)))
            self.run_button.set_purpose_tooltip('Open the read-only orientation check in the originating calibration window; '
                'open Calibrate first when this action is unavailable. Opening does not start a job.')
            self._draw_figure()
        elif self.figure_view is not None:
            for widget in self._figure_widgets:
                widget.hide()

    def _ensure_figures(self):
        if self.figure_view is not None:
            return
        from .controls_panel import FormRow, ActionButton
        from xrd_tools.session.readiness import ControlActionSpec, ControlAction, SectionId
        from xdart.gui.tabs.scattering.scientific_view import ScientificView
        from xdart.gui.tabs.scattering.results_table import ResultsTable
        self.figure_view = ScientificView(self)
        self.figure_view.setMinimumHeight(260)
        self.figure_view.setToolTip('Computed synthetic figures in declared coordinates; native right-click Export is available.')
        self.figure_row = FormRow(label='Figure', path=('help', 'figure'), kind='combo', value='')
        self.figure_row.set_purpose_tooltip('Choose a computed example for this topic. No experimental file is read.')
        self.step_row = FormRow(label='Orientation check', path=('help', 'step'), kind='combo',
            value='1 · Zero-angle rings', choices=('1 · Zero-angle rings', '2 · Moved-arm evidence',
                                                  '3 · Known-feature direction', '4 · Physical checks and ties'))
        self.step_row.set_purpose_tooltip('Step through a synthetic orientation check; run on your own frames in calibration.')
        from xrd_tools.geometry_figures import camera_candidates
        self.candidate_row = FormRow(label='Candidate mounting', path=('help', 'camera'), kind='combo',
            choices=tuple(c.label for c in camera_candidates()), value=camera_candidates()[0].label)
        self.candidate_row.set_purpose_tooltip('Select a camera hypothesis to compare with the fixed synthetic truth; this changes only the illustration.')
        self.feature_row = FormRow(label='Known feature direction', path=('help', 'feature'), kind='combo',
            value='Undeclared', choices=('Undeclared', 'x+', 'x-', 'z+', 'z-'))
        self.feature_row.set_purpose_tooltip('Declare the synthetic feature’s transverse lab direction independently, then click it; this does not certify a mounting.')
        self.score_table = ResultsTable(('Mounting', 'del', 'nu', 'AUC', 'SE'), self,
            header_tooltips={i: tip for i, tip in enumerate(('Camera hypothesis; never applied.',
                'Candidate del sense.', 'Candidate nu sense.', 'Ring/background AUC, not probability.',
                'Conditional segment standard error; geometry uncertainty is excluded.'))}, max_height=150)
        self.score_table.setToolTip('Computed synthetic scores, with fixed inputs and explicit ties; no file is read.')
        self.previous_button = ActionButton(ControlActionSpec(ControlAction.HELP, 'Previous step', SectionId.EXPERIMENT), self)
        self.next_button = ActionButton(ControlActionSpec(ControlAction.HELP, 'Next step', SectionId.EXPERIMENT), self)
        self.check_button = ActionButton(ControlActionSpec(ControlAction.HELP, 'Check detector orientation', SectionId.EXPERIMENT), self)
        self.run_button = ActionButton(ControlActionSpec(ControlAction.HELP, 'Run on my frames', SectionId.EXPERIMENT), self)
        for button, tip in ((self.previous_button, 'Show the previous synthetic check step.'),
                            (self.next_button, 'Show the next synthetic check step.'),
                            (self.check_button, 'Start the four-step synthetic orientation walkthrough.'),
                            (self.run_button, 'Open the check in the originating calibration window.')):
            button.set_purpose_tooltip(tip)
        self._figure_widgets = (self.figure_row, self.step_row, self.candidate_row, self.feature_row, self.score_table,
            self.check_button, self.previous_button, self.next_button, self.run_button, self.figure_view)
        for index, widget in enumerate(self._figure_widgets, 2):
            self.insert_content(widget, index)
        self.figure_row.editor.currentIndexChanged.connect(self._draw_figure)
        self.candidate_row.editor.currentIndexChanged.connect(self._draw_figure)
        self.step_row.editor.currentIndexChanged.connect(self._draw_step)
        self.previous_button.clicked.connect(lambda: self.step_row.editor.setCurrentIndex(max(0, self.step_row.editor.currentIndex()-1)))
        self.next_button.clicked.connect(lambda: self.step_row.editor.setCurrentIndex(min(3, self.step_row.editor.currentIndex()+1)))
        self.check_button.clicked.connect(self.show_orientation_check)
        self.run_button.clicked.connect(self._run_on_frames)
        self.figure_view.figureImageClicked.connect(self._feature_clicked)
        self.feature_row.editor.currentIndexChanged.connect(lambda: self._feature_clicked(None, None, None))

    def show_orientation_check(self):
        self.show_topic('geometry.orientation', self._details)
        self._walkthrough = True
        self.step_row.editor.setCurrentIndex(0)
        self._draw_step()

    def _draw_step(self, *_):
        if self.figure_view is None or not self._walkthrough:
            return
        step = self.step_row.editor.currentIndex()
        key = ('off-centre', 'mirror-tie', 'off-centre', 'centred')[step]
        topic = get_topic(self.topic_id)
        if topic is None:
            return
        self.set_text(topic.title+'\n\n'+topic.text+'\n\nSynthetic check\n'+
            ('Compare fixed-geometry ring candidates; trust the beam, wavelength, zeros and mount first.',
             'Moved rings constrain motor sense only together with camera mounting. Mirror-line flips can remain tied.',
             'Declare an asymmetric feature’s lab direction before checking χ. Powder rings alone cannot certify azimuth.',
             'A centred or mirror/diagonal frame may not identify the mounting. Check the physical detector and obtain separating evidence.')[self.step_row.editor.currentIndex()])
        self._draw_figure_key(key)
        self.feature_row.setVisible(step == 2)
        from xrd_tools.geometry_figures import synthetic_orientation_finding
        if step not in self._synthetic_findings:
            self._synthetic_findings[step] = synthetic_orientation_finding(step)
        finding = self._synthetic_findings[step]
        rows = tuple((c.camera.rows+'/'+c.camera.columns, *c.senses,
                      f'{c.score:.4g}', f'{c.standard_error:.4g}') for c in finding.candidates)
        self.score_table.show_rows(('Mounting', 'del', 'nu', 'AUC', 'SE'), rows,
            cell_tooltips={(r, c): 'Synthetic fixed-input hypothesis. '+finding.reason
                           for r in range(len(rows)) for c in range(5)})
        self.score_table.setVisible(step != 2)
        self.pane.appendPlainText('\n'+finding.reason)

    def _feature_clicked(self, _key, column, row):
        if not self._walkthrough or self.step_row.editor.currentIndex() != 2:
            return
        from xrd_tools.geometry_figures import GeometryInputs, camera_candidates, feature_azimuth
        direction = self.feature_row.editor.currentText()
        if column is None or direction == 'Undeclared':
            self.pane.appendPlainText('Declare the known feature’s lab direction, then click it in the image.')
            return
        result = feature_azimuth(GeometryInputs(), camera_candidates()[self.candidate_row.editor.currentIndex()].camera,
                                 (column, row), expected_lab=direction)
        self.pane.appendPlainText(result.reason)

    def _draw_figure(self, *_):
        if self.figure_view is None:
            return
        topic = get_topic(self.topic_id)
        if topic and topic.figures:
            if self._walkthrough and self.sender() is self.candidate_row.editor:
                self._draw_step()
                return
            index = max(0, self.figure_row.editor.currentIndex())
            self._draw_figure_key(topic.figures[index])

    def _draw_figure_key(self, key):
        from xrd_tools.geometry_figures import help_figures
        from xrd_tools.session.display_logic import PanelKey, PanelRole
        from xdart.gui.tabs.scattering.shell_values import KeyedFigureProjection, KeyedPanelsProjection
        figures = help_figures(key, max(0, self.candidate_row.editor.currentIndex()))
        keys = tuple(PanelKey(PanelRole.FIGURE, f'help-{i}') for i in range(len(figures)))
        layout = tuple(tuple(keys[i:i+2]) for i in range(0, len(keys), 2))
        self.figure_view.reconcile_keyed(KeyedPanelsProjection(layout,
            tuple(KeyedFigureProjection(k, f.title, f) for k, f in zip(keys, figures))))

    def _run_on_frames(self):
        capability = getattr(self.parentWidget(), 'open_orientation_check', None)
        if callable(capability):
            capability()


class ContextHelpOwner(QtCore.QObject):
    """Own help and its F1 binding only for one top-level widget's lifetime."""

    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.window = None
        self.shortcut = None
        self._held_row = None
        self._focus_generation = 0
        self._resume_generation = 0
        self._resume_timer = QtCore.QTimer(self)
        self._resume_timer.setSingleShot(True)
        self._resume_timer.timeout.connect(self._resume_queued)
        root.installEventFilter(self)
        QtWidgets.QApplication.instance().focusChanged.connect(self._focus_changed)
        self._bind_shortcut()

    def _bind_shortcut(self):
        if self.shortcut is None:
            self.shortcut = QtGui.QShortcut(QtGui.QKeySequence("F1"), self.root)
            self.shortcut.activated.connect(self.show_focused)

    def _trace_focus(self, event, **details):
        if os.environ.get('XDART_DEBUG_HELP_FOCUS'):
            logging.getLogger(__name__).warning('help_owner %s', dict(event=event,
                generation=self._focus_generation, held=self._held_row,
                focus=QtWidgets.QApplication.focusWidget(), root_focus=self.root.focusWidget(), **details))

    def eventFilter(self, watched, event):
        if watched is self.root:
            if event.type() == QtCore.QEvent.Type.Hide:
                self.shutdown()
            elif event.type() == QtCore.QEvent.Type.Show:
                self._bind_shortcut()
            elif event.type() == QtCore.QEvent.Type.WindowActivate:
                # Run after Qt restores the root's former focus widget.
                self._queue_resume()
        elif watched is self.window and event.type() == QtCore.QEvent.Type.Hide:
            self.root.activateWindow()
            self._queue_resume()
        return False

    def _queue_resume(self):
        self._resume_generation = self._focus_generation
        self._trace_focus('queue_resume')
        self._resume_timer.start(0)

    def _resume_queued(self):
        generation = self._resume_generation
        self._trace_focus('queued_callback', requested_generation=generation)
        if generation == self._focus_generation:
            self._resume_draft()

    def _focus_changed(self, _previous, current):
        self._trace_focus('focus_changed', previous=_previous, current=current)
        row = self._held_row
        if row is None or current is None or current.window() is not self.root:
            return
        # Qt has now restored root focus. Resolve this transition immediately;
        # a zero timer can arrive after a synchronous root action captures its
        # form. A help-button click *within* the root still preserves the draft.
        if _previous is not None and _previous.window() is self.window:
            self._resume_draft()
            return
        if current is row.editor or _help_action(current):
            return
        self.release_draft()

    def release_draft(self):
        """Resolve the held row once before a deliberate root focus action."""
        row = self._held_row
        self._trace_focus('release_draft')
        if row is None:
            return
        self._focus_generation += 1
        self._held_row = None
        row.destroyed.disconnect(self._forget_draft)
        row.resolve_help_draft()

    def hold_draft(self, row):
        """One row reference for this visit; FormRow remains the sole value owner."""
        if self._held_row is row:
            return
        self._resume_draft(restore=False)
        self._focus_generation += 1
        self._held_row = row
        row.destroyed.connect(self._forget_draft)

    def _forget_draft(self):
        if self.sender() is self._held_row:
            self._held_row = None

    def _resume_draft(self, *, restore=True):
        row = self._held_row
        self._trace_focus('resume_draft', restore=restore)
        if row is None:
            return
        current = QtWidgets.QApplication.focusWidget()
        if (restore and current is not None and current.window() is self.root
                and current is not row.editor and not _help_action(current)):
            self.release_draft()
            return
        self._focus_generation += 1
        self._held_row = None
        row.destroyed.disconnect(self._forget_draft)
        held = row._help_focus_transfer
        row._help_focus_transfer = False
        if restore and held and row.editor.isVisible() and row.editor.isEnabled():
            row.editor.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _open(self):
        preserve_focused_draft(self.root)
        self._bind_shortcut()
        if self.window is None:
            self.window = HelpWindow(self.root)
            self.window.installEventFilter(self)
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        return self.window

    def show_index(self):
        self._open().show_index()

    def preserve_draft(self):
        preserve_focused_draft(self.root)

    def show_topic(self, identifier: str, details: str = ""):
        self._open().show_topic(identifier, details)

    def show_context(self, widget):
        topic, details = _context(widget, self.root)
        if topic:
            self.show_topic(topic, details)
        else:
            self._open().show_index(details=details)

    def show_focused(self):
        widget = QtWidgets.QApplication.focusWidget()
        self.show_context(widget if widget is not None and widget.window() is self.root else self.root)

    def shutdown(self):
        """Release callbacks/windows on actual retirement; ignored busy Close keeps both."""
        self._resume_timer.stop()
        self._resume_draft(restore=False)
        if self.shortcut is not None:
            self.shortcut.setEnabled(False)
            self.shortcut.deleteLater()
            self.shortcut = None
        if self.window is not None:
            self.window.removeEventFilter(self)
            self.window.close()
            self.window.deleteLater()
            self.window = None


def owner_for(widget) -> ContextHelpOwner:
    root = widget.window()
    owner = getattr(root, "_context_help_owner", None)
    if owner is None:
        owner = root._context_help_owner = ContextHelpOwner(root)
    return owner
