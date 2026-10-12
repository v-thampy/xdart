"""The Tools panel's Reference Phases window, owned by ScientificView (REFERENCE-PHASES-20261011).

It mounts Int's shared phase source in its working-list mode (calibrants, library entries, new CIFs and lattices,
each with an include check and a minimum-intensity row), shows the view's rings notice on the source's note line,
and applies every edit at once through the host's plot-option commands. It keeps no selection of its own: it
reconciles from the accepted options, and orders its commands so each intermediate value is valid (a removed
phase's minimum is pruned before the phase leaves the list; a new phase joins the list before its minimum).
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from .shell_values import ScientificPlotOptions

WINDOW_TOOLTIP = ('Reference phases drawn on the plots: built-in calibrants, CIF phases and lattices, each with '
                  'its own colour, legend entry and minimum intensity. Display only.')
RINGS_TOOLTIP = ('Whether current-Run detector rings can be drawn for the listed phases. Reference lines on q or '
                 '2θ plots stay available.')


class ReferencePhasesWindow(QtWidgets.QDialog):
    """A projection-reconciled editor of the reference-phase options; edits apply live."""

    #: one plot-option edit: (command path, value)
    optionEdited = QtCore.Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        from xdart.gui.analysis.phase_source import PhaseCutoffs, PhaseSourceWidget
        self._cutoff_type = PhaseCutoffs
        self.setObjectName('referencePhasesWindow')
        self.setWindowTitle('Reference Phases')
        self.setModal(False)
        self.setToolTip(WINDOW_TOOLTIP)
        self._projection = ScientificPlotOptions()
        self._reconciling = False
        layout = QtWidgets.QVBoxLayout(self)
        self.phase_source = PhaseSourceWidget(self, checkable=True, show_library=False, calibrants=True,
                                              cutoffs=True)
        layout.addWidget(self.phase_source)
        #: the rings notice is the shared source's note line: the window builds no display of its own (Rule 14)
        self.rings_note = self.phase_source.note
        self.phase_source.set_note('', RINGS_TOOLTIP)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.close_button = QtWidgets.QPushButton('Close', self)
        self.close_button.setToolTip('Close this window; every edit is already applied.')
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)
        self.close_button.clicked.connect(self.close)
        self.phase_source.changed.connect(self._selection_edited)
        self.phase_source.cutoffsChanged.connect(self._cutoffs_edited)

    @property
    def projection(self) -> ScientificPlotOptions:
        return self._projection

    def reconcile(self, options: ScientificPlotOptions) -> None:
        if type(options) is not ScientificPlotOptions:
            raise TypeError('plot options must be an exact frozen value')
        self._reconciling = True
        try:
            self._projection = options
            self.phase_source.set_selection(options.phase_references)
            self.phase_source.set_cutoffs(self._cutoff_type(options.phase_global_floor, options.phase_min_intensity,
                                                            options.phase_floors))
        finally:
            self._reconciling = False

    def set_available(self, available: bool, reason: str = '', *, visible: bool = True) -> None:
        self.phase_source.set_available(available, reason)
        if not visible:
            self.hide()

    def set_rings_reason(self, reason: str) -> None:
        self.phase_source.set_note(reason, RINGS_TOOLTIP)

    def showEvent(self, event) -> None:
        # Opening the window lists entries other tools added; reconciles never read the library.
        self.phase_source.refresh_library()
        super().showEvent(event)

    def _emit(self, field, value) -> None:
        self.optionEdited.emit(('other', field), value)

    def _selection_edited(self, selection) -> None:
        if self._reconciling:
            return
        # Read everything first: a host applies each command at once, and its reconcile reprojects this window.
        options, selection, floors = self._projection, tuple(selection), self.phase_source.cutoffs.per_phase
        listed = {identifier for identifier, _visible in selection}
        pruned = tuple(pair for pair in options.phase_floors if pair[0] in listed)
        if pruned != options.phase_floors:
            self._emit('phase_floors', pruned)  # before the phase leaves the list
        self._emit('phase_references', selection)
        if floors != pruned:
            self._emit('phase_floors', floors)  # a new phase's own minimum, after it joined the list

    def _cutoffs_edited(self, cutoffs) -> None:
        if self._reconciling:
            return
        options = self._projection
        if cutoffs.global_value != options.phase_min_intensity:
            self._emit('phase_min_intensity', cutoffs.global_value)
        if cutoffs.per_phase != options.phase_floors:
            self._emit('phase_floors', cutoffs.per_phase)
        if cutoffs.global_on != options.phase_global_floor:
            self._emit('phase_global_floor', cutoffs.global_on)
