"""The reference-phase plot options bound to a mounted phase source (REFERENCE-PHASES-20261011; PHASE-ANALYSIS-128).

The retired Reference Phases window's code, now a binding the Phase Analysis window mounts over its shared phase
source (calibrants, library entries, new CIFs and lattices, each with an include check and a minimum-intensity
row). It keeps no selection of its own: it reconciles the source from the accepted options, shows the view's rings
notice on the source's note line, and applies every edit at once as plot-option edits, ordered so each intermediate
value is valid (a removed phase's minimum is pruned before the phase leaves the list; a new phase joins the list
before its minimum). ScientificView attaches it (``attach_phase_list``) and drives its availability and notice.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore

from .shell_values import ScientificPlotOptions

RINGS_TOOLTIP = ('Whether current-Run detector rings can be drawn for the listed phases. Reference lines on q or '
                 '2θ plots stay available.')


class ReferencePhaseBinding(QtCore.QObject):
    """A projection-reconciled binding of the reference-phase options to one phase source; edits apply live."""

    #: one plot-option edit: (command path, value)
    optionEdited = QtCore.Signal(object, object)
    #: an explicit library refresh of the source (open, a library edit, an add): the view's lookups are stale
    refreshRequested = QtCore.Signal()

    def __init__(self, source, parent=None):
        super().__init__(parent)
        from xdart.gui.analysis.phase_source import PhaseCutoffs
        self._cutoff_type = PhaseCutoffs
        self.source = source
        self._projection = ScientificPlotOptions()
        self._reconciling = False
        #: the rings notice is the shared source's note line: the binding builds no display of its own (Rule 14)
        self.rings_note = source.note
        source.set_note('', RINGS_TOOLTIP)
        source.changed.connect(self._selection_edited)
        source.cutoffsChanged.connect(self._cutoffs_edited)
        source.libraryRefreshed.connect(self._library_refreshed)

    @property
    def projection(self) -> ScientificPlotOptions:
        return self._projection

    def reconcile(self, options: ScientificPlotOptions) -> None:
        if type(options) is not ScientificPlotOptions:
            raise TypeError('plot options must be an exact frozen value')
        self._reconciling = True
        try:
            self._projection = options
            self.source.set_selection(options.phase_references)
            self.source.set_cutoffs(self._cutoff_type(options.phase_global_floor, options.phase_min_intensity,
                                                      options.phase_floors))
        finally:
            self._reconciling = False

    def set_available(self, available: bool, reason: str = '', *, visible: bool = True) -> None:
        """The list's availability (the 2D Viewer, keyed views); ``visible`` is kept for the view's callers."""
        self.source.set_available(available, reason)

    def set_rings_reason(self, reason: str) -> None:
        self.source.set_note(reason, RINGS_TOOLTIP)

    def submit_cutoffs(self, cutoffs) -> None:
        """Apply a minimum-intensity choice made outside the rows (Load settings), as an edit of the options."""
        self.source.set_cutoffs(cutoffs)
        self._cutoffs_edited(cutoffs)

    def _library_refreshed(self) -> None:
        if not self._reconciling:
            self.refreshRequested.emit()

    def _emit(self, field, value) -> None:
        self.optionEdited.emit(('other', field), value)

    def _selection_edited(self, selection) -> None:
        if self._reconciling:
            return
        # Read everything first: a host applies each command at once, and its reconcile reprojects this binding.
        options, selection, floors = self._projection, tuple(selection), self.source.cutoffs.per_phase
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
