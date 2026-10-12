"""Shared CIF/library and manual-lattice source for fitting and reference overlays.

This owner mounts Int's keyed BrowserView and Controls rows. The headless phase
library owns all valid sources; selection is only an immutable tuple of IDs and
visibility flags. No reflection listing or detector prediction runs here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xdart.gui.tabs.scattering.browser_view import (BrowserResult, BrowserView, ResultValueSpec,
                                                    keyed_list_profile)
from xdart.gui.widgets.controls_panel import FormRow, SectionCard
from xdart.gui.widgets.file_dialogs import browse

CIF_FILTER = 'CIF files (*.cif);;All files (*)'
_PYMATGEN_REASON = 'Phase sources need pymatgen; install xdart[fitting].'
_ADD_CALIBRANT, _ADD_FROM_LIBRARY = 'Add calibrant…', 'Add from library…'
_ROW_TIP = 'This phase\'s minimum intensity, in percent of its strongest reachable reflection.'
_SAME_TIP = 'Same minimum for all phases is on; turn it off to give each CIF phase its own minimum.'
_NO_INTENSITY_TIP = {'lattice': 'Lattice phases have no structural intensities; every allowed position is shown.',
                     'calibrant': 'Calibrants carry positions only, with no intensities; every line is shown.'}


@dataclass(frozen=True, slots=True)
class PhaseCutoffs:
    """The minimum-intensity choice (REFERENCE-PHASES-20261011): one value for every CIF phase while
    ``global_on``, else each listed CIF phase's own ``per_phase`` value (stored values survive the switch)."""

    global_on: bool = True
    global_value: float = 5.
    per_phase: tuple[tuple[str, float], ...] = ()
_SOURCE_TIP = ('Choose structural CIF phases or enter lattice parameters. CIF entries retain their bytes in the '
               'phase library; a lattice entry predicts positions only, without structural intensities.')


class PhaseSourceWidget(QtWidgets.QWidget):
    """One shared source editor; consumers own only what its selection means."""

    changed = QtCore.Signal(object)
    messageChanged = QtCore.Signal(str)
    #: a user's minimum-intensity edit (row, global switch or global value), as the new ``PhaseCutoffs``
    cutoffsChanged = QtCore.Signal(object)

    def __init__(self, parent=None, *, library=None, checkable=True, show_library=True, key_by_index=False,
                 calibrants=False, cutoffs=False, status_visible=True):
        super().__init__(parent)
        from xrd_tools.analysis.phase_references import PhaseLibrary
        self.library = PhaseLibrary() if library is None else library
        self._selection = ()
        #: the current selection's entries as last resolved; reconciles read no library files
        self._resolved = {}
        self._show_library = bool(show_library)
        self._checkable = bool(checkable)
        self._key_by_index = bool(key_by_index)
        self._shown_ids = ()
        self._details = {}
        self._reason = ''
        self._availability_reason = ''
        self._source_reason = ''
        self._cutoffs_enabled = bool(cutoffs)
        self._cutoffs = PhaseCutoffs()
        self.setObjectName('phaseSourceWidget')
        self.setToolTip(_SOURCE_TIP)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        values = ResultValueSpec(suffix=' %', minimum=0., maximum=100., decimals=1) if cutoffs else None
        self.browser = BrowserView(self, profile=keyed_list_profile('Phases', checkable=checkable, values=values))
        self.browser.setObjectName('phaseSourceBrowser')
        self.browser.setMaximumHeight(150)
        self.browser.setToolTip(_SOURCE_TIP)
        self.browser.scans.setToolTip('Phases in library order. Check a phase to show its reference overlay.'
                                      if checkable else 'The phases selected for this fit, in order.')
        self.browser.scans_label.setToolTip('Persistent CIF and lattice entries available to this tool.')
        # The keyed profile hides Int's file actions, but they remain children of
        # the mounted owner and retain useful help if a profile reveals them.
        for control, tip in ((self.browser.show_all, 'Show every phase in this source list.'),
                             (self.browser.metadata, 'Show the selected phase source details.'),
                             (self.browser.auto_last, 'Follow the last phase added to the list.'),
                             (self.browser.date_sort, 'Sort the source list by time.'),
                             (self.browser.refresh, 'Refresh the available phase sources.')):
            control.setToolTip(tip)
        layout.addWidget(self.browser)
        buttons = QtWidgets.QHBoxLayout()
        self.add_cif_btn = QtWidgets.QPushButton('Add CIF…', self)
        self.add_cif_btn.setToolTip('Add structural CIF bytes to the phase library and select the phase.')
        self.add_lattice_btn = QtWidgets.QPushButton('Add lattice…', self)
        self.add_lattice_btn.setToolTip('Enter unit-cell lengths, angles and space group for reflection positions.')
        self.remove_btn = QtWidgets.QPushButton('Remove', self)
        self.remove_btn.setToolTip('Remove the selected phase from this tool; keep its library entry.')
        for button in (self.add_cif_btn, self.add_lattice_btn, self.remove_btn):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.calibrant_row = self.library_row = None
        if calibrants:
            from xrd_tools.integrate.calibration import CALIBRANT_NAMES
            self.calibrant_row = FormRow(label='Calibrant', path=('phase_source', 'calibrant'), value=_ADD_CALIBRANT,
                                         kind='combo', choices=(_ADD_CALIBRANT, *CALIBRANT_NAMES), parent=self)
            self.calibrant_row.set_purpose_tooltip(
                'Add a built-in calibrant: its pyFAI d-spacings as reference lines and current-Run detector rings. '
                'Positions only; needs no pymatgen.')
            self.calibrant_row.valueChanged.connect(lambda _path, value: self._choose_calibrant(value))
            layout.addWidget(self.calibrant_row)
        if calibrants and not show_library:
            self.library_row = FormRow(label='Library', path=('phase_source', 'library'), value='', kind='combo',
                                       choices=('',), choice_labels={'': _ADD_FROM_LIBRARY}, parent=self)
            self.library_row.set_purpose_tooltip('Add a CIF or lattice entry already kept in the phase library.')
            self.library_row.valueChanged.connect(lambda _path, value: self._choose_library_entry(value))
            layout.addWidget(self.library_row)
        self.global_switch = self.global_row = None
        if cutoffs:
            self.global_switch = FormRow(label='Same minimum for all phases', path=('phase_source', 'global_floor'),
                                         value=True, kind='bool', parent=self)
            self.global_switch.set_purpose_tooltip(
                'On: one minimum intensity applies to every CIF phase and the row values are off. '
                'Off: each CIF phase uses its own value, starting from this one.')
            self.global_row = FormRow(label='Minimum intensity', path=('phase_source', 'min_intensity'), value=5.,
                                      kind='float', minimum=0., maximum=100., decimals=1, suffix=' %', parent=self)
            self.global_switch.valueChanged.connect(lambda _path, value: self._switch_edited(bool(value)))
            self.global_row.valueChanged.connect(lambda _path, value: self._global_edited(float(value)))
            self.browser.resultValueChanged.connect(self._row_edited)
            layout.addWidget(self.global_switch)
            layout.addWidget(self.global_row)
        self.lattice_form = SectionCard('Lattice parameters', self, collapsible=False)
        self.lattice_form.setToolTip('A unit cell and space group determine allowed reflection positions, not intensities.')
        self.lattice_rows = {}
        fields = (('name', 'Name', 'Lattice', 'A name for this lattice entry.'),
                  ('a', 'a (Å)', '', 'Unit-cell length a, in ångströms.'),
                  ('b', 'b (Å)', '', 'Unit-cell length b; blank uses a.'),
                  ('c', 'c (Å)', '', 'Unit-cell length c; blank uses a.'),
                  ('alpha', 'α (°)', '90', 'Unit-cell angle between b and c, in degrees.'),
                  ('beta', 'β (°)', '90', 'Unit-cell angle between a and c, in degrees.'),
                  ('gamma', 'γ (°)', '90', 'Unit-cell angle between a and b, in degrees.'),
                  ('space_group', 'Space group', '1', 'International space-group number or Hermann–Mauguin symbol.'))
        for key, label, value, tip in fields:
            row = FormRow(label=label, path=('phase_source', key), value=value, parent=self.lattice_form)
            row.set_purpose_tooltip(tip)
            if key not in ('name', 'space_group'):
                row.editor.setValidator(QtGui.QDoubleValidator(row))
            self.lattice_rows[key] = row
            self.lattice_form.add_row(row)
        self.add_lattice_commit = QtWidgets.QPushButton('Add phase', self.lattice_form)
        self.add_lattice_commit.setToolTip('Validate these lattice parameters, retain the entry and select it.')
        self.lattice_form.add_row(self.add_lattice_commit)
        self.lattice_form.hide()
        layout.addWidget(self.lattice_form)
        self.status = QtWidgets.QLabel('', self)
        self.status.setWordWrap(True)
        self.status.setToolTip('Phase-source validation and missing-library-entry notices.')
        layout.addWidget(self.status)
        #: False: a host shows the messages (``messageChanged``) on its own status line, so each shows once
        self.status.setVisible(bool(status_visible))
        #: a host's notice about its use of the list (``set_note``), hidden while empty
        self.note = QtWidgets.QLabel('', self)
        self.note.setObjectName('phaseSourceNote')
        self.note.setWordWrap(True)
        self.note.hide()
        layout.addWidget(self.note)
        self._note = ('', '')
        controls = (self.browser, self.browser.scans, self.browser.scans_label,
                    self.browser.show_all, self.browser.metadata, self.browser.auto_last,
                    self.browser.date_sort, self.browser.refresh, self.add_cif_btn,
                    self.add_lattice_btn, self.remove_btn, self.lattice_form,
                    self.add_lattice_commit, self.status, *self.lattice_rows.values(),
                    *(row for row in (self.calibrant_row, self.library_row, self.global_switch) if row is not None))
        self._purpose_tooltips = {control: control.toolTip() for control in controls}
        self.add_cif_btn.clicked.connect(self._browse_cifs)
        self.add_lattice_btn.clicked.connect(lambda: self.lattice_form.setVisible(not self.lattice_form.isVisible()))
        self.add_lattice_commit.clicked.connect(self._commit_lattice)
        self.remove_btn.clicked.connect(self.remove_current)
        if checkable:
            self.browser.checksChanged.connect(self._checks_changed)
        self.refresh_library()

    @property
    def selection(self):
        return self._selection

    @property
    def reason(self):
        return self._reason

    def _message(self, message):
        if str(message) == self._reason:
            return
        self._reason = str(message)
        self.status.setText(self._reason)
        self._sync_tooltips()
        self.messageChanged.emit(self._reason)

    def set_note(self, text, purpose):
        """Show a host's notice under the list (hidden while empty); its tooltip is the purpose plus the notice."""
        text, purpose = str(text), str(purpose)
        if (text, purpose) == self._note:
            return
        self._note = (text, purpose)
        self.note.setText(text)
        self.note.setToolTip(purpose + (' ' + text if text else ''))
        self.note.setVisible(bool(text))

    def _sync_tooltips(self):
        suffix = ' ' + self._availability_reason if self._availability_reason else ''
        self.setToolTip(_SOURCE_TIP + suffix + (' ' + self._reason if self._reason else ''))
        needs_backend = (self.add_cif_btn, self.add_lattice_btn, self.lattice_form, self.add_lattice_commit,
                         *self.lattice_rows.values())
        for control, purpose in self._purpose_tooltips.items():
            extra = suffix or (' ' + self._source_reason if self._source_reason and control in needs_backend else '')
            if isinstance(control, FormRow):
                control.set_purpose_tooltip(purpose + extra)
            else:
                control.setToolTip(purpose + extra)
        self._sync_cutoff_rows()

    def selected_entries(self):
        """The visible selected entries as last resolved by refresh_library."""
        return tuple(self._resolved[identifier] for identifier, visible in self._selection
                     if visible and identifier in self._resolved)

    def listed_entries(self):
        """Every listed entry with its visibility, in list order, as last resolved."""
        return tuple((self._resolved[identifier], visible) for identifier, visible in self._selection
                     if identifier in self._resolved)

    @property
    def cutoffs(self) -> PhaseCutoffs:
        return self._cutoffs

    def set_cutoffs(self, cutoffs) -> None:
        """Project the minimum-intensity choice silently (no ``cutoffsChanged``)."""
        if type(cutoffs) is not PhaseCutoffs:
            raise TypeError('cutoffs must be an exact PhaseCutoffs')
        self._cutoffs = cutoffs
        self._sync_cutoff_rows()

    def cutoff_editor(self, identifier):
        """The listed phase's row editor (None: not listed, or no cutoffs)."""
        if identifier not in self._shown_ids:
            return None
        key = str(self._shown_ids.index(identifier)) if self._key_by_index else identifier
        return self.browser.result_value_editor(key)

    def refresh_cutoffs(self):
        """Re-project the cutoff rows after a host reconciled this source's list itself."""
        self._sync_cutoff_rows()

    def _cif_ids(self):
        return [entry.id for entry, _visible in self.listed_entries() if entry.source_kind == 'cif']

    def _sync_cutoff_rows(self):
        if not self._cutoffs_enabled:
            return
        cutoffs, available = self._cutoffs, self.isEnabled() and not self._availability_reason
        own, rows = dict(cutoffs.per_phase), {}
        for index, identifier in enumerate(self._shown_ids):
            entry = self._resolved.get(identifier)
            if entry is None:
                continue
            key = str(index) if self._key_by_index else identifier
            if entry.source_kind != 'cif':
                rows[key] = (cutoffs.global_value, False, _NO_INTENSITY_TIP.get(entry.source_kind, _ROW_TIP))
            else:
                value = cutoffs.global_value if cutoffs.global_on else own.get(identifier, cutoffs.global_value)
                tip = _ROW_TIP + (' ' + _SAME_TIP if cutoffs.global_on else '')
                rows[key] = (value, available and not cutoffs.global_on, tip)
        self.browser.reconcile_result_values(rows)
        blocked = self.global_switch.editor.blockSignals(True)
        try:
            self.global_switch.editor.setChecked(cutoffs.global_on)
        finally:
            self.global_switch.editor.blockSignals(blocked)
        self.global_row.set_number(cutoffs.global_value)
        cif = bool(self._cif_ids())
        tip = 'Minimum intensity as a percent of the strongest reflection in the reachable range, from 0 to 100.'
        if not cutoffs.global_on:
            tip += ' Each CIF phase uses its own value now; new phases start from this one.'
        elif not cif:
            tip += (' Lattice and calibrant phases bypass the minimum.' if self._selection
                    else ' Select a CIF phase to edit the minimum.')
        self.global_row.editor.setEnabled(available and cutoffs.global_on and cif)
        self.global_row.set_purpose_tooltip(tip + (' ' + self._availability_reason if self._availability_reason else ''))

    def _adjust_cutoffs(self):
        """Keep stored values for listed phases only; with the switch off a new CIF phase starts from the global."""
        cutoffs = self._cutoffs
        listed = [identifier for identifier, _visible in self._selection]
        own = dict(cutoffs.per_phase)
        values = [(identifier, own[identifier]) for identifier in listed if identifier in own]
        if not cutoffs.global_on:
            values += [(identifier, cutoffs.global_value) for identifier in self._cif_ids() if identifier not in own]
            order = {identifier: index for index, identifier in enumerate(listed)}
            values.sort(key=lambda pair: order[pair[0]])
        self._cutoffs = replace(cutoffs, per_phase=tuple(values))

    def _switch_edited(self, on):
        cutoffs = self._cutoffs
        if on == cutoffs.global_on:
            return
        self._cutoffs = replace(cutoffs, global_on=on)
        if not on:
            self._adjust_cutoffs()  # materialize the global value for each CIF phase without its own
        self._sync_cutoff_rows()
        self.cutoffsChanged.emit(self._cutoffs)

    def _global_edited(self, value):
        if value == self._cutoffs.global_value:
            return
        self._cutoffs = replace(self._cutoffs, global_value=value)
        self._sync_cutoff_rows()
        self.cutoffsChanged.emit(self._cutoffs)

    def _row_edited(self, key, value):
        identifier = self._shown_ids[int(key)] if self._key_by_index else key
        own = dict(self._cutoffs.per_phase)
        if own.get(identifier) == value:
            return
        own[identifier] = float(value)
        order = [identifier for identifier, _visible in self._selection]
        self._cutoffs = replace(self._cutoffs, per_phase=tuple(
            (identifier, own[identifier]) for identifier in order if identifier in own))
        self.cutoffsChanged.emit(self._cutoffs)

    def set_selection(self, selection):
        selection = tuple(selection)
        if len(selection) > 8 or any(type(item) is not tuple or len(item) != 2
                or type(item[0]) is not str or not item[0] or type(item[1]) is not bool for item in selection):
            raise ValueError('Choose at most eight identified phases with boolean visibility.')
        if len({identifier for identifier, _visible in selection}) != len(selection):
            raise ValueError('A phase can be selected only once.')
        if selection == self._selection:
            return
        self._selection = selection
        self.refresh_library()
        if self._cutoffs_enabled:
            self._adjust_cutoffs()
            self._sync_cutoff_rows()
        self.changed.emit(self._selection)

    def refresh_library(self):
        """Read the library once for the list and the selection (user actions, never reconciles)."""
        from xrd_tools.analysis.phase_references import PhaseLibraryLookup, pymatgen_available
        selected = dict(self._selection)
        if self._show_library:
            entries = tuple(self.library.entries())
            known = {entry.id: entry for entry in entries}
            lookups = {identifier: PhaseLibraryLookup(known[identifier]) if identifier in known
                       else self.library.resolve(identifier) for identifier in selected}
        else:
            lookups = {identifier: self.library.resolve(identifier) for identifier in selected}
            entries = tuple(lookup.entry for lookup in lookups.values() if lookup.entry is not None)
        self._resolved = {identifier: lookup.entry for identifier, lookup in lookups.items()
                          if lookup.entry is not None}
        self._shown_ids = tuple(entry.id for entry in entries)
        rows = tuple(BrowserResult(str(index) if self._key_by_index else entry.id, entry.name,
                                   self._details.get(entry.id, f'{entry.source_kind} phase: {entry.name}'))
                     for index, entry in enumerate(entries))
        if self._checkable:
            checked = tuple(row.identifier for row, entry in zip(rows, entries) if selected.get(entry.id, False))
            self.browser.reconcile_checked_results(rows, checked)
        else:
            self.browser.reconcile_results(rows, self.browser.current_result())
        missing = [lookup.reason or f'Missing phase library entry: {identifier}'
                   for identifier, lookup in lookups.items() if lookup.entry is None]
        self._message('; '.join(missing))
        if self.library_row is not None:
            editor = self.library_row.editor
            blocked = editor.blockSignals(True)
            try:
                editor.clear()
                editor.addItem(_ADD_FROM_LIBRARY, '')
                for entry in self.library.entries():
                    if entry.id not in selected:
                        editor.addItem(entry.name, entry.id)
            finally:
                editor.blockSignals(blocked)
        self._sync_backend()
        self._sync_tooltips()

    def set_available(self, available, reason=''):
        self.setEnabled(bool(available))
        self._availability_reason = str(reason) if not available else ''
        self._sync_backend()
        self._sync_tooltips()

    def _sync_backend(self):
        """Calibrants need no pymatgen (P3a's rings never did); CIF and lattice sources say why they are off."""
        from xrd_tools.analysis.phase_references import pymatgen_available
        self._source_reason = '' if pymatgen_available() else _PYMATGEN_REASON
        for control in (self.add_cif_btn, self.add_lattice_btn, self.lattice_form):
            control.setEnabled(not self._source_reason)

    def set_details(self, details):
        self._details = dict(details)
        self.refresh_library()

    def _checks_changed(self, checked):
        checked = set(checked)
        visible_ids = {identifier for index, identifier in enumerate(self._shown_ids)
                       if (str(index) if self._key_by_index else identifier) in checked}
        known_ids = {identifier for identifier, _visible in self._selection}
        selected = tuple((identifier, identifier in visible_ids)
                         for identifier, _visible in self._selection)
        selected += tuple((identifier, True) for identifier in self._shown_ids
                          if identifier in visible_ids and identifier not in known_ids)
        if len(selected) > 8:
            self._message('Choose at most eight phases.'); self.refresh_library(); return
        self.set_selection(selected)

    def _add_entry(self, entry):
        before = self._selection
        if entry.id not in dict(before):
            if len(before) >= 8:
                raise ValueError('Choose at most eight phases.')
            self.set_selection((*before, (entry.id, True)))
        else:
            self.set_selection(tuple((identifier, visible or identifier == entry.id)
                                     for identifier, visible in before))
        if self._selection == before:
            self.refresh_library()  # the library entry itself may have been rewritten
        return entry

    def add_cif_paths(self, paths):
        added = []
        for path in paths:
            try:
                if len(self._selection) >= 8:
                    raise ValueError('Choose at most eight phases.')
                entry = self.library.add_cif(path)
                added.append(self._add_entry(entry))
            except (OSError, ValueError, ImportError) as error:
                self._message(str(error)); break
        return tuple(added)

    def add_calibrant(self, name):
        """Add a built-in calibrant entry (no library file) and show it."""
        from xrd_tools.analysis.phase_references import calibrant_entry
        try:
            if len(self._selection) >= 8 and calibrant_entry(name).id not in dict(self._selection):
                raise ValueError('Choose at most eight phases.')
            return self._add_entry(calibrant_entry(name))
        except ValueError as error:
            self._message(str(error))
            return None

    def add_library_entry(self, identifier):
        lookup = self.library.resolve(identifier)
        if lookup.entry is None:
            self._message(lookup.reason or f'Missing phase library entry: {identifier}')
            return None
        try:
            if len(self._selection) >= 8 and identifier not in dict(self._selection):
                raise ValueError('Choose at most eight phases.')
            return self._add_entry(lookup.entry)
        except ValueError as error:
            self._message(str(error))
            return None

    def _choose_calibrant(self, value):
        if value and value != _ADD_CALIBRANT:
            self.add_calibrant(value)
        self._reset_choice(self.calibrant_row, 0)

    def _choose_library_entry(self, value):
        if value:
            self.add_library_entry(value)
        self._reset_choice(self.library_row, 0)

    @staticmethod
    def _reset_choice(row, index):
        blocked = row.editor.blockSignals(True)
        try:
            row.editor.setCurrentIndex(index)
        finally:
            row.editor.blockSignals(blocked)

    def add_lattice(self, name, **lattice):
        try:
            if len(self._selection) >= 8:
                raise ValueError('Choose at most eight phases.')
            return self._add_entry(self.library.add_lattice(name, **lattice))
        except (ValueError, ImportError) as error:
            self._message(str(error))
            return None

    def remove_current(self):
        row = self.browser.current_result_row()
        if 0 <= row < len(self._shown_ids):
            identifier = self._shown_ids[row]
            self.set_selection(tuple(pair for pair in self._selection if pair[0] != identifier))

    def _browse_cifs(self):
        paths = browse(self, 'Add CIF phase(s)', many=True, file_filter=CIF_FILTER)
        if paths:
            self.add_cif_paths(paths)

    def _commit_lattice(self):
        text = {key: row.editor.text().strip() for key, row in self.lattice_rows.items()}
        try:
            values = {key: float(text[key]) for key in ('a', 'alpha', 'beta', 'gamma')}
            values.update({key: float(text[key]) for key in ('b', 'c') if text[key]})
            group = text['space_group']
            values['space_group'] = int(group) if group.isdigit() else group
            entry = self.add_lattice(text['name'], **values)
        except ValueError as error:
            self._message(f'Lattice parameters: {error}'); return
        if entry is not None:
            self.lattice_form.hide()
