"""Shared CIF/library and manual-lattice source for fitting and reference overlays.

This owner mounts Int's keyed BrowserView and Controls rows. The headless phase
library owns all valid sources; selection is only an immutable tuple of IDs and
visibility flags. No reflection listing or detector prediction runs here.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xdart.gui.tabs.scattering.browser_view import BrowserResult, BrowserView, keyed_list_profile
from xdart.gui.widgets.controls_panel import FormRow, SectionCard
from xdart.gui.widgets.file_dialogs import browse

CIF_FILTER = 'CIF files (*.cif);;All files (*)'
_SOURCE_TIP = ('Choose structural CIF phases or enter lattice parameters. CIF entries retain their bytes in the '
               'phase library; a lattice entry predicts positions only, without structural intensities.')


class PhaseSourceWidget(QtWidgets.QWidget):
    """One shared source editor; consumers own only what its selection means."""

    changed = QtCore.Signal(object)
    messageChanged = QtCore.Signal(str)

    def __init__(self, parent=None, *, library=None, checkable=True, show_library=True, key_by_index=False):
        super().__init__(parent)
        from xrd_tools.analysis.phase_references import PhaseLibrary
        self.library = PhaseLibrary() if library is None else library
        self._selection = ()
        self._show_library = bool(show_library)
        self._checkable = bool(checkable)
        self._key_by_index = bool(key_by_index)
        self._shown_ids = ()
        self._details = {}
        self._reason = ''
        self.setObjectName('phaseSourceWidget')
        self.setToolTip(_SOURCE_TIP)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.browser = BrowserView(self, profile=keyed_list_profile('Phases', checkable=checkable))
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
        self._reason = str(message)
        self.status.setText(self._reason)
        self.setToolTip(_SOURCE_TIP + (' ' + self._reason if self._reason else ''))
        self.messageChanged.emit(self._reason)

    def selected_entries(self):
        entries = []
        missing = []
        for identifier, visible in self._selection:
            if not visible:
                continue
            lookup = self.library.resolve(identifier)
            if lookup.entry is None:
                missing.append(lookup.reason or f'Missing phase library entry: {identifier}')
            else:
                entries.append(lookup.entry)
        self._message('; '.join(missing))
        return tuple(entries)

    def set_selection(self, selection):
        selection = tuple(selection)
        if len(selection) > 8 or any(type(item) is not tuple or len(item) != 2
                or type(item[0]) is not str or not item[0] or type(item[1]) is not bool for item in selection):
            raise ValueError('Choose at most eight identified phases with boolean visibility.')
        if len({identifier for identifier, _visible in selection}) != len(selection):
            raise ValueError('A phase can be selected only once.')
        changed = selection != self._selection
        self._selection = selection
        self.refresh_library()
        if changed:
            self.changed.emit(self._selection)

    def refresh_library(self):
        selected = dict(self._selection)
        if self._show_library:
            entries = tuple(self.library.entries())
        else:
            entries = tuple(lookup.entry for identifier, _visible in self._selection
                            if (lookup := self.library.resolve(identifier)).entry is not None)
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
                   for identifier, _visible in self._selection
                   if (lookup := self.library.resolve(identifier)).entry is None]
        self._message('; '.join(missing))
        from xrd_tools.analysis import phase
        self.set_available(bool(phase.HAS_PYMATGEN),
                           '' if phase.HAS_PYMATGEN else 'Phase sources need pymatgen; install xdart[fitting].')

    def set_available(self, available, reason=''):
        self.setEnabled(bool(available))
        tip = _SOURCE_TIP + (' ' + str(reason) if reason else '')
        self.setToolTip(tip + (' ' + self._reason if self._reason else ''))
        for button, purpose in ((self.add_cif_btn, 'Add a structural CIF to the library.'),
                                (self.add_lattice_btn, 'Enter a lattice and space group.'),
                                (self.remove_btn, 'Remove this tool’s selected phase.')):
            button.setToolTip(purpose + (' ' + str(reason) if reason else ''))

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
        if entry.id not in dict(self._selection):
            if len(self._selection) >= 8:
                raise ValueError('Choose at most eight phases.')
            self.set_selection((*self._selection, (entry.id, True)))
        else:
            self.set_selection(tuple((identifier, visible or identifier == entry.id)
                                     for identifier, visible in self._selection))
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
