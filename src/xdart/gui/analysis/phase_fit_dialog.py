# -*- coding: utf-8 -*-
"""The Phase Analysis window (PHASE-ANALYSIS-128): one phase list for the plots' reference overlays and the fit, and
every phase-fitting option of the HZO notebook, each with its "?" help.

The maintainer, 2026-10-28: "there are two tools right now - Phase Fitting and Reference Phases. they can be folded
into one - Phase Analysis". Its host creates it (Int's analysis mount, Stitch's ``SnapshotFitHost``) and fits through
``vnext_submit``, on the host's own displayed input and worker. The phase list is Int's shared phase source in its
reference mode (an include check and a minimum intensity per phase), bound by ``ReferencePhaseBinding`` to the
reference-phase plot options of the ``ScientificView`` the host attaches it to: a checked phase is drawn and fitted,
and a built-in calibrant is drawn only.

It builds no display or editor widget of its own (hard rule 14): its options are the shared Controls rows
(``FormRow``, ``RangeRow``) in ``SectionCard``s, the rarely changed ones the shared ``AdvancedDialog``'s groups; its
plots ``FitView``, its trend ``TrendView``, its table ``ResultsTable``; Save and Load settings the shared
``PageSettings``; Browse the shared ``file_dialogs``.

Needs the ``xrd-tools[fitting]`` extra: ``lmfit`` (the fit) and ``pymatgen`` (CIF parsing), both imported lazily.
"""
import math

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xrd_tools.help_topics import diagnostic_text
from xrd_tools.session.readiness import (
    ControlAction, ControlActionSpec, ControlFieldKind, ControlFormField, SectionId, integer_field_refusal,
)

from xdart.gui.pages.values import CloseReceipt, PageCleanup
from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
from xdart.gui.tabs.scattering.fit_view import FitPanel, FitProjection, FitSeries, FitView, fit_result_series
from xdart.gui.tabs.scattering.results_table import PHASE_RESULT_HEADERS, ResultsTable, phase_result_rows
from xdart.gui.tabs.scattering.trend_view import TrendView
from xdart.gui.tools.page_settings import PageSettings, SettingsKind, check_fields, plain, saved_fields
from xdart.gui.widgets.controls_panel import ActionButton, FormRow, RangeRow, SectionCard

#: each section's help topic (``xrd_tools.help_topics``)
TOPICS = {"phases": "phase-analysis.phases", "shape": "phase-analysis.peak-shape",
          "background": "phase-analysis.background", "constraints": "phase-analysis.constraints",
          "batch": "phase-analysis.batch"}
#: the window's saved settings (Save settings…, Load settings…); no last-session restore
PHASE_ANALYSIS_SETTINGS = SettingsKind("xdart-phase-analysis-settings", 1, "phase_analysis_settings", "Phase Analysis",
                                       sections=("phases",))

# (display label, plan value)
_PROFILES = (("Pseudo-Voigt", "pseudovoigt"), ("Gaussian", "gaussian"), ("Lorentzian", "lorentzian"),
             ("Voigt", "voigt"), ("Lorentzian²", "lorentzian_squared"), ("Pearson VII", "pearson7"),
             ("Pearson IV", "pearson4"), ("Split Lorentzian", "splitlorentzian"), ("Moffat", "moffat"),
             ("Student's t", "studentst"), ("Skewed Gaussian", "skewedgaussian"))
_WIDTH_MODELS = (("Caglioti U,V,W", "caglioti"), ("Scherrer size and strain", "scherrer"), ("Fixed width", "fixed"))
_TEXTURES = (("None", "none"), ("March-Dollase", "march_dollase"), ("Free (per-peak)", "free"))
_BACKGROUNDS = (("None", "none"), ("SNIP", "snip"), ("Chebyshev", "chebyshev"))
_FIT_BACKGROUNDS = (("None", "none"), ("Linear", "linear"), ("Polynomial", "polynomial"), ("Chebyshev", "chebyshev"),
                    ("Spline", "spline"), ("Substrate", "template"), ("Substrate + linear", "template+linear"),
                    ("Substrate + polynomial", "template+polynomial"),
                    ("Substrate + Chebyshev", "template+chebyshev"))
_AMORPHOUS = (("None", "none"), ("Gaussian", "gaussian"), ("Pseudo-Voigt", "pseudovoigt"),
              ("Lorentzian", "lorentzian"), ("Voigt", "voigt"), ("Lorentzian²", "lorentzian_squared"))
#: the fitted backgrounds that take a degree (a spline's knots)
_DEGREE_BACKGROUNDS = frozenset({"polynomial", "chebyshev", "spline", "template+polynomial", "template+chebyshev"})

_WINDOW_TIP = ('Phase Analysis: one phase list drawn on the plots and fitted, with every option of the phase-fitting '
               'notebook. Each "?" opens its section\'s help.')
_TEXTURE_TIP = ('How relative intensities within each CIF phase may depart from the structure: none, March-Dollase '
                'along the axis in Advanced, or free per-peak heights within the range in Advanced.')
_TEXTURE_LATTICE_TIP = ' Lattice phases always fit free per-peak heights: they have no structural intensities.'
_FIT_BACKGROUND_TIP = ('A background refined with the phases: linear, polynomial, Chebyshev or spline of the degree '
                       'below, or the captured substrate scaled by one amplitude, alone or plus a polynomial.')
_FIT_TIP = 'Fit the checked CIF and lattice phases to the displayed pattern; calibrants are reference lines only.'
_BATCH_TIP = ('Fit every selected result with these settings, then plot fractions and lattice against the result '
              'order or a scan column.')
_CALIBRANT_ONLY = 'Calibrants are reference lines only and are not fitted: check a CIF or lattice phase to fit.'
_SUBSTRATE_NEEDED = ('The Substrate background needs a captured substrate: show the bare substrate\'s pattern and '
                     'click Use displayed as substrate.')
_TEMPLATE_EXPORT_REASON = ('PHASE_TEMPLATE_NOTEBOOK_UNAVAILABLE: a fit that scales a captured substrate cannot be '
                           'exported as a notebook; clear the substrate or choose another background to export.')
_RECAPTURE_NOTICE = 'The substrate is not saved: capture it again before fitting with it.'


def _path(name):
    return ("phase_fit", name)


def _combo(label, name, choices):
    values = tuple(value for _text, value in choices)
    return FormRow(label=label, path=_path(name), value=values[0], kind="combo", choices=values,
                   choice_labels={value: text for text, value in choices})


def _range(label, name, low, high, placeholders):
    return RangeRow(label=label, low={"path": _path(f"{name}_low"), "value": low, "placeholder": placeholders[0]},
                    high={"path": _path(f"{name}_high"), "value": high, "placeholder": placeholders[1]})


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


class PhaseFitDialog(QtWidgets.QDialog):
    """The Phase Analysis window; ``vnext_submit(action, values)`` is its host's ("reload", "fit", "close",
    "capture_template"), and ``allow_batch`` whether the host can fit a batch (Stitch can; Int's is held)."""

    def __init__(self, parent=None, *, vnext_submit=None, allow_batch=False):
        super().__init__(parent)
        if not callable(vnext_submit):
            raise TypeError("Phase Analysis fits through its host: Int's analysis mount or a SnapshotFitHost")
        self._vnext_submit = vnext_submit
        self._allow_batch = bool(allow_batch)
        self._batch_reason = '' if allow_batch else diagnostic_text("P3_7_BATCH_DISPLAY_PROJECTION_UNAVAILABLE")
        self._fit_reason = ''
        self._running = False
        #: a host's probe of a fit it runs (Load settings waits for it); None: the host runs none in the background
        self.busy_probe = None
        self._export_trace = None
        self._export_wavelength = None
        self._export_snapshot_provider = None
        self._x = self._y = None
        self._x_label, self._x_unit = "q", ""
        #: the substrate pattern the host captured (a q ``DisplayedTraceInput``), kept until cleared and never saved
        self._template = None
        self._receipts = {}
        self._load_notice = ''
        self._closing = False
        #: each option row's purpose tooltip, and each whole-number row's bounds (minimum, maximum, blank allowed)
        self._tips, self._int_bounds = {}, {}
        self.setObjectName("phaseFitDialog")
        self.setWindowTitle("Phase Analysis")
        self.setToolTip(_WINDOW_TIP)
        self._build_ui()
        from .fit_export import FitExportActions
        self.export_actions = FitExportActions(self, tool='phase', snapshot=self.export_snapshot,
                                               settings=self.export_fit_values)
        self._build_settings()
        self._sync_options()
        self._sync_actions()
        # 1400 x 1200 (the maintainer, after the mockup: "the window can be wider. 1400x1200 is also fine"), within
        # the screen; the controls and the displays scroll in their own panes when the screen is smaller.
        screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1400, 1200)
        width, height = min(1400, int(.95 * available.width())), min(1200, int(.95 * available.height()))
        self.resize(width, height)
        controls = min(440, width // 2)
        self.splitter.setSizes([controls, max(1, width - controls)])

    # ---- UI construction ------------------------------------------------
    def _card(self, title, section, tip):
        card = SectionCard(title, self, collapsible=False)
        card.set_purpose_tooltip(tip)
        card.set_help_topic(TOPICS[section])
        self._cards.addWidget(card)
        return card

    def _option(self, card, section, row, tip):
        self._tips[row] = tip
        row.set_purpose_tooltip(tip)
        row.set_help_topic(TOPICS[section], button=True)
        if card is not None:
            card.add_row(row)
        return row

    def _int_row(self, label, name, value, minimum, maximum, *, blank=False):
        self._int_bounds[name] = (minimum, maximum, blank)
        return FormRow(label=label, path=_path(name), value=value, kind="int", minimum=minimum, maximum=maximum,
                       allow_blank=blank)

    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(11, 11, 11, 11)
        lay.setSpacing(8)
        # The options' column on the left, the displays on the right; each scrolls when the window is small.
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)
        controls = QtWidgets.QScrollArea(self.splitter)
        controls.setObjectName("phaseAnalysisControls")
        controls.setWidgetResizable(True)
        column = QtWidgets.QWidget(controls)
        self._cards = QtWidgets.QVBoxLayout(column)
        self._cards.setContentsMargins(0, 0, 6, 0)
        self._build_phases()
        self._build_shape()
        self._build_background()
        self._build_constraints()
        self._build_batch()
        self._cards.addStretch(1)
        controls.setWidget(column)
        displays = QtWidgets.QScrollArea(self.splitter)
        displays.setObjectName("phaseAnalysisDisplays")
        displays.setWidgetResizable(True)
        self.displays = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, displays)
        self.displays.setChildrenCollapsible(False)
        # Data and fit above (the fit range dragged on it), the residual below: the shared fit display.
        self.fit_view = FitView(self.displays, name="phaseFitView", summary=False, stretch=(3, 1),
                                upper_min_height=220, lower_max_height=110, link_x=True)
        self.plot, self.resid_plot = self.fit_view.plot, self.fit_view.residuals
        self.fit_view.reconcile(FitProjection(lower=FitPanel(zero_line=True)))
        # Parameters (fractions, lattice) against the result order or a scan column, on the shared trend.
        self.trend = TrendView(self.displays, name="phaseFitTrend")
        # Each phase's fraction and lattice, on the shared table.
        self.table = ResultsTable(PHASE_RESULT_HEADERS, self.displays, name="phaseFitTable", max_height=150)
        self.table.setToolTip("Each fitted phase's fraction and lattice; a '—' cell's tooltip says why it is empty.")
        for index, (widget, stretch) in enumerate(((self.fit_view, 5), (self.trend, 2), (self.table, 1))):
            self.displays.addWidget(widget)
            self.displays.setStretchFactor(index, stretch)
        displays.setWidget(self.displays)
        self.splitter.addWidget(controls)
        self.splitter.addWidget(displays)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 5)
        for splitter, tip in ((self.splitter, 'Drag to share the width between the options and the plots.'),
                              (self.displays, 'Drag to share the height between the fit, the trend and the table.')):
            for index in range(1, splitter.count()):
                splitter.handle(index).setToolTip(tip)
        lay.addWidget(self.splitter, 1)

        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(7)
        self.advanced_btn = QtWidgets.QPushButton("Advanced…", self)
        self.advanced_btn.setToolTip('The rarely changed options: the solver\'s budget and weighting, the SNIP width, '
                                     'the March-Dollase axis and free-height range, and the σ limits.')
        self.load_settings_btn = QtWidgets.QPushButton("Load settings…", self)
        self.load_settings_btn.setToolTip('Restore every option and the phase list from a saved Phase Analysis '
                                          'settings file. A substrate is not saved: capture it again.')
        self.save_settings_btn = QtWidgets.QPushButton("Save settings…", self)
        self.save_settings_btn.setToolTip('Save every option and the phase list as a versioned settings file; the '
                                          'substrate and Sequential\'s seeds are not saved.')
        self.refresh_btn = QtWidgets.QPushButton("Reload", self)
        self.refresh_btn.setToolTip("Re-grab the currently selected frame's pattern")
        self.fit_btn = QtWidgets.QPushButton("Fit", self)
        self.fit_btn.setObjectName("peakFitGo")
        self.batch_btn = QtWidgets.QPushButton("Batch", self)
        for button in (self.advanced_btn, self.load_settings_btn, self.save_settings_btn):
            actions.addWidget(button)
        actions.addStretch(1)
        for button in (self.refresh_btn, self.fit_btn, self.batch_btn):
            actions.addWidget(button)
        lay.addLayout(actions)

        self.status = QtWidgets.QLabel("", self)
        self.status.setObjectName("peakFitStatus")
        self.status.setWordWrap(True)
        self.status.setToolTip("Phase Analysis status: the phase list's notices, refusals and the fit's result.")
        lay.addWidget(self.status)
        self._build_advanced()

        self.phase_source.messageChanged.connect(self.status.setText)
        self.phase_source.changed.connect(lambda _selection: self._sync_options())
        self.trend.messageChanged.connect(self.status.setText)
        self.refresh_btn.clicked.connect(self.refresh_pattern)
        self.fit_btn.clicked.connect(self._do_fit)
        self.advanced_btn.clicked.connect(self._show_advanced)
        self.load_settings_btn.clicked.connect(lambda: self._settings_action(self.settings_io.load))
        self.save_settings_btn.clicked.connect(lambda: self._settings_action(self.settings_io.save))
        self.substrate_capture_btn.clicked.connect(lambda: self._vnext_submit("capture_template", None))
        self.substrate_clear_btn.clicked.connect(lambda: self.set_substrate_template(None))
        self.fit_view.rangeChanged.connect(self._region_to_fields)
        self.fit_range_row.valueChanged.connect(lambda _path, _value: self._fields_to_region())
        self.full_btn.clicked.connect(self._on_full)
        for row in (self.texture_row, self.bkg_row, self.fit_background_row, self.amorphous_row):
            row.editor.currentIndexChanged.connect(lambda _index: self._sync_options())

    def _build_phases(self):
        from xdart.gui.tabs.scattering.reference_phases import ReferencePhaseBinding
        from .phase_source import PhaseSourceWidget
        self.phases_card = self._card("Phases", "phases", 'The phase list, shared with the plots: each checked phase '
                                      'is drawn as a reference and fitted; a built-in calibrant is drawn only.')
        # The Reference Phases list: calibrants, library entries, CIFs and lattices, each with an include check and a
        # minimum intensity, keyed by library id; this window's status line shows its messages.
        self.phase_source = PhaseSourceWidget(self.phases_card, checkable=True, show_library=False, calibrants=True,
                                              cutoffs=True, status_visible=False)
        self.phase_source.browser.setMaximumHeight(230)  # room for all eight phases
        self.phase_source.set_help_topic(TOPICS["phases"])
        self.phases_card.add_row(self.phase_source)
        #: the list is the host view's reference-phase plot options; the host attaches this binding to its view
        self.phase_binding = ReferencePhaseBinding(self.phase_source, self)
        self.rings_note = self.phase_binding.rings_note
        # Existing tool integrations retain these aliases to the shared owner.
        self.cif_browser = self.phase_source.browser
        self.cif_browser.setObjectName("phaseCifBrowser")
        self.cif_list = self.cif_browser.scans
        self.add_cif_btn = self.phase_source.add_cif_btn
        self.remove_cif_btn = self.phase_source.remove_btn
        self.add_lattice_btn = self.phase_source.add_lattice_btn
        self.min_intensity = self.phase_source.global_row.editor

    def _build_shape(self):
        card = self.shape_card = self._card("Peak shape", "shape", 'The peak profile fitted to every reflection, how '
                                            'its width varies with q, and texture.')
        self.profile_row = self._option(card, "shape", _combo("Profile", "profile", _PROFILES), (
            'The peak shape fitted to every reflection of every phase; extra shape parameters (fraction, exponent, '
            'skew) are fitted for the profiles that have them.'))
        self.width_model_row = self._option(card, "shape", _combo("Width model", "width_model", _WIDTH_MODELS), (
            'How each phase\'s peak width varies with q: Caglioti U, V, W (σ² = U·q² + V·q + W), Scherrer '
            'crystallite size and microstrain, or one fixed width per phase.'))
        self.texture_row = self._option(card, "shape", _combo("Texture", "texture", _TEXTURES), _TEXTURE_TIP)
        self.profile_combo, self.texture_combo = self.profile_row.editor, self.texture_row.editor

    def _build_background(self):
        card = self.background_card = self._card("Background", "background", (
            'What is not crystalline: a baseline removed before the fit, a background refined with the phases, a '
            'captured substrate and an amorphous peak.'))
        self.bkg_row = self._option(card, "background", _combo("Pre-fit background", "background", _BACKGROUNDS), (
            'A baseline subtracted before the fit, with no free parameters: SNIP (its width in Advanced) or '
            'Chebyshev.'))
        self.bkg_combo = self.bkg_row.editor
        self.fit_background_row = self._option(card, "background", _combo(
            "Fitted background", "fit_background", _FIT_BACKGROUNDS), _FIT_BACKGROUND_TIP)
        self.fit_background_degree_row = self._option(card, "background", self._int_row(
            "Degree", "fit_background_degree", "3", 2, 15), (
            'The fitted background\'s polynomial or Chebyshev degree, or a spline\'s knots (at least four), from 2 '
            'to 15.'))
        # The substrate, captured through the host's own fit input: the card's status names it.
        self.substrate_capture_btn = ActionButton(ControlActionSpec(
            ControlAction.PREVIEW, "Use displayed as substrate", SectionId.PROCESSING), self)
        self.substrate_capture_btn.set_purpose_tooltip(
            'Capture the displayed trace, in Å⁻¹, as the substrate template: show the bare substrate\'s pattern '
            'first. It is kept until cleared and is not saved with the settings.')
        self.substrate_clear_btn = ActionButton(ControlActionSpec(
            ControlAction.PREVIEW, "Clear substrate", SectionId.PROCESSING), self)
        self.substrate_clear_btn.set_purpose_tooltip('Forget the captured substrate template.')
        substrate = QtWidgets.QWidget(card)
        buttons = QtWidgets.QHBoxLayout(substrate)
        buttons.setContentsMargins(0, 0, 0, 0)
        for button in (self.substrate_capture_btn, self.substrate_clear_btn):
            button.setAutoDefault(False)
            buttons.addWidget(button)
        buttons.addStretch(1)
        card.add_row(substrate)
        self.amorphous_row = self._option(card, "background", _combo("Amorphous peak", "amorphous_peak", _AMORPHOUS),
                                          'A broad peak for an amorphous component, fitted with the phases from the '
                                          'starting centre and σ below.')
        self.amorphous_center_row = self._option(card, "background", FormRow(
            label="Amorphous centre", path=_path("amorphous_center"), value=1.5, kind="float", minimum=0.,
            maximum=100., decimals=3, suffix=" Å⁻¹"),
            'The amorphous peak\'s starting centre, in Å⁻¹; it must lie within the displayed trace.')
        self.amorphous_sigma_row = self._option(card, "background", FormRow(
            label="Amorphous σ", path=_path("amorphous_sigma"), value=.3, kind="float", minimum=.001, maximum=10.,
            decimals=3, suffix=" Å⁻¹"), 'The amorphous peak\'s starting width σ, in Å⁻¹.')

    def _build_constraints(self):
        card = self.constraints_card = self._card("Constraints", "constraints", (
            'Bounds on each lattice and on one global shift, the cross-phase lock, and the interval the fit solves '
            'over.'))
        self.lattice_row = self._option(card, "constraints", FormRow(
            label="Lattice ±%", path=_path("lattice_pct"), value=5., kind="float", minimum=0., maximum=25.,
            decimals=2, suffix=" %"),
            'Each cell length may move within this percent of its starting value; cell angles are not refined.')
        self.lattice_pct = self.lattice_row.editor
        self.q_shift_row = self._option(card, "constraints", FormRow(
            label="q-shift bound", path=_path("q_shift"), value=.05, kind="float", minimum=.001, maximum=1.,
            decimals=3, suffix=" Å⁻¹"), 'One shift of every reflection, bounded to ± this value in Å⁻¹.')
        self.cross_phase_row = self._option(card, "constraints", FormRow(
            label="Cross-phase lock", path=_path("lock_cross_phase"), value=False, kind="bool"),
            'Keep the same axis letter ordered across phases (lock the a, b, c order), as the notebook\'s lock.')
        self.fit_range_row = _range("Fit range", "fit_range", "", "", ("full", "full"))
        self.full_btn = QtWidgets.QPushButton("Full", self)
        self.full_btn.setToolTip("Fit the whole trace: clear the fit range.")
        self.full_btn.setAutoDefault(False)
        self.fit_range_row.add_trailing_widget(self.full_btn)
        self._option(card, "constraints", self.fit_range_row, (
            'The interval, in the displayed axis, whose points enter the solve; the whole trace is still drawn. '
            'Drag it on the plot or type it; blank fits the whole trace.'))
        for edit in self.fit_range_row.editors:
            edit.setValidator(QtGui.QDoubleValidator(self))

    def _build_batch(self):
        card = self.batch_card = self._card("Batch", "batch", (
            'Fitting many results with these settings, each seeded from the previous fit when Sequential is on.'))
        self.sequential_row = self._option(card, "batch", FormRow(
            label="Sequential", path=_path("sequential"), value=False, kind="bool", enabled=self._allow_batch,
            reason=self._batch_reason),
            'In a Batch, seed each fit from the previous fit\'s values; a fit whose reflections differ starts fresh.')

    def add_batch_row(self, row):
        """A host's own Batch row (Stitch's Trend column), in the Batch card."""
        self.batch_card.add_row(row)

    def _build_advanced(self):
        """The rarely changed options, on the shared Advanced dialog's rows. The minimum intensity is the Phases
        card's (REFERENCE-PHASES-20261011): one value or each phase's own."""
        self.advanced = AdvancedDialog(self, title="Advanced — Phase Analysis", object_name="phaseFitAdvancedDialog")
        self._maxfev_row = self._option(None, "constraints", self._int_row(
            "Max iterations", "max_nfev", "", 0, 1_000_000, blank=True),
            'The solver\'s evaluation budget (max_nfev); blank or 0 uses lmfit\'s default.')
        self.uncertainty_row = self._option(None, "constraints", FormRow(
            label="Use uncertainties", path=_path("use_uncertainties"), value=False, kind="bool"),
            'Weight points by their measured σ; no uncertainties are inferred. Available when the trace has them.')
        self.snip_width_row = self._option(None, "background", self._int_row(
            "SNIP width", "snip_width", "", 1, 1_000_000, blank=True),
            'The SNIP baseline\'s window, in points; blank uses 5 % of the trace.')
        self.march_rows = tuple(self._option(None, "shape", self._int_row(
            f"March-Dollase {axis}", f"march_{axis}", str(value), -20, 20),
            f'The March-Dollase preferred-orientation axis, {axis}; the axis must not be 0 0 0.')
            for axis, value in zip("hkl", (0, 0, 1)))
        self.free_height_row = self._option(None, "shape", _range("Free-height range", "free_height", "0", "10",
                                                                 ("0", "10")),
                                            'Bounds on each free per-peak height, relative to the structure\'s: '
                                            '0 ≤ low < high.')
        self.width_limits_row = self._option(None, "shape", _range("σ min / max", "width", "", "", ("auto", "auto")),
                                             'Bounds on each peak\'s width σ, in Å⁻¹; blank uses automatic limits.')
        for row in (self.free_height_row, self.width_limits_row):
            for edit in row.editors:
                edit.setValidator(QtGui.QDoubleValidator(self))
        for title, section, rows in (("Solver", "constraints", (self._maxfev_row,)),
                                     ("Weighting", "constraints", (self.uncertainty_row,)),
                                     ("Background", "background", (self.snip_width_row,)),
                                     ("Texture", "shape", (*self.march_rows, self.free_height_row)),
                                     ("Peak widths", "shape", (self.width_limits_row,))):
            self.advanced.add_group(title, rows).set_help_topic(TOPICS[section])
        self.use_uncertainties = self.uncertainty_row.editor
        self.use_uncertainties.setEnabled(False)
        self.max_nfev = self._maxfev_row.editor
        self.max_nfev.setPlaceholderText("auto")
        self.snip_width_row.editor.setPlaceholderText("auto")

    def _show_advanced(self):
        self.advanced.show(); self.advanced.raise_(); self.advanced.activateWindow()

    # ---- options --------------------------------------------------------
    def _fitted_entries(self):
        """The included CIF and lattice phases, in list order: a calibrant is a reference line only."""
        return tuple(entry for entry, shown in self.phase_source.listed_entries()
                     if shown and entry.source_kind != 'calibrant')

    def _included_calibrants(self):
        return tuple(entry.name for entry, shown in self.phase_source.listed_entries()
                     if shown and entry.source_kind == 'calibrant')

    def _template_in_use(self):
        return self._template is not None and self.fit_background_row.current_value().startswith("template")

    def _sync_options(self):
        """Each row that a choice makes irrelevant is off and says why; Texture's tip names lattice phases; the
        export buttons say why a fit cannot be exported."""
        fit_background = self.fit_background_row.current_value()
        texture = self.texture_row.current_value()
        amorphous = self.amorphous_row.current_value() != "none"
        lattice = any(entry.source_kind == 'lattice' for entry in self._fitted_entries())
        self.texture_row.set_purpose_tooltip(_TEXTURE_TIP + (_TEXTURE_LATTICE_TIP if lattice else ''))
        self.fit_background_row.set_purpose_tooltip(_FIT_BACKGROUND_TIP + (
            ' ' + _SUBSTRATE_NEEDED if fit_background.startswith("template") and self._template is None else ''))
        for rows, on, reason in (
                ((self.fit_background_degree_row,), fit_background in _DEGREE_BACKGROUNDS,
                 'Choose a polynomial, Chebyshev or spline background to set its degree.'),
                ((self.amorphous_center_row, self.amorphous_sigma_row), amorphous,
                 'Choose an amorphous peak to set its starting centre and σ.'),
                (self.march_rows, texture == "march_dollase", 'Choose March-Dollase texture to set its axis.'),
                ((self.free_height_row,), texture == "free",
                 'Choose free (per-peak) texture to set its height range.'),
                ((self.snip_width_row,), self.bkg_row.current_value() == "snip",
                 'Choose the SNIP pre-fit background to set its width.')):
            for row in rows:
                for editor in row.editors if isinstance(row, RangeRow) else (row.editor,):
                    editor.setEnabled(on)
                row.set_purpose_tooltip(self._tips[row] if on else f'{self._tips[row]} {reason}')
        if hasattr(self, 'export_actions'):
            reason = (self._lattice_export_reason() if lattice else
                      _TEMPLATE_EXPORT_REASON if self._template_in_use() else '')
            self.export_actions.set_available(not reason, reason)
        self._sync_substrate()

    def _sync_substrate(self):
        """The Background card's status names the captured substrate; Clear is available while one is."""
        template, wanted = self._template, self.fit_background_row.current_value().startswith("template")
        if template is not None:
            name = f"{template.title} ({template.label})" if template.title else template.label
            self.background_card.set_status_text(f"Substrate: {name}", tooltip=(
                'The captured substrate template the Substrate backgrounds scale, captured in Å⁻¹ through the fit\'s '
                'own input. Clear substrate forgets it.'))
        else:
            self.background_card.set_status_text("Substrate: not captured" if wanted else "", tooltip=(
                _SUBSTRATE_NEEDED if wanted else ''))
        spec = self.substrate_clear_btn.spec
        self.substrate_clear_btn.apply_spec(ControlActionSpec(
            spec.action, spec.label, spec.section, enabled=template is not None,
            reason='' if template is not None else 'No substrate is captured.'))

    def set_substrate_template(self, trace):
        """The substrate pattern the host captured through its own fit input (a q ``DisplayedTraceInput``), or None
        to clear it. A trace not in Å⁻¹ is refused, keeping the substrate before it; True when it is set."""
        if trace is not None:
            from xrd_tools.analysis.axis_units import require_inverse_angstrom
            from xrd_tools.analysis.display_fit_operations import DisplayedTraceInput
            if not isinstance(trace, DisplayedTraceInput):
                raise TypeError("a substrate is a DisplayedTraceInput")
            try:
                require_inverse_angstrom(trace.axis_unit, operation="a substrate template")
            except ValueError as error:
                self.status.setText(str(error))
                return False
        self._template = trace
        self._sync_options()
        if trace is not None:
            self.status.setText(f"Substrate captured: {trace.title or trace.label}.")
        return True

    @property
    def substrate_template(self):
        return self._template

    def sequential_batch(self):
        """Whether a Batch seeds each fit from the previous one (an analyzer option, not a plan field)."""
        return self._allow_batch and bool(self.sequential_row.current_value())

    @staticmethod
    def _range_values(row):
        """A RangeRow's two bounds as numbers, None for a blank one; ValueError names a bound that is no number."""
        values = []
        for edit in row.editors:
            text = edit.text().strip()
            try:
                values.append(float(text) if text else None)
            except ValueError:
                raise ValueError(f"{text!r} is not a number") from None
        return tuple(values)

    def _range_or(self, row, default):
        try:
            low, high = self._range_values(row)
        except ValueError:
            return default
        return (default[0] if low is None else low, default[1] if high is None else high)

    def _fit_range(self):
        """The fit range in the displayed axis (a blank bound is the trace's), or None: the whole trace."""
        try:
            low, high = self._range_values(self.fit_range_row)
        except ValueError:
            return None
        if low is None and high is None:
            return None
        if low is None or high is None:
            if self._x is None:
                return None
            extent = self._data_extent()
            low, high = extent[0] if low is None else low, extent[1] if high is None else high
        return (low, high)

    def _range_unit(self):
        """The displayed axis as a fit-range unit: a 2θ trace's own (Stitch), else Å⁻¹."""
        return self._x_unit if self._x_unit in ("2th_deg", "2th_rad") else "q_A^-1"

    def fit_refusal(self):
        """Why Fit or Batch cannot start with these options ("" when they can): a refused number, a list of only
        calibrants, or a Substrate background without a captured substrate."""
        for row in (self._maxfev_row, self.fit_background_degree_row, self.snip_width_row, *self.march_rows):
            if row.refusal():
                return f"{row.label.text()}: {row.refusal()}"
        for row in (self.fit_range_row, self.free_height_row, self.width_limits_row):
            try:
                self._range_values(row)
            except ValueError as error:
                return f"{row.label.text()}: {error}"
        if not self._fitted_entries() and self._included_calibrants():
            return _CALIBRANT_ONLY
        if self.fit_background_row.current_value().startswith("template") and self._template is None:
            return _SUBSTRATE_NEEDED
        return ""

    def vnext_fit_values(self):
        """The plan's options (``DisplayedPhaseFitPlan``'s fields but its trace and wavelength): every option, with
        the included CIF and lattice phases in list order and their own minimums."""
        cutoffs = self.phase_source.cutoffs
        entries = self._fitted_entries()
        fit_background = self.fit_background_row.current_value()
        fit_range = self._fit_range()
        values = {
            "cif_paths": (), "phase_names": tuple(entry.name for entry in entries), "expected_hashes": (),
            "phase_entries": entries,
            "prefit_background": self.bkg_row.current_value(), "phase_profile": self.profile_row.current_value(),
            "lattice_pct": float(self.lattice_row.current_value()) / 100.0, "min_intensity": cutoffs.global_value,
            "max_nfev": self.max_nfev.value() or None,
            "width_model": self.width_model_row.current_value(),
            "width_limits": self._range_or(self.width_limits_row, (None, None)),
            "texture": self.texture_row.current_value(),
            "march_axis": tuple(row.editor.value() for row in self.march_rows),
            "free_height_range": self._range_or(self.free_height_row, (0.0, 10.0)),
            "q_shift_bound": float(self.q_shift_row.current_value()),
            "lock_cross_phase": bool(self.cross_phase_row.current_value()),
            "snip_width": self.snip_width_row.editor.value(),
            "fit_background": fit_background, "fit_background_degree": self.fit_background_degree_row.editor.value(),
            "background_template": self._template if fit_background.startswith("template") else None,
            "amorphous_peak": self.amorphous_row.current_value(),
            "amorphous_center": float(self.amorphous_center_row.current_value()),
            "amorphous_sigma": float(self.amorphous_sigma_row.current_value()),
            "fit_range": fit_range, "fit_range_unit": "q_A^-1" if fit_range is None else self._range_unit(),
        }
        if not cutoffs.global_on:
            # each included phase's own minimum, in the fit's phase order (a lattice phase's is ignored)
            own = dict(cutoffs.per_phase)
            values['phase_min_intensities'] = tuple(float(own.get(entry.id, cutoffs.global_value))
                                                    for entry in entries)
        if self.use_uncertainties.isChecked() and self.use_uncertainties.isEnabled():
            values['use_uncertainties'] = True
        return values

    @staticmethod
    def _lattice_export_reason():
        return ('PHASE_LATTICE_NOTEBOOK_UNAVAILABLE: manual lattice phases cannot be exported '
                'in the existing CIF notebook format.')

    def export_fit_values(self):
        """Keep structural-library exports on the existing portable CIF recipe; Sequential travels with them."""
        values = dict(self.vnext_fit_values())
        if values.get("background_template") is not None:
            raise ValueError(_TEMPLATE_EXPORT_REASON)
        entries = values.pop('phase_entries', ())
        if any(entry.source_kind == 'lattice' for entry in entries):
            raise ValueError(self._lattice_export_reason())
        if entries:
            values.update(cif_paths=tuple(str(entry.cif_path) for entry in entries),
                          phase_names=tuple(entry.name for entry in entries),
                          expected_hashes=tuple(entry.id for entry in entries))
        if self.sequential_batch():
            values["sequential"] = True
        return values

    # ---- actions and the host's projections -----------------------------
    def _sync_actions(self):
        """Fit and Batch say why they cannot start (the host's reason, Int's held Batch); the list stays editable."""
        fit_reason = self._fit_reason
        self.fit_btn.setEnabled(not self._running and not fit_reason)
        self.fit_btn.setToolTip(fit_reason or _FIT_TIP)
        batch_reason = self._batch_reason or fit_reason
        if self._running:
            self.batch_btn.setEnabled(self._allow_batch)
            self.batch_btn.setToolTip("Cancel the batch; the running solve finishes first.")
        else:
            self.batch_btn.setEnabled(not batch_reason)
            self.batch_btn.setToolTip(batch_reason or _BATCH_TIP)
        # a capture shares the host's fit worker: it waits for a running batch rather than replace it
        spec = self.substrate_capture_btn.spec
        self.substrate_capture_btn.apply_spec(ControlActionSpec(
            spec.action, spec.label, spec.section, enabled=not self._running,
            reason='A batch is fitting; capture the substrate when it is done.' if self._running else ''))

    def set_fit_reason(self, reason=''):
        """The host's reason no fit can run now (Stitch: no result is open), shown on Fit and Batch."""
        self._fit_reason = str(reason or '')
        self._sync_actions()

    def keyPressEvent(self, event):
        # Esc must not discard the fit setup (phases/range/results); the default
        # QDialog Esc -> reject closes the popup.  Match the metadata plotter.
        # Other keys pass through.
        if event.key() == QtCore.Qt.Key.Key_Escape:
            event.accept()
            return
        super().keyPressEvent(event)

    def refresh_pattern(self):
        """Re-grab the host's displayed pattern (Reload)."""
        self._vnext_submit("reload", None)

    def _do_fit(self):
        refusal = self.fit_refusal()
        if refusal:
            self.status.setText(refusal)
            return
        if self._x is None:  # nothing shown yet: the displayed trace first, as Reload, so its fit can be drawn
            self._vnext_submit("reload", None)
            if self._x is None:
                return
        self._vnext_submit("fit", self.vnext_fit_values())

    def _show_pattern(self, x, y, x_label, *, clear_fit=False):
        """Draw the pattern; ``clear_fit``: the last fit goes too (its residual and rows), in the same redraw."""
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._x_label = x_label or "q"
        self.trend.set_value_unit(self._x_label)
        lower = FitPanel(zero_line=True) if clear_fit else self.fit_view.projection.lower
        self.fit_view.reconcile(FitProjection(self._pattern_panel(), lower))
        if clear_fit:
            self.table.clear_rows()

    def _pattern_panel(self, *series, markers=()):
        """The upper panel: the pattern as a line ("data"), then ``series`` (fit, background,
        components) and the peak ``markers``."""
        return FitPanel((FitSeries(self._x, self._y, "data_line", "data"), *series),
                        self._x_label, "Intensity", markers=markers)

    def set_vnext_trace(self, trace):
        self._export_trace = trace
        self._x_unit = trace.axis_unit
        self.use_uncertainties.setEnabled(trace.sigma is not None)
        if trace.sigma is None:
            self.use_uncertainties.setChecked(False)
        self._show_pattern(trace.axis, trace.intensity, trace.axis_unit or trace.label, clear_fit=True)
        self._fields_to_region()
        self.status.setText("Displayed trace captured. Check the phases to fit and click Fit.")

    def clear_input(self):
        """Release a detached displayed input when its host changes scope or closes."""
        self._x = self._y = self._export_trace = None
        self.fit_view.set_fit_range(None)
        self.fit_view.reconcile(FitProjection(FitPanel(), FitPanel(zero_line=True)))
        self.table.clear_rows()
        self.fit_btn.setEnabled(False)
        self.use_uncertainties.setChecked(False)
        self.use_uncertainties.setEnabled(False)
        if self._receipts:
            self._receipts = {}
            self.phase_source.set_details({})

    def export_snapshot(self):
        if self._export_snapshot_provider is not None:
            return self._export_snapshot_provider()
        if self._export_trace is None:
            return None
        from .fit_export import snapshot_from_trace, plain_settings
        return snapshot_from_trace(self._export_trace, wavelength=self._export_wavelength,
                                   settings=plain_settings(self.vnext_fit_values()))

    def set_vnext_result(self, result):
        fit_x = self._x[np.isfinite(self._x) & np.isfinite(self._y)]
        upper, markers, lower = fit_result_series(fit_x, result,
            tuple(zip(getattr(result, 'phase_names', ()) or tuple(receipt.phase_name for receipt in result.cif_receipts),
                      result.phase_components, strict=True)))
        self.fit_view.reconcile(FitProjection(self._pattern_panel(*upper, markers=markers),
                                              FitPanel(lower, zero_line=True)))
        # Each fitted CIF row shows the receipt of the bytes the fit read, by its library id (never its position).
        self._receipts = {receipt.expected_sha256 or receipt.sha256: "\n".join(map(str, (
            receipt.phase_name, receipt.lexical_path, receipt.resolved_path, receipt.byte_count, receipt.sha256,
            receipt.expected_sha256, receipt.pre_state, receipt.post_state, receipt.receipt_fingerprint)))
            for receipt in result.cif_receipts}
        self.phase_source.set_details(self._receipts)
        self._show_phase_rows(phase_result_rows(result), getattr(result, 'phase_fractions_reason', ''))
        detail = "; ".join((*result.diagnostics,
                            *((result.phase_fractions_reason,) if getattr(result, 'phase_fractions_reason', '') else ())))
        calibrants = self._included_calibrants()
        self.status.setText((result.message or result.code) +
            (f"; wavelength={result.wavelength_angstrom}" if result.wavelength_angstrom else "") +
            (f"; {detail}" if detail else "") +
            (f". Calibrants {', '.join(calibrants)} are reference lines only and were not fitted." if calibrants
             else ""))
        self.status.setToolTip("\n".join((result.plan_fingerprint,
            result.policy_fingerprint, result.result_fingerprint)))

    def showEvent(self, event):
        # Opening the window lists entries other tools added (an explicit library read); reconciles never read it.
        self.phase_source.refresh_library()
        super().showEvent(event)

    def closeEvent(self, event):
        self.phase_source.invalidate_add()  # a CIF still being read selects nothing in a closed window
        self._vnext_submit("close", None)
        super().closeEvent(event)

    def _show_phase_rows(self, projection, reason=''):
        """The table's rows; a withheld fraction (a "—" Value cell) and the fraction notice carry its reason (R4)."""
        headers, rows = projection
        tips = {}
        for row_index, row in enumerate(rows):
            for column, value in enumerate(row):
                tip = (reason if headers[column] == "Value" else '') if value == '—' else (
                    reason if reason and value == reason else '')
                if tip:
                    tips[row_index, column] = tip
        self.table.show_rows(headers, rows, cell_tooltips=tips)

    # ---- the fit range --------------------------------------------------
    def _data_extent(self):
        if self._x is None or self._x.size == 0:
            return (0.0, 1.0)
        return (float(np.nanmin(self._x)), float(np.nanmax(self._x)))

    def _region_to_fields(self, lo, hi):
        """The fit range dragged on the plot, written into the Fit range row."""
        low, high = self.fit_range_row.editors
        low.setText(f"{lo:.6g}")
        high.setText(f"{hi:.6g}")

    def _fields_to_region(self):
        """The Fit range row's values (the whole trace while blank), shown as the plot's fit range."""
        if self._x is None:
            self.fit_view.set_fit_range(None)
            return
        extent = self._data_extent()
        bounds = self._fit_range()
        self.fit_view.set_fit_range(extent if bounds is None else bounds, limits=extent)

    def _on_full(self):
        for edit in self.fit_range_row.editors:
            edit.setText("")
        self._fields_to_region()

    # ---- batch hooks (driven by the host) -------------------------------
    def batch_x_range(self):
        """The whole finite pattern: the fit range masks only the solve (the shared batch contract)."""
        return (-np.inf, np.inf)

    def set_batch_running(self, running):
        self._running = bool(running)
        self.batch_btn.setText("Cancel" if running else "Batch")
        self._sync_actions()

    def set_batch_progress(self, done, total):
        self.status.setText(f"Phase-fitting… {done}/{total}")

    def reset_param_trend(self):
        """A fresh trend (Reload, a new batch run)."""
        self.trend.clear()

    # ---- saved settings (the shared PageSettings owner) -----------------
    def _build_settings(self):
        """Save settings… and Load settings…: every option as a plain field and the phase list, through the shared
        owner, with its page surface (``_status``, ``_readiness``, ``_value``, ``_closing``, the chooser)."""
        rows = {"profile": self.profile_row, "width_model": self.width_model_row, "texture": self.texture_row,
                "prefit_background": self.bkg_row, "fit_background": self.fit_background_row,
                "fit_background_degree": self.fit_background_degree_row, "amorphous_peak": self.amorphous_row,
                "amorphous_center": self.amorphous_center_row, "amorphous_sigma": self.amorphous_sigma_row,
                "lattice_pct": self.lattice_row, "q_shift": self.q_shift_row,
                "lock_cross_phase": self.cross_phase_row, "sequential": self.sequential_row,
                "max_nfev": self._maxfev_row, "use_uncertainties": self.uncertainty_row,
                "snip_width": self.snip_width_row,
                **{f"march_{axis}": row for axis, row in zip("hkl", self.march_rows)}}
        #: each saved field: its row and, for a range, the bound
        self._settings_rows = {key: (row, None) for key, row in rows.items()}
        for key, row in (("fit_range", self.fit_range_row), ("free_height", self.free_height_row),
                         ("width", self.width_limits_row)):
            self._settings_rows[f"{key}_low" if key != "width" else "width_min"] = (row, 0)
            self._settings_rows[f"{key}_high" if key != "width" else "width_max"] = (row, 1)
        self._int_rows = (self.fit_background_degree_row, self._maxfev_row, self.snip_width_row, *self.march_rows)
        self._fields = {key: self._descriptor(key, row, bound) for key, (row, bound) in self._settings_rows.items()}
        self._settings_path_chooser = lambda action, start: self.settings_io.choose_path(action, start)
        self.settings_io = PageSettings(self, PHASE_ANALYSIS_SETTINGS, document=self._settings_document,
                                        apply=self._apply_settings, busy=self._busy_reason,
                                        close=lambda: CloseReceipt(PageCleanup.CLEAN),
                                        snapshot=lambda: self._template, restore=self._restore_runtime)

    def _descriptor(self, key, row, bound):
        """The shared field descriptor ``check_fields`` reads: a choice among the stored values, a toggle, a whole
        number with its bounds, or a number."""
        label = row.label.text() + ("" if bound is None else (" (low)", " (high)")[bound])
        if bound is None and isinstance(row.editor, QtWidgets.QComboBox):
            choices = tuple(row.editor.itemData(index) for index in range(row.editor.count()))
            return ControlFormField(SectionId.PROCESSING, label, row.path, kind=ControlFieldKind.COMBO, choices=choices)
        if bound is None and isinstance(row.editor, QtWidgets.QPushButton):
            return ControlFormField(SectionId.PROCESSING, label, row.path, kind=ControlFieldKind.BOOL)
        if bound is None and key in self._int_bounds:
            minimum, maximum, blank = self._int_bounds[key]
            return ControlFormField(SectionId.PROCESSING, label, row.path, kind=ControlFieldKind.INT, minimum=minimum,
                                    maximum=maximum, allow_blank=blank)
        return ControlFormField(SectionId.PROCESSING, label, _path(key))

    def _setting(self, key):
        """A saved field's current value; ValueError names a row whose text is refused."""
        row, bound = self._settings_rows[key]
        if bound is not None:
            try:
                return self._range_values(row)[bound]
            except ValueError as error:
                raise ValueError(f"{row.label.text()}: {error}") from None
        if row in self._int_rows:
            if row.refusal():
                raise ValueError(f"{row.label.text()}: {row.refusal()}")
            return row.editor.value()
        return row.current_value()

    @property
    def _readiness(self):
        return ("", True)

    def _value(self, key):
        return "" if key == "project_root" else self._setting(key)

    def _status(self, text, *, ready=True):
        """The settings owner's message, on this window's status line; a load's notice (phases it skipped, a
        substrate to capture again) stays after its success message."""
        notice, self._load_notice = self._load_notice, ''
        self.status.setText(" ".join(part for part in (str(text), notice) if part))

    def _busy_reason(self):
        if self._running:
            return "Phase Analysis is fitting a batch; load settings when it is done."
        if self.phase_source.add_worker.busy:
            return "Phase Analysis is reading CIF files; load settings once they are added."
        if self.busy_probe is not None and self.busy_probe():
            return "Phase Analysis is fitting; load settings when the fit is done."
        return ""

    def _settings_action(self, action):
        outcome = action()
        if hasattr(outcome, 'reason'):
            self.status.setText(outcome.reason)

    def _settings_document(self, _page):
        """Every option (plain values, a range as its two bounds) and the phase list: each row's id, include flag
        and own minimum, the switch and the global minimum. The substrate and Sequential's seeds are not saved."""
        cutoffs = self.phase_source.cutoffs
        own = dict(cutoffs.per_phase)
        names = {entry.id: entry.name for entry, _shown in self.phase_source.listed_entries()}
        rows = [{"id": identifier, "included": bool(shown),
                 **({"name": names[identifier]} if identifier in names else {}),
                 **({"minimum": float(own[identifier])} if identifier in own else {})}
                for identifier, shown in self.phase_source.selection]
        return {"kind": PHASE_ANALYSIS_SETTINGS.kind, "version": PHASE_ANALYSIS_SETTINGS.version,
                "fields": {key: plain(self._setting(key)) for key in self._settings_rows},
                "phases": {"rows": rows, "global_on": bool(cutoffs.global_on), "value": float(cutoffs.global_value)}}

    def _checked_fields(self, document):
        """The document's fields this window has, each within its row's choices and bounds, and the ranges in
        order: ValueError names the first that is not (nothing applies)."""
        fields = saved_fields(self, document)
        check_fields(self, fields)
        for key, value in fields.items():
            row, bound = self._settings_rows[key]
            field = self._fields[key]
            if field.kind in (ControlFieldKind.COMBO, ControlFieldKind.BOOL):
                continue
            if field.kind is ControlFieldKind.INT:
                if value is None and field.allow_blank:
                    continue
                refusal = ('is not a whole number' if type(value) is not int else integer_field_refusal(
                    str(value), minimum=field.minimum, maximum=field.maximum, allow_blank=field.allow_blank))
            elif bound is not None:
                refusal = '' if value is None or _number(value) else 'is not a number'
            else:
                editor = row.editor
                refusal = ('is not a number' if not _number(value) else
                           '' if editor.minimum() <= value <= editor.maximum() else
                           f'is outside {editor.minimum():g} to {editor.maximum():g}')
            if refusal:
                raise ValueError(f"{field.label}: {value!r} {refusal}")
        current = {key: self._setting(key) for key in self._settings_rows}
        merged = {**current, **fields}
        for key in ("fit_range", "free_height", "width"):
            low, high = (merged[f"{key}_low"], merged[f"{key}_high"]) if key != "width" else (
                merged["width_min"], merged["width_max"])
            label = self._settings_rows[f"{key}_low" if key != "width" else "width_min"][0].label.text()
            if key == "free_height":  # a blank bound is the row's own default, as the fit reads it
                low, high = 0.0 if low is None else low, 10.0 if high is None else high
                if not 0 <= low < high:
                    raise ValueError(f"{label}: needs 0 ≤ low < high")
            if key == "width" and any(value is not None and value <= 0 for value in (low, high)):
                raise ValueError(f"{label}: σ limits must be positive")
            if key != "free_height" and low is not None and high is not None and not low < high:
                raise ValueError(f"{label}: low must be less than high")
        return fields

    def _checked_phases(self, document):
        """The document's phase list as a selection, its minimums and the rows the library no longer holds, or None
        when the document keeps the list; ValueError says why a list is malformed."""
        from .phase_source import PhaseCutoffs
        phases = document.get("phases")
        if phases is None:
            return None
        rows = phases.get("rows", [])
        if type(rows) is not list or len(rows) > 8:
            raise ValueError("Phase Analysis settings phases need a list of at most eight phases")
        cutoffs = self.phase_source.cutoffs
        global_on, value = phases.get("global_on", cutoffs.global_on), phases.get("value", cutoffs.global_value)
        if type(global_on) is not bool or not _number(value) or not 0 <= value <= 100:
            raise ValueError("Phase Analysis settings minimum intensity is malformed")
        seen = set()
        for row in rows:
            identifier = row.get("id") if type(row) is dict else None
            minimum = row.get("minimum") if type(row) is dict else None
            if (type(identifier) is not str or not identifier or identifier in seen
                    or type(row.get("included")) is not bool
                    or minimum is not None and not (_number(minimum) and 0 <= minimum <= 100)):
                raise ValueError("Phase Analysis settings phases are malformed")
            seen.add(identifier)
        selection, own, missing, kinds = [], {}, [], {}
        for row in rows:
            lookup = self.phase_source.library.resolve(row["id"])
            if lookup.entry is None:
                name = row.get("name") if type(row.get("name")) is str else ""
                missing.append(f"{name} ({row['id'][:8]}…)" if name else f"{row['id'][:12]}…")
                continue
            selection.append((row["id"], row["included"]))
            kinds[row["id"]] = lookup.entry.source_kind
            if row.get("minimum") is not None:
                own[row["id"]] = float(row["minimum"])
        # as the source keeps them: stored values for listed phases; with the switch off, each CIF phase its own
        if global_on:
            per_phase = tuple((identifier, own[identifier]) for identifier, _shown in selection if identifier in own)
        else:
            per_phase = tuple((identifier, own.get(identifier, float(value))) for identifier, _shown in selection
                              if identifier in own or kinds[identifier] == 'cif')
        return tuple(selection), PhaseCutoffs(global_on, float(value), per_phase), tuple(missing)

    def _apply_settings(self, _page, document):
        """Check the whole document first, then restore its options and its phase list (as ordered plot-option
        edits, through the binding); a successful load clears the substrate, which is never saved."""
        document = PHASE_ANALYSIS_SETTINGS.validate(document)
        fields = self._checked_fields(document)
        phases = self._checked_phases(document)
        for key, value in fields.items():
            row, bound = self._settings_rows[key]
            if bound is not None:
                edit = row.editors[bound]
                edit.setText("" if value is None else f"{float(value):.6g}")
                edit.setModified(False)
            elif isinstance(row.editor, QtWidgets.QComboBox):
                row.editor.setCurrentIndex(row.editor.findData(value))
            elif isinstance(row.editor, QtWidgets.QPushButton):
                row.editor.setChecked(bool(value))
            else:
                row.set_number(value)
        notices = []
        if phases is not None:
            selection, cutoffs, missing = phases
            self.phase_source.set_cutoffs(cutoffs)
            self.phase_source.set_selection(selection)
            self.phase_binding.submit_cutoffs(cutoffs)
            if missing:
                notices.append(f"Missing phase library entries were skipped: {', '.join(missing)}.")
        self._template = None
        self._sync_options()
        self._fields_to_region()
        if self.fit_background_row.current_value().startswith("template"):
            notices.append(_RECAPTURE_NOTICE)
        self._load_notice = " ".join(notices)

    def _restore_runtime(self, template):
        """A load that could not apply: the substrate before it, and no load notice."""
        self._template = template
        self._load_notice = ''
        self._sync_options()
