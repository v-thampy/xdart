"""Native Stitch workspace over the existing finite operation owner.

Source admission, calibration, raw reads and publication stay with their shared
headless owners. This page queues their exact prepared requests and presents
the resulting curves and original contributing frames.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import logging
import os
import time

import re

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from xdart.gui.pages.handle import AppMenuHosts, PageHandle
from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.pages.values import (
    ActionAccepted, ActionCompleted, ActionRefused, PageCleanup,
    RSM_TOOL_KEY, SCATTERING_PAGE_KEY, STITCH_TOOL_KEY,
)
from xdart.gui.tools.stitch_owner import (
    StitchOwnerAction, StitchOwnerFinalization, StitchOwnerOutcomeKind,
    StitchToolOwner,
)
from xdart.gui.tools.stitch_values import StitchFrameSelector, StitchToolForm
from xdart.gui.widgets.controls_panel import (
    PROCESSING_ROW_GAP, PROCESSING_ROW_LABEL_WIDTH, ControlsPanel, FormRow, RangeRow, SubsectionCard,
)
from xdart.gui.widgets.run_controls import RunControlsBar
from xdart.gui.widgets.scientific_plot import scaled_intensity_label
from xdart.gui.tabs.scattering.shell_values import ScientificPlotOptions
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.workspace_shell import _HugWidthScrollArea
from xrd_tools.analysis.module_transaction import MetadataColumnSelector, ModuleDisposition
from xrd_tools.analysis.stitch_operation import StitchGeometryKind
from xrd_tools.analysis.stitch_sources import StitchSourceSelection
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, AnalysisArtifactPayload
from xrd_tools.session.readiness import ControlFieldKind, ControlFormField, SectionId
from xrd_tools.session.threshold_editing import (
    ThresholdEditRefusal, ThresholdState, threshold_transition,
)

_FRAME_CHECKS = {"Exclude empty frames": "empty", "Exclude empty and partial frames": "partial",
                 "Keep all frames": "none"}
_SAMPLE_POSITIONS = {"As calibrated": "calibrated", "On rotation axis": "rotation_axis"}


_LOG = logging.getLogger(__name__)
_PAIR_SEPARATOR = re.compile(r"[,;\n]+")


def _parse_pairs(text):
    """``name=value`` entries separated by commas, semicolons or lines, sorted."""
    pairs = []
    for token in _PAIR_SEPARATOR.split(str(text)):
        token = token.strip()
        if not token:
            continue
        left, separator, right = token.partition("=")
        if not separator or not left.strip() or not right.strip():
            raise ValueError("mapping entries must use name=value")
        pairs.append((left.strip(), right.strip()))
    return tuple(sorted(pairs))


#: A position's motor defaults to its name without one of these (del_value → del).
_POSITION_SUFFIX = re.compile(r"(?:_angle|_value|_deg|_pos|_position)$", re.IGNORECASE)


def _goniometer_positions(path) -> tuple[str, ...]:
    """The positions a pyFAI goniometer JSON declares, or () when it cannot be read."""
    try:
        file = Path(str(path or "").strip())
        if file.suffix.lower() != ".json" or file.stat().st_size > 1 << 20:
            return ()
        names = json.loads(file.read_text(encoding="utf-8"))["trans_function"]["pos_names"]
    except (OSError, ValueError, KeyError, TypeError):
        return ()
    if type(names) is not list or not names or any(type(name) is not str or not name.strip()
                                                   for name in names):
        return ()
    return tuple(names)


def _parse_references(text):
    return tuple((name, float(value)) for name, value in _parse_pairs(text))


def _choice(choices, value):
    return next(label for label, key in choices.items() if key == value)


def frame_screen_note(summary):
    """One status phrase for the frames the dropped-frame check excluded or flagged."""
    if not summary:
        return ""
    excluded, partial = list(summary["excluded"]), list(summary["partial"])
    flagged = [label for label in partial if label not in excluded]
    parts = []
    if excluded:
        parts.append(f"excluded dropped frames {', '.join(map(str, excluded))}")
    if flagged:
        parts.append(f"partial frames kept {', '.join(map(str, flagged))}")
    return " · ".join(parts)


def saved_frame_screen_note(records):
    """One status phrase for the frame check verdicts a saved result recorded."""
    if not records:
        return ""
    if any("screen" not in record for record in records):
        return "frame check not recorded"
    excluded = sum(record["excluded"] for record in records)
    partial = sum(record["screen"] == "partial" and not record["excluded"] for record in records)
    parts = []
    if excluded:
        parts.append(f"{excluded} dropped frame{'' if excluded == 1 else 's'} excluded")
    if partial:
        parts.append(f"{partial} partial frame{'' if partial == 1 else 's'} kept")
    return " · ".join(parts)


class StitchPage(QtWidgets.QWidget):
    """One retained page; one worker owns preview, science and strict reopen."""

    toolRequested = QtCore.Signal(str)

    def __init__(self, services, parent=None):
        super().__init__(parent)
        self.setObjectName("stitchPage")
        self._services = services
        self._owner = StitchToolOwner()
        self._source_paths = ()
        self._paths_by_kind = {}
        self._range_rows = []
        self._range_keys = set()
        self._threshold_state = ThresholdState()
        self._auto_threshold_max = None
        self._queued_jobs = []
        self._job_total = 0
        self._batch_started = None
        self._raw_revision = 0
        self._raw_request_revision = None
        self._raw_pending = False
        self._raw_popup = None
        self._raw_frame_view = None
        self._calibration_dialog = None
        # Saved calibration-dialog inputs, applied when the dialog is created.
        self._calibration_settings = None
        # (action, start directory) -> path for File ▸ Save/Load Settings; tests replace it.
        self._settings_path_chooser = self._choose_settings_path
        self._closing_settings = None
        self._settings_remembered = False
        self._metadata_dialog = None
        self._metadata_request_dialog = None
        self._metadata_view_revision = None
        self._authoring_dialogs = []
        self._fields = {}
        self._rows = {}
        self._revision = 0
        self._prepared_revision = None
        self._action_revision = None
        self._view_revision = 0
        self._action_view_revision = None
        self._started = None
        self._closing = False
        self._payloads = {}
        self._plot_options = ScientificPlotOptions()
        self._diagnostic = "Intensity"
        #: One display key per result, created once: the view compares keys by identity.
        self._result_keys = {}
        self._plot_mode, self._plot_axis = "Single", "Q"
        self._color_map, self._log_scale = "Default", False
        self._build_ui()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)
        self._authoring_timer = QtCore.QTimer(self)
        self._authoring_timer.setInterval(50)
        self._authoring_timer.timeout.connect(self._sync_actions)
        self._status("Ready · Stitch 1D")
        self._sync_actions()
        self._refresh_inputs_view()
        self._draw()  # the view takes Stitch's layout before the first result

    def _build_ui(self):
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(self.splitter)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(5, 5, 5, 5)
        menus = QtWidgets.QHBoxLayout()
        for title, name in (("File", "fileMenuButton"), ("Config", "configMenuButton"),
                            ("Help", "helpMenuButton")):
            button = QtWidgets.QToolButton()
            button.setObjectName(name)
            button.setText(title)
            button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
            menu = QtWidgets.QMenu(button)
            button.setMenu(menu)
            setattr(self, title.lower() + "_menu", menu)
            menus.addWidget(button)
        left_layout.addLayout(menus)
        self.file_menu.addAction("Open result…", self.open_result)
        self.file_menu.addAction(
            "Integration workspace", lambda: self.toolRequested.emit(str(SCATTERING_PAGE_KEY)),
        )
        heading = QtWidgets.QLabel("DATA BROWSER")
        heading.setObjectName("browserHeading")
        left_layout.addWidget(heading)
        self.results = QtWidgets.QListWidget()
        self.results.setObjectName("stitchResults")
        self.results.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.results.itemSelectionChanged.connect(self._draw)
        self.results.currentItemChanged.connect(self._current_result_changed)
        self.frames = QtWidgets.QListWidget()
        self.frames.setObjectName("stitchFrames")
        self.frames.currentItemChanged.connect(self._frame_changed)
        browser = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        for caption, widget in (("Results", self.results), ("Frames", self.frames)):
            pane = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(pane)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(QtWidgets.QLabel(caption))
            layout.addWidget(widget, 1)
            browser.addWidget(pane)
        browser.setSizes([160, 70])
        left_layout.addWidget(browser, 1)
        browse_actions = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("Open…")
        self.open_button.clicked.connect(self.open_result)
        self.metadata_button = QtWidgets.QPushButton("Metadata")
        self.metadata_button.clicked.connect(self._show_metadata)
        browse_actions.addWidget(self.open_button)
        browse_actions.addWidget(self.metadata_button)
        self.raw_button = QtWidgets.QPushButton("Raw image")
        self.raw_button.clicked.connect(self._show_raw)
        left_layout.addWidget(self.raw_button)
        left_layout.addLayout(browse_actions)
        self.splitter.addWidget(left)

        # Int's scientific display, fed Stitch results: one pane for both pages, not a copy.
        from xdart.gui.tabs.scattering.scientific_view import ScientificView
        self.scientific = ScientificView()
        # Stitch results are integrated 1-D curves (Int 1D's layout: Q or 2θ); Stitch has no
        # normalization channel, 1-D Background or raw popup on this pane.
        self.scientific.set_hidden_tools("norm", "background", "raw")
        self.scientific.commandRequested.connect(self._view_command)
        self.curve = self.scientific.curve
        self._install_diagnostic_option()
        self.splitter.addWidget(self.scientific)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.controls = ControlsPanel(show_section_numbers=False)
        self.controls.experiment_card.title.setText("EXPERIMENT")
        scroll = _HugWidthScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.controls)
        right_layout.addWidget(scroll, 1)
        self._build_controls()
        self._source_changed()
        self._render_ranges()
        self.run_bar = RunControlsBar()
        # The same modes as Int: another one leaves for the integration workspace.
        from xdart.gui.tabs.scattering.run_mode_projection import (
            RUN_MODE_CHOICES, UNOWNED_RUN_MODE_REASONS,
        )
        self.run_bar.apply_profile(modes=RUN_MODE_CHOICES, live=False, batch=False, cores=False)
        self.run_bar.set_unavailable_modes(UNOWNED_RUN_MODE_REASONS)
        self.run_bar.modeCombo.setCurrentText("Stitch 1D")
        self.run_bar.modeChanged.connect(self._mode_chosen)
        self.run_bar.writeModeButton.setText("Replace ⇄")
        self.run_bar.writeModeButton.setChecked(True)
        self.run_bar.writeModeButton.setEnabled(False)
        self.run_bar.writeModeButton.setToolTip("Replace only this output after successful publication")
        self.run_bar.actionClicked.connect(self.begin_run)
        self.run_bar.stopClicked.connect(self.stop)
        right_layout.addWidget(self.run_bar)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([230, 720, 360])

    def _mode_chosen(self, mode):
        """Leave for the chosen mode's page (the host refuses while Stitch runs)."""
        if mode == "Stitch 1D":
            return
        blocker = QtCore.QSignalBlocker(self.run_bar.modeCombo)
        self.run_bar.modeCombo.setCurrentText("Stitch 1D")
        del blocker
        self.toolRequested.emit(str(RSM_TOOL_KEY if mode == "RSM" else SCATTERING_PAGE_KEY))

    def _field(self, card, section, key, label, value="", *, choices=(), browse=False):
        kind = ControlFieldKind.COMBO if choices else ControlFieldKind.LINE
        path = ("Project", "project_folder") if key == "project_root" else ("Stitch", key)
        field = ControlFormField(section, label, path, value, kind, tuple(choices), browse)
        row = FormRow(label=label, path=path, value=value, kind=kind.value,
                      choices=choices, browse=browse)
        # The editor keeps the shared Controls object name, so Int's styling (one height
        # for every box) applies here too.
        row.editor.setProperty("stitchField", key)
        row.valueChanged.connect(self._edited)
        row.draftChanged.connect(self._edited)
        row.browseRequested.connect(lambda _path, name=key: self._browse(name))
        if card is not None:
            card.add_row(row)
        self._fields[key], self._rows[key] = field, row
        return row

    @staticmethod
    def _align_to_ranges(row):
        """As Int's Axis row: this box shares the range boxes' left edge."""
        row.label.setMinimumWidth(PROCESSING_ROW_LABEL_WIDTH)
        row.layout().setSpacing(PROCESSING_ROW_GAP)

    def _build_controls(self):
        project, experiment, source, processing = (
            self.controls.project_card, self.controls.experiment_card,
            self.controls.source_card, self.controls.processing_card,
        )
        self._field(project, SectionId.PROJECT, "project_root", "Folder", browse=True)
        self._rows["project_root"].valueChanged.connect(self._refresh_xu_geometry)
        self._field(project, SectionId.PROJECT, "output_path", "Save Path", browse=True)
        self._field(experiment, SectionId.EXPERIMENT, "backend", "Backend",
                    "pyFAI MultiGeometry", choices=("pyFAI MultiGeometry", "xrayutilities histogram"))
        authoring = QtWidgets.QWidget()
        authoring_layout = QtWidgets.QHBoxLayout(authoring)
        authoring_layout.setContentsMargins(0, 0, 0, 0)
        self.calibrate_button = QtWidgets.QPushButton("⌖ Calibrate")
        self.calibrate_button.clicked.connect(self._calibrate)
        self.mask_button = QtWidgets.QPushButton("▦ Make Mask")
        self.mask_button.clicked.connect(self._make_mask)
        authoring_layout.addWidget(self.calibrate_button)
        authoring_layout.addWidget(self.mask_button)
        experiment.add_row(authoring)
        self._field(experiment, SectionId.EXPERIMENT, "geometry_path", "Calibration", browse=True)
        self._rows["geometry_path"].valueChanged.connect(self._refresh_xu_geometry)
        self._field(experiment, SectionId.EXPERIMENT, "mask_path", "Mask File", browse=True)
        self._rows["backend"].valueChanged.connect(self._backend_changed)
        self._field(source, SectionId.SOURCE, "source_kind", "Source", "Spec",
                    choices=("Spec", "NeXus", "Directory"))
        self._field(source, SectionId.SOURCE, "spec_path", "SPEC files", browse=True)
        self._field(source, SectionId.SOURCE, "image_dir", "Images", browse=True)
        self._field(source, SectionId.SOURCE, "scan_mode", "Scans", "Select Scans",
                    choices=("Stitch All", "Select Scans"))
        self._field(source, SectionId.SOURCE, "scan", "Selection", "1")
        self._rows["scan"].setToolTip("1,2,[2-5,8] produces three outputs. Brackets combine scans into one result.")
        self._rows["source_kind"].valueChanged.connect(self._source_changed)
        self._rows["scan_mode"].valueChanged.connect(self._source_changed)
        self._rows["spec_path"].valueChanged.connect(self._source_path_edited)
        grid = SubsectionCard("1-D", accent="processing")
        processing.add_row(grid)
        axis = self._field(grid, SectionId.PROCESSING, "unit", "Axis", "Q (Å⁻¹)", choices=("Q (Å⁻¹)", "2θ (°)"))
        self._rows["unit"].valueChanged.connect(self._processing_axis_changed)
        # As Int: the Pts cluster sits on the Axis row (ControlsPanel._attach_points_to_row).
        self._align_to_ranges(axis)
        points_label = QtWidgets.QLabel("Pts")
        points_label.setObjectName("controlsHeaderLabel")
        axis.add_trailing_widget(points_label)
        points = self._field(None, SectionId.PROCESSING, "npt_1d", "Pts", "1500")
        points.label.hide()
        points.setMaximumWidth(58)
        axis.add_trailing_widget(points)
        self.radial_range = self._range(grid, "Q (Å⁻¹)", ("q_min", "q_max", "radial_auto"), (0., 6.2, True))
        conditioning = SubsectionCard("Conditioning", accent="processing")
        processing.add_row(conditioning)
        self.threshold_range = self._range(conditioning, "Threshold", ("threshold_min", "threshold", "threshold_enabled"), (0., None, True),
                                           display_decimals=0)  # as Int's threshold row
        self.threshold_range._high.setPlaceholderText("Auto")
        self._align_to_ranges(self._field(conditioning, SectionId.PROCESSING, "monitor", "Monitor"))
        self.advanced = QtWidgets.QDialog(self)
        self.advanced.setWindowTitle("Advanced — Stitch 1D")
        advanced_layout = QtWidgets.QVBoxLayout(self.advanced)
        advanced_card = SubsectionCard("Source and geometry")
        advanced_layout.addWidget(advanced_card)
        fields = (
            ("image_stem", "Image stem", ""),
            ("detector_rows", "Rows", "195"), ("detector_cols", "Columns", "1475"),
            ("raw_dtype", "Data type", "int32"), ("raw_header_skip", "Header bytes", "0"),
            ("source_motors", "Angle mapping", "del_angle=del, nu_angle=nu"),
            ("poni_references", "PONI reference", "del=0, nu=0"),
            ("image_rotation", "Rotation (°)", "0"),
            ("frame_start", "First frame", "0"), ("frame_stop", "Last frame", ""),
            ("frame_step", "Frame step", "1"), ("max_frame_mib", "Max frame MiB", "256"),
        )
        self._field(advanced_card, SectionId.EXPERIMENT, "geometry_kind", "Geometry",
                    "Goniometer JSON", choices=("Goniometer JSON", "PONI + offsets"))
        for key, label, value in fields:
            self._field(advanced_card, SectionId.SOURCE, key, label, value)
        self._field(advanced_card, SectionId.SOURCE, "output_family", "Result names")
        self._rows["output_family"].setToolTip(
            "Blank names are automatic: <file>-<scan> for one scan, and the scans listed for a "
            "combined result ([2-5,8] gives <file>-2-5+8). To override, enter one name, or names in "
            "output order separated by ; (a blank entry stays automatic). At most 80 characters; "
            "processing and frame selection keep the same name.")
        for key in ("geometry_path", "geometry_kind"):
            self._rows[key].valueChanged.connect(self._sync_goniometer_mapping)
        frames_card = SubsectionCard("Frames and sample")
        advanced_layout.addWidget(frames_card)
        self._field(frames_card, SectionId.PROCESSING, "frame_check", "Frame check",
                    "Exclude empty frames", choices=tuple(_FRAME_CHECKS))
        self._rows["frame_check"].setToolTip(
            "Compares each frame's counts with its neighbours on the same outer-circle row: "
            "below 1 % is an empty (dropped) frame, below 20 % a partial one.")
        self._field(frames_card, SectionId.PROCESSING, "sample_position", "Sample",
                    "As calibrated", choices=tuple(_SAMPLE_POSITIONS))
        self._rows["sample_position"].setToolTip(
            "On rotation axis drops a stacked-arm calibration's fitted sample offset, "
            "for a sample aligned onto the axis (for example after GI alignment).")
        self.corrections_card = SubsectionCard("Corrections")
        advanced_layout.addWidget(self.corrections_card)
        card = self.corrections_card
        self._field(card, SectionId.PROCESSING, "solid_angle", "Solid angle", "On", choices=("On", "Off"))
        self._rows["solid_angle"].setToolTip(
            "Keep on for a stacked-arm calibration with a sample offset: its frames sit at "
            "different distances.")
        self._field(card, SectionId.PROCESSING, "polarization", "Polarization", "0.99")
        self._rows["polarization"].setToolTip("Polarization factor from -1 to 1; leave blank for none.")
        self._field(card, SectionId.PROCESSING, "error_model", "Error model", "None",
                    choices=("None", "Poisson"))
        self._field(card, SectionId.PROCESSING, "dark_path", "Dark", browse=True)
        self._field(card, SectionId.PROCESSING, "flat_path", "Flat field", browse=True)
        self._field(card, SectionId.PROCESSING, "flat_cutoff", "Flat cutoff (% of median)", "10")
        self._rows["flat_cutoff"].setToolTip(
            "Flat pixels below this share of the flat's median are treated as dead and masked. "
            "Raise it to mask a region shadowed in the flat; lower it to keep weak but "
            "working pixels. 0 masks only zero, negative and non-finite pixels.")
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.advanced.close)
        advanced_layout.addWidget(close)
        actions = QtWidgets.QWidget()
        actions_layout = QtWidgets.QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_button = QtWidgets.QPushButton("Preview")
        self.preview_button.clicked.connect(self.begin_preview)
        self.retry_button = QtWidgets.QPushButton("Retry")
        self.retry_button.clicked.connect(self._retry)
        self.advanced_button = QtWidgets.QPushButton("Advanced")
        self.advanced_button.clicked.connect(self.advanced.show)
        for button in (self.preview_button, self.retry_button, self.advanced_button):
            actions_layout.addWidget(button)
        processing.add_row(actions)
        self.preview_dialog = QtWidgets.QDialog(self)
        self.preview_dialog.setWindowTitle("Stitch preview")
        preview_layout = QtWidgets.QVBoxLayout(self.preview_dialog)
        self.preview_text = QtWidgets.QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        preview_layout.addWidget(self.preview_text)
        self.preview_dialog.resize(680, 480)

    def _range(self, card, label, keys, values, *, display_decimals=None):
        specs = []
        for key, value in zip(keys, values):
            field = ControlFormField(SectionId.PROCESSING, label, ("Stitch", key), value,
                ControlFieldKind.BOOL if isinstance(value, bool) else ControlFieldKind.LINE)
            self._fields[key] = field
            self._range_keys.add(key)
            specs.append(dict(path=field.path, value=value))
        row = RangeRow(label=label, low=specs[0], high=specs[1], toggle=specs[2],
                       display_decimals=display_decimals)
        row.valueChanged.connect(self._range_edited)
        row.draftChanged.connect(self._edited)
        self._range_rows.append(row)
        card.add_row(row)
        return row

    def _render_ranges(self):
        values = {field.path: field for key, field in self._fields.items() if key in self._range_keys}
        auto = bool(self._fields["radial_auto"].value)
        for key in ("q_min", "q_max"):
            values[("Stitch", key)] = replace(self._fields[key], enabled=not auto,
                                              reason="Disable Auto to edit this range." if auto else "")
        off = self._threshold_mode == "off"  # Int's rule: the limits follow the Threshold toggle
        for key in ("threshold_min", "threshold"):
            values[("Stitch", key)] = replace(self._fields[key], enabled=not off,
                                              reason="Enable Threshold to edit this limit." if off else "")
        for row in self._range_rows:
            row.apply_fields(values)

    @property
    def _threshold_mode(self):
        state = self._threshold_state
        return "manual" if state.apply_threshold else "auto" if state.mask_saturation else "off"

    @property
    def _threshold_manual(self):
        state = self._threshold_state
        return state.apply_threshold or state.threshold_min is not None or state.threshold_max is not None

    def _range_edited(self, path, value):
        key = path[-1]
        if key in {"threshold_min", "threshold", "threshold_enabled"}:
            result = threshold_transition(self._threshold_state,
                {"threshold_min": "min", "threshold": "max", "threshold_enabled": "enabled"}[key],
                value, self._auto_threshold_max)
            if isinstance(result, ThresholdEditRefusal):
                self._render_ranges()
                self._status(result.reason, ready=False)
                return
            if result is None:
                self._render_ranges()
                return
            self._threshold_state = result
            for name, projected in (
                    ("threshold_min", 0.0 if result.threshold_min is None else result.threshold_min),
                    ("threshold", result.threshold_max if result.threshold_max is not None or result.apply_threshold else self._auto_threshold_max),
                    ("threshold_enabled", result.apply_threshold or result.mask_saturation)):
                self._fields[name] = replace(self._fields[name], value=projected)
            self._render_ranges()
            self._edited()
            return
        old = self._fields[key].value
        if key != "radial_auto":
            try:
                value = None if value is None or not str(value).strip() else float(value)
            except ValueError:
                self._status("Enter a number or clear the threshold maximum for Auto", ready=False)
                return
        if value == old:
            return
        self._fields[key] = replace(self._fields[key], value=value)
        self._render_ranges()
        self._edited()

    def set_sources(self, paths):
        self._auto_threshold_max = None
        if not self._threshold_manual:
            self._fields["threshold"] = replace(self._fields["threshold"], value=None)
            self._render_ranges()
        self._source_paths = tuple(str(path) for path in paths)
        self._paths_by_kind[self._value("source_kind")] = self._source_paths
        shown = self._source_paths[0] if len(self._source_paths) == 1 else f"{len(self._source_paths)} files selected"
        field = replace(self._fields["spec_path"], value=shown)
        self._fields["spec_path"] = field
        self._rows["spec_path"].apply_field(field)
        self._rows["spec_path"].setToolTip("\n".join(self._source_paths))
        self._edited()

    def _source_path_edited(self, _path, value):
        if str(value) and not str(value).endswith("files selected"):
            self._auto_threshold_max = None
            if not self._threshold_manual:
                self._fields["threshold"] = replace(self._fields["threshold"], value=None)
                self._render_ranges()
            self._source_paths = (str(value),)
            self._paths_by_kind[self._value("source_kind")] = self._source_paths

    def _source_changed(self, *_args):
        kind = self._value("source_kind")
        for key in ("image_dir", "scan_mode", "scan"):
            self._rows[key].setVisible(kind != "NeXus" and (key == "image_dir" or kind == "Spec"))
        self._rows["scan"].setVisible(kind == "Spec" and self._value("scan_mode") == "Select Scans")
        self._rows["spec_path"].label.setText({"Spec": "SPEC files", "NeXus": "NeXus files", "Directory": "Folder"}[kind])
        if _args and _args[0][-1] == "source_kind":
            self.set_sources(self._paths_by_kind.get(kind, ()))

    def _processing_axis_changed(self, *_args):
        q = self._value("unit") == "Q (Å⁻¹)"
        self.radial_range.label.setText("Q (Å⁻¹)" if q else "2θ (°)")
        self._fields["q_min"] = replace(self._fields["q_min"], value=0.)
        self._fields["q_max"] = replace(self._fields["q_max"], value=6.2 if q else 90.)
        self._render_ranges()

    def set_field(self, key, value):
        if key in self._range_keys:
            self._range_edited(("Stitch", key), value)
            return
        field = replace(self._fields[key], value=value)
        self._fields[key] = field
        self._rows[key].apply_field(field)
        self._edited()
        if key == "backend":
            self._backend_changed()
        elif key == "spec_path":
            self._source_path_edited(field.path, value)
        elif key in {"source_kind", "scan_mode"}:
            self._source_changed(field.path, value)
        elif key == "unit":
            self._processing_axis_changed()
        elif key == "project_root" and not self._value("output_path"):
            self.set_field("output_path", str(Path(value) / "xdart_processed_data"))
        if key in {"geometry_path", "project_root"}:
            self._refresh_xu_geometry()
        if key in {"geometry_path", "geometry_kind"}:
            self._sync_goniometer_mapping()

    def _value(self, key):
        if key in self._range_keys:
            return self._fields[key].value
        return self._rows[key].current_value()

    def _edited(self, *_args):
        self._revision += 1
        self._prepared_revision = None
        self._queued_jobs.clear()
        if hasattr(self, "run_bar"):
            self._status("Inputs changed · Preview required", ready=False)
            self._sync_actions()
            self._refresh_inputs_view()

    def _automatic_names(self, kind, sources, scans):
        """The result names the selection gets when Result names is blank (Preview confirms)."""
        from xdart.gui.tools.stitch_native_values import automatic_family
        from xrd_tools.sources.grouping import parse_stitch_scan_groups
        project = str(self._value("project_root") or "").strip()
        try:
            if kind == "Spec" and self._value("scan_mode") != "Stitch All":
                groups = [[(path, scan) for scan in group]
                          for path in sources for group in parse_stitch_scan_groups(scans)]
            else:
                groups = [[(path, None) for path in sources]] if kind == "NeXus" and sources else []
            names = [automatic_family(group, project) for group in groups[:3]] if project else []
        except ValueError:
            names = []
        if not names:
            return "automatic"
        more = f"; … {len(groups)} outputs" if len(groups) > 3 else ""
        return "auto · " + "; ".join(names) + more

    def _refresh_inputs_view(self):
        """Int's section headers; the automatic image folder, stem and result names."""
        if "output_family" not in self._rows:  # the Advanced rows are still being built
            return
        from xrd_tools.sources.grouping import parse_stitch_scan_groups
        value = self._value
        kind = value("source_kind")
        sources = tuple(Path(path) for path in self._source_paths)
        spec = sources[0] if sources and kind == "Spec" else None
        scans = str(value("scan") or "")
        first = re.match(r"\s*\[?\s*(\d+)", scans)
        self._rows["image_dir"].editor.setPlaceholderText(
            f"auto · {spec.parent}" if spec else "auto · the SPEC file's folder")
        self._rows["image_stem"].editor.setPlaceholderText(
            f"auto · {spec.stem if spec else '<SPEC name>'}_scan{first[1] if first else '<N>'}_")
        self._rows["output_family"].editor.setPlaceholderText(self._automatic_names(kind, sources, scans))
        project = Path(str(value("project_root") or ""))
        project_ready = bool(str(value("project_root")).strip()) and project.is_dir()
        self.controls.project_card.set_valid_marker(project_ready, "" if project_ready else "Choose an existing Project folder")
        calibration = str(value("geometry_path") or "").strip()
        mask = str(value("mask_path") or "").strip()
        experiment_ready = bool(calibration) and Path(calibration).is_file() and (not mask or Path(mask).is_file())
        xu = value("backend") == "xrayutilities histogram"
        self.controls.experiment_card.set_status_text("xrayutilities" if xu else "multigeometry")
        self.controls.experiment_card.set_valid_marker(
            experiment_ready, "" if experiment_ready else "Choose an existing calibration (and mask)")
        present = bool(sources) and all(path.exists() for path in sources)
        try:
            groups = len(parse_stitch_scan_groups(scans)) if kind == "Spec" and value("scan_mode") != "Stitch All" else None
        except ValueError:
            groups = 0
        if kind == "Spec":
            text = f"{len(sources)} SPEC file{'s' * (len(sources) != 1)} · " + (
                "all scans" if groups is None else f"{groups} output{'s' * (groups != 1)}")
        else:
            text = f"{len(sources)} {kind} source{'s' * (len(sources) != 1)}"
        source_ready = present and groups != 0
        self.controls.source_card.set_status_text(text if sources else "")
        self.controls.source_card.set_valid_marker(source_ready, "" if source_ready else "Choose sources and scans")
        try:
            processing_ready = int(str(value("npt_1d"))) > 0
        except ValueError:
            processing_ready = False
        self.controls.processing_card.set_status_text("stitch 1d")
        self.controls.processing_card.set_valid_marker(processing_ready, "" if processing_ready else "Enter the number of points")

    def _backend_changed(self, *_args):
        xu = self._value("backend") == "xrayutilities histogram"
        self._sync_advanced()
        self.set_field("source_motors", "del=del, nu=nu" if xu else "del_angle=del, nu_angle=nu")
        if not xu and not str(self._value("image_rotation")).strip():
            self.set_field("image_rotation", "0")
        self.set_field("q_max", "5.2" if xu else "6.2")
        self._refresh_xu_geometry()
        self._sync_goniometer_mapping()
        self._sync_actions()

    def _sync_advanced(self):
        """XU always applies its solid angle and has no uncertainty or sample-position
        model; polarization, dark, flat and the frame check apply to both backends."""
        xu = self._value("backend") == "xrayutilities histogram"
        for key, enabled in (("frame_check", True), ("sample_position", not xu),
                             ("solid_angle", not xu), ("error_model", not xu)):
            self._fields[key] = replace(self._fields[key], enabled=enabled)
            self._rows[key].apply_field(self._fields[key])

    def _sync_goniometer_mapping(self, *_args):
        """Bind every position the goniometer JSON declares (Angle mapping, Advanced).

        A position's motor defaults to its name without its suffix; a mapping that already
        binds exactly these positions is the user's and stays. XU owns its own mapping.
        """
        if (not hasattr(self, "advanced") or self._value("backend") == "xrayutilities histogram"
                or self._value("geometry_kind") != "Goniometer JSON"):
            return
        positions = _goniometer_positions(self._value("geometry_path"))
        if not positions:
            return
        try:
            current = dict(_parse_pairs(self._value("source_motors") or ""))
        except ValueError:
            current = {}
        if set(current) == set(positions):
            return
        self.set_field("source_motors", ", ".join(
            f"{name}={current.get(name) or _POSITION_SUFFIX.sub('', name) or name}"
            for name in positions))

    def _refresh_xu_geometry(self, *_args, calibration=None):
        """Present asset-owned settings through the existing strict admission owner."""
        if self._value("backend") != "xrayutilities histogram":
            return
        from xrd_tools.analysis.xu_stitch_calibration import (
            XuStitchCalibrationInput, capture_xu_stitch_calibration,
        )
        try:
            if calibration is None:
                calibration = capture_xu_stitch_calibration(
                    XuStitchCalibrationInput(self._value("geometry_path")),
                    project_root=Path(self._value("project_root")))
            acquisition = calibration.projection.value["acquisition"]
        except (ValueError, TypeError, OSError):
            values = {"source_motors": "", "image_rotation": ""}
            tip = "Select a valid XU calibration to inspect its angle mapping and rotation."
        else:
            values = dict(source_motors=", ".join(
                f"{name}={source}" for name, source in acquisition["source_motors"].items()),
                image_rotation=90 * acquisition["image_rotation_quarter_turns"])
            tip = "Defined by the selected XU calibration; change it in Calibrate."
        values.update(geometry_kind="Goniometer JSON", poni_references="")
        for key, value in values.items():
            field = replace(self._fields[key], value=value)
            self._fields[key] = field
            self._rows[key].apply_field(field)
            self._rows[key].setToolTip(tip)

    def _browse(self, key):
        """Browse as Int does: the last folder picked, the same file types for the same inputs."""
        from xdart.gui.widgets import file_dialogs as dialogs
        kind = self._value("source_kind")
        folders = {"project_root": "Choose folder", "output_path": "Choose processed-data folder",
                   "image_dir": "Choose images folder"}
        files = {"geometry_path": ("Choose calibration", dialogs.CALIBRATION_FILTER),
                 "mask_path": ("Choose detector mask", dialogs.MASK_FILTER),
                 "dark_path": ("Choose dark", dialogs.CORRECTION_FILTER),
                 "flat_path": ("Choose flat field", dialogs.CORRECTION_FILTER)}
        current = self._source_paths[0] if key == "spec_path" and self._source_paths else self._value(key)
        common = dict(current=str(current or ""), fallback=str(self._value("project_root") or ""))
        if key in folders or key == "spec_path" and kind == "Directory":
            picks = dialogs.browse(self, folders.get(key, "Choose folder"), folder=True, **common)
        elif key == "spec_path":
            picks = dialogs.browse(self, "Choose source files", many=True, **common,
                                   file_filter=dialogs.NEXUS_FILTER if kind == "NeXus" else dialogs.ANY_FILTER)
            if picks:
                self.set_sources(picks)
            return
        else:
            title, file_filter = files.get(key, ("Choose " + self._fields[key].label, dialogs.ANY_FILTER))
            picks = dialogs.browse(self, title, file_filter=file_filter, **common)
        if picks:
            self.set_field(key, picks[0])

    def build_form(self):
        value = self._value
        kind = (StitchGeometryKind.PYFAI_GONIOMETER_JSON if value("geometry_kind") == "Goniometer JSON"
                else StitchGeometryKind.PONI)
        monitor = str(value("monitor")).strip()
        selection = StitchSourceSelection(value("source_kind"), self._source_paths,
            scans=str(value("scan")), stitch_all=value("scan_mode") == "Stitch All",
            image_dir=str(value("image_dir")), image_stem=str(value("image_stem")))
        conditioning = StitchConditioning(self._threshold_mode,
            value("threshold_min") if self._threshold_mode == "manual" else None,
            value("threshold") if self._threshold_mode == "manual" else None)
        native = dict(source_selection=selection,
            unit="q_A^-1" if value("unit") == "Q (Å⁻¹)" else "2th_deg",
            conditioning=conditioning, mask_path=str(value("mask_path")) or None,
            output_family=str(value("output_family")).strip() or None,
            frame_screen=_FRAME_CHECKS[value("frame_check")])
        polarization = str(value("polarization")).strip()
        try:
            factor = float(polarization) if polarization else None
        except ValueError as error:
            raise ValueError("Polarization must be a number from -1 to 1, or blank") from error
        cutoff = str(value("flat_cutoff")).strip()
        try:
            fraction = float(cutoff) / 100.0 if cutoff else 0.1
        except ValueError as error:
            raise ValueError("Flat cutoff must be a percentage from 0 to below 100") from error
        native.update(polarization_factor=factor,
                      dark_path=str(value("dark_path")).strip() or None,
                      flat_path=str(value("flat_path")).strip() or None,
                      flat_min_fraction=fraction)
        if value("backend") != "xrayutilities histogram":
            native.update(
                sample_position=_SAMPLE_POSITIONS[value("sample_position")],
                correct_solid_angle=value("solid_angle") == "On",
                error_model="poisson" if value("error_model") == "Poisson" else None)
        if ";" in str(value("output_family")):
            native.update(output_family=None,
                output_families=tuple(part.strip() for part in str(value("output_family")).split(";")))
        output = str(value("output_path"))
        if not output:
            output = str(Path(value("project_root")) / "xdart_processed_data")
        directory = Path(output)
        if not directory.is_absolute():
            directory = Path(value("project_root")) / directory
        output = str(directory / "stitch_stitch1d.nexus")
        return StitchToolForm(
            project_root=value("project_root"),
            spec_path=self._source_paths[0], scan=value("scan"),
            image_dir=value("image_dir") or value("project_root"), image_stem=value("image_stem"),
            detector_shape=(int(value("detector_rows")), int(value("detector_cols"))),
            raw_dtype=value("raw_dtype"), raw_header_skip=int(value("raw_header_skip")),
            frame_selector=StitchFrameSelector(int(value("frame_start")),
                None if not str(value("frame_stop")).strip() else int(value("frame_stop")), int(value("frame_step"))),
            geometry_path=value("geometry_path"), geometry_kind=kind,
            source_motors=_parse_pairs(value("source_motors")),
            poni_references=(_parse_references(value("poni_references"))
                             if kind is StitchGeometryKind.PONI else ()),
            image_rotation=int(value("image_rotation")),
            threshold=None if value("threshold") is None or not str(value("threshold")).strip() else float(value("threshold")),
            q_range=None if value("radial_auto") else (float(value("q_min")), float(value("q_max"))), npt_1d=int(value("npt_1d")),
            output_path=output, monitor_selector=MetadataColumnSelector(monitor) if monitor else None,
            backend="xu_hist" if value("backend") == "xrayutilities histogram" else "multigeometry",
            max_frame_bytes=int(value("max_frame_mib")) * 1024 * 1024,
            **native,
        )

    def load_form(self, form):
        """Restore a prepared form, including native source and conditioning intent."""
        if form.source_selection is None:
            raise ValueError("the Stitch page takes a native source selection")
        fields = {key: getattr(form, key) for key in (
            "project_root", "spec_path", "scan", "image_dir", "image_stem", "raw_dtype",
            "raw_header_skip", "geometry_path", "threshold", "npt_1d", "output_path")}
        fields["threshold"] = "" if form.threshold is None else form.threshold
        fields.update(unit="Q (Å⁻¹)" if form.unit == "q_A^-1" else "2θ (°)",
            output_family=";".join(form.output_families) if form.output_families else form.output_family or "",
            mask_path=form.mask_path or "",
            geometry_kind="PONI + offsets" if form.geometry_kind is StitchGeometryKind.PONI else "Goniometer JSON",
            poni_references=", ".join(f"{key}={value}" for key, value in form.poni_references),
            backend="xrayutilities histogram" if form.backend == "xu_hist" else "pyFAI MultiGeometry",
            detector_rows=form.detector_shape[0], detector_cols=form.detector_shape[1],
            source_motors=", ".join(f"{key}={value}" for key, value in form.source_motors),
            image_rotation=form.image_rotation, frame_start=form.frame_selector.start_label,
            frame_stop="" if form.frame_selector.stop_label is None else form.frame_selector.stop_label,
            frame_step=form.frame_selector.step, radial_auto=form.q_range is None,
            frame_check=_choice(_FRAME_CHECKS, form.frame_screen),
            sample_position=_choice(_SAMPLE_POSITIONS, form.sample_position),
            solid_angle="On" if form.correct_solid_angle else "Off",
            polarization="" if form.polarization_factor is None else form.polarization_factor,
            error_model="Poisson" if form.error_model == "poisson" else "None",
            dark_path=form.dark_path or "", flat_path=form.flat_path or "",
            flat_cutoff=f"{100 * form.flat_min_fraction:g}")
        if form.q_range is not None:
            fields.update(q_min=form.q_range[0], q_max=form.q_range[1])
        for key, value in fields.items():
            self.set_field(key, value)
        selection = form.source_selection
        self.set_field("source_kind", selection.kind)
        self.set_sources(selection.paths)
        self.set_field("scan_mode", "Stitch All" if selection.stitch_all else "Select Scans")
        self.set_field("scan", selection.scans)
        self.set_field("output_path", str(Path(form.output_path).parent))
        self.set_field("image_dir", selection.image_dir)
        self.set_field("image_stem", selection.image_stem)
        policy = form.conditioning or StitchConditioning()
        self._threshold_state = ThresholdState(policy.mode == "manual",
            policy.threshold_min, policy.threshold_max, policy.mode == "auto")
        for key, value in (("threshold_min", policy.threshold_min if policy.mode == "manual" else 0.),
                           ("threshold", policy.threshold_max), ("threshold_enabled", policy.mode != "off")):
            self._fields[key] = replace(self._fields[key], value=value)
        self._render_ranges()
        self._sync_advanced()
        self._refresh_xu_geometry()

    def _start(self, action, begin):
        if self._closing:
            return False
        if self._authoring_active():
            self._status("Finish the calibration or mask editor before starting another operation", ready=False)
            return False
        try:
            identity = begin()
        except (TypeError, ValueError, OSError) as error:
            self._queued_jobs.clear()
            self._status(str(error), ready=False)
            return False
        if identity is None:
            self._status("Finish the current operation or refresh Preview", ready=False)
            return False
        self._action_revision = self._revision
        self._action_view_revision = self._view_revision
        self._started = time.monotonic()
        self._status(action)
        self._timer.start()
        self._sync_actions()
        return True

    def begin_preview(self):
        def begin():
            self._owner.set_form(self.build_form())
            self._prepared_revision = None
            return self._owner.begin_preflight()
        self._start("Preparing Preview…", begin)

    def begin_run(self):
        if self._prepared_revision != self._revision or self._owner.prepared is None:
            self._status("Preview the current selection before Run", ready=False)
            return
        count = len(self._owner.prepared.requests)
        self._queued_jobs = list(range(count))
        self._job_total = count
        self._batch_started = time.monotonic()
        self._run_next_job()

    def _run_next_job(self):
        if not self._queued_jobs or self._closing:
            return
        index = self._queued_jobs.pop(0)
        if not self._start(f"Running · Stitch 1D · {index + 1}/{self._job_total}",
                           lambda: self._owner.begin_run(job_index=index)):
            self._queued_jobs.clear()

    def open_result(self, path=None):
        if path is None or isinstance(path, bool):
            from xdart.gui.widgets.file_dialogs import browse
            picks = browse(self, "Open stitched result", file_filter="NeXus (*.nexus)",
                           fallback=str(self._value("project_root") or ""))
            path = picks[0] if picks else ""
        if not path:
            return
        self._view_revision += 1
        self._start("Opening saved result…", lambda: self._owner.begin_reopen(path))

    def stop(self):
        self._queued_jobs.clear()
        if self._owner.cancel():
            self._status("Stopping…")
        return ActionAccepted("stitch-stop")

    def _retry(self):
        begin = (self._owner.begin_retry_cleanup if self._owner.finalization is StitchOwnerFinalization.CLEANUP_PENDING
                 else self._owner.begin_retry_verification)
        self._start("Retrying finalization…", begin)

    def _status(self, text, *, ready=True):
        self.run_bar.set_readiness_summary(str(text), ready=ready)

    def _poll(self):
        update = self._owner.poll()
        if update is not None:
            if update.progress is not None:
                p = update.progress
                self._status(f"{p.stage} · {p.completed}/{p.total}")
            elif update.terminal_status is OperationTerminalStatus.FAILED:
                self._queued_jobs.clear()
                self._status(update.failure_message or update.failure_type, ready=False)
                if update.action is StitchOwnerAction.METADATA and self._metadata_current():
                    self._metadata_dialog.status.setText(update.failure_message or update.failure_type)
                if update.action is StitchOwnerAction.FRAME_READ and self._raw_popup is not None and self._raw_request_revision == self._raw_revision:
                    self._raw_popup.caption.setText("Image unavailable · " + (update.failure_message or update.failure_type))
            elif update.terminal_status is OperationTerminalStatus.CANCELLED:
                self._queued_jobs.clear()
                self._status("Cancelled")
            elif update.outcome is not None:
                self._accept_outcome(update)
        if not self._owner.busy:
            self._timer.stop()
            if self._raw_pending and not self._queued_jobs:
                QtCore.QTimer.singleShot(0, self._request_raw)
        self._sync_actions()

    def _accept_outcome(self, update):
        outcome = update.outcome
        if outcome.kind is StitchOwnerOutcomeKind.METADATA:
            if self._metadata_current():
                self._metadata_dialog.adopt_sources(outcome.metadata)
            self._status("Ready · saved Stitch 1D result")
        elif outcome.kind is StitchOwnerOutcomeKind.PREFLIGHT_READY:
            if update.stale or self._action_revision != self._revision:
                self._status("Inputs changed · Preview required", ready=False)
                return
            self._prepared_revision = self._revision
            preflight = outcome.preflight
            if preflight.requests and preflight.form.backend == "xu_hist":
                self._refresh_xu_geometry(calibration=preflight.requests[0].plan.calibration)
            bounds = []
            for members in preflight.preview_members:
                for member in members:
                    dtype = np.dtype(member["source_snapshot"]["raw_dtype"])
                    bounds.append(FilesystemSourceAdapter._threshold_max_for_dtype(dtype))
            self._auto_threshold_max = bounds[0] if bounds and all(value == bounds[0] for value in bounds) else None
            if not self._threshold_manual:
                self._fields["threshold"] = replace(self._fields["threshold"], value=self._auto_threshold_max)
                self._render_ranges()
            lines = [f"{len(preflight.requests)} stitched outputs"]
            if preflight.warnings:
                lines.extend(("\nWarnings", *preflight.warnings))
            for index, request in enumerate(preflight.requests, 1):
                count = len(request.module.source.selected_labels)
                lines.append(f"\n{index}. {Path(request.module.output.target).name} · {count} frames")
                for member in preflight.preview_members[index - 1]:
                    identity = member["identity"]
                    lines.append(f"  {identity['scan_id']}-{identity['frame_index']} · {member['source_path']}")
            if preflight.skipped:
                lines.append("\nSkipped scans")
                lines.extend(f"{Path(path).name} · scan {scan}: {reason}" for path, scan, reason in preflight.skipped)
            self.preview_text.setPlainText("\n".join(lines))
            self.preview_dialog.show()
            self._status(f"Ready · Stitch 1D · {len(preflight.requests)} outputs", ready=bool(preflight.requests))
        elif outcome.kind is StitchOwnerOutcomeKind.FRAME_READ:
            if (self._raw_request_revision == self._raw_revision and not self._closing
                    and self._raw_popup is not None and self._raw_popup.isVisible()
                    and self._current_payload() is not None
                    and outcome.artifact_path == self._current_payload().inspection.path):
                self._raw_frame_view = outcome.frame_view
                self._render_raw()
                self._status("Ready · saved Stitch 1D result")
        elif outcome.kind is StitchOwnerOutcomeKind.REOPENED:
            if self._action_view_revision == self._view_revision and not self._closing:
                self._mount_payload(outcome.artifact_path, outcome.payload)
                note = saved_frame_screen_note(outcome.payload.contributing_frames)
                self._status("Ready · saved Stitch 1D result" + (f" · {note}" if note else ""))
        elif outcome.kind is StitchOwnerOutcomeKind.RESULT:
            result = outcome.result
            if result.terminal.disposition is ModuleDisposition.COMMITTED:
                if not update.stale and self._action_revision == self._revision and not self._closing:
                    self._mount_payload(result.request.module.output.target, result.payload)
                elapsed = time.monotonic() - (self._batch_started or self._started)
                count = len(result.request.module.source.selected_labels)
                note = frame_screen_note(result.frame_screen)
                self._status(f"Complete · {count} frames · {elapsed:.2f} s" + (f" · {note}" if note else ""))
                if self._queued_jobs and not update.stale and self._action_revision == self._revision:
                    self._run_next_job()
            else:
                self._queued_jobs.clear()
                self._status(f"{result.terminal.disposition.value}: {result.terminal.diagnostic or result.terminal.code}", ready=False)
        else:
            self._queued_jobs.clear()
            self._status(outcome.finalization_message or "Output retained · finalization retry required", ready=False)

    def _current_payload(self):
        item = self.results.currentItem()
        return None if item is None else self._payloads.get(item.data(QtCore.Qt.ItemDataRole.UserRole))

    def _current_result_changed(self, *_args):
        if self._metadata_dialog is not None:
            self._metadata_dialog.close()
        self._view_revision += 1
        self._raw_revision += 1
        self._raw_frame_view = None
        self.frames.clear()
        payload = self._current_payload()
        if payload is not None:
            records = payload.contributing_frames
            manifest = json.loads(payload.provenance_json).get("source", {}).get("input_manifest", {})
            sources = {entry["source"]["module_source_fingerprint"]: entry["source"]["relative_path"]
                for entry in manifest.get("members", (manifest,)) if "source" in entry}
            members = tuple(dict.fromkeys((row["identity"]["source_id"], row["identity"]["scan_id"]) for row in records))
            for record in records:
                identity = record["identity"]
                frame = identity["frame_index"]
                storage_scan = record["key"].split("/")[0][5:] if "/" in record["key"] else None
                scan = identity["scan_id"]
                # SPEC uses occurrence-qualified identities; the first occurrence
                # displays the familiar scan number, while the lookup stays exact.
                if scan.endswith(".1") and scan[:-2].isdecimal():
                    scan = str(int(scan[:-2]))
                if sum(member[1] == identity["scan_id"] for member in members) > 1:
                    path = Path(sources.get(identity["source_id"], record["source_path"]))
                    scan = f"{path.as_posix()} · {scan}"
                label = str(frame) if len(members) == 1 else f"{scan}-{frame}"
                screen = record.get("screen")
                if record.get("excluded"):
                    label += " · excluded"
                elif screen == "partial":
                    label += " · partial"
                item = QtWidgets.QListWidgetItem(label)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, (storage_scan, frame))
                if record.get("excluded"):
                    item.setForeground(self.frames.palette().color(
                        QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text))
                check = ("not recorded" if screen is None
                         else f"{screen}, excluded from the result" if record["excluded"] else screen)
                item.setToolTip(f"{record['source_path']}\nscan {identity['scan_id']} · frame {frame}"
                                f"\nFrame check: {check}")
                self.frames.addItem(item)
        if self.frames.count():
            self.frames.setCurrentRow(0)
        elif self._raw_popup is not None:
            self._raw_popup.image.clear_image_buffers()
        self._draw()
        if hasattr(self, "run_bar"):
            self._sync_actions()

    def _frame_changed(self, *_args):
        self._raw_revision += 1
        self._raw_frame_view = None
        if self._raw_popup is not None and self._raw_popup.isVisible():
            self._raw_popup.image.clear_image_buffers()
            self._raw_popup.caption.setText("Loading selected frame…")
            self._raw_pending = True
            QtCore.QTimer.singleShot(0, self._request_raw)

    def _show_raw(self):
        from .raw_popup import StitchRawPopup
        if self._raw_popup is None:
            self._raw_popup = StitchRawPopup(self)
            self._raw_popup.modeChanged.connect(self._frame_changed)
            self._raw_popup.closed.connect(self._raw_closed)
        self._raw_popup.show()
        self._raw_popup.raise_()
        self._frame_changed()

    def _request_raw(self):
        if self._closing or self._raw_popup is None or not self._raw_popup.isVisible():
            self._raw_pending = False
            return
        if self._owner.busy or self._queued_jobs:
            return
        item, payload = self.frames.currentItem(), self._current_payload()
        if item is None or payload is None:
            self._raw_pending = False
            return
        scan, frame = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self._raw_request_revision = self._raw_revision
        self._raw_pending = False
        self._start("Loading contributing frame…", lambda: self._owner.begin_frame_read(
            payload, frame, scan=scan, source_root=str(self._value("project_root")) or None,
            full_raw=self._raw_popup.mode.currentText() == "Full Raw"))

    def _render_raw(self):
        item = self.frames.currentItem()
        if self._raw_popup is not None and self._raw_popup.isVisible() and self._raw_frame_view is not None and item is not None:
            self._raw_popup.render(self._raw_frame_view, label=item.text(),
                color_map=self._color_map, log=self._log_scale)

    def _raw_closed(self):
        self._raw_revision += 1
        self._raw_pending = False
        self._raw_frame_view = None

    def _source_spec_for_tools(self):
        from xrd_tools.core.scan import SourceSpec, SourceKind
        from xrd_tools.sources.grouping import parse_stitch_scan_groups
        if not self._source_paths or self._value("source_kind") == "Directory":
            return None
        path = self._source_paths[0]
        if self._value("source_kind") == "NeXus":
            return SourceSpec(path, SourceKind.NEXUS_STACK)
        scan = parse_stitch_scan_groups(str(self._value("scan") or "1"))[0][0]
        stem = str(self._value("image_stem")).format(spec=Path(path).stem, scan=scan.split(".")[0], occurrence=scan)
        return SourceSpec(path, SourceKind.SPEC, options={
            "scan": scan, "image_dir": str(self._value("image_dir")),
            "image_stem": stem or None,
            "read_image_kwargs": dict(detector_shape=(int(self._value("detector_rows")), int(self._value("detector_cols"))),
                raw_dtype=str(self._value("raw_dtype")), raw_header_skip=int(self._value("raw_header_skip")))})

    def _calibrate(self):
        from .calibration import StitchCalibrationDialog
        if self._owner.busy or self._closing or self._authoring_active():
            return
        try:
            project = str(self._value("project_root"))
            backend = "xu_hist" if self._value("backend") == "xrayutilities histogram" else "multigeometry"
            if not project:
                raise ValueError("Choose a Project folder first")
            if self._calibration_dialog is not None and (self._calibration_project, self._calibration_backend) != (project, backend):
                self._calibration_settings = self._calibration_dialog.settings()
                self._calibration_dialog.hide()
                self._calibration_dialog = None
            # The processing scan only fills a new dialog with no calibration source: the
            # calibrant scan is usually another scan, and saved or edited inputs are kept.
            unconfigured = False
            if self._calibration_dialog is None:
                self._calibration_dialog = StitchCalibrationDialog(str(self._value("project_root")),
                    backend=backend, parent=self)
                saved = self._calibration_settings or {}
                if saved:
                    self._calibration_dialog.apply_settings(saved)
                unconfigured = not (saved.get("fields") or {}).get("source")
                self._calibration_project, self._calibration_backend = project, backend
                self._calibration_dialog.calibrationSaved.connect(self._adopt_calibration)
                self._authoring_dialogs.append(self._calibration_dialog)
            source = self._source_spec_for_tools() if unconfigured else None
            if source is not None:
                motors = dict(_parse_pairs(self._value("source_motors") or "del=del, nu=nu"))
                self._calibration_dialog.set_source_context(source, image_rotation=int(self._value("image_rotation") or 0),
                    del_motor=motors.get("del_angle", motors.get("del", "del")),
                    nu_motor=motors.get("nu_angle", motors.get("nu", "nu")),
                    geometry_path=(self._value("geometry_path") or None
                                   if self._value("geometry_kind") == "Goniometer JSON" else None))
            self._calibration_dialog.show()
            self._calibration_dialog.raise_()
            self._authoring_timer.start()
        except (TypeError, ValueError, OSError) as error:
            self._status(str(error), ready=False)

    def _adopt_calibration(self, path, backend):
        if self._closing or self._calibration_project != str(self._value("project_root")):
            self._status(f"Calibration saved: {path} · Project changed; select the calibration to use it")
            return
        self.set_field("backend", "xrayutilities histogram" if backend == "xu_hist" else "pyFAI MultiGeometry")
        self.set_field("geometry_kind", "Goniometer JSON")
        self.set_field("geometry_path", path)
        dialog = self._calibration_dialog
        if dialog is not None:
            motors = dialog.saved_source_motors
            if motors:
                self.set_field("source_motors", ", ".join(f"{name}={source}" for name, source in motors.items()))
            self.set_field("image_rotation", dialog.saved_image_rotation)

    def _make_mask(self):
        from .mask import StitchMaskDialog
        if self._owner.busy or self._closing or self._authoring_active():
            return
        try:
            source = self._source_spec_for_tools()
            dialog = StitchMaskDialog(str(self._value("project_root")), source=source,
                rotation=int(self._value("image_rotation")), reader_options=dict(
                    detector_shape=(int(self._value("detector_rows")), int(self._value("detector_cols"))),
                    raw_dtype=str(self._value("raw_dtype")), raw_header_skip=int(self._value("raw_header_skip")),
                    rotation=0), parent=self)
            self._authoring_dialogs.append(dialog)
            revision = self._revision
            dialog.maskSaved.connect(lambda path: self.set_field("mask_path", path)
                if revision == self._revision and not self._closing
                else self._status(f"Mask saved: {path} · Inputs changed; select the mask to use it"))
            dialog.show()
            self._authoring_timer.start()
        except (TypeError, ValueError, OSError) as error:
            self._status(str(error), ready=False)

    def _mount_payload(self, path, payload):
        if type(payload) is not AnalysisArtifactPayload or payload.inspection.kind is not AnalysisArtifactKind.STITCH_1D:
            raise TypeError("Stitch page requires an admitted 1D artifact")
        key = str(path)
        self._payloads[key] = payload
        item = next((self.results.item(i) for i in range(self.results.count())
                     if self.results.item(i).data(QtCore.Qt.ItemDataRole.UserRole) == key), None)
        if item is None:
            item = QtWidgets.QListWidgetItem(Path(key).name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            item.setToolTip(key)
            self.results.addItem(item)
            self._result_key(key)
        self.results.setCurrentItem(item)
        self._current_result_changed()

    def _selected(self):
        items = self.results.selectedItems()
        if self._plot_mode == "Single":
            item = self.results.currentItem()
            items = [] if item is None else [item]
        return [(item, self._payloads[item.data(QtCore.Qt.ItemDataRole.UserRole)]) for item in items]

    def _result_key(self, path):
        """The result's display key (one object per result; results share one run and one
        navigation artifact, so the footer steps through them)."""
        from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
        from xdart.gui.tabs.scattering.events import RunIdentity
        key = self._result_keys.get(path)
        if key is None:
            number = len(self._result_keys) + 1
            key = DisplayFrameKey(RunIdentity(1, "stitch-page"), Path(path).name,
                                  "stitch-results", number, number)
            self._result_keys[path] = key
        return key

    def _draw(self, *_args):
        """Project the selected results into Int's scientific view."""
        if not hasattr(self, "scientific"):
            return
        from xdart.gui.tabs.scattering.shell_values import (
            AxisProjection, FrameNavigationProjection, ScientificProjection, TraceProjection,
        )
        role = QtCore.Qt.ItemDataRole.UserRole
        items = [self.results.item(row) for row in range(self.results.count())]
        keys = tuple(self._result_key(item.data(role)) for item in items)
        selected = self._selected()
        available = all(payload.wavelength_A is not None for _, payload in selected)
        if not available and self._plot_axis == "2theta":
            self._plot_axis = "Q"
        unit = "q_A^-1" if self._plot_axis == "Q" else "2th_deg"
        axis_label, axis_unit = ("Q", "Å⁻¹") if unit == "q_A^-1" else ("2θ", "°")
        traces = []
        for item, payload in selected:
            values = {"Intensity": payload.intensity, "Coverage": payload.coverage,
                      "Normalization": payload.normalization}[self._diagnostic]
            if values is None:
                continue
            try:
                radial = payload.radial_axis(unit)
            except ValueError:
                if payload.native_unit != unit:
                    continue
                radial = payload.axis("q")
            traces.append(TraceProjection(
                self._result_key(item.data(role)),
                AxisProjection(np.asarray(radial, dtype=float), axis_label, axis_unit),
                np.asarray(values, dtype=float), title=item.text()))
        current = self.results.currentItem()
        state = ScientificProjection(
            heavy_available=frozenset(keys), traces=tuple(traces),
            title=current.text() if current else "Current", processing_mode="Int 1D",
            plot_axis=self._plot_axis, plot_mode=self._plot_mode, plot_options=self._plot_options,
            color_map=self._color_map, log_scale=self._log_scale, background_enabled=False,
            status=current.text() if current else "")
        navigation = FrameNavigationProjection(
            frames=keys, current=None if current is None else self._result_key(current.data(role)),
            selected=tuple(self._result_key(item.data(role)) for item, _ in selected))
        self.scientific.reconcile(state, navigation, completed=self.results.currentRow() + 1,
                                  total=len(keys), detail="")
        self.curve.setLabel("left", scaled_intensity_label(self._diagnostic,
                                                           self._plot_options.intensity_scale))
        blocker = QtCore.QSignalBlocker(self._diagnostic_combo)
        self._diagnostic_combo.setCurrentText(self._diagnostic)
        del blocker

    def _view_command(self, command):
        """The view's typed commands, for the Stitch results it shows."""
        from xdart.gui.tabs.scattering.scientific_view import _PLOT_OPTION_FIELDS
        from xdart.gui.tabs.scattering.scientific_plot_options import PLOT_OPTION_COMMAND_PATHS
        from xdart.gui.tabs.scattering.shell_values import ShellCommandKind
        kind, value = command.kind, command.value
        if kind in {ShellCommandKind.SELECT_FRAME, ShellCommandKind.HYDRATE_FRAME}:
            path = next((path for path, key in self._result_keys.items() if key is command.frame), None)
            role = QtCore.Qt.ItemDataRole.UserRole
            row = next((row for row in range(self.results.count())
                        if self.results.item(row).data(role) == path), -1)
            if row >= 0 and row != self.results.currentRow():
                flag = (QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
                        if self._plot_mode == "Single" else QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
                self.results.setCurrentRow(row, flag)
            return
        if kind is ShellCommandKind.SET_PLOT_MODE:
            self._plot_mode = str(value)
            if self._plot_mode == "Single":
                self.results.setCurrentRow(self.results.currentRow(),
                                           QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect)
        elif kind is ShellCommandKind.SET_PLOT_AXIS and value in {"Q", "2theta"}:
            self._plot_axis = value
        elif kind is ShellCommandKind.SET_COLOR_MAP:
            self._color_map = str(value)
            self._render_raw()
        elif kind is ShellCommandKind.SET_LOG_SCALE:
            self._log_scale = bool(value)
            self._render_raw()
        elif kind is ShellCommandKind.SET_PLOT_OPTION:
            field = dict(zip(PLOT_OPTION_COMMAND_PATHS, _PLOT_OPTION_FIELDS)).get(command.path)
            if field is None:
                return
            self._plot_options = replace(self._plot_options, **{field: value})
        elif kind is ShellCommandKind.CLEAR_1D:
            self.clear_plot()
            return
        else:
            return
        self._draw()

    def _install_diagnostic_option(self):
        """Stitch's Diagnostic (Intensity, Coverage or Normalization) as a row of the view's
        own plot-options dialog."""
        dialog = self.scientific.plot_options_dialog
        self._diagnostic_combo = QtWidgets.QComboBox()
        self._diagnostic_combo.addItems(("Intensity", "Coverage", "Normalization"))
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Diagnostic"))
        layout.addWidget(self._diagnostic_combo)
        dialog.layout().insertWidget(dialog.layout().count() - 1, row)
        dialog.accepted.connect(self._diagnostic_accepted)

    def _diagnostic_accepted(self):
        if self._diagnostic_combo.currentText() != self._diagnostic:
            self._diagnostic = self._diagnostic_combo.currentText()
            self._draw()

    def clear_plot(self):
        self._view_revision += 1
        self.results.clearSelection()
        self.results.setCurrentRow(-1)
        self._draw()

    def _show_metadata(self):
        payload = self._current_payload()
        if payload is None or self._owner.busy or self._closing:
            return
        from .metadata import StitchMetadataDialog
        if self._metadata_dialog is not None:
            self._metadata_dialog.close()
        dialog = StitchMetadataDialog(payload.provenance_json, self)
        self._metadata_dialog = self._metadata_request_dialog = dialog
        self._metadata_view_revision = self._view_revision
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.finished.connect(lambda _result: self._metadata_closed(dialog))
        dialog.status.setText("Loading experimental metadata…")
        dialog.show()
        if not self._start("Loading experimental metadata…", lambda: self._owner.begin_metadata(
                payload, source_root=str(self._value("project_root")) or None)):
            dialog.status.setText("Metadata reader is busy; close and retry")

    def _metadata_current(self):
        return (not self._closing and self._metadata_dialog is not None
                and self._metadata_dialog is self._metadata_request_dialog
                and self._metadata_view_revision == self._view_revision)

    def _metadata_closed(self, dialog):
        if self._metadata_dialog is dialog:
            self._metadata_dialog = None
            identity = self._owner.current_identity
            if identity is not None and identity.request.action is StitchOwnerAction.METADATA:
                self._owner.cancel()

    def _sync_actions(self):
        busy, final = self._owner.busy, self._owner.finalization
        authoring = self._authoring_active()
        idle = not busy and not authoring and final is StitchOwnerFinalization.NONE and not self._closing
        self.controls.setEnabled(idle)
        for dialog in self._authoring_dialogs:
            dialog.setEnabled(not busy and not self._closing and (not authoring or dialog.active()))
        if not any(dialog.isVisible() or dialog.active() for dialog in self._authoring_dialogs):
            self._authoring_timer.stop()
        for row in self._rows.values():
            row.setEnabled(idle)
        if self._value("backend") == "xrayutilities histogram":
            locked = ("geometry_kind", "source_motors", "poni_references", "image_rotation")
            for key in locked:
                self._rows[key].setEnabled(False)
        elif self._value("geometry_kind") != "PONI + offsets":
            self._rows["poni_references"].setEnabled(False)
        for row in self._range_rows:
            row.setEnabled(idle)
        self.calibrate_button.setEnabled(idle)
        self.mask_button.setEnabled(idle)
        self.raw_button.setEnabled(self.frames.count() > 0 and not self._closing)
        self.advanced.setEnabled(idle)
        self.open_button.setEnabled(idle)
        self.metadata_button.setEnabled(idle and self._current_payload() is not None)
        self.preview_button.setEnabled(idle)
        self.run_bar.startButton.setEnabled(idle and self._prepared_revision == self._revision
                                            and self._owner.prepared is not None
                                            and bool(self._owner.prepared.requests))
        self.run_bar.set_stop_enabled(busy and self._owner.current_identity.request.action is StitchOwnerAction.RUN)
        self.retry_button.setVisible(final is not StitchOwnerFinalization.NONE)
        self.retry_button.setEnabled(not busy)

    def _authoring_active(self):
        return any(dialog.active() for dialog in self._authoring_dialogs)

    def active(self):
        return self._owner.busy or self._owner.finalization is not StitchOwnerFinalization.NONE or self._closing or self._authoring_active()

    def shutdown(self):
        self._closing = True
        self._queued_jobs.clear()
        self._raw_closed()
        self._view_revision += 1
        self._timer.stop()
        self._authoring_timer.stop()
        receipt = self._owner.close()
        child_receipts = [dialog.shutdown() for dialog in self._authoring_dialogs]
        if any(child.status is PageCleanup.PENDING for child in child_receipts):
            from xdart.gui.pages.values import CloseReceipt
            return CloseReceipt(PageCleanup.PENDING, "Calibration or mask editor is closing")
        if receipt.status is PageCleanup.CLEAN:
            for dialog in self.findChildren(QtWidgets.QDialog):
                dialog.close()
            self.curve.clear()
            self._payloads.clear()
        return receipt

    # -- saved settings (File ▸ Save/Load Settings and the last session) -----------------

    def settings(self):
        """Every input, the source files, the threshold policy and the calibration dialog."""
        from .settings import page_settings
        return page_settings(self)

    def apply_settings(self, document):
        """Restore a settings document; one that cannot apply leaves the inputs as they were."""
        from .settings import apply_page_settings
        before = self.settings()
        try:
            apply_page_settings(self, document)
        except Exception:
            apply_page_settings(self, before)
            raise

    def _calibration_settings_now(self):
        dialog = self._calibration_dialog
        return dialog.settings() if dialog is not None else self._calibration_settings

    def _set_calibration_settings(self, values):
        self._calibration_settings = values if isinstance(values, dict) else None
        if self._calibration_dialog is not None and self._calibration_settings:
            self._calibration_dialog.apply_settings(self._calibration_settings)

    def _choose_settings_path(self, action, start):
        caption, file_filter = "Stitch settings", "Stitch settings (*.json);;All files (*)"
        if action == "save":
            selected, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save " + caption, start, file_filter)
        else:
            selected, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load " + caption, start, file_filter)
        return selected or None

    def _settings_path(self, action):
        from xdart.utils.browse import browse_start_dir, remember_browse_path
        selected = self._settings_path_chooser(action, browse_start_dir("", fallback=self._value("project_root")))
        if not selected:
            return None
        path = Path(selected).expanduser()
        if action == "save" and not path.suffix:
            path = path.with_suffix(".json")
        remember_browse_path(path)
        return path

    def save(self):
        """File ▸ Save Settings."""
        from .settings import dump_settings
        path = self._settings_path("save")
        if path is None:
            return ActionCompleted("stitch-settings-save-cancelled")
        try:
            text = dump_settings(self.settings())
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)
        except (OSError, ValueError) as error:
            self._status(f"Could not save the Stitch settings: {error}", ready=False)
            return ActionRefused(str(error))
        self._status(f"Stitch settings saved: {path.name}")
        return ActionCompleted("stitch-settings-saved")

    def load(self):
        """File ▸ Load Settings."""
        from .settings import load_settings
        if self._owner.busy:
            return ActionRefused("Stitch is busy; load settings when it is idle")
        path = self._settings_path("load")
        if path is None:
            return ActionCompleted("stitch-settings-load-cancelled")
        try:
            self.apply_settings(load_settings(path.read_text(encoding="utf-8")))
        except Exception as error:  # noqa: BLE001 - a settings file is user input
            reason = str(error) if isinstance(error, (OSError, ValueError)) else (
                f"these Stitch settings cannot be applied here ({type(error).__name__}: {error})")
            self._status(f"Could not load the Stitch settings: {reason}", ready=False)
            return ActionRefused(reason)
        self._status(f"Stitch settings loaded: {path.name} · Preview required", ready=False)
        return ActionCompleted("stitch-settings-loaded")

    def restore_last_settings(self):
        """Restore the settings the last session left (nothing in a fresh session)."""
        from xdart.utils.session import load_session
        from .settings import SESSION_KEY
        document = load_session().get(SESSION_KEY)
        if document is None:
            return
        try:
            self.apply_settings(document)
        except Exception:  # noqa: BLE001 - saved settings are a convenience
            _LOG.warning("Could not restore the last Stitch settings", exc_info=True)
            self._status("Could not restore the last Stitch settings", ready=False)

    def close_and_remember(self):
        """Close the page and keep its settings for the next session once it is clean."""
        if self._closing_settings is None and not self._closing:
            try:
                self._closing_settings = self.settings()
            except Exception:  # noqa: BLE001
                _LOG.warning("Could not capture the Stitch settings", exc_info=True)
        receipt = self.shutdown()
        if (receipt.status is PageCleanup.CLEAN and not self._settings_remembered
                and self._closing_settings is not None):
            from xdart.utils.session import save_session
            from .settings import SESSION_KEY
            save_session({SESSION_KEY: self._closing_settings})
            self._settings_remembered = True
        return receipt

    def mount_points(self):
        return AppMenuHosts(self.config_menu, self.help_menu, tool_requested=self.toolRequested)

    def request(self):
        self.open_result()
        return ActionAccepted("open-stitch-result")

    def run_pause(self):
        self.begin_run()
        return ActionAccepted("stitch-run")


def build_stitch_page(services, parent):
    page = StitchPage(services, parent)
    page.restore_last_settings()
    return PageHandle(key=STITCH_TOOL_KEY, widget=page, close=page.close_and_remember,
                      open_folder=page, settings_io=page, run_control=page, activity=page,
                      app_menus=page)
