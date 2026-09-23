"""Native Stitch workspace over the existing finite operation owner.

Source admission, calibration, raw reads and publication stay with their shared
headless owners. This page queues their exact prepared requests and presents
the resulting curves and original contributing frames.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import time

import numpy as np

import pyqtgraph as pg
from matplotlib import colormaps
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.pages.handle import AppMenuHosts, PageHandle
from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.pages.values import (
    ActionAccepted, PageCleanup,
    SCATTERING_PAGE_KEY, STITCH_TOOL_KEY,
)
from xdart.gui.themes import apply_seaborn_plot_style
from xdart.gui.tools.stitch_owner import (
    StitchOwnerAction, StitchOwnerFinalization, StitchOwnerOutcomeKind,
    StitchToolOwner,
)
from xdart.gui.tools.stitch_tool import StitchToolDialog
from xdart.gui.tools.stitch_values import StitchFrameSelector, StitchToolForm
from xdart.gui.widgets.controls_panel import ControlsPanel, FormRow, RangeRow, SubsectionCard
from xdart.gui.widgets.run_controls import RunControlsBar
from xdart.gui.widgets.scientific_plot import (
    PlotCursorReadout, scaled_intensity, scaled_intensity_label, overlay_step, trace_colors,
)
from xdart.gui.tabs.scattering.scientific_plot_options import PlotOptionsDialog
from xdart.gui.tabs.scattering.shell_values import ScientificPlotOptions
from xdart.gui.tabs.scattering.workspace_shell import _HugWidthScrollArea
from xrd_tools.analysis.module_transaction import MetadataColumnSelector, ModuleDisposition
from xrd_tools.analysis.stitch_operation import StitchGeometryKind
from xrd_tools.analysis.stitch_sources import StitchSourceSelection
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, AnalysisArtifactPayload
from xrd_tools.session.readiness import ControlFieldKind, ControlFormField, SectionId


class StitchPage(QtWidgets.QWidget):
    """One retained page; one worker owns preview, science and strict reopen."""

    toolRequested = QtCore.Signal(str)

    def __init__(self, services, parent=None):
        super().__init__(parent)
        self.setObjectName("stitchPage")
        self._services = services
        self._owner = StitchToolOwner()
        self._native_mode = True
        self._source_paths = ()
        self._paths_by_kind = {}
        self._range_rows = []
        self._range_keys = set()
        self._threshold_mode = "auto"
        self._threshold_manual = False
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
        self._options = None
        self._build_ui()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)
        self._authoring_timer = QtCore.QTimer(self)
        self._authoring_timer.setInterval(50)
        self._authoring_timer.timeout.connect(self._sync_actions)
        self._status("Ready · Stitch 1D")
        self._sync_actions()

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

        centre = QtWidgets.QWidget()
        centre_layout = QtWidgets.QVBoxLayout(centre)
        centre_layout.setContentsMargins(0, 5, 0, 0)
        header = QtWidgets.QHBoxLayout()
        self.result_title = QtWidgets.QLabel("Current")
        self.result_title.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.result_title.setMinimumWidth(0)
        self.color_map = QtWidgets.QComboBox()
        self.color_map.setObjectName("stitchColorMap")
        self.color_map.addItems(("Default", *tuple(colormaps)))
        self.color_map.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.color_map.setMinimumContentsLength(9)
        self.log_button = QtWidgets.QPushButton("Log")
        self.log_button.setObjectName("stitchLog")
        self.log_button.setCheckable(True)
        self.log_button.toggled.connect(self._set_log)
        self.color_map.currentTextChanged.connect(self._display_changed)
        header.addWidget(self.result_title, 1)
        header.addWidget(self.color_map)
        header.addWidget(self.log_button)
        centre_layout.addLayout(header)
        self.curve = pg.PlotWidget()
        self.curve.setObjectName("stitchCurve")
        apply_seaborn_plot_style(self.curve)
        self.legend = self.curve.addLegend()
        self._cursor_readout = PlotCursorReadout(self.curve)
        centre_layout.addWidget(self.curve, 1)
        plot_bar = QtWidgets.QHBoxLayout()
        self.viewer_axis = QtWidgets.QComboBox()
        self.viewer_axis.addItem("Q (Å⁻¹)", "q_A^-1")
        self.viewer_axis.addItem("2θ (°)", "2th_deg")
        self.viewer_axis.currentIndexChanged.connect(self._draw)
        self.viewer_axis.setToolTip("Saved radial coordinate")
        self.display_mode = QtWidgets.QComboBox()
        self.display_mode.setObjectName("stitchDisplayMode")
        self.display_mode.addItems(("Single", "Overlay"))
        self.display_mode.currentTextChanged.connect(self._display_mode_changed)
        options = QtWidgets.QPushButton("Options")
        options.clicked.connect(self._show_options)
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(self.clear_plot)
        plot_bar.addWidget(self.viewer_axis)
        plot_bar.addStretch(1)
        plot_bar.addWidget(self.display_mode)
        plot_bar.addWidget(options)
        plot_bar.addWidget(clear)
        centre_layout.addLayout(plot_bar)
        footer = QtWidgets.QHBoxLayout()
        self.result_status = QtWidgets.QLabel("")
        self.result_status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.result_status.setMinimumWidth(0)
        self.result_status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                         QtWidgets.QSizePolicy.Policy.Preferred)
        self.previous = QtWidgets.QPushButton("<")
        self.next = QtWidgets.QPushButton(">")
        for button in (self.previous, self.next):
            button.setMaximumWidth(32)
        self.result_selector = QtWidgets.QComboBox()
        self.result_selector.setMinimumWidth(45)
        self.result_selector.currentIndexChanged.connect(self._select_result)
        self.result_count = QtWidgets.QLabel("0/0")
        self.previous.clicked.connect(lambda: self._step_result(-1))
        self.next.clicked.connect(lambda: self._step_result(1))
        for widget in (self.result_status, self.previous, self.result_selector, self.next, self.result_count):
            footer.addWidget(widget, 1 if widget is self.result_status else 0)
        centre_layout.addLayout(footer)
        self.splitter.addWidget(centre)

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
        self.run_bar.apply_profile(modes=("Stitch 1D",), live=False, batch=False, cores=False)
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

    def _field(self, card, section, key, label, value="", *, choices=(), browse=False):
        kind = ControlFieldKind.COMBO if choices else ControlFieldKind.LINE
        path = ("Project", "project_folder") if key == "project_root" else ("Stitch", key)
        field = ControlFormField(section, label, path, value, kind, tuple(choices), browse)
        row = FormRow(label=label, path=path, value=value, kind=kind.value,
                      choices=choices, browse=browse)
        row.editor.setObjectName("stitch_" + key)
        row.valueChanged.connect(self._edited)
        row.draftChanged.connect(self._edited)
        row.browseRequested.connect(lambda _path, name=key: self._browse(name))
        card.add_row(row)
        self._fields[key], self._rows[key] = field, row
        return row

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
        self._field(source, SectionId.SOURCE, "output_family", "Result names")
        self._rows["output_family"].setToolTip(
            "Required when combining sources. One name applies to the sole combined result. "
            "For several outputs, separate names in output order with ; and leave single-source "
            "names blank for automatic naming. Example for 1,2,[1,2]: ;;combined. "
            "Names are at most 80 characters; processing and frame selection keep the same name.")
        self._rows["source_kind"].valueChanged.connect(self._source_changed)
        self._rows["scan_mode"].valueChanged.connect(self._source_changed)
        self._rows["spec_path"].valueChanged.connect(self._source_path_edited)
        source.set_status_text("metadata required")
        grid = SubsectionCard("1-D", accent="processing")
        processing.add_row(grid)
        self._field(grid, SectionId.PROCESSING, "unit", "Axis", "Q (Å⁻¹)", choices=("Q (Å⁻¹)", "2θ (°)"))
        self._rows["unit"].valueChanged.connect(self._processing_axis_changed)
        self._field(grid, SectionId.PROCESSING, "npt_1d", "Pts", "1500")
        self.radial_range = self._range(grid, "Q (Å⁻¹)", ("q_min", "q_max", "radial_auto"), (0., 6.2, True))
        conditioning = SubsectionCard("Conditioning", accent="processing")
        processing.add_row(conditioning)
        self.threshold_range = self._range(conditioning, "Threshold", ("threshold_min", "threshold", "threshold_enabled"), (0., None, True))
        self.threshold_range._high.setPlaceholderText("Auto")
        self._field(conditioning, SectionId.PROCESSING, "monitor", "Monitor")
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

    def _range(self, card, label, keys, values):
        specs = []
        for key, value in zip(keys, values):
            field = ControlFormField(SectionId.PROCESSING, label, ("Stitch", key), value,
                ControlFieldKind.BOOL if isinstance(value, bool) else ControlFieldKind.LINE)
            self._fields[key] = field
            self._range_keys.add(key)
            specs.append(dict(path=field.path, value=value))
        row = RangeRow(label=label, low=specs[0], high=specs[1], toggle=specs[2])
        row.valueChanged.connect(self._range_edited)
        row.draftChanged.connect(self._edited)
        self._range_rows.append(row)
        card.add_row(row)
        return row

    def _render_ranges(self):
        values = {field.path: field for key, field in self._fields.items() if key in self._range_keys}
        auto = bool(self._fields["radial_auto"].value)
        for key in ("q_min", "q_max"):
            values[("Stitch", key)] = replace(self._fields[key], enabled=not auto)
        for row in self._range_rows:
            row.apply_fields(values)

    def _range_edited(self, path, value):
        key = path[-1]
        old = self._fields[key].value
        if key not in {"radial_auto", "threshold_enabled"}:
            try:
                value = None if value is None or not str(value).strip() else float(value)
            except ValueError:
                self._status("Enter a number or clear the threshold maximum for Auto", ready=False)
                return
        if value == old:
            return
        self._fields[key] = replace(self._fields[key], value=value)
        if key == "threshold" and value is None:
            self._threshold_mode, self._threshold_manual = "auto", False
            self._fields["threshold_min"] = replace(self._fields["threshold_min"], value=0.)
            self._fields["threshold"] = replace(self._fields["threshold"], value=self._auto_threshold_max)
            self._fields["threshold_enabled"] = replace(self._fields["threshold_enabled"], value=True)
        elif key in {"threshold_min", "threshold"}:
            if self._fields["threshold"].value is None:
                self._status("Enter an upper threshold before selecting a manual lower bound", ready=False)
                self._fields[key] = replace(self._fields[key], value=old)
            else:
                self._threshold_mode, self._threshold_manual = "manual", True
                self._fields["threshold_enabled"] = replace(self._fields["threshold_enabled"], value=True)
        elif key == "threshold_enabled":
            self._threshold_mode = ("manual" if self._threshold_manual else "auto") if value else "off"
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

    def _backend_changed(self, *_args):
        xu = self._value("backend") == "xrayutilities histogram"
        self.set_field("source_motors", "del=del, nu=nu" if xu else "del_angle=del, nu_angle=nu")
        if not xu and not str(self._value("image_rotation")).strip():
            self.set_field("image_rotation", "0")
        self.set_field("q_max", "5.2" if xu else "6.2")
        if xu and not self._native_mode:
            for key, value in (
                ("geometry_kind", "Goniometer JSON"), ("poni_references", ""),
                ("detector_rows", "195"), ("detector_cols", "1475"),
                ("raw_dtype", "int32"), ("raw_header_skip", "0"),
                ("image_rotation", "0"), ("threshold", "800000"),
            ):
                self.set_field(key, value)
        self._refresh_xu_geometry()
        self._sync_actions()

    def _refresh_xu_geometry(self, *_args, calibration=None):
        """Present asset-owned settings through the existing strict admission owner."""
        if not self._native_mode or self._value("backend") != "xrayutilities histogram":
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
        start = str(self._value("project_root") or Path.home())
        if key in {"project_root", "image_dir", "output_path"} or key == "spec_path" and self._value("source_kind") == "Directory":
            value = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose folder", start)
        elif key == "spec_path":
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose source files", start,
                "NeXus / HDF5 (*.nexus *.nxs *.h5 *.hdf5);;All files (*)" if self._value("source_kind") == "NeXus" else "All files (*)")
            if files:
                self.set_sources(files)
            return
        else:
            value, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose " + self._fields[key].label, start)
        if value:
            self.set_field(key, value)

    def build_form(self):
        value = self._value
        kind = (StitchGeometryKind.PYFAI_GONIOMETER_JSON if value("geometry_kind") == "Goniometer JSON"
                else StitchGeometryKind.PONI)
        monitor = str(value("monitor")).strip()
        native = {}
        output = str(value("output_path"))
        if self._native_mode:
            selection = StitchSourceSelection(value("source_kind"), self._source_paths,
                scans=str(value("scan")), stitch_all=value("scan_mode") == "Stitch All",
                image_dir=str(value("image_dir")), image_stem=str(value("image_stem")))
            conditioning = StitchConditioning(self._threshold_mode,
                value("threshold_min") if self._threshold_mode == "manual" else None,
                value("threshold") if self._threshold_mode == "manual" else None)
            native = dict(source_selection=selection,
                unit="q_A^-1" if value("unit") == "Q (Å⁻¹)" else "2th_deg",
                conditioning=conditioning, mask_path=str(value("mask_path")) or None,
                output_family=str(value("output_family")).strip() or None)
            if ";" in str(value("output_family")):
                native.update(output_family=None,
                    output_families=tuple(part.strip() for part in str(value("output_family")).split(";")))
            if not output:
                output = str(Path(value("project_root")) / "xdart_processed_data")
            directory = Path(output)
            if not directory.is_absolute():
                directory = Path(value("project_root")) / directory
            output = str(directory / "stitch_stitch1d.nexus")
        return StitchToolForm(
            project_root=value("project_root"),
            spec_path=self._source_paths[0] if self._native_mode else value("spec_path"), scan=value("scan"),
            image_dir=(value("image_dir") or value("project_root")) if self._native_mode else value("image_dir"), image_stem=value("image_stem"),
            detector_shape=(int(value("detector_rows")), int(value("detector_cols"))),
            raw_dtype=value("raw_dtype"), raw_header_skip=int(value("raw_header_skip")),
            frame_selector=StitchFrameSelector(int(value("frame_start")),
                None if not str(value("frame_stop")).strip() else int(value("frame_stop")), int(value("frame_step"))),
            geometry_path=value("geometry_path"), geometry_kind=kind,
            source_motors=StitchToolDialog._parse_pairs(value("source_motors")),
            poni_references=(StitchToolDialog._parse_references(value("poni_references"))
                             if kind is StitchGeometryKind.PONI else ()),
            image_rotation=int(value("image_rotation")),
            threshold=None if value("threshold") is None or not str(value("threshold")).strip() else float(value("threshold")),
            q_range=None if self._native_mode and value("radial_auto") else (float(value("q_min")), float(value("q_max"))), npt_1d=int(value("npt_1d")),
            output_path=output, monitor_selector=MetadataColumnSelector(monitor) if monitor else None,
            backend="xu_hist" if value("backend") == "xrayutilities histogram" else "multigeometry",
            max_frame_bytes=int(value("max_frame_mib")) * 1024 * 1024,
            **native,
        )

    def load_form(self, form):
        """Restore a prepared form, including native source and conditioning intent."""
        self._native_mode = form.source_selection is not None
        fields = {key: getattr(form, key) for key in (
            "project_root", "spec_path", "scan", "image_dir", "image_stem", "raw_dtype",
            "raw_header_skip", "geometry_path", "threshold", "npt_1d", "output_path")}
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
            frame_step=form.frame_selector.step, radial_auto=form.q_range is None)
        if form.q_range is not None:
            fields.update(q_min=form.q_range[0], q_max=form.q_range[1])
        for key, value in fields.items():
            self.set_field(key, value)
        if form.source_selection is not None:
            selection = form.source_selection
            self.set_field("source_kind", selection.kind)
            self.set_sources(selection.paths)
            self.set_field("scan_mode", "Stitch All" if selection.stitch_all else "Select Scans")
            self.set_field("scan", selection.scans)
            self.set_field("output_path", str(Path(form.output_path).parent))
            self.set_field("image_dir", selection.image_dir)
            self.set_field("image_stem", selection.image_stem)
            policy = form.conditioning or StitchConditioning()
            self._threshold_mode = policy.mode
            self._threshold_manual = policy.mode == "manual"
            for key, value in (("threshold_min", policy.threshold_min if policy.mode == "manual" else 0.),
                               ("threshold", policy.threshold_max), ("threshold_enabled", policy.mode != "off")):
                self._fields[key] = replace(self._fields[key], value=value)
            self._render_ranges()

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
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open stitched result", "", "NeXus (*.nexus)")
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
        if outcome.kind is StitchOwnerOutcomeKind.PREFLIGHT_READY:
            if update.stale or self._action_revision != self._revision:
                self._status("Inputs changed · Preview required", ready=False)
                return
            self._prepared_revision = self._revision
            preflight = outcome.preflight
            if self._native_mode:
                if preflight.requests and preflight.form.backend == "xu_hist":
                    self._refresh_xu_geometry(calibration=preflight.requests[0].plan.calibration)
                bounds = []
                for request in preflight.requests:
                    for member in request.provenance.get("frame_members", ()):
                        dtype = np.dtype(member["source_snapshot"]["raw_dtype"])
                        bounds.append(float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else None)
                self._auto_threshold_max = bounds[0] if bounds and all(value == bounds[0] for value in bounds) else None
                if not self._threshold_manual:
                    self._fields["threshold"] = replace(self._fields["threshold"], value=self._auto_threshold_max)
                    self._render_ranges()
            if preflight.summary is not None:
                StitchToolDialog._render_preflight(self, preflight.summary)
            else:
                lines = [f"{len(preflight.requests)} stitched outputs"]
                for index, request in enumerate(preflight.requests, 1):
                    count = len(request.module.source.selected_labels)
                    lines.append(f"\n{index}. {Path(request.module.output.target).name} · {count} frames")
                    for member in request.provenance.get("frame_members", ()):
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
                self._status("Ready · saved Stitch 1D result")
        elif outcome.kind is StitchOwnerOutcomeKind.RESULT:
            result = outcome.result
            if result.terminal.disposition is ModuleDisposition.COMMITTED:
                if not update.stale and self._action_revision == self._revision and not self._closing:
                    self._mount_payload(result.request.module.output.target, result.payload)
                elapsed = time.monotonic() - (self._batch_started or self._started)
                count = len(result.request.module.source.selected_labels)
                self._status(f"Complete · {count} frames · {elapsed:.2f} s")
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
                item = QtWidgets.QListWidgetItem(label)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, (storage_scan, frame))
                item.setToolTip(f"{record['source_path']}\nscan {identity['scan_id']} · frame {frame}")
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
                color_map=self.color_map.currentText(), log=self._plot_options.intensity_scale == "Log")

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
                self._calibration_dialog.hide()
                self._calibration_dialog = None
            if self._calibration_dialog is None:
                self._calibration_dialog = StitchCalibrationDialog(str(self._value("project_root")),
                    backend=backend, parent=self)
                self._calibration_project, self._calibration_backend = project, backend
                self._calibration_dialog.calibrationSaved.connect(self._adopt_calibration)
                self._authoring_dialogs.append(self._calibration_dialog)
            source = self._source_spec_for_tools()
            if source is not None:
                motors = dict(StitchToolDialog._parse_pairs(self._value("source_motors") or "del=del, nu=nu"))
                self._calibration_dialog.set_source_context(source, image_rotation=int(self._value("image_rotation") or 0),
                    del_motor=motors.get("del_angle", motors.get("del", "del")),
                    nu_motor=motors.get("nu_angle", motors.get("nu", "nu")))
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
            self.result_selector.addItem(str(self.results.count()))
        self.results.setCurrentItem(item)
        self._current_result_changed()

    def _selected(self):
        items = self.results.selectedItems()
        if self.display_mode.currentText() == "Single":
            item = self.results.currentItem()
            items = [] if item is None else [item]
        return [(item, self._payloads[item.data(QtCore.Qt.ItemDataRole.UserRole)]) for item in items]

    def _draw(self, *_args):
        if not hasattr(self, "curve"):
            return
        self.curve.clear()
        self.legend.clear()
        selected = self._selected()
        scale = self._plot_options.intensity_scale
        available = all(payload.wavelength_A is not None for _, payload in selected)
        self.viewer_axis.model().item(1).setEnabled(available)
        if not available and self.viewer_axis.currentIndex() == 1:
            blocker = QtCore.QSignalBlocker(self.viewer_axis)
            self.viewer_axis.setCurrentIndex(0)
            del blocker
        self.viewer_axis.setToolTip("Saved radial coordinate" if available else "This earlier result has no saved wavelength for 2θ conversion")
        unit = self.viewer_axis.currentData()
        self.curve.setLabel("bottom", "Q" if unit == "q_A^-1" else "2θ", units="Å⁻¹" if unit == "q_A^-1" else "°")
        self.curve.setLabel("left", scaled_intensity_label(self._diagnostic, scale))
        arrays, items = [], []
        for item, payload in selected:
            values = {"Intensity": payload.intensity, "Coverage": payload.coverage,
                      "Normalization": payload.normalization}[self._diagnostic]
            if values is None:
                continue
            arrays.append(scaled_intensity(values, scale))
            items.append((item, payload))
        step = overlay_step(arrays, self._plot_options.overlay_offset)
        colors = trace_colors(self.color_map.currentText(), len(items))
        for index, ((item, payload), values, color) in enumerate(zip(items, arrays, colors)):
            try:
                radial = payload.radial_axis(unit)
            except ValueError:
                if payload.native_unit != unit:
                    continue
                radial = payload.axis("q")
            self.curve.plot(radial, values + index * step,
                            pen=pg.mkPen(color, width=1.4), symbol="o", symbolSize=4,
                            symbolBrush=color, symbolPen=color, connect="finite", name=item.text())
        self.legend.setVisible(self._plot_options.show_legend)
        current = self.results.currentItem()
        self.result_title.setText(current.text() if current else "Current")
        self.result_status.setText(current.text() if current else "")
        index = self.results.currentRow()
        blocked = QtCore.QSignalBlocker(self.result_selector)
        self.result_selector.setCurrentIndex(index)
        del blocked
        self.result_count.setText(f"{index + 1}/{self.results.count()}")

    def clear_plot(self):
        self._view_revision += 1
        self.results.clearSelection()
        self.results.setCurrentRow(-1)
        self._draw()

    def _select_result(self, index):
        if index >= 0 and index != self.results.currentRow():
            flag = (QtCore.QItemSelectionModel.SelectionFlag.NoUpdate
                    if self.display_mode.currentText() == "Overlay"
                    else QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect)
            self.results.setCurrentRow(index, flag)

    def _step_result(self, step):
        index = self.results.currentRow() + step
        if 0 <= index < self.results.count():
            self._select_result(index)

    def _display_mode_changed(self, mode):
        if mode == "Single":
            self.results.setCurrentRow(self.results.currentRow(),
                QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect)
        self._draw()

    def _set_log(self, checked):
        self._plot_options = replace(self._plot_options, intensity_scale="Log" if checked else "Linear")
        self._display_changed()

    def _display_changed(self, *_args):
        self._draw()
        self._render_raw()

    def _show_options(self):
        if self._options is None:
            self._options = PlotOptionsDialog(self, show_waterfall=False)
            self._diagnostics_combo = QtWidgets.QComboBox()
            self._diagnostics_combo.addItems(("Intensity", "Coverage", "Normalization"))
            row = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(row)
            layout.addWidget(QtWidgets.QLabel("Diagnostic"))
            layout.addWidget(self._diagnostics_combo)
            self._options.layout().insertWidget(self._options.layout().count() - 1, row)
            self._options.accepted.connect(self._accept_options)
        self._options.reconcile(self._plot_options)
        self._diagnostics_combo.setCurrentText(self._diagnostic)
        self._options.show()
        self._options.raise_()

    def _accept_options(self):
        self._plot_options = self._options.edited_options()
        self._diagnostic = self._diagnostics_combo.currentText()
        blocker = QtCore.QSignalBlocker(self.log_button)
        self.log_button.setChecked(self._plot_options.intensity_scale == "Log")
        del blocker
        self._draw()

    def _show_metadata(self):
        current = self.results.currentItem()
        if current is None:
            return
        payload = self._payloads[current.data(QtCore.Qt.ItemDataRole.UserRole)]
        dialog = QtWidgets.QDialog(self)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle("Saved Stitch provenance")
        layout = QtWidgets.QVBoxLayout(dialog)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(payload.provenance_json)
        layout.addWidget(text)
        dialog.resize(680, 450)
        dialog.show()

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
            if not self._native_mode:
                locked += ("detector_rows", "detector_cols", "raw_dtype", "raw_header_skip")
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
    return PageHandle(key=STITCH_TOOL_KEY, widget=page, close=page.shutdown,
                      open_folder=page, run_control=page, activity=page, app_menus=page)
