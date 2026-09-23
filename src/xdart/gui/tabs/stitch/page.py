"""Native Stitch workspace over the existing finite operation owner.

The first mounted path consumes the already admitted single-SPEC operation.
Extended source and calibration admission belongs to that headless path, never
to this presentation or a second publisher.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time

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
from xdart.gui.widgets.controls_panel import ControlsPanel, FormRow, SubsectionCard
from xdart.gui.widgets.run_controls import RunControlsBar
from xdart.gui.widgets.scientific_plot import (
    PlotCursorReadout, scaled_intensity, scaled_intensity_label, overlay_step, trace_colors,
)
from xdart.gui.tabs.scattering.scientific_plot_options import PlotOptionsDialog
from xdart.gui.tabs.scattering.shell_values import ScientificPlotOptions
from xdart.gui.tabs.scattering.workspace_shell import _HugWidthScrollArea
from xrd_tools.analysis.module_transaction import MetadataColumnSelector, ModuleDisposition
from xrd_tools.analysis.stitch_operation import StitchGeometryKind
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
        self.results.currentItemChanged.connect(self._draw)
        left_layout.addWidget(self.results, 1)
        browse_actions = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("Open…")
        self.open_button.clicked.connect(self.open_result)
        self.metadata_button = QtWidgets.QPushButton("Metadata")
        self.metadata_button.clicked.connect(self._show_metadata)
        browse_actions.addWidget(self.open_button)
        browse_actions.addWidget(self.metadata_button)
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
        self.color_map.currentTextChanged.connect(self._draw)
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
        self.viewer_axis.addItem("Q (Å⁻¹)")
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
        self._field(project, SectionId.PROJECT, "output_path", "Output", browse=True)
        self._field(experiment, SectionId.EXPERIMENT, "backend", "Backend",
                    "pyFAI MultiGeometry", choices=("pyFAI MultiGeometry", "xrayutilities histogram"))
        self._field(experiment, SectionId.EXPERIMENT, "geometry_path", "Calibration", browse=True)
        self._rows["backend"].valueChanged.connect(self._backend_changed)
        self._field(source, SectionId.SOURCE, "spec_path", "SPEC file", browse=True)
        self._field(source, SectionId.SOURCE, "image_dir", "Images", browse=True)
        self._field(source, SectionId.SOURCE, "scan", "Scan", "1")
        source.set_status_text("metadata required")
        grid = SubsectionCard("1-D", accent="processing")
        processing.add_row(grid)
        self._field(grid, SectionId.PROCESSING, "q_min", "Q min (Å⁻¹)", "1")
        self._field(grid, SectionId.PROCESSING, "q_max", "Q max (Å⁻¹)", "6.2")
        self._field(grid, SectionId.PROCESSING, "npt_1d", "Pts", "1500")
        conditioning = SubsectionCard("Conditioning", accent="processing")
        processing.add_row(conditioning)
        self._field(conditioning, SectionId.PROCESSING, "threshold", "Threshold max", "800000")
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

    def set_field(self, key, value):
        field = replace(self._fields[key], value=value)
        self._fields[key] = field
        self._rows[key].apply_field(field)
        self._edited()
        if key == "backend":
            self._backend_changed()

    def _value(self, key):
        return self._rows[key].current_value()

    def _edited(self, *_args):
        self._revision += 1
        self._prepared_revision = None
        if hasattr(self, "run_bar"):
            self._status("Inputs changed · Preview required", ready=False)
            self._sync_actions()

    def _backend_changed(self, *_args):
        xu = self._value("backend") == "xrayutilities histogram"
        self.set_field("source_motors", "del=del, nu=nu" if xu else "del_angle=del, nu_angle=nu")
        self.set_field("q_max", "5.2" if xu else "6.2")
        if xu:
            for key, value in (
                ("geometry_kind", "Goniometer JSON"), ("poni_references", ""),
                ("detector_rows", "195"), ("detector_cols", "1475"),
                ("raw_dtype", "int32"), ("raw_header_skip", "0"),
                ("image_rotation", "0"), ("threshold", "800000"),
            ):
                self.set_field(key, value)
        self._sync_actions()

    def _browse(self, key):
        start = str(self._value("project_root") or Path.home())
        if key in {"project_root", "image_dir"}:
            value = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose folder", start)
        elif key == "output_path":
            value, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Choose Stitch output", start, "NeXus (*.nexus)")
        else:
            value, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose " + self._fields[key].label, start)
        if value:
            self.set_field(key, value)

    def build_form(self):
        value = self._value
        kind = (StitchGeometryKind.PYFAI_GONIOMETER_JSON if value("geometry_kind") == "Goniometer JSON"
                else StitchGeometryKind.PONI)
        monitor = str(value("monitor")).strip()
        return StitchToolForm(
            project_root=value("project_root"), spec_path=value("spec_path"), scan=value("scan"),
            image_dir=value("image_dir"), image_stem=value("image_stem"),
            detector_shape=(int(value("detector_rows")), int(value("detector_cols"))),
            raw_dtype=value("raw_dtype"), raw_header_skip=int(value("raw_header_skip")),
            frame_selector=StitchFrameSelector(int(value("frame_start")),
                None if not str(value("frame_stop")).strip() else int(value("frame_stop")), int(value("frame_step"))),
            geometry_path=value("geometry_path"), geometry_kind=kind,
            source_motors=StitchToolDialog._parse_pairs(value("source_motors")),
            poni_references=(StitchToolDialog._parse_references(value("poni_references"))
                             if kind is StitchGeometryKind.PONI else ()),
            image_rotation=int(value("image_rotation")),
            threshold=None if not str(value("threshold")).strip() else float(value("threshold")),
            q_range=(float(value("q_min")), float(value("q_max"))), npt_1d=int(value("npt_1d")),
            output_path=value("output_path"), monitor_selector=MetadataColumnSelector(monitor) if monitor else None,
            backend="xu_hist" if value("backend") == "xrayutilities histogram" else "multigeometry",
            max_frame_bytes=int(value("max_frame_mib")) * 1024 * 1024,
        )

    def _start(self, action, begin):
        if self._closing:
            return
        try:
            identity = begin()
        except (TypeError, ValueError, OSError) as error:
            self._status(str(error), ready=False)
            return
        if identity is None:
            self._status("Finish the current operation or refresh Preview", ready=False)
            return
        self._action_revision = self._revision
        self._action_view_revision = self._view_revision
        self._started = time.monotonic()
        self._status(action)
        self._timer.start()
        self._sync_actions()

    def begin_preview(self):
        def begin():
            self._owner.set_form(self.build_form())
            self._prepared_revision = None
            return self._owner.begin_preflight()
        self._start("Preparing Preview…", begin)

    def begin_run(self):
        def begin():
            if self._prepared_revision != self._revision:
                return None
            self._owner.set_form(self.build_form())
            return self._owner.begin_run()
        self._start("Running · Stitch 1D", begin)

    def open_result(self, path=None):
        if path is None or isinstance(path, bool):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open stitched result", "", "NeXus (*.nexus)")
        if not path:
            return
        self._view_revision += 1
        self._start("Opening saved result…", lambda: self._owner.begin_reopen(path))

    def stop(self):
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
                self._status(update.failure_message or update.failure_type, ready=False)
            elif update.terminal_status is OperationTerminalStatus.CANCELLED:
                self._status("Cancelled")
            elif update.outcome is not None:
                self._accept_outcome(update)
        if not self._owner.busy:
            self._timer.stop()
        self._sync_actions()

    def _accept_outcome(self, update):
        outcome = update.outcome
        if outcome.kind is StitchOwnerOutcomeKind.PREFLIGHT_READY:
            if update.stale or self._action_revision != self._revision:
                self._status("Inputs changed · Preview required", ready=False)
                return
            self._prepared_revision = self._revision
            summary = outcome.preflight.summary
            StitchToolDialog._render_preflight(self, summary)
            self.preview_dialog.show()
            self._status(f"Ready · Stitch 1D · {len(summary.selected_labels)} frames")
        elif outcome.kind is StitchOwnerOutcomeKind.REOPENED:
            if self._action_view_revision == self._view_revision and not self._closing:
                self._mount_payload(outcome.artifact_path, outcome.payload)
                self._status("Ready · saved Stitch 1D result")
        elif outcome.kind is StitchOwnerOutcomeKind.RESULT:
            result = outcome.result
            if result.terminal.disposition is ModuleDisposition.COMMITTED:
                if not update.stale and self._action_revision == self._revision and not self._closing:
                    self._mount_payload(result.request.module.output.target, result.payload)
                elapsed = time.monotonic() - self._started
                self._status(f"Complete · Stitch 1D · {elapsed:.2f} s")
            else:
                self._status(f"{result.terminal.disposition.value}: {result.terminal.diagnostic or result.terminal.code}", ready=False)
        else:
            self._status(outcome.finalization_message or "Output retained · finalization retry required", ready=False)

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
        self._draw()

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
        self.curve.setLabel("bottom", "Q", units="Å⁻¹")
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
            self.curve.plot(payload.axis("q"), values + index * step,
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
        self._draw()

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
        idle = not busy and final is StitchOwnerFinalization.NONE and not self._closing
        self.controls.setEnabled(not busy and not self._closing)
        for row in self._rows.values():
            row.setEnabled(idle)
        if self._value("backend") == "xrayutilities histogram":
            for key in ("geometry_kind", "source_motors", "poni_references", "image_rotation",
                        "detector_rows", "detector_cols", "raw_dtype", "raw_header_skip", "threshold"):
                self._rows[key].setEnabled(False)
        elif self._value("geometry_kind") != "PONI + offsets":
            self._rows["poni_references"].setEnabled(False)
        self.advanced.setEnabled(idle)
        self.open_button.setEnabled(idle)
        self.preview_button.setEnabled(idle)
        self.run_bar.startButton.setEnabled(idle and self._prepared_revision == self._revision
                                            and self._owner.prepared is not None)
        self.run_bar.set_stop_enabled(busy and self._owner.current_identity.request.action is StitchOwnerAction.RUN)
        self.retry_button.setVisible(final is not StitchOwnerFinalization.NONE)
        self.retry_button.setEnabled(not busy)

    def active(self):
        return self._owner.busy or self._owner.finalization is not StitchOwnerFinalization.NONE or self._closing

    def shutdown(self):
        self._closing = True
        self._view_revision += 1
        self._timer.stop()
        receipt = self._owner.close()
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
