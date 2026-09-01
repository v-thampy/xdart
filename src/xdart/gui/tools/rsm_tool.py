"""Thin Qt presentation for the standalone, headless RSM operation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.pages.handle import PageHandle
from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.pages.values import PageCleanup, RSM_TOOL_KEY
from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector,
    ModuleDisposition,
)
from xrd_tools.analysis.rsm_operation import (
    RSMDetectorGeometry,
    RSMImageConditioning,
    RSMNormalizationMode,
    RSMNormalizationPolicy,
    RSMOperationPlan,
    RSMOperationResult,
)
from xrd_tools.core.geometry import DetectorHeader, ImageOrientation
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite

from .rsm_owner import (
    RSMOwnerAction,
    RSMOwnerFinalization,
    RSMOwnerOutcomeKind,
    RSMOwnerUpdate,
    RSMToolOwner,
)
from .rsm_values import RSMFrameSelector, RSMToolForm, rsm_tool_preset


logger = logging.getLogger(__name__)
_CURRENT_FORM_REVISION = object()
_PSIC_ROLES = ("mu", "eta", "chi", "phi", "nu", "del")


class RSMToolDialog(QtWidgets.QDialog):
    """Cached nonmodal RSM dialog; source and science I/O stay in its owner."""

    def __init__(self, services, parent=None, *, owner=None):
        super().__init__(parent)
        self._services = services
        self._owner = RSMToolOwner() if owner is None else owner
        self._prepared_form_fingerprint = None
        self._active_action = None
        self._form_revision = 0
        self._active_form_revision = None
        self._execution_form_revision = None
        self._painted_result_fingerprint = None
        self._shutdown_complete = False
        self._suppress_form_changes = False
        self.setObjectName("rsmToolDialog")
        self.setWindowTitle("Reciprocal Space Map")
        self.setModal(False)
        self.resize(1240, 920)
        self._build_ui()
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_owner)
        self._connect_form_changes()
        self._apply_scan43_preset()
        self._sync_actions()

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        input_group = QtWidgets.QGroupBox("Exact RSM request", self)
        self._input_group = input_group
        form = QtWidgets.QFormLayout(input_group)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self.project_edit, project_row = self._path_row(
            directory=True,
            caption="Choose Project root",
        )
        self.project_edit.setObjectName("rsmProject")
        form.addRow("Project", project_row)

        self.preset_button = QtWidgets.QPushButton("Apply STO_align scan 43 preset")
        self.preset_button.setObjectName("rsmScan43Preset")
        self.preset_button.setToolTip(
            "Applies filesystem-free notebook defaults. Preview still authenticates "
            "the SPEC file and every raw member."
        )
        form.addRow("Preset", self.preset_button)

        self.spec_edit, spec_row = self._path_row(
            caption="Choose extensionless SPEC file",
            file_filter="SPEC files (*);;All files (*)",
        )
        self.spec_edit.setObjectName("rsmSpec")
        form.addRow("SPEC", spec_row)

        self.scan_edit = QtWidgets.QLineEdit()
        self.scan_edit.setObjectName("rsmScan")
        self.scan_edit.setPlaceholderText("N or N.M")
        form.addRow("Scan", self.scan_edit)

        self.image_dir_edit, image_row = self._path_row(
            directory=True,
            caption="Choose raw-image directory",
        )
        self.image_dir_edit.setObjectName("rsmImageDirectory")
        form.addRow("Images", image_row)

        self.image_stem_edit = QtWidgets.QLineEdit()
        self.image_stem_edit.setObjectName("rsmImageStem")
        form.addRow("Image stem", self.image_stem_edit)

        decoder = QtWidgets.QWidget()
        decoder_layout = QtWidgets.QHBoxLayout(decoder)
        decoder_layout.setContentsMargins(0, 0, 0, 0)
        self.detector_rows = QtWidgets.QSpinBox()
        self.detector_rows.setObjectName("rsmDetectorRows")
        self.detector_rows.setRange(2, 1_000_000)
        self.detector_columns = QtWidgets.QSpinBox()
        self.detector_columns.setObjectName("rsmDetectorColumns")
        self.detector_columns.setRange(2, 1_000_000)
        self.raw_dtype_combo = QtWidgets.QComboBox()
        self.raw_dtype_combo.setObjectName("rsmRawDtype")
        self.raw_dtype_combo.addItems(("int32", "uint16", "float32", "float64"))
        self.raw_header_skip = QtWidgets.QSpinBox()
        self.raw_header_skip.setObjectName("rsmRawHeaderSkip")
        self.raw_header_skip.setRange(0, 1 << 30)
        for label, widget in (
            ("rows", self.detector_rows),
            ("columns", self.detector_columns),
            ("dtype", self.raw_dtype_combo),
            ("header bytes", self.raw_header_skip),
        ):
            decoder_layout.addWidget(QtWidgets.QLabel(label))
            decoder_layout.addWidget(widget)
        decoder_layout.addStretch(1)
        form.addRow("Raw decoder", decoder)

        frame_row = QtWidgets.QWidget()
        frame_layout = QtWidgets.QHBoxLayout(frame_row)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_start = QtWidgets.QSpinBox()
        self.frame_start.setObjectName("rsmFrameStart")
        self.frame_start.setRange(0, 2_147_483_647)
        self.frame_stop = QtWidgets.QLineEdit()
        self.frame_stop.setObjectName("rsmFrameStop")
        self.frame_stop.setMaximumWidth(100)
        self.frame_stop.setPlaceholderText("last")
        self.frame_step = QtWidgets.QSpinBox()
        self.frame_step.setObjectName("rsmFrameStep")
        self.frame_step.setRange(1, 2_147_483_647)
        for label, widget in (
            ("start", self.frame_start),
            ("stop", self.frame_stop),
            ("step", self.frame_step),
        ):
            frame_layout.addWidget(QtWidgets.QLabel(label))
            frame_layout.addWidget(widget)
        frame_layout.addStretch(1)
        form.addRow("Frame labels (inclusive)", frame_row)

        detector = QtWidgets.QWidget()
        detector_layout = QtWidgets.QGridLayout(detector)
        detector_layout.setContentsMargins(0, 0, 0, 0)
        self.cch1_edit = QtWidgets.QLineEdit()
        self.cch1_edit.setObjectName("rsmCch1")
        self.cch2_edit = QtWidgets.QLineEdit()
        self.cch2_edit.setObjectName("rsmCch2")
        self.pwidth1_edit = QtWidgets.QLineEdit()
        self.pwidth1_edit.setObjectName("rsmPixelWidth1")
        self.pwidth2_edit = QtWidgets.QLineEdit()
        self.pwidth2_edit.setObjectName("rsmPixelWidth2")
        self.distance_edit = QtWidgets.QLineEdit()
        self.distance_edit.setObjectName("rsmDistance")
        for column, (label, widget) in enumerate(
            (
                ("cch1", self.cch1_edit),
                ("cch2", self.cch2_edit),
                ("pwidth1", self.pwidth1_edit),
                ("pwidth2", self.pwidth2_edit),
                ("distance", self.distance_edit),
            )
        ):
            detector_layout.addWidget(QtWidgets.QLabel(label), 0, column)
            detector_layout.addWidget(widget, 1, column)
        form.addRow("psic detector", detector)

        roi = QtWidgets.QWidget()
        roi_layout = QtWidgets.QHBoxLayout(roi)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        self.roi_r0 = self._roi_spin("rsmRoiRowStart", 0)
        self.roi_r1 = self._roi_spin("rsmRoiRowStop", -1)
        self.roi_c0 = self._roi_spin("rsmRoiColumnStart", 0)
        self.roi_c1 = self._roi_spin("rsmRoiColumnStop", -1)
        for label, widget in (
            ("row start", self.roi_r0),
            ("row stop", self.roi_r1),
            ("column start", self.roi_c0),
            ("column stop", self.roi_c1),
        ):
            roi_layout.addWidget(QtWidgets.QLabel(label))
            roi_layout.addWidget(widget)
        roi_layout.addStretch(1)
        form.addRow("ROI", roi)

        selectors = QtWidgets.QWidget()
        selector_layout = QtWidgets.QGridLayout(selectors)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.addWidget(QtWidgets.QLabel("role"), 0, 0)
        selector_layout.addWidget(QtWidgets.QLabel("physical column"), 0, 1)
        selector_layout.addWidget(QtWidgets.QLabel("occurrence"), 0, 2)
        self.selector_edits = {}
        for row, role in enumerate((*_PSIC_ROLES, "foil", "exposure"), start=1):
            name = QtWidgets.QLineEdit()
            name.setObjectName(f"rsmSelector_{role}_name")
            occurrence = QtWidgets.QSpinBox()
            occurrence.setObjectName(f"rsmSelector_{role}_occurrence")
            occurrence.setRange(0, 1024)
            occurrence.setToolTip(
                "Zero-based physical occurrence: 0 is the first column with this name."
            )
            selector_layout.addWidget(QtWidgets.QLabel(role), row, 0)
            selector_layout.addWidget(name, row, 1)
            selector_layout.addWidget(occurrence, row, 2)
            self.selector_edits[role] = (name, occurrence)
        form.addRow("Exact SPEC selectors", selectors)

        conditioning = QtWidgets.QWidget()
        conditioning_layout = QtWidgets.QHBoxLayout(conditioning)
        conditioning_layout.setContentsMargins(0, 0, 0, 0)
        self.offset_edit = QtWidgets.QLineEdit()
        self.offset_edit.setObjectName("rsmAdditiveOffset")
        self.high_threshold_edit = QtWidgets.QLineEdit()
        self.high_threshold_edit.setObjectName("rsmHighThreshold")
        self.high_threshold_edit.setPlaceholderText("blank = none")
        self.static_hot_edit = QtWidgets.QLineEdit()
        self.static_hot_edit.setObjectName("rsmStaticHotThreshold")
        self.static_hot_edit.setPlaceholderText("blank = none")
        for label, widget in (
            ("offset", self.offset_edit),
            ("high → NaN", self.high_threshold_edit),
            ("all-frame hot", self.static_hot_edit),
        ):
            conditioning_layout.addWidget(QtWidgets.QLabel(label))
            conditioning_layout.addWidget(widget)
        conditioning_layout.addStretch(1)
        form.addRow("Conditioning", conditioning)

        absorption = QtWidgets.QWidget()
        absorption_layout = QtWidgets.QHBoxLayout(absorption)
        absorption_layout.setContentsMargins(0, 0, 0, 0)
        self.absorption_edits = []
        for index in range(4):
            edit = QtWidgets.QLineEdit()
            edit.setObjectName(f"rsmAbsorption{index}")
            self.absorption_edits.append(edit)
            absorption_layout.addWidget(QtWidgets.QLabel(str(index + 1)))
            absorption_layout.addWidget(edit)
        absorption_layout.addStretch(1)
        form.addRow("Foil absorption lengths", absorption)

        execution = QtWidgets.QWidget()
        execution_layout = QtWidgets.QHBoxLayout(execution)
        execution_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_combo = QtWidgets.QComboBox()
        self.grid_combo.setObjectName("rsmGrid")
        self.grid_combo.addItem("Quick 40³", (40, 40, 40))
        self.grid_combo.addItem("Full 200³", (200, 200, 200))
        self.chunk_size = QtWidgets.QSpinBox()
        self.chunk_size.setObjectName("rsmChunkSize")
        self.chunk_size.setRange(1, 1024)
        self.max_frame_mib = QtWidgets.QSpinBox()
        self.max_frame_mib.setObjectName("rsmMaxFrameMiB")
        self.max_frame_mib.setRange(1, 4096)
        self.max_chunk_mib = QtWidgets.QSpinBox()
        self.max_chunk_mib.setObjectName("rsmMaxChunkMiB")
        self.max_chunk_mib.setRange(1, 4096)
        for label, widget in (
            ("grid", self.grid_combo),
            ("chunk", self.chunk_size),
            ("max frame MiB", self.max_frame_mib),
            ("max chunk MiB", self.max_chunk_mib),
        ):
            execution_layout.addWidget(QtWidgets.QLabel(label))
            execution_layout.addWidget(widget)
        execution_layout.addStretch(1)
        form.addRow("Grid / memory", execution)

        self.output_edit, output_row = self._path_row(
            save=True,
            caption="Choose RSM output",
            file_filter="NeXus artifact (*.nexus)",
        )
        self.output_edit.setObjectName("rsmOutput")
        form.addRow("Output", output_row)

        self.overwrite_combo = QtWidgets.QComboBox()
        self.overwrite_combo.setObjectName("rsmOverwrite")
        self.overwrite_combo.addItem(
            "Create new (refuse if present)",
            AnalysisArtifactOverwrite.CREATE_NEW,
        )
        self.overwrite_combo.addItem(
            "Replace transactionally",
            AnalysisArtifactOverwrite.REPLACE,
        )
        form.addRow("Output policy", self.overwrite_combo)

        hold = QtWidgets.QLabel(
            "R1: one SPEC scan · canonical psic · identity orientation · exact "
            "source-derived q bounds. Held: GI/refraction, multi-scan RSM, and "
            "volume rendering. Output publication assumes an operator-owned "
            "Project tree; hostile/shared namespace mutation is held."
        )
        hold.setObjectName("rsmBoundary")
        hold.setWordWrap(True)
        form.addRow("Scientific boundary", hold)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(input_group)
        scroll.setMaximumHeight(520)
        outer.addWidget(scroll)

        buttons = QtWidgets.QHBoxLayout()
        self.preview_button = QtWidgets.QPushButton("Preview")
        self.preview_button.setObjectName("rsmPreview")
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setObjectName("rsmRun")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setObjectName("rsmCancel")
        self.retry_cleanup_button = QtWidgets.QPushButton("Retry Cleanup")
        self.retry_cleanup_button.setObjectName("rsmRetryCleanup")
        self.retry_verification_button = QtWidgets.QPushButton(
            "Retry Verification"
        )
        self.retry_verification_button.setObjectName("rsmRetryVerification")
        for button in (
            self.preview_button,
            self.run_button,
            self.cancel_button,
            self.retry_cleanup_button,
            self.retry_verification_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setObjectName("rsmProgress")
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        outer.addWidget(self.progress)

        detail = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.preview_text = QtWidgets.QPlainTextEdit()
        self.preview_text.setObjectName("rsmPreflightSummary")
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText(
            "Preview lists every exact raw member and source-derived science fact."
        )
        detail.addWidget(self.preview_text)

        plots = QtWidgets.QWidget()
        plots_layout = QtWidgets.QGridLayout(plots)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        self.slice_plots = []
        self.slice_images = []
        for index, (title, horizontal, vertical) in enumerate(
            (
                ("H-normal central slice", "L", "K"),
                ("K-normal central slice", "L", "H"),
                ("L-normal central slice", "K", "H"),
            )
        ):
            plot = pg.PlotWidget()
            plot.setObjectName(f"rsmSlicePlot{index}")
            plot.setTitle(title)
            plot.setLabel("bottom", horizontal)
            plot.setLabel("left", vertical)
            image = pg.ImageItem(axisOrder="row-major")
            plot.addItem(image)
            self.slice_plots.append(plot)
            self.slice_images.append(image)
            plots_layout.addWidget(plot, index // 2, index % 2)
        self.result_facts = QtWidgets.QLabel("No committed RSM loaded")
        self.result_facts.setObjectName("rsmResultFacts")
        self.result_facts.setWordWrap(True)
        plots_layout.addWidget(self.result_facts, 1, 1)
        detail.addWidget(plots)
        detail.setStretchFactor(0, 1)
        detail.setStretchFactor(1, 2)
        outer.addWidget(detail, 1)

        self.status_label = QtWidgets.QLabel("Ready for an exact Preview")
        self.status_label.setObjectName("rsmStatus")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        self.preset_button.clicked.connect(self._apply_scan43_preset)
        self.preview_button.clicked.connect(self._begin_preflight)
        self.run_button.clicked.connect(self._begin_run)
        self.cancel_button.clicked.connect(self._cancel)
        self.retry_cleanup_button.clicked.connect(self._begin_retry_cleanup)
        self.retry_verification_button.clicked.connect(
            self._begin_retry_verification
        )

    @staticmethod
    def _roi_spin(name, value):
        widget = QtWidgets.QSpinBox()
        widget.setObjectName(name)
        widget.setRange(-1, 1_000_000)
        widget.setValue(value)
        return widget

    def _path_row(self, *, directory=False, save=False, caption, file_filter=""):
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QtWidgets.QLineEdit()
        button = QtWidgets.QPushButton("Choose…")
        layout.addWidget(edit, 1)
        layout.addWidget(button)

        def choose():
            from xdart.utils.browse import browse_start_dir, remember_browse_path

            start = browse_start_dir(edit.text())
            if directory:
                selected = QtWidgets.QFileDialog.getExistingDirectory(
                    self, caption, start
                )
            elif save:
                selected, _chosen = QtWidgets.QFileDialog.getSaveFileName(
                    self, caption, start, file_filter
                )
            else:
                selected, _chosen = QtWidgets.QFileDialog.getOpenFileName(
                    self, caption, start, file_filter
                )
            if selected:
                remember_browse_path(selected)
                edit.setText(selected)

        button.clicked.connect(choose)
        return edit, row

    def _connect_form_changes(self):
        line_edits = (
            self.project_edit,
            self.spec_edit,
            self.scan_edit,
            self.image_dir_edit,
            self.image_stem_edit,
            self.frame_stop,
            self.cch1_edit,
            self.cch2_edit,
            self.pwidth1_edit,
            self.pwidth2_edit,
            self.distance_edit,
            self.offset_edit,
            self.high_threshold_edit,
            self.static_hot_edit,
            self.output_edit,
            *(item[0] for item in self.selector_edits.values()),
            *self.absorption_edits,
        )
        for edit in line_edits:
            edit.textChanged.connect(self._form_changed)
        spins = (
            self.detector_rows,
            self.detector_columns,
            self.raw_header_skip,
            self.frame_start,
            self.frame_step,
            self.roi_r0,
            self.roi_r1,
            self.roi_c0,
            self.roi_c1,
            self.chunk_size,
            self.max_frame_mib,
            self.max_chunk_mib,
            *(item[1] for item in self.selector_edits.values()),
        )
        for spin in spins:
            spin.valueChanged.connect(self._form_changed)
        for combo in (
            self.raw_dtype_combo,
            self.grid_combo,
            self.overwrite_combo,
        ):
            combo.currentIndexChanged.connect(self._form_changed)

    def _form_changed(self, *_args):
        if self._suppress_form_changes:
            return
        self._form_revision += 1
        if self._painted_result_fingerprint is not None:
            self._clear_painted_result("Result cleared because inputs changed")
            self.status_label.setText("Inputs changed · previous result cleared")
        elif self._prepared_form_fingerprint is not None:
            self.status_label.setText("Inputs changed · run Preview again")
        self._prepared_form_fingerprint = None
        self.run_button.setEnabled(False)

    def _apply_scan43_preset(self):
        preset = rsm_tool_preset()
        self._suppress_form_changes = True
        try:
            root_text = self.project_edit.text().strip()
            if root_text:
                root = Path(root_text)
                self.spec_edit.setText(str(root / preset.spec_relative_path))
                self.image_dir_edit.setText(
                    str(root / preset.image_directory_relative_path)
                )
                if not self.output_edit.text().strip():
                    self.output_edit.setText(str(root / "rsm_scan43.nexus"))
            else:
                self.spec_edit.setText(preset.spec_relative_path)
                self.image_dir_edit.setText(preset.image_directory_relative_path)
                if not self.output_edit.text().strip():
                    self.output_edit.setText("rsm_scan43.nexus")
            self.scan_edit.setText(preset.scan)
            self.image_stem_edit.setText(preset.image_stem)
            self.detector_rows.setValue(preset.detector_shape[0])
            self.detector_columns.setValue(preset.detector_shape[1])
            dtype_index = self.raw_dtype_combo.findText("int32")
            self.raw_dtype_combo.setCurrentIndex(max(0, dtype_index))
            self.raw_header_skip.setValue(preset.raw_header_skip)
            self.frame_start.setValue(preset.frame_selector.start_label)
            self.frame_stop.setText(str(preset.frame_selector.stop_label))
            self.frame_step.setValue(preset.frame_selector.step)
            header = preset.header
            for edit, value in (
                (self.cch1_edit, header.cch1),
                (self.cch2_edit, header.cch2),
                (self.pwidth1_edit, header.pwidth1),
                (self.pwidth2_edit, header.pwidth2),
                (self.distance_edit, header.distance),
            ):
                edit.setText(f"{value:.12g}")
            for widget, value in zip(
                (self.roi_r0, self.roi_r1, self.roi_c0, self.roi_c1),
                preset.roi,
            ):
                widget.setValue(value)
            selectors = dict(preset.motor_selectors)
            selectors["foil"] = preset.normalization.foil_selector
            selectors["exposure"] = preset.normalization.exposure_selector
            for role, selector in selectors.items():
                name, occurrence = self.selector_edits[role]
                name.setText(selector.name)
                occurrence.setValue(selector.occurrence)
            self.offset_edit.setText(
                f"{preset.conditioning.additive_offset:.12g}"
            )
            self.high_threshold_edit.setText(
                f"{preset.conditioning.high_threshold:.12g}"
            )
            self.static_hot_edit.setText(
                f"{preset.conditioning.static_hot_threshold:.12g}"
            )
            for edit, value in zip(
                self.absorption_edits,
                preset.normalization.absorption_lengths,
            ):
                edit.setText(f"{value:.12g}")
            self.grid_combo.setCurrentIndex(0)
            self.chunk_size.setValue(preset.chunk_size)
            self.max_frame_mib.setValue(
                preset.max_frame_bytes // (1024 * 1024)
            )
            self.max_chunk_mib.setValue(
                preset.max_chunk_bytes // (1024 * 1024)
            )
        finally:
            self._suppress_form_changes = False
        self._form_changed()

    @staticmethod
    def _optional_float(text):
        value = str(text).strip()
        return None if not value else float(value)

    def _selector(self, role):
        name, occurrence = self.selector_edits[role]
        return MetadataColumnSelector(name.text().strip(), occurrence.value())

    def _build_form(self):
        project_text = self.project_edit.text().strip()
        if not project_text:
            raise ValueError("Project root is required")
        project_path = Path(project_text).expanduser()

        def in_project(text, name):
            spelling = str(text).strip()
            if not spelling:
                raise ValueError(f"{name} is required")
            path = Path(spelling).expanduser()
            return path if path.is_absolute() else project_path / path

        rows = self.detector_rows.value()
        columns = self.detector_columns.value()
        header = DetectorHeader(
            float(self.cch1_edit.text()),
            float(self.cch2_edit.text()),
            float(self.pwidth1_edit.text()),
            float(self.pwidth2_edit.text()),
            float(self.distance_edit.text()),
            rows,
            columns,
        )
        geometry = RSMDetectorGeometry(
            header,
            tuple((role, self._selector(role)) for role in _PSIC_ROLES),
            ImageOrientation(),
            (
                self.roi_r0.value(),
                self.roi_r1.value(),
                self.roi_c0.value(),
                self.roi_c1.value(),
            ),
        )
        conditioning = RSMImageConditioning(
            float(self.offset_edit.text()),
            self._optional_float(self.high_threshold_edit.text()),
            self._optional_float(self.static_hot_edit.text()),
        )
        normalization = RSMNormalizationPolicy(
            RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE,
            self._selector("foil"),
            self._selector("exposure"),
            tuple(float(edit.text()) for edit in self.absorption_edits),
        )
        plan = RSMOperationPlan(
            geometry,
            conditioning,
            normalization,
            bins=tuple(self.grid_combo.currentData()),
            chunk_size=self.chunk_size.value(),
            max_frame_bytes=self.max_frame_mib.value() * 1024 * 1024,
            max_chunk_bytes=self.max_chunk_mib.value() * 1024 * 1024,
        )
        stop_text = self.frame_stop.text().strip()
        return RSMToolForm(
            project_text,
            in_project(self.spec_edit.text(), "SPEC path"),
            self.scan_edit.text().strip(),
            in_project(self.image_dir_edit.text(), "image directory"),
            self.image_stem_edit.text().strip(),
            RSMFrameSelector(
                self.frame_start.value(),
                None if not stop_text else int(stop_text),
                self.frame_step.value(),
            ),
            (rows, columns),
            self.raw_dtype_combo.currentText(),
            plan,
            in_project(self.output_edit.text(), "output path"),
            self.raw_header_skip.value(),
            (
                AnalysisArtifactOverwrite.CREATE_NEW,
                AnalysisArtifactOverwrite.REPLACE,
            )[self.overwrite_combo.currentIndex()],
        )

    def _current_form(self):
        try:
            return self._build_form()
        except (TypeError, ValueError, OverflowError) as error:
            self._notice(f"Invalid RSM request: {error}")
            return None

    def _notice(self, text):
        message = str(text)
        self.status_label.setText(message)
        try:
            self._services.status.show(message, 8000)
        except Exception:
            logger.debug("RSM status presenter failed", exc_info=True)

    def _begin_preflight(self):
        form = self._current_form()
        if form is None:
            return
        self._owner.set_form(form)
        identity = self._owner.begin_preflight()
        if identity is None:
            self._notice("Preview refused while another RSM action is active")
            return
        if self._painted_result_fingerprint is not None:
            self._clear_painted_result("Result cleared for a new Preview")
        self._prepared_form_fingerprint = None
        self._begin_polling(RSMOwnerAction.PREFLIGHT, "Preparing exact preview…")

    def _begin_run(self):
        form = self._current_form()
        if form is None:
            return
        self._owner.set_form(form)
        identity = self._owner.begin_run()
        if identity is None:
            self._notice("Run requires an unchanged successful Preview")
            self._prepared_form_fingerprint = None
            self._sync_actions()
            return
        self._execution_form_revision = self._form_revision
        self._begin_polling(RSMOwnerAction.RUN, "Running RSM…")

    def _begin_retry_cleanup(self):
        identity = self._owner.begin_retry_cleanup()
        if identity is None:
            self._notice("No retryable RSM cleanup is available")
            return
        self._begin_polling(
            RSMOwnerAction.RETRY_CLEANUP,
            "Retrying cleanup only…",
            form_revision=self._execution_form_revision,
        )

    def _begin_retry_verification(self):
        identity = self._owner.begin_retry_verification()
        if identity is None:
            self._notice("No retryable RSM verification is available")
            return
        self._begin_polling(
            RSMOwnerAction.RETRY_VERIFICATION,
            "Retrying strict reload only…",
            form_revision=self._execution_form_revision,
        )

    def _begin_polling(
        self,
        action,
        message,
        *,
        form_revision=_CURRENT_FORM_REVISION,
    ):
        self._active_action = action
        self._active_form_revision = (
            self._form_revision
            if form_revision is _CURRENT_FORM_REVISION
            else form_revision
        )
        self.progress.setRange(0, 0)
        self.status_label.setText(message)
        self._poll_timer.start()
        self._sync_actions()

    def _cancel(self):
        if self._owner.cancel():
            self.status_label.setText("Cancellation requested…")

    def _poll_owner(self):
        try:
            update = self._owner.poll()
            if update is not None:
                self._accept_update(update)
        except Exception as error:
            logger.exception("RSM owner update failed")
            self._notice(f"RSM owner or presentation failed: {error}")
            self._poll_timer.stop()
            self._sync_actions()
            return
        if not self._owner.busy:
            self._poll_timer.stop()
            self._active_action = None
            self._active_form_revision = None
            self._sync_actions()

    def _accept_update(self, update):
        if type(update) is not RSMOwnerUpdate:
            raise TypeError("RSM dialog requires an exact owner update")
        if update.progress is not None:
            progress = update.progress
            self.progress.setRange(0, progress.total)
            self.progress.setValue(progress.completed)
            self.status_label.setText(
                f"{progress.stage} · {progress.completed}/{progress.total}"
            )
            return
        if update.terminal_status is OperationTerminalStatus.FAILED:
            detail = update.failure_message or update.failure_type
            self._notice(f"{update.action.value} failed: {detail}")
            return
        if update.terminal_status is OperationTerminalStatus.CANCELLED:
            self.status_label.setText("RSM cancelled")
            return
        outcome = update.outcome
        if outcome is None:
            raise RuntimeError("returned RSM update omitted its outcome")
        if outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_READY:
            preflight = outcome.preflight
            if (
                update.stale
                or self._active_form_revision != self._form_revision
                or preflight is None
            ):
                self._prepared_form_fingerprint = None
                self.status_label.setText("Preview completed, but inputs changed")
                return
            current = self._current_form()
            if current is None or not preflight.is_current(current):
                self._prepared_form_fingerprint = None
                self.status_label.setText("Preview completed, but inputs changed")
                return
            self._prepared_form_fingerprint = preflight.form.fingerprint
            self._render_preflight(preflight.summary)
            self.status_label.setText(
                f"Preview ready · {len(preflight.summary.selected_labels)} exact frames"
            )
            return
        if outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_REFUSED:
            detail = outcome.refusal_message or outcome.refusal_code
            self._notice(f"Preview refused: {detail}")
            return
        if outcome.kind is RSMOwnerOutcomeKind.CLEANUP_PENDING:
            self._notice("Output cleanup needs retry; science will not rerun")
            return
        if outcome.kind is RSMOwnerOutcomeKind.VERIFICATION_PENDING:
            self._notice("Output committed; strict reload needs verification retry")
            return
        result = outcome.result
        if outcome.kind is not RSMOwnerOutcomeKind.RESULT or result is None:
            raise RuntimeError("RSM owner returned an invalid outcome")
        terminal = result.terminal
        if terminal.disposition is ModuleDisposition.COMMITTED:
            if update.stale or self._active_form_revision != self._form_revision:
                self._notice(
                    "RSM committed, but inputs changed; the exact output was "
                    "retained and was not painted"
                )
                return
            self._paint_result(result)
            self.status_label.setText(
                f"Committed {Path(result.request.module.output.target).name}"
            )
        elif terminal.disposition is ModuleDisposition.CANCELLED:
            self.status_label.setText("RSM cancelled")
        elif terminal.disposition is ModuleDisposition.REFUSED:
            self._notice(f"RSM refused: {terminal.code}")
        else:
            self._notice(
                f"RSM failed: {terminal.code}"
                + (f" · {terminal.diagnostic}" if terminal.diagnostic else "")
            )

    def _render_preflight(self, summary):
        selectors = ", ".join(
            f"{item.name}[{item.occurrence}]" for item in summary.required_selectors
        )
        ub = "; ".join(
            " ".join(f"{value:.10g}" for value in row) for row in summary.ub
        )
        bounds = ", ".join(
            f"{name} {low:.10g}..{high:.10g}"
            for name, (low, high) in zip(("H", "K", "L"), summary.q_bounds)
        )
        offset, high, static = summary.conditioning
        lines = [
            f"Project: {summary.project_root}",
            f"Source: {summary.source_relative_path} · scan {summary.source_scan}",
            f"Images: {summary.image_directory_relative_path} · stem {summary.image_stem}",
            f"Frames: {len(summary.selected_labels)} · labels {summary.selected_labels}",
            (
                "Raw decoder: "
                f"{summary.detector_shape[0]}×{summary.detector_shape[1]} · "
                f"{summary.raw_dtype} · header {summary.raw_header_skip} bytes · "
                "source threshold none · source rotation 0°"
            ),
            f"Output: {summary.output_relative_path} ({summary.overwrite.value})",
            f"Required physical columns: {selectors}",
            f"Energy: {summary.energy_eV:.12g} eV",
            f"UB: {ub}",
            f"Exact q bounds: {bounds}",
            f"Grid: {summary.bins} · cropped detector {summary.cropped_shape}",
            (
                f"Conditioning: +{offset:.10g} · high {high} · "
                f"all-frame hot {static}"
            ),
            (
                f"Normalization: {summary.normalization_mode.value} · "
                f"absorption {summary.absorption_lengths} · numerator before grid"
            ),
            (
                f"Memory: chunk {summary.chunk_size} · frame "
                f"{summary.max_frame_bytes / (1024 * 1024):.3f} MiB · chunk "
                f"{summary.max_chunk_bytes / (1024 * 1024):.3f} MiB"
            ),
            f"Request fingerprint: {summary.request_fingerprint}",
            f"Preflight fingerprint: {summary.preflight_fingerprint}",
            f"Plan fingerprint: {summary.plan_fingerprint}",
            "Holds: " + ", ".join(summary.holds),
            "",
            "Exact raw membership and divisors:",
        ]
        for member in summary.members:
            values = ", ".join(
                f"{name}[{occurrence}]={value:.10g}"
                for name, occurrence, value in member.values
            )
            lines.append(
                f"  {member.label}: {member.relative_path}"
                f"#{member.source_frame_index} · divisor "
                f"{member.normalization_divisor:.12g} · {values}"
            )
        lines.extend(("", "Exact dependency files:"))
        lines.extend(f"  {path}" for path in summary.dependency_files)
        self.preview_text.setPlainText("\n".join(lines))

    @staticmethod
    def _axis_edges(values):
        axis = np.asarray(values, dtype=np.float64)
        if (
            axis.ndim != 1
            or axis.size < 2
            or not np.all(np.isfinite(axis))
            or not np.all(np.diff(axis) > 0)
        ):
            raise ValueError("strict RSM display axis is invalid")
        return (
            float(axis[0] - (axis[1] - axis[0]) / 2),
            float(axis[-1] + (axis[-1] - axis[-2]) / 2),
        )

    def _clear_painted_result(self, detail="No committed RSM loaded"):
        self.setUpdatesEnabled(False)
        try:
            for image in self.slice_images:
                image.clear()
            self.result_facts.setText(str(detail))
            self._painted_result_fingerprint = None
        finally:
            self.setUpdatesEnabled(True)
            self.update()

    def _paint_result(self, result):
        if type(result) is not RSMOperationResult:
            raise TypeError("RSM painter requires an exact operation result")
        if (
            result.terminal.disposition is not ModuleDisposition.COMMITTED
            or result.payload is None
        ):
            raise ValueError("RSM painter accepts only a strict committed payload")
        payload = result.payload
        intensity = payload.intensity
        if intensity.ndim != 3:
            raise ValueError("strict RSM payload is not three-dimensional")
        middle = tuple(size // 2 for size in intensity.shape)
        axes = tuple(payload.axis(name) for name in ("h", "k", "l"))
        presentations = (
            (intensity[middle[0], :, :], axes[2], axes[1]),
            (intensity[:, middle[1], :], axes[2], axes[0]),
            (intensity[:, :, middle[2]], axes[1], axes[0]),
        )
        staged = []
        for values, horizontal, vertical in presentations:
            x0, x1 = self._axis_edges(horizontal)
            y0, y1 = self._axis_edges(vertical)
            image = pg.ImageItem(axisOrder="row-major")
            image.setImage(np.asarray(values), autoLevels=True)
            image.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
            staged.append(image)
        facts = (
            f"Shape {intensity.shape} · finite voxels "
            f"{int(np.isfinite(intensity).sum())} · "
            f"H {axes[0][0]:.8g}..{axes[0][-1]:.8g} · "
            f"K {axes[1][0]:.8g}..{axes[1][-1]:.8g} · "
            f"L {axes[2][0]:.8g}..{axes[2][-1]:.8g}"
        )
        old_images = tuple(self.slice_images)
        old_facts = self.result_facts.text()
        old_fingerprint = self._painted_result_fingerprint
        self.setUpdatesEnabled(False)
        added = []
        removed = []
        try:
            for plot, image in zip(self.slice_plots, staged):
                plot.addItem(image)
                added.append((plot, image))
            for plot, image in zip(self.slice_plots, old_images):
                plot.removeItem(image)
                removed.append((plot, image))
            self.result_facts.setText(facts)
            self.slice_images = staged
            self._painted_result_fingerprint = payload.result_fingerprint
        except BaseException as primary:
            rollback_errors = []
            for plot, image in reversed(added):
                try:
                    plot.removeItem(image)
                except BaseException as error:
                    rollback_errors.append(error)
            for plot, image in reversed(removed):
                try:
                    plot.addItem(image)
                except BaseException as error:
                    rollback_errors.append(error)
            self.slice_images = list(old_images)
            self._painted_result_fingerprint = old_fingerprint
            try:
                self.result_facts.setText(old_facts)
            except BaseException as error:
                rollback_errors.append(error)
            for error in rollback_errors:
                try:
                    primary.add_note(
                        "RSM presentation rollback also failed: "
                        f"{type(error).__module__}.{type(error).__qualname__}: "
                        f"{error}"
                    )
                except BaseException:
                    break
            raise primary.with_traceback(primary.__traceback__)
        finally:
            self.setUpdatesEnabled(True)
            self.update()

    def _sync_actions(self):
        busy = bool(self._owner.busy)
        finalization = self._owner.finalization
        current = self._owner.form
        prepared = self._owner.prepared
        current_preview = (
            not busy
            and finalization is RSMOwnerFinalization.NONE
            and prepared is not None
            and current is not None
            and prepared.is_current(current)
            and self._prepared_form_fingerprint == current.fingerprint
        )
        self._input_group.setEnabled(not busy)
        self.preview_button.setEnabled(
            not busy and finalization is RSMOwnerFinalization.NONE
        )
        self.run_button.setEnabled(current_preview)
        self.cancel_button.setEnabled(
            busy
            and self._active_action in {RSMOwnerAction.PREFLIGHT, RSMOwnerAction.RUN}
        )
        self.retry_cleanup_button.setEnabled(
            not busy and finalization is RSMOwnerFinalization.CLEANUP_PENDING
        )
        self.retry_verification_button.setEnabled(
            not busy and finalization is RSMOwnerFinalization.VERIFICATION_PENDING
        )

    def active(self):
        return bool(
            self._owner.busy
            or self._owner.finalization is not RSMOwnerFinalization.NONE
        )

    def shutdown(self):
        if self._shutdown_complete:
            from xdart.gui.pages.values import CloseReceipt

            return CloseReceipt(PageCleanup.CLEAN)
        self._poll_timer.stop()
        receipt = self._owner.close()
        if receipt.status is PageCleanup.CLEAN:
            self.setUpdatesEnabled(False)
            try:
                for image in self.slice_images:
                    image.clear()
                self.preview_text.clear()
                self.result_facts.setText("No committed RSM loaded")
                self._painted_result_fingerprint = None
            finally:
                self.setUpdatesEnabled(True)
            self._shutdown_complete = True
        return receipt


def build_rsm_tool(services, parent):
    dialog = RSMToolDialog(services, parent)
    return PageHandle(
        key=RSM_TOOL_KEY,
        widget=dialog,
        close=dialog.shutdown,
        activity=dialog,
    )


__all__ = ["RSMToolDialog", "build_rsm_tool"]
