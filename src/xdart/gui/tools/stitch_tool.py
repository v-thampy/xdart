"""Thin Qt presentation for the standalone, headless Stitch operation."""

from __future__ import annotations

import logging
from pathlib import Path
import re

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.analysis.scan_source_widget import ScanSourceWidget
from xdart.gui.pages.handle import PageHandle
from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.pages.values import PageCleanup, STITCH_TOOL_KEY
from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector,
    ModuleDisposition,
)
from xrd_tools.analysis.stitch_operation import (
    StitchGeometryKind,
    StitchOperationResult,
)
from xrd_tools.analysis.xu_stitch_calibration import (
    CANONICAL_XU_STITCH_CALIBRATION_LOCATOR,
    XuStitchCalibrationRefused,
    install_canonical_xu_stitch_calibration,
)
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite

from .stitch_owner import (
    StitchOwnerAction,
    StitchOwnerFinalization,
    StitchOwnerOutcomeKind,
    StitchOwnerUpdate,
    StitchToolOwner,
)
from .stitch_values import StitchFrameSelector, StitchToolForm


logger = logging.getLogger(__name__)
_PAIR_SEPARATOR = re.compile(r"[,;\n]+")
_CURRENT_FORM_REVISION = object()


class StitchToolDialog(QtWidgets.QDialog):
    """Cached nonmodal Stitch dialog; all source/science I/O stays in its owner."""

    def __init__(self, services, parent=None, *, owner=None):
        super().__init__(parent)
        self._services = services
        self._owner = StitchToolOwner() if owner is None else owner
        self._prepared_form_fingerprint = None
        self._active_action = None
        self._form_revision = 0
        self._active_form_revision = None
        self._execution_form_revision = None
        self._painted_result_fingerprint = None
        self._shutdown_complete = False
        self._suppress_form_changes = False
        self.setObjectName("stitchToolDialog")
        self.setWindowTitle("Stitching")
        self.setModal(False)
        self.resize(1120, 880)
        self._build_ui()
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_owner)
        self._connect_form_changes()
        self._sync_actions()

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        input_group = QtWidgets.QGroupBox("Exact Stitch request", self)
        self._input_group = input_group
        form = QtWidgets.QFormLayout(input_group)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self.project_edit, project_row = self._path_row(
            directory=True, caption="Choose Project root"
        )
        self.project_edit.setObjectName("stitchProject")
        self.project_edit.setToolTip(
            "The SPEC file, raw members, geometry, and output must resolve inside Project."
        )
        form.addRow("Project", project_row)

        self.source_widget = ScanSourceWidget(mode="vnext_analysis", parent=self)
        self.source_widget.setObjectName("stitchSource")
        self.source_widget.dir_check.setChecked(False)
        self.source_widget.dir_check.setVisible(False)
        self.source_widget.dir_kind_combo.setVisible(False)
        self.source_widget.kind_label.setText("SPEC")
        self.source_widget.kind_label.setVisible(True)
        self.source_widget.group_row.setVisible(False)
        self.source_widget.raw_dot.setText("○ checked during Preview")
        self.source_widget.raw_dot.setToolTip(
            "Preview admits metadata and exact raw locators without decoding a detector frame."
        )
        self.source_widget.adv_btn.setChecked(True)
        self.source_widget.det_rows.setText("195")
        self.source_widget.det_cols.setText("1475")
        dtype_index = self.source_widget.dtype_combo.findText("int32")
        if dtype_index >= 0:
            self.source_widget.dtype_combo.setCurrentIndex(dtype_index)
        form.addRow("SPEC source", self.source_widget)

        self.scan_edit = QtWidgets.QLineEdit("14")
        self.scan_edit.setObjectName("stitchScan")
        self.scan_edit.setPlaceholderText("N or N.M")
        form.addRow("Scan", self.scan_edit)

        self.threshold_edit = QtWidgets.QLineEdit("800000")
        self.threshold_edit.setObjectName("stitchThreshold")
        self.threshold_edit.setPlaceholderText("blank = no threshold")
        form.addRow("Hot-pixel threshold", self.threshold_edit)

        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.setObjectName("stitchBackend")
        self.backend_combo.addItem(
            "pyFAI MultiGeometry",
            "multigeometry",
        )
        self.backend_combo.addItem(
            "xrayutilities histogram (calibrated multi-axis reference)",
            "xu_hist",
        )
        self.backend_combo.setToolTip(
            "Use xrayutilities histogram for the authenticated SURFACE calibrated "
            "multi-axis path; use pyFAI MultiGeometry for validated pyFAI goniometers."
        )
        form.addRow("Backend", self.backend_combo)

        self.geometry_edit, geometry_row = self._path_row(
            caption="Choose Stitch geometry",
            file_filter="Geometry (*.json *.poni);;All files (*)",
        )
        self.geometry_edit.setObjectName("stitchGeometry")
        self.install_xu_asset_button = QtWidgets.QPushButton("Install canonical")
        self.install_xu_asset_button.setObjectName("stitchInstallXuAsset")
        self.install_xu_asset_button.setToolTip(
            "Create the exact bundled SURFACE v1 calibration inside this Project. "
            "Existing different bytes are never replaced."
        )
        geometry_row.layout().addWidget(self.install_xu_asset_button)
        self.geometry_label = QtWidgets.QLabel("Geometry")
        form.addRow(self.geometry_label, geometry_row)

        self.geometry_kind_combo = QtWidgets.QComboBox()
        self.geometry_kind_combo.setObjectName("stitchGeometryKind")
        self.geometry_kind_combo.addItem(
            "Fitted pyFAI goniometer JSON",
            StitchGeometryKind.PYFAI_GONIOMETER_JSON,
        )
        self.geometry_kind_combo.addItem("PONI + motor offsets", StitchGeometryKind.PONI)
        form.addRow("Geometry kind", self.geometry_kind_combo)

        self.geometry_sha_edit = QtWidgets.QLineEdit()
        self.geometry_sha_edit.setObjectName("stitchGeometrySha256")
        self.geometry_sha_edit.setPlaceholderText("optional expected SHA-256")
        form.addRow("Geometry SHA-256", self.geometry_sha_edit)

        self.motor_mapping_edit = QtWidgets.QLineEdit(
            "del_angle=del, nu_angle=nu"
        )
        self.motor_mapping_edit.setObjectName("stitchMotorMapping")
        self.motor_mapping_edit.setToolTip(
            "One-to-one geometry-position=SPEC-column mappings. Every fitted JSON "
            "position must be bound exactly once."
        )
        form.addRow("Motor mapping", self.motor_mapping_edit)

        self.poni_reference_edit = QtWidgets.QLineEdit("del=0, nu=0")
        self.poni_reference_edit.setObjectName("stitchPoniReferences")
        form.addRow("PONI reference positions", self.poni_reference_edit)

        self.rotation_combo = QtWidgets.QComboBox()
        self.rotation_combo.setObjectName("stitchImageRotation")
        for rotation in (0, 90, 180, 270):
            self.rotation_combo.addItem(f"{rotation}°", rotation)
        self.rotation_combo.setToolTip(
            "Recorded Stitch-owned raw-array rotation. The source itself stays unrotated."
        )
        form.addRow("Trusted orientation", self.rotation_combo)

        frame_row = QtWidgets.QWidget()
        frame_layout = QtWidgets.QHBoxLayout(frame_row)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_start = QtWidgets.QSpinBox()
        self.frame_start.setObjectName("stitchFrameStart")
        self.frame_start.setRange(0, 2_147_483_647)
        self.frame_stop = QtWidgets.QLineEdit()
        self.frame_stop.setObjectName("stitchFrameStop")
        self.frame_stop.setPlaceholderText("last")
        self.frame_stop.setMaximumWidth(100)
        self.frame_step = QtWidgets.QSpinBox()
        self.frame_step.setObjectName("stitchFrameStep")
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

        q_row = QtWidgets.QWidget()
        q_layout = QtWidgets.QHBoxLayout(q_row)
        q_layout.setContentsMargins(0, 0, 0, 0)
        self.q_min = QtWidgets.QDoubleSpinBox()
        self.q_min.setObjectName("stitchQMin")
        self.q_max = QtWidgets.QDoubleSpinBox()
        self.q_max.setObjectName("stitchQMax")
        for widget, value in ((self.q_min, 1.0), (self.q_max, 6.2)):
            widget.setDecimals(6)
            widget.setRange(-1_000_000.0, 1_000_000.0)
            widget.setValue(value)
        self.npt_1d = QtWidgets.QSpinBox()
        self.npt_1d.setObjectName("stitchBins")
        self.npt_1d.setRange(1, 1_000_000)
        self.npt_1d.setValue(1500)
        q_layout.addWidget(QtWidgets.QLabel("q min"))
        q_layout.addWidget(self.q_min)
        q_layout.addWidget(QtWidgets.QLabel("q max"))
        q_layout.addWidget(self.q_max)
        q_layout.addWidget(QtWidgets.QLabel("bins"))
        q_layout.addWidget(self.npt_1d)
        q_layout.addStretch(1)
        form.addRow("1-D grid", q_row)

        science_row = QtWidgets.QWidget()
        science_layout = QtWidgets.QHBoxLayout(science_row)
        science_layout.setContentsMargins(0, 0, 0, 0)
        self.monitor_edit = QtWidgets.QLineEdit()
        self.monitor_edit.setObjectName("stitchMonitor")
        self.monitor_edit.setPlaceholderText("blank = none")
        self.monitor_edit.setMaximumWidth(180)
        self.detector_mask_check = QtWidgets.QCheckBox("Use detector mask")
        self.detector_mask_check.setObjectName("stitchDetectorMask")
        self.detector_mask_check.setChecked(True)
        self.max_frame_mib = QtWidgets.QSpinBox()
        self.max_frame_mib.setObjectName("stitchMaxFrameMiB")
        self.max_frame_mib.setRange(1, 4096)
        self.max_frame_mib.setValue(256)
        science_layout.addWidget(QtWidgets.QLabel("Monitor"))
        science_layout.addWidget(self.monitor_edit)
        science_layout.addWidget(self.detector_mask_check)
        science_layout.addWidget(QtWidgets.QLabel("max frame MiB"))
        science_layout.addWidget(self.max_frame_mib)
        science_layout.addStretch(1)
        form.addRow("Normalization / memory", science_row)

        self.output_edit, output_row = self._path_row(
            save=True,
            caption="Choose Stitch output",
            file_filter="NeXus artifact (*.nexus)",
        )
        self.output_edit.setObjectName("stitchOutput")
        form.addRow("Output", output_row)

        self.output_policy_label = QtWidgets.QLabel(
            "Create new (refuse if present)"
        )
        self.output_policy_label.setObjectName("stitchOutputPolicy")
        form.addRow("Output policy", self.output_policy_label)

        hold = QtWidgets.QLabel(
            "Mode: 1-D only · 2-D held pending the scan-14 orientation parity oracle"
        )
        self.parity_hold = hold
        hold.setObjectName("stitch2dHold")
        hold.setWordWrap(True)
        form.addRow("Parity boundary", hold)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(input_group)
        scroll.setMaximumHeight(430)
        outer.addWidget(scroll)

        buttons = QtWidgets.QHBoxLayout()
        self.preview_button = QtWidgets.QPushButton("Preview")
        self.preview_button.setObjectName("stitchPreview")
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setObjectName("stitchRun")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setObjectName("stitchCancel")
        self.retry_cleanup_button = QtWidgets.QPushButton("Retry Cleanup")
        self.retry_cleanup_button.setObjectName("stitchRetryCleanup")
        self.retry_verification_button = QtWidgets.QPushButton(
            "Retry Verification"
        )
        self.retry_verification_button.setObjectName("stitchRetryVerification")
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
        self.progress.setObjectName("stitchProgress")
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        outer.addWidget(self.progress)

        detail_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.preview_text = QtWidgets.QPlainTextEdit()
        self.preview_text.setObjectName("stitchPreflightSummary")
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText(
            "Preview lists every exact selected raw member before Run is enabled."
        )
        detail_splitter.addWidget(self.preview_text)

        plots = QtWidgets.QWidget()
        plots_layout = QtWidgets.QVBoxLayout(plots)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        self.intensity_plot, self.intensity_curve = self._plot(
            "Stitched intensity", "Intensity"
        )
        self.coverage_plot, self.coverage_curve = self._plot(
            "Coverage", "Coverage"
        )
        self.normalization_plot, self.normalization_curve = self._plot(
            "Normalization", "Weight"
        )
        for plot in (
            self.intensity_plot,
            self.coverage_plot,
            self.normalization_plot,
        ):
            plots_layout.addWidget(plot)
        detail_splitter.addWidget(plots)
        detail_splitter.setStretchFactor(0, 1)
        detail_splitter.setStretchFactor(1, 2)
        outer.addWidget(detail_splitter, 1)

        self.status_label = QtWidgets.QLabel("Ready for an exact Preview")
        self.status_label.setObjectName("stitchStatus")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        self.geometry_kind_combo.currentIndexChanged.connect(
            self._sync_geometry_fields
        )
        self.backend_combo.currentIndexChanged.connect(self._sync_backend_fields)
        self.install_xu_asset_button.clicked.connect(self._install_xu_asset)
        self.preview_button.clicked.connect(self._begin_preflight)
        self.run_button.clicked.connect(self._begin_run)
        self.cancel_button.clicked.connect(self._cancel)
        self.retry_cleanup_button.clicked.connect(self._begin_retry_cleanup)
        self.retry_verification_button.clicked.connect(
            self._begin_retry_verification
        )
        self._sync_geometry_fields()
        self._sync_backend_fields()

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

    @staticmethod
    def _plot(title, left_label):
        widget = pg.PlotWidget()
        widget.setTitle(title)
        widget.setLabel("bottom", "q", units="Å⁻¹")
        widget.setLabel("left", left_label)
        curve = widget.plot([], [])
        return widget, curve

    def _connect_form_changes(self):
        line_edits = (
            self.project_edit,
            self.scan_edit,
            self.threshold_edit,
            self.geometry_edit,
            self.geometry_sha_edit,
            self.motor_mapping_edit,
            self.poni_reference_edit,
            self.frame_stop,
            self.monitor_edit,
            self.output_edit,
            self.source_widget.image_dir_edit,
            self.source_widget.image_stem_edit,
            self.source_widget.det_rows,
            self.source_widget.det_cols,
            self.source_widget.header_skip,
        )
        for edit in line_edits:
            edit.textChanged.connect(self._form_changed)
        for spin in (
            self.frame_start,
            self.frame_step,
            self.q_min,
            self.q_max,
            self.npt_1d,
            self.max_frame_mib,
        ):
            spin.valueChanged.connect(self._form_changed)
        for combo in (
            self.backend_combo,
            self.geometry_kind_combo,
            self.rotation_combo,
            self.source_widget.dtype_combo,
        ):
            combo.currentIndexChanged.connect(self._form_changed)
        self.detector_mask_check.toggled.connect(self._form_changed)
        self.source_widget.sigExternalSyntaxChanged.connect(self._form_changed)

    def _form_changed(self, *_args):
        if self._suppress_form_changes:
            return
        self._form_revision += 1
        if self._prepared_form_fingerprint is not None:
            self.status_label.setText("Inputs changed · run Preview again")
        self._prepared_form_fingerprint = None
        self.run_button.setEnabled(False)

    def _sync_geometry_fields(self, *_args):
        if self.backend_combo.currentData() == "xu_hist":
            self.poni_reference_edit.setEnabled(False)
            return
        poni = self.geometry_kind_combo.currentIndex() == 1
        self.poni_reference_edit.setEnabled(poni)

    def _sync_backend_fields(self, *_args):
        xu = self.backend_combo.currentData() == "xu_hist"
        self._suppress_form_changes = True
        try:
            if xu:
                self.geometry_label.setText("XU calibration asset")
                self.geometry_kind_combo.setCurrentIndex(0)
                self.geometry_sha_edit.clear()
                self.motor_mapping_edit.setText("del=del, nu=nu")
                self.poni_reference_edit.clear()
                self.rotation_combo.setCurrentIndex(0)
                self.source_widget.det_rows.setText("195")
                self.source_widget.det_cols.setText("1475")
                dtype_index = self.source_widget.dtype_combo.findText("int32")
                if dtype_index >= 0:
                    self.source_widget.dtype_combo.setCurrentIndex(dtype_index)
                self.source_widget.header_skip.setText("0")
                self.threshold_edit.setText("800000")
                self.detector_mask_check.setChecked(True)
                self.q_min.setValue(1.0)
                self.q_max.setValue(5.2)
                self.parity_hold.setText(
                    "Mode: XU 1-D only · GI, 2-D/chi, sensor/parallax, new "
                    "corrections, and unvalidated platforms are held"
                )
            else:
                self.geometry_label.setText("Geometry")
                self.motor_mapping_edit.setText("del_angle=del, nu_angle=nu")
                self.q_min.setValue(1.0)
                self.q_max.setValue(6.2)
                self.parity_hold.setText(
                    "Mode: 1-D only · 2-D held pending the scan-14 orientation "
                    "parity oracle"
                )
        finally:
            self._suppress_form_changes = False
        for widget in (
            self.geometry_kind_combo,
            self.geometry_sha_edit,
            self.motor_mapping_edit,
            self.poni_reference_edit,
            self.rotation_combo,
            self.source_widget.det_rows,
            self.source_widget.det_cols,
            self.source_widget.dtype_combo,
            self.source_widget.header_skip,
            self.threshold_edit,
            self.detector_mask_check,
        ):
            widget.setEnabled(not xu)
        self.install_xu_asset_button.setVisible(xu)
        self._sync_geometry_fields()

    def _install_xu_asset(self):
        project = self.project_edit.text().strip()
        if not project:
            self._notice("Choose Project before installing the XU calibration")
            return
        try:
            receipt = install_canonical_xu_stitch_calibration(
                project_root=project,
            )
        except (TypeError, ValueError, XuStitchCalibrationRefused) as error:
            code = getattr(error, "code", type(error).__name__)
            self._notice(f"XU calibration install refused: {code}")
            return
        target = Path(receipt.project_root) / CANONICAL_XU_STITCH_CALIBRATION_LOCATOR
        self.geometry_edit.setText(str(target))
        self._notice(
            "Canonical XU calibration ready · "
            f"{receipt.raw_sha256[:12]}… · {receipt.byte_count} bytes"
        )

    @staticmethod
    def _parse_pairs(text):
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

    @classmethod
    def _parse_references(cls, text):
        return tuple((name, float(value)) for name, value in cls._parse_pairs(text))

    def _source_path(self):
        syntax = self.source_widget.external_source_syntax()
        if (
            type(syntax) is not tuple
            or len(syntax) != 2
            or syntax[0] != "exact"
        ):
            raise ValueError("choose one exact extensionless SPEC file")
        value = syntax[1]
        return str(getattr(value, "uri", value))

    def _build_form(self):
        rows = int(self.source_widget.det_rows.text())
        columns = int(self.source_widget.det_cols.text())
        header_text = self.source_widget.header_skip.text().strip()
        stop_text = self.frame_stop.text().strip()
        threshold_text = self.threshold_edit.text().strip()
        monitor_text = self.monitor_edit.text().strip()
        geometry_kind = (
            StitchGeometryKind.PYFAI_GONIOMETER_JSON,
            StitchGeometryKind.PONI,
        )[self.geometry_kind_combo.currentIndex()]
        references = (
            self._parse_references(self.poni_reference_edit.text())
            if geometry_kind is StitchGeometryKind.PONI
            else ()
        )
        return StitchToolForm(
            project_root=self.project_edit.text().strip(),
            spec_path=self._source_path(),
            scan=self.scan_edit.text().strip(),
            image_dir=self.source_widget.image_dir_edit.text().strip(),
            image_stem=self.source_widget.image_stem_edit.text().strip(),
            frame_selector=StitchFrameSelector(
                self.frame_start.value(),
                None if not stop_text else int(stop_text),
                self.frame_step.value(),
            ),
            detector_shape=(rows, columns),
            raw_dtype=self.source_widget.dtype_combo.currentText(),
            raw_header_skip=0 if not header_text else int(header_text),
            threshold=None if not threshold_text else float(threshold_text),
            geometry_path=self.geometry_edit.text().strip(),
            geometry_kind=geometry_kind,
            expected_geometry_sha256=(
                self.geometry_sha_edit.text().strip() or None
            ),
            source_motors=self._parse_pairs(self.motor_mapping_edit.text()),
            poni_references=references,
            image_rotation=self.rotation_combo.currentData(),
            q_range=(self.q_min.value(), self.q_max.value()),
            npt_1d=self.npt_1d.value(),
            monitor_selector=(
                None if not monitor_text else MetadataColumnSelector(monitor_text)
            ),
            use_detector_mask=self.detector_mask_check.isChecked(),
            output_path=self.output_edit.text().strip(),
            overwrite=AnalysisArtifactOverwrite.REPLACE,
            max_frame_bytes=self.max_frame_mib.value() * 1024 * 1024,
            mode="1d",
            backend=self.backend_combo.currentData(),
        )

    def _current_form(self):
        try:
            return self._build_form()
        except (TypeError, ValueError, OverflowError) as error:
            self._notice(f"Invalid Stitch request: {error}")
            return None

    def _notice(self, text):
        message = str(text)
        self.status_label.setText(message)
        try:
            self._services.status.show(message, 8000)
        except Exception:
            logger.debug("Stitch status presenter failed", exc_info=True)

    def _begin_preflight(self):
        form = self._current_form()
        if form is None:
            return
        self._owner.set_form(form)
        identity = self._owner.begin_preflight()
        if identity is None:
            self._notice("Preview refused while another Stitch action is active")
            return
        self._prepared_form_fingerprint = None
        self._begin_polling(StitchOwnerAction.PREFLIGHT, "Preparing exact preview…")

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
        # Finalization retries retain this exact execution even though the
        # inputs become editable after a pending terminal.  Preserve the
        # revision that launched science so a later cleanup/verification retry
        # cannot paint that old result under newly edited controls.
        self._execution_form_revision = self._form_revision
        self._begin_polling(StitchOwnerAction.RUN, "Running Stitch…")

    def _begin_retry_cleanup(self):
        identity = self._owner.begin_retry_cleanup()
        if identity is None:
            self._notice("No retryable Stitch cleanup is available")
            return
        self._begin_polling(
            StitchOwnerAction.RETRY_CLEANUP,
            "Retrying cleanup only…",
            form_revision=self._execution_form_revision,
        )

    def _begin_retry_verification(self):
        identity = self._owner.begin_retry_verification()
        if identity is None:
            self._notice("No retryable Stitch verification is available")
            return
        self._begin_polling(
            StitchOwnerAction.RETRY_VERIFICATION,
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
        except Exception as error:
            logger.exception("Stitch owner poll failed")
            self._notice(f"Stitch owner failed: {error}")
            self._poll_timer.stop()
            self._sync_actions()
            return
        if update is not None:
            self._accept_update(update)
        if not self._owner.busy:
            self._poll_timer.stop()
            self._active_action = None
            self._active_form_revision = None
            self._sync_actions()

    def _accept_update(self, update):
        if type(update) is not StitchOwnerUpdate:
            raise TypeError("Stitch dialog requires an exact owner update")
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
            self.status_label.setText("Stitch cancelled")
            return
        outcome = update.outcome
        if outcome is None:
            raise RuntimeError("returned Stitch update omitted its outcome")
        if outcome.kind is StitchOwnerOutcomeKind.PREFLIGHT_READY:
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
        if outcome.kind is StitchOwnerOutcomeKind.CLEANUP_PENDING:
            self._notice("Output finalization needs cleanup retry; science will not rerun")
            return
        if outcome.kind is StitchOwnerOutcomeKind.VERIFICATION_PENDING:
            self._notice("Output committed; strict reload needs verification retry")
            return
        result = outcome.result
        if outcome.kind is not StitchOwnerOutcomeKind.RESULT or result is None:
            raise RuntimeError("Stitch owner returned an invalid outcome")
        terminal = result.terminal
        if terminal.disposition is ModuleDisposition.COMMITTED:
            if (
                update.stale
                or self._active_form_revision != self._form_revision
            ):
                self._notice(
                    "Stitch committed, but the displayed inputs changed; "
                    "the exact output was retained and was not painted"
                )
                return
            self._paint_result(result)
            self.status_label.setText(
                f"Committed {Path(result.request.module.output.target).name}"
            )
        elif terminal.disposition is ModuleDisposition.CANCELLED:
            self.status_label.setText("Stitch cancelled")
        elif terminal.disposition is ModuleDisposition.REFUSED:
            self._notice(f"Stitch refused: {terminal.code}")
        else:
            self._notice(
                f"Stitch failed: {terminal.code}"
                + (f" · {terminal.diagnostic}" if terminal.diagnostic else "")
            )

    def _render_preflight(self, summary):
        selectors = ", ".join(
            f"{item.name}[{item.occurrence}]" for item in summary.required_selectors
        )
        motors = ", ".join(
            f"{geometry}={source}" for geometry, source in summary.source_motors
        )
        references = ", ".join(
            f"{name}={value:.10g}" for name, value in summary.poni_references
        ) or "none"
        monitor = (
            "none"
            if summary.monitor_selector is None
            else (
                f"{summary.monitor_selector.name}"
                f"[{summary.monitor_selector.occurrence}]"
            )
        )
        threshold = "none" if summary.threshold is None else f"{summary.threshold:.10g}"
        lines = [
            f"Backend: {summary.backend}",
            f"Project: {summary.project_root}",
            f"Source: {summary.source_relative_path} · scan {summary.source_scan}",
            f"Images: {summary.image_directory_relative_path} · stem {summary.image_stem}",
            f"Frames: {len(summary.selected_labels)} · labels {summary.selected_labels}",
            f"Geometry: {summary.geometry_relative_path} ({summary.geometry_kind.value})",
            f"Geometry SHA-256: {summary.geometry_sha256}",
            f"Motor mapping: {motors}",
            f"PONI references: {references}",
            f"Image rotation: {summary.image_rotation}°",
            (
                "Raw decoder: "
                f"{summary.detector_shape[0]}×{summary.detector_shape[1]} · "
                f"{summary.raw_dtype} · header {summary.raw_header_skip} bytes · "
                f"threshold {threshold}"
            ),
            f"Output: {summary.output_relative_path} ({summary.overwrite.value})",
            f"Grid: q {summary.q_range[0]:.8g}..{summary.q_range[1]:.8g} · {summary.npt_1d} bins",
            (
                f"Normalization: monitor {monitor} · detector mask "
                f"{'on' if summary.use_detector_mask else 'off'}"
            ),
            f"Max processed frame: {summary.max_frame_bytes / (1024 * 1024):.3f} MiB",
            f"Required columns: {selectors}",
            f"Request fingerprint: {summary.request_fingerprint}",
            f"Manifest fingerprint: {summary.manifest_fingerprint}",
            f"Geometry fingerprint: {summary.geometry_fingerprint}",
        ]
        if summary.backend == "xu_hist":
            lines.extend(
                (
                    "Asset semantic fingerprint: "
                    f"{summary.asset_semantic_fingerprint}",
                    "Effective geometry fingerprint: "
                    f"{summary.effective_geometry_fingerprint}",
                )
            )
        lines.extend(("Holds: " + ", ".join(summary.holds), "", "Exact raw membership:"))
        for member in summary.members:
            values = ", ".join(
                f"{name}[{occurrence}]={value:.10g}"
                for name, occurrence, value in member.values
            )
            lines.append(
                f"  {member.label}: {member.relative_path}"
                f"#{member.source_frame_index} · {values}"
            )
        lines.extend(("", "Exact dependency files:"))
        lines.extend(f"  {path}" for path in summary.dependency_files)
        self.preview_text.setPlainText("\n".join(lines))

    def _paint_result(self, result):
        if type(result) is not StitchOperationResult:
            raise TypeError("Stitch painter requires an exact operation result")
        if (
            result.terminal.disposition is not ModuleDisposition.COMMITTED
            or result.payload is None
        ):
            raise ValueError("Stitch painter accepts only a strict committed payload")
        payload = result.payload
        q = payload.axis("q")
        intensity = payload.intensity
        coverage = payload.coverage
        normalization = payload.normalization
        if coverage is None or normalization is None:
            raise ValueError("strict Stitch payload omitted its diagnostics")
        self.setUpdatesEnabled(False)
        try:
            self.intensity_curve.setData(q, intensity)
            self.coverage_curve.setData(q, coverage)
            self.normalization_curve.setData(q, normalization)
            self._painted_result_fingerprint = payload.result_fingerprint
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
            and finalization is StitchOwnerFinalization.NONE
            and prepared is not None
            and current is not None
            and prepared.is_current(current)
            and self._prepared_form_fingerprint == current.fingerprint
        )
        self._input_group.setEnabled(not busy)
        self.preview_button.setEnabled(
            not busy and finalization is StitchOwnerFinalization.NONE
        )
        self.run_button.setEnabled(current_preview)
        self.cancel_button.setEnabled(
            busy and self._active_action is StitchOwnerAction.RUN
        )
        self.retry_cleanup_button.setEnabled(
            not busy and finalization is StitchOwnerFinalization.CLEANUP_PENDING
        )
        self.retry_verification_button.setEnabled(
            not busy
            and finalization is StitchOwnerFinalization.VERIFICATION_PENDING
        )

    def active(self):
        return bool(
            self._owner.busy
            or self._owner.finalization is not StitchOwnerFinalization.NONE
        )

    def shutdown(self):
        if self._shutdown_complete:
            from xdart.gui.pages.values import CloseReceipt

            return CloseReceipt(PageCleanup.CLEAN)
        self._poll_timer.stop()
        receipt = self._owner.close()
        if receipt.status is PageCleanup.CLEAN:
            self.source_widget.shutdown_probe_worker()
            self.setUpdatesEnabled(False)
            try:
                for curve in (
                    self.intensity_curve,
                    self.coverage_curve,
                    self.normalization_curve,
                ):
                    curve.setData([], [])
                self.preview_text.clear()
                self._painted_result_fingerprint = None
            finally:
                self.setUpdatesEnabled(True)
            self._shutdown_complete = True
        return receipt


def build_stitch_tool(services, parent):
    dialog = StitchToolDialog(services, parent)
    return PageHandle(
        key=STITCH_TOOL_KEY,
        widget=dialog,
        close=dialog.shutdown,
        activity=dialog,
    )


__all__ = ["StitchToolDialog", "build_stitch_tool"]
