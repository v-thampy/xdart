"""Thin Qt presentation for the standalone, headless RSM v2 operation."""

from __future__ import annotations

from dataclasses import dataclass
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
from xrd_tools.analysis.rsm_geometry_asset import (
    CANONICAL_RSM_GEOMETRY_LOCATOR,
    RSMGeometryAssetRefused,
    install_canonical_rsm_geometry_asset,
    rsm_geometry_asset_input,
)
from xrd_tools.analysis.rsm_operation import (
    RSMImageConditioning,
    RSMNormalizationMode,
    RSMNormalizationPolicy,
    RSMOperationResultV2,
)
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from xrd_tools.session.display_logic import PanelRole
from xrd_tools.session.rsm_viewer_model import (
    RSMViewerModel,
    RSMViewerSnapshot,
    RSMViewerValues,
    make_rsm_viewer_values,
)

from .rsm_owner import (
    RSMOwnerAction,
    RSMOwnerFinalization,
    RSMOwnerOutcomeKind,
    RSMOwnerUpdate,
    RSMToolOwner,
)
from .rsm_values import (
    RSMFrameSelector,
    RSMScanMemberForm,
    RSMToolFormV2,
    RSMToolPreflightSummaryV2,
    rsm_tool_preset,
)


logger = logging.getLogger(__name__)
_CURRENT_FORM_REVISION = object()
_PSIC_ROLES = ("mu", "eta", "chi", "phi", "nu", "del")
_MIB = 1024 * 1024


@dataclass(slots=True)
class _MemberDraft:
    spec_path: str
    scan: str
    image_dir: str
    image_stem: str
    frame_start: int
    frame_stop: str
    frame_step: int
    detector_rows: int
    detector_columns: int
    raw_dtype: str
    raw_header_skip: int
    selector_names: tuple[str, ...]
    selector_occurrences: tuple[int, ...]

    @classmethod
    def from_scan43(cls):
        preset = rsm_tool_preset()
        selectors = tuple(selector for _role, selector in preset.motor_selectors)
        return cls(
            preset.spec_relative_path,
            preset.scan,
            preset.image_directory_relative_path,
            preset.image_stem,
            preset.frame_selector.start_label,
            str(preset.frame_selector.stop_label),
            preset.frame_selector.step,
            preset.detector_shape[0],
            preset.detector_shape[1],
            preset.raw_dtype,
            preset.raw_header_skip,
            tuple(selector.name for selector in selectors),
            tuple(selector.occurrence for selector in selectors),
        )

    def clone(self):
        return _MemberDraft(
            self.spec_path,
            self.scan,
            self.image_dir,
            self.image_stem,
            self.frame_start,
            self.frame_stop,
            self.frame_step,
            self.detector_rows,
            self.detector_columns,
            self.raw_dtype,
            self.raw_header_skip,
            tuple(self.selector_names),
            tuple(self.selector_occurrences),
        )

    def selected_count_text(self) -> str:
        try:
            stop = int(self.frame_stop)
        except ValueError:
            return "open"
        if stop < self.frame_start:
            return "invalid"
        return str((stop - self.frame_start) // self.frame_step + 1)


class RSMToolDialog(QtWidgets.QDialog):
    """Cached nonmodal RSM v2 dialog; source/science I/O stay in its owner."""

    def __init__(self, services, parent=None, *, owner=None):
        super().__init__(parent)
        self._services = services
        self._owner = (
            RSMToolOwner.v2()
            if owner is None
            else owner
        )
        self._prepared_form_fingerprint = None
        self._active_action = None
        self._form_revision = 0
        self._active_form_revision = None
        self._execution_form_revision = None
        self._painted_result_fingerprint = None
        self._painted_snapshot: RSMViewerSnapshot | None = None
        self._viewer_values: RSMViewerValues | None = None
        self._viewer_model: RSMViewerModel | None = None
        self._members: list[_MemberDraft] = []
        self._shutdown_complete = False
        self._suppress_form_changes = False
        self._suppress_slice_changes = False
        self.setObjectName("rsmToolDialog")
        self.setWindowTitle("Reciprocal Space Map")
        self.setModal(False)
        self.resize(1480, 980)
        self._build_ui()
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_owner)
        self._connect_form_changes()
        self._apply_scan43_preset()
        self._sync_actions()

    @staticmethod
    def _spin(name, minimum, maximum):
        widget = QtWidgets.QSpinBox()
        widget.setObjectName(name)
        widget.setRange(minimum, maximum)
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

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        input_group = QtWidgets.QGroupBox("Exact RSM v2 request", self)
        self._input_group = input_group
        input_layout = QtWidgets.QVBoxLayout(input_group)
        common = QtWidgets.QFormLayout()
        input_layout.addLayout(common)

        self.project_edit, project_row = self._path_row(
            directory=True, caption="Choose Project root"
        )
        self.project_edit.setObjectName("rsmProject")
        common.addRow("Project", project_row)
        asset_row = QtWidgets.QWidget()
        asset_layout = QtWidgets.QHBoxLayout(asset_row)
        asset_layout.setContentsMargins(0, 0, 0, 0)
        self.geometry_asset_edit = QtWidgets.QLineEdit()
        self.geometry_asset_edit.setObjectName("rsmGeometryAsset")
        self.install_geometry_button = QtWidgets.QPushButton("Install canonical")
        self.install_geometry_button.setObjectName("rsmInstallCanonicalGeometry")
        asset_layout.addWidget(self.geometry_asset_edit, 1)
        asset_layout.addWidget(self.install_geometry_button)
        common.addRow("RSM geometry asset", asset_row)
        self.geometry_identity = QtWidgets.QLabel(
            "Asset identity not yet authenticated"
        )
        self.geometry_identity.setObjectName("rsmGeometryIdentity")
        self.geometry_identity.setWordWrap(True)
        common.addRow("Asset identity", self.geometry_identity)
        self.coordinate_frame_combo = QtWidgets.QComboBox()
        self.coordinate_frame_combo.setObjectName("rsmCoordinateFrame")
        for frame in RSMCoordinateFrame:
            self.coordinate_frame_combo.addItem(frame.display_name, frame)
        common.addRow("Coordinates", self.coordinate_frame_combo)

        member_group = QtWidgets.QGroupBox("Ordered scan members")
        member_layout = QtWidgets.QVBoxLayout(member_group)
        self.member_table = QtWidgets.QTableWidget(0, 6)
        self.member_table.setObjectName("rsmMemberTable")
        self.member_table.setHorizontalHeaderLabels(
            ("#", "SPEC", "scan", "frames", "images", "stem")
        )
        self.member_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.member_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.member_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.member_table.verticalHeader().setVisible(False)
        header = self.member_table.horizontalHeader()
        for column in (0, 2, 3):
            header.setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
            )
        for column in (1, 4, 5):
            header.setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.ResizeMode.Stretch,
            )
        self.member_table.setMaximumHeight(150)
        member_layout.addWidget(self.member_table)
        member_buttons = QtWidgets.QHBoxLayout()
        self.add_member_button = QtWidgets.QPushButton("Add")
        self.add_member_button.setObjectName("rsmAddMember")
        self.add_scan42_button = QtWidgets.QPushButton("Add Scan 42")
        self.add_scan42_button.setObjectName("rsmAddScan42")
        self.remove_member_button = QtWidgets.QPushButton("Remove")
        self.remove_member_button.setObjectName("rsmRemoveMember")
        self.move_up_button = QtWidgets.QPushButton("Move Up")
        self.move_up_button.setObjectName("rsmMoveMemberUp")
        self.move_down_button = QtWidgets.QPushButton("Move Down")
        self.move_down_button.setObjectName("rsmMoveMemberDown")
        self.preset_button = QtWidgets.QPushButton("Reset to Scan 43 preset")
        self.preset_button.setObjectName("rsmScan43Preset")
        for button in (
            self.add_member_button,
            self.add_scan42_button,
            self.remove_member_button,
            self.move_up_button,
            self.move_down_button,
            self.preset_button,
        ):
            member_buttons.addWidget(button)
        member_buttons.addStretch(1)
        member_layout.addLayout(member_buttons)

        editor = QtWidgets.QGroupBox("Selected member")
        editor_form = QtWidgets.QFormLayout(editor)
        self.spec_edit, spec_row = self._path_row(
            caption="Choose extensionless SPEC file",
            file_filter="SPEC files (*);;All files (*)",
        )
        self.spec_edit.setObjectName("rsmSpec")
        editor_form.addRow("SPEC", spec_row)
        self.scan_edit = QtWidgets.QLineEdit()
        self.scan_edit.setObjectName("rsmScan")
        editor_form.addRow("Scan", self.scan_edit)
        self.image_dir_edit, image_row = self._path_row(
            directory=True, caption="Choose raw-image directory"
        )
        self.image_dir_edit.setObjectName("rsmImageDirectory")
        editor_form.addRow("Images", image_row)
        self.image_stem_edit = QtWidgets.QLineEdit()
        self.image_stem_edit.setObjectName("rsmImageStem")
        editor_form.addRow("Image stem", self.image_stem_edit)

        frame_row = QtWidgets.QWidget()
        frame_layout = QtWidgets.QHBoxLayout(frame_row)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_start = self._spin("rsmFrameStart", 0, 2_147_483_647)
        self.frame_stop = QtWidgets.QLineEdit()
        self.frame_stop.setObjectName("rsmFrameStop")
        self.frame_stop.setMaximumWidth(100)
        self.frame_step = self._spin("rsmFrameStep", 1, 2_147_483_647)
        for label, widget in (
            ("start", self.frame_start),
            ("stop inclusive", self.frame_stop),
            ("step", self.frame_step),
        ):
            frame_layout.addWidget(QtWidgets.QLabel(label))
            frame_layout.addWidget(widget)
        frame_layout.addStretch(1)
        editor_form.addRow("Frame labels", frame_row)

        decoder = QtWidgets.QWidget()
        decoder_layout = QtWidgets.QHBoxLayout(decoder)
        decoder_layout.setContentsMargins(0, 0, 0, 0)
        self.detector_rows = self._spin("rsmDetectorRows", 2, 1_000_000)
        self.detector_columns = self._spin("rsmDetectorColumns", 2, 1_000_000)
        self.raw_dtype_combo = QtWidgets.QComboBox()
        self.raw_dtype_combo.setObjectName("rsmRawDtype")
        self.raw_dtype_combo.addItems(("int32", "uint16", "float32", "float64"))
        self.raw_header_skip = self._spin("rsmRawHeaderSkip", 0, 1 << 30)
        for label, widget in (
            ("rows", self.detector_rows),
            ("columns", self.detector_columns),
            ("dtype", self.raw_dtype_combo),
            ("header bytes", self.raw_header_skip),
        ):
            decoder_layout.addWidget(QtWidgets.QLabel(label))
            decoder_layout.addWidget(widget)
        decoder_layout.addStretch(1)
        editor_form.addRow("Raw decoder", decoder)

        selectors = QtWidgets.QWidget()
        selector_layout = QtWidgets.QGridLayout(selectors)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        self.member_selector_edits = {}
        for column, role in enumerate(_PSIC_ROLES):
            selector_layout.addWidget(QtWidgets.QLabel(role), 0, column)
            name = QtWidgets.QLineEdit()
            name.setObjectName(f"rsmSelector_{role}_name")
            occurrence = self._spin(f"rsmSelector_{role}_occurrence", 0, 1024)
            pair = QtWidgets.QWidget()
            pair_layout = QtWidgets.QHBoxLayout(pair)
            pair_layout.setContentsMargins(0, 0, 0, 0)
            pair_layout.addWidget(name, 1)
            pair_layout.addWidget(occurrence)
            selector_layout.addWidget(pair, 1, column)
            self.member_selector_edits[role] = (name, occurrence)
        editor_form.addRow("Exact motor columns", selectors)
        member_layout.addWidget(editor)
        input_layout.addWidget(member_group)

        advanced = QtWidgets.QGroupBox("Advanced common controls")
        advanced.setCheckable(True)
        advanced.setChecked(False)
        advanced_form = QtWidgets.QFormLayout(advanced)
        normalization = QtWidgets.QWidget()
        normalization_layout = QtWidgets.QHBoxLayout(normalization)
        normalization_layout.setContentsMargins(0, 0, 0, 0)
        self.normalization_combo = QtWidgets.QComboBox()
        self.normalization_combo.setObjectName("rsmNormalization")
        self.normalization_combo.addItem(
            "Foil transmission / exposure",
            RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE,
        )
        self.normalization_combo.addItem("Identity", RSMNormalizationMode.IDENTITY)
        normalization_layout.addWidget(self.normalization_combo)
        self.common_selector_edits = {}
        for role in ("foil", "exposure"):
            name = QtWidgets.QLineEdit()
            name.setObjectName(f"rsmSelector_{role}_name")
            occurrence = self._spin(f"rsmSelector_{role}_occurrence", 0, 1024)
            self.common_selector_edits[role] = (name, occurrence)
            normalization_layout.addWidget(QtWidgets.QLabel(role))
            normalization_layout.addWidget(name)
            normalization_layout.addWidget(occurrence)
        advanced_form.addRow("Normalization", normalization)
        self.selector_edits = {
            **self.member_selector_edits,
            **self.common_selector_edits,
        }

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
        advanced_form.addRow("Foil absorption lengths", absorption)

        conditioning = QtWidgets.QWidget()
        conditioning_layout = QtWidgets.QHBoxLayout(conditioning)
        conditioning_layout.setContentsMargins(0, 0, 0, 0)
        self.offset_edit = QtWidgets.QLineEdit()
        self.offset_edit.setObjectName("rsmAdditiveOffset")
        self.high_threshold_edit = QtWidgets.QLineEdit()
        self.high_threshold_edit.setObjectName("rsmHighThreshold")
        self.static_hot_edit = QtWidgets.QLineEdit()
        self.static_hot_edit.setObjectName("rsmStaticHotThreshold")
        for label, widget in (
            ("offset", self.offset_edit),
            ("high to NaN", self.high_threshold_edit),
            ("all-frame hot", self.static_hot_edit),
        ):
            conditioning_layout.addWidget(QtWidgets.QLabel(label))
            conditioning_layout.addWidget(widget)
        conditioning_layout.addStretch(1)
        advanced_form.addRow("Conditioning", conditioning)

        execution = QtWidgets.QWidget()
        execution_layout = QtWidgets.QHBoxLayout(execution)
        execution_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_combo = QtWidgets.QComboBox()
        self.grid_combo.setObjectName("rsmGrid")
        self.grid_combo.addItem("Quick 40x40x40", (40, 40, 40))
        self.grid_combo.addItem("Full 200x200x200", (200, 200, 200))
        self.chunk_size = self._spin("rsmChunkSize", 1, 1024)
        self.max_frame_mib = self._spin("rsmMaxFrameMiB", 1, 4096)
        self.max_chunk_mib = self._spin("rsmMaxChunkMiB", 1, 4096)
        for label, widget in (
            ("grid", self.grid_combo),
            ("chunk", self.chunk_size),
            ("max frame MiB", self.max_frame_mib),
            ("max chunk MiB", self.max_chunk_mib),
        ):
            execution_layout.addWidget(QtWidgets.QLabel(label))
            execution_layout.addWidget(widget)
        execution_layout.addStretch(1)
        advanced_form.addRow("Grid / memory", execution)
        input_layout.addWidget(advanced)

        output_form = QtWidgets.QFormLayout()
        self.output_edit, output_row = self._path_row(
            save=True,
            caption="Choose RSM output",
            file_filter="NeXus artifact (*.nexus)",
        )
        self.output_edit.setObjectName("rsmOutput")
        output_form.addRow("Output", output_row)
        self.output_policy_label = QtWidgets.QLabel(
            "Create new (refuse if present)"
        )
        self.output_policy_label.setObjectName("rsmOutputPolicy")
        output_form.addRow("Output policy", self.output_policy_label)
        hold = QtWidgets.QLabel(
            "R2: ordered exact SPEC members, canonical psic geometry authority, "
            "one common selected-coordinate grid, immutable new output, and "
            "committed-result display only. Held: "
            "GI/refraction corrections, volume rendering, and hostile/shared "
            "Project output namespaces."
        )
        hold.setObjectName("rsmBoundary")
        hold.setWordWrap(True)
        output_form.addRow("Scientific boundary", hold)
        input_layout.addLayout(output_form)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(input_group)
        scroll.setMaximumHeight(560)
        outer.addWidget(scroll)

        buttons = QtWidgets.QHBoxLayout()
        self.preview_button = QtWidgets.QPushButton("Preview")
        self.preview_button.setObjectName("rsmPreview")
        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setObjectName("rsmRun")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setObjectName("rsmCancel")
        self.retry_cleanup_button = QtWidgets.QPushButton("Retry Finalization")
        self.retry_cleanup_button.setObjectName("rsmRetryCleanup")
        self.retry_verification_button = QtWidgets.QPushButton("Retry Verification")
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
            "Preview lists ordered membership and every exact RSM v2 identity."
        )
        detail.addWidget(self.preview_text)
        surface = QtWidgets.QWidget()
        surface_layout = QtWidgets.QVBoxLayout(surface)
        surface_layout.setContentsMargins(0, 0, 0, 0)
        indices = QtWidgets.QHBoxLayout()
        indices.addWidget(QtWidgets.QLabel("Committed-result slice indices"))
        self.slice_index_controls = []
        self.slice_index_labels = []
        for axis in ("H", "K", "L"):
            control = self._spin(f"rsm{axis}SliceIndex", 0, 0)
            control.setEnabled(False)
            label = QtWidgets.QLabel(axis)
            indices.addWidget(label)
            indices.addWidget(control)
            self.slice_index_labels.append(label)
            self.slice_index_controls.append(control)
        indices.addStretch(1)
        self.result_facts = QtWidgets.QLabel("No committed RSM loaded")
        self.result_facts.setObjectName("rsmResultFacts")
        self.result_facts.setWordWrap(True)
        indices.addWidget(self.result_facts, 1)
        surface_layout.addLayout(indices)
        plots = QtWidgets.QWidget()
        plots_layout = QtWidgets.QGridLayout(plots)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        panel_specs = (
            ("HK slice", "H", "K", PanelRole.SLICE_2D),
            ("HL slice", "H", "L", PanelRole.SLICE_2D),
            ("KL slice", "K", "L", PanelRole.SLICE_2D),
            ("H mean projection", "H", "Mean I", PanelRole.PROJ_1D),
            ("K mean projection", "K", "Mean I", PanelRole.PROJ_1D),
            ("L mean projection", "L", "Mean I", PanelRole.PROJ_1D),
        )
        self.surface_plots = []
        initial_items = []
        for index, (title, horizontal, vertical, role) in enumerate(panel_specs):
            plot = pg.PlotWidget()
            plot.setObjectName(f"rsmSurfacePlot{index}")
            plot.setTitle(title)
            plot.setLabel("bottom", horizontal)
            plot.setLabel("left", vertical)
            item = (
                pg.ImageItem(axisOrder="row-major")
                if role is PanelRole.SLICE_2D
                else pg.PlotDataItem()
            )
            plot.addItem(item)
            self.surface_plots.append(plot)
            initial_items.append(item)
            plots_layout.addWidget(plot, index // 3, index % 3)
        self.surface_items = tuple(initial_items)
        self._surface_title_texts = tuple(item[0] for item in panel_specs)
        self._surface_axis_label_texts = tuple(
            (item[1], item[2]) for item in panel_specs
        )
        self.slice_plots = self.surface_plots[:3]
        self.slice_images = list(self.surface_items[:3])
        surface_layout.addWidget(plots)
        detail.addWidget(surface)
        detail.setStretchFactor(0, 1)
        detail.setStretchFactor(1, 3)
        detail.setSizes((380, 1100))
        outer.addWidget(detail, 1)
        self.status_label = QtWidgets.QLabel("Ready for an exact Preview")
        self.status_label.setObjectName("rsmStatus")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        self.install_geometry_button.clicked.connect(self._install_geometry_asset)
        self.member_table.currentCellChanged.connect(self._member_selection_changed)
        self.add_member_button.clicked.connect(self._add_member)
        self.add_scan42_button.clicked.connect(self._add_scan42_member)
        self.remove_member_button.clicked.connect(self._remove_member)
        self.move_up_button.clicked.connect(lambda: self._move_member(-1))
        self.move_down_button.clicked.connect(lambda: self._move_member(1))
        self.preset_button.clicked.connect(self._apply_scan43_preset)
        self.preview_button.clicked.connect(self._begin_preflight)
        self.run_button.clicked.connect(self._begin_run)
        self.cancel_button.clicked.connect(self._cancel)
        self.retry_cleanup_button.clicked.connect(self._begin_retry_cleanup)
        self.retry_verification_button.clicked.connect(self._begin_retry_verification)
        for control in self.slice_index_controls:
            control.valueChanged.connect(self._slice_index_changed)

    def _connect_form_changes(self):
        for edit in (
            self.project_edit,
            self.geometry_asset_edit,
            self.output_edit,
            self.offset_edit,
            self.high_threshold_edit,
            self.static_hot_edit,
            *(item[0] for item in self.common_selector_edits.values()),
            *self.absorption_edits,
        ):
            edit.textChanged.connect(self._form_changed)
        for spin in (
            self.chunk_size,
            self.max_frame_mib,
            self.max_chunk_mib,
            *(item[1] for item in self.common_selector_edits.values()),
        ):
            spin.valueChanged.connect(self._form_changed)
        for combo in (
            self.coordinate_frame_combo,
            self.normalization_combo,
            self.grid_combo,
        ):
            combo.currentIndexChanged.connect(self._form_changed)
        for edit in (
            self.spec_edit,
            self.scan_edit,
            self.image_dir_edit,
            self.image_stem_edit,
            self.frame_stop,
            *(item[0] for item in self.member_selector_edits.values()),
        ):
            edit.textChanged.connect(self._member_editor_changed)
        for spin in (
            self.frame_start,
            self.frame_step,
            self.detector_rows,
            self.detector_columns,
            self.raw_header_skip,
            *(item[1] for item in self.member_selector_edits.values()),
        ):
            spin.valueChanged.connect(self._member_editor_changed)
        self.raw_dtype_combo.currentIndexChanged.connect(self._member_editor_changed)

    def _selected_member_index(self):
        row = self.member_table.currentRow()
        return row if 0 <= row < len(self._members) else 0

    def _commit_member_editor(self):
        if not self._members:
            return
        index = self._selected_member_index()
        member = self._members[index]
        member.spec_path = self.spec_edit.text().strip()
        member.scan = self.scan_edit.text().strip()
        member.image_dir = self.image_dir_edit.text().strip()
        member.image_stem = self.image_stem_edit.text().strip()
        member.frame_start = self.frame_start.value()
        member.frame_stop = self.frame_stop.text().strip()
        member.frame_step = self.frame_step.value()
        member.detector_rows = self.detector_rows.value()
        member.detector_columns = self.detector_columns.value()
        member.raw_dtype = self.raw_dtype_combo.currentText()
        member.raw_header_skip = self.raw_header_skip.value()
        selectors = tuple(self.member_selector_edits[role] for role in _PSIC_ROLES)
        member.selector_names = tuple(name.text().strip() for name, _occ in selectors)
        member.selector_occurrences = tuple(occ.value() for _name, occ in selectors)
        self._refresh_member_row(index)

    def _load_member_editor(self, index):
        if not 0 <= index < len(self._members):
            return
        member = self._members[index]
        self._suppress_form_changes = True
        try:
            self.spec_edit.setText(member.spec_path)
            self.scan_edit.setText(member.scan)
            self.image_dir_edit.setText(member.image_dir)
            self.image_stem_edit.setText(member.image_stem)
            self.frame_start.setValue(member.frame_start)
            self.frame_stop.setText(member.frame_stop)
            self.frame_step.setValue(member.frame_step)
            self.detector_rows.setValue(member.detector_rows)
            self.detector_columns.setValue(member.detector_columns)
            self.raw_dtype_combo.setCurrentIndex(
                max(0, self.raw_dtype_combo.findText(member.raw_dtype))
            )
            self.raw_header_skip.setValue(member.raw_header_skip)
            for role, name, occurrence in zip(
                _PSIC_ROLES,
                member.selector_names,
                member.selector_occurrences,
                strict=True,
            ):
                name_edit, occurrence_edit = self.member_selector_edits[role]
                name_edit.setText(name)
                occurrence_edit.setValue(occurrence)
        finally:
            self._suppress_form_changes = False
        self._sync_member_actions()

    def _member_selection_changed(
        self, current_row, _current_column, previous_row, _previous_column
    ):
        if self._suppress_form_changes:
            return
        if 0 <= previous_row < len(self._members):
            self._refresh_member_row(previous_row)
        self._load_member_editor(current_row)

    def _member_editor_changed(self, *_args):
        if self._suppress_form_changes:
            return
        self._commit_member_editor()
        self._form_changed()

    def _refresh_member_table(self, selected=0):
        self._suppress_form_changes = True
        try:
            self.member_table.setRowCount(len(self._members))
            for row in range(len(self._members)):
                self._refresh_member_row(row)
            if self._members:
                selected = min(max(0, selected), len(self._members) - 1)
                self.member_table.setCurrentCell(selected, 0)
        finally:
            self._suppress_form_changes = False
        if self._members:
            self._load_member_editor(selected)
        self._sync_member_actions()

    def _refresh_member_row(self, row):
        if not 0 <= row < len(self._members):
            return
        member = self._members[row]
        values = (
            str(row + 1),
            member.spec_path,
            member.scan,
            member.selected_count_text(),
            member.image_dir,
            member.image_stem,
        )
        for column, value in enumerate(values):
            item = self.member_table.item(row, column)
            if item is None:
                item = QtWidgets.QTableWidgetItem()
                self.member_table.setItem(row, column, item)
            item.setText(value)
            item.setToolTip(value)

    def _sync_member_actions(self):
        count = len(self._members)
        row = self._selected_member_index() if count else -1
        self.add_member_button.setEnabled(count < 16)
        self.add_scan42_button.setEnabled(count < 16)
        self.remove_member_button.setEnabled(count > 1)
        self.move_up_button.setEnabled(row > 0)
        self.move_down_button.setEnabled(0 <= row < count - 1)

    def _add_member(self):
        self._commit_member_editor()
        row = self._selected_member_index()
        draft = (
            self._members[row].clone()
            if self._members
            else _MemberDraft.from_scan43()
        )
        insert = min(row + 1, len(self._members))
        self._members.insert(insert, draft)
        self._refresh_member_table(insert)
        self._form_changed()

    def _add_scan42_member(self):
        draft = _MemberDraft.from_scan43()
        draft.scan = "42.1"
        draft.image_stem = "b_thampy_STO_align_scan42_"
        draft.frame_start = 0
        draft.frame_stop = "160"
        draft.frame_step = 1
        insert = min(self._selected_member_index() + 1, len(self._members))
        self._members.insert(insert, draft)
        self._refresh_member_table(insert)
        self._form_changed()

    def _remove_member(self):
        if len(self._members) <= 1:
            return
        row = self._selected_member_index()
        del self._members[row]
        self._refresh_member_table(min(row, len(self._members) - 1))
        self._form_changed()

    def _move_member(self, offset):
        self._commit_member_editor()
        row = self._selected_member_index()
        target = row + int(offset)
        if not 0 <= target < len(self._members):
            return
        self._members[row], self._members[target] = (
            self._members[target],
            self._members[row],
        )
        self._refresh_member_table(target)
        self._form_changed()

    def _form_changed(self, *_args):
        if self._suppress_form_changes:
            return
        self._form_revision += 1
        self.geometry_identity.setText("Asset identity requires a new Preview")
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
            self.geometry_asset_edit.setText(CANONICAL_RSM_GEOMETRY_LOCATOR)
            self._members = [_MemberDraft.from_scan43()]
            if not self.output_edit.text().strip():
                self.output_edit.setText("rsm_scan43.nexus")
            self.coordinate_frame_combo.setCurrentIndex(
                self.coordinate_frame_combo.findData(RSMCoordinateFrame.HKL)
            )
            self.normalization_combo.setCurrentIndex(0)
            for role, selector in (
                ("foil", preset.normalization.foil_selector),
                ("exposure", preset.normalization.exposure_selector),
            ):
                name, occurrence = self.common_selector_edits[role]
                name.setText(selector.name)
                occurrence.setValue(selector.occurrence)
            for edit, value in zip(
                self.absorption_edits,
                preset.normalization.absorption_lengths,
                strict=True,
            ):
                edit.setText(f"{value:.12g}")
            self.offset_edit.setText(f"{preset.conditioning.additive_offset:.12g}")
            self.high_threshold_edit.setText(
                f"{preset.conditioning.high_threshold:.12g}"
            )
            self.static_hot_edit.setText(
                f"{preset.conditioning.static_hot_threshold:.12g}"
            )
            self.grid_combo.setCurrentIndex(0)
            self.chunk_size.setValue(preset.chunk_size)
            self.max_frame_mib.setValue(preset.max_frame_bytes // _MIB)
            self.max_chunk_mib.setValue(preset.max_chunk_bytes // _MIB)
        finally:
            self._suppress_form_changes = False
        self._refresh_member_table(0)
        self._form_changed()

    @staticmethod
    def _optional_float(text):
        value = str(text).strip()
        return None if not value else float(value)

    def _common_selector(self, role):
        name, occurrence = self.common_selector_edits[role]
        return MetadataColumnSelector(name.text().strip(), occurrence.value())

    @staticmethod
    def _inside_project(project_path, text, name):
        spelling = str(text).strip()
        if not spelling:
            raise ValueError(f"{name} is required")
        path = Path(spelling).expanduser()
        return path if path.is_absolute() else project_path / path

    def _member_form(self, member, project_path):
        stop = member.frame_stop.strip()
        return RSMScanMemberForm(
            self._inside_project(project_path, member.spec_path, "SPEC path"),
            member.scan,
            self._inside_project(project_path, member.image_dir, "image directory"),
            member.image_stem,
            RSMFrameSelector(
                member.frame_start,
                None if not stop else int(stop),
                member.frame_step,
            ),
            (member.detector_rows, member.detector_columns),
            member.raw_dtype,
            member.raw_header_skip,
            tuple(
                (role, MetadataColumnSelector(name, occurrence))
                for role, name, occurrence in zip(
                    _PSIC_ROLES,
                    member.selector_names,
                    member.selector_occurrences,
                    strict=True,
                )
            ),
        )

    def _build_form(self):
        self._commit_member_editor()
        project_text = self.project_edit.text().strip()
        if not project_text:
            raise ValueError("Project root is required")
        project_path = Path(project_text).expanduser()
        if self.normalization_combo.currentIndex() == 1:
            normalization = RSMNormalizationPolicy.identity()
        else:
            normalization = RSMNormalizationPolicy(
                RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE,
                self._common_selector("foil"),
                self._common_selector("exposure"),
                tuple(float(edit.text()) for edit in self.absorption_edits),
            )
        return RSMToolFormV2(
            project_text,
            rsm_geometry_asset_input(self.geometry_asset_edit.text().strip()),
            tuple(self._member_form(member, project_path) for member in self._members),
            RSMImageConditioning(
                float(self.offset_edit.text()),
                self._optional_float(self.high_threshold_edit.text()),
                self._optional_float(self.static_hot_edit.text()),
            ),
            normalization,
            tuple(self.grid_combo.currentData()),
            self.chunk_size.value(),
            self.max_frame_mib.value() * _MIB,
            self.max_chunk_mib.value() * _MIB,
            self._inside_project(project_path, self.output_edit.text(), "output path"),
            AnalysisArtifactOverwrite.CREATE_NEW,
            coordinate_frame=RSMCoordinateFrame(
                self.coordinate_frame_combo.currentData()
            ),
        )

    def _current_form(self):
        try:
            return self._build_form()
        except (TypeError, ValueError, OverflowError) as error:
            self._notice(f"Invalid RSM request: {error}")
            return None

    def _install_geometry_asset(self):
        project = self.project_edit.text().strip()
        if not project:
            self._notice("Install canonical requires a Project root")
            return
        try:
            receipt = install_canonical_rsm_geometry_asset(project_root=project)
        except RSMGeometryAssetRefused as error:
            self._notice(f"{error.code}: {error}")
            return
        self._suppress_form_changes = True
        try:
            self.geometry_asset_edit.setText(receipt.request.locator)
        finally:
            self._suppress_form_changes = False
        self._form_changed()
        self._show_geometry_identity(
            receipt.raw_sha256,
            receipt.semantic_fingerprint,
            receipt.receipt_fingerprint,
        )
        self._notice(f"Canonical RSM geometry ready: {receipt.lexical_relative_path}")

    def _show_geometry_identity(self, raw, semantic, receipt):
        self.geometry_identity.setText(
            f"raw {raw} · semantic {semantic} · receipt {receipt}"
        )

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
        if self._prepared_form_fingerprint != form.fingerprint:
            self._notice("Run requires an unchanged successful Preview")
            self._prepared_form_fingerprint = None
            self._sync_actions()
            return
        self._owner.set_form(form)
        identity = self._owner.begin_run()
        if identity is None:
            self._notice("Run requires an unchanged successful Preview")
            self._prepared_form_fingerprint = None
            self._sync_actions()
            return
        self._execution_form_revision = self._form_revision
        self._begin_polling(RSMOwnerAction.RUN, "Running RSM v2…")

    def _begin_retry_cleanup(self):
        identity = self._owner.begin_retry_cleanup()
        if identity is None:
            self._notice("No retryable RSM cleanup is available")
            return
        self._begin_polling(
            RSMOwnerAction.RETRY_CLEANUP,
            "Retrying finalization; science will not rerun…",
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

    def _begin_polling(self, action, message, *, form_revision=_CURRENT_FORM_REVISION):
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
            self._notice(
                f"RSM_VIEW_PRESENTATION_FAILED: presentation failed: {error}"
            )
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
            self._notice(
                f"{update.action.value} failed: "
                f"{update.failure_message or update.failure_type}"
            )
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
                "Preview ready · "
                f"{preflight.summary.total_selected_frames} exact frames"
            )
            return
        if outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_REFUSED:
            self._notice(
                f"Preview refused: {outcome.refusal_message or outcome.refusal_code}"
            )
            return
        if outcome.kind is RSMOwnerOutcomeKind.CLEANUP_PENDING:
            message = outcome.finalization_message or "Output finalization needs retry"
            self._notice(f"{message}; use Retry Finalization. Science will not rerun")
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
                    "RSM committed, but inputs changed; the exact output was retained "
                    "and was not painted"
                )
                return
            self._paint_result(result)
            self.status_label.setText(
                f"Committed {Path(result.request.module.output.target).name}"
            )
        elif terminal.disposition is ModuleDisposition.CANCELLED:
            self._notice("RSM cancelled" + (f" · {terminal.diagnostic}" if terminal.diagnostic else ""))
        elif terminal.disposition is ModuleDisposition.REFUSED:
            self._notice(f"RSM refused: {terminal.code}" + (f" · {terminal.diagnostic}" if terminal.diagnostic else ""))
        else:
            self._notice(
                f"RSM failed: {terminal.code}"
                + (f" · {terminal.diagnostic}" if terminal.diagnostic else "")
            )

    def _render_preflight(self, summary):
        if type(summary) is not RSMToolPreflightSummaryV2:
            raise TypeError("RSM v2 presenter requires exact summary")
        frame = summary.coordinate_frame
        axis_symbols = frame.axis_symbols
        if frame is RSMCoordinateFrame.HKL:
            source_ub_authority = (
                "Source UB authority: authenticated source UB drives H/K/L "
                "conversion"
            )
        else:
            source_ub_authority = (
                "Source UB authority: source UB is unused and non-driving; "
                "explicit identity drives Cartesian Q"
            )
        union = ", ".join(
            f"{name} {low:.10g}..{high:.10g}"
            for name, (low, high) in zip(
                axis_symbols,
                summary.union_q_bounds,
                strict=True,
            )
        )
        lines = [
            f"Project: {summary.project_root}",
            f"Output: {summary.output_relative_path}",
            f"Coordinate frame: {frame.display_name}",
            "Stored axes: "
            + ", ".join(
                label if unit is None else f"{label} [{unit}]"
                for label, unit in zip(
                    frame.axis_symbols,
                    frame.axis_units,
                    strict=True,
                )
            ),
            f"xrayutilities matrix policy: {frame.matrix_policy}",
            source_ub_authority,
            (
                f"Ordered members: {len(summary.members)} · total frames "
                f"{summary.total_selected_frames}"
            ),
            f"Asset raw: {summary.geometry_asset_raw_sha256}",
            f"Asset semantic: {summary.geometry_asset_semantic_fingerprint}",
            f"Asset receipt: {summary.geometry_asset_receipt_fingerprint}",
            f"Effective geometry: {summary.effective_geometry_fingerprint}",
            f"Common grid: {summary.common_grid_fingerprint}",
            (
                "Union q bounds: "
                if frame is RSMCoordinateFrame.HKL
                else "Union coordinate bounds: "
            )
            + union,
            f"Grid bins: {summary.bins} · detector {summary.detector_shape}",
            (
                f"Normalization: {summary.normalization_mode.value} · divisor "
                f"{summary.normalization_divisor_range[0]:.12g}.."
                f"{summary.normalization_divisor_range[1]:.12g}"
            ),
            f"Mask intent: {summary.mask_intent}",
            f"Group source: {summary.group_source_fingerprint}",
            f"Group preflight: {summary.group_preflight_fingerprint}",
            f"Plan: {summary.plan_fingerprint}",
            f"Request: {summary.request_fingerprint}",
            f"Output identity: {summary.output_fingerprint}",
            "Holds: " + ", ".join(summary.holds),
            "",
        ]
        for member in summary.members:
            coordinate_matrix = "; ".join(
                " ".join(f"{value:.10g}" for value in row) for row in member.ub
            )
            bounds = ", ".join(
                f"{name} {low:.10g}..{high:.10g}"
                for name, (low, high) in zip(
                    axis_symbols,
                    member.q_bounds,
                    strict=True,
                )
            )
            lines.extend(
                (
                    (
                        f"Member {member.ordinal + 1}: "
                        f"{member.source_relative_path} · scan "
                        f"{member.source_scan} · "
                        f"{member.selected_frame_count} frames"
                    ),
                    f"  labels: {member.selected_labels}",
                    f"  form: {member.member_form_fingerprint}",
                    f"  module source: {member.module_source_fingerprint}",
                    f"  source: {member.source_fingerprint}",
                    f"  table: {member.table_fingerprint}",
                    f"  member preflight: {member.member_preflight_fingerprint}",
                    f"  geometry binding: {member.geometry_binding_fingerprint}",
                    f"  energy: {member.energy_eV:.12g} eV",
                    f"  xrayutilities matrix: {coordinate_matrix}",
                    f"  coordinate bounds: {bounds}",
                    (
                        "  divisor: "
                        f"{member.normalization_divisor_range[0]:.12g}.."
                        f"{member.normalization_divisor_range[1]:.12g}"
                    ),
                    f"  raw dependencies ({member.dependency_file_count}):",
                    *(f"    {path}" for path in member.dependency_files),
                    "",
                )
            )
        self.preview_text.setPlainText("\n".join(lines))
        self._show_geometry_identity(
            summary.geometry_asset_raw_sha256,
            summary.geometry_asset_semantic_fingerprint,
            summary.geometry_asset_receipt_fingerprint,
        )

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

    def _stage_surface(self, snapshot):
        staged = []
        for product in snapshot.products:
            if product.panel_key.role is PanelRole.SLICE_2D:
                x0, x1 = self._axis_edges(product.x_axis)
                y0, y1 = self._axis_edges(product.y_axis_or_none)
                item = pg.ImageItem(axisOrder="row-major")
                item.setImage(np.asarray(product.values), autoLevels=True)
                item.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
            else:
                item = pg.PlotDataItem(
                    np.asarray(product.x_axis),
                    np.asarray(product.values),
                    connect="finite",
                )
            staged.append(item)
        return tuple(staged)

    @staticmethod
    def _surface_titles(snapshot):
        state = snapshot.state
        axis0, axis1, axis2 = state.coordinate_frame.axis_symbols
        return (
            f"{axis0}{axis1} slice · {axis2} index {state.l_index}",
            f"{axis0}{axis2} slice · {axis1} index {state.k_index}",
            f"{axis1}{axis2} slice · {axis0} index {state.h_index}",
            f"{axis0} mean projection",
            f"{axis1} mean projection",
            f"{axis2} mean projection",
        )

    @staticmethod
    def _surface_axis_labels(values):
        axis0, axis1, axis2 = values.coordinate_frame.axis_labels
        return (
            (axis0, axis1),
            (axis0, axis2),
            (axis1, axis2),
            (axis0, "Mean I"),
            (axis1, "Mean I"),
            (axis2, "Mean I"),
        )

    @staticmethod
    def _surface_facts(values, snapshot):
        axes = tuple(axis for _name, axis in values.axes)
        state = snapshot.state
        axis0, axis1, axis2 = values.coordinate_frame.axis_symbols
        return (
            f"{values.coordinate_frame.display_name} · shape {values.shape} · "
            f"indices {axis0}/{axis1}/{axis2} {state.h_index}/"
            f"{state.k_index}/{state.l_index} · finite products "
            f"{snapshot.finite_counts} · cache {snapshot.cache_bytes} bytes · "
            f"{axis0} {axes[0][0]:.8g}..{axes[0][-1]:.8g} · "
            f"{axis1} {axes[1][0]:.8g}..{axes[1][-1]:.8g} · "
            f"{axis2} {axes[2][0]:.8g}..{axes[2][-1]:.8g}"
        )

    def _slice_control_state(self):
        return tuple(
            (control.minimum(), control.maximum(), control.value(), control.isEnabled())
            for control in self.slice_index_controls
        )

    def _restore_slice_controls(self, states):
        self._suppress_slice_changes = True
        try:
            for control, (minimum, maximum, value, enabled) in zip(
                self.slice_index_controls, states, strict=True
            ):
                control.setRange(minimum, maximum)
                control.setValue(value)
                control.setEnabled(enabled)
        finally:
            self._suppress_slice_changes = False

    def _set_slice_controls(self, values, snapshot):
        state = snapshot.state
        self._suppress_slice_changes = True
        try:
            for label, control, symbol, size, index in zip(
                self.slice_index_labels,
                self.slice_index_controls,
                values.coordinate_frame.axis_symbols,
                values.shape,
                (state.h_index, state.k_index, state.l_index),
                strict=True,
            ):
                label.setText(symbol)
                control.setRange(0, size - 1)
                control.setValue(index)
                control.setEnabled(True)
        finally:
            self._suppress_slice_changes = False

    def _replace_surface(self, values, model, snapshot):
        staged = self._stage_surface(snapshot)
        titles = self._surface_titles(snapshot)
        axis_labels = self._surface_axis_labels(values)
        facts = self._surface_facts(values, snapshot)
        old_items = self.surface_items
        old_titles = self._surface_title_texts
        old_axis_labels = self._surface_axis_label_texts
        old_slice_labels = tuple(label.text() for label in self.slice_index_labels)
        old_facts = self.result_facts.text()
        old_fingerprint = self._painted_result_fingerprint
        old_values = self._viewer_values
        old_model = self._viewer_model
        old_snapshot = self._painted_snapshot
        old_controls = self._slice_control_state()
        self.setUpdatesEnabled(False)
        try:
            for plot, item in zip(self.surface_plots, staged, strict=True):
                plot.addItem(item)
            for plot, item in zip(self.surface_plots, old_items, strict=True):
                plot.removeItem(item)
            for plot, title in zip(self.surface_plots, titles, strict=True):
                plot.setTitle(title)
            for plot, (horizontal, vertical) in zip(
                self.surface_plots,
                axis_labels,
                strict=True,
            ):
                plot.setLabel("bottom", horizontal)
                plot.setLabel("left", vertical)
            self.result_facts.setText(facts)
            self._set_slice_controls(values, snapshot)
            self.surface_items = staged
            self._surface_title_texts = titles
            self._surface_axis_label_texts = axis_labels
            self.slice_images = list(staged[:3])
            self._painted_result_fingerprint = values.result_fingerprint
            self._viewer_values = values
            self._viewer_model = model
            self._painted_snapshot = snapshot
        except BaseException as primary:
            rollback_errors = []
            for plot, item in reversed(
                tuple(zip(self.surface_plots, staged, strict=True))
            ):
                try:
                    if item in plot.getPlotItem().items:
                        plot.removeItem(item)
                except BaseException as error:
                    rollback_errors.append(error)
            for plot, item in reversed(
                tuple(zip(self.surface_plots, old_items, strict=True))
            ):
                try:
                    if item not in plot.getPlotItem().items:
                        plot.addItem(item)
                except BaseException as error:
                    rollback_errors.append(error)
            for plot, title in zip(self.surface_plots, old_titles, strict=True):
                try:
                    plot.setTitle(title)
                except BaseException as error:
                    rollback_errors.append(error)
            for plot, (horizontal, vertical) in zip(
                self.surface_plots,
                old_axis_labels,
                strict=True,
            ):
                try:
                    plot.setLabel("bottom", horizontal)
                    plot.setLabel("left", vertical)
                except BaseException as error:
                    rollback_errors.append(error)
            self.surface_items = old_items
            self._surface_title_texts = old_titles
            self._surface_axis_label_texts = old_axis_labels
            self.slice_images = list(old_items[:3])
            self._painted_result_fingerprint = old_fingerprint
            self._viewer_values = old_values
            self._viewer_model = old_model
            self._painted_snapshot = old_snapshot
            try:
                self.result_facts.setText(old_facts)
            except BaseException as error:
                rollback_errors.append(error)
            try:
                self._restore_slice_controls(old_controls)
                for label, text in zip(
                    self.slice_index_labels,
                    old_slice_labels,
                    strict=True,
                ):
                    label.setText(text)
            except BaseException as error:
                rollback_errors.append(error)
            for error in rollback_errors:
                try:
                    primary.add_note(
                        "RSM presentation rollback also failed: "
                        f"{type(error).__module__}.{type(error).__qualname__}: {error}"
                    )
                except BaseException:
                    break
            raise primary.with_traceback(primary.__traceback__)
        finally:
            self.setUpdatesEnabled(True)
            self.update()

    def _paint_result(self, result):
        if type(result) is not RSMOperationResultV2:
            raise TypeError("RSM painter requires an exact v2 operation result")
        if (
            result.terminal.disposition is not ModuleDisposition.COMMITTED
            or result.payload is None
        ):
            raise ValueError("RSM painter accepts only a strict committed payload")
        values = make_rsm_viewer_values(result.payload)
        model = RSMViewerModel()
        snapshot = model.snapshot(values)
        self._replace_surface(values, model, snapshot)

    def _slice_index_changed(self, *_args):
        if self._suppress_slice_changes:
            return
        values = self._viewer_values
        model = self._viewer_model
        old_snapshot = self._painted_snapshot
        if values is None or model is None or old_snapshot is None:
            return
        indices = tuple(control.value() for control in self.slice_index_controls)
        try:
            snapshot = model.snapshot(
                h_index=indices[0], k_index=indices[1], l_index=indices[2]
            )
            self._replace_surface(values, model, snapshot)
        except Exception as error:
            try:
                model.snapshot(
                    h_index=old_snapshot.state.h_index,
                    k_index=old_snapshot.state.k_index,
                    l_index=old_snapshot.state.l_index,
                )
            except Exception:
                logger.exception("RSM viewer cache restore failed")
            self._set_slice_controls(values, old_snapshot)
            self._notice(
                f"RSM_VIEW_PRESENTATION_FAILED: presentation failed: {error}"
            )

    def _clear_painted_result(self, detail="No committed RSM loaded"):
        self.setUpdatesEnabled(False)
        try:
            for item in self.surface_items:
                item.clear()
            self.result_facts.setText(str(detail))
            self._painted_result_fingerprint = None
            self._painted_snapshot = None
            self._viewer_values = None
            self._viewer_model = None
            self._suppress_slice_changes = True
            try:
                for control in self.slice_index_controls:
                    control.setRange(0, 0)
                    control.setValue(0)
                    control.setEnabled(False)
            finally:
                self._suppress_slice_changes = False
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
            and self._active_action
            in {RSMOwnerAction.PREFLIGHT, RSMOwnerAction.RUN}
        )
        self.retry_cleanup_button.setEnabled(
            not busy and finalization is RSMOwnerFinalization.CLEANUP_PENDING
        )
        self.retry_verification_button.setEnabled(
            not busy and finalization is RSMOwnerFinalization.VERIFICATION_PENDING
        )
        self._sync_member_actions()

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
                for item in self.surface_items:
                    item.clear()
                self.preview_text.clear()
                self.result_facts.setText("No committed RSM loaded")
                self._painted_result_fingerprint = None
                self._painted_snapshot = None
                self._viewer_values = None
                self._viewer_model = None
                for control in self.slice_index_controls:
                    control.setEnabled(False)
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
