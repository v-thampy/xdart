"""Native RSM workspace page (AUD-9, stage 1) over the RSM v2 owner.

The maintainer, 2026-09-30: "Can you start aud 9 in a separate branch". The page is the RSM dialog's
workflow in the main workspace, on the shared components as the Stitch page has them: Int's Controls panel
and run bar, its section actions (Preview · Retry · Advanced), and its scientific panes for the six
products. The RSM v2 owner (``tools.rsm_owner.RSMToolOwner.v2``) is the one operation owner: the page builds
its exact ``RSMToolFormV2`` and shows what the owner returns; it holds no operation state or science of
its own. A saved result is reopened on the shared single-worker owner, off the GUI thread.

Stage 2 (the maintainer, 2026-10-01: "This should have the same replace append option like stitch"): the
run bar's Append ⇄ Replace is Stitch's. The operation replaces its same-kind ``_rsm.nexus`` in place (a
foreign or other-kind file there stays refused), and a Run in Append skips an output already gridded with
the same settings over the same frames. Preview creates no folder: the run makes the output's.

Stage 2b (design §2; the maintainer, 2026-10-01: "the control panel can be almost identical to the stitch control
panel"): Experiment is Stitch's (Calibrate · Make Mask in the shared windows, Calibration, Mask File), and Source
is Stitch's rows over Stitch's own source owner (``rsm_member_groups``), its scans found on the shared single
worker before Preview and Run.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import logging
from pathlib import Path
import re

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.pages.handle import AppMenuHosts, PageHandle
from xdart.gui.pages.operation_owner import OperationTerminalStatus, SingleWorkerOwner
from xdart.gui.pages.values import (
    ActionAccepted, ActionCompleted, ActionRefused, CloseReceipt, PageCleanup,
    RSM_TOOL_KEY, SCATTERING_PAGE_KEY, STITCH_TOOL_KEY,
)
from xdart.gui.tabs.rsm.display import rsm_keyed_projection, rsm_slice_normals
from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
from xdart.gui.tabs.scattering.browser_view import BrowserProfile, BrowserResult, BrowserView
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tabs.scattering.controls_inventory import (
    MASK_FILE, PONI_FILE, SOURCE_DIRECTORY, SOURCE_FILTER, SOURCE_RECURSIVE, SOURCE_TYPE,
)
from xdart.gui.tabs.scattering.controls_readiness import SectionHeaderProjection
from xdart.gui.tabs.scattering.shell_values import ShellCommandKind
from xdart.gui.tabs.scattering.shell_widgets import apply_section_header
from xdart.gui.tabs.scattering.workspace_shell import _HugWidthScrollArea
from xdart.gui.tools.rsm_owner import (
    RSMOwnerAction, RSMOwnerFinalization, RSMOwnerOutcomeKind, RSMOwnerUpdate, RSMToolOwner,
)
from xdart.gui.tools.rsm_values import (
    RSMToolFormV2, format_rsm_angle_mapping, parse_rsm_angle_mapping, rsm_frame_selector,
    rsm_preflight_summary_lines, rsm_tool_preset,
)
from xdart.gui.widgets.controls_panel import ControlsPanel, FormRow
from xdart.gui.widgets.run_controls import RunControlsBar
from xrd_tools.analysis.module_transaction import MetadataColumnSelector, ModuleDisposition
from xrd_tools.analysis.rsm_geometry_asset import rsm_calibration_geometry_input
from xrd_tools.analysis.rsm_operation import (
    RSMFrameCorrections, RSMImageConditioning, RSMNormalizationMode, RSMNormalizationPolicy, RSMOperationResultV2,
)
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from xrd_tools.session.readiness import (
    ControlAction, ControlActionSpec, ControlFieldKind, ControlFormField, SectionId,
)
from xrd_tools.session.rsm_viewer_model import RSMViewerModel, make_rsm_viewer_values

_LOG = logging.getLogger(__name__)
_MIB = 1024 * 1024
#: The grids the RSM dialog offers (the scan-43 preset's quick and full).
_GRIDS = {"Quick 40×40×40": (40, 40, 40), "Full 200×200×200": (200, 200, 200)}
_NORMALIZATIONS = {"Foil transmission / exposure": RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE,
                   "Identity": RSMNormalizationMode.IDENTITY}
_FRAMES = {frame.display_name: frame for frame in RSMCoordinateFrame}
#: The run bar's modes this page runs.
_RSM_MODES = ("RSM",)
#: Int's Data Browser as a results profile: the page's File actions, Config and Help for the host's menus, the
#: Results list and Open… (hard rule 14; Codex's RSM reuse audit R1)
_BROWSER = BrowserProfile(
    menus=(("File", ("Open result…", "Integration workspace")), ("Config", ()), ("Help", ())),
    raw_resolution_menus=False, list_caption="Results", directory=False, frames=False, actions=("Open…",),
    date_sort=False, refresh=False, results=True,
)
#: The operation's stages in words (its progress counts frames plus the masks and four steps).
_STAGE_TEXT = {"revalidate": "Checking the inputs", "static-mask": "Masking hot pixels", "science": "Gridding",
               "projection": "Projecting", "publish": "Saving", "reload": "Checking the saved result"}
#: Results kept in memory at once (the maintainer, 2026-10-01: "I'm happy keeping even more, about 4-5 for
#: quick access"): float32, 32 MB each at Full 200³ and 160 MB for five; the shown one's viewer values share
#: its buffer, and its six products are 0.5 MB (probe 3). Another is read again from its file, off the GUI
#: thread, when it is shown.
_RESIDENT_RESULTS = 5
#: The page's fields: (card, key, label, default, choices, browse).
_FIELDS = (
    (SectionId.PROJECT, "project_root", "Folder", "", (), True),
    (SectionId.PROJECT, "output_path", "Save Path", "", (), True),
    (SectionId.PROCESSING, "coordinates", "Coordinates", RSMCoordinateFrame.HKL.display_name, tuple(_FRAMES),
     False),
    (SectionId.PROCESSING, "grid", "Grid", "Quick 40×40×40", tuple(_GRIDS), False),
    (SectionId.PROCESSING, "normalization", "Normalization", "Foil transmission / exposure",
     tuple(_NORMALIZATIONS), False),
    (SectionId.PROCESSING, "foil", "Foil column", "foil status", (), False),
    (SectionId.PROCESSING, "exposure", "Exposure column", "Seconds", (), False),
    (SectionId.PROCESSING, "absorption", "Foil absorption", "1.06, 3.04, 4.65, 9.5", (), False),
    (SectionId.PROCESSING, "offset", "Offset", "1e-06", (), False),
    (SectionId.PROCESSING, "high_threshold", "High → NaN", "20000000000", (), False),
    (SectionId.PROCESSING, "static_hot", "All-frame hot", "100", (), False),
)
#: The whole-number rows (FormRow kind "int"): (minimum, maximum, blank allowed). The row outlines a refused
#: text; ``rsm_values`` stays the authority that refuses it (Codex's stage-1 review F1). The frame range is
#: Stitch's one range for every member, in Advanced (design §2).
_INTEGER_FIELDS = {"frame_first": (0, None, False), "frame_last": (0, None, True), "frame_step": (1, None, False),
                   # Advanced (the dialog's spin boxes' ranges, as rows)
                   "adv_rows": (2, 1_000_000, False), "adv_columns": (2, 1_000_000, False),
                   "adv_header": (0, 1 << 30, False),
                   "adv_foil_occurrence": (0, 1024, False), "adv_exposure_occurrence": (0, 1024, False),
                   "adv_chunk": (1, 1024, False), "adv_max_frame_mib": (1, 4096, False),
                   "adv_max_chunk_mib": (1, 4096, True)}
#: Stitch's Source kinds (the maintainer: "nexus files as source as well. And directory mode just like stitch").
_SOURCE_KINDS = ("Spec", "NeXus", "Directory")
#: One RSM per run for now (Stitch queues one output per group; the RSM page does not yet).
_ONE_GROUP = "one RSM per run for now: brackets combine scans into one RSM, as [42-44]"
_RAW_DTYPES = ("int32", "uint16", "float32", "float64")


def _resolve_rsm_groups(arguments, cancel, _progress):
    """The member groups of a source selection (``rsm_member_groups``: Stitch's own discovery, the Selection
    syntax and Directory mode), off the GUI thread: it reads SPEC files and lists images."""
    from xdart.gui.tools.rsm_values import rsm_member_groups
    if cancel.is_set():
        return None
    selection, options = arguments
    return rsm_member_groups(selection, **options)


def _read_rsm_record(path, cancel, _progress):
    """A saved RSM record, strictly admitted and detached (the shared reader's runner)."""
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, read_analysis_artifact
    if cancel.is_set():
        return None
    return read_analysis_artifact(Path(path), expected_kind=AnalysisArtifactKind.RSM)


def _progress_text(stage, completed, total):
    """A run's progress in words."""
    if stage == "science" and total > 4:
        return f"Gridding · {completed} of {total - 4}"
    return f"{_STAGE_TEXT.get(stage, stage)}…"


def _family_pair(root, member):
    """A member's (source path, scan or None) as the slot policy names one: a SPEC scan's file and key, a NeXus
    stack's file alone (stage 2b C5c)."""
    if isinstance(member, tuple):
        path, scan = member
    elif hasattr(member, "nexus_path"):
        path, scan = member.nexus_path, None
    else:
        path, scan = member.spec_path, member.scan
    return Path(root) / path, scan


def suggested_family(project_root, members):
    """The family a page suggests for its ordered members (forms of either kind, or (path, scan) pairs), as the
    slot policy derives it from them (``rsm`` when it cannot)."""
    from xrd_tools.io.output_path import automatic_artifact_family, is_artifact_family
    root = Path(project_root)
    if not members:
        return "rsm"
    try:
        family = automatic_artifact_family(tuple(_family_pair(root, member) for member in members), root)
    except (ValueError, OSError):
        family = ""
    return family if is_artifact_family(family) else "rsm"


def suggested_output(project_root, members):
    """The output a page suggests for its ordered members: the stable slot ``<Project>/xdart_processed_data/
    <family>_rsm.nexus`` of ``suggested_family``."""
    from xrd_tools.io.output_path import resolve_finite_output_target
    return str(resolve_finite_output_target(Path(project_root) / "xdart_processed_data",
                                            suggested_family(project_root, members), operation_token="rsm"))


class RSMPage(QtWidgets.QWidget):
    """The native RSM workspace."""

    toolRequested = QtCore.Signal(str)
    #: (page key, processing mode): open that page in that mode
    modeRequested = QtCore.Signal(str, str)

    def __init__(self, services, parent=None, *, owner=None, reader=None, resolver=None):
        super().__init__(parent)
        self.setObjectName("rsmPage")
        self._services = services
        self._owner = RSMToolOwner.v2() if owner is None else owner
        self._reader = SingleWorkerOwner(_read_rsm_record) if reader is None else reader
        self._reading = None  # (identity, path) of the record being read
        # the Source's member groups, found on the shared single worker before Preview and Run (it reads the
        # SPEC files and lists the images): (identity, action, key) while found; (key, groups, skipped) after
        self._resolver = SingleWorkerOwner(_resolve_rsm_groups) if resolver is None else resolver
        self._resolving = None
        self._groups = None
        self._fields, self._rows = {}, {}
        self._source_paths = ()  # Stitch's selection: the SPEC or NeXus files, or the folder, as chosen
        self._source_shown = ""  # the SPEC files row's text for them ("3 files selected")
        self._paths_by_kind = {}
        self._readiness = ("", False)
        self._output_suggested = True  # the output follows the Source until the user names one
        self._run_requested = False
        self._form_revision = 0
        self._active_revision = None
        self._prepared_fingerprint = None
        self._results = OrderedDict()  # path → payload, the most recent last
        self._listed = []  # the Results list's paths, in the order they were first listed
        self._requested = None  # the result last asked to be shown: a late read shows only it (Codex F4)
        # each resident result's viewer values and model, built once (Codex's reuse audit rec. 3): showing it
        # again recomputes nothing; dropped with its payload when it leaves the five resident
        self._viewers = {}  # path → (payload, values, model)
        self._shown_path = None
        self._shown = None  # (values, snapshot) drawn now
        self._color_map = "Default"  # Int's top bar: viridis
        self._log_scale = False
        self._closing = False
        # the shared calibration and mask windows (tabs/calibration), opened with RSM's context
        self._authoring_dialogs = []
        self._calibration_dialog = None
        self._calibration_project = None
        self._build_ui()
        self._authoring_timer = QtCore.QTimer(self)
        self._authoring_timer.setInterval(50)
        self._authoring_timer.timeout.connect(self._sync_actions)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)
        self._source_changed()
        self._edited()

    # ── layout ──
    def _build_ui(self):
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(self.splitter)
        self.browser = BrowserView(profile=_BROWSER)  # Int's Data Browser (hard rule 14)
        self.browser.commandRequested.connect(self._browser_command)
        self.splitter.addWidget(self.browser)

        # Int's scientific view, its keyed 2×3 layout (hard rule 14): the slices and mean projections under
        # Int's top bar (colour map, Log); the tools RSM has no operation for are left out
        self.view = ScientificView()
        self.view.setObjectName("rsmPageView")
        self.view.set_hidden_tools("norm", "background", "raw", "raw_pane", "slice")
        # the maintainer's two extra tools, "as extra tools that pop up separate windows to keep the scientific
        # view clean for now"; their windows come with G9
        self.view.set_page_tools(
            ("3d", "3D Viewer", "The whole RSM volume in 3-D, in its own window"),
            ("custom", "Custom Slice", "A slice along any two directions (for example HH and L), in its own window"))
        self.view.commandRequested.connect(self._view_command)
        self.splitter.addWidget(self.view)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.controls = ControlsPanel(experiment_title="Configuration", show_section_numbers=False)  # Int's
        self.controls.experiment_card.title.setText("EXPERIMENT")
        self.controls.controlActionRequested.connect(self._control_action)
        scroll = _HugWidthScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.controls)
        right_layout.addWidget(scroll, 1)
        self._build_controls()
        self.run_bar = RunControlsBar()
        from xdart.gui.tabs.scattering.run_mode_projection import (
            RUN_MODE_CHOICES, UNOWNED_RUN_MODE_REASONS,
        )
        self.run_bar.apply_profile(modes=RUN_MODE_CHOICES, live=False, batch=False, cores=False)
        self.run_bar.set_unavailable_modes(UNOWNED_RUN_MODE_REASONS)
        self.run_bar.modeCombo.setCurrentText("RSM")
        self.run_bar.modeChanged.connect(self._mode_chosen)
        # Stitch's Append ⇄ Replace: Append skips an output already gridded with these settings, Replace
        # grids it again (the maintainer, 2026-10-01: "the same replace append option like stitch")
        self.run_bar.writeModeChanged.connect(self._write_mode_changed)
        self._write_mode_changed()
        self.run_bar.actionClicked.connect(self.begin_run)
        self.run_bar.stopClicked.connect(self.stop)
        right_layout.addWidget(self.run_bar)
        self.splitter.addWidget(right)  # the browser keeps Int's own minimum width (255)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.setStretchFactor(2, 1)
        self.splitter.setSizes([289, 868, 323])  # Int's columns, so the pages switch in place

    def _write_mode_changed(self, _mode=None):
        self.run_bar.writeModeButton.setToolTip(
            "Append: skip the output when it was already gridded from the same frames with the same settings"
            if self.run_bar.write_mode() == "Append" else
            "Replace: grid again; the output is replaced once its new result is written")

    @staticmethod
    def _new_field(section, key, label, value, choices=(), browse=False, path=None):
        """One of the page's fields; ``path`` Int's for a row Int's composer lays out."""
        kind = (ControlFieldKind.COMBO if choices else ControlFieldKind.BOOL if isinstance(value, bool)
                else ControlFieldKind.INT if key in _INTEGER_FIELDS else ControlFieldKind.LINE)
        if path is None:
            path = ("Project", "project_folder") if key == "project_root" else ("RSM", key)
        minimum, maximum, allow_blank = _INTEGER_FIELDS.get(key, (None, None, False))
        return ControlFormField(section, label, path, value, kind, tuple(choices), browse,
                                minimum=minimum, maximum=maximum, allow_blank=allow_blank)

    def _bind(self, key, field, row):
        """A row the page reads and edits by ``key``, built here or by Int's composer."""
        row.editor.setProperty("rsmField", key)
        row.valueChanged.connect(lambda _path, _value, name=key: self._committed(name))
        row.browseRequested.connect(lambda _path, name=key: self._browse(name))
        self._fields[key], self._rows[key] = field, row
        return row

    def _row(self, section, key, label, value, choices, browse):
        field = self._new_field(section, key, label, value, choices, browse)
        return self._bind(key, field, FormRow(
            label=label, path=field.path, value=value, kind=field.kind.value, choices=choices, browse=browse,
            minimum=field.minimum, maximum=field.maximum, allow_blank=field.allow_blank))

    def _build_controls(self):
        cards = {SectionId.PROJECT: self.controls.project_card, SectionId.EXPERIMENT: self.controls.experiment_card,
                 SectionId.SOURCE: self.controls.source_card, SectionId.PROCESSING: self.controls.processing_card}
        self._build_experiment()  # Stitch's Experiment rows (the maintainer: its control panel)
        self._build_source()  # and its Source rows
        for section, key, label, value, choices, browse in _FIELDS:
            cards[section].add_row(self._row(section, key, label, value, choices, browse))
        buttons = self.controls.render_actions(self.controls.processing_card, tuple(
            ControlActionSpec(action, label, SectionId.PROCESSING) for action, label in (
                (ControlAction.PREVIEW, "Preview"), (ControlAction.RETRY, "Retry"),
                (ControlAction.ADVANCED_PROCESSING, "Advanced"))), expand=True)
        self.preview_button, self.retry_button, self.advanced_button = (
            buttons[action] for action in (ControlAction.PREVIEW, ControlAction.RETRY,
                                           ControlAction.ADVANCED_PROCESSING))
        self.preview_button.setToolTip("Check the inputs and list every exact identity, without gridding.")
        self.retry_button.setToolTip("Retry the last result's saving or its check.")
        self._build_advanced()
        self.preview_dialog = QtWidgets.QDialog(self)
        self.preview_dialog.setWindowTitle("RSM preview")
        preview_layout = QtWidgets.QVBoxLayout(self.preview_dialog)
        self.preview_text = QtWidgets.QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        preview_layout.addWidget(self.preview_text)
        self.preview_dialog.resize(760, 560)

    def _build_experiment(self):
        """Stitch's Experiment rows, from Int's composer (no Grazing: the maintainer): Calibrate · Make Mask on
        top, the Detector subsection (the Calibration and the Mask File), then the calibration's refinement. RSM
        calibrates its own data in the shared calibration window (the maintainer, 2026-10-01: "the idea is to
        reuse what was developed for stitch, not use the calibration that was done in stitch"), and its geometry
        is the Calibration row's (AUD-9 stage 2b C7b)."""
        experiment = self.controls.experiment_card
        detector = {"geometry_path": self._new_field(SectionId.EXPERIMENT, "geometry_path", "Calibration", "",
                                                     browse=True, path=PONI_FILE),
                    "mask_path": self._new_field(SectionId.EXPERIMENT, "mask_path", "Mask File", "", browse=True,
                                                 path=MASK_FILE)}
        built = self.controls.render_experiment_rows(experiment, tuple(detector.values()), (
            ControlActionSpec(ControlAction.CALIBRATE, "Calibrate", SectionId.EXPERIMENT),
            ControlActionSpec(ControlAction.MAKE_MASK, "Make Mask", SectionId.EXPERIMENT)))
        for key, field in detector.items():
            self._bind(key, field, built[field.path])
        self.calibrate_button, self.mask_button = built[ControlAction.CALIBRATE], built[ControlAction.MAKE_MASK]
        self.calibrate_button.setToolTip("Calibrate the detector from a calibrant or direct-beam scan in the "
                                         "shared calibration window; its saved file becomes the Calibration.")
        self.mask_button.setToolTip("Draw a detector mask on one of the scan's frames.")
        self._tip("geometry_path", "The calibration window's pyFAI file (its xrayutilities partner, "
                                   "<name>.xu.json, is saved beside it): the distance, beam pixel, pixel size "
                                   "and camera. Inside the Project.")
        self._tip("mask_path", "A detector mask, applied to every frame with the detector's own.")
        refinement = self._row(SectionId.EXPERIMENT, "refinement", "Use the calibration's refinement", True,
                               (), False)
        refinement.setToolTip("When the calibration carries a refined fit: its del/nu zero offsets, detector "
                              "mount tilts and energy offset (each scan keeps its own recorded energy plus the "
                              "offset). Never the calibrant's sample offset.")
        experiment.add_row(refinement)

    def _build_source(self):
        """Stitch's Source rows, from Int's composer (design §2; the maintainer, 2026-10-01: "nexus files as
        source as well. And directory mode just like stitch"): the kind with Directory's Subdirs on its row, the
        SPEC or NeXus files or the folder, Directory's Filter, the Images folder, the Image stem, and the Scans
        with their Selection. The Selection's syntax is Stitch's: ``[42-44]`` makes one RSM of three scans.
        Every row is built once; a kind shows its own (``_source_changed``)."""
        fields = {
            "source_kind": self._new_field(SectionId.SOURCE, "source_kind", "Source", "Spec", _SOURCE_KINDS,
                                           path=SOURCE_TYPE),
            "spec_path": self._new_field(SectionId.SOURCE, "spec_path", "SPEC files", "", browse=True,
                                         path=SOURCE_DIRECTORY),
            "subdirs": self._new_field(SectionId.SOURCE, "subdirs", "Subdirs", False, path=SOURCE_RECURSIVE),
            "name_filter": self._new_field(SectionId.SOURCE, "name_filter", "Filter", "", path=SOURCE_FILTER),
            "image_dir": self._new_field(SectionId.SOURCE, "image_dir", "Images", "", browse=True),
            # on the card, not in Advanced: a stem left from another dataset matches the wrong images
            "image_stem": self._new_field(SectionId.SOURCE, "image_stem", "Image stem", ""),
            "scan_mode": self._new_field(SectionId.SOURCE, "scan_mode", "Scans", "Select Scans",
                                         ("All Scans", "Select Scans")),
            "scan": self._new_field(SectionId.SOURCE, "scan", "Selection", "1"),
        }
        rows = self.controls.render_source_rows(self.controls.source_card, tuple(fields.values()))
        for key, field in fields.items():
            self._bind(key, field, rows[field.path])
        self._tip("source_kind", "Source kind: SPEC files and their scans' images, raw NeXus files, or a folder "
                                 "of both.")
        self._tip("subdirs", "Also the SPEC and NeXus files in the folder's subfolders.")
        self._tip("name_filter", "Only the files whose names match: a space is and, | is or, -term is not.")
        self._tip("image_dir", "The folder of the scans' detector images. Blank: the SPEC file's folder.")
        self._tip("image_stem",
                  "The part of the image file names that identifies the scan; any prefix before it is "
                  "fine (b_thampy_R1C2_scan3_0000.raw has R1C2_scan3_). Blank: <SPEC name>_scan<N>_ for "
                  "each scan. {spec} and {scan} fill in the SPEC name and scan number.")
        self._tip("scan_mode", "All Scans: each scan of the files is its own RSM. Select Scans: the Selection.")
        self._tip("scan", "Scan numbers, as Stitch's Selection: brackets combine scans into one RSM ([42-44] "
                          "is one RSM of three scans); 42,43 would be two RSMs, and the page grids one RSM per "
                          "run for now.")

    def _build_advanced(self):
        """The rarely changed settings: Int's Advanced dialog, its groups the page's shared rows (Codex's reuse
        audits, 2026-10-01). Stitch's Source and geometry rows (design §2): one raw decoder, one Angle mapping
        and one frame range for every scan, as Stitch's."""
        preset = rsm_tool_preset()
        self.advanced = AdvancedDialog(self, title="Advanced — RSM", object_name="rsmAdvancedDialog")
        groups = (
            ("Source and geometry", SectionId.SOURCE, (
                ("adv_rows", "Rows", str(preset.detector_shape[0]), ()),
                ("adv_columns", "Columns", str(preset.detector_shape[1]), ()),
                ("adv_dtype", "Data type", preset.raw_dtype, _RAW_DTYPES),
                ("adv_header", "Header bytes", str(preset.raw_header_skip), ()),
                ("adv_angle_mapping", "Angle mapping", format_rsm_angle_mapping(preset.motor_selectors), ()),
                ("frame_first", "First frame", "0", ()),
                ("frame_last", "Last frame", "", ()),
                ("frame_step", "Frame step", "1", ()))),
            ("Normalization column occurrences", SectionId.PROCESSING, (
                ("adv_foil_occurrence", "Foil column occurrence", "0", ()),
                ("adv_exposure_occurrence", "Exposure column occurrence", "0", ()))),
            ("Chunk and memory", SectionId.PROCESSING, (
                ("adv_chunk", "Frames per chunk", str(preset.chunk_size), ()),
                ("adv_max_frame_mib", "Max frame MiB", str(preset.max_frame_bytes // _MIB), ()),
                ("adv_max_chunk_mib", "Max chunk MiB", "", ()))),
        )
        for title, section, rows in groups:
            self.advanced.add_group(title, tuple(self._row(section, key, label, value, choices, False)
                                                 for key, label, value, choices in rows))
        self._tip("adv_rows", "The raw images' rows (a NeXus stack's frames are this shape too).")
        self._tip("adv_header", "Bytes before the pixels in each raw image file.")
        self._tip("adv_angle_mapping", "Which source column each psic circle reads, as role=column: mu, eta, chi, "
                                       "phi, nu and del, each once. A column the scan repeats takes its "
                                       "occurrence after # (nu=Nu#1 is the second Nu; #0, the first, is the "
                                       "default).")
        self._tip("frame_first", "The first frame of every scan (0 is the first).")
        self._tip("frame_last", "The last frame of every scan; blank is each scan's last.")
        self._tip("frame_step", "Every n-th frame from the first.")
        self._tip("adv_max_chunk_mib", "Blank is Auto: 256 MiB, or one frame's need when larger (an Eiger 4M), "
                                       "the chunk then holding as many frames as fit. A number is your cap; one "
                                       "below a frame's need is named.")

    def _tip(self, key, text):
        self._rows[key].setToolTip(text)

    def _advanced_refusal(self, section=None):
        """The first refused Advanced whole-number row, as readiness and ``build_form`` name it ("" if none)."""
        for key in _INTEGER_FIELDS:
            if (key.startswith(("adv_", "frame_")) and (section is None or self._fields[key].section is section)
                    and self._rows[key].refusal()):
                return f"{self._fields[key].label} ({self._rows[key].refusal()})"
        return ""

    # ── values ──
    def _value(self, key):
        return self._rows[key].current_value()

    def set_field(self, key, value):
        self._fields[key] = replace(self._fields[key], value=value)
        self._rows[key].apply_field(self._fields[key])
        self._committed(key)

    def _committed(self, key):
        """A committed edit: a typed source path is the selection, a kind shows its own rows; the output
        follows the Source until a name is chosen."""
        if key == "spec_path":
            text = str(self._value("spec_path") or "").strip()
            if text != self._source_shown:  # the row's own display of the selection is not a path (Stitch's)
                self._adopt_sources((text,) if text else ())
                self._source_shown = text
                self._show_source_tip()
        elif key in {"source_kind", "scan_mode"}:
            self._source_changed(key)
        if key == "output_path":
            self._output_suggested = not str(self._value("output_path")).strip()
        self._edited()

    # ── the Source (Stitch's rows and owner) ──
    def set_sources(self, paths):
        """The SPEC or NeXus files, or the folder (Directory), as Stitch's ``set_sources``: one is shown, more
        as "N files selected"."""
        self._adopt_sources(paths)
        count = len(self._source_paths)
        self._source_shown = ("" if not count else self._source_paths[0] if count == 1
                              else f"{count} files selected")
        self._fields["spec_path"] = replace(self._fields["spec_path"], value=self._source_shown)
        self._rows["spec_path"].apply_field(self._fields["spec_path"])
        self._show_source_tip()
        self._edited()

    def _adopt_sources(self, paths):
        self._source_paths = tuple(str(path) for path in paths if str(path).strip())
        self._paths_by_kind[self._value("source_kind")] = self._source_paths

    def _show_source_tip(self):
        row = self._rows["spec_path"]
        for widget in (row, row.editor, row.label):
            widget.setToolTip("\n".join(self._source_paths))

    def _source_changed(self, key=None):
        """A kind shows its own rows (Stitch's): Images and the stem for SPEC scans (a Directory's too), the
        Scans and Selection for SPEC files, Subdirs and Filter for a folder. A new kind takes back its own
        files."""
        kind = self._value("source_kind")
        for name in ("image_dir", "image_stem"):
            self._rows[name].setVisible(kind != "NeXus")
        self._rows["scan_mode"].setVisible(kind == "Spec")
        self._rows["scan"].setVisible(kind == "Spec" and self._value("scan_mode") == "Select Scans")
        for name in ("subdirs", "name_filter"):
            self._rows[name].setVisible(kind == "Directory")
        self._rows["spec_path"].label.setText({"Spec": "SPEC files", "NeXus": "NeXus files",
                                               "Directory": "Folder"}[kind])
        if key == "source_kind":
            self.set_sources(self._paths_by_kind.get(kind, ()))

    def _source_files(self, root):
        """The chosen files or folder, a relative one inside the Project (the selection would take it from
        the working directory)."""
        return tuple(str(self._inside(root, path, "source")) for path in self._source_paths)

    def _group_key(self):
        """Everything the Source's member groups are found from, as committed (the cache's key)."""
        return (str(self._value("project_root")).strip(), self._source_paths,
                *(str(self._value(key)) for key in (
                    "source_kind", "scan_mode", "scan", "image_dir", "image_stem", "subdirs", "name_filter",
                    "frame_first", "frame_last", "frame_step", "adv_rows", "adv_columns", "adv_dtype",
                    "adv_header", "adv_angle_mapping")))

    def _resolution_request(self):
        """(Stitch's source selection, the members' options) for ``rsm_member_groups``: the Project's paths
        absolute, one raw decoder, frame range and Angle mapping for every scan. ValueError names the row."""
        from xrd_tools.analysis.stitch_sources import StitchSourceSelection
        project = str(self._value("project_root")).strip()
        if not project:
            raise ValueError("Project root is required")
        root = Path(project).expanduser()
        refused = self._advanced_refusal(SectionId.SOURCE)
        if refused:
            raise ValueError(f"Advanced {refused}")
        kind = self._value("source_kind")
        if not self._source_paths:
            raise ValueError({"Spec": "SPEC files", "NeXus": "NeXus files", "Directory": "a source folder"}[kind]
                             + " to read")
        images = str(self._value("image_dir")).strip()
        directory = kind == "Directory"
        selection = StitchSourceSelection(
            kind, self._source_files(root), scans=str(self._value("scan")),
            stitch_all=self._value("scan_mode") == "All Scans",
            image_dir=str(self._inside(root, images, "Images")) if images and kind != "NeXus" else "",
            image_stem=str(self._value("image_stem")).strip() if kind != "NeXus" else "",
            recursive=directory and bool(self._value("subdirs")),
            name_filter=str(self._value("name_filter")).strip() if directory else "")
        options = dict(
            frame_selector=rsm_frame_selector(self._value("frame_first"), self._value("frame_last"),
                                              self._value("frame_step")),
            detector_shape=(int(self._value("adv_rows")), int(self._value("adv_columns"))),
            raw_dtype=str(self._value("adv_dtype")), raw_header_skip=int(self._value("adv_header")),
            motor_selectors=parse_rsm_angle_mapping(self._value("adv_angle_mapping")))
        return selection, options

    def resolve_source(self):
        """The Source's member groups found now, on the calling thread (scripts and rows; Preview and Run find
        them on the shared single worker first): (groups, skipped). ``build_form`` reads them."""
        from xdart.gui.tools.rsm_values import rsm_member_groups
        key = self._group_key()
        selection, options = self._resolution_request()
        groups, skipped = rsm_member_groups(selection, **options)
        self._found(key, groups, skipped)
        return groups, skipped

    def _found(self, key, groups, skipped):
        """Member groups found for ``key``: kept for ``build_form``, and the suggested output follows them."""
        self._groups = (key, tuple(groups), tuple(skipped))
        if self._apply_suggestion():
            self._edited()

    def _group_members(self):
        """The one RSM's ordered members, from the groups found for the Source as it stands (ValueError
        naming what is missing or why it is not one RSM)."""
        from xdart.gui.tools.rsm_values import _MAX_RSM_MEMBERS
        found = self._groups
        if found is None or found[0] != self._group_key():
            raise ValueError("the Source's scans are not found yet (Preview finds them)")
        _key, groups, skipped = found
        if not groups:
            reason = ""
            if skipped:  # the first skip's file, scan and why (Stitch's skips)
                path, scan, why = skipped[0]
                reason = f" ({' '.join(part for part in (Path(path).name, str(scan)) if part)}: {why})"
            raise ValueError("the Source has no scan to grid" + reason)
        if len(groups) > 1:
            raise ValueError(f"{len(groups)} RSMs in the Source: {_ONE_GROUP}")
        if len(groups[0]) > _MAX_RSM_MEMBERS:
            raise ValueError(f"{len(groups[0])} scans in one RSM: at most {_MAX_RSM_MEMBERS}")
        return groups[0]

    def _suggestion_members(self, root):
        """The (path, scan) pairs the suggested output's family is named from: the found group's members when
        the Source was found as it stands, else the rows' own without reading (one SPEC file's selected group,
        or the NeXus files); none (the family ``rsm``) until then."""
        from xrd_tools.sources.grouping import parse_stitch_scan_groups
        found = self._groups
        if found is not None and found[0] == self._group_key():
            return found[1][0] if len(found[1]) == 1 else ()
        kind = self._value("source_kind")
        try:
            files = self._source_files(root)
            if kind == "NeXus":
                return tuple((path, None) for path in files)
            if kind == "Spec" and len(files) == 1 and self._value("scan_mode") == "Select Scans":
                groups = parse_stitch_scan_groups(str(self._value("scan")))
                return tuple((files[0], token) for token in groups[0]) if len(groups) == 1 else ()
        except ValueError:
            pass
        return ()

    def _apply_suggestion(self):
        """The suggested output while the user has named none; True when it changed."""
        project = str(self._value("project_root")).strip()
        if not (self._output_suggested and project):
            return False
        suggestion = suggested_output(project, self._suggestion_members(Path(project).expanduser()))
        if suggestion == self._value("output_path"):
            return False
        self._fields["output_path"] = replace(self._fields["output_path"], value=suggestion)
        self._rows["output_path"].apply_field(self._fields["output_path"])
        return True

    @staticmethod
    def _inside(project, text, name):
        """A path as the RSM dialog reads one: required; a relative one is inside the Project."""
        spelling = str(text).strip()
        if not spelling:
            raise ValueError(f"{name} is required")
        path = Path(spelling).expanduser()
        return path if path.is_absolute() else project / path

    def _calibration_locator(self, root):
        """The Calibration row's file, Project relative, as RSM's calibration input takes it; ValueError
        naming what is wrong."""
        text = str(self._value("geometry_path")).strip()
        if not text:
            raise ValueError("a Calibration (Calibrate, or choose its file)")
        path = Path(text).expanduser()
        try:
            locator = (path.as_posix() if not path.is_absolute()
                       else path.resolve().relative_to(Path(root).expanduser().resolve()).as_posix())
            rsm_calibration_geometry_input(locator)  # its exact Project-relative spelling (no ".."), no reads
        except (TypeError, ValueError, OSError):
            raise ValueError("a Calibration inside the Project") from None
        return locator

    def _mask_path(self, root):
        """The Mask File as the run reads it (None when blank)."""
        text = str(self._value("mask_path")).strip()
        return None if not text else str(self._inside(root, text, "Mask File"))

    def _max_chunk_bytes(self):
        """Advanced Max chunk MiB in bytes; None (blank) is Auto."""
        text = str(self._value("adv_max_chunk_mib")).strip()
        return None if not text else int(text) * _MIB

    def _chunk_cap_refusal(self):
        """The Max chunk MiB a user set below one frame's need, named as the run would refuse it ("" when
        Auto or enough). Arithmetic only (``rsm_v2_chunk_bytes``): no file is read; the raw decoder's shape
        stands for every frame, uncropped, so the need named is an upper bound."""
        from xrd_tools.analysis.rsm_operation import rsm_v2_chunk_bytes
        if any(self._rows[key].refusal() for key in ("adv_max_chunk_mib", "adv_rows", "adv_columns")):
            return ""  # the row's own refusal names it
        cap = self._max_chunk_bytes()
        if cap is None:
            return ""
        static_hot = bool(str(self._value("static_hot")).strip())
        corrections = RSMFrameCorrections(solid_angle=True, polarization_factor=0.99)
        shape = (int(self._value("adv_rows")), int(self._value("adv_columns")))
        need = rsm_v2_chunk_bytes(shape, shape, 1, static_hot=static_hot, corrections=corrections)
        mib = -(-need // _MIB)
        return "" if cap >= need else f"Max chunk MiB of {mib} or more (one frame needs {mib} MiB), or blank for Auto"

    def build_form(self):
        """The exact RSM v2 form of the page's inputs: the Source's one member group as found for it
        (``rsm_member_groups``: Preview and Run find it on the shared worker first, ``resolve_source`` on the
        calling thread; this never reads), the Calibration row's geometry (AUD-9 stage 2b C7b) and the Mask
        File (C4)."""
        refused = self._advanced_refusal()
        if refused:
            raise ValueError(f"Advanced: {refused}")
        project_text = str(self._value("project_root")).strip()
        if not project_text:
            raise ValueError("Project root is required")
        project = Path(project_text).expanduser()
        members = self._group_members()
        mode = _NORMALIZATIONS[self._value("normalization")]
        if mode is RSMNormalizationMode.IDENTITY:
            normalization = RSMNormalizationPolicy.identity()
        else:
            normalization = RSMNormalizationPolicy(
                mode, MetadataColumnSelector(str(self._value("foil")).strip(),
                                             int(self._value("adv_foil_occurrence"))),
                MetadataColumnSelector(str(self._value("exposure")).strip(),
                                       int(self._value("adv_exposure_occurrence"))),
                tuple(float(part) for part in str(self._value("absorption")).replace(";", ",").split(",")
                      if part.strip()))

        def optional(key):
            text = str(self._value(key)).strip()
            return None if not text else float(text)
        output = self._inside(project, str(self._value("output_path")).strip(), "output path")
        # the suggested output is its family's stable slot, and the result records that family
        family = (suggested_family(project, members)
                  if str(output) == suggested_output(project, members) else None)
        geometry = rsm_calibration_geometry_input(self._calibration_locator(project),
                                                  refinement=bool(self._value("refinement")))
        # decision 5's defaults (solid angle, polarization 0.99) with the Mask File
        corrections = RSMFrameCorrections(mask_path=self._mask_path(project), solid_angle=True,
                                          polarization_factor=0.99)
        return RSMToolFormV2(
            project_text, geometry, members,
            RSMImageConditioning(float(self._value("offset")), optional("high_threshold"), optional("static_hot")),
            normalization, tuple(_GRIDS[self._value("grid")]), int(self._value("adv_chunk")),
            int(self._value("adv_max_frame_mib")) * _MIB, self._max_chunk_bytes(), output,
            AnalysisArtifactOverwrite.REPLACE,  # Append is the page's skip, as Stitch's (_start_run)
            coordinate_frame=_FRAMES[self._value("coordinates")], output_family=family, corrections=corrections)

    # ── readiness ──
    def _edited(self, *_args):
        """Any edit: the output suggestion, the section gate and the readiness line (Int's)."""
        self._form_revision += 1
        self._prepared_fingerprint = None
        self._apply_suggestion()
        project = str(self._value("project_root")).strip()
        self.controls.apply_card_visibility(project)
        self._source_hints(project)
        named = bool(str(self._value("geometry_path")).strip())
        refinement = self._fields["refinement"]
        if refinement.enabled != named:
            self._fields["refinement"] = replace(refinement, enabled=named,
                                                 reason="" if named else "Choose a Calibration first")
            self._rows["refinement"].apply_field(self._fields["refinement"])
        self._readiness = self._readiness_now()
        self._show_readiness()
        self._sync_actions()

    def _readiness_now(self):
        """("Needs …", False) or ("Ready · RSM · …", True), from committed values; the disk is asked only
        whether the named files and folders exist. Each section's header (Int's ``SectionHeaderProjection``:
        its status, its ✓ when ready, what it still needs as the tip) is left in ``self._headers``."""
        needs = []  # in the readiness line's order
        sections = {key: [] for key in ("project", "experiment", "source", "processing")}

        def need(section, text):
            needs.append(text)
            sections[section].append(text)
        project = str(self._value("project_root")).strip()
        root = Path(project).expanduser() if project else None
        if not project:
            need("project", "a Project folder")
        elif not root.is_dir():
            need("project", f"an existing Project folder ({root.name} not found)")
        else:
            try:
                locator = self._calibration_locator(root)
            except ValueError as error:
                need("experiment", str(error))
            else:
                if not (root / locator).is_file():
                    need("experiment", f"the Calibration ({Path(locator).name} not found)")
            mask = str(self._value("mask_path")).strip()
            if mask and not self._inside(root, mask, "Mask File").is_file():
                need("experiment", f"the Mask File ({Path(mask).name} not found)")
            for text in self._source_needs(root):
                need("source", text)
            output = str(self._value("output_path")).strip()
            if not output.endswith(".nexus"):
                need("project", "an Output ending in .nexus")
        for key, section in (("source", SectionId.SOURCE), ("processing", SectionId.PROCESSING)):
            refused = self._advanced_refusal(section)
            if refused:
                need(key, f"Advanced {refused}")
        cap = self._chunk_cap_refusal()
        if cap:
            need("processing", f"Advanced {cap}")
        grid = "×".join(str(size) for size in _GRIDS[self._value("grid")])
        ready_detail = {"project": "Project folder and Output are ready.",
                        "experiment": "The Calibration (and any Mask File) is ready.",
                        "source": "The source, its Selection and the frames are ready; Preview finds the scans.",
                        "processing": "The coordinates, grid and normalization are set."}
        calibration = str(self._value("geometry_path")).strip()
        texts = {"project": "", "experiment": Path(calibration).name if calibration else "",
                 "source": self._source_text(),
                 "processing": f"{self._value('coordinates')} · grid {grid}"}
        self._headers = {key: SectionHeaderProjection(
            texts[key], not sections[key],
            ready_detail[key] if not sections[key] else "Needs " + ", ".join(sections[key]))
            for key in sections}
        if needs:
            return "Needs " + ", ".join(needs), False
        return f"Ready · RSM · {texts['source']} · {self._value('coordinates')} · grid {grid}", True

    def _selected_groups(self):
        """How many RSMs a SPEC Selection makes (Stitch's syntax; groups never span files), None when it is
        known only once the scans are found (All Scans, NeXus, a Directory). ValueError: not a Selection."""
        from xrd_tools.sources.grouping import parse_stitch_scan_groups
        if self._value("source_kind") != "Spec" or self._value("scan_mode") != "Select Scans":
            return None
        return len(parse_stitch_scan_groups(str(self._value("scan")))) * max(len(self._source_paths), 1)

    def _source_text(self):
        """The Source header's status, as Stitch's: the files and what they make."""
        kind, count = self._value("source_kind"), len(self._source_paths)
        if not count:
            return ""
        if kind == "Directory":
            return (Path(self._source_paths[0]).name + (" + subfolders" if self._value("subdirs") else "")
                    + (" · filtered" if str(self._value("name_filter")).strip() else ""))
        if kind == "NeXus":
            return f"{count} NeXus file{'s' * (count != 1)}"
        try:
            groups = self._selected_groups()
        except ValueError:
            groups = 0
        return f"{count} SPEC file{'s' * (count != 1)} · " + (
            "all scans" if groups is None else f"{groups} RSM{'s' * (groups != 1)}")

    def _source_needs(self, root):
        """What the Source still needs, without reading a file (the disk is asked only whether the chosen
        files and folders exist): its files, the Images folder, an Image stem and Selection that parse, one RSM
        (a SPEC Selection's count is known here, All Scans' and a Directory's once Preview finds the scans),
        and the Advanced frame range and Angle mapping."""
        kind = self._value("source_kind")
        needs = []
        files = [Path(path) for path in self._source_files(root)]
        if not files:
            needs.append({"Spec": "SPEC files", "NeXus": "NeXus files", "Directory": "a source Folder"}[kind])
        gone = [path.name for path in files if not (path.is_dir() if kind == "Directory" else path.is_file())]
        if gone:
            needs.append(f"the source ({', '.join(gone[:3])}{' …' * (len(gone) > 3)} not found)")
        if kind != "NeXus":
            images = str(self._value("image_dir")).strip()
            if images and not self._inside(root, images, "Images").is_dir():
                needs.append(f"the Images folder ({Path(images).name} not found)")
            stem = str(self._value("image_stem")).strip()
            try:
                stem.format(spec="spec", scan="1", occurrence="1.1")
                if any(mark in stem for mark in ("/", "\\", "\0")):
                    raise ValueError
            except (KeyError, IndexError, ValueError):
                needs.append("an Image stem of a file name, its braces {spec}, {scan} or {occurrence}")
        try:
            groups = self._selected_groups()
        except ValueError as error:
            needs.append(f"a Selection ({error})")
        else:
            if groups is not None and groups > 1:
                needs.append(f"one RSM ({groups} selected; {_ONE_GROUP})")
        if not any(self._rows[key].refusal() for key in ("frame_first", "frame_last", "frame_step")):
            try:  # the rows' own refusals are named as Advanced's; here, a Last before the First
                rsm_frame_selector(self._value("frame_first"), self._value("frame_last"), self._value("frame_step"))
            except ValueError as error:
                needs.append(f"Advanced frames ({error})")
        try:
            parse_rsm_angle_mapping(self._value("adv_angle_mapping"))
        except ValueError as error:
            needs.append(f"Advanced {error}")
        return needs

    def _source_hints(self, project):
        """Stitch's placeholders: the automatic Images folder and Image stem of the first SPEC file and scan."""
        kind = self._value("source_kind")
        spec = None
        if kind == "Spec" and self._source_paths:
            spec = Path(self._source_paths[0]).expanduser()
            spec = spec if spec.is_absolute() or not project else Path(project).expanduser() / spec
        first = re.match(r"\s*\[?\s*(\d+)", str(self._value("scan") or "")) if kind == "Spec" else None
        self._rows["image_dir"].editor.setPlaceholderText(
            f"auto · {spec.parent}" if spec else "auto · the SPEC file's folder")
        self._rows["image_stem"].editor.setPlaceholderText(
            f"auto · {spec.stem if spec else '<SPEC name>'}_scan{first[1] if first else '<N>'}_")

    def _show_readiness(self):
        text, ready = self._readiness
        self.run_bar.set_readiness_summary(text, ready=ready)
        cards = {"project": self.controls.project_card, "experiment": self.controls.experiment_card,
                 "source": self.controls.source_card, "processing": self.controls.processing_card}
        for key, header in getattr(self, "_headers", {}).items():  # Int's header adapter
            apply_section_header(cards[key], header)

    def _status(self, text, *, ready=True):
        self.run_bar.set_readiness_summary(str(text), ready=ready)

    def _notice(self, text, *, ready=False):
        self._status(text, ready=ready)
        try:
            self._services.status.show(str(text), 8000)
        except Exception:
            _LOG.debug("RSM status presenter failed", exc_info=True)

    def _sync_actions(self):
        busy = bool(self._owner.busy)
        final = self._owner.finalization
        reading = self._reading is not None
        authoring = self._authoring_active()
        resolving = self._resolving is not None
        idle = (not busy and not reading and not authoring and not resolving
                and final is RSMOwnerFinalization.NONE and not self._closing)
        self.controls.set_editors_enabled(idle)  # the action row stays usable: Retry (Codex F2)
        self.calibrate_button.setEnabled(idle)
        self.mask_button.setEnabled(idle)
        for dialog in self._authoring_dialogs:
            dialog.setEnabled(not busy and not self._closing and (not authoring or dialog.active()))
        if not any(dialog.isVisible() or dialog.active() for dialog in self._authoring_dialogs):
            self._authoring_timer.stop()
        self.run_bar.set_mode_row_enabled(idle)
        self.advanced.setEnabled(idle)
        self.browser.set_action_enabled("Open…", not busy and not reading and not self._closing)
        self.preview_button.setEnabled(idle)
        self.advanced_button.setEnabled(idle)
        self.retry_button.setEnabled(not busy and final is not RSMOwnerFinalization.NONE and not self._closing)
        # Stitch's: "▶ Run" throughout (RSM has no pause), enabled while idle and ready
        self.run_bar.startButton.setEnabled(idle and self._readiness[1])
        identity = self._owner.current_identity if busy else None
        self.run_bar.set_stop_enabled((resolving and self._resolving[1] is not None) or (
            identity is not None and identity.request.action in {
                RSMOwnerAction.PREFLIGHT, RSMOwnerAction.RUN}))  # what the owner cancels, or finding the scans

    # ── the owner ──
    def _control_action(self, action):
        {ControlAction.CALIBRATE: self._calibrate, ControlAction.MAKE_MASK: self._make_mask,
         ControlAction.PREVIEW: self.begin_preview, ControlAction.RETRY: self.retry,
         ControlAction.ADVANCED_PROCESSING: self.advanced.show}.get(action, lambda: None)()

    # ── the shared calibration and mask windows ──
    def _authoring_active(self):
        return any(dialog.active() for dialog in self._authoring_dialogs)

    def _reader_options(self):
        """The raw decoder as the calibration and mask windows read a frame with it (ValueError naming a refused
        Advanced row)."""
        refused = self._advanced_refusal(SectionId.SOURCE)
        if refused:
            raise ValueError(f"Advanced {refused}")
        return dict(detector_shape=(int(self._value("adv_rows")), int(self._value("adv_columns"))),
                    raw_dtype=str(self._value("adv_dtype")), raw_header_skip=int(self._value("adv_header")))

    def _source_spec_for_tools(self):
        """The first selected scan (or NeXus file), for the calibration and mask windows, from the Source rows
        without reading (Stitch's ``_source_spec_for_tools``); None for a folder or without a source."""
        from xrd_tools.core.scan import SourceKind, SourceSpec
        from xrd_tools.sources.grouping import parse_stitch_scan_groups
        project = str(self._value("project_root")).strip()
        kind = self._value("source_kind")
        if not project or not self._source_paths or kind == "Directory":
            return None
        root = Path(project).expanduser()
        path = Path(self._source_files(root)[0])
        if kind == "NeXus":
            return SourceSpec(str(path), SourceKind.NEXUS_STACK)
        try:  # a Selection or stem readiness refuses opens the window without a scan
            scan = parse_stitch_scan_groups(str(self._value("scan") or "1"))[0][0]
            stem = str(self._value("image_stem")).strip().format(spec=path.stem, scan=scan.split(".")[0],
                                                                  occurrence=scan)
        except (KeyError, IndexError, ValueError):
            return None
        images = str(self._value("image_dir")).strip()
        return SourceSpec(str(path), SourceKind.SPEC, options={
            "scan": scan, "image_dir": str(self._inside(root, images, "Images")) if images else str(path.parent),
            "image_stem": stem or None, "read_image_kwargs": self._reader_options()})

    def _calibrate(self):
        """Calibrate: the shared calibration window with RSM's context ("Save geometry and use in RSM"), so RSM
        calibrates its own data; its saved file becomes the Calibration row's."""
        from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
        if self._owner.busy or self._closing or self._authoring_active():
            return
        try:
            project = str(self._value("project_root")).strip()
            if not project:
                raise ValueError("Choose a Project folder first")
            if self._calibration_dialog is not None and self._calibration_project != project:
                self._calibration_dialog.hide()
                self._calibration_dialog = None
            fresh = self._calibration_dialog is None
            if fresh:
                self._calibration_dialog = CalibrationDialog(CalibrationContext("RSM", project), parent=self)
                self._calibration_project = project
                self._calibration_dialog.calibrationSaved.connect(self._adopt_calibration)
                self._authoring_dialogs.append(self._calibration_dialog)
            source = self._source_spec_for_tools() if fresh else None
            if source is not None:  # RSM reads raw images: no turn or mirror of its own
                self._calibration_dialog.set_source_context(source, image_rotation=0, image_flip=False)
            self._calibration_dialog.show()
            self._calibration_dialog.raise_()
            self._authoring_timer.start()
        except (TypeError, ValueError, OSError) as error:
            self._notice(str(error))

    def _adopt_calibration(self, path, _backend=None):
        """A calibration the window saved: the Calibration row takes it (and its mask, the detector's)."""
        if self._closing or self._calibration_project != str(self._value("project_root")).strip():
            self._notice(f"Calibration saved: {path} · Project changed; select the calibration to use it")
            return
        self.set_field("geometry_path", str(path))
        dialog = self._calibration_dialog
        if dialog is not None and dialog.saved_mask:
            self.set_field("mask_path", dialog.saved_mask)

    def _make_mask(self):
        """Make Mask: the shared mask window on the first selected scan; its saved mask fills Mask File."""
        from xdart.gui.tabs.calibration.mask import MaskDialog
        if self._owner.busy or self._closing or self._authoring_active():
            return
        try:
            dialog = MaskDialog(str(self._value("project_root")), source=self._source_spec_for_tools(), rotation=0,
                                reader_options=dict(self._reader_options(), rotation=0), parent=self)
            self._authoring_dialogs.append(dialog)
            revision = self._form_revision
            dialog.maskSaved.connect(lambda path: self.set_field("mask_path", str(path))
                                     if revision == self._form_revision and not self._closing
                                     else self._notice(f"Mask saved: {path} · Inputs changed; select the mask "
                                                       "to use it"))
            dialog.show()
            self._authoring_timer.start()
        except (TypeError, ValueError, OSError) as error:
            self._notice(str(error))

    def _current_form(self, *, quiet=False):
        try:
            return self.build_form()
        except (TypeError, ValueError, OverflowError) as error:
            if not quiet:
                self._notice(f"Invalid RSM request: {error}")
            return None

    def _commit_typing(self):
        """Commit a field still being typed in before an action reads the inputs (Stitch's live lesson: a
        click on Run leaves the focus in the field on macOS, and disabling the controls for the check then
        committed it mid-check, which discarded the check as "the inputs changed")."""
        focused = QtWidgets.QApplication.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit) and (self.isAncestorOf(focused)
                                                         or self.advanced.isAncestorOf(focused)):
            focused.clearFocus()  # the row commits on the focus-out, as Return or Tab would

    def begin_preview(self):
        """Preview: find the Source's scans, check the inputs and list every exact identity, without
        gridding."""
        self._commit_typing()
        self._find_scans("preview")

    def begin_run(self):
        """Run: find the Source's scans again, then grid them, Previewing first unless the prepared Preview is
        still the current form (as Stitch's Run prepares first)."""
        self._commit_typing()
        if not self._readiness[1]:
            self._notice(self._readiness[0])
            return
        self._find_scans("run")

    def _find_scans(self, action):
        """Find the Source's member groups on the shared single worker (``rsm_member_groups`` reads the SPEC
        files and lists the images, off the GUI thread; S2), then go on to ``action`` (``_continue``)."""
        if self._owner.busy or self._resolving is not None:  # a running result keeps its form
            self._notice("RSM is busy · try again once it is done")
            return
        try:
            request = self._resolution_request()
        except (TypeError, ValueError) as error:
            self._notice(f"Invalid RSM request: {error}")
            return
        identity = self._resolver.begin(request)
        if identity is None:
            self._notice("RSM is busy · try again once it is done")
            return
        self._resolving = (identity, action, self._group_key())
        self._status("Finding the source's scans…")
        self._timer.start()
        self._sync_actions()

    def _poll_resolver(self):
        if self._resolving is None:
            return
        identity, action, key = self._resolving
        update = self._resolver.poll(identity)
        if update is None or update.terminal is None:
            return
        self._resolving = None
        terminal = update.terminal
        if action is None:  # stopped: whatever was found is dropped
            self._notice("RSM stopped")
        elif terminal.status is OperationTerminalStatus.FAILED:
            self._notice(f"Source refused: {terminal.failure_message or terminal.failure_type}")
        elif terminal.status is OperationTerminalStatus.RETURNED and terminal.payload is not None:
            groups, skipped = terminal.payload
            if key != self._group_key():
                self._notice("The scans were found, but the Source changed: Preview again")
                return
            self._found(key, groups, skipped)
            self._continue(action)
        else:
            self._notice("RSM stopped")

    def _continue(self, action):
        """Preview, or Run, on the member groups just found."""
        form = self._current_form()
        if form is None:
            return
        if action == "run" and self._prepared_fingerprint == form.fingerprint and self._owner.prepared is not None:
            self._start_run(form)
        else:
            self._begin_preflight(run=action == "run", form=form)

    def _begin_preflight(self, *, run, form=None):
        if self._owner.busy:  # a running result keeps its form
            self._notice("RSM is busy · try again once it is done")
            return
        form = self._current_form() if form is None else form
        if form is None:
            return
        self._owner.set_form(form)
        if self._owner.begin_preflight() is None:
            self._notice("RSM is busy · try again once it is done")
            return
        self._run_requested = run
        self._active_revision = self._form_revision
        self._status("Checking the inputs…")
        self._timer.start()
        self._sync_actions()

    def _start_run(self, form):
        prepared = self._owner.prepared
        if (self.run_bar.write_mode() == "Append" and prepared is not None and prepared.is_current(form)
                and prepared.output_current and Path(form.output_path).is_file()):
            # Stitch's Append: the output already holds this RSM; the next Run checks again
            self._prepared_fingerprint = None
            self._status(f"Up to date · {Path(form.output_path).name} already gridded with these settings "
                         "(Replace ⇄ grids again)")
            return
        self._owner.set_form(form)
        if self._owner.begin_run() is None:
            self._prepared_fingerprint = None
            self._notice("RSM could not start: Preview again")
            return
        self._run_requested = False
        self._active_revision = self._form_revision
        self._status("Gridding…")
        self._timer.start()
        self._sync_actions()

    def stop(self):
        if self._resolving is not None and self._resolving[1] is not None:
            identity, _action, key = self._resolving
            self._resolver.cancel(identity)
            self._resolving = (identity, None, key)  # finding the scans has no stop point: its result is dropped
            self._status("Stopping…")
            self._sync_actions()
        elif self._owner.cancel():
            self._status("Stopping…")
        return ActionAccepted("rsm-stop")

    def retry(self):
        final = self._owner.finalization
        begin = (self._owner.begin_retry_cleanup if final is RSMOwnerFinalization.CLEANUP_PENDING
                 else self._owner.begin_retry_verification)
        if begin() is None:
            self._notice("No RSM retry is available")
            return
        self._status("Retrying the result's saving…" if final is RSMOwnerFinalization.CLEANUP_PENDING
                     else "Checking the saved result again…")
        self._timer.start()
        self._sync_actions()

    def _poll(self):
        try:
            update = self._owner.poll()
            if update is not None:
                self._accept(update)
            self._poll_reader()
            self._poll_resolver()
        except Exception as error:
            _LOG.exception("RSM page update failed")
            self._notice(f"RSM_VIEW_PRESENTATION_FAILED: {error}")
        if not self._owner.busy and self._reading is None and self._resolving is None:
            self._timer.stop()
        self._sync_actions()

    def _accept(self, update):
        if type(update) is not RSMOwnerUpdate:
            raise TypeError("the RSM page requires an exact owner update")
        if update.progress is not None:
            progress = update.progress
            self._status(_progress_text(progress.stage, progress.completed, progress.total))
            return
        if update.terminal_status is OperationTerminalStatus.FAILED:
            self._notice(f"RSM {update.action.value} failed: {update.failure_message or update.failure_type}")
            return
        if update.terminal_status is OperationTerminalStatus.CANCELLED:
            self._notice("RSM stopped")
            return
        outcome = update.outcome
        if outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_READY:
            preflight = outcome.preflight
            current = self._current_form(quiet=True)
            if update.stale or self._active_revision != self._form_revision or current is None \
                    or not preflight.is_current(current):
                self._run_requested = False
                self._notice("Preview completed, but the inputs changed: Preview again")
                return
            self._prepared_fingerprint = preflight.form.fingerprint
            summary = preflight.summary
            lines = rsm_preflight_summary_lines(summary)
            for path, scan, reason in (self._groups[2] if self._groups else ()):  # what the Source skipped
                lines.append(f"Skipped {' '.join(part for part in (Path(path).name, str(scan)) if part)}: {reason}")
            if preflight.output_current:  # as Stitch's Preview says it
                lines.append(f"Output {Path(preflight.form.output_path).name}: already gridded with these "
                             "settings (Append skips it)")
            self.preview_text.setPlainText("\n".join(lines))  # it names the geometry and its energies
            if self._run_requested:
                self._start_run(current)
            else:
                self._status(f"Preview ready · {summary.total_selected_frames} frames", ready=True)
                self.preview_dialog.show()
            return
        self._run_requested = False
        if outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_REFUSED:
            self._notice(f"Preview refused: {outcome.refusal_message or outcome.refusal_code}")
            return
        if outcome.kind is RSMOwnerOutcomeKind.CLEANUP_PENDING:
            self._notice((outcome.finalization_message or "Saving needs a retry") + " · Retry (no regridding)")
            return
        if outcome.kind is RSMOwnerOutcomeKind.VERIFICATION_PENDING:
            self._notice("Saved; its check needs a retry · Retry")
            return
        result = outcome.result
        if outcome.kind is not RSMOwnerOutcomeKind.RESULT or result is None:
            raise RuntimeError("RSM owner returned an invalid outcome")
        terminal = result.terminal
        if terminal.disposition is ModuleDisposition.COMMITTED:
            if type(result) is not RSMOperationResultV2 or result.payload is None:
                raise ValueError("the RSM page shows only a strict committed payload")
            target = str(result.request.module.output.target)
            current = not update.stale and self._active_revision == self._form_revision
            self._add_result(target, result.payload, show=current)
            self._edited()  # the next Run checks again: in Append it finds this output up to date
            if current:
                self._status(f"Saved {Path(target).name}", ready=True)
            else:
                self._notice(f"Saved {Path(target).name}, but the inputs changed: listed, not shown")
            return
        detail = f" · {terminal.diagnostic}" if terminal.diagnostic else ""
        if terminal.disposition is ModuleDisposition.CANCELLED:
            self._notice("RSM stopped" + detail)
        else:
            self._notice(f"RSM {terminal.disposition.value}: {terminal.code}{detail}")

    # ── results ──
    @property
    def results(self):
        """The Data Browser's Results list, Int's own widget. The rows read it and choose in it as a user
        does; the page fills it only through ``reconcile_results``."""
        return self.browser.scans

    def _browser_command(self, command):
        if command.kind is ShellCommandKind.SELECT_RESULT:
            self._result_selected(command.value)
        elif command.kind is ShellCommandKind.OPEN_RESULT or (
                command.kind is ShellCommandKind.MENU and command.value == "File:Open result…"):
            self.open_result()
        elif command.kind is ShellCommandKind.MENU and command.value == "File:Integration workspace":
            self.toolRequested.emit(str(SCATTERING_PAGE_KEY))

    def _publish_results(self):
        self.browser.reconcile_results(
            tuple(BrowserResult(path, Path(path).name, path) for path in self._listed),
            self._requested if self._requested in self._listed else None)

    def open_result(self):
        from xdart.gui.widgets import file_dialogs as dialogs
        picks = dialogs.browse(self, "Open RSM result", file_filter=dialogs.NEXUS_FILTER,
                               current=str(self._value("output_path") or ""),
                               fallback=str(self._value("project_root") or ""))
        if picks:
            self.reopen(picks[0])

    def reopen(self, path):
        """Read a saved RSM record on the shared single-worker reader, then show it."""
        identity = self._reader.begin(str(path))
        if identity is None:
            self._notice("A result is still being read")
            return False
        self._reading = (identity, str(path))
        self._requested = str(path)
        self._status(f"Reading {Path(path).name}…")
        self._timer.start()
        self._sync_actions()
        return True

    def _poll_reader(self):
        if self._reading is None:
            return
        identity, path = self._reading
        update = self._reader.poll(identity)
        if update is None or update.terminal is None:
            return
        self._reading = None
        terminal = update.terminal
        if terminal.status is OperationTerminalStatus.RETURNED and terminal.payload is not None:
            # kept either way; shown only if it is still the result asked for (Codex's stage-1 review F4)
            shown = path == self._requested
            self._add_result(path, terminal.payload, show=shown)
            if shown:
                self._status(f"Opened {Path(path).name}", ready=True)
            else:
                self._show_readiness()
        elif terminal.status is OperationTerminalStatus.FAILED:
            self._notice(f"Could not open {Path(path).name}: {terminal.failure_message or terminal.failure_type}")
        # The selection moved on while this was read: the result now asked for is read next, after a failed
        # read as well (Codex's stage-2 review S2-F1). Never the same failed record again, never while closing.
        latest = self._requested
        if (latest != path and not self._closing and latest in self._results
                and self._results[latest] is None):
            self.reopen(latest)

    def _add_result(self, path, payload, *, show=True):
        """List a result and keep its payload; the oldest beyond the newest few are read again when shown
        (the one asked to be shown stays: a late read kept beside it does not push it out)."""
        if show:
            self._requested = path
        self._results[path] = payload
        self._viewers.pop(path, None)  # a result admitted again (Replace, a read back) is drawn from itself
        self._results.move_to_end(path)
        resident = [key for key, value in self._results.items() if value is not None]
        excess = len(resident) - _RESIDENT_RESULTS
        for key in resident:
            if excess > 0 and key != self._requested:
                self._results[key] = None
                self._viewers.pop(key, None)
                excess -= 1
        if path not in self._listed:
            self._listed.append(path)
        self._publish_results()
        if show:
            self._show(path, payload)

    def _result_selected(self, path):
        self._requested = path
        payload = self._results.get(path)
        if payload is None:
            if self._reading is None:  # else it is read once the read in flight returns (_poll_reader)
                self.reopen(path)
            return
        try:
            self._show(path, payload)
        except Exception as error:  # the prior six stay
            _LOG.exception("RSM result presentation failed")
            self._notice(f"RSM_VIEW_PRESENTATION_FAILED: {error}")

    @property
    def shown(self):
        """The result drawn now: (its viewer values, the viewer-model snapshot drawn), or None."""
        return self._shown

    def _viewer(self, path, payload):
        """The result's viewer values and model, built once while it stays resident."""
        viewer = self._viewers.get(path)
        if viewer is None or viewer[0] is not payload:  # a new result under the same path is built again
            values = make_rsm_viewer_values(payload)
            model = RSMViewerModel()
            model.snapshot(values)
            viewer = self._viewers[path] = (payload, values, model)
        return viewer

    def _show(self, path, payload):
        _payload, values, model = self._viewer(path, payload)
        self._draw(values, model.current_snapshot, path)
        self._shown_path = path

    def _draw(self, values, snapshot, path=None):
        path = self._shown_path if path is None else path
        self.view.reconcile_keyed(rsm_keyed_projection(
            values, snapshot, color_map=self._color_map, log_scale=self._log_scale,
            title=Path(path).name if path else ""))
        self._shown = (values, snapshot)

    def _redraw(self):
        shown = self.shown
        if shown is not None:
            self._draw(*shown)

    def _view_command(self, command):
        """Int's view's commands for the RSM drawn: its colour map and Log (as Int's top bar sets them), and a
        slice stepped along its normal."""
        kind = command.kind
        if kind is ShellCommandKind.SET_COLOR_MAP:
            self._color_map = str(command.value)
            self._redraw()
        elif kind is ShellCommandKind.SET_LOG_SCALE:
            self._log_scale = bool(command.value)
            self._redraw()
        elif kind is ShellCommandKind.SET_PANEL_INDEX:
            self._step_slice(command.path[0] if command.path else "", command.value)
        elif kind is ShellCommandKind.PAGE_TOOL:
            self._page_tool(str(command.value))

    def _page_tool(self, name):
        """The 3D Viewer and Custom Slice buttons: their windows come with stage 2b's G9."""
        _LOG.info("RSM page tool %r pressed; its window is not built yet (stage 2b G9)", name)

    def _step_slice(self, instance, index):
        shown = self._viewers.get(self._shown_path) if self._shown_path else None
        if shown is None or type(index) is not int:
            return
        _payload, values, model = shown
        normal = rsm_slice_normals(values).get(instance)
        if normal is None:
            return
        indices = list(model.current_snapshot.state.axis_indices)
        indices[normal] = index
        try:
            self._draw(values, model.snapshot(values, h_index=indices[0], k_index=indices[1],
                                              l_index=indices[2]))
        except Exception as error:  # the view keeps the six it drew
            _LOG.exception("RSM slice presentation failed")
            self._notice(f"RSM_VIEW_PRESENTATION_FAILED: {error}")

    # ── paths ──
    def _browse(self, key):
        """Browse as Int and Stitch do: the last folder picked, the same file types for the same inputs; the
        Source's files by its kind (SPEC files, NeXus files, or a folder)."""
        from xdart.gui.widgets import file_dialogs as dialogs
        kind = self._value("source_kind")
        current = self._source_paths[0] if key == "spec_path" and self._source_paths else self._value(key)
        common = dict(current=str(current or ""), fallback=str(self._value("project_root") or ""))
        if key in {"project_root", "image_dir"} or key == "spec_path" and kind == "Directory":
            picks = dialogs.browse(self, {"image_dir": "Choose images folder"}.get(key, "Choose folder"),
                                   folder=True, **common)
            if picks and key == "spec_path":
                self.set_sources(picks[:1])
                return
        elif key == "spec_path":
            if kind == "Spec":  # SPEC files have no extension: only they are listed
                picks = dialogs.browse_spec(self, "Choose SPEC files", many=True, **common)
            else:
                picks = dialogs.browse(self, "Choose NeXus files", many=True, file_filter=dialogs.NEXUS_FILTER,
                                       **common)
            if picks:
                self.set_sources(picks)
            return
        elif key in {"geometry_path", "mask_path"}:  # Int's file types for the same inputs
            title, file_filter = (("Choose calibration", dialogs.CALIBRATION_FILTER) if key == "geometry_path"
                                  else ("Choose detector mask", dialogs.MASK_FILTER))
            picks = dialogs.browse(self, title, file_filter=file_filter, **common)
        elif key == "output_path":  # the shared save: the last folder, the current output's name
            picks = dialogs.browse(self, "Choose RSM output", save=True, file_filter="NeXus result (*.nexus)",
                                   default_name=Path(common["current"]).name if common["current"] else "",
                                   **common)
        else:
            picks = dialogs.browse(self, "Choose " + self._fields[key].label, **common)
        if picks:
            self.set_field(key, picks[0])

    # ── modes and the host ──
    def _mode_chosen(self, mode):
        """RSM here; another mode leaves for its page (the host refuses while RSM runs)."""
        blocker = QtCore.QSignalBlocker(self.run_bar.modeCombo)
        self.run_bar.modeCombo.setCurrentText("RSM")
        del blocker
        if mode in _RSM_MODES:
            return
        if mode in {"Stitch 1D", "Stitch 2D"}:
            self.modeRequested.emit(str(STITCH_TOOL_KEY), str(mode))
        else:
            self.modeRequested.emit(str(SCATTERING_PAGE_KEY), str(mode))

    def active(self):
        return bool(self._owner.busy or self._reading is not None or self._resolving is not None
                    or self._owner.finalization is not RSMOwnerFinalization.NONE or self._authoring_active())

    def mount_points(self):
        return AppMenuHosts(self.browser.menus["Config"], self.browser.menus["Help"],
                            tool_requested=self.toolRequested,
                            mode_requested=self.modeRequested)

    def request(self):
        self.open_result()
        return ActionAccepted("open-rsm-result")

    def run_pause(self):
        self.begin_run()
        return ActionAccepted("rsm-run")

    def close_page(self):
        """Close the owner and the reader; PENDING while either still has a worker."""
        self._closing = True
        self._timer.stop()
        self._authoring_timer.stop()
        receipts = (self._owner.close(), self._reader.close(), self._resolver.close(),
                    *(dialog.shutdown() for dialog in self._authoring_dialogs))
        pending = [receipt for receipt in receipts if receipt.status is not PageCleanup.CLEAN]
        if pending:
            self._closing = False
            return pending[0]
        self.view.clear_keyed("No RSM result shown")
        self._viewers.clear()
        self._shown_path = self._shown = None
        return CloseReceipt(PageCleanup.CLEAN)


class _RSMProcessingMode:
    """Another page's request for RSM (Int's or Stitch's run bar)."""

    def __init__(self, page):
        self._page = page

    def select(self, mode):
        from xdart.gui.pages.values import MODE_REFUSED
        if str(mode) in _RSM_MODES:
            return ActionCompleted("processing-mode-RSM")
        return ActionRefused(MODE_REFUSED)


def build_rsm_page(services, parent):
    page = RSMPage(services, parent)
    return PageHandle(key=RSM_TOOL_KEY, widget=page, close=page.close_page, open_folder=page,
                      run_control=page, activity=page, app_menus=page, processing_mode=_RSMProcessingMode(page))


__all__ = ["RSMPage", "build_rsm_page", "suggested_output"]
