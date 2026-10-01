"""Native Stitch workspace over the existing finite operation owner.

Source admission, calibration, raw reads and publication stay with their shared
headless owners. This page queues their exact prepared requests and presents
the resulting curves and original contributing frames.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import functools
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
from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
from xdart.gui.tabs.scattering.workspace_shell import _HugWidthScrollArea
from xrd_tools.analysis.module_transaction import MetadataColumnSelector, ModuleDisposition
from xrd_tools.analysis.stitch_native import native_ceilings
from xrd_tools.analysis.stitch_operation import StitchGeometryKind
from xrd_tools.core.geometry import ImageOrientation
from xrd_tools.analysis.stitch_sources import StitchSourceSelection
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, AnalysisArtifactPayload
from xrd_tools.session.readiness import (
    ControlAction, ControlActionSpec, ControlFieldKind, ControlFormField, SectionId,
)
from xdart.gui.tabs.scattering.controls_inventory import (
    GI_1D_AXES, GI_ENABLED, GI_MOTOR, GI_ORIENTATION, GI_THETA, INT_1D_AXIS, INT_1D_POINTS, INT_1D_RADIAL_AUTO,
    MASK_FILE, PONI_FILE,
    INT_1D_RADIAL_HIGH, INT_1D_RADIAL_LOW, INT_2D_AXIS, INT_2D_AZIM_AUTO, INT_2D_AZIM_HIGH, INT_2D_AZIM_LOW,
    INT_2D_AZIM_POINTS, INT_2D_RADIAL_AUTO, INT_2D_RADIAL_HIGH, INT_2D_RADIAL_LOW, INT_2D_RADIAL_POINTS,
    SOURCE_DIRECTORY, SOURCE_FILTER, SOURCE_RECURSIVE, SOURCE_TYPE, STANDARD_1D_AXES, STANDARD_2D_AXES,
    THRESHOLD_ENABLED, THRESHOLD_MAX, THRESHOLD_MIN,
)
from xrd_tools.session.threshold_editing import (
    ThresholdEditRefusal, ThresholdState, threshold_transition,
)

_FRAME_CHECKS = {"Exclude empty frames": "empty", "Exclude empty and partial frames": "partial",
                 "Keep all frames": "none"}
_SAMPLE_POSITIONS = {"As calibrated": "calibrated", "On rotation axis": "rotation_axis"}
#: A GI result's cut: its plot-axis key (Int's) and axis label and unit.
_CUT_AXES = {"q_ip": ("Qᵢₚ", "Å⁻¹"), "q_oop": ("Qₒₒₚ", "Å⁻¹")}
#: The Axis choices: Q and 2θ, and in Grazing the GI cuts (named as Int's GI axes). A GI axis writes
#: I(q) and both cuts, which one pass gives (maintainer, 2026-09-29: "it will be good to save both by
#: default along with I(q)"); the choice is the cut a result opens on.
_STANDARD_AXES = tuple(label for label in STANDARD_1D_AXES if label != "χ (°)")  # Stitch makes no I(χ)
#: Grazing's Axis, Int's GI axes a Stitch makes: every Grazing Stitch writes I(q) and both cuts; the choice
#: is the one a result opens on.
_GI_AXES = {label: GI_1D_AXES[label] for label in ("Q", "Qip", "Qoop")}
#: The 2-D Axis, as Int's: Standard's Q-χ and 2θ-χ bin the map in Q or 2θ; a Grazing Stitch 2D writes
#: the (χ, q) and q_ip-q_oop maps, and the choice is the one a result opens on.
_STANDARD_MAP_AXES = tuple(STANDARD_2D_AXES)
_GI_MAP_AXES = ("Qip-Qoop", "Q-χ")
#: Axis names settings saved before Stitch took Int's GI names.
_LEGACY_AXES = {"Qip (Å⁻¹)": "Qip", "Qoop (Å⁻¹)": "Qoop"}
#: What Cores do on this page: each worker holds its frame's arrays (Codex terminal review); on the
#: xrayutilities backend its own work stays on one thread and the workers bin the frames (PERF-STITCH-3)
_CORES_TIP = "Frames stitched at once; each needs its frame's arrays in memory, so fewer use less"
#: The Processing fields, in their rows' order: (group, key, path, default). Int's paths take Int's
#: labels, rules and rows (``project_control_fields``, ``ControlsPanel.render_processing_rows``); Stitch's
#: own are its GI ranges and Monitor.
_PROCESSING = (
    ("1-D", "unit", INT_1D_AXIS, "Q (Å⁻¹)"),
    ("1-D", "npt_1d", INT_1D_POINTS, "1500"),
    ("1-D", "radial_auto", INT_1D_RADIAL_AUTO, True),
    ("1-D", "q_min", INT_1D_RADIAL_LOW, 0.0),
    ("1-D", "q_max", INT_1D_RADIAL_HIGH, 6.2),
    ("1-D", "gi_ip_auto", ("Stitch", "gi_ip_auto"), True),
    ("1-D", "gi_ip_min", ("Stitch", "gi_ip_low"), -2.0),
    ("1-D", "gi_ip_max", ("Stitch", "gi_ip_high"), 2.0),
    ("1-D", "gi_oop_auto", ("Stitch", "gi_oop_auto"), True),
    ("1-D", "gi_oop_min", ("Stitch", "gi_oop_low"), 0.0),
    ("1-D", "gi_oop_max", ("Stitch", "gi_oop_high"), 1.0),
    ("2-D", "map_axis", INT_2D_AXIS, "Q-χ"),
    ("2-D", "npt_rad_2d", INT_2D_RADIAL_POINTS, "1000"),
    ("2-D", "npt_azim_2d", INT_2D_AZIM_POINTS, "360"),
    ("2-D", "map_auto", INT_2D_RADIAL_AUTO, True),
    ("2-D", "map_min", INT_2D_RADIAL_LOW, 0.0),
    ("2-D", "map_max", INT_2D_RADIAL_HIGH, 6.2),
    ("2-D", "chi_auto", INT_2D_AZIM_AUTO, True),
    ("2-D", "chi_min", INT_2D_AZIM_LOW, -180.0),
    ("2-D", "chi_max", INT_2D_AZIM_HIGH, 180.0),
    ("Conditioning", "threshold_enabled", THRESHOLD_ENABLED, True),
    ("Conditioning", "threshold_min", THRESHOLD_MIN, 0.0),
    ("Conditioning", "threshold", THRESHOLD_MAX, None),
    ("Conditioning", "monitor", ("Stitch", "monitor"), "None"),
)
#: A GI range's label: (the rows' label, what it bins and windows).
_GI_RANGES = {"gi_ip": ("Qip (Å⁻¹)", "Bins I(Qip) (Auto: the data's span) and is I(Qoop)'s window (Auto: every "
                                     "pixel); a Stitch 2D's q_ip-q_oop map spans both."),
              "gi_oop": ("Qoop (Å⁻¹)", "Bins I(Qoop) (Auto: the data's span) and is I(Qip)'s window (Auto: every "
                                       "pixel); a Stitch 2D's q_ip-q_oop map spans both.")}


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


def _goniometer_record(path):
    """A pyFAI goniometer JSON with valid position names, or None when it cannot be read."""
    try:
        file = Path(str(path or "").strip())
        state = file.stat()
        if file.suffix.lower() != ".json" or state.st_size > 1 << 20:
            return None
        return _read_goniometer(str(file), state.st_mtime_ns, state.st_size)
    except OSError:
        return None


def _calibration_detector(path):
    """The calibration's detector (pyFAI's name), for decision 39's documented counter limit: a pyFAI
    goniometer JSON's, else an xrayutilities calibration's; None when unknown (read through the
    stat-keyed cache the readiness line uses)."""
    record = _goniometer_record(path)
    if record is None:
        record = _xu_calibration_record(path)
        detector = record.get("detector") if record is not None else None
        name = detector.get("detector_factory_name") if isinstance(detector, dict) else None
    else:
        name = record.get("detector")
    return name if isinstance(name, str) and name.strip() else None


def _xu_calibration_record(path):
    try:
        file = Path(str(path or "").strip())
        state = file.stat()
        if file.suffix.lower() != ".json" or state.st_size > 1 << 20:
            return None
        return _read_json(str(file), state.st_mtime_ns, state.st_size)
    except OSError:
        return None


@functools.lru_cache(maxsize=4)
def _read_json(path, _mtime_ns, _size):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


@functools.lru_cache(maxsize=8)
def _container_ceiling(path, _state):
    """A NeXus container's (recorded saturation, data type), from its header alone; None unread."""
    from xrd_tools.io.image import read_detector_image_layout
    try:
        layout = read_detector_image_layout(path)
    except Exception:  # noqa: BLE001 - an unreadable container shows no hint
        return None
    return layout.saturation, layout.dtype


@functools.lru_cache(maxsize=4)
def _read_goniometer(path, _mtime_ns, _size):  # the readiness line asks on every edit
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        names = record["trans_function"]["pos_names"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if type(names) is not list or not names or any(type(name) is not str or not name.strip()
                                                   for name in names):
        return None
    return record


@functools.lru_cache(maxsize=8)
def _bound_arm(path, _mtime_ns, _size, bindings):
    """The detector circles a goniometer JSON moves and holds, as bound to the scan's
    motors: ("ok", moving, holds) or ("unknown", why)."""
    from xrd_tools.session.instrument import calibration_arm
    try:
        return ("ok", *calibration_arm(_read_goniometer(path, _mtime_ns, _size), dict(bindings)))
    except ValueError as error:
        return ("unknown", str(error))


@functools.lru_cache(maxsize=8)
def _spec_motor_names(path, _state, scan):
    """The real motors a SPEC scan records (its ``#O`` motors, the scanned one among them; no
    counters) from its header alone, or None when the scan cannot be read. ``scan`` None: the
    file's first. The only source read the page makes: on a committed source or scan edit (Codex
    audit G-F2; Preview decides everything else on the worker)."""
    from xrd_tools.io.spec import read_spec_scan_motor_names
    try:
        return read_spec_scan_motor_names(path, scan)
    except (OSError, ValueError, KeyError, IndexError):
        return None


#: Monitor lists every counter a scan records but what is clearly no monitor (maintainer, 2026-09-30:
#: "all the metadata columns that could be beam monitors of any kind listed (i0, i1, mon, monitor, bs,
#: beamstop, pd, photod, etc). if in doubt, list it"): the likely monitors first, in the file's order.
def monitor_choices(counters):
    """"None" (no monitor), then the scan's counters as beam-monitor choices: likely monitors first, the
    time stamp and H, K, L left out (``core.metadata.monitor_candidates``, which Int's Norm channel takes)."""
    from xrd_tools.core.metadata import monitor_candidates
    return ("None", *monitor_candidates(name for name in counters if name.strip()))


#: The detector camera's eight values, "rows, columns" (xrayutilities' detectorDir1, detectorDir2), Stitch's
#: frame first: an image rotation of 0 before the camera (maintainer, 2026-09-30).
CAMERAS = tuple(f"{rows}, {columns}" for rows, columns in (
    ("x-", "z+"), ("x-", "z-"), ("x+", "z-"), ("x+", "z+"),
    ("z-", "x-"), ("z-", "x+"), ("z+", "x+"), ("z+", "x-")))


def camera_text(orientation) -> str:
    """The camera ("rows, columns") whose images Stitch turns and mirrors by ``orientation``."""
    from xrd_tools.core.geometry.camera import stitch_camera
    camera = stitch_camera(orientation)
    return f"{camera.rows}, {camera.columns}"


def _camera_field(field, value):
    """The camera field showing ``value``: a value no camera is (an invalid xrayutilities file's blank) stays shown."""
    return replace(field, value=value, choices=CAMERAS + (() if value in CAMERAS else (str(value),)))


def camera_orientation(text) -> ImageOrientation:
    """The turn and mirror bringing a camera's ("rows, columns") images into Stitch's frame."""
    from xrd_tools.core.geometry.camera import DetectorCamera, stitch_orientation
    parts = tuple(part.strip() for part in str(text or "").split(","))
    if len(parts) != 2:
        raise ValueError("choose the detector camera (Advanced): its rows' and columns' directions, e.g. x-, z+")
    return stitch_orientation(DetectorCamera(*parts))


@functools.lru_cache(maxsize=8)
def _spec_counter_names(path, _state, scan):
    """A SPEC scan's counters (``#L`` columns but its motors) from its header alone, or None unread."""
    from xrd_tools.io.spec import read_spec_scan_counter_names
    try:
        return read_spec_scan_counter_names(path, scan)
    except (OSError, ValueError, KeyError, IndexError):
        return None


@functools.lru_cache(maxsize=8)
def _spec_frame_counts(path, _state, scans):
    """Each selected scan's frames (its SPEC points; ``scans`` None: every scan's), or None when the file
    cannot be read. On a committed source or scan edit only, as the incidence choices (maintainer,
    2026-09-30: "display the number of images in a scan in the readiness tab the moment it is selected like
    in int ... only if it is quick": measured about 0.5 ms a scan)."""
    from silx.io.specfile import SfError
    from xrd_tools.io.spec import count_spec_scan_points
    try:
        return count_spec_scan_points(path, scans)
    except (OSError, ValueError, LookupError, SfError):
        return None


def _file_states(*paths):
    """Each file's (mtime, size), or None when it is absent: a cache key that follows edits."""
    states = []
    for path in paths:
        try:
            state = Path(path).stat()
            states.append((state.st_mtime_ns, state.st_size))
        except OSError:
            states.append(None)
    return tuple(states)


@functools.lru_cache(maxsize=4)
def _xu_partner(path, project, _states):  # the readiness line asks on every edit
    """(the xrayutilities file Stitch reads for this pyFAI calibration, "") or ("", why none)."""
    from xrd_tools.integrate.stitch_calibration import xu_calibration_path
    try:
        return str(xu_calibration_path(path, project or None)), ""
    except ValueError as error:
        return "", str(error)


def _goniometer_positions(path) -> tuple[str, ...]:
    """The positions a pyFAI goniometer JSON declares, or () when it cannot be read."""
    record = _goniometer_record(path)
    return () if record is None else tuple(record["trans_function"]["pos_names"])


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


#: What a Stitch run is doing, by the operation's stage names.
_STAGE_TEXT = {"calibration": "Checking the calibration", "geometry": "Checking the geometry",
               "science": "Stitching", "projection": "Building the result", "publish": "Saving",
               "reload": "Checking the saved result"}


def _progress_text(stage, completed, total):
    """A run's progress in words. The operation counts steps, its frames plus four (the geometry,
    closing the stitch, saving, the check), and its "science" stage reports 1 + the frames done."""
    if stage == "science" and total > 4:
        return f"Stitching · frame {max(0, min(completed - 1, total - 4))} of {total - 4}"
    return f"{_STAGE_TEXT.get(stage, stage)}…"

#: The run bar's modes this page runs.
_STITCH_MODES = ("Stitch 1D", "Stitch 2D")


def _gi_motor(payload):
    """A GI result's incidence motor as its plan recorded it ("Manual" for a typed angle)."""
    gi = json.loads(payload.provenance_json).get("plan", {}).get("gi") or {}
    return str(gi.get("incidence_motor") or "Manual")


def _kind_name(payload):
    """A saved result's mode, as the run bar names it."""
    return ("Stitch 2D" if payload is not None and payload.kind is AnalysisArtifactKind.STITCH_2D
            else "Stitch 1D")


#: Recently shown results stay in memory within this many bytes, beside the shown ones (C12): about 55 Stitch 2D
#: results of 287 frames (4.65 MB each), far more Stitch 1D ones; least recently shown go first.
_RECENT_BYTES = 256 * 1024 * 1024

class StitchPage(QtWidgets.QWidget):
    """One retained page; one worker owns preview, science and strict reopen."""

    toolRequested = QtCore.Signal(str)
    #: (page key, processing mode): open that page in that mode (Codex audit G-F3)
    modeRequested = QtCore.Signal(str, str)

    def __init__(self, services, parent=None):
        super().__init__(parent)
        self.setObjectName("stitchPage")
        self._services = services
        self._owner = StitchToolOwner()
        self._source_paths = ()
        self._source_shown = ""  # the source box's display of the selection ("3 files selected")
        self._paths_by_kind = {}
        self._range_rows = []
        self._range_keys = set()
        #: Each field's key by its path (Int's for the rows Int's composer builds).
        self._keys_by_path = {}
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
        # What the readiness line last asked the disk, reused while a value is typed (AUD-3R)
        self._drafting, self._disk_facts = False, None
        #: What follows the preflight: "preview" shows the plan, "run" continues into the jobs
        #: (Run prepares changed inputs itself), "stopped" does neither (Stop while preparing).
        self._after_prepare = None
        #: Skipped scans and warnings of the prepared plan, for the status line.
        self._prepared_notes = ""
        #: The outputs a Run in Append mode skipped as already stitched, for the status line.
        self._skip_note = ""
        #: Int's readiness line: (summary, ready) from the section checks.
        self._readiness = ("", False)
        #: The committed SPEC selection and the frames its scans record (None: unknown), as Int's "N frames".
        self._frames = None
        self._action_revision = None
        self._view_revision = 0
        self._action_view_revision = None
        self._started = None
        self._closing = False
        #: The results in memory, by path: the shown ones (current, selected, holding a pin) and the most
        #: recently shown others within _RECENT_BYTES; the rest are listed by name and reloaded from their
        #: files when shown again (C12; a Stitch 2D of 287 frames holds 4.65 MB).
        self._payloads = {}
        #: Paths in the order they were last shown (the latest last), and each loaded result's bytes.
        self._recent, self._payload_bytes = [], {}
        #: Shown results not in memory, to reload in order, and the one reloading.
        self._reload_pending = []
        self._reloading = None
        self._diagnostic = "Intensity"
        #: One display key per result, created once: the view compares keys by identity.
        self._result_keys = {}
        #: The view's settings as Int keeps them: plot mode and axis, plot options, color map and log
        #: scale, and for a map its image axis, slice and pins.
        self._view = ScientificPreferences()
        #: The last map projection shown (a pin takes its slice); None while a pattern shows.
        self._last_map_state = None
        #: What a Run stitches: 'Stitch 1D' or 'Stitch 2D' (the run bar's modes).
        self._mode = "Stitch 1D"
        self._build_ui()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)
        self._authoring_timer = QtCore.QTimer(self)
        self._authoring_timer.setInterval(50)
        self._authoring_timer.timeout.connect(self._sync_actions)
        self._refresh_inputs_view()
        self._status(self._readiness[0], ready=self._readiness[1])
        self._sync_actions()
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
        self.results.itemSelectionChanged.connect(self._update_residency)
        self.results.currentItemChanged.connect(self._update_residency)
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
        # A Stitch 1D shows in Int 1D's layout (Q or 2θ), a Stitch 2D in Int 2D's without the raw
        # detector pane (the map has no one detector image; the contributing frames open in the raw
        # popup). Stitch has no normalization channel, 1-D Background or raw popup on this pane.
        self.scientific.set_hidden_tools("norm", "background", "raw", "raw_pane")
        self.scientific.set_plot_modes("Single", "Overlay")
        self.scientific.commandRequested.connect(self._view_command)
        self.curve = self.scientific.curve
        self._install_diagnostic_option()
        self.splitter.addWidget(self.scientific)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.controls = ControlsPanel(experiment_title="Configuration", show_section_numbers=False)  # as Int's
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
        # Int's Cores: the frames stitched at once, on either backend
        self.run_bar.apply_profile(modes=RUN_MODE_CHOICES, live=False, batch=False, cores=True)
        self.run_bar.set_unavailable_modes(UNOWNED_RUN_MODE_REASONS)
        self.run_bar.modeCombo.setCurrentText("Stitch 1D")
        self.run_bar.modeChanged.connect(self._mode_chosen)
        # Append skips an output already stitched with these settings; Replace stitches every one again
        # (maintainer, 2026-09-29: "skip if something has already been processed with the same
        # settings as before, else overwrite").
        self.run_bar.writeModeChanged.connect(self._write_mode_changed)
        self._write_mode_changed()
        self.run_bar.actionClicked.connect(self.begin_run)
        self.run_bar.stopClicked.connect(self.stop)
        right_layout.addWidget(self.run_bar)
        self.splitter.addWidget(right)
        # Int's columns (workspace_shell): the same proportions, so the two pages switch in place
        left.setMinimumWidth(255)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.setStretchFactor(2, 1)
        self.splitter.setSizes([289, 868, 323])

    def _control_action(self, action):
        """Int's action buttons on this page (``ControlsPanel.controlActionRequested``)."""
        {ControlAction.CALIBRATE: self._calibrate, ControlAction.MAKE_MASK: self._make_mask,
         ControlAction.PREVIEW: self.begin_preview, ControlAction.RETRY: self._retry,
         ControlAction.ADVANCED_PROCESSING: self.advanced.show}.get(action, lambda: None)()

    def _write_mode_changed(self, _mode=None):
        self.run_bar.writeModeButton.setToolTip(
            "Append: skip each output already stitched from the same frames with the same settings, "
            "and stitch the rest" if self.run_bar.write_mode() == "Append" else
            "Replace: stitch every output again; each file is replaced once its new stitch is written")

    def _mode_chosen(self, mode):
        """Stitch 1D or 2D here; another mode leaves for its page (the host refuses while Stitch runs)."""
        if mode in _STITCH_MODES and self.select_mode(mode):
            return
        blocker = QtCore.QSignalBlocker(self.run_bar.modeCombo)
        self.run_bar.modeCombo.setCurrentText(self._mode)
        del blocker
        if mode in _STITCH_MODES:
            self._status("Stitch is busy · change the mode once it is done", ready=False)
            return
        if mode == "RSM":
            self.toolRequested.emit(str(RSM_TOOL_KEY))
        else:  # the integration page, in the mode chosen here
            self.modeRequested.emit(str(SCATTERING_PAGE_KEY), str(mode))

    def select_mode(self, mode):
        """Stitch 1D or Stitch 2D (maintainer, 2026-09-29: a Stitch 2D shows as Int 2D does, without
        the raw pane): what a Run stitches, with Int 2D's rows for a map. Refused while Stitch is busy;
        returns whether the page took the mode."""
        if mode not in _STITCH_MODES or (mode != self._mode and self.active()):
            return False
        blocker = QtCore.QSignalBlocker(self.run_bar.modeCombo)
        self.run_bar.modeCombo.setCurrentText(mode)
        del blocker
        if mode != self._mode:
            self._mode = mode
            self._map_card.setVisible(mode == "Stitch 2D")
            self.advanced.setWindowTitle(f"Advanced — {mode}")
            self._edited()  # a prepared plan is another mode's; readiness checks this one
            self._draw()  # the mode's layout, as Int's follows its run-bar mode
        return True

    def _field(self, card, section, key, label, value="", *, choices=(), browse=False):
        field = self._new_field(section, key, label, value, choices=choices, browse=browse)
        row = FormRow(label=label, path=field.path, value=value, kind=field.kind.value,
                      choices=choices, browse=browse)
        if card is not None:
            card.add_row(row)
        return self._bind(key, field, row)

    @staticmethod
    def _new_field(section, key, label, value="", *, choices=(), browse=False, path=None):
        """One of this page's fields; ``path`` Int's for a row Int's composer lays out."""
        kind = (ControlFieldKind.COMBO if choices else ControlFieldKind.BOOL if isinstance(value, bool)
                else ControlFieldKind.LINE)
        if path is None:
            path = ("Project", "project_folder") if key == "project_root" else ("Stitch", key)
        return ControlFormField(section, label, path, value, kind, tuple(choices), browse)

    def _bind(self, key, field, row):
        """A row this page reads and edits by ``key``, built here or by Int's composer."""
        # The editor keeps the shared Controls object name, so Int's styling (one height
        # for every box) applies here too.
        row.editor.setProperty("stitchField", key)
        row.valueChanged.connect(self._edited)
        row.draftChanged.connect(self._draft_edited)
        row.browseRequested.connect(lambda _path, name=key: self._browse(name))
        self._fields[key], self._rows[key] = field, row
        self._keys_by_path[field.path] = key
        return row

    def _tip(self, key, text):
        """A tooltip on the row and its editor and label (Int's composer sets its own on all three)."""
        row = self._rows[key]
        for widget in (row, row.editor, row.label):
            widget.setToolTip(text)

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
        # Int's EXPERIMENT rows, from its composer: Calibrate · Make Mask on top, the Detector subsection (the
        # calibration and mask), then Configuration (Grazing, below); every action button is Int's
        detector = {"geometry_path": self._new_field(SectionId.EXPERIMENT, "geometry_path", "Calibration",
                                                     browse=True, path=PONI_FILE),
                    "mask_path": self._new_field(SectionId.EXPERIMENT, "mask_path", "Mask File", browse=True,
                                                 path=MASK_FILE)}
        built = self.controls.render_experiment_rows(experiment, tuple(detector.values()), (
            ControlActionSpec(ControlAction.CALIBRATE, "Calibrate", SectionId.EXPERIMENT),
            ControlActionSpec(ControlAction.MAKE_MASK, "Make Mask", SectionId.EXPERIMENT)))
        for key, field in detector.items():
            self._bind(key, field, built[field.path])
        self._detector_card = self._rows["geometry_path"].parentWidget()
        while not isinstance(self._detector_card, SubsectionCard):
            self._detector_card = self._detector_card.parentWidget()
        self.calibrate_button, self.mask_button = built[ControlAction.CALIBRATE], built[ControlAction.MAKE_MASK]
        self.calibrate_button.setToolTip("Calibrate the detector arm from a calibrant scan: the pyFAI goniometer "
                                         "file and its xrayutilities file.")
        self.mask_button.setToolTip("Draw a detector mask on one of the scan's frames.")
        self.controls.controlActionRequested.connect(self._control_action)
        self._rows["geometry_path"].valueChanged.connect(self._refresh_xu_geometry)
        self._rows["geometry_path"].valueChanged.connect(self._show_early_ceiling)
        # Grazing incidence as Int sets it, in the Experiment section (maintainer, 2026-09-29):
        # Int's own rows, rebuilt from this page's values (_render_gi).
        # tilt_motor None: the default (chi, gonchi, else Manual) once the scan's motors are known
        self._gi = dict(enabled=False, motor=None, angle="0.1", orientation="4", tilt_motor=None, tilt_angle="0")
        self._gi_catalog = None
        self._gi_host = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(self._gi_host).setContentsMargins(0, 0, 0, 0)
        experiment.add_row(self._gi_host)
        self.controls.fieldValueChanged.connect(self._gi_edited)
        # Int's Source rows, from its composer: the kind and (Directory only) Subdirs on one row, then the
        # files or folder and, as Int's Image Directory, the Filter (maintainer, 2026-09-30: "Directory mode
        # should have subfolders option as well as filters like in Int 2d"; "the subdirs toggle button should
        # be in the same row a the source like in int 1d/2d"). Every row is built once; a kind shows its own.
        fields = {
            "source_kind": self._new_field(SectionId.SOURCE, "source_kind", "Source", "Spec",
                                           choices=("Spec", "NeXus", "Directory"), path=SOURCE_TYPE),
            "spec_path": self._new_field(SectionId.SOURCE, "spec_path", "SPEC files", browse=True,
                                         path=SOURCE_DIRECTORY),
            "subdirs": self._new_field(SectionId.SOURCE, "subdirs", "Subdirs", False, path=SOURCE_RECURSIVE),
            "name_filter": self._new_field(SectionId.SOURCE, "name_filter", "Filter", path=SOURCE_FILTER),
            "image_dir": self._new_field(SectionId.SOURCE, "image_dir", "Images", browse=True),
            # on the card, not in Advanced: a stem left from another dataset matches the wrong images
            "image_stem": self._new_field(SectionId.SOURCE, "image_stem", "Image stem"),
            "scan_mode": self._new_field(SectionId.SOURCE, "scan_mode", "Scans", "Select Scans",
                                         choices=("Stitch All", "Select Scans")),
            "scan": self._new_field(SectionId.SOURCE, "scan", "Selection", "1"),
        }
        rows = self.controls.render_source_rows(source, tuple(fields.values()))
        for key, field in fields.items():
            self._bind(key, field, rows[field.path])
        self._tip("source_kind", "Source kind: SPEC files and their scans' images, NeXus files, or a folder "
                                 "of both.")
        self._tip("subdirs", "Also the SPEC and NeXus files in the folder's subfolders.")
        self._tip("image_stem",
                  "The part of the image file names that identifies the scan; any prefix before it is "
                  "fine (b_thampy_R1C2_scan3_0000.raw has R1C2_scan3_). Blank: <SPEC name>_scan<N>_ for "
                  "each scan. {spec} and {scan} fill in the SPEC name and scan number.")
        self._tip("scan", "1,2,[2-5,8] produces three outputs. Brackets combine scans into one result.")
        self._rows["source_kind"].valueChanged.connect(self._source_changed)
        self._rows["scan_mode"].valueChanged.connect(self._source_changed)
        self._rows["spec_path"].valueChanged.connect(self._source_path_edited)
        self._rows["scan"].valueChanged.connect(self._source_committed)  # committed, not drafts
        # A calibration serves both backends (its xrayutilities file is saved beside it), so the
        # backend is a processing choice (maintainer, 2026-09-29).
        self._field(processing, SectionId.PROCESSING, "backend", "Backend",
                    "pyFAI MultiGeometry", choices=("pyFAI MultiGeometry", "xrayutilities histogram"))
        self._rows["backend"].setToolTip(
            "pyFAI MultiGeometry reads the Calibration's pyFAI file; xrayutilities histogram reads the "
            "xrayutilities file saved beside it (<name>.xu.json), or an xrayutilities calibration chosen "
            "directly.")
        self._rows["backend"].valueChanged.connect(self._backend_changed)
        # Int's Processing rows, from its composer (maintainer, 2026-09-30: "it still feels like the conventions
        # and rules that were implemented in int 2d for display etc are not being uniformly applied"): each
        # group's Axis with its Pts, a RangeRow per range and the Threshold, on Int's paths with Int's labels and
        # rules; Stitch's GI ranges and Monitor beside them. Built once: the mode shows the 2-D group (the map on
        # its own axis and ranges, as Int 2D's), Grazing the GI ranges.
        self._axis_q = self._map_q = True  # each radial range's unit: Q (every axis but 2θ, 2θ-χ)
        self._gi_axis = "Q"  # the Grazing axis last chosen: the one a loaded GI form opens on
        for _group, key, path, value in _PROCESSING:
            self._fields[key] = self._new_field(SectionId.PROCESSING, key, key, value, path=path,
                                                choices=("None",) if key == "monitor" else ())
            if path[-1].endswith(("_auto", "_low", "_high")) or path[0] == "Mask":
                self._range_keys.add(key)
        fields = self._processing_fields()
        self._processing_cards = {}
        for name in ("1-D", "2-D", "Conditioning"):
            card = self._processing_cards[name] = SubsectionCard(name, accent="processing")
            processing.add_row(card)
            rows = self.controls.render_processing_rows(
                card, tuple(fields[key] for group, key, _path, _value in _PROCESSING if group == name))
            for group, key, path, _value in _PROCESSING:
                if group != name:
                    continue
                self._keys_by_path[path] = key
                row = rows[path]
                if key in self._range_keys:
                    if row not in self._range_rows:
                        row.valueChanged.connect(self._range_edited)
                        row.draftChanged.connect(self._edited)
                        self._range_rows.append(row)
                else:  # its projected field (a combo's kind and choices), which set_field applies
                    self._bind(key, fields[key], row)
        self._map_card = self._processing_cards["2-D"]
        range_row = {key: row for row in self._range_rows for path, key in self._keys_by_path.items()
                     if key in self._range_keys and path in (row._low_path, row._high_path)}
        self.radial_range, self.map_range, self.chi_range, self.threshold_range = (
            range_row[key] for key in ("q_min", "map_min", "chi_min", "threshold_min"))
        self._gi_rows = (range_row["gi_ip_min"], range_row["gi_oop_min"])
        for row, prefix in zip(self._gi_rows, _GI_RANGES):
            row.setToolTip(_GI_RANGES[prefix][1])
        self._tip("unit", "Grazing's Q, Qip and Qoop each write I(q) and both cuts (one pass gives both); the "
                          "result opens on the one chosen.")
        self._tip("map_axis", "The map's axes, as Int 2D's: Q-χ or 2θ-χ over the ranges below. Grazing writes "
                              "the (χ, q) map and the q_ip-q_oop map (over the Qip and Qoop ranges above); the "
                              "result opens on the one chosen.")
        self._tip("npt_rad_2d", "The map's radial points")
        self._tip("npt_azim_2d", "The map's χ points")
        self.map_range.setToolTip("The map's radial range (Auto: its frames' extent), its own as Int 2D's; "
                                  "I(q) beside it keeps the 1-D range.")
        self.chi_range.setToolTip("The map's χ range (Auto: its frames' extent). I(q) beside it takes every χ, "
                                  "as a Stitch 1D.")
        self._align_to_ranges(self._rows["monitor"])
        for key in ("unit", "map_axis"):
            self._rows[key].valueChanged.connect(self._processing_axis_changed)
        self._render_gi()
        self._map_card.setVisible(False)
        self.advanced = QtWidgets.QDialog(self)
        self.advanced.setWindowTitle("Advanced — Stitch 1D")
        advanced_layout = QtWidgets.QVBoxLayout(self.advanced)
        advanced_card = SubsectionCard("Source and geometry")
        advanced_layout.addWidget(advanced_card)
        fields = (
            ("detector_rows", "Rows", "195"), ("detector_cols", "Columns", "1475"),
            ("raw_dtype", "Data type", "int32"), ("raw_header_skip", "Header bytes", "0"),
            ("source_motors", "Angle mapping", "del_angle=del, nu_angle=nu"),
            ("poni_references", "PONI reference", "del=0, nu=0"),
            ("camera", "Camera (rows, columns)", CAMERAS[0]),
            ("frame_start", "First frame", "0"), ("frame_stop", "Last frame", ""),
            ("frame_step", "Frame step", "1"), ("max_frame_mib", "Max frame MiB", "256"),
        )
        self._field(advanced_card, SectionId.EXPERIMENT, "geometry_kind", "Geometry",
                    "Goniometer JSON", choices=("Goniometer JSON", "PONI + offsets"))
        for key, label, value in fields:
            self._field(advanced_card, SectionId.SOURCE, key, label, value,
                        choices=CAMERAS if key == "camera" else ())
        self._rows["camera"].setToolTip(
            "The detector's camera, as Calibrate ▸ 1 · Instrument ▸ Detector gives it: the lab directions the "
            "stored images' rows and columns run along (x up, z to the left looking downstream). Stitch turns "
            "and mirrors every frame, its mask, dark and flat into its frame (x-, z+, the default) as the "
            "calibration was made.")
        self._field(advanced_card, SectionId.SOURCE, "output_family", "Result names")
        self._rows["output_family"].setToolTip(
            "Blank names are automatic: <file>-<scan> for one scan, and the scans listed for a "
            "combined result ([2-5,8] gives <file>-2-5_8). To override, enter one name, or names in "
            "output order separated by ; (a blank entry stays automatic). At most 80 characters; "
            "processing and frame selection keep the same name.")
        for key in ("geometry_path", "geometry_kind"):
            self._rows[key].valueChanged.connect(self._sync_goniometer_mapping)
        self._rows["source_motors"].valueChanged.connect(self._mapping_edited)
        self._rows["source_motors"].setToolTip(
            "Which motor each calibration position reads. A position naming a circle (del_angle, "
            "nu_value; in fourc or twoc, del is the 2θ circle) follows that circle's motor under "
            "Calibrate ▸ 1 · Instrument, and editing it here renames that motor there. Any other "
            "position keeps the motor given here. An XU calibration defines its own.")
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
        # Grazing incidence's tilt, read per frame like the incidence (maintainer, 2026-09-29): the
        # motor is psic's chi circle; Manual takes Int's one tilt angle for every frame (2026-09-30,
        # option (b): "okay with (b)"). Rendered from self._gi.
        self._tilt_rows = (
            FormRow(label="GI tilt motor", path=("Stitch", "gi_tilt_motor"), value="Manual", kind="combo",
                    choices=("Manual",)),
            FormRow(label="GI tilt angle (°)", path=("Stitch", "gi_tilt_angle"), value="0", kind="line"))
        self._tilt_rows[0].setToolTip(
            "Grazing incidence: the motor giving each frame's chi circle (psic's χ; 90° is level at a "
            "chi-90 mounting). With the incidence, mu and phi it sets the sample surface for both "
            "backends. Default: chi, then gonchi, else Manual (one tilt angle, as Int's).")
        self._tilt_rows[1].setToolTip("Manual tilt: Int's Tilt angle, pyFAI's tilt of the sample (0 is level), for "
                                      "every frame; the incidence still comes from the circles.")
        for row in self._tilt_rows:
            row.valueChanged.connect(self._tilt_edited)
            row.draftChanged.connect(self._edited)
            frames_card.add_row(row)
        self._render_tilt(self._gi_catalog)
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
        # Int's action row: Preview · Retry (after a failure) · Advanced
        buttons = self.controls.render_actions(processing, tuple(
            ControlActionSpec(action, label, SectionId.PROCESSING) for action, label in (
                (ControlAction.PREVIEW, "Preview"), (ControlAction.RETRY, "Retry"),
                (ControlAction.ADVANCED_PROCESSING, "Advanced"))), expand=True)
        self.preview_button, self.retry_button, self.advanced_button = (
            buttons[action] for action in (ControlAction.PREVIEW, ControlAction.RETRY, ControlAction.ADVANCED_PROCESSING))
        self.preview_button.setToolTip("Check the inputs and list what a Run stitches, without stitching.")
        self.retry_button.setToolTip("Retry the last result's saving or cleanup.")
        self.advanced_button.setToolTip("The source, geometry, frame and correction settings.")
        self.preview_dialog = QtWidgets.QDialog(self)
        self.preview_dialog.setWindowTitle("Stitch preview")
        preview_layout = QtWidgets.QVBoxLayout(self.preview_dialog)
        self.preview_text = QtWidgets.QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        preview_layout.addWidget(self.preview_text)
        self.preview_dialog.resize(680, 480)

    _GI_KEYS = {GI_ENABLED: "enabled", GI_MOTOR: "motor", GI_THETA: "angle", GI_ORIENTATION: "orientation"}

    def _gi_catalog_now(self):
        """The real motors of the first selected SPEC scan (Int's incidence choices), or None when
        the source does not say (another source kind, or an unreadable file)."""
        first = self._first_spec_scan()
        return None if first is None else _spec_motor_names(first[0], _file_states(first[0]), first[1])

    def _first_spec_scan(self):
        """(the first SPEC file, the first selected scan's N.M key, None: the file's first), or None
        for another source kind."""
        if self._value("source_kind") != "Spec" or not self._source_paths:
            return None
        first = re.search(r"\d+(?:\.\d+)?", str(self._value("scan") or ""))
        return self._source_paths[0], (None if self._value("scan_mode") == "Stitch All" or first is None
                                       else first[0] if "." in first[0] else f"{first[0]}.1")  # SPEC's N.M key

    def _monitor_choices(self, keep):
        """Monitor's choices: the first selected SPEC scan's counters (``monitor_choices``), and ``keep``
        (the value chosen or restored) whatever that scan records."""
        first = self._first_spec_scan()
        counters = (None if first is None else _spec_counter_names(first[0], _file_states(first[0]), first[1]))
        choices = monitor_choices(counters or ())
        keep = str(keep or "").strip() or "None"
        return choices if keep in choices else (*choices, keep)

    def _refresh_monitor_choices(self):
        """On a committed source or scan edit (Codex audit G-F2): the new scan's counters."""
        field = replace(self._fields["monitor"], value=self._value("monitor"),
                        choices=self._monitor_choices(self._value("monitor")))
        self._fields["monitor"] = field
        self._rows["monitor"].apply_field(field)

    def _gi_sample_motors(self):
        """The Geometry's other sample circles (mu, phi) by its motor names: both backends read them
        with the incidence and tilt. Preview decides from every selected scan (Codex review F3): a
        circle no scan records reads 0 and Preview says so; one only some scans record is refused."""
        from xrd_tools.integrate.stitch_gi import GI_SAMPLE_ROLES
        geometry = self._instrument()
        circles = {role: circle for circle, role in geometry.as_psic().items()}
        names = dict(geometry.motors)
        return tuple((role, names.get(circles.get(role, role))) for role in GI_SAMPLE_ROLES)

    def _gi_fields(self):
        """Int's GI fields (its paths and rows) for this page's values."""
        gi = self._gi
        fields = [ControlFormField(SectionId.EXPERIMENT, "Grazing", GI_ENABLED, gi["enabled"], ControlFieldKind.BOOL)]
        if not gi["enabled"]:
            return tuple(fields)
        motor = gi["motor"] or "Manual"
        catalog = self._gi_catalog
        choices = ("Manual", *(catalog if catalog is not None else (() if motor == "Manual" else (motor,))))
        fields.append(ControlFormField(SectionId.EXPERIMENT, "Incidence motor", GI_MOTOR, motor,
                                       ControlFieldKind.COMBO, tuple(dict.fromkeys(choices))))
        if motor == "Manual":
            fields.append(ControlFormField(SectionId.EXPERIMENT, "Incidence angle", GI_THETA, gi["angle"]))
        # the tilt is read per frame like the incidence: Advanced ▸ GI tilt motor
        fields.append(ControlFormField(SectionId.EXPERIMENT, "Sample orientation", GI_ORIENTATION, gi["orientation"]))
        return tuple(fields)

    def _gi_status(self):
        """Int's fixed status (``ControlsPanel._experiment_status``): a longer one widened the panel (maintainer,
        2026-09-30); what a Q or 2θ Axis gives in Grazing is on the eliding readiness line."""
        return ControlsPanel._experiment_status(self._gi_fields())

    def _render_gi(self):
        """Int's GI rows in the Experiment section, rebuilt from this page's values; the incidence
        motor defaults as Int's does, in Stitch's order (eta, gonth, halpha, …), never to a motor
        the scan does not record."""
        from xrd_tools.session.gi_motor import (
            STITCH_GI_MOTOR_PREFERENCE, STITCH_TILT_MOTOR_PREFERENCE, pick_default_gi_motor,
        )
        self._gi_catalog = catalog = self._gi_catalog_now() if self._gi["enabled"] else None
        motor = self._gi["motor"]
        if catalog is not None and (motor is None or (motor != "Manual" and motor not in catalog)):
            self._gi["motor"] = pick_default_gi_motor(catalog, STITCH_GI_MOTOR_PREFERENCE)
        tilt = self._gi["tilt_motor"]
        if catalog is not None and (tilt is None or (tilt != "Manual" and tilt not in catalog)):
            self._gi["tilt_motor"] = pick_default_gi_motor(catalog, STITCH_TILT_MOTOR_PREFERENCE,
                                                           rotation_fallback=False)
        if hasattr(self, "_tilt_rows"):  # built with the Advanced dialog, after Experiment
            self._render_tilt(catalog)
        layout = self._gi_host.layout()
        while layout.count():
            widget = layout.takeAt(0).widget()
            if widget is not None:  # hidden, not orphaned: a parentless widget is a top-level window
                widget.hide()
                widget.deleteLater()
        # parented at birth: a parentless card is a top-level window until it is laid out
        self._gi_card = group = SubsectionCard(self.controls.experiment_title, self._gi_host,
                                               status=self._gi_status(), accent="experiment")
        self.controls.render_gi_rows(group, self._gi_fields())
        layout.addWidget(group)
        self._render_gi_axis()

    def _render_tilt(self, catalog):
        """The Advanced tilt rows from this page's values: the scan's motors as choices."""
        tilt = self._gi["tilt_motor"] or "Manual"
        choices = ("Manual", *(catalog if catalog is not None else (() if tilt == "Manual" else (tilt,))))
        motor, chi = self._tilt_rows
        motor.apply_field(ControlFormField(SectionId.PROCESSING, "GI tilt motor", motor._path, tilt,
                                           ControlFieldKind.COMBO, tuple(dict.fromkeys(choices))))
        chi.apply_field(ControlFormField(SectionId.PROCESSING, "GI tilt angle (°)", chi._path,
                                         self._gi["tilt_angle"], ControlFieldKind.LINE))
        chi.setVisible(tilt == "Manual")

    def _tilt_edited(self, path, value):
        key = {"gi_tilt_motor": "tilt_motor", "gi_tilt_angle": "tilt_angle"}[path[-1]]
        if self._gi[key] == str(value):
            return
        self._gi[key] = str(value)
        self._tilt_rows[1].setVisible((self._gi["tilt_motor"] or "Manual") == "Manual")
        self._edited()

    def _render_gi_axis(self):
        """The Axis choices, as Int's: Standard's Q and 2θ (the 2-D Q-χ and 2θ-χ), Grazing's Q, Qip and
        Qoop (the 2-D Qip-Qoop and Q-χ); a choice Grazing lacks becomes its Q (its last axis for the 1-D), the
        GI ranges in Grazing, and the Experiment status."""
        grazing = self._gi["enabled"]
        for key, choices, fallback in (
                ("unit", tuple(_GI_AXES) if grazing else _STANDARD_AXES,
                 (self._gi_axis if self._gi_axis in _GI_AXES else "Q") if grazing else "Q (Å⁻¹)"),
                ("map_axis", _GI_MAP_AXES if grazing else _STANDARD_MAP_AXES, "Q-χ")):
            field, current = self._fields[key], self._value(key)
            if field.choices != choices or current not in choices:
                self._fields[key] = replace(field, choices=choices, value=current if current in choices else fallback)
                self._rows[key].apply_field(self._fields[key])
        for row in self._gi_rows:
            row.setVisible(grazing)
        status = self._gi_status()
        if getattr(self, "_gi_card", None) is not None:
            self._gi_card.status.setText(status)
            self._gi_card.status.setVisible(bool(status))
        self.controls.experiment_card.set_status_text(status)  # as Int's EXPERIMENT header
        self._processing_axis_changed()

    def _gi_edited(self, path, value):
        key = self._GI_KEYS.get(tuple(path))
        if key is None:
            return
        value = bool(value) if key == "enabled" else str(value)
        if self._gi[key] == value:
            return
        self._gi[key] = value
        if key in ("enabled", "motor"):  # the rows shown change
            self._render_gi()
        self._edited()

    def _gi_value(self):
        """The form's grazing incidence in Grazing (I(q) and both cuts, whichever axis it opens on, as Int's
        Grazing is a GI run), else None."""
        value = self._value
        if not self._gi["enabled"]:
            return None
        from xrd_tools.integrate.stitch_gi import GI_CUTS, StitchGI
        motor, tilt = self._gi["motor"] or "Manual", self._gi["tilt_motor"] or "Manual"
        ip, oop = ((None if value(f"{prefix}_auto") else (float(value(f"{prefix}_min")), float(value(f"{prefix}_max"))))
                   for prefix in ("gi_ip", "gi_oop"))
        sample_motors = self._gi_sample_motors()
        read = {None if motor == "Manual" else motor, None if tilt == "Manual" else tilt,
                *(name for _role, name in sample_motors)} - {None}
        geometry = self._instrument()  # the Diffractometer panel's motor zeros (item 7)
        zeros = tuple((name, geometry.zero(circle)) for circle, name in geometry.motors
                      if name in read and geometry.zero(circle) is not None)
        # and its senses: a circle turning opposite to psic's (maintainer, 2026-09-30)
        flipped = tuple(name for circle, name in geometry.motors if name in read and geometry.sense(circle) < 0)
        return StitchGI(
            GI_CUTS, incidence_motor=None if motor == "Manual" else motor,
            incident_angle_deg=float(self._gi["angle"]) if motor == "Manual" else None,
            tilt_motor=None if tilt == "Manual" else tilt,
            tilt_angle_deg=float(self._gi["tilt_angle"]) if tilt == "Manual" else None,
            sample_orientation=int(self._gi["orientation"]), sample_motors=sample_motors,
            ip_range=ip, oop_range=oop, motor_zeros=zeros, reversed_motors=flipped)

    def _load_gi(self, gi):
        """A form's (or settings') grazing incidence onto the page. A GI form writes both cuts; the cut a
        result opens on is the page's view (the GI axis last chosen, which settings keep), not the
        form's."""
        self._gi["enabled"] = gi is not None
        if gi is not None:
            self._gi.update(motor=gi.incidence_motor or "Manual", tilt_motor=gi.tilt_motor or "Manual",
                            orientation=str(gi.sample_orientation))
            if gi.incident_angle_deg is not None:
                self._gi["angle"] = f"{gi.incident_angle_deg:g}"
            if gi.tilt_angle_deg is not None:
                self._gi["tilt_angle"] = f"{gi.tilt_angle_deg:g}"
            self._render_gi()  # Grazing's axis choices, before choosing one
            self.set_field("unit", self._gi_axis)  # the one it opens on: the Grazing axis last chosen
            for prefix, bounds in (("gi_ip", gi.ip_range), ("gi_oop", gi.oop_range)):
                self.set_field(f"{prefix}_auto", bounds is None)
                if bounds is not None:
                    self.set_field(f"{prefix}_min", bounds[0])
                    self.set_field(f"{prefix}_max", bounds[1])
        self._render_gi()
        self._edited()

    def _processing_fields(self):
        """This page's Processing fields by key, as Int projects them (``project_control_fields``: its labels,
        the range rows' following the axes, and its rules: a range's bounds are greyed under Auto, the
        Threshold's limits when it is off); Stitch's GI ranges and Monitor with Int's range rule."""
        from xdart.gui.tabs.scattering.controls_inventory import project_control_fields

        def current(key):
            return self._value(key) if key in self._rows else self._fields[key].value
        values = {path: current(key) for _group, key, path, _value in _PROCESSING if path[0] != "Stitch"}
        # the 1-D range is I(q)'s whatever the Grazing axis; the 2-D ranges the (χ, q) map's
        values.update({("Int1D", "unit"): "2th_deg" if current("unit") == "2θ (°)" else "q_A^-1",
                       ("Int1D", "gi_mode"): "q_total",
                       ("Int2D", "unit"): STANDARD_2D_AXES.get(current("map_axis"), "q_A^-1"),
                       ("Int2D", "gi_mode"): "q_chi"})
        grazing = self._gi["enabled"]
        projected = {field.path: field for field in project_control_fields(values, {
            INT_1D_AXIS: tuple(_GI_AXES) if grazing else _STANDARD_AXES,
            INT_2D_AXIS: _GI_MAP_AXES if grazing else _STANDARD_MAP_AXES})}
        fields = {}
        for _group, key, path, _value in _PROCESSING:
            if path in projected:
                fields[key] = projected[path]
            elif key == "monitor":
                fields[key] = replace(self._fields[key], label="Monitor", value=current(key),
                                      choices=self._monitor_choices(current(key)))
            else:  # a GI range's bound or Auto, with Int's range rule
                prefix, part = key.rsplit("_", 1)
                label, auto = _GI_RANGES[prefix][0], bool(self._fields[f"{prefix}_auto"].value)
                fields[key] = replace(
                    self._fields[key], value=current(key),
                    label=f"{label} {dict(min='Low', max='High', auto='Auto')[part]}",
                    kind=ControlFieldKind.BOOL if part == "auto" else ControlFieldKind.LINE,
                    enabled=part == "auto" or not auto, reason="Disable Auto to edit this range." if (
                        part != "auto" and auto) else "")
        # the decision-39 hint: the automatic ceiling is the Max box's grey text, never its value
        fields["threshold"] = replace(fields["threshold"], placeholder=self._fields["threshold"].placeholder)
        return fields

    def _render_ranges(self):
        """Int's projection of the Processing fields onto their rows in place (Int's same-schema update):
        values, what is greyed, and the range rows' labels following the axes."""
        fields = self._processing_fields()
        by_path = {field.path: field for field in fields.values()}
        for row in self._range_rows:
            row.apply_fields(by_path)
            label = by_path[row._low_path].label
            row.label.setText("Threshold" if row._low_path == THRESHOLD_MIN else label.removesuffix(" Low"))

    def _show_auto_ceiling(self, ceilings):
        """The automatic ceiling (decision 39) as the Max box's grey hint, never its value: the one
        value every scan shares, else "per scan". The transitions' default maximum is the largest
        value it keeps; typing any number makes the threshold manual."""
        values = {ceiling[0] for ceiling in ceilings if ceiling is not None}
        if len(ceilings) and None not in ceilings and len(values) == 1:
            value = values.pop()
            source = ("saturation" if any(ceiling[1] == "saturation" for ceiling in ceilings)
                      else "data type maximum")
            self._auto_threshold_max, hint = float(value - 1), f"Auto · {value} ({source})"
        else:
            self._auto_threshold_max, hint = None, "Auto · per scan" if values else "Auto"
        self._fields["threshold"] = replace(self._fields["threshold"], placeholder=hint)
        self._render_ranges()

    @property
    def _threshold_mode(self):
        state = self._threshold_state
        return "manual" if state.apply_threshold else "auto" if state.mask_saturation else "off"

    @property
    def _threshold_manual(self):
        state = self._threshold_state
        return state.apply_threshold or state.threshold_min is not None or state.threshold_max is not None

    def _range_edited(self, path, value):
        key = self._keys_by_path.get(tuple(path), path[-1])
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
                    ("threshold", result.threshold_max if result.threshold_max is not None or result.apply_threshold else None),
                    ("threshold_enabled", result.apply_threshold or result.mask_saturation)):
                self._fields[name] = replace(self._fields[name], value=projected)
            self._render_ranges()
            self._edited()
            return
        old = self._fields[key].value
        if not key.endswith("_auto"):
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
        self._adopt_sources(paths)
        count = len(self._source_paths)
        self._source_shown = "" if not count else self._source_paths[0] if count == 1 else f"{count} files selected"
        field = replace(self._fields["spec_path"], value=self._source_shown)
        self._fields["spec_path"] = field
        self._rows["spec_path"].apply_field(field)
        self._show_source_tip()
        self._edited()
        self._source_committed()

    def _source_path_edited(self, _path, value):
        """A committed edit of the source box. Only a typed path is a selection: the box's own
        display of the current one ("3 files selected") is not a path, and an empty box is no
        source (Codex R19-F1: it kept the previous one, and Run stayed ready)."""
        text = str(value or "").strip()
        if text == self._source_shown:
            return
        self._adopt_sources((text,) if text else ())
        self._source_shown = text
        self._show_source_tip()
        self._edited()  # the row's own edit signal refreshed readiness before the selection changed
        self._source_committed()

    def _adopt_sources(self, paths):
        """The selection Run, Preview and the saved settings use, with what depends on it."""
        self._show_auto_ceiling(())
        if not self._threshold_manual:
            self._fields["threshold"] = replace(self._fields["threshold"], value=None)
            self._render_ranges()
        self._source_paths = tuple(str(path) for path in paths)
        self._paths_by_kind[self._value("source_kind")] = self._source_paths

    def _show_source_tip(self):
        row = self._rows["spec_path"]
        for widget in (row, row.editor, row.label):
            widget.setToolTip("\n".join(self._source_paths))

    def _source_changed(self, *_args):
        kind = self._value("source_kind")
        # the stem also names the images of the SPEC scans a Directory finds: shown for both, so no
        # hidden value governs them (Codex R20-F2); NeXus sources have no stem
        for key in ("image_dir", "image_stem", "scan_mode", "scan"):
            self._rows[key].setVisible(kind != "NeXus" and (key in ("image_dir", "image_stem") or kind == "Spec"))
        for key in ("subdirs", "name_filter"):
            self._rows[key].setVisible(kind == "Directory")
        self._rows["scan"].setVisible(kind == "Spec" and self._value("scan_mode") == "Select Scans")
        self._rows["spec_path"].label.setText({"Spec": "SPEC files", "NeXus": "NeXus files", "Directory": "Folder"}[kind])
        if _args and tuple(_args[0]) == self._fields["source_kind"].path:
            self.set_sources(self._paths_by_kind.get(kind, ()))
        self._source_committed()

    def _processing_axis_changed(self, *_args):
        """An Axis choice: a change of unit resets its range to the unit's defaults (every 1-D axis but 2θ
        bins I(q); a map Q-χ or 2θ-χ), and the range rows' labels follow."""
        if self._gi["enabled"] and self._value("unit") in _GI_AXES:
            self._gi_axis = self._value("unit")
        for axis, q, low, high in (("unit", self._value("unit") != "2θ (°)", "q_min", "q_max"),
                                   ("map_axis", self._value("map_axis") != "2θ-χ", "map_min", "map_max")):
            attribute = "_axis_q" if axis == "unit" else "_map_q"
            if q != getattr(self, attribute):
                setattr(self, attribute, q)
                self._fields[low] = replace(self._fields[low], value=0.)
                self._fields[high] = replace(self._fields[high], value=6.2 if q else 90.)
        self._render_ranges()

    def set_field(self, key, value):
        if key in self._range_keys:
            self._range_edited(self._fields[key].path, value)
            return
        if key == "unit":
            value = _LEGACY_AXES.get(value, value)
        field = replace(self._fields[key], value=value)
        if key == "camera":
            field = _camera_field(field, value)
        if key == "monitor":  # "" in older settings: none; any other name stays a choice
            value = str(value or "").strip() or "None"
            field = replace(field, value=value, choices=self._monitor_choices(value))
        self._fields[key] = field
        self._rows[key].apply_field(field)
        self._edited()
        if key == "backend":
            self._backend_changed()
        elif key == "spec_path":
            self._source_path_edited(field.path, value)
        elif key in {"source_kind", "scan_mode"}:
            self._source_changed(field.path, value)
        elif key == "scan":
            self._source_committed()
        elif key in {"unit", "map_axis"}:
            self._processing_axis_changed()
        elif key == "project_root" and not self._value("output_path"):
            self.set_field("output_path", str(Path(value) / "xdart_processed_data"))
        if key in {"geometry_path", "project_root"}:
            self._refresh_xu_geometry()
        if key == "geometry_path":
            self._show_early_ceiling()
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
            self._refresh_inputs_view()
            self._status(self._readiness[0], ready=self._readiness[1])
            self._sync_actions()

    def _draft_edited(self, *_args):
        """A keystroke: an edit like any other, except that the readiness line asks the disk nothing (AUD-3R,
        maintainer 2026-09-30: "yes for cheap fix"): its file checks and calibration reads stay those of
        the last committed values until the typed one is committed."""
        self._drafting = True
        try:
            self._edited()
        finally:
            self._drafting = False

    def _ask_disk(self, project, calibration, mask, sources, xu):
        """The readiness line's disk facts: (the Project exists, the calibration exists, the mask exists, why
        the calibration cannot drive the arm or None, the sources not found). A draft reuses the last ones."""
        if self._drafting and self._disk_facts is not None:
            return self._disk_facts
        project_ready = bool(str(project).strip()) and Path(str(project)).is_dir()
        calibration_found = bool(calibration) and Path(calibration).is_file()
        mask_found = not mask or Path(mask).is_file()
        arm_problem = ((self._calibration_pair(calibration, xu) or self._arm_problem())
                       if calibration_found and mask_found else None)
        gone = tuple(path.name for path in sources if not path.exists())
        self._disk_facts = (project_ready, calibration_found, mask_found, arm_problem, gone)
        return self._disk_facts

    def _source_committed(self, *_args):
        """A committed source or scan edit, never a draft (Codex audit G-F2: the page's source reads):
        the frames the selection's scans record, and the first scan's motors."""
        frames = self._frames
        self._frames = (self._selection_key(), self._selection_frames())
        if self._frames != frames and hasattr(self, "run_bar"):  # the run bar is built after the sections
            self._refresh_inputs_view()
            self._status(self._readiness[0], ready=self._readiness[1])
        self._refresh_gi_catalog()
        self._refresh_monitor_choices()
        self._show_early_ceiling()

    def _show_early_ceiling(self, *_args):
        """Threshold Auto's ceiling as soon as the selection is committed, as Int shows it (maintainer,
        2026-09-30: "the threshold Auto doesn't show the saturation value"): what the selection gives
        without reading its images (Codex audit G-F2), a NeXus container's recorded value and data type,
        else the calibration detector's documented counter limit. Preview's per-scan value replaces it."""
        from xrd_tools.core.invalid import resolve_detector_saturation
        recorded = dtype = None
        if self._value("source_kind") == "NeXus" and self._source_paths:
            first = self._source_paths[0]
            read = _container_ceiling(first, _file_states(first))
            if read is not None:
                recorded, dtype = read
        ceiling = resolve_detector_saturation(
            recorded=recorded, detector=_calibration_detector(self._value("geometry_path")), dtype=dtype)
        self._show_auto_ceiling(() if ceiling is None else (
            (ceiling.value, "data type" if ceiling.source == "dtype" else "saturation"),))

    def _selection_key(self):
        return (self._value("source_kind"), self._source_paths, self._value("scan_mode"), str(self._value("scan")))

    def _selection_frames(self):
        """The frames a SPEC selection's scans record (a scan in two results counts twice, as it is
        stitched twice), or None for another kind or a scan a file lacks."""
        from xrd_tools.sources.grouping import parse_stitch_scan_groups
        kind, paths, mode, scans = self._selection_key()
        if kind != "Spec" or not paths:
            return None
        try:
            groups = None if mode == "Stitch All" else parse_stitch_scan_groups(scans)
        except ValueError:
            return None
        wanted = None if groups is None else tuple(sorted({scan for group in groups for scan in group}))
        total = 0
        for path in paths:
            counts = _spec_frame_counts(path, _file_states(path), wanted)
            if counts is None or None in counts.values():
                return None
            total += sum(counts.values()) if groups is None else sum(
                sum(counts[scan] for scan in dict.fromkeys(group)) for group in groups)
        return total

    def _frames_text(self):
        """" · N frames" for the committed selection shown, else "" (a draft, or unknown)."""
        if self._frames is None or self._frames[0] != self._selection_key() or self._frames[1] is None:
            return ""
        count = self._frames[1]
        return f" · {count} frame{'s' * (count != 1)}"

    def _refresh_gi_catalog(self, *_args):
        """On a committed source or scan edit, never a draft: another first scan's motors are the
        incidence and tilt choices (the page's one source read, Codex audit G-F2)."""
        if self._gi["enabled"] and self._gi_catalog_now() != self._gi_catalog:
            self._render_gi()
            self._edited()  # a default motor may have changed

    def _automatic_names(self, kind, sources):
        """The result names the selection gets when Result names is blank, as Int names results:
        a NeXus container's stem, _gi for grazing incidence. A SPEC scan's is its images' name, which
        only Preview reads (Codex audit G-F2: no folder is listed on an edit)."""
        from xrd_tools.analysis.stitch_sources import int_scan_name
        from xrd_tools.io.output_path import generated_artifact_family, scan_family
        gi = self._gi["enabled"]
        if kind == "Spec":
            return "auto · as Int names the scan's images" + (", with _gi" if gi else "")
        if kind != "NeXus" or not sources:
            return "automatic"
        try:
            return "auto · " + generated_artifact_family(
                scan_family([int_scan_name(path) for path in sources]), grazing_incidence=gi)
        except ValueError:
            return "automatic"

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
        # a Directory stitches every scan it finds: its stem hint names no scan number
        first = re.match(r"\s*\[?\s*(\d+)", scans) if kind == "Spec" else None
        self._rows["image_dir"].editor.setPlaceholderText(
            f"auto · {spec.parent}" if spec else "auto · the SPEC file's folder")
        self._rows["image_stem"].editor.setPlaceholderText(
            f"auto · {spec.stem if spec else '<SPEC name>'}_scan{first[1] if first else '<N>'}_")
        self._rows["output_family"].editor.setPlaceholderText(self._automatic_names(kind, sources))
        self.controls.apply_card_visibility(str(value("project_root") or ""))  # Int's gate: a Project first
        calibration = str(value("geometry_path") or "").strip()
        mask = str(value("mask_path") or "").strip()
        xu = value("backend") == "xrayutilities histogram"
        project_ready, calibration_found, mask_found, arm_problem, gone = self._ask_disk(
            value("project_root") or "", calibration, mask, sources, xu)
        experiment_ready = bool(calibration) and calibration_found and mask_found
        detector = ControlsPanel._detector_status(tuple(  # Int's Detector status: fitted, or a mask
            replace(self._fields[key], value=value(key)) for key in ("geometry_path", "mask_path")))
        self._detector_card.status.setText(detector)
        self._detector_card.status.setVisible(bool(detector))
        arm_problem = arm_problem if experiment_ready else None
        experiment_ready = experiment_ready and arm_problem is None
        present = bool(sources) and not gone
        try:
            groups = len(parse_stitch_scan_groups(scans)) if kind == "Spec" and value("scan_mode") != "Stitch All" else None
        except ValueError:
            groups = 0
        if kind == "Spec":
            text = f"{len(sources)} SPEC file{'s' * (len(sources) != 1)} · " + (
                "all scans" if groups is None else f"{groups} output{'s' * (groups != 1)}") + self._frames_text()
        elif kind == "Directory":
            text = (f"{sources[0].name} + subfolders" if value("subdirs") else sources[0].name) if sources else ""
            text += " · filtered" if sources and str(value("name_filter")).strip() else ""
        else:
            text = f"{len(sources)} {'NeXus file' if kind == 'NeXus' else f'{kind} source'}{'s' * (len(sources) != 1)}"
        source_ready = present and groups != 0
        self.controls.source_card.set_status_text(text if sources else "")
        map_mode = self._mode == "Stitch 2D"
        try:
            processing_ready = all(int(str(value(key))) > 0 for key in (
                ("npt_1d", "npt_rad_2d", "npt_azim_2d") if map_mode else ("npt_1d",)))
        except ValueError:
            processing_ready = False
        self.controls.processing_card.set_status_text(
            f"{self._mode.lower()} · " + ("xrayutilities" if xu else "multigeometry"))
        # What each input still needs, in a few words, as Int's "Needs …" (maintainer, 2026-09-30:
        # "slightly more descriptive (but brief) messages")
        needs = {}
        if not project_ready:
            needs["project"] = "an existing Project folder" if str(value("project_root")).strip() else "a Project folder"
        if not experiment_ready and not arm_problem:
            needs["experiment"] = ("a calibration" if not calibration else
                                   f"the calibration {Path(calibration).name} (not found)"
                                   if not calibration_found
                                   else f"the mask {Path(mask).name} (not found)")
        if not source_ready:
            needs["source"] = ({"Spec": "SPEC files", "NeXus": "NeXus files", "Directory": "a source folder"}
                               .get(kind, "a source") if not sources else f"{gone[0]} (not found)" if gone
                               else "a scan selection such as 1,2,[3-5]")
        if not processing_ready:
            needs["processing"] = "Pts above 0"
        checks = ((self.controls.project_card, "project"), (self.controls.experiment_card, "experiment"),
                  (self.controls.source_card, "source"), (self.controls.processing_card, "processing"))
        for card, key in checks:
            problem = arm_problem if key == "experiment" and arm_problem else (
                f"Needs {needs[key]}" if key in needs else "")
            card.set_valid_marker(not problem, problem)
        # As Int's readiness line; Run then checks the inputs in full (Preview's preflight).
        if arm_problem:  # the calibration's arm and the Geometry disagree: a sentence of its own
            self._readiness = (arm_problem, False)
        elif needs:
            self._readiness = ("Needs " + ", ".join(needs.values()), False)
        else:
            self._readiness = (f"Ready · {self._mode} · {self._ready_text(kind, sources, scans, groups)}", True)

    def _ready_text(self, kind, sources, scans, groups):
        """What a Run stitches, briefly: the SPEC file and scans (or the folder and its options, or the
        NeXus files), the backend, and grazing incidence when it is on."""
        value = self._value
        if kind == "Spec":
            label = sources[0].name if len(sources) == 1 else f"{len(sources)} SPEC files"
            shown = scans.strip()
            if groups is None:
                text = f"{label}, all scans"
            elif len(shown) <= 24:
                text = f"{label}, scan{'s' * any(mark in shown for mark in ',-[')} {shown}"
            else:
                text = f"{label}, {groups} scan groups"
            outputs = (groups or 0) * len(sources)
            text += (f" → {outputs} outputs" if outputs > 1 else "") + self._frames_text()
        elif kind == "Directory":
            text = f"every scan in {sources[0].name}" + (" and its subfolders" if value("subdirs") else "")
            name_filter = str(value("name_filter")).strip()
            text += f" matching “{name_filter}”" if name_filter else ""
        elif kind == "NeXus":
            text = (sources[0].name if len(sources) == 1 else f"{len(sources)} NeXus files") + " → 1 output"
        else:  # a kind a settings file names that this version lacks (it is refused before a Run)
            text = f"{len(sources)} {kind} source{'s' * (len(sources) != 1)}"
        text += " · " + ("xrayutilities" if value("backend") == "xrayutilities histogram" else "pyFAI MultiGeometry")
        if self._gi["enabled"]:
            text += f" · grazing {value('unit')}"
        return text

    def _backend_changed(self, *_args):
        xu = self._value("backend") == "xrayutilities histogram"
        self._sync_advanced()
        self.set_field("source_motors", "del=del, nu=nu" if xu else "del_angle=del, nu_angle=nu")
        if not xu and not str(self._value("camera")).strip():
            self.set_field("camera", CAMERAS[0])
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
        """Bind every position the goniometer JSON declares (Angle mapping, Advanced) to the
        motor of its circle under Calibrate ▸ Geometry (del_value → the del motor; in fourc or
        twoc del is the 2θ circle), so a calibration file's position names never have to match
        the scan's. A position that names no circle keeps the motor the mapping gives it, else
        its name without its suffix. PONI + offsets keeps the del/nu positions its mapping
        names (a fresh one maps the Geometry's arm), one reference each. XU owns its mapping.
        """
        if not hasattr(self, "advanced") or self._value("backend") == "xrayutilities histogram":
            return
        geometry = self._instrument()
        try:
            given = dict(_parse_pairs(self._value("source_motors") or ""))
        except ValueError:
            given = {}
        if self._value("geometry_kind") == "Goniometer JSON":
            positions = _goniometer_positions(self._value("geometry_path"))
        else:  # PONI + offsets: the circles the mapping names (a loaded form's, kept as it is);
            # a fresh mapping takes the Geometry's arm
            positions = tuple(name for name in ("del", "nu") if name in given) or tuple(
                circle for circle, motor in zip(("del", "nu"), geometry.stitch_arm()) if motor is not None)
            self._sync_poni_references(positions)
        if not positions:
            return
        pairs = []
        for name in positions:
            circle = geometry.circle_for_position(name)
            motor = geometry.motor(circle) if circle else given.get(name) or _POSITION_SUFFIX.sub("", name) or name
            pairs.append(f"{name}={motor}")
        if ", ".join(pairs) != self._value("source_motors"):
            self.set_field("source_motors", ", ".join(pairs))

    def _sync_poni_references(self, positions):
        """PONI + offsets: one reference angle per arm circle (0 unless one is given)."""
        try:
            given = dict(_parse_pairs(self._value("poni_references") or ""))
        except ValueError:
            given = {}
        text = ", ".join(f"{name}={given.get(name, '0')}" for name in positions)
        if text != self._value("poni_references"):
            self.set_field("poni_references", text)

    def _mapping_edited(self, *_args):
        """An edited Angle mapping: a circle's position renames that circle's motor under
        Geometry; any other position keeps the motor given here."""
        try:
            pairs = _parse_pairs(self._value("source_motors") or "")
        except ValueError as error:
            self._status(str(error), ready=False)
            return
        self._adopt_mapping(pairs)

    def _calibration_pair(self, calibration, xu):
        """Name the Calibration's files on its row; return why Backend xrayutilities cannot use
        it (a pyFAI file with no partner made from it), else None."""
        from xrd_tools.integrate.stitch_calibration import xu_partner_path
        path = Path(calibration)
        if _goniometer_record(path) is None:  # an xrayutilities calibration, or a PONI
            self._rows["geometry_path"].setToolTip("")
            return None
        partner_file = xu_partner_path(path)
        partner, problem = _xu_partner(str(path), str(self._value("project_root") or ""), _file_states(
            path, partner_file, partner_file.with_name(partner_file.name + ".provenance.json")))
        if problem:
            self._rows["geometry_path"].setToolTip(
                f"pyFAI MultiGeometry reads {path.name}; it has no xrayutilities file ({partner_file.name}).")
            return problem if xu else None
        self._rows["geometry_path"].setToolTip(
            f"pyFAI MultiGeometry reads {path.name}; xrayutilities histogram reads {Path(partner).name}, saved with it.")
        return None

    def _arm_problem(self):
        """Why the calibration cannot drive the Geometry's detector arm (None when it can):
        the circles it moves and holds come from what it is bound to — a goniometer JSON's
        transformation, a PONI + offsets form's del/nu mapping — never from position names."""
        geometry = self._instrument()
        try:
            bindings = tuple(_parse_pairs(self._value("source_motors") or ""))
        except ValueError:
            return None  # the form refuses a malformed mapping itself
        if self._value("backend") == "xrayutilities histogram":  # the XU calibration's own mapping
            moving = {name for name, _motor in bindings} & {"del", "nu"}
            if not moving:
                return None  # an unusable XU calibration is refused where it is read
            return geometry.arm_mismatch(moving, {"del", "nu"} - moving)
        if self._value("geometry_kind") != "Goniometer JSON":  # PONI + offsets: del and/or nu
            moving = {name for name, _motor in bindings} & {"del", "nu"}
            # a mapping naming neither is refused by the form itself
            return geometry.arm_mismatch(moving, {"del", "nu"} - moving) if moving else None
        path = Path(str(self._value("geometry_path") or "").strip())
        if _goniometer_record(path) is None:
            return None  # an unreadable calibration is refused where it is read
        state = path.stat()
        found = _bound_arm(str(path), state.st_mtime_ns, state.st_size, bindings)
        if found[0] == "ok":
            return geometry.arm_mismatch(found[1], found[2])
        if None not in geometry.stitch_arm():
            return None  # psic holds no circle: the operation reads the calibration itself
        return (f"This calibration's detector arm cannot be established ({found[1]}), and "
                f"{geometry.geometry} holds a detector circle at zero. Choose psic under Calibrate ▸ "
                "1 · Instrument, or calibrate its 2θ arm.")

    def _instrument(self):
        """The diffractometer and motor names: Calibrate ▸ 1 · Instrument (saved with the
        settings); older settings name the del and nu motors, or the Angle mapping does. An older
        window's normal-incidence zero becomes its circle's motor zero (Codex review F1)."""
        from xrd_tools.session.instrument import InstrumentGeometry
        from .calibration import migrated_instrument
        if self._calibration_dialog is not None:
            return self._calibration_dialog.geometry_editor.value()
        saved = self._calibration_settings or {}
        try:
            return migrated_instrument(InstrumentGeometry.from_record(saved["instrument_geometry"]), saved)
        except (KeyError, ValueError, TypeError):
            pass
        fields = saved.get("fields") or {}
        motors = {circle: fields[f"{circle}_motor"] for circle in ("del", "nu")
                  if isinstance(fields.get(f"{circle}_motor"), str) and fields[f"{circle}_motor"].strip()}
        if not motors:
            try:
                pairs = _parse_pairs(self._value("source_motors") or "")
            except ValueError:
                pairs = ()
            default = InstrumentGeometry()
            motors = {circle: motor for position, motor in pairs
                      if (circle := default.circle_for_position(position)) in ("del", "nu")}
        return migrated_instrument(InstrumentGeometry("psic", tuple(motors.items())), saved)

    def _set_instrument(self, geometry):
        """Keep one geometry: the calibration window's, or the settings it will open with."""
        if self._calibration_dialog is not None:
            self._calibration_dialog.set_instrument(geometry)
            return
        saved = dict(self._calibration_settings or {"fields": {}, "choices": {}})
        saved["instrument_geometry"] = geometry.to_record()
        self._calibration_settings = saved
        self._instrument_changed()

    def _instrument_changed(self, *_args):
        """The mapping follows the new geometry, and readiness and any prepared plan are
        rechecked: the calibration may no longer drive its arm. The GI card shows nothing of the
        geometry (Preview decides the sample circles, G-F2), so it is not rebuilt: rebuilding it on
        every Diffractometer keystroke and on Read metadata closed the calibration window on macOS
        (maintainer, 2026-09-29)."""
        self._sync_goniometer_mapping()
        self._edited()

    def _adopt_mapping(self, pairs):
        """Explicit position → motor pairs (a loaded form) become the geometry's motors (its plane
        and motor zeros kept)."""
        geometry = self._instrument()
        motors = dict(geometry.motors)
        for position, motor in pairs:
            circle = geometry.circle_for_position(position)
            if circle is not None:
                motors[circle] = motor
        self._set_instrument(replace(geometry, motors=tuple(motors.items())))

    def _refresh_xu_geometry(self, *_args, calibration=None):
        """Present asset-owned settings through the existing strict admission owner."""
        if self._value("backend") != "xrayutilities histogram":
            return
        from xrd_tools.analysis.xu_stitch_calibration import (
            XuStitchCalibrationInput, capture_xu_stitch_calibration,
        )
        try:
            if calibration is None:
                from xrd_tools.integrate.stitch_calibration import xu_calibration_path
                project = Path(self._value("project_root"))
                calibration = capture_xu_stitch_calibration(
                    XuStitchCalibrationInput(xu_calibration_path(self._value("geometry_path"), project)),
                    project_root=project)
            acquisition = calibration.projection.value["acquisition"]
        except (ValueError, TypeError, OSError):
            values = {"source_motors": "", "camera": ""}
            tip = "Select a valid XU calibration to inspect its angle mapping and camera."
        else:
            values = dict(source_motors=", ".join(  # a single 2θ arm's asset names no nu motor
                f"{name}={source}" for name, source in acquisition["source_motors"].items() if source),
                camera=camera_text(ImageOrientation(rotation=90 * acquisition["image_rotation_quarter_turns"],
                                                    flip_horizontal=acquisition.get("image_flip", False))))
            tip = "Defined by the selected XU calibration; change it in Calibrate."
        values.update(geometry_kind="Goniometer JSON", poni_references="")
        for key, value in values.items():
            field = replace(self._fields[key], value=value)
            if key == "camera":
                field = _camera_field(field, value)
            self._fields[key] = field
            self._rows[key].apply_field(field)
            self._rows[key].setToolTip(tip)
        self._edited()  # readiness rechecks the arm the calibration's own mapping names

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
            if kind == "Spec":  # SPEC files have no extension: only they are listed
                picks = dialogs.browse_spec(self, "Choose SPEC files", many=True, **common)
            else:
                picks = dialogs.browse(self, "Choose source files", many=True, **common,
                                       file_filter=dialogs.NEXUS_FILTER)
            if picks:
                self.set_sources(picks)
            return
        else:
            title, file_filter = files.get(key, ("Choose " + self._fields[key].label, dialogs.ANY_FILTER))
            picks = dialogs.browse(self, title, file_filter=file_filter, **common)
        if picks:
            self.set_field(key, picks[0])

    def build_form(self):
        problem = self._arm_problem()  # Run and Preview never stitch another instrument's arm
        if problem:
            raise ValueError(problem)
        value = self._value
        kind = (StitchGeometryKind.PYFAI_GONIOMETER_JSON if value("geometry_kind") == "Goniometer JSON"
                else StitchGeometryKind.PONI)
        orientation = camera_orientation(value("camera"))  # the detector camera's turn and mirror
        monitor = str(value("monitor")).strip()
        monitor = "" if monitor == "None" else monitor
        directory = value("source_kind") == "Directory"
        selection = StitchSourceSelection(value("source_kind"), self._source_paths,
            scans=str(value("scan")), stitch_all=value("scan_mode") == "Stitch All",
            image_dir=str(value("image_dir")), image_stem=str(value("image_stem")),
            recursive=directory and bool(value("subdirs")),
            name_filter=str(value("name_filter")).strip() if directory else "")
        conditioning = StitchConditioning(self._threshold_mode,
            value("threshold_min") if self._threshold_mode == "manual" else None,
            value("threshold") if self._threshold_mode == "manual" else None)
        native = dict(source_selection=selection,
            unit="2th_deg" if value("unit") == "2θ (°)" else "q_A^-1",
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
        map_mode = self._mode == "Stitch 2D"
        output = str(directory / ("stitch_stitch2d.nexus" if map_mode else "stitch_stitch1d.nexus"))
        if map_mode:  # the map on its own axis and ranges, as Int 2D's (Auto: the frames' own extent)
            native.update(mode="2d", npt_rad_2d=int(value("npt_rad_2d")), npt_azim_2d=int(value("npt_azim_2d")),
                          chi_range=None if value("chi_auto") else (float(value("chi_min")), float(value("chi_max"))),
                          map_unit=STANDARD_2D_AXES.get(value("map_axis"), "q_A^-1"),  # Grazing's maps: Q
                          map_radial_range=None if value("map_auto")
                          else (float(value("map_min")), float(value("map_max"))))
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
            image_rotation=orientation.rotation, image_flip=orientation.flip_horizontal,
            threshold=None if value("threshold") is None or not str(value("threshold")).strip() else float(value("threshold")),
            q_range=None if value("radial_auto") else (float(value("q_min")), float(value("q_max"))), npt_1d=int(value("npt_1d")),
            output_path=output, monitor_selector=MetadataColumnSelector(monitor) if monitor else None,
            backend="xu_hist" if value("backend") == "xrayutilities histogram" else "multigeometry",
            max_frame_bytes=int(value("max_frame_mib")) * 1024 * 1024,
            gi=self._gi_value(),
            **native,
        )

    def load_form(self, form):
        """Restore a prepared form, including native source and conditioning intent. A GI form's
        results open on the GI axis this page last chose (view state, restored with the settings;
        Codex re-review R2-F3)."""
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
            camera=camera_text(ImageOrientation(rotation=form.image_rotation, flip_horizontal=form.image_flip)),
            frame_start=form.frame_selector.start_label,
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
        if form.mode == "2d":  # Auto is None, as saved: an explicit −180 to 180 stays explicit
            chi, radial = form.chi_range, form.map_radial_range
            if form.gi is None:  # a Grazing map's 2-D Axis is the one it opens on: the page's
                fields.update(map_axis="2θ-χ" if form.map_unit == "2th_deg" else "Q-χ")
            fields.update(npt_rad_2d=form.npt_rad_2d, npt_azim_2d=form.npt_azim_2d,
                          chi_auto=chi is None, map_auto=radial is None)
            if chi is not None:
                fields.update(chi_min=chi[0], chi_max=chi[1])
            if radial is not None:
                fields.update(map_min=radial[0], map_max=radial[1])
        self.select_mode("Stitch 2D" if form.mode == "2d" else "Stitch 1D")
        for key, value in fields.items():
            self.set_field(key, value)
        # the form's own mapping and references, as saved: an earlier field change may have
        # synced them to the page's geometry meanwhile
        for key in ("source_motors", "poni_references"):
            self.set_field(key, fields[key])
        self._adopt_mapping(form.source_motors)  # the form's motors become the geometry's
        selection = form.source_selection
        self.set_field("source_kind", selection.kind)
        self.set_sources(selection.paths)
        self.set_field("scan_mode", "Stitch All" if selection.stitch_all else "Select Scans")
        self.set_field("scan", selection.scans)
        self.set_field("output_path", str(Path(form.output_path).parent))
        self.set_field("image_dir", selection.image_dir)
        self.set_field("image_stem", selection.image_stem)
        self.set_field("subdirs", selection.recursive)
        self.set_field("name_filter", selection.name_filter)
        policy = form.conditioning or StitchConditioning()
        self._threshold_state = ThresholdState(policy.mode == "manual",
            policy.threshold_min, policy.threshold_max, policy.mode == "auto")
        for key, value in (("threshold_min", policy.threshold_min if policy.mode == "manual" else 0.),
                           ("threshold", policy.threshold_max), ("threshold_enabled", policy.mode != "off")):
            self._fields[key] = replace(self._fields[key], value=value)
        self._render_ranges()
        self._sync_advanced()
        self._refresh_xu_geometry()
        self._load_gi(form.gi)

    def _commit_typing(self):
        """Commit a field still being typed in, before an action reads the inputs. A click on Run
        leaves the focus in the field on macOS, and disabling the controls for the action then
        committed it mid-check, which discarded the check as "Inputs changed" (live)."""
        focused = QtWidgets.QApplication.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit) and (self.isAncestorOf(focused)
                                                         or self.advanced.isAncestorOf(focused)):
            focused.clearFocus()  # the row commits on the focus-out, as Return or Tab would

    def _start(self, action, begin):
        if self._closing:
            return False
        if self._authoring_active():
            self._status("Finish the calibration or mask editor before starting another operation", ready=False)
            return False
        self._commit_typing()
        try:
            identity = begin()
        except (TypeError, ValueError, OSError) as error:
            self._queued_jobs.clear()
            self._status(str(error), ready=False)
            return False
        if identity is None:
            self._status("Finish the current operation first", ready=False)
            return False
        self._action_revision = self._revision
        self._action_view_revision = self._view_revision
        self._started = time.monotonic()
        self._status(action)
        self._timer.start()
        self._sync_actions()
        return True

    def begin_preview(self):
        self._prepare("Preparing Preview…", then="preview")

    def _prepare(self, action, *, then):
        def begin():
            self._owner.set_form(self.build_form())
            self._prepared_revision = None
            return self._owner.begin_preflight()
        self._after_prepare = then
        if not self._start(action, begin):
            self._after_prepare = None

    def begin_run(self):
        # Typing still in a field is part of this Run's inputs: committed first, it makes the check
        # below run again instead of Append reusing Preview's (Codex testing-fixes review F1).
        self._commit_typing()
        if self._prepared_revision != self._revision or self._owner.prepared is None:
            # As Int's Run: the inputs are checked on the way (Preview's preflight), then run.
            self._prepare("Checking the inputs…", then="run")
            return
        self._run_prepared()

    def _run_prepared(self):
        prepared = self._owner.prepared
        count = len(prepared.requests) if prepared is not None else 0
        if not count:
            self._status("Nothing to stitch" + self._prepared_notes, ready=False)
            return
        self._prepared_revision = None  # a run uses its plan once: the next Run checks again
        # A Run right after Preview takes Preview's check: an output since removed is stitched again.
        kept = tuple(index for index in prepared.unchanged
                     if Path(prepared.requests[index].module.output.target).is_file()
                     ) if self.run_bar.write_mode() == "Append" else ()
        self._queued_jobs = [index for index in range(count) if index not in kept]
        if not self._queued_jobs:
            self._status(f"Up to date · {count} {'output' if count == 1 else 'outputs'} already stitched "
                         "with these settings (Replace ⇄ stitches again)" + self._prepared_notes)
            return
        self._skip_note = f" · {len(kept)} already stitched, skipped" if kept else ""
        self._job_total = len(self._queued_jobs)
        self._batch_started = time.monotonic()
        self._run_next_job()

    def _run_next_job(self):
        if not self._queued_jobs or self._closing:
            return
        index = self._queued_jobs.pop(0)
        position = self._job_total - len(self._queued_jobs)
        if not self._start(f"Running · {self._mode} · {position}/{self._job_total}{self._skip_note}{self._prepared_notes}",
                           lambda: self._owner.begin_run(job_index=index, workers=self.run_bar.get_cores())):
            self._queued_jobs.clear()
            self._prepared_revision = None  # the next Run prepares again

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
        if self._after_prepare == "run":  # the check finishes; nothing is stitched after it
            self._after_prepare = "stopped"
            self._status("Stopping…")
        if self._owner.cancel():
            self._status("Stopping…")
        self._sync_actions()
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
                self._status(_progress_text(p.stage, p.completed, p.total))
            elif update.terminal_status is OperationTerminalStatus.FAILED:
                self._queued_jobs.clear()
                self._after_prepare = None
                reloading, self._reloading = self._reloading, None
                self._status((f"Could not reload {Path(reloading).name}: " if update.action is StitchOwnerAction.REOPEN
                              and reloading is not None else "") + (update.failure_message or update.failure_type),
                             ready=False)
                if update.action is StitchOwnerAction.METADATA and self._metadata_current():
                    self._metadata_dialog.status.setText(update.failure_message or update.failure_type)
                if update.action is StitchOwnerAction.FRAME_READ and self._raw_popup is not None and self._raw_request_revision == self._raw_revision:
                    self._raw_popup.caption.setText("Image unavailable · " + (update.failure_message or update.failure_type))
            elif update.terminal_status is OperationTerminalStatus.CANCELLED:
                self._queued_jobs.clear()
                self._after_prepare = None
                self._reloading = None
                self._status("Cancelled")
            elif update.outcome is not None:
                self._accept_outcome(update)
        if not self._owner.busy:
            self._timer.stop()
            if self._raw_pending and not self._queued_jobs:
                QtCore.QTimer.singleShot(0, self._request_raw)
            if self._reload_pending and not self._queued_jobs:
                QtCore.QTimer.singleShot(0, self._request_reload)
        self._sync_actions()

    def _accept_outcome(self, update):
        outcome = update.outcome
        if outcome.kind is StitchOwnerOutcomeKind.METADATA:
            if self._metadata_current():
                self._metadata_dialog.adopt_sources(outcome.metadata)
            self._status(f"Ready · saved {_kind_name(self._current_payload())} result")
        elif outcome.kind is StitchOwnerOutcomeKind.PREFLIGHT_READY:
            then, self._after_prepare = self._after_prepare, None
            if then == "stopped":
                self._status("Stopped before stitching", ready=False)
                return
            if update.stale or self._action_revision != self._revision:
                self._status("Inputs changed while checking · " + ("Run" if then == "run" else "Preview")
                             + " again", ready=False)
                return
            self._prepared_revision = self._revision
            preflight = outcome.preflight
            if preflight.requests and preflight.form.backend == "xu_hist":
                self._refresh_xu_geometry(calibration=preflight.requests[0].plan.calibration)
            self._show_auto_ceiling({ceiling for request, members in zip(preflight.requests, preflight.preview_members)
                                     for ceiling in native_ceilings(request, members)})
            done = len(preflight.unchanged)
            lines = [f"{len(preflight.requests)} stitched outputs"
                     + (f" · {done} already stitched with these settings (Append skips them)" if done else "")]
            if preflight.warnings:
                lines.extend(("\nWarnings", *preflight.warnings))
            for index, request in enumerate(preflight.requests, 1):
                count = len(request.module.source.selected_labels)
                lines.append(f"\n{index}. {Path(request.module.output.target).name} · {count} frames"
                             + (" · already stitched" if index - 1 in preflight.unchanged else ""))
                for member in preflight.preview_members[index - 1]:
                    identity = member["identity"]
                    lines.append(f"  {identity['scan_id']}-{identity['frame_index']} · {member['source_path']}")
            if preflight.skipped:
                lines.append("\nSkipped scans")
                lines.extend(f"{Path(path).name} · scan {scan}: {reason}" for path, scan, reason in preflight.skipped)
            self.preview_text.setPlainText("\n".join(lines))
            counts = ((len(preflight.skipped), "scan skipped", "scans skipped"),
                      (len(preflight.warnings), "warning", "warnings"))
            notes = [f"{n} {one if n == 1 else many}" for n, one, many in counts if n]
            self._prepared_notes = f" · {', '.join(notes)} (Preview lists them)" if notes else ""
            if then == "run":
                self._run_prepared()
                return
            self.preview_dialog.show()
            self._status(f"Ready · {self._mode} · {len(preflight.requests)} outputs"
                         + (f" · {done} already stitched" if done else "") + self._prepared_notes,
                         ready=bool(preflight.requests))
        elif outcome.kind is StitchOwnerOutcomeKind.FRAME_READ:
            if (self._raw_request_revision == self._raw_revision and not self._closing
                    and self._raw_popup is not None and self._raw_popup.isVisible()
                    and self._current_payload() is not None
                    and outcome.artifact_path == self._current_payload().inspection.path):
                self._raw_frame_view = outcome.frame_view
                self._render_raw()
                self._status(f"Ready · saved {_kind_name(self._current_payload())} result")
        elif outcome.kind is StitchOwnerOutcomeKind.REOPENED and self._reloading is not None:
            path, self._reloading = self._reloading, None
            if not self._closing and path in self._resident_paths():
                self._restore_payload(path, outcome.payload)
                self._status(f"Ready · saved {_kind_name(outcome.payload)} result")
        elif outcome.kind is StitchOwnerOutcomeKind.REOPENED:
            if self._action_view_revision == self._view_revision and not self._closing:
                self._mount_payload(outcome.artifact_path, outcome.payload)
                note = saved_frame_screen_note(outcome.payload.contributing_frames)
                self._status(f"Ready · saved {_kind_name(outcome.payload)} result" + (f" · {note}" if note else ""))
        elif outcome.kind is StitchOwnerOutcomeKind.RESULT:
            result = outcome.result
            if result.terminal.disposition is ModuleDisposition.COMMITTED:
                if not update.stale and self._action_revision == self._revision and not self._closing:
                    self._mount_payload(result.request.module.output.target, result.payload)
                elapsed = time.monotonic() - (self._batch_started or self._started)
                count = len(result.request.module.source.selected_labels)
                note = frame_screen_note(result.frame_screen)
                self._status(f"Complete · {count} frames · {elapsed:.2f} s" + (f" · {note}" if note else "")
                             + self._skip_note + self._prepared_notes)
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
                color_map=self._view.color_map, log=self._view.log_scale)

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
            if not project:
                raise ValueError("Choose a Project folder first")
            if self._calibration_dialog is not None and self._calibration_project != project:
                self._calibration_settings = self._calibration_dialog.settings()
                self._calibration_dialog.hide()
                self._calibration_dialog = None
            # The processing scan only fills a new dialog with no calibration source: the
            # calibrant scan is usually another scan, and saved or edited inputs are kept.
            unconfigured = False
            if self._calibration_dialog is None:
                geometry = self._instrument()  # before the window exists: settings, or migrated
                self._calibration_dialog = StitchCalibrationDialog(str(self._value("project_root")), parent=self)
                saved = self._calibration_settings or {}
                if saved:
                    self._calibration_dialog.apply_settings(saved)
                self._calibration_dialog.set_instrument(geometry)
                self._calibration_dialog.geometryChanged.connect(self._instrument_changed)
                unconfigured = not (saved.get("fields") or {}).get("source")
                self._calibration_project = project
                self._calibration_dialog.calibrationSaved.connect(self._adopt_calibration)
                self._authoring_dialogs.append(self._calibration_dialog)
            source = self._source_spec_for_tools() if unconfigured else None
            if source is not None:  # the window names the motors from the geometry it was given
                orientation = self._image_orientation()
                self._calibration_dialog.set_source_context(source, image_rotation=orientation.rotation,
                    image_flip=orientation.flip_horizontal,
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
        self.set_field("geometry_kind", "Goniometer JSON")
        self.set_field("geometry_path", path)
        dialog = self._calibration_dialog
        if dialog is not None:  # the fit's own instrument and motors, as saved with it
            from xrd_tools.session.instrument import InstrumentGeometry
            try:
                fitted = InstrumentGeometry.from_record(dialog.saved_instrument_geometry)
            except (ValueError, TypeError):
                fitted = None
            if fitted is not None and fitted != self._instrument():
                self._set_instrument(fitted)
            if dialog.saved_mask:  # the calibration's mask is this detector's: Stitch leaves it out too
                self.set_field("mask_path", dialog.saved_mask)
            # the pyFAI side always (Backend may change later); xrayutilities then shows its partner's
            self._adopt_mapping(tuple(dialog.saved_source_motors.items()))
            self._sync_goniometer_mapping()
            self.set_field("camera", camera_text(ImageOrientation(rotation=dialog.saved_image_rotation,
                                                                  flip_horizontal=dialog.saved_image_flip)))
            self._refresh_xu_geometry()

    def _image_orientation(self):
        """The camera's turn and mirror into Stitch's frame (none while it is blank or malformed)."""
        try:
            return camera_orientation(self._value("camera"))
        except ValueError:
            return ImageOrientation()

    def _make_mask(self):
        from .mask import StitchMaskDialog
        if self._owner.busy or self._closing or self._authoring_active():
            return
        try:
            source = self._source_spec_for_tools()
            dialog = StitchMaskDialog(str(self._value("project_root")), source=source,
                rotation=self._image_orientation().rotation, reader_options=dict(
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

    def _set_view(self, **changes):
        self._view = replace(self._view, **changes)

    def _mount_payload(self, path, payload):
        if type(payload) is not AnalysisArtifactPayload or payload.inspection.kind not in {
                AnalysisArtifactKind.STITCH_1D, AnalysisArtifactKind.STITCH_2D}:
            raise TypeError("Stitch page requires an admitted Stitch result")
        key = str(path)
        self._payloads[key] = payload
        self._payload_bytes[key] = payload.resident_bytes
        item = next((self.results.item(i) for i in range(self.results.count())
                     if self.results.item(i).data(QtCore.Qt.ItemDataRole.UserRole) == key), None)
        if item is None:
            item = QtWidgets.QListWidgetItem(Path(key).name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            item.setToolTip(key)
            self.results.addItem(item)
            self._result_key(key)
        if payload.gi:  # a GI result opens on the axis Grazing chose (I(q) or a cut), else on a cut it has
            modes = {mode for mode, _cut in payload.gi}
            chosen = _GI_AXES.get(self._value("unit")) if self._gi["enabled"] else None
            if chosen == "q_total":
                self._set_view(plot_axis="Q")
            elif chosen in modes:
                self._set_view(plot_axis=chosen)
            elif self._view.plot_axis not in modes:
                self._set_view(plot_axis=payload.gi[0][0])
            if payload.gi_map is not None and self._gi["enabled"]:  # and on the map the 2-D Axis chose
                self._set_view(image_axis="qip_qoop" if self._value("map_axis") == "Qip-Qoop" else "q_chi")
        # shown once: selecting it would rebuild the frame list and redraw through the list's signals,
        # and again here (Codex testing-fixes review, startup recommendation 2)
        blocker = QtCore.QSignalBlocker(self.results)
        self.results.setCurrentItem(item)
        del blocker
        self._update_residency()  # a run's outputs replace each other in memory as they are shown
        self._current_result_changed()

    def _selected(self):
        """The selected results in memory (a reloading one joins when it is back)."""
        items = self.results.selectedItems()
        if self._view.plot_mode == "Single":
            item = self.results.currentItem()
            items = [] if item is None else [item]
        role = QtCore.Qt.ItemDataRole.UserRole
        return [(item, self._payloads[item.data(role)]) for item in items if item.data(role) in self._payloads]

    def _resident_paths(self):
        """The results shown: the current one, the selected ones (Single shows the current one only)
        and those holding a slice pin."""
        role = QtCore.Qt.ItemDataRole.UserRole
        paths = ({item.data(role) for item in self.results.selectedItems()}
                 if self._view.plot_mode != "Single" else set())
        current = self.results.currentItem()
        if current is not None:
            paths.add(current.data(role))
        pinned = {id(pin.frame) for pin in self._view.slice_pins}
        return paths | {path for path, key in self._result_keys.items() if id(key) in pinned}

    def _update_residency(self, *_args):
        """C12 (STITCH-2D-20260930; Codex general audit AUD-5, narrowed by architecture invariant 12): the
        results shown stay in memory, and so do the most recently shown others within _RECENT_BYTES, so
        going back to one is immediate (maintainer, 2026-09-30: "can we hold more than one, say abot 20, or
        even 48 so browsing is smoother?"); another is dropped and read from its file again when shown (a
        processed file can always be read, or stitched again)."""
        resident = self._resident_paths()
        self._recent = [path for path in self._recent if path not in resident] + sorted(resident)
        kept, used = set(resident), 0
        for path in reversed(self._recent):  # the latest shown first
            if path not in resident and path in self._payloads:
                used += self._payload_bytes.get(path, 0)
                if used > _RECENT_BYTES:
                    break
                kept.add(path)
        for path in [path for path in self._payloads if path not in kept]:
            del self._payloads[path]
            self._payload_bytes.pop(path, None)
        self._recent = [path for path in self._recent if path in self._payloads or path in resident]
        role = QtCore.Qt.ItemDataRole.UserRole
        listed = [self.results.item(row).data(role) for row in range(self.results.count())]
        self._reload_pending = [path for path in dict.fromkeys(self._reload_pending + listed)
                                if path in resident and path not in self._payloads]
        if self._reload_pending:
            QtCore.QTimer.singleShot(0, self._request_reload)

    def _request_reload(self):
        """Read the next shown result back from its file, through the owner, between its other work:
        never inside a run's queue of outputs (a refused start would end the run's queue)."""
        if self._closing or self._owner.busy or self._queued_jobs or self._reloading is not None:
            return
        resident = self._resident_paths()
        self._reload_pending = [path for path in self._reload_pending
                                if path in resident and path not in self._payloads]
        if not self._reload_pending:
            return
        path = self._reload_pending[0]
        self._reloading = path
        if self._start(f"Loading {Path(path).name}…", lambda: self._owner.begin_reopen(path)):
            self._reload_pending.pop(0)
        else:  # refused (an open editor, a pending finalization): tried again once Stitch is idle
            self._reloading = None

    def _restore_payload(self, path, payload):
        """A reloaded result back in memory: the list and its selection stay as they are."""
        self._payloads[path] = payload
        self._payload_bytes[path] = payload.resident_bytes
        current = self.results.currentItem()
        if current is not None and current.data(QtCore.Qt.ItemDataRole.UserRole) == path:
            self._current_result_changed()  # its frames, and the view
        else:
            self._draw()

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
        """Project the selected results into Int's scientific view in the mode's layout, as Int's follows
        its run-bar mode (maintainer, 2026-09-30: "see how this is done in int 2d"): Stitch 2D in Int 2D's
        (the map, with I(q) and its slices below; no map for a Stitch 1D result), Stitch 1D in Int 1D's
        (a Stitch 2D result's I(q), as Int 1D shows a 2-D record's)."""
        if not hasattr(self, "scientific"):
            return
        self._draw_projection("Int 2D" if self._mode == "Stitch 2D" else "Int 1D")
        self.curve.setLabel("left", scaled_intensity_label(self._diagnostic,
                                                           self._view.plot_options.intensity_scale))
        blocker = QtCore.QSignalBlocker(self._diagnostic_combo)
        self._diagnostic_combo.setCurrentText(self._diagnostic)
        del blocker

    def _navigation(self):
        """Every result's key, the current result's and the selected results' (the view's footer)."""
        from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection
        role = QtCore.Qt.ItemDataRole.UserRole
        keys = tuple(self._result_key(self.results.item(row).data(role)) for row in range(self.results.count()))
        current = self.results.currentItem()
        return keys, FrameNavigationProjection(
            frames=keys, current=None if current is None else self._result_key(current.data(role)),
            selected=tuple(self._result_key(item.data(role)) for item, _ in self._selected()))

    def _display_payload(self, item, payload, with_map=True):
        """A result as Int's frame (``StandardDisplayPayload``), through Int's conversion
        (``FrameView.from_results``): a Stitch 2D's map and the I(q) beside it, a Stitch 1D's I(q); the
        Diagnostic shows their coverage or normalization instead. A GI result is a GI frame, as Int's: the
        GI cut the plot axis chooses as its pattern, and a Stitch 2D's q_ip-q_oop map beside the (χ, q) one."""
        from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
        from xrd_tools.core.frame_view import FrameView
        from xdart.gui.tabs.scattering.display_values import StandardDisplayPayload
        key = self._result_key(item.data(QtCore.Qt.ItemDataRole.UserRole))
        shown = self._diagnostic.lower()

        def as_2d(projection):
            (_q, x), (_chi, y) = projection.axes
            (_q, x_unit), (_chi, y_unit) = projection.axis_units
            return IntegrationResult2D(x, y, getattr(projection, shown), None, x_unit, y_unit)
        map_result, cut = None, dict(payload.gi).get(self._view.plot_axis)
        if payload.kind is AnalysisArtifactKind.STITCH_2D:
            if with_map:  # Int 1D's layout shows a map's I(q) alone
                (_q, q), (_chi, chi) = payload.axes
                (_q, q_unit), (_chi, chi_unit) = payload.inspection.axis_units
                map_result = IntegrationResult2D(q, chi, getattr(payload, shown), None, q_unit, chi_unit)
            pattern = payload.profile if cut is None else cut
        else:
            pattern = payload if cut is None else cut
        if pattern is payload:
            radial, unit = payload.axis("q"), payload.native_unit
        else:
            (_q, radial), = pattern.axes
            (_q, unit), = pattern.axis_units
        view = FrameView.from_results(label=key.local_frame_label,
                                      result_1d=IntegrationResult1D(radial, getattr(pattern, shown), None, unit),
                                      result_2d=map_result)
        gi = {}
        if payload.gi_map is not None or payload.gi:
            gi = dict(measurement_mode="GI", gi_resolved_motor=_gi_motor(payload),
                      gi_mode_2d="" if map_result is None else "q_chi",
                      gi_mode_1d="q_total" if cut is None else self._view.plot_axis)
        if payload.gi_map is not None and map_result is not None:
            gi["extra_views_2d"] = {"qip_qoop": FrameView.from_results(
                label=key.local_frame_label, result_2d=as_2d(payload.gi_map))}
        return StandardDisplayPayload(0, key, item.text(), view, status="complete",
            wavelength_m=None if payload.wavelength_A is None else float(payload.wavelength_A) * 1e-10, **gi)

    def _draw_projection(self, processing_mode):
        """Int's projection of the selected results (``build_scientific_projection``) in Int 2D's layout
        (the map as the cake, I(q) below, its slices and pins as Int 2D makes them from the same map) or
        Int 1D's (each result's I(q), or the GI cut the plot axis chooses), with Int's axis rules: Q and 2θ
        from one axis and the wavelength, and the Q the shown results have when they lack the axis chosen
        (the choice stays, Codex terminal review F2)."""
        from xdart.gui.tabs.scattering.shell_projection import build_scientific_projection
        map_mode = processing_mode == "Int 2D"
        role = QtCore.Qt.ItemDataRole.UserRole
        keys, navigation = self._navigation()
        items = {id(self._result_key(item.data(role))): (item, payload) for item, payload in self._selected()}
        current = self.results.currentItem()
        if current is not None and current.data(role) in self._payloads:
            items.setdefault(id(self._result_key(current.data(role))),
                             (current, self._payloads[current.data(role)]))
        if map_mode:
            # a pinned result's map gives its pinned slice; Int's projection keeps it out of the selection
            # (Codex terminal review F1: the pin stayed while its trace vanished)
            pinned = {id(pin.frame) for pin in self._view.slice_pins}
            for row in range(self.results.count()):
                item = self.results.item(row)
                key = self._result_key(item.data(role))
                if id(key) in pinned and item.data(role) in self._payloads:
                    items.setdefault(id(key), (item, self._payloads[item.data(role)]))
        state = build_scientific_projection(
            tuple(self._display_payload(item, payload, with_map=map_mode) for item, payload in items.values()),
            navigation, frozenset(keys), self._view, "", processing_mode=processing_mode)
        names = {id(key): (key.source_scan, f"{key.source_scan}_{key.local_frame_label}") for key in keys}

        def named(trace):  # the result's name, as the list shows it, before any slice note
            name, prefix = names[id(trace.frame)]
            return replace(trace, title=name + trace.title[len(prefix):]) if trace.title.startswith(prefix) else trace
        cuts = tuple(mode for mode in _CUT_AXES
                     if any(mode in dict(payload.gi) for _item, payload in items.values()))
        state = replace(state, traces=tuple(named(trace) for trace in state.traces),
                        pinned_traces=tuple(replace(pinned, trace=named(pinned.trace)) for pinned in state.pinned_traces),
                        gi_modes_1d=("q_total", *cuts) if state.measurement_mode == "GI" else (),
                        title=current.text() if current is not None else "Current",
                        status=current.text() if current is not None else "")
        self._last_map_state = state if map_mode else None
        self.scientific.reconcile(state, navigation, completed=self.results.currentRow() + 1,
                                  total=len(keys), detail="")

    def _view_command(self, command):
        """The view's typed commands for the Stitch results it shows: the preference fields each changes are
        Int's reducer's (``scientific_preference_updates``: a pin lives in Overlay, a plot axis carries its
        pins, Share axis follows the map shown), with this page's own side effects."""
        from xdart.gui.tabs.scattering.scientific_axes import shown_slice_pin
        from xdart.gui.tabs.scattering.scientific_preferences import scientific_preference_updates
        from xdart.gui.tabs.scattering.shell_values import ShellCommandKind
        kind, view = command.kind, self._view
        role = QtCore.Qt.ItemDataRole.UserRole
        if kind in {ShellCommandKind.SELECT_FRAME, ShellCommandKind.HYDRATE_FRAME}:
            path = next((path for path, key in self._result_keys.items() if key is command.frame), None)
            row = next((row for row in range(self.results.count())
                        if self.results.item(row).data(role) == path), -1)
            if row >= 0 and row != self.results.currentRow():
                flag = (QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
                        if view.plot_mode == "Single" else QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
                self.results.setCurrentRow(row, flag)
            return
        if kind is ShellCommandKind.CLEAR_1D:
            self.clear_plot()
            return
        if kind is ShellCommandKind.PIN_SLICE:  # the slice shown, on the current result's map
            pin = shown_slice_pin(self._last_map_state)
            current = self.results.currentItem()
            if (pin is None or current is None or pin.frame is not self._result_key(current.data(role))
                    or pin.projection_id in {other.projection_id for other in view.slice_pins}):
                return
            self._set_view(slice_pins=(*view.slice_pins, pin))
            self._draw()
            return
        updates = scientific_preference_updates(view, command, shown_image_axis=(
            None if self._last_map_state is None else self._last_map_state.image_axis))
        if updates is None:
            return
        self._set_view(**updates)
        if kind is ShellCommandKind.SET_PLOT_MODE:
            if command.value == "Single":
                self.results.setCurrentRow(self.results.currentRow(),
                                           QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect)
            self._update_residency()  # Overlay shows the selected results, Single the current one
        elif kind in {ShellCommandKind.SET_COLOR_MAP, ShellCommandKind.SET_LOG_SCALE}:
            self._render_raw()  # the raw popup's image too
        self._draw()

    def _install_diagnostic_option(self):
        """Stitch's Diagnostic (Intensity, Coverage or Normalization) as a row of the view's
        own plot-options dialog."""
        dialog = self.scientific.plot_options_dialog
        self._diagnostic_combo = QtWidgets.QComboBox()
        self._diagnostic_combo.addItems(("Intensity", "Coverage", "Normalization"))
        dialog.add_row("Diagnostic", self._diagnostic_combo)
        # read before the view's own option commands redraw (and so reset) the row
        self.scientific.plotOptionsAccepted.connect(self._diagnostic_accepted)
        dialog.rejected.connect(lambda: self._diagnostic_combo.setCurrentText(self._diagnostic))

    def _diagnostic_accepted(self):
        if self._diagnostic_combo.currentText() != self._diagnostic:
            self._diagnostic = self._diagnostic_combo.currentText()
            self._draw()

    def clear_plot(self):
        self._view_revision += 1
        self._set_view(slice_pins=())
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
        self.run_bar.set_mode_row_enabled(idle)  # as Int's: the mode and Append/Replace hold during a run
        for widget in (self.run_bar.coresSpin, self.run_bar.coresLabel):
            widget.setToolTip(_CORES_TIP)
        for dialog in self._authoring_dialogs:
            dialog.setEnabled(not busy and not self._closing and (not authoring or dialog.active()))
        if not any(dialog.isVisible() or dialog.active() for dialog in self._authoring_dialogs):
            self._authoring_timer.stop()
        for row in self._rows.values():
            row.setEnabled(idle)
        if self._value("backend") == "xrayutilities histogram":
            locked = ("geometry_kind", "source_motors", "poni_references", "camera")
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
        self.run_bar.startButton.setEnabled(idle and self._readiness[1])  # Run prepares the inputs itself
        action = self._owner.current_identity.request.action if busy else None
        self.run_bar.set_stop_enabled(action is StitchOwnerAction.RUN or (
            action is StitchOwnerAction.PREFLIGHT and self._after_prepare == "run"))
        self.retry_button.setVisible(final is not StitchOwnerFinalization.NONE)
        self.retry_button.setEnabled(not busy)
        if idle and self._reload_pending and self._reloading is None and not self._queued_jobs:
            QtCore.QTimer.singleShot(0, self._request_reload)

    def _authoring_active(self):
        return any(dialog.active() for dialog in self._authoring_dialogs)

    def active(self):
        return self._owner.busy or self._owner.finalization is not StitchOwnerFinalization.NONE or self._closing or self._authoring_active()

    def shutdown(self):
        self._closing = True
        self._queued_jobs.clear()
        self._reload_pending.clear()
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
            self._payload_bytes.clear()
            self._recent.clear()
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
        self._status(f"Stitch settings loaded: {path.name}", ready=self._readiness[1])
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
        return AppMenuHosts(self.config_menu, self.help_menu, tool_requested=self.toolRequested,
                            mode_requested=self.modeRequested)

    def request(self):
        self.open_result()
        return ActionAccepted("open-stitch-result")

    def run_pause(self):
        self.begin_run()
        return ActionAccepted("stitch-run")


class _StitchProcessingMode:
    """Another page's request for Stitch 1D or 2D (Int's run bar), through the page's own mode choice;
    completed only when the page took it."""

    def __init__(self, page):
        self._page = page

    def select(self, mode):
        from xdart.gui.pages.values import MODE_REFUSED
        if self._page.select_mode(str(mode)):
            return ActionCompleted(f"processing-mode-{mode}")
        return ActionRefused(MODE_REFUSED)


def build_stitch_page(services, parent):
    page = StitchPage(services, parent)
    page.restore_last_settings()
    return PageHandle(key=STITCH_TOOL_KEY, widget=page, close=page.close_and_remember,
                      open_folder=page, settings_io=page, run_control=page, activity=page,
                      app_menus=page, processing_mode=_StitchProcessingMode(page))
