# -*- coding: utf-8 -*-
"""
@author: thampy, walroth
"""

# Standard library imports
import logging
import os
import fnmatch
import time
import numpy as np
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Qt imports
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.parametertree import ParameterTree, Parameter

# Project imports
from xrd_tools.core.containers import PONI
from xrd_tools.io.metadata import read_image_metadata
from xrd_tools.session.readiness import (
    append_config_difference_lines,
    append_config_mismatch_check,
    processing_config_from_mapping,
    processing_config_from_scan,
)
# ``_extract_scan_info`` is a private helper but the wrangler uses it
# to predict whether the SPEC sidecar exists for a given image file —
# same parser the SSRL reader uses, so the UI's existence check stays
# in sync with what ``read_image_metadata`` will actually look for.
from xrd_tools.io.metadata import _extract_scan_info
from xrd_tools.session.run_configuration import RunConfigurationRefused
from .wrangler_widget import (
    GIMotorHydration,
    _run_owner_refusal_status,
    wranglerWidget,
)
from .image_wrangler_thread import (  # noqa: F401
    GISourceMotorDiscovery,
    imageThread,
    _get_scan_info,
)
from .ui.specUI import Ui_Form
from xdart.modules.live import LiveScan
from xdart.utils import get_fname_dir, match_img_detector
from xdart.utils.browse import browse_start_dir, remember_browse_path
from xdart.utils.session import load_session, save_session
from ..run_config_debug import (
    parameter_change_summary,
    run_config_debug_enabled,
    run_config_debug_log,
)


QFileDialog = QtWidgets.QFileDialog
QDialog = QtWidgets.QDialog
QMessageBox = QtWidgets.QMessageBox
QPushButton = QtWidgets.QPushButton


def_poni_file = ''
def_img_file = ''
def_meta_ext = 'auto'


def _normalize_meta_ext(value):
    """Return the internal metadata mode; ``None`` means GUI metadata off."""
    if value is None:
        return None
    text = str(value).strip()
    if text == '' or text.lower() == 'none':
        return None
    lowered = text.lower()
    if lowered in ('auto', 'txt', 'pdi', 'metadata', 'spec'):
        return lowered
    return text


def _meta_ext_parameter_value(value):
    """Return a value accepted by the Meta File dropdown."""
    normalized = _normalize_meta_ext(value)
    return 'none' if normalized is None else normalized


def _path_exists_case_insensitive(path):
    candidate = Path(path)
    if candidate.exists():
        return True
    target = candidate.name.lower()
    try:
        return any(
            sibling.name.lower() == target and sibling.is_file()
            for sibling in candidate.parent.iterdir()
        )
    except OSError:
        return False

params = [
    # N1: the portable Project Folder.  Setting it stamps entry/@source_base and
    # makes each frame's raw source path RELATIVE to it, so the processed .nxs
    # resolves its raw images after the data moves machines.  Blank -> absolute
    # paths (back-compat).  (The full progressive-disclosure / folder-change
    # reset UX is a follow-up; the portable storage is active once a folder is
    # set.)
    # Direction-A Stage 3: group `title`s are the card-header band text
    # (uppercase per spec).  Only the display `title` changes — every group
    # `name` (Project/Calibration/Signal/BG) stays put, so .child() paths,
    # session save/load, and the disclosure/toggle logic are untouched.
    {'name': 'Project', 'title': 'PROJECT', 'type': 'group', 'children': [
        {'name': 'project_folder', 'title': 'Folder', 'type': 'str_browse', 'value': ''},
        {'name': 'h5_dir', 'title': 'Save Path', 'type': 'str_browse',
         'value': '', 'enabled': True},
    ], 'expanded': True},
    # PONI is the first row of DATA — the standalone CALIBRATION group is gone.
    # This collapses the N1 progressive disclosure to two stages (Folder -> DATA),
    # since the PONI picker now lives inside DATA (see _apply_disclosure).  A
    # proper redo belongs in the Tier-B custom-card wrangler migration.
    {'name': 'Signal', 'title': 'DATA', 'type': 'group', 'children': [
        {'name': 'poni_file', 'title': 'Poni', 'type': 'str_browse', 'value': def_poni_file},
        {'name': 'inp_type', 'title': 'Source', 'type': 'list',
         'values': ['Image Series', 'Image Directory', 'Single Image'], 'value': 'Image Series'},
        {'name': 'File', 'title': 'Image File   ', 'type': 'str_browse', 'value': def_img_file},
        {'name': 'img_dir', 'title': 'Directory', 'type': 'str_browse', 'value': '', 'visible': False},
        {'name': 'include_subdir', 'title': 'Subdirectories', 'type': 'bool', 'value': False, 'visible': False},
        # O-1a-W1R (review §39.4, ".hdf5 selector reachability"): the backend,
        # the single-file browse filter, the container emitter branch and the
        # directory contracts all support ``.hdf5``, but the operator could not
        # SELECT it here, so the supported branch was unreachable outside a
        # test-side limits override.  ``hdf5`` sits next to ``h5`` because it is
        # the same container format with the long spelling.
        {'name': 'img_ext', 'title': 'File Type  ', 'type': 'list',
         'values': ['tif', 'raw', 'h5', 'hdf5', 'nxs', 'mar3450'],
         'value': 'tif', 'visible': False},
        {'name': 'series_average', 'title': 'Average Scan', 'type': 'bool', 'value': False, 'visible': True},
        {'name': 'meta_ext', 'title': 'Meta File', 'type': 'list',
         'values': ['auto', 'none', 'txt', 'pdi', 'metadata', 'spec'],
         'value': def_meta_ext},
        # Optional override for the SPEC file's directory.  Shown only
        # when ``meta_ext == 'spec'`` (see ``set_meta_ext``).  Blank
        # → use the default search (image dir + immediate parent);
        # set → that directory is searched first.
        {'name': 'meta_dir', 'title': 'Meta Directory', 'type': 'str_browse',
         'value': '', 'visible': False},
        {'name': 'Filter', 'type': 'str', 'value': '', 'visible': False,
         'tip': "Filename filter: space-separated terms = AND (any order), '|' or OR = either, leading -term or NOT = exclude.  Case-insensitive substrings; empty = all files."},
        {'name': 'mask_file', 'title': 'Mask File', 'type': 'str_browse', 'value': ''},
        # (Write Mode moved to the Controls run bar — it's a run/output property,
        # not a data input.  setup() reads it from the shared StaticControls.)
    ], 'expanded': True, 'visible': False},
    {'name': 'GI', 'title': 'Grazing Incidence', 'type': 'group',
     'children': [
        # UI-1 (#81): the group HEADER carries a real checkbox — the on/off
        # toggle (see wranglerWidget._install_group_toggles).  The bool is
        # the hidden source of truth the rest of the code reads -- a hidden
        # bool can't repaint-uncheck when the tree is disabled (#56).
        {'name': 'Grazing', 'type': 'bool', 'value': False, 'visible': False},
        {'name': 'th_motor', 'title': 'Theta Motor', 'type': 'list',
         'values': ['th', 'Manual'], 'value': 'th'},
        {'name': 'th_val', 'title': 'Theta', 'type': 'str', 'value': '0.1', 'visible': False},
        {'name': 'sample_orientation', 'title': 'Sample Orientation', 'type': 'int', 'value': 4,
         'limits': (1, 8), 'step': 1,
         'tip': 'EXIF sample orientation (1-8) for pyFAI FiberIntegrator'},
        {'name': 'tilt_angle', 'title': 'Tilt Angle', 'type': 'float', 'value': 0.0,
         'step': 0.1, 'tip': 'Chi offset / tilt angle in degrees for FiberIntegrator'},
        # gi_mode_1d / gi_mode_2d are controlled by the integrator panel
        # (axis1D / axis2D combos).  Kept here for session persistence only.
        {'name': 'gi_mode_1d', 'title': '1D Mode', 'type': 'list',
         'values': {'Q': 'q_total', u'Q\u1D62\u209A': 'q_ip', u'Q\u2092\u2092\u209A': 'q_oop'},
         'value': 'q_total', 'visible': False},
        {'name': 'gi_mode_2d', 'title': '2D Mode', 'type': 'list',
         'values': {u'Q\u1D62\u209A\u2013Q\u2092\u2092\u209A': 'qip_qoop', u'Q-\u03C7': 'q_chi'},
         'value': 'qip_qoop', 'visible': False},
    ], 'expanded': False, 'visible': False},
    {'name': 'Mask', 'title': 'Intensity Threshold', 'type': 'group',
     'children': [
        # UI-1 (#81): header checkbox is the on/off toggle; bool hidden (see GI).
        {'name': 'Threshold', 'type': 'bool', 'value': False, 'visible': False},
        # float (was int) so the integrator's text Min/Max round-trip exactly
        # into this hidden carrier (no int coercion → live == reintegrate).
        {'name': 'min', 'title': 'Min', 'type': 'float', 'value': 0},
        {'name': 'max', 'title': 'Max', 'type': 'float', 'value': 0},
    ], 'expanded': False, 'visible': False},
    # Auto-mask the uint16 detector ceiling (65535) as a saturated/dead
    # sentinel.  Its OWN UI-1 header-checkbox group (a checkbox on the left,
    # like Intensity Threshold) — independent of the Min/Max band.  ON by
    # default preserves the long-standing behaviour; OFF lets a genuinely-
    # saturated Bragg peak at the ceiling be seen + integrated.  Non-finite +
    # the uint32 ceiling stay always masked (unambiguous invalids).  The bool
    # is hidden; the group header checkbox is the toggle (see
    # wranglerWidget._install_group_toggles).
    {'name': 'MaskSat', 'title': 'Mask Saturated', 'type': 'group',
     'children': [
        {'name': 'mask_sentinel', 'type': 'bool', 'value': True,
         'visible': False},
    ], 'expanded': False, 'visible': False},
    {'name': 'BG', 'title': 'BACKGROUND', 'type': 'group', 'children': [
        {'name': 'bg_type', 'title': '', 'type': 'list',
         'values': ['None', 'Single BG File', 'Series Average', 'BG Directory'], 'value': 'None'},
        {'name': 'File', 'title': 'BG File', 'type': 'str_browse', 'value': '', 'visible': False},
        {'name': 'Match', 'title': 'Match Parameter', 'type': 'group', 'children': [
            {'name': 'Parameter', 'type': 'list', 'values': ['None'], 'value': 'None'},
            {'name': 'match_fname', 'title': 'Match File Root', 'type': 'bool', 'value': False},
            {'name': 'bg_dir', 'title': 'Directory', 'type': 'str_browse', 'value': ''},
            {'name': 'Filter', 'type': 'str', 'value': '',
             'tip': "Filename filter: space-separated terms = AND (any order), '|' or OR = either, leading -term or NOT = exclude.  Case-insensitive substrings; empty = all files."},
        ], 'expanded': True, 'visible': False},
        {'name': 'Scale', 'type': 'float', 'value': 1, 'visible': False},
        {'name': 'norm_channel', 'title': 'Normalize', 'type': 'list', 'values': ['bstop'], 'value': 'bstop',
         'visible': False},
    ], 'expanded': True, 'visible': False},   # expanded so the BG dropdown shows (was collapsed)
    # (h5_dir / Save Path moved into the PROJECT group above.)
]

ctr = 1


# The GI incidence-motor SHOW-all + default-select policy is shared with the
# integrator combo so the wrangler's th_motor param and the integrator's
# gi_motor combo never disagree (see gi_motor_defaults.pick_default_gi_motor).
from ..gi_motor_defaults import pick_default_gi_motor

# HDF5 container extensions that may embed their own scan metadata (Bluesky/
# NXWriter).  Only these are probed for embedded motors/counters; every other
# source (TIFF/EDF/CBF/…) skips the probe entirely, so the sidecar path — and
# the light-holder tests that bind only the metadata methods — are untouched.
_EMBEDDED_META_EXTS = ('.nxs', '.h5', '.hdf5')

# O-1b R4A-1: the metadata-preview budget.  BOTH bounds cover the COMPLETE
# preview attempt (direct children AND any Subdirs descent), and they are
# independently load-bearing: whichever trips first stops the attempt.  These
# are the retired probe-budget constants from the pre-R4-A recursive seed.
_PREVIEW_MAX_CONTENT_OPENS = 8
_PREVIEW_DEADLINE_S = 0.75
# R4A-III.1 (review §54.3 item 3): a preview-only CONTAINMENT ceiling on how
# many directory entries may reach the final natural-key sort.  Ordering has no
# cooperative point inside it, so rather than leave one opaque unbounded phase,
# a directory wider than this is simply not previewed -- UNKNOWN/Manual, never a
# truncated candidate set.  This is name-only containment, NOT a content open,
# and it is set far above any real acquisition directory (the 529-file
# beamline benchmark is two orders of magnitude below it) so it is a runaway
# guard rather than a limit operators can reach.
_PREVIEW_MAX_PREVIEW_ENTRIES = 20000


class DirectoryMetadataPreview(NamedTuple):
    """THE immutable, detached result of inspecting ONE candidate container.

    O-1b R4A-2 (review §49.3 item 1, §50.1).  Candidate discovery used to pick
    ``natural_sort_ints(candidates)[0]`` while adoption independently decided
    usability, so discovery could select a candidate adoption then rejected --
    and the whole GI/BG option set was cleared with a usable sibling right
    there.  There is now ONE usability operation
    (:meth:`imageWrangler._metadata_preview_result`) returning either this value
    or ``None``; discovery and adoption consume the SAME result, so a candidate
    is never content-opened twice and the two can no longer drift apart.

    ``motors``/``counters`` are detached name tuples: nothing here aliases a
    live HDF5 handle, a provider, or a mutable panel list.
    """

    path: str
    kind: str                 # 'embedded' (Bluesky columns) | 'sidecar'
    motors: tuple
    counters: tuple


class _PreviewBudget:
    """The WHOLE-attempt content-open + wall-clock bound for one preview.

    Both bounds must be independently discriminating (review §50.2): enforcing
    eight opens without a deadline, or a deadline without the open cap, each
    passes one frozen row and fails the other.  The clock is read through the
    module-level ``time`` import so the oracle's injected ``monotonic`` is
    load-bearing -- a deadline captured from ``time.time``/``perf_counter``, or
    from a clock imported into another module, would look correct and never
    trip.
    """

    __slots__ = ("deadline", "opens")

    def __init__(self, *, deadline, opens):
        self.deadline = float(deadline)
        self.opens = int(opens)

    def deadline_passed(self):
        """The TIME bound alone (review §52.3 item 3).

        Name enumeration is charged against the clock but NOT against the
        content-open cap, so the two bounds stay independently discriminating:
        a delay in ``scandir`` iteration with no candidate opens must trip this
        and nothing else.
        """
        return time.monotonic() >= self.deadline

    def exhausted(self):
        return self.opens <= 0 or self.deadline_passed()

    def charge(self):
        """Authorize ONE content open, or refuse because a bound tripped."""
        if self.exhausted():
            return False
        self.opens -= 1
        return True


def _directory_preview_entries(directory, budget):
    """``(files, subdirectories)`` in GLOBAL natural order, or ``None``.

    Name-only: ``os.scandir`` entries are classified with the cached
    ``is_file``/``is_dir`` bits and never opened, so listing a candidate root
    stays inside the lazy-discovery contract.

    O-1b R4A-III / R4A-III.1 (review §52.2, §54.2).  The ONE shared deadline
    covers every phase of the attempt that can block, because each of them was
    the same user-visible high-level-directory freeze wearing a different name:

    * **enumeration** — the deadline is charged while the scandir iterator is
      consumed, before anything is materialized.  Draining a wide accidental
      root into a dict first spent arbitrarily long with ZERO content opens, so
      the eight-open cap never fired;
    * **classification** — ``is_file``/``is_dir`` are checked immediately before
      AND after each call.  They are NOT guaranteed cached value reads: on NFS,
      and on any filesystem whose directory entries carry no usable ``d_type``,
      each is a stat-like call whose latency is outside Python's control, so a
      listing could complete just under the deadline and then block for O(N)
      more; and
    * **ordering** — a natural-key sort has no cooperative point inside it, so
      instead of reintroducing one opaque unbounded phase, the number of entries
      that may reach the sort is capped by ``_PREVIEW_MAX_PREVIEW_ENTRIES``.

    Every failure mode produces the SAME truthful answer, ``None``: a partial
    listing is not a globally ordered candidate set, and inspecting its head
    would silently reintroduce the ``candidates[0]``-order bug R4A-2 exists to
    fix.  The caller abandons preview and the admitted worker remains the later
    authoritative discovery owner.

    When the preview does complete, ordering is exactly as before: one global
    natural sort, so ``scan_2`` precedes ``scan_10`` however wide the directory.
    """
    from .image_wrangler_thread import natural_keys_int

    try:
        with os.scandir(directory) as entries:
            retained = []
            for entry in entries:
                if budget.deadline_passed():
                    return None
                if len(retained) >= _PREVIEW_MAX_PREVIEW_ENTRIES:
                    # Containment, not an open: past this the final ordering
                    # step is no longer conservatively bounded.  Abandon rather
                    # than order a truncated set.
                    return None
                retained.append(entry)
    except OSError:
        return (), ()
    # Natural-key preparation + sort: bounded by the ceiling above, and still
    # charged, so a large in-budget listing cannot spend the whole attempt here.
    if budget.deadline_passed():
        return None
    ordered = sorted(retained, key=lambda entry: natural_keys_int(entry.name))
    files, subdirs = [], []
    for entry in ordered:
        if budget.deadline_passed():
            return None
        try:
            is_file = entry.is_file(follow_symlinks=False)
            is_dir = False if is_file else entry.is_dir(follow_symlinks=False)
        except OSError:
            # An entry that vanished mid-listing is skipped, but the call still
            # consumed real time, so the bound is re-checked before continuing.
            if budget.deadline_passed():
                return None
            continue
        if budget.deadline_passed():
            return None
        if is_file:
            files.append(Path(entry.path))
        elif is_dir:
            subdirs.append(Path(entry.path))
    return tuple(files), tuple(subdirs)


class imageWrangler(wranglerWidget):
    """Widget for integrating detector image files (TIFF/EDF/CBF/Eiger
    master; as an image series, image directory, or single image).
    Per-frame metadata is read from an optional sidecar — a SPEC file,
    a ``.txt``/``.pdi`` file, or nothing (the ``Meta Format`` option).
    Can be used "live": it polls the data folder for new images (and
    their metadata, if any) until the scan is complete.

    attributes:
        command_queue: Queue, used to send commands to thread
        file_lock, mp.Condition, process safe lock for file access
        fname: str, path to data file
        parameters: pyqtgraph Parameter, stores parameters from user
        scan_name: str, current scan name, used to handle syncing data
        scan_args: dict, used as **kwargs in scan initialization.
            see LiveScan.
        thread: wranglerThread or subclass, QThread for controlling
            processes
        tree: pyqtgraph ParameterTree, stores and organizes parameters
        ui: Ui_Form from qtdesigner

    methods:
        stop: function to pass stop command to thread via command_queue
        enabled: Enables or disables interactivity
        set_image_dir: sets the image directory
        set_poni_file: sets the calibration poni file
        set_spec_file: sets the spec data file
        set_fname: Method to safely change file name
        setup: Syncs thread parameters prior to starting

    signals:
        finished: Connected to thread.finished signal
        sigStart: Tells tthetaWidget to start the thread and prepare
            for new data.
        sigUpdateData: int, signals a new frame has been added.
        sigUpdateFile: (str, str, bool, str, bool, bool), sends new scan_name, file name
            GI flag (grazing incidence), theta motor for GI, single_image and
            series_average flag to static_scan_Widget.
        sigUpdateGI: bool, signals the grazing incidence condition has changed.
        showLabel: str, connected to thread showLabel signal, sets text
            in specLabel
    """
    showLabel = QtCore.Signal(str)
    sigSavePathChanged = QtCore.Signal(str)

    def __init__(self, fname, file_lock, scan, parent=None):
        """fname: str, file path
        file_lock: mp.Condition, process safe lock
        """
        super().__init__(fname, file_lock, parent)

        # Scan Parameters
        self.poni = None
        self.scan_parameters = []
        self.counters = []
        self.motors = []
        self.command = None
        self.scan = scan
        self.run_configuration = None
        self._admitted_run_configuration = None
        # O-1a-W1R: frozen values this wrangler's worker must never take from a
        # mutable mirror.  Admission refuses a configuration missing any of them,
        # which makes the projection's absent-value leg unreachable for a run.
        # O-1a-W1R-D1 (review §40.1 P1-A, §40.3 D1 item 4): an image run needs an
        # output target AND a typed source.  Two real GUI paths admitted
        # ``source is None`` -- an N1 project switch and an operator clearing the
        # editable Image File field -- and handed 58 worker reads back to writable
        # state.  Legitimate seedless typed Directory runs are unaffected: they
        # freeze a real DirectorySourceSpec.
        self._admission_required_frozen_values = ("save_path", "source")
        # O-1a-W1R: the loaded-scan calibration CANDIDATE for the click in
        # progress.  The freeze owner reads it; only exact admission publishes it.
        self._staged_run_calibration = None
        # H19: immutable Source-card baseline + serialized headless index owner.
        # Populated immediately before setup() for container-directory runs.
        self.source_run_plan = None
        self.source_spec = None
        self.source_index_session = None
        self.source_frame_count_snapshot = {}
        self.source_pending_count = 0

        # Setup gui elements
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        # Deferred for this release: the NeXus Viewer mode is not yet usable on
        # stacked datasets (integrated_1d/2d store the whole (N, ...) stack
        # under one key, so it needs a per-frame slider).  Hide the option to
        # avoid the half-baked viewer; the controller/mode code stays dormant
        # and the option returns once the slider lands.
        _nx_idx = self.ui.processingModeCombo.findText('NeXus Viewer')
        if _nx_idx >= 0:
            self.ui.processingModeCombo.removeItem(_nx_idx)
        # Run-control signal wiring moved to attach_controls() — the controls are
        # the shared StaticControls widget now, connected to this wrangler's
        # handlers on wrangler swap (and disconnected on the previous swap).
        self._on_mode_changed()
        self._set_wrangler_tooltips()

        # Long thread messages (e.g. the live-GI clip advisory) must not
        # force the window wider — elide into the label, full text in the
        # tooltip (see wranglerWidget._guard_status_label).
        self._guard_status_label()
        self.showLabel.connect(self._set_status_text)

        # Setup parameter tree
        self.tree = ParameterTree()
        # Stage 3a: object-named so the global QSS themes it (card-band group
        # headers + field tint) and live-switches Dark/Light, replacing the old
        # widget-local Dracula stylesheet (see stylize_ParameterTree + the
        # QTreeView#WranglerTree rules in themes/dark.py).
        self.tree.setObjectName('WranglerTree')
        # This (and the name-column width below) is the dominant floor on
        # how narrow the right panel drags — the param tree is the widest
        # fixed content.  Keep it small so the panel can shrink.
        self.tree.setMinimumWidth(80)
        self.stylize_ParameterTree()
        self.parameters = Parameter.create(
            name='image_wrangler', type='group', children=params
        )
        self.tree.setParameters(self.parameters, showTop=False)
        # Squeeze parameter tree columns to reduce panel width
        header = self.tree.header()
        header.setStretchLastSection(True)
        # Stage 3a: wider name column (was nominally 79) so card-row labels like
        # "Average Scan" / "Write Mode" / "Calibration" never clip, and a
        # *consistent* label width per the spec.  ParameterTree defaults column 0
        # to ResizeToContents (which ignored the old resizeSection AND marginally
        # clipped "Average Scan" at ~98px); switch it to Interactive so the fixed
        # 120 actually applies.
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Interactive)
        # Root decoration (below) adds a small per-row gutter, so the name column
        # needs a touch more room or labels like "Average Scan" clip.
        header.resizeSection(0, 132)
        header.setMinimumSectionSize(40)
        # The group expand/collapse chevrons live in the indentation column; the
        # old shallow indent (3px) squeezed them to near-invisibility.  Root
        # decoration makes the top-level group rows draw a branch arrow the theme
        # replaces with a clear, high-contrast triangle (themes/dark:
        # QTreeView#WranglerTree::branch).  Keep the indent small (a touch above
        # the old 3) so the field labels are not pushed far right or clipped —
        # just enough for the chevron to read.
        self.tree.setIndentation(9)
        self.tree.setRootIsDecorated(True)
        # Hide the "Parameter / Value" header bar — it's just visual noise above
        # the wrangler tree.  Column sizing above still applies (the header is
        # hidden, not removed).
        self.tree.setHeaderHidden(True)
        self.layout = QtWidgets.QVBoxLayout(self.ui.paramFrame)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.tree)

        # Set attributes from Parameter Tree and a couple more
        # N1: Project Folder (portable @source_base).  Blank -> None (absolute).
        # _restoring gates the folder-change reset off during session restore.
        self._restoring = False
        self.project_folder = self.parameters.child('Project').child('project_folder').value()
        self.source_base = self._compute_source_base()
        # Calibration
        self.poni_file = self.parameters.child('Signal').child('poni_file').value()
        # Applies the N1 progressive disclosure for the fresh-start state (no
        # folder -> only Project visible).
        self.get_poni_dict()

        # Signal
        self.inp_type = self.parameters.child('Signal').child('inp_type').value()
        self.img_file = self.parameters.child('Signal').child('File').value()
        self.img_dir = self.parameters.child('Signal').child('img_dir').value()
        self.include_subdir = self.parameters.child('Signal').child('include_subdir').value()
        self.img_ext = self.parameters.child('Signal').child('img_ext').value()
        self._img_dir_probe_cache = (None, None, 0.0)
        # Cache for the Bluesky/NXWriter embedded-metadata probe (see
        # _read_bluesky_source_columns): (key, result), key=(path, mtime, size).
        self._bluesky_cols_cache = None
        self.single_img = True if self.inp_type == 'Single Image' else False
        self.file_filter = self.parameters.child('Signal').child('Filter').value()
        self.series_average = self.parameters.child('Signal').child('series_average').value()
        self.meta_ext = _normalize_meta_ext(
            self.parameters.child('Signal').child('meta_ext').value()
        )
        # Optional explicit dir for SPEC files; '' falls back to the
        # xrd_tools default (image dir + parent search).  Wired
        # to the thread so workers pass it to read_image_metadata.
        self.meta_dir = self.parameters.child('Signal').child('meta_dir').value()

        # Mask
        self.mask_file = self.parameters.child('Signal').child('mask_file').value()

        # Threshold
        self.apply_threshold = self.parameters.child('Mask').child('Threshold').value()
        self.threshold_min = self.parameters.child('Mask').child('min').value()
        self.threshold_max = self.parameters.child('Mask').child('max').value()
        self.mask_sentinel = self.parameters.child('MaskSat').child('mask_sentinel').value()

        # Write Mode
        self.write_mode = self._active_write_mode()

        # Background
        self.bg_type = self.parameters.child('BG').child('bg_type').value()
        self.bg_file = self.parameters.child('BG').child('File').value()
        self.bg_dir = self.parameters.child('BG').child('Match').child('bg_dir').value()
        self.bg_matching_par = self.parameters.child('BG').child('Match').child('Parameter').value()
        self.bg_match_fname = self.parameters.child('BG').child('Match').child('match_fname').value()
        self.bg_file_filter = self.parameters.child('BG').child('Match').child('Filter').value()
        self.bg_scale = self.parameters.child('BG').child('Scale').value()
        self.bg_norm_channel = self.parameters.child('BG').child('norm_channel').value()

        # Grazing Incidence
        self.gi = self.parameters.child('GI').child('Grazing').value()
        self.incidence_motor = self.parameters.child('GI').child('th_motor').value()
        self.sample_orientation = self.parameters.child('GI').child('sample_orientation').value()
        self.tilt_angle = self.parameters.child('GI').child('tilt_angle').value()
        # gi_mode_1d / gi_mode_2d are driven by the integrator panel;
        # default here, actual values set from scan.bai_*_args at thread start.
        self.gi_mode_1d = self.scan.bai_1d_args.get('gi_mode_1d', 'q_total')
        self.gi_mode_2d = self.scan.bai_2d_args.get('gi_mode_2d', 'qip_qoop')

        # HDF5 Save Path
        self.h5_dir = self.parameters.child('Project').child('h5_dir').value()

        # NOTE: Integration Advanced button (self.ui.advancedButton) is wired
        # in static_scan_widget.set_wrangler() to show the integratorTree's
        # existing advancedWidget1D / advancedWidget2D dialogs directly.

        # Wire signals from parameter tree based buttons.
        # setup() re-syncs params to the thread AND re-applies the N1 progressive
        # disclosure, which setOpts(visible=...) on the param GROUPS.  Those
        # setOpts calls fire sigTreeStateChanged with change type 'options'; if
        # setup ran on them it re-entered setup -> re-disclosed -> 'options' -> ...
        # an unbounded loop (froze the panel when a click/scroll in the poni/mask
        # field kicked it off; sigUpdateGI -> V2 refresh rode along as spam).  Run
        # setup ONLY on real 'value' edits, never on the disclosure's own
        # 'options'/'limits' churn.  (Direct setup() calls at run start are
        # unaffected.)
        self.parameters.sigTreeStateChanged.connect(self._setup_on_value_change)
        self.parameters.sigTreeStateChanged.connect(self._save_to_session)

        # UI-1 (#81): put a real checkbox on the GI / Intensity-Threshold
        # group headers — the checkbox is the on/off toggle, driving the
        # hidden enabling bool (see wranglerWidget._install_group_toggles).
        self._install_group_toggles(self.tree)

        self.parameters.child('Project').child('project_folder').sigActivated.connect(
            self.set_project_folder
        )
        # N1 Decision 2: a folder change (browse OR direct edit) resets the
        # dependent paths.  Guarded by _restoring so a session restore doesn't
        # trip it (see _restore_from_session).
        self.parameters.child('Project').child('project_folder').sigValueChanged.connect(
            self._on_project_folder_changed
        )
        self.parameters.child('Signal').child('poni_file').sigActivated.connect(
            self.set_poni_file
        )
        self.parameters.child('Signal').child('poni_file').sigValueChanged.connect(
            self.get_poni_dict
        )
        self.parameters.child('Signal').child('inp_type').sigValueChanged.connect(
            self.set_inp_type
        )
        self.parameters.child('Signal').child('File').sigActivated.connect(
            self.set_img_file
        )
        self.parameters.child('Signal').child('img_dir').sigActivated.connect(
            self.set_img_dir
        )
        self.parameters.child('Signal').child('mask_file').sigActivated.connect(
            self.set_mask_file
        )
        self.parameters.child('Signal').child('series_average').sigValueChanged.connect(
            self.set_series_average
        )
        self.parameters.child('Signal').child('meta_ext').sigValueChanged.connect(
            self.set_meta_ext
        )
        self.parameters.child('Signal').child('meta_dir').sigActivated.connect(
            self.set_meta_dir
        )
        self.parameters.child('BG').child('bg_type').sigValueChanged.connect(
            self.set_bg_type
        )
        self.parameters.child('BG').child('File').sigActivated.connect(
            self.set_bg_file
        )
        self.parameters.child('BG').child('Match').child('bg_dir').sigActivated.connect(
            self.set_bg_dir
        )
        self.parameters.child('BG').child('Match').child('Parameter').sigValueChanged.connect(
            self.set_bg_matching_par
        )
        self.parameters.child('BG').child('norm_channel').sigValueChanged.connect(
            self.set_bg_norm_channel
        )
        self.parameters.child('GI').child('th_motor').sigValueChanged.connect(
            self.set_gi_th_motor
        )
        self.parameters.child('GI').child('th_val').sigValueChanged.connect(
            self.set_gi_th_motor
        )
        self.parameters.child('Project').child('h5_dir').sigActivated.connect(
            self.set_h5_dir
        )

        # Setup thread
        self.thread = imageThread(
            self.command_queue,
            self.scan_args,
            self.file_lock,
            self.fname,
            self.h5_dir,
            self.scan_name,
            self.single_img,
            self.poni,
            self.inp_type,
            self.img_file,
            self.img_dir,
            self.include_subdir,
            self.img_ext,
            self.series_average,
            self.meta_ext,
            self.file_filter,
            self.mask_file,
            self.write_mode,
            self.bg_type,
            self.bg_file,
            self.bg_dir,
            self.bg_matching_par,
            self.bg_match_fname,
            self.bg_file_filter,
            self.bg_scale,
            self.bg_norm_channel,
            self.gi,
            self.incidence_motor,
            self.sample_orientation,
            self.tilt_angle,
            self.gi_mode_1d,
            self.gi_mode_2d,
            self.command,
            self.scan,
            live_mode=self.live_mode,
            max_cores=self.ui.maxCoresSpinBox.value(),
            parent=self,
        )

        self.thread.showLabel.connect(self._set_status_text)
        # Mid-run Append config mismatch: the worker stopped the run cleanly
        # (target preserved) — surface the one-modal warning here, because the
        # Run-click mismatch modal (_confirm_append_config_replace) only fires
        # at Run-click and a MID-RUN settings change bypasses it.
        self.thread.sigAppendMismatch.connect(self._on_append_mismatch)
        self.thread.sigUpdateFile.connect(self.sigUpdateFile.emit)
        self.thread.finished.connect(self.finished.emit)
        self.thread.sigUpdate.connect(self.sigUpdateData.emit)
        # self.thread.sigUpdateFrame.connect(self.sigUpdateFrame.emit)
        self.thread.sigUpdateGI.connect(self.sigUpdateGI.emit)
        self.thread.sigXyeOutputReady.connect(self.sigXyeOutputReady.emit)
        # R4A-1(iii): the first JIT-classified container's motor/counter names.
        # Delivered as a VALUE the receiver qualifies by exact accepted-object
        # identity; it is never re-emitted onward unqualified.
        self.thread.sigGISourceMotors.connect(self._on_gi_source_motors)
        # Pause (Phase B): the worker emits sigPaused once it has drained+flushed
        # at a frame boundary.  Morph the action button to Resume here AND
        # re-emit at the wrangler level so the host (staticWidget) can lift the
        # freeze guard for browsing.  _run_phase tracks the Start/Pause/Resume
        # state machine (idle | running | paused).
        self._run_phase = 'idle'
        self.thread.sigPaused.connect(self._on_paused)
        self.thread.sigPaused.connect(self.sigPaused.emit)

        # Enable/disable buttons initially
        self.ui.stopButton.setEnabled(False)

        self.setup()
        self._restore_from_session()
        # Open the GI / Threshold / Background groups when their toggle is on
        # (e.g. from a restored session) so the relevant controls are visible
        # instead of folded; collapsed when off.
        self._expand_active_groups()

    # UI-1 (#81): the GI / Intensity-Threshold groups carry a header CHECKBOX
    # as their on/off toggle, mapped to the hidden bool that is their source
    # of truth (see wranglerWidget._install_group_toggles).
    # Mask / MaskSat moved to the integrator panel (hidden carriers here), so no
    # header-checkbox toggles for them.
    # GI moved to the integrator panel (hidden carrier here) — no header toggle.
    _GROUP_TOGGLES = {}

    def _expand_active_groups(self):
        """Sync each wrangler group's expanded state to its enabling param.

        For the GI / Intensity-Threshold groups the header checkbox is the
        on/off toggle (UI-1); this folds the group to match the restored
        state (open when on, collapsed when off) -- e.g. after a session
        restore.  The Background group only expands-when-on (its toggle
        is the ``bg_type`` list, not the header)."""
        groups = (
            ('GI', ('GI', 'Grazing'), lambda v: bool(v), True),
            ('BG', ('BG', 'bg_type'), lambda v: v not in (None, '', 'None'), False),
        )
        for group_name, child_path, is_on, collapse_when_off in groups:
            try:
                group = self.parameters.child(group_name)
                value = self.parameters.child(*child_path).value()
            except Exception:
                logger.debug("expand-active-group skipped for %s", group_name,
                             exc_info=True)
                continue
            on = is_on(value)
            if on:
                group.setOpts(expanded=True)
            elif collapse_when_off:
                group.setOpts(expanded=False)

    # --- Session persistence ---

    # Flat map: (session_key, param_path_tuple, is_path, self_attr)
    # is_path=True  → only restore if the path exists on disk
    # self_attr     → attribute to sync on restore (None = handled by sigValueChanged)
    _SESSION_PARAMS = [
        ('project_folder', ('Project', 'project_folder'),            True,  'project_folder'),
        ('poni_file',      ('Signal', 'poni_file'),                  True,  'poni_file'),
        ('inp_type',       ('Signal', 'inp_type'),                   False, 'inp_type'),
        ('img_file',       ('Signal', 'File'),                       True,  'img_file'),
        ('img_dir',        ('Signal', 'img_dir'),                    True,  'img_dir'),
        ('include_subdir', ('Signal', 'include_subdir'),             False, 'include_subdir'),
        ('img_ext',        ('Signal', 'img_ext'),                    False, 'img_ext'),
        ('series_average', ('Signal', 'series_average'),             False, 'series_average'),
        ('meta_ext',       ('Signal', 'meta_ext'),                   False, None),
        ('meta_dir',       ('Signal', 'meta_dir'),                   True,  'meta_dir'),
        ('file_filter',    ('Signal', 'Filter'),                     False, 'file_filter'),
        ('mask_file',      ('Signal', 'mask_file'),                  True,  'mask_file'),
        ('bg_type',        ('BG', 'bg_type'),                        False, 'bg_type'),
        ('bg_file',        ('BG', 'File'),                           True,  'bg_file'),
        ('bg_dir',         ('BG', 'Match', 'bg_dir'),                True,  'bg_dir'),
        ('bg_match_fname', ('BG', 'Match', 'match_fname'),           False, 'bg_match_fname'),
        ('bg_file_filter', ('BG', 'Match', 'Filter'),                False, 'bg_file_filter'),
        ('bg_scale',       ('BG', 'Scale'),                          False, 'bg_scale'),
        ('gi',             ('GI', 'Grazing'),                        False, 'gi'),
        # J1: json session key stays 'th_mtr' for back-compat with
        # old session.json files; attr name is now 'incidence_motor'.
        ('th_mtr',         ('GI', 'th_motor'),                       False, 'incidence_motor'),
        ('sample_orientation', ('GI', 'sample_orientation'),         False, 'sample_orientation'),
        ('tilt_angle',     ('GI', 'tilt_angle'),                     False, 'tilt_angle'),
        ('gi_mode_1d',     ('GI', 'gi_mode_1d'),                    False, 'gi_mode_1d'),
        ('gi_mode_2d',     ('GI', 'gi_mode_2d'),                    False, 'gi_mode_2d'),
        ('apply_threshold', ('Mask', 'Threshold'),                   False, 'apply_threshold'),
        ('threshold_min',  ('Mask', 'min'),                          False, 'threshold_min'),
        ('threshold_max',  ('Mask', 'max'),                          False, 'threshold_max'),
        ('mask_sentinel',  ('MaskSat', 'mask_sentinel'),             False, 'mask_sentinel'),
        ('h5_dir',         ('Project', 'h5_dir'),                    True,  'h5_dir'),
    ]

    def _setup_on_value_change(self, param, changes):
        """Gate setup() to real value edits only.

        pyqtgraph fires sigTreeStateChanged for EVERY change including 'options'
        (visibility) and 'limits' (dropdown repopulation).  setup()'s own N1
        progressive disclosure toggles group visibility via setOpts(visible=...),
        so running setup on 'options' changes re-entered it forever (the freeze).
        Only a genuine 'value' edit should re-run the full sync.
        """
        try:
            for ch in changes:
                if ch[1] == 'value':
                    debug_trace = run_config_debug_enabled()
                    summary = (
                        parameter_change_summary(changes)
                        if debug_trace
                        else None
                    )
                    if debug_trace:
                        run_config_debug_log(
                            logger,
                            "legacy_parameter_setup_enter",
                            wrangler=self,
                            origin="hidden_parameter_tree_value",
                            changes=summary,
                        )
                    self.setup()
                    if debug_trace:
                        run_config_debug_log(
                            logger,
                            "legacy_parameter_setup_exit",
                            wrangler=self,
                            origin="hidden_parameter_tree_value",
                            changes=summary,
                        )
                    return
        except (IndexError, TypeError):
            # Malformed change tuple: fall back to the historical behaviour.
            self.setup()

    def _save_to_session(self, *args):
        # Don't persist the transient half-restored state while a session restore
        # is in flight (each setValue fires sigTreeStateChanged -> here): it's a
        # redundant write storm and a crash-corruption window.  The on-disk file
        # already holds the complete values restore is loading.
        if getattr(self, '_restoring', False):
            return
        data = {}
        for key, path, _, _ in self._SESSION_PARAMS:
            try:
                p = self.parameters
                for segment in path:
                    p = p.child(segment)
                data[key] = p.value()
            except (AttributeError, KeyError, TypeError) as e:
                logger.debug("Failed to save session parameter %s: %s", key, e)
        data['processing_mode'] = self.ui.processingModeCombo.currentText()
        # Live is a momentary start/stop control now (not a persisted mode);
        # its checked state means "a live run is active", which must never be
        # restored — doing so would auto-start a run on launch.
        data['batch_mode'] = self.ui.batchCheckBox.isChecked()
        save_session(data)

    def _restore_from_session(self):
        session = load_session()
        # N1: restoring project_folder fires its sigValueChanged ->
        # _on_project_folder_changed, which would DESTRUCTIVELY clear the very
        # poni_file/img_file this loop is about to restore.  Gate it inert for
        # the duration of the restore.  (A project_folder whose root no longer
        # exists is is_path-skipped below -> stays blank -> Decision 3 fresh-start
        # fallback, prompting the user to re-enter the folder.)
        self._restoring = True
        try:
            for key, path, is_path, attr in self._SESSION_PARAMS:
                if key not in session:
                    continue
                val = session.get(key)
                if key == 'meta_ext':
                    # Legacy sessions (pre-'auto') encoded "never set" as
                    # None/'' — the OLD metadata-off default, not a deliberate
                    # choice.  Modern saves write the literal dropdown string,
                    # so only a saved 'none' is an explicit off (MD-2 boundary).
                    # Migrate the legacy-unset encodings to the modern default
                    # instead of resurrecting them as a permanent 'none'.
                    # AND (maintainer, 2026-07-13): a stored literal 'none'
                    # does not persist either — the old nxs sync force-wrote
                    # 'none' into sessions for weeks, so it cannot be trusted
                    # as deliberate.  Metadata-off stays selectable PER
                    # SESSION; every restore comes back at the 'auto' default.
                    if _normalize_meta_ext(val) is None:
                        val = def_meta_ext
                    else:
                        val = _meta_ext_parameter_value(val)
                elif val is None:
                    continue
                if is_path and not Path(val).exists():
                    continue
                try:
                    p = self.parameters
                    for segment in path:
                        p = p.child(segment)
                    p.setValue(val)
                    if key == 'meta_ext':
                        self.meta_ext = _normalize_meta_ext(val)
                    elif attr is not None:
                        setattr(self, attr, val)
                except (AttributeError, KeyError, TypeError, ValueError) as e:
                    logger.debug("Failed to restore session parameter %s: %s", key, e)
            # Restore processing mode dropdown and checkboxes
            mode = session.get('processing_mode')
            if mode:
                idx = self.ui.processingModeCombo.findText(mode)
                if idx >= 0:
                    self.ui.processingModeCombo.setCurrentIndex(idx)
            else:
                # Fresh start (no saved session): default to Int 2D so the cake
                # + raw panels are shown without the user switching modes.
                idx = self.ui.processingModeCombo.findText('Int 2D')
                if idx >= 0:
                    self.ui.processingModeCombo.setCurrentIndex(idx)
            # Deliberately do NOT restore Live's checked state — it's a
            # start/stop control, and setChecked(True) would fire its toggled
            # handler and auto-start a live run on launch.
            if 'batch_mode' in session:
                self.ui.batchCheckBox.setChecked(session['batch_mode'])
            else:
                # Fresh start: default to non-Batch (live per-frame display).
                self.ui.batchCheckBox.setChecked(False)
        finally:
            self._restoring = False
        # meta_ext needs None conversion (sigValueChanged fires set_meta_ext automatically)
        # poni_file needs poni_dict loaded; this re-applies the disclosure for the
        # restored Project-Folder + PONI state (or the fresh-start fallback when
        # the root was missing/skipped).
        self.source_base = self._compute_source_base()
        self.get_poni_dict()

    def _sync_h5_dir_from_parameters(self):
        """Sync the Save Path parameter and notify the scans browser on change.

        Containment rule (Vivek): with a Project Folder set, the Save Path
        must live INSIDE it (the .nxs stores source paths relative to the
        project root, so processed data outside the project breaks the N1
        portability story).  A typed/browsed path outside reverts to the
        default ``<project>/xdart_processed_data`` with a status message
        (the setValue re-enters this method via the tree-change cascade and
        passes validation on the second pass).
        """
        path = self.parameters.child('Project').child('h5_dir').value()
        base = self._compute_source_base()
        if path and base:
            _abs = os.path.abspath(os.path.expanduser(str(path)))
            try:
                inside = os.path.commonpath([_abs, base]) == base
            except ValueError:          # different drives (Windows)
                inside = False
            if not inside:
                # REJECT (Vivek): restore the previous valid value when there
                # is one, else the default.  Note the .nxs never embeds raw
                # data -- sources are stored as references -- the issue with
                # an outside path is purely that the output leaves the
                # portable project tree.
                prev = (getattr(self, 'h5_dir', '') or '').strip()
                try:
                    prev_ok = bool(prev) and os.path.commonpath(
                        [os.path.abspath(os.path.expanduser(prev)), base]
                    ) == base
                except ValueError:
                    prev_ok = False
                fallback = (prev if prev_ok
                            else os.path.join(base, 'xdart_processed_data'))
                imageWrangler._safe_status_text(
                    self,
                    'Save Path rejected: must be inside the Project Folder. '
                    f'Kept {os.path.basename(fallback) or fallback}.',
                )
                self.parameters.child('Project').child('h5_dir').setValue(fallback)
                return
        old_path = getattr(self, 'h5_dir', None)
        self.h5_dir = path
        if path and path != old_path:
            self.sigSavePathChanged.emit(path)

    # Signal to notify static_scan_widget that viewer mode changed.
    # Emits the viewer_mode string ('image', 'xye') or '' for normal.
    sigViewerModeChanged = QtCore.Signal(str)

    # A Run click in a Stitch mode diverts here instead of a wrangler run; the
    # host (staticWidget.start_stitch) launches the stitch worker.  Arg: '1d'|'2d'.
    sigStitchRequested = QtCore.Signal(str)
    # Emitted ('1d'|'2d'|'') when the dropdown enters/leaves a Stitch mode, so the
    # host can route the display to the persistent StitchDisplayController (mirrors
    # sigViewerModeChanged).  Prev-tracked to fire only on an actual change.
    sigStitchModeChanged = QtCore.Signal(str)

    def _on_mode_changed(self, *args):
        """Update all flags from the processing mode dropdown and checkboxes."""
        mode_text = self.ui.processingModeCombo.currentText()
        is_viewer = mode_text in ('Image Viewer', 'XYE Viewer', 'NeXus Viewer')
        is_file_viewer = mode_text in ('Image Viewer', 'XYE Viewer')
        is_xye = mode_text == 'Int 1D (XYE)'

        # Pre-process state overrides
        self.ui.liveCheckBox.blockSignals(True)
        self.ui.batchCheckBox.blockSignals(True)

        if is_viewer:
            self.ui.liveCheckBox.setEnabled(False)
            self.ui.batchCheckBox.setEnabled(False)
            self.ui.coresLabel.setEnabled(False)
            self.ui.maxCoresSpinBox.setEnabled(False)
        elif is_xye:
            # Stash the user's Batch choice once on entering XYE so it can be
            # restored on leaving -- XYE force-checks Batch, and the normal
            # branch used to leave it stuck checked (UI-2).
            if getattr(self, '_pre_xye_batch', None) is None:
                self._pre_xye_batch = self.ui.batchCheckBox.isChecked()
            self.ui.liveCheckBox.setChecked(False)
            self.ui.liveCheckBox.setEnabled(False)
            self.ui.batchCheckBox.setChecked(True)
            self.ui.batchCheckBox.setEnabled(False)
            self.ui.coresLabel.setEnabled(True)
            self.ui.maxCoresSpinBox.setEnabled(True)
        else:
            # Restore the pre-XYE Batch choice when leaving XYE (UI-2).
            if getattr(self, '_pre_xye_batch', None) is not None:
                self.ui.batchCheckBox.setChecked(self._pre_xye_batch)
                self._pre_xye_batch = None
            # Both toggles stay clickable in a normal processing mode; they
            # are kept mutually exclusive by auto-unchecking the other one
            # rather than greying it out.  Greying Live out whenever Batch
            # was checked left it dead after a mode switch until a run
            # finished and enabled(True) reset it (bug #1).
            self.ui.liveCheckBox.setEnabled(True)
            self.ui.batchCheckBox.setEnabled(True)

            is_live = self.ui.liveCheckBox.isChecked()
            is_batch = self.ui.batchCheckBox.isChecked()
            if is_live and is_batch:
                # Uncheck whichever one was NOT just toggled.  When the
                # trigger is the mode combo (or unknown), prefer Live.
                if self.sender() is self.ui.batchCheckBox:
                    self.ui.liveCheckBox.setChecked(False)
                    is_live = False
                else:
                    self.ui.batchCheckBox.setChecked(False)
                    is_batch = False

            # Cores drives the reduction worker pool for BOTH batch (parallel
            # reprocess) and LIVE (the streaming acquisition pool reads max_cores
            # at run start), so keep it usable in any normal processing mode.
            self.ui.coresLabel.setEnabled(True)
            self.ui.maxCoresSpinBox.setEnabled(True)

        self.ui.liveCheckBox.blockSignals(False)
        self.ui.batchCheckBox.blockSignals(False)

        # Cores applies to ANY processing mode (batch parallel reprocess or the
        # live streaming pool), so it is visible in general — hidden ONLY in the
        # file viewers (Image/XYE/NeXus viewer) where the whole processing row is
        # hidden.
        cores_visible = not is_viewer
        self.ui.coresLabel.setVisible(cores_visible)
        self.ui.maxCoresSpinBox.setVisible(cores_visible)

        self.batch_mode = self.ui.batchCheckBox.isChecked()
        self.live_mode = self.ui.liveCheckBox.isChecked()
        self.xye_only = is_xye
        # Stitch 1D / Stitch 2D are post-load batch reductions of the loaded
        # scan, not a wrangler acquisition run; start() diverts to the host's
        # stitch worker when this is set (viewer mode texts never contain it).
        self.stitch_mode = ('Stitch' in mode_text)

        if mode_text == 'Image Viewer':
            self.viewer_mode = 'image'
            self.scan.skip_2d = False
        elif mode_text == 'XYE Viewer':
            self.viewer_mode = 'xye'
            self.scan.skip_2d = False
        elif mode_text == 'NeXus Viewer':
            self.viewer_mode = 'nexus'
            self.scan.skip_2d = False
        else:
            self.viewer_mode = None
            self.scan.skip_2d = ('1D' in mode_text) and ('2D' not in mode_text)

        # Sync to thread

        # Gray out integration controls in viewer mode
        self._set_integration_controls_enabled(not is_viewer)
        # Apply the per-mode param visibility: image/xye viewers HIDE all
        # processing groups (leaving Project Folder + Save Path); any other mode
        # restores the normal Project->Calibration->rest progressive disclosure.
        self._apply_disclosure()
        # Image/XYE viewers are file-inspection modes: the PROCESSING groups
        # (Calibration/Signal/BG/Mask/GI, handled per-group above) are
        # disabled so masks/background/calibration cannot be edited there --
        # but the tree itself stays enabled so Project Folder and Save Path
        # remain usable (they drive the file browser, which is exactly what
        # viewer modes are for).  The run-lock still disables the whole tree
        # during runs.
        try:
            self.tree.setEnabled(True)
        except AttributeError:
            pass
        # Image/XYE viewers are pure file browsers.  Collapse the run controls
        # down to the mode row instead of showing a disabled Run/Stop/Append row.
        # Int 1D (XYE) is NOT a viewer here; it remains a processing mode.
        _controls = getattr(self, '_controls', None)
        if _controls is None:
            self.ui.frame.setVisible(not is_file_viewer)
            self.ui.frame.setEnabled(not is_viewer)
        else:
            _controls.set_run_row_visible(not is_file_viewer)
            _controls.set_run_row_enabled(not is_viewer)
            # No batch processing in a file viewer -> hide the Batch toggle (and
            # the Cores it gates) in viewer modes; apply_profile restores it for
            # processing modes.
            _controls.batchButton.setVisible(not is_viewer)
        # Notify parent only when viewer mode actually changed (avoids
        # unnecessary layout resets when just toggling Live/Batch).
        new_vm = self.viewer_mode or ''
        if not hasattr(self, '_prev_viewer_mode'):
            self._prev_viewer_mode = ''
        if new_vm != self._prev_viewer_mode:
            self._prev_viewer_mode = new_vm
            self.sigViewerModeChanged.emit(new_vm)

        # Stitch-mode dropdown change → the host routes the display to/from the
        # persistent StitchDisplayController.  '' on any non-stitch mode.
        new_sm = ('1d' if mode_text == 'Stitch 1D'
                  else '2d' if mode_text == 'Stitch 2D' else '')
        if new_sm != getattr(self, '_prev_stitch_mode', ''):
            self._prev_stitch_mode = new_sm
            self.sigStitchModeChanged.emit(new_sm)

    def _set_integration_controls_enabled(self, enabled, *, include_gi=True):
        """Enable or disable parameter tree groups related to integration."""
        group_names = ['Signal', 'BG', 'Mask', 'MaskSat']  # poni_file lives in Signal now
        if include_gi:
            group_names.append('GI')
        for group_name in group_names:
            try:
                grp = self.parameters.child(group_name)
                grp.setOpts(enabled=enabled)
            except (AttributeError, KeyError) as e:
                logger.debug("Failed to set enabled state for %s: %s", group_name, e)
        # Also disable write mode and mask file in Signal group
        for child_name in ('write_mode', 'mask_file'):
            try:
                self.parameters.child('Signal').child(child_name).setOpts(enabled=enabled)
            except (AttributeError, KeyError) as e:
                logger.debug("Failed to set enabled state for Signal.%s: %s", child_name, e)
        # Save Path deliberately NOT touched here: it drives the scans
        # browser, which viewer modes rely on; the run-lock (enabled())
        # disables the whole tree during runs.

    def _active_write_mode(self):
        """Output mode ('Append'/'Overwrite') from the shared run Controls.

        Write Mode moved out of the wrangler param tree into the Controls run bar
        (it's a run/output property).  Defaults to the SAFE 'Append' when the
        controls aren't attached (headless/test holders) — never silently
        Overwrite."""
        controls = getattr(self, '_controls', None)
        getter = getattr(controls, 'write_mode', None)
        return getter() if callable(getter) else 'Append'

    def _candidate_append_target_file(self, *, refresh_source=False):
        """Return the .nxs path this raw-source run would append to, if known."""

        if refresh_source:
            try:
                self.get_img_fname()
            except Exception:
                logger.debug("append target source refresh failed", exc_info=True)
        img_file = str(getattr(self, "img_file", "") or "")
        try:
            inp_type = str(
                self.parameters.child("Signal").child("inp_type").value() or ""
            )
            if inp_type != "Image Directory":
                configured = self.parameters.child("Signal").child("File").value()
                if configured:
                    img_file = str(configured)
        except Exception:
            pass
        if not img_file:
            return ""
        try:
            scan_name = imageWrangler._append_scan_name_for_source(img_file)
        except Exception:
            logger.debug("append target scan-name parse failed", exc_info=True)
            return ""
        if not scan_name:
            return ""
        try:
            h5_dir = self.parameters.child("Project").child("h5_dir").value()
        except Exception:
            h5_dir = getattr(self, "h5_dir", "")
        if not h5_dir:
            return ""
        return os.path.abspath(
            os.path.expanduser(os.path.join(str(h5_dir), scan_name + ".nxs"))
        )

    @staticmethod
    def _append_scan_name_for_source(path):
        """Mirror imageThread's output scan-name derivation for Append targets —
        delegates to the ONE canonical ``scan_name_from_source`` (Codex F2)."""
        from .image_wrangler_thread import scan_name_from_source
        return scan_name_from_source(path)

    def _append_target_matches_scan_file(self, scan, *, refresh_source=False):
        target = imageWrangler._candidate_append_target_file(
            self,
            refresh_source=refresh_source,
        )
        data_file = getattr(scan, "data_file", None)
        if not target or not data_file:
            return False
        try:
            current = os.path.abspath(os.path.expanduser(os.fspath(data_file)))
        except (TypeError, ValueError):
            return False
        return current == target

    def _append_target_snapshot_scan(self, target):
        """Read the append target's stored processing config without displaying it."""

        if not target or not os.path.exists(target):
            return None
        scan_name = Path(str(target)).stem
        try:
            scan = LiveScan(
                scan_name,
                data_file=str(target),
                static=True,
                file_lock=getattr(self, "file_lock", None),
            )
            scan.load_from_h5(replace=False, mode="r")
            return scan
        except Exception:
            logger.warning(
                "append target config check could not read %s; proceeding "
                "without pre-run mismatch modal",
                target,
                exc_info=True,
            )
            return None

    def _candidate_processing_mapping(self):
        """Processing mapping of the configuration this Run click would freeze.

        Delegates to the host's PURE staging owner (§9.10 step 3-stage).  Returns
        ``None`` when the host cannot offer one — Controls V2 off, a duck-typed
        holder without the host seam, or a staging refusal (the ordinary Run
        preparation then raises the typed refusal) — so the caller falls back to
        the live display projection exactly as before."""
        host = getattr(self, "_h19_host", None)
        getter = getattr(
            host, "_controls_v2_candidate_processing_mapping", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            logger.debug("candidate run configuration unavailable",
                         exc_info=True)
            return None

    def _append_config_mismatch_details(self):
        scan = getattr(self, "scan", None)
        active_write_mode = getattr(self, "_active_write_mode", None)
        write_mode = (
            active_write_mode()
            if callable(active_write_mode)
            else imageWrangler._active_write_mode(self)
        )
        if str(write_mode or "").strip().lower() != "append":
            return None, None, None

        target = imageWrangler._candidate_append_target_file(
            self,
            refresh_source=True,
        )
        processed = None
        if target and os.path.exists(target):
            # Cold-launch path (CF-3): no viewer-loaded scan yet, but Append is
            # about to target an existing .nxs. Read the target's stored config
            # before any frame can process so the overwrite modal fires pre-run
            # instead of relying on the writer's mid-run backstop. This
            # disk config is authoritative when the append target already exists.
            target_scan = imageWrangler._append_target_snapshot_scan(self, target)
            if target_scan is not None:
                processed = processing_config_from_scan(
                    target_scan,
                    prefer_stored=True,
                )
                try:
                    index = getattr(
                        getattr(target_scan, "frames", None),
                        "index",
                        None,
                    )
                    self._append_target_frame_count = (
                        len(index) if index is not None else 0
                    )
                except Exception:
                    self._append_target_frame_count = 0

        if (
            processed is None
            and scan is not None
            and imageWrangler._append_target_matches_scan_file(self, scan)
        ):
            # Fast path/fallback: when the viewer-loaded scan is the same file
            # this run would append to, use its already-restored reduction config.
            processed = processing_config_from_scan(scan, prefer_stored=True)

        # §10.3: compare the processed target against the CANDIDATE configuration
        # THIS Run click would freeze — never ``self.run_configuration``, which
        # normally belongs to the PREVIOUS run (Append could then pass the pre-run
        # gate and discover the mismatch inside the worker, after raw-data work).
        # The candidate is a pure stage of the revisioned journal winners, so an
        # accepted modal decision is still folded in before the single freeze.
        candidate = imageWrangler._candidate_processing_mapping(self)
        if processed is None or (scan is None and candidate is None):
            return None, None, None
        if candidate is not None:
            current = processing_config_from_mapping(candidate)
        else:
            current = processing_config_from_scan(scan)
        check = append_config_mismatch_check(
            write_mode,
            processed,
            current,
        )
        return check, processed, current

    def _append_config_mismatch_message(self):
        check, _processed, _current = imageWrangler._append_config_mismatch_details(
            self
        )
        if check is None:
            return ""
        return "" if check.ok else check.reason

    @staticmethod
    def _format_append_axis(axis):
        text = str(axis or "").strip()
        lookup = {
            "q": "Q",
            "q-chi": "Q-Chi",
            "2theta": "2theta",
            "2theta-chi": "2theta-Chi",
            "chi": "Chi",
            "q_total": "Q",
            "q_ip": "Qip",
            "q_oop": "Qoop",
            "qip_qoop": "Qip-Qoop",
            "qip_exit": "Qip-Exit",
            "qoop_exit": "Qoop-Exit",
        }
        return lookup.get(text, text.replace("_", "-") or "data grid")

    @staticmethod
    def _format_append_range(value):
        if value is None:
            return ""
        try:
            lo, hi = value
        except (TypeError, ValueError):
            return ""
        try:
            return f" {float(lo):g}-{float(hi):g}"
        except (TypeError, ValueError):
            return f" {lo}-{hi}"

    @staticmethod
    def _format_append_config(sig):
        if sig is None:
            return "unknown"
        parts = [str(getattr(sig, "display_mode", "") or "").strip()]
        # Show BOTH legs with their point counts -- the difference is often in the
        # 1D config (axis/npt), which the old 2D-only summary hid, so the two
        # lines looked identical.
        ax1 = imageWrangler._format_append_axis(getattr(sig, "axis_1d", ""))
        if ax1 and ax1 != "data grid":
            n1 = getattr(sig, "npt_1d", None)
            parts.append(f"1D {ax1}" + (f" ({n1} pts)" if n1 else ""))
        ax2 = getattr(sig, "axis_2d", None)
        if ax2 not in (None, "", "q-chi"):
            nr = getattr(sig, "npt_rad_2d", None)
            na = getattr(sig, "npt_azim_2d", None)
            pts2 = f" ({nr}×{na} pts)" if nr and na else ""
            parts.append(f"2D {imageWrangler._format_append_axis(ax2)}{pts2}")
        return ", ".join(p for p in parts if p) or "unknown"

    def _append_config_mismatch_modal_text(self, processed, current,
                                           mismatched_fields=()):
        details = append_config_difference_lines(
            processed, current, mismatched_fields)
        detail_text = "\n".join(f"- {line}" for line in details)
        if not detail_text:
            detail_text = "- Stored and current integration settings differ."
        return (
            "The existing processed scan was created with integration settings "
            "that do not match the current run.\n\n"
            f"Changed settings:\n{detail_text}\n\n"
            f"Existing scan: {imageWrangler._format_append_config(processed)}\n"
            f"Current run: {imageWrangler._format_append_config(current)}\n\n"
            "Append cannot combine data produced with different settings.\n\n"
            "Choose Replace to overwrite the existing processed data using "
            "the current settings, or Cancel to leave it unchanged."
        )

    def _confirm_append_config_replace(self, check, processed, current):
        parent = self if isinstance(self, QtWidgets.QWidget) else None
        box = QMessageBox(parent)
        try:
            box.setIcon(QMessageBox.Warning)
        except Exception:
            pass
        box.setWindowTitle("Replace existing integration?")
        box.setText(imageWrangler._append_config_mismatch_modal_text(
            self, processed, current,
            getattr(check, "mismatched_fields", ())))
        replace = box.addButton("Replace", QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        return box.clickedButton() is replace

    def _on_append_mismatch(self, message):
        """Mid-run Append config mismatch: the worker already stopped the run
        cleanly and preserved the append target — tell the user WHY the run
        stopped.  Counterpart of the Run-click ``_confirm_append_config_replace``
        modal (which cannot fire for a mid-run settings change); informational
        only, so a single-button warning instead of Yes/No.
        """
        parent = self if isinstance(self, QtWidgets.QWidget) else None
        box = QMessageBox(parent)
        try:
            box.setIcon(QMessageBox.Warning)
        except Exception:
            pass
        box.setWindowTitle("Integration settings changed mid-run")
        box.setText(message)
        box.exec()
        # AFTER the modal: wrangler_finished → stop() clears the status label
        # while the modal is up (its queued delivery runs inside the modal's
        # event loop), so assert the stop reason once the box is dismissed.
        imageWrangler._safe_status_text(self, message)

    def _set_active_write_mode(self, mode):
        controls = getattr(self, "_controls", None)
        setter = getattr(controls, "set_write_mode", None)
        if callable(setter):
            setter(mode)
        else:
            button = getattr(controls, "writeModeButton", None)
            if button is not None and hasattr(button, "setChecked"):
                button.setChecked(str(mode) == "Overwrite")
        self.write_mode = mode

    def _confirm_or_cancel_append_mismatch(self):
        check, processed, current = imageWrangler._append_config_mismatch_details(
            self
        )
        if check is None or check.ok:
            return True
        confirm = getattr(self, "_confirm_append_config_replace", None)
        accepted = (
            confirm(check, processed, current)
            if callable(confirm)
            else imageWrangler._confirm_append_config_replace(
                self, check, processed, current)
        )
        if not accepted:
            imageWrangler._safe_status_text(self, check.reason)
            return False
        imageWrangler._set_active_write_mode(self, "Overwrite")
        return True

    def setup(self):
        """Sets up the child thread, syncs all parameters.
        """
        # Calibration
        global ctr
        ctr += 1

        self.poni_file = self.parameters.child('Signal').child('poni_file').value()
        self.thread.poni = self.poni

        # N1: Project Folder -> @source_base (raw source paths stored RELATIVE to
        # it -> portable .nxs).  Blank -> None -> absolute paths (back-compat).
        self.project_folder = (
            self.parameters.child('Project').child('project_folder').value() or '')
        self.source_base = self._compute_source_base()
        self.thread.source_base = self.source_base

        # Signal
        self.file_filter = self.parameters.child('Signal').child('Filter').value()
        self.thread.publication_store = getattr(self, "publication_store", None)

        self.inp_type = self.parameters.child('Signal').child('inp_type').value()
        self.thread.source_run_plan = self.source_run_plan
        self.thread.source_index_session = self.source_index_session
        self.thread.source_frame_count_snapshot = dict(
            self.source_frame_count_snapshot or {})
        self.thread.source_pending_count = int(
            self.source_pending_count or 0)

        self.get_img_fname()
        self.thread.img_file = self.img_file

        # Container numeric suffixes are scan identity, while image-series
        # numeric suffixes are frame identity.  Use the canonical source rule
        # here too; feeding an already-bare container name back through the
        # image-series parser later made sibling scans 00005/00006 collide.
        self.scan_name = self._append_scan_name_for_source(self.img_file)
        self.thread.scan_name = self.scan_name


        if self.inp_type != "Image Directory":
            self.include_subdir = self.parameters.child(
                'Signal').child('include_subdir').value()

        self.thread.meta_dir = self.meta_dir

        self._sync_h5_dir_from_parameters()
        self.fname = os.path.join(self.h5_dir, self.scan_name + '.nxs')
        self.thread.fname = self.fname

        self.mask_file = self.parameters.child('Signal').child('mask_file').value()

        # Threshold
        self.apply_threshold = self.parameters.child('Mask').child('Threshold').value()
        self.threshold_min = self.parameters.child('Mask').child('min').value()
        self.threshold_max = self.parameters.child('Mask').child('max').value()
        self.mask_sentinel = self.parameters.child('MaskSat').child('mask_sentinel').value()

        # Write Mode
        self.write_mode = self._active_write_mode()

        # Background
        self.bg_type = self.parameters.child('BG').child('bg_type').value()
        self.thread.bg_type = self.bg_type

        self.bg_file = self.parameters.child('BG').child('File').value()
        self.thread.bg_file = self.bg_file

        self.bg_matching_par = self.parameters.child('BG').child('Match').child('Parameter').value()
        self.thread.bg_matching_par = self.bg_matching_par

        self.bg_dir = self.parameters.child('BG').child('Match').child('bg_dir').value()
        self.thread.bg_dir = self.bg_dir

        self.bg_match_fname = self.parameters.child('BG').child('Match').child('match_fname').value()
        self.thread.bg_match_fname = self.bg_match_fname

        self.bg_file_filter = self.parameters.child('BG').child('Match').child('Filter').value()
        self.thread.bg_file_filter = self.bg_file_filter

        self.bg_scale = self.parameters.child('BG').child('Scale').value()
        self.thread.bg_scale = self.bg_scale

        self.bg_norm_channel = self.parameters.child('BG').child('norm_channel').value()
        self.thread.bg_norm_channel = self.bg_norm_channel

        # Grazing Incidence
        self.gi = self.parameters.child('GI').child('Grazing').value()

        # self.incidence_motor = self.parameters.child('GI').child('th_motor').value()

        self.sample_orientation = self.parameters.child('GI').child('sample_orientation').value()

        self.tilt_angle = self.parameters.child('GI').child('tilt_angle').value()

        # GI modes are driven by the integrator panel (axis1D / axis2D),
        # so read them from scan.bai_*_args which the integrator updates.
        self.gi_mode_1d = self.scan.bai_1d_args.get('gi_mode_1d', 'q_total')
        self.gi_mode_2d = self.scan.bai_2d_args.get('gi_mode_2d', 'qip_qoop')
        self.thread.gi_mode_1d = self.gi_mode_1d
        self.thread.gi_mode_2d = self.gi_mode_2d

        # N3: record the GI output mode + geometry as a first-class
        # ``scan.gi_config`` (persisted by the writer to
        # /entry/reduction/config/gi_config), so read_scan can recover the GI
        # mode + axis meaning without sniffing the q-unit string or digging the
        # mode key out of bai_*_args.
        if self.gi:
            self.scan.gi_config = {
                'gi_mode_1d': str(self.gi_mode_1d),
                'gi_mode_2d': str(self.gi_mode_2d),
                'incidence_motor': str(getattr(self, 'incidence_motor', '') or ''),
                'tilt_angle': float(getattr(self, 'tilt_angle', 0.0) or 0.0),
                'sample_orientation': int(getattr(self, 'sample_orientation', 1) or 1),
            }
        else:
            self.scan.gi_config = {}

        # Notify integrator panel so labels/widgets update immediately.
        self.sigUpdateGI.emit(self.gi)

        # Processing mode flags and parallel cores
        # Parallelism is frozen run policy; the display scan keeps its own copy
        # for scan_threads (a GUI consumer, not the run).
        self.scan.max_cores = self.ui.maxCoresSpinBox.value()

        self.thread.command = self.command

        self.thread.file_lock = self.file_lock

        self.thread.scan = self.scan

    def _set_wrangler_tooltips(self):
        """Hover tooltips for the wrangler command/run controls."""
        tips = {
            'processingModeCombo': 'What to produce: integrate (1D/2D/XYE), '
                                   'stitch, or just view images/patterns.',
            'liveCheckBox': 'Start/stop live acquisition — process frames as '
                            'they arrive.',
            'batchCheckBox': 'Process all frames as a batch (parallel across '
                             'Cores) instead of one-at-a-time.',
            'maxCoresSpinBox': 'CPU cores for parallel batch/live processing.',
            'coresLabel': 'CPU cores for parallel batch/live processing.',
            'advancedButton': 'Advanced integration / detector options.',
            'startButton': 'Start processing with the current settings.',
            'stopButton': 'Stop the running process.',
        }
        for name, tip in tips.items():
            w = getattr(self.ui, name, None)
            if w is not None:
                w.setToolTip(tip)

    def controls_profile(self):
        """Image wrangler: Live + Batch + cores, full mode list (NeXus Viewer is
        deferred / hidden for this wrangler).  ``current`` is read from the
        wrangler's own (pre-alias) combo, which __init__ already restored from
        the session — so the shared combo can be restored to it on attach."""
        try:
            current = self.ui.processingModeCombo.currentText()
        except Exception:
            current = ''
        return {
            'modes': ['Int 1D', 'Int 2D', 'Int 1D (XYE)',
                      'Stitch 1D', 'Stitch 2D',
                      'Image Viewer', 'XYE Viewer'],
            'live': True, 'batch': True, 'cores': True,
            'current': current,
        }

    def attach_controls(self, controls):
        """Adopt the shared run controls: hide this wrangler's own (now-dead)
        specUI command/run rows, ALIAS the ``self.ui.*`` control refs onto the
        shared widgets (so all existing run-lifecycle logic drives them), and
        wire the shared signals to this wrangler's handlers."""
        super().attach_controls(controls)
        # Hide the now-dead specUI rows: the command/run rows (controls moved to
        # StaticControls) AND specLabel — the status bar moved to the control
        # layer (controls.statusLabel via _status_label), so specLabel would just
        # sit as an empty bar above the integrator.
        for name in ('commandFrame', 'frame', 'specLabel'):
            w = getattr(self.ui, name, None)
            if w is not None:
                w.hide()
        self.ui.startButton = controls.startButton
        self.ui.stopButton = controls.stopButton
        self.ui.processingModeCombo = controls.modeCombo
        self.ui.liveCheckBox = controls.liveButton
        self.ui.batchCheckBox = controls.batchButton
        self.ui.maxCoresSpinBox = controls.coresSpin
        # Alias coresLabel too, so _on_mode_changed's batch-gated cores show/hide
        # (+ enable) drives the SHARED label/spin as a unit (Cores hidden when
        # Batch is off).
        self.ui.coresLabel = controls.coresLabel
        self._connect_control(controls.startButton.clicked, self._on_start_clicked)
        # Stop is NOT connected here: the host (staticWidget._on_stop_clicked)
        # owns the shared Stop button and dispatches to the active run (a
        # reintegrate, else this wrangler's stop()).
        self._connect_control(controls.modeCombo.currentTextChanged,
                              self._on_mode_changed)
        self._connect_control(controls.liveButton.toggled, self._on_mode_changed)
        self._connect_control(controls.batchButton.toggled, self._on_mode_changed)
        self._connect_control(controls.liveButton.toggled, self._on_live_toggled)
        self._connect_control(controls.modeCombo.currentTextChanged,
                              lambda _=None: self._save_to_session())
        self._connect_control(controls.liveButton.toggled,
                              lambda _=None: self._save_to_session())
        self._connect_control(controls.batchButton.toggled,
                              lambda _=None: self._save_to_session())
        self._set_action_button('idle')          # reset the morph on (re)attach

    def _on_start_clicked(self):
        """The single action button is a 3-state machine (Phase B):

        * **idle** -> Start a run honoring the Live/Batch MODE toggles
          (``live_mode``/``batch_mode``, already synced by ``_on_mode_changed``).
          Live is no longer force-unchecked — it is an honored mode, not a
          competing action.
        * **running** -> Pause (freeze at a frame boundary; browse from disk).
        * **paused** -> Resume.

        Stop (separate red button) remains the terminal action from any state."""
        phase = getattr(self, '_run_phase', 'idle')
        if phase == 'pausing':
            return            # transient (button disabled); ignore stray re-dispatch
        if phase == 'running':
            self.pause()
        elif phase == 'paused':
            self._on_resume()
        else:
            self.start()

    def _inputs_valid(self, staged=None):
        """Whether the wrangler can start a run.  PURE -- it publishes nothing.

        A loaded PONI calibration is required; without this gate a Start/Live
        click with no (or an invalid) PONI ran the *previous* scan with the
        stale calibration (BUG-1).  The image source / save path remain guarded
        inside the run thread.

        O-1a-W1R (review §39.2 W1R-P1-8): validation used to CALL
        ``_adopt_loaded_scan_run_inputs()``, which published the loaded scan's
        PONI/integrator onto ``self.poni`` and three worker carriers -- before
        admission had a chance to refuse.  A refused Start was therefore not
        zero-delta.  Adoption is now STAGED (see
        :meth:`_stage_loaded_scan_calibration`) and only projected onto the
        legacy carriers after exact admission, so this predicate decides from the
        candidate values without writing any of them.
        """
        if staged is None:
            staged = imageWrangler._stage_loaded_scan_calibration(self)
        # GENERIC-DETECTOR FIX (block, not crash): a processed .nxs saved WITHOUT
        # detector pixel sizes AND without a resolvable detector name (an
        # unnamed/generic 'Detector') restores no usable calibration — its
        # ``_restore_calibration_from_group`` returns False, leaving both
        # ``_cached_poni`` AND ``_cached_integrator`` None.  Adoption then can't
        # seed ``self.poni`` and a Run would otherwise build a pixel-less
        # integrator and crash mid-write in pyFAI's calc_cartesian_positions
        # ('NoneType' * float).  Refuse up front with a SPECIFIC message when a
        # scan is loaded but carries no usable calibration; the user can still
        # supply their own .poni (a different poni object than the scan's).
        scan = getattr(self, "scan", None)
        scan_loaded = bool(
            list(getattr(getattr(scan, "frames", None), "index", ()) or ()))
        scan_has_integrator = getattr(scan, "_cached_integrator", None) is not None
        # The calibration THIS click would run with: the already-installed one,
        # else the staged candidate.  Neither is published here.
        candidate_poni = self.poni
        candidate_is_adopted = False
        if candidate_poni is None and staged is not None:
            candidate_poni = staged.get("poni")
            candidate_is_adopted = candidate_poni is not None
        adopted_poni = getattr(self.thread, "_adopted_poni", None)
        candidate_is_adopted = candidate_is_adopted or (
            adopted_poni is not None and candidate_poni is adopted_poni)
        adopted_pixel_less = candidate_is_adopted and not scan_has_integrator
        if candidate_poni is None or adopted_pixel_less:
            if scan_loaded and not scan_has_integrator:
                imageWrangler._safe_status_text(
                    self,
                    'This scan was saved without a usable detector calibration '
                    '— load a PONI calibration file to integrate.',
                )
            else:
                imageWrangler._safe_status_text(
                    self,
                    'Load a PONI calibration file to begin.',
                )
            return False
        if (not imageWrangler._authoritative_source_selected(self)
                and not getattr(self, 'stitch_mode', False)
                and not imageWrangler._directory_run_without_seed_ok(self)):
            imageWrangler._safe_status_text(
                self,
                'Choose an image source to run. Use Reintegrate for a loaded processed scan.',
            )
            return False
        if not imageWrangler._confirm_or_cancel_append_mismatch(self):
            return False
        return True

    def _authoritative_source_mode(self):
        """The source MODE the Source card selects (review §42.2).

        ``self.inp_type`` is display/compatibility state and may not decide which
        source leg runs.  There is deliberately no fallback to it: a host that
        exercises these paths provides the card production reads.
        """
        return str(
            self.parameters.child('Signal').child('inp_type').value() or '')

    def _authoritative_source_selected(self):
        """Whether the CURRENT selection names a usable source.

        O-1a-W1R-D1 (review §40.3 D1 item 2).  ``_inputs_valid`` used to test
        ``self.img_file``, an instance cursor that could outlive the selection it
        described; a Start therefore passed validation on a file the operator had
        already cleared.  Validation now reads the authoritative parameter, so a
        blank or non-existent selection cannot be certified by a stale cursor even
        if one is restored by hand.
        """
        signal = self.parameters.child('Signal')
        # O-1a-W1R-D1 (review §41.3.D): the MODE comes from the Source card too.
        # Branching on the mutable ``self.inp_type`` mirror let a poisoned mode
        # send the gate to the Directory leg, whose any-nonblank-string rule is
        # the weaker one.
        if imageWrangler._authoritative_source_mode(self) == 'Image Directory':
            root = str(signal.child('img_dir').value() or '').strip()
            # A nonexistent directory is not a source.  An EMPTY but existing
            # directory stays legitimate: the worker discovers candidates after
            # Run, so this must not demand a representative frame.
            return bool(root) and os.path.isdir(os.path.expanduser(root))
        selected = str(signal.child('File').value() or '').strip()
        return bool(selected) and os.path.exists(selected)

    def _directory_run_without_seed_ok(self):
        """Whether a configured Image Directory may start without ``img_file``.

        Directory selection no longer searches for a representative container.
        The worker discovers and inspects candidates after Run, so both batch
        and Live directory runs intentionally begin with an empty ``img_file``.
        """
        try:
            inp_type = str(
                self.parameters.child("Signal").child("inp_type").value() or "")
        except Exception:
            return False
        if inp_type != "Image Directory":
            return False
        try:
            img_dir = str(
                self.parameters.child("Signal").child("img_dir").value() or "").strip()
        except Exception:
            return False
        if not img_dir or not os.path.isdir(os.path.expanduser(img_dir)):
            return False
        # O-1a-W1R-D1 (review §40.3 D0 item 8): the eligibility question is
        # "does the freeze owner produce a typed directory source", not "is this
        # a CONTAINER directory".  Gating on the container index config made
        # every plain-image directory selection unstartable while leaving it
        # selectable in the production File Type list.
        host = getattr(self, "_h19_host", None)
        freeze = getattr(host, "_controls_v2_freeze_source_spec", None)
        try:
            from xrd_tools.sources import DirectorySourceSpec

            return bool(callable(freeze)
                        and isinstance(freeze(), DirectorySourceSpec))
        except Exception:
            return False

    # Compatibility name retained for focused tests and older callers.
    def _h19_empty_directory_live_run_ok(self):
        return imageWrangler._directory_run_without_seed_ok(self)

    def _stage_loaded_scan_calibration(self):
        """CANDIDATE loaded-scan calibration for this click -- published nowhere.

        O-1a-W1R (review §39.2 W1R-P1-8, §39.5 Phase 3 item 5).  A processed
        xdart ``.nxs`` can carry the original PONI/integrator, which keeps
        Reintegrate and an explicitly configured fresh Run calibrated.  Adoption
        must reach the FROZEN provenance (§10.5), so the candidate is staged in
        ``self._staged_run_calibration`` where the freeze owner can read it, and
        the legacy wrapper/worker carriers are written only by
        :meth:`_publish_staged_calibration` after exact admission.

        Returns the candidate mapping, or ``None`` when there is nothing to
        adopt.  Absence, foreign identity, stale identity and freeze failure all
        leave every carrier unchanged, because none was touched.
        """
        scan = getattr(self, "scan", None)
        if self.poni is not None:
            self._staged_run_calibration = None
            return None
        cached_poni = getattr(scan, "_cached_poni", None)
        if cached_poni is None:
            self._staged_run_calibration = None
            return None
        staged = {
            "poni": cached_poni,
            "integrator": getattr(scan, "_cached_integrator", None),
            "fiber_integrator": getattr(scan, "_cached_fiber_integrator", None),
        }
        self._staged_run_calibration = staged
        return staged

    def _publish_staged_calibration(self, staged):
        """Project a STAGED calibration candidate onto the legacy carriers.

        Called only after exact admission (review §39.5 Phase 3 item 5).  This is
        the only writer of the adopted PONI/integrator carriers.
        """
        if not staged:
            return None
        cached_poni = staged.get("poni")
        if cached_poni is None or self.poni is not None:
            return None
        self.poni = cached_poni
        try:
            self.thread.poni = cached_poni
            # GENERIC-DETECTOR FIX: the restored ``_cached_integrator`` carries
            # the detector PIXEL SIZE that a pixel-less PONI dataclass (detector
            # NAME only) cannot rebuild.  Carry it to the run thread KEYED on the
            # adopted poni, so the thread's poni-identity rebuild block REUSES it
            # instead of clobbering it with a pixel-less ``poni_to_integrator``.
            # A genuinely-new user-loaded .poni is a DIFFERENT object, so it
            # still rebuilds (the key won't match).
            self.thread._adopted_poni = cached_poni
            self.thread._adopted_integrator = staged.get("integrator")
            self.thread._adopted_fiber_integrator = staged.get(
                "fiber_integrator")
        except Exception:
            pass
        return cached_poni

    def _adopt_loaded_scan_run_inputs(self):
        """Compatibility shim: stage THEN publish in one step.

        Retained for non-run callers (and older focused tests) that legitimately
        want the adoption applied immediately.  The RUN path deliberately does
        not use it -- see :meth:`_stage_loaded_scan_calibration`.
        """
        return imageWrangler._publish_staged_calibration(
            self, imageWrangler._stage_loaded_scan_calibration(self))

    def start(self):
        """The composed Start boundary (O-1a-T3, §9.10 steps 2 and 5).

        Exact order, and the order matters at every step::

            refuse active/stopping        <- BEFORE any mutation whatsoever
            -> validate + adopt calibration + Append modal on the CANDIDATE
            -> harvest revision winners -> pure stage -> checked commit
            -> ONE freeze -> publish that SAME frozen object to every owner
            -> start

        M2 / the r1 fast-Start defect: Stop morphs the button back to green
        immediately, but the worker can take seconds to unwind (final flush,
        bounded writer join).  A Start click in that window must not revive the
        old run, and — the part r1 got wrong — must not consume a next-run edit,
        adopt calibration, run the modal, freeze, or publish onto the live worker
        BEFORE refusing.  So the refusal is FIRST, through the ONE predicate that
        also covers integration/reintegration, stitch, and the host run latch.
        """
        owner = imageWrangler._active_run_owner(self)
        if owner is not None:
            # T-3.1: `owner` is now non-None for an UNOBSERVABLE owner too, so
            # the refusal text distinguishes "still stopping" from "could not
            # confirm".  Either way this returns before `_inputs_valid`, command
            # mutation, the button morph, preparation, and `sigStart`.
            imageWrangler._safe_status_text(
                self, _run_owner_refusal_status(owner))
            return
        # Validation, PONI adoption, and the Append-mismatch modal all run BEFORE
        # the freeze now: adoption must reach the frozen provenance (§10.5), and
        # an accepted modal decision must be applied to the candidate rather than
        # arriving after the run configuration was already frozen (§10.3/§10.4).
        # O-1a-W1R (review §39.2 W1R-P1-8): STAGE the loaded-scan calibration
        # candidate and validate from it.  Nothing is published yet, so every
        # refusal below is zero-delta on the PONI carriers.
        staged = imageWrangler._stage_loaded_scan_calibration(self)
        # O-1a-W1R-D1 (review §41.3.C): ONE bounded cleanup owner for the staged
        # calibration candidate.  The branch-local clears covered the admission
        # refusal paths and the success path, but an ``_inputs_valid()`` refusal
        # and the Stitch diversion below both returned with the candidate still
        # parked -- so a later freeze could read a calibration staged by a click
        # that never ran.  Every early return, refusal, success and exception now
        # leaves the slot empty.
        try:
            if not self._inputs_valid(staged):
                return
            if getattr(self, 'stitch_mode', False):
                # Stitch is a one-shot batch reduction of the already-loaded
                # scan, not a wrangler acquisition run.  Divert to the host's
                # stitch worker and return BEFORE the Pause/Resume morph or
                # sigStart, so the action button stays a green "Run" (stitch is
                # Start/Stop only).
                self.sigStitchRequested.emit(
                    '1d' if '1D' in self.ui.processingModeCombo.currentText()
                    else '2d')
                return
            # O-1a-W1A: the frozen configuration is MANDATORY on the run path.
            # It is produced here, before any command/button/session/source/
            # output/worker side effect, and absence or a foreign result is a
            # typed refusal — never a fall back to display state (W-1.2 case 5).
            try:
                imageWrangler._admit_run_configuration(self, "image-start")
            except RunConfigurationRefused as exc:
                imageWrangler._report_run_configuration_refusal(
                    self, exc, origin="image_wrangler_start")
                return
            except Exception as exc:
                logger.exception("could not freeze Controls run configuration")
                imageWrangler._safe_status_text(
                    self, f"Run configuration is invalid: {exc}")
                return
            # Admitted: NOW the staged calibration may reach the legacy carriers.
            imageWrangler._publish_staged_calibration(self, staged)
            self.command = 'start'
            self.thread.command = 'start'
            self.ui.stopButton.setEnabled(True)
            # morph green Start -> orange Pause
            self._set_action_button('running')
            self.sigStart.emit()
        finally:
            self._staged_run_calibration = None

    def pause(self):
        """Request a Pause: freeze processing at a frame boundary without
        tearing down the scan/session (worker's _wait_if_paused handles it).
        Mirror the command onto self + thread (same delivery as stop())."""
        # RS-2: never overwrite a stop — the worker self-stops by writing
        # thread.command='stop' directly (write-failure stop, GI freeze
        # abort); blindly writing 'pause' here would silently revive a run
        # that just declared itself dead.  command_lock makes the
        # check-then-set atomic against those worker writes.
        with self.thread.command_lock:
            if self.command == 'stop' or self.thread.command == 'stop':
                return
            self.command = 'pause'
            self.thread.command = 'pause'
        # Transient state until the worker confirms via sigPaused -> _on_paused
        # (which morphs to Resume).  Disabled so a double-click can't race.
        self._set_action_button('pausing')

    def _on_paused(self):
        """GUI slot for the worker's sigPaused (run frozen at a frame boundary).
        The host (staticWidget) lifts the freeze guard off the same signal."""
        # sigPaused is queued from the worker thread.  If a Stop landed during
        # the transient 'Pausing…' window (stop() already morphed the button to
        # green 'idle'), a late sigPaused must NOT flash the button back to orange
        # 'Resume' — self.command (GUI-thread mirror) is 'stop' by then.
        # Also honor a WORKER self-stop (thread.command) for symmetry (RS-2).
        if self.command == 'stop' or self.thread.command == 'stop':
            return
        self._set_action_button('paused')    # orange 'Resume'

    def _on_resume(self):
        """Resume from paused.  Re-engage the freeze guard FIRST (the host's
        sigResuming slot runs synchronously, same GUI thread), THEN flip the
        command back to the run state so a browse read can't race the restarted
        writer."""
        # RS-2: a stop that landed while paused must stay a stop (see pause()).
        with self.thread.command_lock:
            if self.command == 'stop' or self.thread.command == 'stop':
                return
        # Emit OUTSIDE the lock (the host's synchronous sigResuming slot must
        # not run under it), then flip the command under a fresh check.
        self.sigResuming.emit()
        with self.thread.command_lock:
            if self.command == 'stop' or self.thread.command == 'stop':
                return
            self.command = 'start'
            self.thread.command = 'start'
        self._set_action_button('running')   # orange 'Pause'

    def _set_action_button(self, phase):
        """Morph the single action button (Start/Pause/Resume) by text + an
        orange-vs-green visual state driven by a dynamic ``runPhase`` Qt property
        (styled in the dark theme).  ``pausing`` is a transient disabled state."""
        btn = self.ui.startButton
        # Labels carry the run-control glyphs (▶ play / ❚❚ pause), matching
        # StaticControls._PHASES — both drive this same button.
        label, prop, enabled = {
            'idle':    ('▶ Run',      'idle',   True),
            'running': ('❚❚ Pause',    'active', True),
            'pausing': ('❚❚ Pausing…', 'active', False),
            'paused':  ('▶ Resume',   'active', True),
        }.get(phase, ('▶ Run', 'idle', True))
        # Keep the transient 'pausing' distinct (not collapsed into 'running'),
        # so a re-dispatch during the disabled 'Pausing…' window is an explicit
        # no-op in _on_start_clicked rather than a redundant second pause().
        self._run_phase = phase if phase in (
            'idle', 'running', 'pausing', 'paused') else 'idle'
        btn.setText(label)
        btn.setEnabled(enabled)
        if btn.property('runPhase') != prop:
            btn.setProperty('runPhase', prop)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _on_live_toggled(self, checked):
        """Live is a pure MODE toggle now (Phase B): it does NOT start/stop a
        run — that is the single Start/Pause/Resume action button's job.  It just
        records live_mode (``_on_mode_changed``, connected first, already does;
        resync defensively)."""
        self.live_mode = bool(checked)

    def stop(self):
        self.command = 'stop'
        self.thread.command = 'stop'
        self.ui.stopButton.setEnabled(False)
        imageWrangler._safe_status_text(self, '')
        self._set_action_button('idle')       # morph back to green 'Start'
        # Keep the Live toggle in sync when stopped via the Stop button or
        # programmatically — uncheck it without re-entering stop().  Because
        # the uncheck is done with signals blocked, ``_on_mode_changed`` does
        # NOT run, so reset ``live_mode`` explicitly here; otherwise it stays
        # stale-True and the next Start click silently runs in live mode.
        if self.ui.liveCheckBox.isChecked():
            self.ui.liveCheckBox.blockSignals(True)
            self.ui.liveCheckBox.setChecked(False)
            self.ui.liveCheckBox.blockSignals(False)
        self.live_mode = False

    def _browse_dir(self, current: str = '') -> str:
        """Start directory for the file dialogs: the LAST successful pick --
        any field, any wrangler, persisted across sessions (maintainer rule,
        2026-07-13; ``xdart.utils.browse``) -- else the current field's folder,
        else the Project Folder, else Qt's default ("")."""
        return browse_start_dir(current, fallback=self.project_folder or '')

    def set_poni_file(self):
        """Opens file dialogue and sets the calibration file
        """
        fname, _ = QFileDialog().getOpenFileName(
            dir=self._browse_dir(self.poni_file),
            filter="PONI (*.poni *.PONI)"
        )
        if fname != '':
            remember_browse_path(fname)
            self.parameters.child('Signal').child('poni_file').setValue(fname)
            self.poni_file = fname
            self._save_to_session()

    # Two-stage disclosure: Project (Folder + Save Path) is always visible; the
    # DATA group (PONI first, then the source rows) + BG reveal once a Project
    # Folder is set.  (The PONI used to be its own CALIBRATION stage; it now lives
    # inside DATA, so the staging collapsed — see _apply_disclosure.)
    # Intensity Threshold ('Mask') + Mask Saturated ('MaskSat') moved to the
    # integrator panel — kept here as hidden carriers the integrator injects into
    # at run-setup, so never disclosed (and re-hidden after "reveal everything").
    _DISCLOSURE_REST = ('Signal', 'BG')
    # GI joins Mask/MaskSat as a HIDDEN CARRIER: the integrator panel owns the GI
    # controls now and injects into this group at run-setup (_push_gi_to_wrangler).
    _DISCLOSURE_CARRIERS = ('Mask', 'MaskSat', 'GI')
    _DISCLOSURE_TOPLEVEL = ()              # Save Path now lives inside PROJECT

    @staticmethod
    def _set_param_visible_if_changed(param, visible):
        """Set pyqtgraph parameter visibility only on a real transition.

        ``Parameter.show()/hide()`` can emit ``sigOptionsChanged`` even when the
        row is already in the requested state, which makes disclosure calls noisy
        enough to feed setup/session/refresh storms. Guard before ``setOpts`` so
        unchanged disclosure passes are silent.
        """
        target = bool(visible)
        current = bool(param.opts.get('visible', True))
        if current != target:
            param.setOpts(visible=target)


    def _apply_disclosure(self):
        """N1 progressive disclosure (design §2): the tree reveals in stages —
        Project Folder (always) -> Calibration (once a folder is set) -> the rest
        (once a folder is set AND a valid PONI loads).  Pure show()/hide() on the
        groups; orthogonal to the run-lock ``enabled()`` (which only greys)."""
        # Defensive: some lightweight/duck wranglers (and GUI test harnesses)
        # don't build the full disclosure param tree.  _on_mode_changed now calls
        # this, so guard against a tree without the 'Project' group (nothing to
        # disclose there) rather than KeyError.
        if 'Project' not in self.parameters.names:
            return
        # Image/XYE viewer modes are pure file-inspection: hide every processing
        # group, leaving only Project Folder + Save Path (the file-browser
        # drivers).  Centralized HERE because _apply_disclosure is re-invoked on
        # every Project-Folder / PONI change and its else-branch re-shows every
        # child, so a check placed only in _on_mode_changed would be silently
        # undone by the next folder/PONI event.
        if getattr(self, 'viewer_mode', None) in ('image', 'xye'):
            # Project (Folder + Save Path) stays; every processing group hides.
            for child in self.parameters.children():
                imageWrangler._set_param_visible_if_changed(
                    child, child.name() == 'Project')
            imageWrangler._safe_status_text(self, '')
            return
        have_root = self._compute_source_base() is not None
        have_poni = self.poni is not None
        # Project (Folder + Save Path) is always visible.
        imageWrangler._set_param_visible_if_changed(
            self.parameters.child('Project'), True)

        # Two-stage disclosure (the PONI picker now lives inside DATA as the first
        # row, so it can no longer be its own reveal stage): Project Folder ->
        # the whole DATA group once a folder is set.  PONI validity is enforced at
        # run time (the run guard / _inputs_valid), not by hiding DATA.
        if not have_root:
            for child in self.parameters.children():
                if child.name() != 'Project':
                    imageWrangler._set_param_visible_if_changed(child, False)
            imageWrangler._safe_status_text(
                self, 'Choose a Project Folder to begin.')
        else:
            # Set each group to its FINAL visibility DIRECTLY.  The old
            # "show() ALL children, THEN hide() the carriers" re-showed the
            # hidden carriers on every call (visible False->True) and hid them
            # again (True->False) — a genuine toggle that fires sigTreeStateChanged
            # ('options') EVERY time.  With _apply_disclosure re-invoked on mode
            # sync, that oscillated carrier visibility forever and froze the panel.
            # Setting the target directly with a guard avoids pyqtgraph show/hide's
            # unconditional options emit and makes disclosure idempotent.
            # (getattr default keeps lightweight duck holders (tests) working.)
            carriers = set(getattr(self, '_DISCLOSURE_CARRIERS', ()))
            for child in self.parameters.children():
                try:
                    imageWrangler._set_param_visible_if_changed(
                        child, child.name() not in carriers)
                except Exception:
                    pass
            imageWrangler._safe_status_text(
                self,
                '' if have_poni else 'Load a PONI calibration to enable a run.')

    def get_poni_dict(self):
        """Load the PONI calibration file and store as a PONI object, then apply
        the N1 progressive disclosure (reveal the rest only with a Project Folder
        AND a valid PONI)."""
        if not os.path.exists(self.poni_file):
            # No calibration: clear any stale PONI so the run guard +
            # _inputs_valid trip (hiding it left the previous scan's PONI live, so
            # a Start click ran the old scan with the stale calibration, BUG-1).
            self.poni = None
            self.thread.poni = None
            self._apply_disclosure()
            return

        try:
            self.poni = PONI.from_poni_file(self.poni_file)
        except (IOError, OSError, ValueError, KeyError) as e:
            logger.debug("Failed to load PONI file %s: %s", self.poni_file, e)
            self.poni = None
        if self.poni is None:
            logger.warning('Invalid Poni File: %s', self.poni_file)
            self.thread.poni = None
            self.thread.signal_q.put(('message', 'Invalid Poni File'))
            self._apply_disclosure()
            return

        self._apply_disclosure()

    def set_inp_type(self):
        """Change Parameter Names depending on Input Type
        """
        self.single_img = False
        self.parameters.child('Signal').child('File').show()
        self.parameters.child('Signal').child('img_dir').hide()
        self.parameters.child('Signal').child('include_subdir').hide()
        self.parameters.child('Signal').child('Filter').hide()
        self.parameters.child('Signal').child('series_average').show()
        self.parameters.child('Signal').child('img_ext').hide()

        inp_type = self.parameters.child('Signal').child('inp_type').value()
        if inp_type == 'Image Directory':
            self.parameters.child('Signal').child('File').hide()
            self.parameters.child('Signal').child('img_dir').show()
            self.parameters.child('Signal').child('include_subdir').show()
            self.parameters.child('Signal').child('Filter').show()
            self.parameters.child('Signal').child('img_ext').show()

        if inp_type == 'Single Image':
            self.single_img = True
            self.parameters.child('Signal').child('series_average').hide()

        self.inp_type = inp_type
        self.get_img_fname()

    def set_img_file(self):
        """Opens file dialogue and sets the spec data file
        """
        fname, _ = QFileDialog().getOpenFileName(
            dir=self._browse_dir(self.img_file),
            filter="Images (*.tiff *.tif *.h5 *.hdf5 *.nxs *.raw *.mar3450)"
        )
        if fname != '':
            remember_browse_path(fname)
            self.parameters.child('Signal').child('File').setValue(fname)

    def set_img_dir(self):
        """Opens file dialogue and sets the signal data folder
        """
        path = QFileDialog().getExistingDirectory(
            caption='Choose Image Directory',
            dir=self._browse_dir(self.img_dir),
            options=QFileDialog.ShowDirsOnly
        )
        if path != '':
            remember_browse_path(path)
            self.parameters.child('Signal').child('img_dir').setValue(path)
            self.img_dir = path

    def _find_image_directory_seed(self, match, suffix):
        """Return the first usable image-directory frame with a short cache.

        ``setup()`` is connected to the whole parameter tree, so one user edit
        can call this several times in a burst. Re-walking a large image
        directory on every signal blocks the GUI; this cache coalesces only that
        burst (sub-second TTL) and still rescans on later Start/setup calls.
        """
        key = (
            self.img_dir,
            suffix.lower(),
            bool(self.include_subdir),
            self.file_filter,
            self.meta_ext,
            self.meta_dir,
            self.poni_file,
        )
        now = time.monotonic()
        cached_key, cached_value, cached_at = self._img_dir_probe_cache
        if cached_key == key:
            # A FOUND seed for these exact inputs never changes, so cache it
            # INDEFINITELY: setup() runs on every parameter edit (threshold, bg,
            # write-mode, ...), and re-walking a large image directory on each one
            # is the dominant source-config GUI stall.  A NEGATIVE (None) result is
            # only burst-coalesced (0.5s TTL) — the directory may still be filling
            # (frames arriving), so re-walk soon rather than poison the cache with
            # a stale "no frames" (cf. the negative-cache trap); a genuine source
            # change (dir/ext/filter/subdir/meta/poni) changes ``key`` and rescans.
            if cached_value is not None or (now - cached_at) <= 0.5:
                return cached_value

        found = None
        for _idx, (subdir, _dirs, files) in enumerate(os.walk(self.img_dir)):
            for file in files:
                fname = os.path.join(subdir, file)
                if not (
                    file.lower().endswith(suffix.lower())
                    and match(file[:-len(suffix)])
                ):
                    continue
                if not match_img_detector(fname, self.poni):
                    continue
                # Container inputs are self-describing raw sources.  Metadata
                # mode may enrich them with a sidecar, but the absence of one
                # must never hide a valid .nxs/.h5 detector file from the
                # directory worker.  This gate became a release blocker when
                # v1.1.3 made ``auto`` the NeXus default: every sidecar-free raw
                # container was rejected before processing could start.
                needs_sidecar = (
                    self.meta_ext
                    and Path(fname).suffix.lower() not in _EMBEDDED_META_EXTS
                )
                if needs_sidecar and not self.exists_meta_file(fname):
                    continue
                found = fname
                break
            if found is not None or (not self.include_subdir):
                break

        self._img_dir_probe_cache = (key, found, now)
        return found

    def _directory_metadata_preview_suffixes(self):
        spec = getattr(self, "source_spec", None)
        suffixes = tuple(getattr(spec, "suffixes", ()) or ())
        if suffixes:
            return suffixes
        ext = str(self.img_ext or "").lstrip(".").lower()
        if ext == "hdf5":
            return ("_master.hdf5", "_master.h5")
        if ext == "h5":
            return ("_master.h5",)
        return (f".{ext}",) if ext else ()

    def _metadata_preview_result(self, path):
        """THE usability definition: an immutable preview value, or ``None``.

        O-1b R4A-2 (review §49.3 item 1).  This is the SINGLE predicate: there
        may not be one "looks like a candidate" rule for discovery and a
        different "can actually adopt" rule for adoption, because the two drift
        (§50.1) and discovery then selects what adoption rejects.  An empty or
        torn file, a processed xdart output, a detectorless container and an
        unreadable candidate all answer ``None`` -- none of them carries the
        embedded Bluesky columns, and none resolves sidecar metadata -- so the
        caller can skip it and keep looking.

        The candidate is content-opened HERE, exactly once; adoption consumes
        this same value rather than re-opening it.  ``_read_bluesky_source_columns``
        is reached through the instance so an injected/counting reader really
        observes every preview open.
        """
        value = str(path or "")
        if not value:
            return None
        try:
            columns = self._read_bluesky_source_columns(value)
        except Exception:
            logger.debug("preview column read failed for %s", value,
                         exc_info=True)
            columns = None
        if columns is not None:
            motors, counters = columns
            return DirectoryMetadataPreview(
                value, 'embedded',
                tuple(str(name) for name in motors),
                tuple(str(name) for name in counters))
        try:
            if self.meta_ext and self.exists_meta_file(value):
                # A sidecar-described candidate is usable, but its columns are
                # parsed by the shared ``set_pars_from_meta`` projection at
                # adoption; the preview only decides usability + identity.
                return DirectoryMetadataPreview(value, 'sidecar', (), ())
        except Exception:
            logger.debug("preview sidecar probe failed for %s", value,
                         exc_info=True)
        return None

    def _first_usable_preview(self, files, suffixes, match, budget):
        """The first usable candidate among *files*, in the order given.

        R4A-2: an unusable candidate is SKIPPED, never allowed to clear a later
        usable sibling.  Every content open is charged to the whole-attempt
        budget, so the caps bound the complete preview rather than one
        directory.
        """
        for path in files:
            low = path.name.lower()
            suffix = next(
                (value for value in suffixes if low.endswith(value)), "")
            if not suffix:
                continue
            if not match(path.name[:-len(suffix)]):
                continue
            if not budget.charge():
                return None
            result = imageWrangler._metadata_preview_result(self, str(path))
            if result is not None:
                return result
        return None

    def _discover_directory_metadata_preview(self):
        """One usable metadata preview for the selected root, or ``None``.

        O-1b R4A-1/R4A-2 (review §49.3 items 2-3).  Direct children are listed
        NAME-ONLY, natural-sorted globally, and inspected in that order until
        the first usable result.  Only when no direct child is usable, and only
        when Subdirs is enabled, does a lazy deterministic descent follow -- the
        pre-R4-A recursive seed's UX without its eager recursion.  The descent
        stops at the first usable result, eight content opens, or 0.75 s,
        whichever comes first, and those bounds cover the COMPLETE attempt.  No
        frame-count or metadata sweep is performed: name enumeration is
        value-only and only a charged candidate is ever opened.
        """
        try:
            from .image_wrangler_thread import _name_filter

            suffixes = imageWrangler._directory_metadata_preview_suffixes(self)
            if not suffixes:
                return None
            match = _name_filter(self.file_filter)
            root = Path(self.img_dir).expanduser()
            budget = _PreviewBudget(
                deadline=time.monotonic() + _PREVIEW_DEADLINE_S,
                opens=_PREVIEW_MAX_CONTENT_OPENS,
            )
            entries = _directory_preview_entries(root, budget)
            if entries is None:
                # R4A-III: the deadline expired while the root was still being
                # listed.  Abandon here -- inspecting the partial head would be
                # a non-global candidate order, and descending would spend more
                # time past a bound that has already tripped.
                return None
            files, subdirs = entries
            result = imageWrangler._first_usable_preview(
                self, files, suffixes, match, budget)
            if result is not None or not self.include_subdir:
                return result
            pending = list(subdirs)
            while pending and not budget.exhausted():
                child = pending.pop(0)
                child_entries = _directory_preview_entries(child, budget)
                if child_entries is None:
                    return None
                child_files, child_dirs = child_entries
                result = imageWrangler._first_usable_preview(
                    self, child_files, suffixes, match, budget)
                if result is not None:
                    return result
                pending.extend(child_dirs)
            return None
        except Exception:
            logger.debug("directory metadata preview discovery failed",
                         exc_info=True)
            return None

    def _directory_metadata_preview_file(self):
        """The usable preview candidate's PATH, or ``""`` -- compatibility name.

        The identity half of :meth:`_discover_directory_metadata_preview`, kept
        because the Source card and older callers ask only "which file".
        """
        result = imageWrangler._discover_directory_metadata_preview(self)
        # Stash the value half beside the identity half so the caller can adopt
        # THE inspected result instead of re-opening the candidate.
        self._directory_metadata_preview_result = result
        return result.path if result is not None else ""

    def _adopt_directory_metadata_preview(self, preview):
        """Project ONE shared preview result onto the option lists.

        *preview* is the :class:`DirectoryMetadataPreview` discovery already
        produced (so nothing is content-opened twice), ``None``/``""`` for "no
        usable candidate", or -- for older callers -- a bare path string, which
        is resolved through the SAME single usability predicate.
        """
        if isinstance(preview, str):
            preview = imageWrangler._metadata_preview_result(self, preview)
        if preview is not None:
            if preview.kind == 'embedded':
                self.motors = list(preview.motors)
                self.counters = list(preview.counters)
                self.scan_parameters = [*self.motors, *self.counters]
                # A container WAS inspected -> the motor result is
                # authoritative (KNOWN_EMPTY when it has none), not UNKNOWN.
                self._gi_motor_knowledge_proved = True
                self.set_gi_motor_options()
                self.set_bg_matching_options()
                self.set_bg_norm_options()
                return
            prior = self.img_file
            try:
                self.img_file = preview.path
                self.set_pars_from_meta()
            finally:
                self.img_file = prior
            return
        # §13.7: nothing usable was found anywhere within the budget (e.g. a
        # lazy recursive root whose matching data has not landed yet), so the
        # motor knowledge is UNKNOWN, NOT known-empty.  Clearing the stale
        # option lists must not resolve an explicit GI motor to Manual.
        self.scan_parameters = []
        self.motors = []
        self.counters = []
        self._gi_motor_knowledge_proved = False
        self.set_gi_motor_options()
        self.set_bg_matching_options()
        self.set_bg_norm_options()
        imageWrangler._report_no_metadata_preview(self)

    def _on_gi_source_motors(self, value):
        """Translate ONE exact-run-qualified worker discovery to the display.

        R4A-1(iii) (review §49.3 item 5).  The pre-Run preview may legitimately
        have found nothing (a lazy recursive root whose data lands during the
        run), and this is how the dropdown fills anyway.

        Qualification is by ``is`` against BOTH the accepted carrier and the
        admission ledger, so the only value that can be applied is the one
        carrying the exact object this wrapper admitted for the CURRENT run.
        There is deliberately no equality fallback and no "reconstruct the
        current run" path: a foreign or equal-VALUED configuration, a stale
        source generation, a later run, and a post-close arrival (the carrier is
        cleared / the receiver is gone) are all inert.  The per-run latch makes
        it at most once, so a duplicate frame or a re-opened master republishes
        nothing.

        Display-only.  The accepted configuration is never mutated, replaced or
        re-frozen here, and the worker keeps integrating with the motor frozen
        at Run; only the visible option lists move, through the SAME projection
        and hydration chain the preview uses.
        """
        if not isinstance(value, GISourceMotorDiscovery):
            return
        if getattr(self, '_gi_hydration_closed', False):
            return
        frozen = getattr(self, 'run_configuration', None)
        if frozen is None or value.run_configuration is not frozen:
            return
        if getattr(self, '_admitted_run_configuration', None) is not frozen:
            return
        if getattr(self, '_gi_source_motors_applied', None) is frozen:
            return
        self._gi_source_motors_applied = frozen
        self.motors = list(value.motors)
        self.counters = list(value.counters)
        self.scan_parameters = [*self.motors, *self.counters]
        # A real container WAS classified, so the motor knowledge is proved.
        self._gi_motor_knowledge_proved = True
        self.set_gi_motor_options()
        self.set_bg_matching_options()
        self.set_bg_norm_options()

    def _report_no_metadata_preview(self):
        """Tell the OPERATOR that motor choices will arrive during the run.

        R4A-1(iii) (review §49.3 item 4, §50.2).  The hint goes through the
        production ``showLabel`` status seam -- the place the operator actually
        reads -- so a private/test-only field cannot satisfy it.
        """
        emit = getattr(getattr(self, 'showLabel', None), 'emit', None)
        if not callable(emit):
            return
        try:
            emit('No readable scan metadata in this directory yet — GI theta '
                 'motor choices will populate during Run.')
        except Exception:
            logger.debug("showLabel emit failed for the preview hint",
                         exc_info=True)

    def get_img_fname(self):
        """Sets file name based on chosen options
        """
        old_fname = self.img_file
        # O-1a-W1R-D2 (review §42.2): read the source MODE from the authoritative
        # Source card, not the mutable ``self.inp_type`` mirror.  A stale mirror
        # sent a real Image-Series selection down the Directory leg, which clears
        # the runtime cursor.  Preventive authority cleanup: admission already
        # consults the card independently.
        if imageWrangler._authoritative_source_mode(self) != 'Image Directory':
            img_file = self.parameters.child('Signal').child('File').value()
            if os.path.exists(img_file):
                self.img_file = img_file
                _p = Path(self.img_file)
                self.img_dir, self.img_ext = str(_p.parent), _p.suffix.lstrip('.')
                self._sync_meta_ext_to_img_ext()
            else:
                # O-1a-W1R-D1 (review §40.1 P1-A, §40.3 D1 item 1): SYNCHRONIZE
                # on every call.  Assigning only when the configured file exists
                # left the previous selection in ``self.img_file`` whenever the
                # authoritative field went blank or named a missing file -- and
                # ``_on_project_folder_changed`` clears that field by design on an
                # N1 project switch.  ``_inputs_valid`` then trusted the stale
                # instance value, the freeze owner produced no typed source, and
                # the run started against the OLD project's frames.  The
                # neighbouring BUG-1 guard already clears ``poni_file`` for
                # exactly this reason; the source needs the same treatment.
                self.img_file = ''

        else:
            self.img_ext = self.parameters.child('Signal').child('img_ext').value()
            self.img_dir = self.parameters.child('Signal').child('img_dir').value()
            self.include_subdir = self.parameters.child('Signal').child('include_subdir').value()
            try:
                from xrd_tools.sources import DirectorySourceSpec

                spec = getattr(self, "source_spec", None)
                if isinstance(spec, DirectorySourceSpec):
                    self.img_dir = str(spec.root)
                    self.include_subdir = bool(spec.recursive)
                    self.file_filter = str(spec.name_filter or "")
                    if spec.suffixes:
                        suffix = spec.suffixes[0]
                        if suffix.startswith("_master."):
                            self.img_ext = suffix.rsplit(".", 1)[-1]
                        else:
                            self.img_ext = suffix.lstrip(".")
            except Exception:
                logger.debug("directory source intent adoption failed",
                             exc_info=True)
            self._sync_meta_ext_to_img_ext()
            # No representative-file search here.  In particular, Subdirs must
            # not trigger a recursive os.walk or embedded-metadata read merely
            # because the operator selected a directory.
            self.img_file = ''
            try:
                directory = Path(self.img_dir).expanduser()
                directory_stat = directory.stat()
                discovery_key = (
                    str(directory.resolve()),
                    imageWrangler._directory_metadata_preview_suffixes(self),
                    str(self.file_filter or ""),
                    # R4A-1: Subdirs is an INPUT to preview discovery now, so it
                    # belongs in the key.  Without it, ticking Subdirs on a root
                    # whose containers live only in subfolders re-served the
                    # cached "nothing usable" answer and the dropdown never
                    # filled until the root's own mtime happened to change --
                    # exactly the case R4A-1 exists to fix.
                    bool(self.include_subdir),
                    int(getattr(directory_stat, "st_dev", 0)),
                    int(getattr(directory_stat, "st_ino", 0)),
                    int(directory_stat.st_mtime_ns),
                )
            except OSError:
                discovery_key = (
                    str(Path(self.img_dir).expanduser()),
                    imageWrangler._directory_metadata_preview_suffixes(self),
                    str(self.file_filter or ""),
                    bool(self.include_subdir),
                    None,
                )
            if discovery_key == getattr(
                self, "_directory_metadata_discovery_key", None
            ):
                # R4A-2: what is cached is the shared PREVIEW RESULT, not a bare
                # path, so a repeat selection re-adopts without opening the
                # candidate again.
                preview_file = getattr(
                    self, "_directory_metadata_preview_path", "")
                preview = getattr(
                    self, "_directory_metadata_preview_result", None)
            else:
                # Discovery keeps going through the identity leg (callers and
                # focused tests substitute THAT method), and the shared result it
                # stashes is reused so discovery and adoption never open the same
                # candidate twice.  A substituted identity is re-resolved through
                # the SAME single predicate rather than a second one.
                preview_file = self._directory_metadata_preview_file()
                preview = getattr(
                    self, "_directory_metadata_preview_result", None)
                if preview is None or preview.path != preview_file:
                    preview = imageWrangler._metadata_preview_result(
                        self, preview_file)
                self._directory_metadata_discovery_key = discovery_key
                self._directory_metadata_preview_result = preview
                self._directory_metadata_preview_path = preview_file
            try:
                preview_stat = Path(preview_file).stat() if preview_file else None
                preview_stamp = (
                    int(preview_stat.st_size),
                    int(preview_stat.st_mtime_ns),
                ) if preview_stat is not None else None
            except OSError:
                preview_stamp = None
            preview_key = (
                preview_file,
                preview_stamp,
            )
            if (
                preview_key != getattr(
                    self, "_directory_metadata_preview_key", None)
            ):
                self._directory_metadata_preview_key = preview_key
                # §13.6 / §13.11 hydration 2 / §15.12-C.3: open the hydration
                # request token HERE — AFTER the new directory source is applied and
                # ONLY when an adopt (which always emits a GIMotorHydration) will
                # follow — so the request token is captured 1:1 with its emit (never
                # leaking an un-emitted token onto the pending FIFO) and carries the
                # fingerprint of the source the request is FOR.
                self._next_gi_hydration_generation()
                self._adopt_directory_metadata_preview(preview)
            return

        if ((self.img_file != old_fname)
                or (self.img_file and (len(self.scan_parameters) < 1))):
            # §13.6 / §13.11 hydration 2 / §15.12-C.3: open the request token AFTER
            # the single-image / series source is applied above and ONLY on the
            # branch that refreshes options (emits), so bump and emit stay 1:1.
            self._next_gi_hydration_generation()
            if (self.meta_ext and self.img_file
                    and self.exists_meta_file(self.img_file)):
                self.set_pars_from_meta()
            elif (self.img_file
                  and Path(self.img_file).suffix.lower() in _EMBEDDED_META_EXTS
                  and self._read_bluesky_source_columns(self.img_file) is not None):
                # Embedded-metadata source (Bluesky/apstools NXWriter .nxs): no
                # sidecar, but the motors + counters live in the HDF5 tree.
                # Drive the SAME option-refresh path as a sidecar — get_scan_
                # parameters harvests them from the file (see that method).
                self.set_pars_from_meta()
            else:
                # No sidecar metadata (e.g. Eiger) OR no file resolved yet (the
                # source just switched, nothing chosen): clear the previous
                # source's stale motor/parameter options and default the GI Theta
                # Motor to Manual, so the incidence angle can be entered directly.
                # §13.7: a RESOLVED file with no readable motors is a proven
                # KNOWN_EMPTY (Eiger); no file resolved yet is UNKNOWN.
                self.scan_parameters = []
                self.motors = []
                self.counters = []
                self._gi_motor_knowledge_proved = bool(self.img_file)
                self.set_gi_motor_options()
                # Refresh the BG Match + norm-channel dropdowns too, so they don't
                # keep the previous format's columns after a format/source switch
                # (F4 — mirror set_pars_from_meta's option refresh on the clear
                # side; with the lists empty both collapse to just 'None').
                self.set_bg_matching_options()
                self.set_bg_norm_options()

    def set_series_average(self):
        self.series_average = self.parameters.child('Signal').child('series_average').value()

    def set_meta_ext(self):
        self.meta_ext = _normalize_meta_ext(
            self.parameters.child('Signal').child('meta_ext').value()
        )
        # Show "Meta Directory" + Browse only for SPEC mode.  Other
        # formats (auto/txt/pdi/metadata) look next to the image — no separate dir
        # makes sense.  This mirrors the bg_dir pattern.
        is_spec = (self.meta_ext == 'spec')
        self.parameters.child('Signal').child('meta_dir').show(is_spec)
        self._save_to_session()
        # The metadata FORMAT changed: drop the previous format's parsed
        # parameters so get_img_fname re-resolves the metadata (and the GI motor
        # list) for the SAME file under the new format.  Without this, its
        # "img_file unchanged" guard would keep the old format's motors (e.g.
        # switching txt -> pdi when there is no .pdi sidecar should clear them).
        self.scan_parameters = []
        self.get_img_fname()

    def set_meta_dir(self):
        """Opens a directory chooser for the SPEC file's location.

        Sets ``meta_dir`` to the picked path; leaves it alone if the
        user cancels.  Empty string means "use the default search
        (image dir + immediate parent)".
        """
        path = QFileDialog().getExistingDirectory(
            caption='Choose Meta (SPEC) Directory',
            dir=self._browse_dir(self.meta_dir),
            options=QFileDialog.ShowDirsOnly,
        )
        if path:
            remember_browse_path(path)
            self.parameters.child('Signal').child('meta_dir').setValue(path)
            self.meta_dir = path

    def _sync_meta_ext_to_img_ext(self):
        """Keep the Meta File row visible and NEVER touch its value.

        This used to force meta_ext='none' and hide the row when the image
        type was NeXus (on the theory that .nxs embeds its metadata).  Live
        bl17-2 (maintainer, 2026-07-13): that silently overwrote the user's
        'auto' after every run, and weeks of force-writes contaminated saved
        sessions with a 'none' nobody chose.  The rule now: the value is the
        USER'S — 'auto' by default, changed only by an explicit edit — and
        the row stays visible so its state is never hidden while it still
        matters ('auto' additionally probes sidecars, harmless for nxs; the
        embedded Bluesky metadata is read regardless of this setting).

        setOpts, NOT show(): pyqtgraph's show/hide emit sigOptionsChanged
        UNCONDITIONALLY (even when visible is unchanged), and this runs
        inside setup() which is wired to sigTreeStateChanged — an
        unconditional emit is an infinite setup() recursion (RecursionError
        at app start, Jun 10).  setOpts skips unchanged values.
        """
        meta_param = self.parameters.child('Signal').child('meta_ext')
        meta_param.setOpts(visible=True)

    def exists_meta_file(self, img_file):
        """Checks for existence of meta file for image file"""
        meta_ext = _normalize_meta_ext(getattr(self, 'meta_ext', None))
        if not meta_ext:
            return False
        if meta_ext == 'auto':
            return bool(read_image_metadata(
                img_file, meta_format='auto', meta_dir=self.meta_dir,
            ))
        if meta_ext != 'spec':
            meta_files = [
                f'{os.path.splitext(img_file)[0]}.{meta_ext}',
                f'{img_file}.{meta_ext}'
            ]
            if (_path_exists_case_insensitive(meta_files[0])
                    or _path_exists_case_insensitive(meta_files[1])):
                return True
        else:
            spec_fname, _, _ = _extract_scan_info(Path(img_file))
            if spec_fname:
                img_fpath = Path(img_file)
                # Honour the optional explicit Meta Directory before
                # falling back to the image dir + parent search, so
                # the existence check matches what ``read_image_metadata``
                # will actually look at when SPEC is selected.
                search_dirs = []
                if getattr(self, 'meta_dir', None):
                    search_dirs.append(Path(self.meta_dir))
                search_dirs.extend([img_fpath.parent, img_fpath.parents[1]])
                for parent in search_dirs:
                    if (parent / spec_fname).is_file():
                        return True

        return False

    def detect_meta_ext(self, img_file):
        """Auto-detect metadata sidecar format for *img_file*.

        Probes for ``.txt`` and ``.pdi`` sidecar files.  If found, updates
        the GUI parameter and ``self.meta_ext``.  Returns the detected
        extension string or ``None``.
        """
        base = os.path.splitext(img_file)[0]
        for ext in ('txt', 'pdi'):
            if (_path_exists_case_insensitive(f'{base}.{ext}')
                    or _path_exists_case_insensitive(f'{img_file}.{ext}')):
                # Update the GUI dropdown so the user sees the change
                param = self.parameters.child('Signal').child('meta_ext')
                param.setValue(ext)          # fires set_meta_ext automatically
                return ext
        return None

    def set_pars_from_meta(self):
        self.get_scan_parameters()
        self.set_bg_matching_options()
        # A targeted metadata read ran -> the motor result is authoritative
        # (KNOWN_EMPTY when it found none), not UNKNOWN (§13.7).
        self._gi_motor_knowledge_proved = True
        self.set_gi_motor_options()
        self.set_bg_norm_options()

    def set_mask_file(self):
        """Opens file dialogue and sets the mask file
        """
        fname, _ = QFileDialog().getOpenFileName(
            dir=self._browse_dir(self.mask_file),
            filter="Mask files (*.edf *.npy);;EDF (*.edf);;NumPy (*.npy)"
        )
        if fname != '':
            remember_browse_path(fname)
            self.parameters.child('Signal').child('mask_file').setValue(fname)
            self.mask_file = fname

    def set_bg_type(self):
        """Change Parameter Names depending on BG Type
        """
        for child in self.parameters.child('BG').children():
            child.hide()
        self.parameters.child('BG').child('bg_type').show()

        self.bg_type = self.parameters.child('BG').child('bg_type').value()
        if self.bg_type == 'None':
            return
        elif self.bg_type != 'BG Directory':
            self.parameters.child('BG').child('File').show()
            if self.bg_type == 'Single BG File':
                opts = {'title': 'File Name'}
            else:
                opts = {'title': 'First File'}
            self.parameters.child('BG').child('File').setOpts(**opts)
        else:
            self.parameters.child('BG').child('Match').show()

        self.parameters.child('BG').child('Scale').show()
        self.parameters.child('BG').child('norm_channel').show()

    def set_bg_file(self):
        """Opens file dialogue and sets the background file
        """
        fname, _ = QFileDialog().getOpenFileName(
            dir=self._browse_dir(self.bg_file),
            filter=f"Images (*.{self.img_ext})"
        )
        if fname != '':
            remember_browse_path(fname)
            self.parameters.child('BG').child('File').setValue(fname)
            self.bg_file = fname

    def set_bg_dir(self):
        """Opens file dialogue and sets the background folder
        """
        path = QFileDialog().getExistingDirectory(
            caption='Choose BG Directory',
            dir=self._browse_dir(self.bg_dir),
            options=QFileDialog.ShowDirsOnly
        )
        if path != '':
            remember_browse_path(path)
            self.parameters.child('BG').child('Match').child('bg_dir').setValue(path)
            self.bg_dir = path

    def set_h5_dir(self):
        """Opens file dialogue and sets the path where processed data is stored
        """
        path = QFileDialog().getExistingDirectory(
            caption='Choose Save Directory',
            dir=self._browse_dir(self.h5_dir),
            options=QFileDialog.ShowDirsOnly
        )
        if path != '':
            remember_browse_path(path)
            Path(path).mkdir(parents=True, exist_ok=True)
            self.parameters.child('Project').child('h5_dir').setValue(path)
            self._sync_h5_dir_from_parameters()

    def _compute_source_base(self):
        """N1: the absolute project root, or None when the Project Folder is
        blank or invalid (-> the writer stores absolute raw paths, back-compat)."""
        pf = (self.parameters.child('Project').child('project_folder').value() or '').strip()
        if not pf:
            return None
        path = os.path.abspath(os.path.expanduser(pf))
        return path if os.path.isdir(path) else None

    def _default_h5_under_project(self):
        """Default the Save Path to ``<project>/xdart_processed_data`` when the
        user hasn't chosen one (blank or still the app default)."""
        base = self._compute_source_base()
        if not base:
            return
        cur_h5 = (self.parameters.child('Project').child('h5_dir').value() or '').strip()
        if not cur_h5 or cur_h5 == get_fname_dir():
            self.parameters.child('Project').child('h5_dir').setValue(
                os.path.join(base, 'xdart_processed_data'))
            self._sync_h5_dir_from_parameters()

    def set_project_folder(self):
        """Browse for the N1 Project Folder.  Setting it stores raw source paths
        RELATIVE to this root (portable .nxs); the value-change handler
        (:meth:`_on_project_folder_changed`) then resets the dependent
        (folder-relative) paths + defaults the Save Path."""
        path = QFileDialog().getExistingDirectory(
            caption='Choose Project Folder',
            dir=self._browse_dir(self.project_folder),
            options=QFileDialog.ShowDirsOnly
        )
        if path != '':
            remember_browse_path(path)
            self.parameters.child('Project').child('project_folder').setValue(path)

    def _on_project_folder_changed(self, *args):
        """N1 Decision 2: a Project Folder change INVALIDATES everything stored
        relative to the OLD root.  Recompute ``source_base``, clear the PONI +
        the dependent source paths (which cascades back to the enter-PONI
        disclosure state via :meth:`get_poni_dict`), and default the Save Path
        under the new folder.  Inert during a session restore (the ``_restoring``
        guard) so it doesn't wipe the values the restore is setting."""
        if getattr(self, '_restoring', False):
            return
        self.source_base = self._compute_source_base()
        if getattr(self, 'thread', None) is not None:
            self.thread.source_base = self.source_base
        # Clear the PONI (cascades through get_poni_dict -> _apply_disclosure) and
        # the source paths that were relative to the now-stale old root.
        # IMPORTANT: get_poni_dict reads the INSTANCE attr self.poni_file, not the
        # param value, so resync it FIRST -- else the cascade re-loads the stale
        # PONI (the old path still exists on disk) and _inputs_valid stays True,
        # letting a Start run the new folder's images against the old calibration
        # (the BUG-1 this reset exists to prevent).
        self.poni_file = ''
        # O-1a-W1R-D1 (review §40.3 D1 item 1): invalidate the runtime SOURCE
        # cursor here too, for the same reason the calibration is invalidated
        # above -- the old file still exists on disk, so nothing downstream could
        # tell that it belongs to the previous project.
        self.img_file = ''
        self.parameters.child('Signal').child('poni_file').setValue('')
        for seg in (('Signal', 'File'), ('Signal', 'img_dir'),
                    ('Signal', 'mask_file')):
            try:
                self.parameters.child(*seg).setValue('')
            except (AttributeError, KeyError, TypeError):
                pass
        # The Save Path is project-relative too: clear it so the default
        # helper re-points it under the NEW folder (its keep-user-value
        # guard otherwise retains the OLD project's path on a switch, and
        # the scans browser never followed).
        self.parameters.child('Project').child('h5_dir').setValue('')
        self._default_h5_under_project()
        self._apply_disclosure()

    def set_bg_matching_options(self):
        """Reads image metadata to populate matching parameters
        """
        pars = [p for p in self.scan_parameters if not any(x.lower() in p.lower() for x in ['ROI', 'PD'])]
        pars.insert(0, 'None')
        if 'TEMP' in pars:
            pars.insert(1, pars.pop(pars.index('TEMP')))

        value = 'None'
        opts = {'values': pars, 'limits': pars, 'value': value}
        self.parameters.child('BG').child('Match').child('Parameter').setOpts(**opts)

    def set_bg_matching_par(self):
        """Changes bg matching parameter
        """
        self.bg_matching_par = self.parameters.child('BG').child('Match').child('Parameter').value()
        if self.bg_matching_par == 'None':
            self.bg_matching_par = None

    def set_bg_norm_options(self):
        """Counter Values used to normalize and subtract background
        """
        pars = self.counters
        pars.insert(0, 'None')

        opts = {'values': pars, 'limits': pars, 'value': 'None'}
        self.parameters.child('BG').child('norm_channel').setOpts(**opts)

    def set_bg_norm_channel(self):
        """Changes bg matching parameter
        """
        self.bg_norm_channel = self.parameters.child('BG').child('norm_channel').value()

    def set_gi_motor_options(self):
        """Reads image metadata to populate possible GI theta motor.

        Always offers a 'Manual' option (enter the incidence angle directly
        via the Theta field).  When no motors are found — e.g. Eiger / no
        metadata — Manual is the default, since there's no ``th`` to read.
        """
        pars = [p for p in self.motors if not any(x.lower() in p.lower() for x in ['ROI', 'PD'])]
        # Default-select by the shared policy: named preference (th/eta/halpha/
        # gonth/theta) if present, else a rotation-sounding motor, else Manual.
        # Matches the integrator's gi_motor combo so the two never disagree.
        value = pick_default_gi_motor(pars)

        pars = ['Manual'] + pars

        opts = {'values': pars, 'limits': pars, 'value': value}
        self.parameters.child('GI').child('th_motor').setOpts(**opts)
        # Sync the Theta-value field visibility + incidence_motor to the
        # (possibly newly-defaulted) selection — setOpts may not re-fire
        # sigValueChanged when the value is set programmatically.
        self.set_gi_th_motor()
        # GI move (Stage B): hand the available motor columns (excl. Manual) to
        # the integrator panel's GI motor dropdown, which owns the selection.
        # §13.6/§13.7: emit a source-qualified GIMotorHydration, not a bare list.
        # ``_gi_motor_knowledge_proved`` is set by the discovery caller to record
        # whether a TARGETED inspection ran (so an empty list is KNOWN_EMPTY only
        # when proved, else UNKNOWN — a lazy recursive directory is never
        # misclassified as known-empty).
        # ``_gi_motor_knowledge_proved`` is True on a fresh wrangler and after a
        # direct establishment of the motor list (a bare ``set_gi_motor_options``
        # / session re-announce with an explicit empty list is KNOWN_EMPTY).  The
        # §13.7 "not inspected -> UNKNOWN" cases (no direct-child preview / no
        # file resolved) set it False EXPLICITLY before calling.  §13.11
        # hydration 5: the getattr FALLBACK is inverted to False, so any future
        # emit path that never established knowledge fails safe to UNKNOWN rather
        # than silently reporting KNOWN_EMPTY.
        _avail = [p for p in self.motors
                  if not any(x.lower() in p.lower() for x in ['ROI', 'PD'])]
        _proved = bool(getattr(self, '_gi_motor_knowledge_proved', False))
        # §21.4 req 4-5: the synchronous image discovery paths OPEN a request in
        # get_img_fname and complete it HERE, so the request's own token is
        # threaded through `_gi_hydration_open_token` rather than re-derived from
        # completion order.  With no open request this is a direct re-announcement
        # of the current source, which goes through the structurally separate
        # announce API and consumes no request.  Read as an ATTRIBUTE so a
        # duck-typed host that binds only the emit methods still works.
        _token = getattr(self, '_gi_hydration_open_token', None)
        if _token is not None:
            self._emit_gi_hydration(_avail, proved=_proved, token=_token)
        else:
            self._announce_gi_hydration(_avail, proved=_proved)

    def set_gi_th_motor(self):
        """Update Grazing theta motor.

        Reveals the Manual 'Theta' value field only when Theta Motor is
        'Manual', and hides it otherwise.  Eiger / metadata-less files have
        no ``th`` to read, so without this input field the Manual path can't
        supply an incidence angle and the GI cake stays blank.  Use
        ``setOpts(visible=...)`` (not hide()/show()) so the param-tree row
        reliably re-renders.
        """
        th_motor = self.parameters.child('GI').child('th_motor').value()
        th_val = self.parameters.child('GI').child('th_val')
        if th_motor == 'Manual':
            th_val.setOpts(visible=True)
            self.incidence_motor = th_val.value()
        else:
            th_val.setOpts(visible=False)
            self.incidence_motor = th_motor

    def _read_bluesky_source_columns(self, img_file):
        """Return ``(motor_names, counter_names)`` when *img_file* is a Bluesky/
        apstools ``NXWriter`` ``.nxs`` whose scan motors + counters are embedded
        in the HDF5 tree; ``None`` for any other source.

        The image half of a ``.nxs`` already reads via ``read_image``; this
        surfaces the embedded metadata the sidecar reader can't see — motors ->
        the GI Theta-Motor dropdown, counters -> Normalize / BG Match.  Defensive
        by contract: a non-``.nxs``/``.h5`` extension, a non-Bluesky HDF5 (plain
        NeXus, xdart-processed, or a bare Eiger master), a locked/corrupt file,
        or a missing reader all return ``None`` so the historical sidecar path is
        completely unaffected.  Cached on ``(path, mtime, size)`` so the
        detection in :meth:`get_img_fname` and the harvest in
        :meth:`get_scan_parameters` share one open per file selection.
        """
        if not img_file:
            return None
        if Path(img_file).suffix.lower() not in _EMBEDDED_META_EXTS:
            return None
        try:
            st = os.stat(img_file)
            key = (os.path.abspath(img_file), st.st_mtime, st.st_size)
        except OSError:
            return None
        cache = getattr(self, '_bluesky_cols_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]
        result = None
        try:
            import h5py
            from xrd_tools.io.bluesky_nexus import (
                bluesky_all_motor_names,
                bluesky_counters,
                is_bluesky_nxwriter,
                resolve_nxentry,
            )
            with h5py.File(img_file, 'r') as h5:
                entry = resolve_nxentry(h5)
                if entry is not None and is_bluesky_nxwriter(entry):
                    # ALL motors (scanned + baseline-fixed), so the GI θ-motor
                    # dropdown offers every real motor of the file — beamlines use
                    # oddly-named incidence axes; the default-pick chooses among
                    # them (see set_gi_motor_options / pick_default_gi_motor).
                    motors = list(bluesky_all_motor_names(entry))
                    counters = list(bluesky_counters(entry).keys())
                    result = (motors, counters)
        except Exception:
            logger.debug("Bluesky source column read failed for %s",
                         img_file, exc_info=True)
            result = None
        self._bluesky_cols_cache = (key, result)
        return result

    def get_scan_parameters(self):
        """ Reads image metadata to populate matching parameters
        """
        if not self.img_file:
            return
        # A Bluesky/apstools NXWriter .nxs embeds its scan motors + counters in
        # the HDF5 tree (no sidecar).  Harvest them directly so the GI Theta
        # Motor, Normalize, and BG-Match dropdowns show the file's real columns
        # (e.g. 'hy', 'i0'..'pd') instead of collapsing to 'th'/'Manual'.
        # Guarded: a non-Bluesky / unreadable / non-HDF5 source returns None and
        # the sidecar path below is byte-identical to before.  The extension
        # fast-path means non-HDF5 sources never invoke the probe at all.
        bluesky_cols = None
        if Path(self.img_file).suffix.lower() in _EMBEDDED_META_EXTS:
            bluesky_cols = self._read_bluesky_source_columns(self.img_file)
        if bluesky_cols is not None:
            motors, counters = bluesky_cols
            self.motors = list(motors)
            self.counters = list(counters)
            self.scan_parameters = list(motors) + list(counters)
            return
        if not self.meta_ext:
            # GUI "none"/blank means metadata OFF.  Do not pass Python None to
            # read_image_metadata here: the headless API deliberately treats
            # None as "auto" for notebook/script callers.
            self.scan_parameters = []
            self.counters = []
            self.motors = []
            return

        # Pass meta_dir so SPEC metadata (which often lives in a separate Meta
        # Directory) is found here too -- the worker threads already do.  Without
        # it the SPEC read returns {} and the GI motor dropdown collapses to its
        # 'th'/'Manual' placeholder.  meta_dir is ignored for txt/pdi sidecars.
        img_meta = read_image_metadata(self.img_file, meta_format=self.meta_ext,
                                       meta_dir=self.meta_dir)
        self.scan_parameters = list(img_meta.keys())
        self.counters = self.scan_parameters
        self.motors = self.scan_parameters

    def enabled(self, enable):
        """Enable/disable the WHOLE wrangler panel for the run lifecycle (#72).

        During a run everything is locked except Stop: the parameter tree is
        hard-disabled (greyed + fully non-interactive, matching the integration
        panel above it), and the non-param widgets (processing-mode combo, Cores
        spinbox + label, Advanced button) are disabled too.  A disabled pyqtgraph
        bool checkbox (Grazing, Average Scan, …) may repaint unchecked during the
        run (#56), but the value is preserved and restored on re-enable — the
        full visible disable was chosen over that cosmetic ("minimize
        complexity").  The running thread uses the setup-time arg snapshot
        regardless.

        args:
            enable: bool, True for enabled False for disabled.
        """
        self.tree.setEnabled(enable)
        # Phase B: the action button stays ENABLED during a run — it is the
        # Pause/Resume control now (morphed by _set_action_button), not just
        # Start.  Only its label/colour changes across the run lifecycle.
        self.ui.startButton.setEnabled(True)
        # Non-param widgets (live outside the ParameterTree): mode combo, Cores
        # spinbox + label, Advanced button.  Stop is left alone (stays enabled).
        for name in ('processingModeCombo', 'maxCoresSpinBox', 'coresLabel',
                     'advancedButton'):
            w = getattr(self.ui, name, None)
            if w is not None:
                w.setEnabled(enable)
        # Live/Batch toggle state vs. the run lifecycle:
        if enable:
            # Run finished — reset the action button to green 'Start' and
            # re-enable both mode toggles.  Reset Live to off (no re-trigger).
            self._set_action_button('idle')
            self.ui.liveCheckBox.blockSignals(True)
            self.ui.liveCheckBox.setChecked(False)
            self.ui.liveCheckBox.blockSignals(False)
            self.ui.liveCheckBox.setEnabled(True)
            self.ui.batchCheckBox.setEnabled(True)
            # Uncheck is signal-blocked → sync the flag (see stop()).
            self.live_mode = False
            # Re-assert per-mode widget state (cores/labels/toggles, viewer
            # dimming) now that the run lock is lifted.
            self._on_mode_changed()
        else:
            # Run active — Live/Batch are pure mode toggles and lock for the
            # run's duration (the morphing action button + Stop drive it now,
            # so Live no longer needs to stay clickable to stop a live run).
            self.ui.liveCheckBox.setEnabled(False)
            self.ui.batchCheckBox.setEnabled(False)

    def stylize_ParameterTree(self):
        # Uniform rows (no light/dark stripe) so the panel reads cleanly.  The
        # group-header band + field tint now come from the GLOBAL QSS, scoped by
        # the tree's object name (QTreeView#WranglerTree in themes/dark.py), so
        # the wrangler tree themes correctly in both Dark AND Light and recolours
        # on a live theme switch — the old widget-local Dracula stylesheet broke
        # under the Light palette (Stage-2 deferral, closed here).
        self.tree.setAlternatingRowColors(False)
