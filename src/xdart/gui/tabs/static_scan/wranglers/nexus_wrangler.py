# -*- coding: utf-8 -*-
"""
nexusWrangler — GUI widget for processing NeXus/HDF5 image stacks.

Provides a simple interface for selecting a NeXus file containing
image frames (e.g., from Bluesky suitcase-nexus exports or Eiger
master files) and a PONI calibration file.

@author: thampy
"""

# Standard library imports
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Used to size the Cores spinbox.  os.cpu_count() returns None on
# exotic platforms; the fallback default mirrors what the SPEC
# wrangler does for the same widget.
_CPU_COUNT = os.cpu_count() or 4

# Qt imports
from pyqtgraph.Qt import QtWidgets, QtCore
from pyqtgraph.parametertree import ParameterTree, Parameter

# Project imports
from xrd_tools.core.containers import PONI
from .wrangler_widget import wranglerWidget
from .nexus_wrangler_thread import nexusThread
from xdart.utils import get_fname_dir
from xdart.utils.session import load_session, save_session

QFileDialog = QtWidgets.QFileDialog

params = [
    # N1: the portable Project Folder.  Setting it stamps entry/@source_base and
    # stores each frame's raw source path RELATIVE to it (portable .nxs); blank
    # -> absolute paths (back-compat).  Mirrors the image wrangler (the fuller
    # progressive-disclosure UX is image-wrangler only; here it's the portable
    # field + wiring).
    {'name': 'Project', 'title': 'Project Folder', 'type': 'group', 'children': [
        # str_browse: path + inline Browse (group header already says the name).
        {'name': 'project_folder', 'title': '', 'type': 'str_browse', 'value': ''},
    ], 'expanded': True},
    {'name': 'Calibration', 'type': 'group', 'children': [
        {'name': 'poni_file', 'title': '', 'type': 'str_browse', 'value': ''},
    ], 'expanded': True},
    {'name': 'NeXus File', 'type': 'group', 'children': [
        {'name': 'nexus_file', 'title': 'File         ', 'type': 'str_browse', 'value': ''},
        {'name': 'entry', 'title': 'Entry', 'type': 'str', 'value': 'entry'},
    ], 'expanded': True},
    {'name': 'Signal', 'type': 'group', 'children': [
        {'name': 'mask_file', 'title': 'Mask File    ', 'type': 'str_browse', 'value': ''},
    ], 'expanded': False},
    # R3-B: detector-saturation masking opt-out for NeXus sources too.  Same
    # UI-1 header-checkbox pattern as GI/the image wrangler's MaskSat: the group
    # header carries a real checkbox (see wranglerWidget._install_group_toggles);
    # the hidden bool is the source of truth.  ON by default preserves the
    # long-standing behaviour; OFF lets a genuinely-saturated module at the
    # integer ceiling be integrated.  Without this, NeXus scans were forced
    # mask_sentinel=True with no way to opt out.
    # Mask Saturated moved to the integrator panel — hidden carrier the
    # integrator injects into at run-setup (setup() still reads mask_sentinel).
    {'name': 'MaskSat', 'title': 'Mask Saturated', 'type': 'group',
     'children': [
        {'name': 'mask_sentinel', 'type': 'bool', 'value': True,
         'visible': False},
    ], 'expanded': False, 'visible': False},
    {'name': 'GI', 'title': 'Grazing Incidence', 'type': 'group',
     'children': [
        # UI-1 (#81): the group HEADER carries a real checkbox — the on/off
        # toggle (see wranglerWidget._install_group_toggles).  The bool is the
        # hidden source of truth (a hidden bool can't repaint-uncheck when the
        # tree is disabled mid-run, #56).
        {'name': 'Grazing', 'title': 'Grazing Incidence', 'type': 'bool',
         'value': False, 'visible': False},
        {'name': 'th_motor', 'title': 'Theta Motor', 'type': 'str', 'value': 'th'},
        {'name': 'th_val', 'title': 'Theta Value', 'type': 'float', 'value': 0.0},
        {'name': 'sample_orientation', 'title': 'Sample Orientation', 'type': 'int',
         'value': 4, 'limits': (1, 4)},
        {'name': 'tilt_angle', 'title': 'Tilt Angle', 'type': 'float', 'value': 0.0},
        # gi_mode_1d / gi_mode_2d are controlled by the integrator panel;
        # kept here for session persistence only.
        {'name': 'gi_mode_1d', 'title': '1D Mode', 'type': 'list',
         'values': ['q_total', 'q_ip', 'q_oop', 'exit_angle'],
         'value': 'q_total', 'visible': False},
        {'name': 'gi_mode_2d', 'title': '2D Mode', 'type': 'list',
         'values': ['qip_qoop', 'q_chi', 'exit_angles'],
         'value': 'qip_qoop', 'visible': False},
    ], 'expanded': False, 'visible': False},  # hidden carrier: GI owned by the integrator panel
    {'name': 'Output', 'type': 'group', 'children': [
        {'name': 'h5_dir', 'title': 'Output Dir   ', 'type': 'str_browse', 'value': ''},
    ], 'expanded': False},
]


class nexusWrangler(wranglerWidget):
    """Widget for processing NeXus/HDF5 image stacks.

    A simpler alternative to imageWrangler for data already stored
    in NeXus format (e.g. from Bluesky).

    signals:
        showLabel: str, status text
    """
    showLabel = QtCore.Signal(str)

    def __init__(self, fname, file_lock, scan, data_1d, data_2d, parent=None):
        super().__init__(fname, file_lock, parent)

        self.poni = None
        self.command = None
        self.scan = scan
        self.data_1d = data_1d
        self.data_2d = data_2d

        # Attributes
        self.nexus_file = ''
        self.entry = 'entry'
        # N1: Project Folder (portable @source_base).  Blank -> None (absolute).
        self.project_folder = ''
        self.source_base = None
        self.poni_file = ''
        self.mask_file = ''
        self.h5_dir = get_fname_dir()
        self.gi = False
        self.incidence_motor = 'th'
        self.sample_orientation = 4
        self.tilt_angle = 0.0
        self.gi_mode_1d = 'q_total'
        self.gi_mode_2d = 'qip_qoop'
        # Per-mode flags reflected from the processingModeCombo
        # selection (see :meth:`_on_mode_changed`).  Threaded down to
        # the worker in :meth:`start` so the user can change mode
        # between runs without restarting the app.
        self.xye_only = False

        # ── Build UI programmatically ────────────────────────────────
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Status label.  Long thread messages (missing-dataset errors etc.)
        # must not force the window wider — elide into the label, full text
        # in the tooltip (see wranglerWidget._guard_status_label).
        self.statusLabel = QtWidgets.QLabel('Ready')
        layout.addWidget(self.statusLabel)
        self._guard_status_label()
        self.showLabel.connect(self._set_status_text)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.startButton = QtWidgets.QPushButton('Start')
        # N8: object names enable the dark-theme QSS to colour
        # these primary CTAs (green/red).  See xdart/gui/themes/dark.py.
        self.startButton.setObjectName('startButton')
        self.stopButton = QtWidgets.QPushButton('Stop')
        self.stopButton.setObjectName('stopButton')

        # Processing-mode dropdown — same items as imageWrangler so the
        # two paths feel interchangeable.  Selecting "Int 1D" or
        # "Int 1D (XYE)" skips 2D integration (faster batches on big
        # detectors).  "Int 1D (XYE)" also bypasses the HDF5 write
        # entirely — useful when the user only wants per-frame XYE
        # files (e.g. handing off to downstream tools that don't read
        # our .nxs schema).
        self.modeLabel = QtWidgets.QLabel('Mode:')
        self.processingModeCombo = QtWidgets.QComboBox()
        self.processingModeCombo.addItems([
            'Int 1D + 2D',
            'Int 1D',
            'Int 1D (XYE)',
        ])

        # Cores spinbox — controls how many worker threads the
        # nexusThread spawns for parallel batch integration.  Same
        # convention as imageWrangler.maxCoresSpinBox.  Pushed down
        # to ``self.thread.max_cores`` before each scan starts (see
        # :meth:`start`).  Default = min(CPU-1, 4) so we don't
        # saturate a busy laptop by default.
        self.coresLabel = QtWidgets.QLabel('Cores:')
        self.maxCoresSpinBox = QtWidgets.QSpinBox()
        self.maxCoresSpinBox.setMinimum(1)
        self.maxCoresSpinBox.setMaximum(_CPU_COUNT)
        self.maxCoresSpinBox.setValue(min(_CPU_COUNT - 1, 4) or 1)

        btn_layout.addWidget(self.startButton)
        btn_layout.addWidget(self.stopButton)
        btn_layout.addWidget(self.modeLabel)
        btn_layout.addWidget(self.processingModeCombo)
        btn_layout.addWidget(self.coresLabel)
        btn_layout.addWidget(self.maxCoresSpinBox)
        # Wrap the legacy per-wrangler controls in a frame so attach_controls()
        # can hide them as one — the run controls are the shared StaticControls
        # widget now (Stage-3 cleanup deletes this build entirely).
        self._legacy_controls = QtWidgets.QFrame()
        self._legacy_controls.setLayout(btn_layout)
        layout.addWidget(self._legacy_controls)

        # Signal wiring moved to attach_controls() (shared controls).  Seed the
        # processing-mode flags once from the init-time local combo.
        self._on_mode_changed(self.processingModeCombo.currentText())

        self.stopButton.setEnabled(False)

        # Parameter tree
        self.tree = ParameterTree()
        self.parameters = Parameter.create(
            name='nexus_wrangler', type='group', children=params
        )
        self.tree.setParameters(self.parameters, showTop=False)
        # Hide the "Parameter / Value" header bar (parity with the image
        # wrangler) — visual noise above the tree.
        self.tree.setHeaderHidden(True)
        # Shallow indent (parity with the image wrangler): grouped param names
        # sit close under the top-level rows and nearer the left edge, giving
        # the value boxes more room.  Chevrons need the column so it stays > 0.
        self.tree.setIndentation(3)
        # Match the image wrangler's tree styling so the group headers (and
        # their expand/collapse arrows) are legible against the Dracula theme
        # -- the nexus tree previously had no styling at all (UI-4).  Uniform
        # dark rows (no stripe), lighter input boxes; group headers keep their
        # band.  Scoped to this tree (left scans list keeps its striping).
        self.tree.setAlternatingRowColors(False)
        self.tree.setStyleSheet("""
        QTreeView { alternate-background-color: #21222c; }
        QTreeView::item:has-children {
            background-color: #44475a;
            color: #f8f8f2;
        }
        QTreeView::item:has-children:disabled {
            background-color: #3a3d4d;
            color: #6272a4;
        }
        QLineEdit, QComboBox, QAbstractSpinBox {
            background-color: #4a4f63;
            color: #f8f8f2;
        }
            """)
        layout.addWidget(self.tree)

        # Connect parameter browse buttons
        self.parameters.child('Project').child('project_folder').sigActivated.connect(
            self.browse_project_folder
        )
        self.parameters.child('Calibration').child('poni_file').sigActivated.connect(
            self.browse_poni
        )
        self.parameters.child('NeXus File').child('nexus_file').sigActivated.connect(
            self.browse_nexus
        )
        self.parameters.child('Signal').child('mask_file').sigActivated.connect(
            self.browse_mask
        )
        self.parameters.child('Output').child('h5_dir').sigActivated.connect(
            self.browse_h5_dir
        )

        # UI-1 (#81): put a real checkbox on the GI group header — the
        # checkbox is the on/off toggle, driving the hidden 'Grazing' bool
        # that setup() reads at start (see wranglerWidget._install_group_toggles).
        self._install_group_toggles(self.tree)

        # Setup thread
        self.thread = nexusThread(
            self.command_queue,
            self.scan_args,
            self.file_lock,
            self.fname,
            self.nexus_file,
            self.poni,
            self.mask_file,
            self.gi,
            self.incidence_motor,
            self.sample_orientation,
            self.tilt_angle,
            self.gi_mode_1d,
            self.gi_mode_2d,
            self.command,
            self.scan,
            self.data_1d,
            self.data_2d,
            entry=self.entry,
            parent=self,
        )
        self.thread.showLabel.connect(self._set_status_text)
        self.thread.sigUpdateFile.connect(self.sigUpdateFile.emit)
        self.thread.finished.connect(self.finished.emit)
        self.thread.sigUpdate.connect(self.sigUpdateData.emit)
        self.thread.sigUpdateGI.connect(self.sigUpdateGI.emit)

        self._restore_from_session()

    # ── Session persistence ──────────────────────────────────────────

    _SESSION_PARAMS = [
        ('project_folder',      ('Project', 'project_folder'),   True,  'project_folder'),
        ('poni_file',           ('Calibration', 'poni_file'),    True,  'poni_file'),
        ('nexus_file',          ('NeXus File', 'nexus_file'),    True,  'nexus_file'),
        ('entry',               ('NeXus File', 'entry'),         False, 'entry'),
        ('mask_file',           ('Signal', 'mask_file'),         True,  'mask_file'),
        ('h5_dir',              ('Output', 'h5_dir'),            True,  'h5_dir'),
        ('gi',                  ('GI', 'Grazing'),               False, 'gi'),
        ('sample_orientation',  ('GI', 'sample_orientation'),    False, 'sample_orientation'),
        ('tilt_angle',          ('GI', 'tilt_angle'),            False, 'tilt_angle'),
        ('gi_mode_1d',          ('GI', 'gi_mode_1d'),            False, 'gi_mode_1d'),
        ('gi_mode_2d',          ('GI', 'gi_mode_2d'),            False, 'gi_mode_2d'),
        ('mask_sentinel',       ('MaskSat', 'mask_sentinel'),    False, 'mask_sentinel'),
    ]

    def _restore_from_session(self):
        """Restore parameters from session.json."""
        data = load_session()
        if not data:
            return
        for skey, param_path, is_path, attr in self._SESSION_PARAMS:
            val = data.get(skey)
            if val is None:
                continue
            if is_path and not os.path.exists(val):
                continue
            try:
                p = self.parameters
                for name in param_path:
                    p = p.child(name)
                p.setValue(val)
                if attr:
                    setattr(self, attr, val)
            except (KeyError, AttributeError, ValueError, TypeError) as e:
                # KeyError: parameter tree was renamed since the session
                # was saved.  Other types: value-coercion mismatches.
                # All are non-fatal — the unrestored param just keeps
                # its default — but we want them visible at debug.
                logger.debug(
                    "session restore failed for %s: %s", skey, e,
                )

        # UI-1 (#81): reflect the restored GI on/off into the group's expanded
        # state (the header checkbox itself is synced via the hidden bool's
        # sigValueChanged) -- expand when on, collapse when off.
        try:
            gi_on = bool(self.parameters.child('GI').child('Grazing').value())
            self.parameters.child('GI').setOpts(expanded=gi_on)
        except Exception:
            logger.debug("GI expand-restore skipped", exc_info=True)

        # Restore PONI
        poni_file = self.parameters.child('Calibration').child('poni_file').value()
        if poni_file and os.path.exists(poni_file):
            self.poni = PONI.from_poni_file(poni_file)

        # Cores spinbox isn't in the parameter tree — restore the
        # last selected worker count directly.  Clamp to the
        # spinbox's [min, max] in case the user opened on a smaller
        # machine than the one that saved the session.
        cores = data.get('max_cores')
        if isinstance(cores, int):
            cores = max(self.maxCoresSpinBox.minimum(),
                        min(cores, self.maxCoresSpinBox.maximum()))
            self.maxCoresSpinBox.setValue(cores)

        # Restore processing mode.  ``findText`` returns -1 if the
        # saved label no longer exists in the combo (e.g. we renamed
        # an option), in which case we silently fall back to the
        # default (index 0).
        mode = data.get('processing_mode')
        if isinstance(mode, str):
            idx = self.processingModeCombo.findText(mode)
            if idx >= 0:
                self.processingModeCombo.setCurrentIndex(idx)

    # UI-1 (#81): the GI group carries a header CHECKBOX as its on/off toggle,
    # mapped to the hidden bool that is its source of truth (see
    # wranglerWidget._install_group_toggles).
    # MaskSat moved to the integrator panel (hidden carrier), so no toggle here.
    # GI moved to the integrator panel (hidden carrier here) — no header toggle.
    _GROUP_TOGGLES = {}

    def _save_to_session(self):
        """Save parameters to session.json."""
        data = load_session() or {}
        for skey, param_path, _is_path, _attr in self._SESSION_PARAMS:
            try:
                p = self.parameters
                for name in param_path:
                    p = p.child(name)
                data[skey] = p.value()
            except (KeyError, AttributeError) as e:
                # Missing parameter tree node — surfaces only when the
                # widget is mid-tear-down or the tree shape changed.
                logger.debug(
                    "session save skipped for %s: %s", skey, e,
                )
        # Persist the Cores spinbox alongside the parameter-tree
        # values — see :meth:`_restore_from_session` for the inverse.
        data['max_cores'] = self.maxCoresSpinBox.value()
        data['processing_mode'] = self.processingModeCombo.currentText()
        save_session(data)

    # ── Processing-mode dropdown ─────────────────────────────────────

    def _on_mode_changed(self, mode_text: str) -> None:
        """Sync scan/thread flags to the dropdown's current text.

        Mirror of ``imageWrangler._on_mode_changed``'s logic, minus the
        viewer modes (the NeXus wrangler always runs the integration
        pipeline — it has no separate display-only viewer panel).

        * ``Int 1D + 2D`` → 1D + 2D, save .nxs, write XYE every frame
        * ``Int 1D``       → 1D only, save .nxs, write XYE every frame
        * ``Int 1D (XYE)`` → 1D only, **no** .nxs save, XYE only
        """
        # ``'1D'`` is the spec-wrangler convention: any mode whose label
        # contains '1D' (vs '2D') skips 2D integration.  Matches even
        # if we add new label variants later (e.g. 'Int 1D (cake-XYE)').
        skip_2d = ('1D' in mode_text) and ('2D' not in mode_text)
        self.scan.skip_2d = skip_2d
        self.xye_only = (mode_text == 'Int 1D (XYE)')
        # Push down to the thread immediately so a mid-session mode
        # change picks up on the next run without going through start().
        if hasattr(self, 'thread') and self.thread is not None:
            self.thread.xye_only = self.xye_only

    # ── Browse dialogs ───────────────────────────────────────────────

    def _compute_source_base(self):
        """N1: the absolute project root, or None when the Project Folder is
        blank (-> the writer stores absolute raw paths, back-compat)."""
        pf = (self.parameters.child('Project').child('project_folder').value() or '').strip()
        return os.path.abspath(os.path.expanduser(pf)) if pf else None

    def browse_project_folder(self):
        """Browse for the N1 Project Folder; setting it makes the processed
        ``.nxs`` store raw source paths RELATIVE to this root (portable)."""
        folder = QFileDialog.getExistingDirectory(self, 'Choose Project Folder', '')
        if folder:
            self.parameters.child('Project').child('project_folder').setValue(folder)
            self.project_folder = folder
            self.source_base = self._compute_source_base()
            self._save_to_session()

    def browse_poni(self):
        poni_file, _ = QFileDialog.getOpenFileName(
            self, 'Select PONI file', '', 'PONI files (*.poni);;All files (*)')
        if poni_file:
            self.parameters.child('Calibration').child('poni_file').setValue(poni_file)
            self.poni_file = poni_file
            self.poni = PONI.from_poni_file(poni_file)
            self._save_to_session()

    def browse_nexus(self):
        nexus_file, _ = QFileDialog.getOpenFileName(
            self, 'Select NeXus/HDF5 file',
            os.path.dirname(self.nexus_file) if self.nexus_file else '',
            'HDF5 files (*.h5 *.hdf5 *.nxs);;All files (*)')
        if nexus_file:
            self.parameters.child('NeXus File').child('nexus_file').setValue(nexus_file)
            self.nexus_file = nexus_file
            self._save_to_session()

    def browse_mask(self):
        mask_file, _ = QFileDialog.getOpenFileName(
            self, 'Select mask file', '', 'Image files (*.tif *.edf *.npy);;All files (*)')
        if mask_file:
            self.parameters.child('Signal').child('mask_file').setValue(mask_file)
            self.mask_file = mask_file
            self._save_to_session()

    def browse_h5_dir(self):
        h5_dir = QFileDialog.getExistingDirectory(
            self, 'Select output directory',
            self.h5_dir, QFileDialog.ShowDirsOnly)
        if h5_dir:
            self.parameters.child('Output').child('h5_dir').setValue(h5_dir)
            self.h5_dir = h5_dir
            self._save_to_session()

    # ── Thread control ───────────────────────────────────────────────

    def setup(self):
        """Sync parameters to thread before starting."""
        self.nexus_file = self.parameters.child('NeXus File').child('nexus_file').value()
        self.entry = self.parameters.child('NeXus File').child('entry').value()
        self.poni_file = self.parameters.child('Calibration').child('poni_file').value()
        self.mask_file = self.parameters.child('Signal').child('mask_file').value()
        # R3-B: detector-saturation masking opt-out (default ON).
        self.mask_sentinel = self.parameters.child('MaskSat').child('mask_sentinel').value()
        # N1: Project Folder -> @source_base (relative raw paths -> portable .nxs).
        self.project_folder = self.parameters.child('Project').child('project_folder').value()
        self.source_base = self._compute_source_base()

        # Load PONI if needed
        if self.poni_file and os.path.exists(self.poni_file):
            self.poni = PONI.from_poni_file(self.poni_file)

        # Output directory
        h5_dir = self.parameters.child('Output').child('h5_dir').value()
        if h5_dir:
            self.h5_dir = h5_dir

        # HDF5 output file
        scan_name = Path(self.nexus_file).stem if self.nexus_file else 'nexus_scan'
        self.fname = os.path.join(self.h5_dir, f'{scan_name}.nxs')
        self.scan_name = scan_name

        # GI parameters
        self.gi = self.parameters.child('GI').child('Grazing').value()
        self.incidence_motor = self.parameters.child('GI').child('th_motor').value()
        self.sample_orientation = self.parameters.child('GI').child('sample_orientation').value()
        self.tilt_angle = self.parameters.child('GI').child('tilt_angle').value()
        # GI modes are driven by the integrator panel (axis1D / axis2D)
        self.gi_mode_1d = self.scan.bai_1d_args.get('gi_mode_1d', 'q_total')
        self.gi_mode_2d = self.scan.bai_2d_args.get('gi_mode_2d', 'qip_qoop')

        # Recreate thread with current params.  Release the PREVIOUS one
        # first: it is parented to this widget, so without deleteLater every
        # run start accumulated one dormant QThread (plus its signal
        # connections and _published_frames remnants) for the app's life.
        _old = getattr(self, 'thread', None)
        if _old is not None and not (hasattr(_old, 'isRunning')
                                     and _old.isRunning()):
            try:
                _old.setParent(None)
                _old.deleteLater()
            except Exception:
                logger.debug("old nexusThread release failed", exc_info=True)
        self.thread = nexusThread(
            self.command_queue,
            self.scan_args,
            self.file_lock,
            self.fname,
            self.nexus_file,
            self.poni,
            self.mask_file,
            self.gi,
            self.incidence_motor,
            self.sample_orientation,
            self.tilt_angle,
            self.gi_mode_1d,
            self.gi_mode_2d,
            self.command,
            self.scan,
            self.data_1d,
            self.data_2d,
            entry=self.entry,
            parent=self,
        )
        self.thread.showLabel.connect(self._set_status_text)
        self.thread.sigUpdateFile.connect(self.sigUpdateFile.emit)
        self.thread.finished.connect(self.finished.emit)
        self.thread.sigUpdate.connect(self.sigUpdateData.emit)
        self.thread.sigUpdateGI.connect(self.sigUpdateGI.emit)
        self.sigUpdateGI.emit(self.gi)

        self.thread.file_lock = self.file_lock
        self.thread.scan_args = self.scan_args
        self.thread.scan = self.scan
        self.thread.data_1d = self.data_1d
        self.thread.data_2d = self.data_2d
        self.thread.command = self.command
        # R3-B: the freshly-recreated thread defaults mask_sentinel=True; push
        # the UI's value so the NeXus opt-out actually reaches _resolve_frame_mask.
        self.thread.mask_sentinel = self.mask_sentinel
        # N1: push the project root so the writer stamps @source_base + relative
        # raw paths (set AFTER the thread recreate above).
        self.thread.source_base = self.source_base

    def controls_profile(self):
        """NeXus wrangler: no Live / Batch / Pause; its own 3 mode items.
        ``current`` is the wrangler's own (pre-alias) combo, which __init__
        already restored from the session."""
        try:
            current = self.processingModeCombo.currentText()
        except Exception:
            current = ''
        return {
            'modes': ['Int 1D + 2D', 'Int 1D', 'Int 1D (XYE)'],
            'live': False, 'batch': False, 'cores': True,
            'current': current,
        }

    def attach_controls(self, controls):
        """Adopt the shared run controls: hide the legacy frame, ALIAS the
        self.* control refs onto the shared widgets (so start/stop/enabled/
        _on_mode_changed drive them), and wire the shared signals to handlers.
        NeXus has no Live/Batch/Pause — the action button is two-state
        (enabled 'Start' / disabled during a run)."""
        super().attach_controls(controls)
        if getattr(self, '_legacy_controls', None) is not None:
            self._legacy_controls.hide()
        # Status moved to the control layer (controls.statusLabel via
        # _status_label) — hide this wrangler's own now-orphaned status label.
        if getattr(self, 'statusLabel', None) is not None:
            self.statusLabel.hide()
        self.startButton = controls.startButton
        self.stopButton = controls.stopButton
        self.processingModeCombo = controls.modeCombo
        self.maxCoresSpinBox = controls.coresSpin
        self.coresLabel = controls.coresLabel
        self._connect_control(controls.startButton.clicked, self.start)
        # Stop is owned by the host (staticWidget._on_stop_clicked), which
        # dispatches to the active run (reintegrate, else this wrangler).
        self._connect_control(controls.modeCombo.currentTextChanged,
                              self._on_mode_changed)
        self._connect_control(controls.modeCombo.currentTextChanged,
                              lambda _=None: self._save_to_session())
        # NeXus never pauses: ensure the action button is the plain green Start.
        controls.set_action_phase('idle')
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    def start(self):
        self.command = 'start'
        self.thread.command = 'start'
        # Push the current Cores selection into the worker thread.
        # The thread caches it as ``self.max_cores`` and uses it when
        # building the ThreadPoolExecutor for parallel integration.
        self.thread.max_cores = self.maxCoresSpinBox.value()
        # Re-sync processing-mode flags onto scan + thread so a
        # stale ``thread`` instance (recreated in :meth:`setup` since
        # the last mode-change signal) sees the latest selection.
        self._on_mode_changed(self.processingModeCombo.currentText())
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self._save_to_session()
        self.sigStart.emit()

    def stop(self):
        self.command = 'stop'
        self.thread.command = 'stop'
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    def enabled(self, enable):
        """Enable/disable the WHOLE wrangler panel for the run lifecycle (#72),
        except Stop.

        Hard-disables the parameter tree (greyed + non-interactive, matching the
        integration panel) plus the non-param widgets (processing-mode combo,
        Cores spinbox + label).  A disabled pyqtgraph bool checkbox may repaint
        unchecked during the run (#56), but the value is preserved/restored on
        re-enable; the full visible disable was chosen over that cosmetic
        ("minimize complexity").
        """
        self.startButton.setEnabled(enable)
        self.tree.setEnabled(enable)
        for w in (self.processingModeCombo, self.maxCoresSpinBox,
                  getattr(self, 'coresLabel', None)):
            if w is not None:
                w.setEnabled(enable)
