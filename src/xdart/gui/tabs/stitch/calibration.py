"""Modeless guided Stitch calibration, using the shared child-process owner."""
from __future__ import annotations

from dataclasses import asdict, replace
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re
import uuid

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.pages.operation_owner import SingleWorkerOwner, OperationCancelled, OperationTerminalStatus
from xdart.gui.widgets.controls_panel import ControlsComboBox, browse_button
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.integrate.calibration_seed import ring_summary, rough_arm_record
from xrd_tools.session.instrument import InstrumentGeometry
from xrd_tools.integrate.stitch_calibration import (
    adopt_calibration_pair, capture_calibration_frames, fit_calibration, scan_motor_names,
    load_calibration_session, save_calibration_fit, save_calibration_session,
    source_frame_choices, suggested_frame_indices, validate_frame_files,
)


#: Share of the progress bar per automatic-refinement stage (the owner needs one fixed total).
_REFINE_STAGES = {'Screening the scan': (0, 250), 'Finding ring points': (250, 700),
                  'Validating models': (700, 980), 'Fitting': (980, 1000)}


def _run_job(request, cancelled, publish):
    try:
        return _dispatch_job(request, cancelled, publish)
    except InterruptedError as error:  # the numerical owners' cancellation
        raise OperationCancelled(str(error)) from error


def _seed_record(seed, cancelled):
    """The calibration a frame's geometry is predicted from: this session's fit, a fit of
    three or more saved frames, else the page's calibration file (None without one)."""
    if seed.get('record') is not None:
        return seed['record']
    if len(seed.get('frames', ())) >= 3:
        try:
            fit = fit_calibration(seed['frames'], backend='multigeometry', del_motor=seed['del_motor'],
                                  nu_motor=seed['nu_motor'], cancelled=cancelled)
            return fit.asset['calibration']
        except (ValueError, RuntimeError, KeyError):
            pass  # the saved frames do not fit yet: fall back to the rough geometry or the file
    return seed.get('rough') if seed.get('rough') is not None else seed.get('file')


def _pyfai_arguments(frame, seed, cancelled=None):
    """pyFAI-calib2 options for one guided frame: its calibrant and energy, and the frame's
    predicted geometry when a calibration is at hand, so "extract rings" numbers the rings."""
    arguments = ['-c', seed['calibrant']]
    record = _seed_record(seed, cancelled)
    if record is not None:
        from xrd_tools.integrate.calibration_seed import predict_frame_seed, write_seed_poni
        try:
            predicted = predict_frame_seed(record, frame, calibrant=seed['calibrant'],
                                           del_motor=seed['del_motor'], nu_motor=seed['nu_motor'])
            path = write_seed_poni(predicted, Path(frame.image_path).parent / f'seed-{uuid.uuid4().hex}.poni',
                                   calibrant=seed['calibrant'])
            return (*arguments, '--poni', str(path))
        except (ValueError, KeyError, OSError):
            pass  # an unusable calibration only loses the head start
    if frame.energy_eV:
        arguments += ['-e', f'{frame.energy_eV / 1000:.9g}']
    if seed.get('detector'):
        arguments += ['-D', seed['detector']]
    return tuple(arguments)


def _dispatch_job(request, cancelled, publish):
    kind, arguments = request
    if kind == 'refine':
        from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
        spec, seed, options = arguments

        def progress(stage, done, total):
            low, high = _REFINE_STAGES[stage]
            publish(f'{stage} {done}/{total}', low + (high - low) * done // max(total, 1), 1000)
        return refine_arm_from_scan(spec, seed, cancelled=cancelled, progress=progress, **options)
    if kind == 'enumerate':
        spec, record = arguments
        names = scan_motor_names(spec)
        geometry = InstrumentGeometry.from_record(record)
        if names:  # the scan's own motor for each circle (a choice the scan has stays)
            geometry = InstrumentGeometry.pick(geometry.geometry, names, keep=geometry)
        motors = tuple(motor for motor in geometry.stitch_arm() if motor)  # a 2θ arm reads one
        return source_frame_choices(spec, motors), names, geometry.to_record()
    if kind == 'prepare':
        spec, indices, directory, rotation, arm = arguments
        return capture_calibration_frames(spec, indices, directory, rotation=rotation,
                                           arm=arm, cancelled=cancelled)
    if kind == 'calibrate':
        from xdart.gui.tabs.scattering.experiment_authoring import prepare_calibration_request, run_calibration
        from xdart.gui.tabs.scattering.operation_values import OperationIdentity, OperationTerminalStatus as ChildStatus
        frame, seed = arguments
        validate_frame_files(frame, require_points=False)
        child_request = prepare_calibration_request(frame.image_path)
        child_request = replace(child_request, points_directory=str(Path(frame.image_path).parent),
                                arguments=_pyfai_arguments(frame, seed, cancelled))
        child = run_calibration(child_request, OperationIdentity(1), cancelled, publish,
                                lambda identity: not cancelled.is_set())
        if child.status is ChildStatus.CANCELLED:
            raise OperationCancelled('calibration child cancelled')
        if child.status is ChildStatus.FAILED:
            raise ValueError(child.diagnostic)
        result = child.payload
        if not result.last_saved_path:
            return None
        if not result.control_points_path:
            raise ValueError('the saved PONI has no matching enabled control points')
        validate_frame_files(frame, require_points=False)
        adopted = adopt_calibration_pair(frame, result.last_saved_path, result.control_points_path,
                                         model_image_sha256=result.loaded_image_sha256)
        if adopted.poni_sha256 != result.candidates[0].proof.sha256 or adopted.npt_sha256 != result.control_points_sha256:
            raise ValueError('saved PONI or points changed during adoption')
        return adopted
    if kind == 'import':
        return adopt_calibration_pair(*arguments)
    if kind == 'fit':
        frames, options = arguments
        return fit_calibration(frames, cancelled=cancelled, **options)
    if kind == 'distance':
        from xrd_tools.sources import open_source
        from xrd_tools.integrate.direct_beam import calibrate_direct_beam, open_direct_beam_scan
        from xrd_tools.core.geometry import ImageOrientation
        spec, motor, options = arguments
        options = dict(options)
        if cancelled.is_set():
            raise OperationCancelled('direct-beam calibration cancelled')
        with ExitStack() as resources:
            source_options = dict(motor=motor, entry=spec.entry or 'entry')
            if spec.kind is SourceKind.SPEC:
                source_options['scan'] = spec.options['scan']
                options['reader_options'] = dict(spec.options.get('read_image_kwargs', {}))
                if spec.options.get('image_file'):
                    source_options['image_file'] = spec.options['image_file']
                else:
                    # Preserve the shared strict series discovery and exact
                    # path/frame references; the loader owns SPEC angle pairing.
                    choices = source_frame_choices(spec, (motor,))
                    source = open_source(replace(spec, options={**spec.options,
                                                              'require_complete_images': True}))
                    if getattr(source, 'close', None) is not None:
                        resources.callback(source.close)
                    frames = (source.frame_for(index) for index, _, _ in choices)
                    source_options['images'] = ((frame.source_path, frame.source_frame_index or 0)
                        if frame.source_path is not None else None for frame in frames)
            inputs = resources.enter_context(open_direct_beam_scan(spec.uri, **source_options))
            def images():
                for frame in inputs.images:
                    if cancelled.is_set():
                        raise OperationCancelled('direct-beam calibration cancelled')
                    yield frame
            result = calibrate_direct_beam(inputs.angles_deg, images(),
                orientation=ImageOrientation(rotation=options.pop('rotation')),
                frame_ids=inputs.frame_ids, point_ids=inputs.point_ids, **options)
            if cancelled.is_set():
                raise OperationCancelled('direct-beam calibration cancelled')
            return result
    raise ValueError('unknown calibration operation')


def fit_summary_text(fit):
    """Plain-text diagnostics for a fitted calibration (guided or automatic)."""
    provenance = fit.asset.get('provenance', {})
    lines = [f'RMS |q|: {fit.rms_q:.6g} Å⁻¹', f'Jacobian condition: {fit.condition_number:.4g}',
             f'Control points: {fit.n_points}']
    if fit.message:
        lines.append(fit.message)
    errors = provenance.get('parameter_errors') or {}
    if provenance.get('arm_model') == 'stacked_extended' and errors:
        record = fit.asset['calibration']
        lines.append('Parameters (± 1σ):')
        for name, value in zip(record['param_names'], record['param']):
            if name not in errors:
                continue
            scale, unit = ((1e3, 'mm') if name.startswith(('dist', 'poni', 'sample_offset'))
                           else (180 / math.pi, '°'))
            lines.append(f'  {name}: {value * scale:.6g} ± {errors[name] * scale:.2g} {unit}')
        correlations = provenance.get('parameter_correlations') or {}
        names, matrix = correlations.get('names', []), correlations.get('matrix', [])
        strong = [f'{names[i]}–{names[j]} {matrix[i][j]:+.2f}' for i in range(len(names))
                  for j in range(i + 1, len(names)) if abs(matrix[i][j]) > .9]
        if strong:
            lines.append('Strong correlations (|r| > 0.9): ' + ', '.join(strong))
    automatic = provenance.get('automatic_refinement')
    if automatic:
        validation = automatic['validation']
        chosen = {'stacked': 'five-parameter arm',
                  'stacked_extended': 'detector mount and sample offset'}[validation['chosen']]
        lines.append(f'Automatic refinement chose the {chosen}'
                     + (f": {validation['reason']}" if validation.get('reason') else ''))
        sigma = validation.get('held_out_sigma_A^-1')
        if sigma:
            lines.append(f"Held-out σ: five-parameter {sigma['stacked']:.3g} Å⁻¹, "
                         f"extended {sigma['stacked_extended']:.3g} Å⁻¹ "
                         f"(improvement {validation['improvement']:.0%})")
        coverage = validation.get('coverage', {})
        if coverage:
            lines.append(f"nu rows used: {coverage['nu_rows_selected']} of {coverage['nu_rows_available']}")
        excluded = automatic.get('excluded_frames') or []
        lines.append('Excluded frames: ' + (', '.join(f"{item['index']} ({item['reason']})"
                                                     for item in excluded) or 'none'))
        lines.append(f"Rejected outlier points: {automatic.get('rejected_points', 0)}")
    return '\n'.join(lines) + '\n\n' + json.dumps(fit.asset['calibration'], indent=2)


def _plain_choice(value):
    """A combo's data as a JSON value (None when it is not one)."""
    return value if value is None or isinstance(value, (bool, int, float, str)) else None


class StitchCalibrationDialog(QtWidgets.QDialog):
    """Retained window. ``calibrationSaved(path, backend)`` adopts a fitted asset.

    ``set_source_context(SourceSpec, image_rotation=0, del_motor='del',
    nu_motor='nu', geometry_path=None)`` seeds the source controls (and the
    automatic refinement's seed geometry) without selecting frames or changing
    previously completed rows. Call ``show/raise_/activateWindow``.
    """
    calibrationSaved = QtCore.Signal(str, str)
    #: The diffractometer or a circle's motor changed (the page keeps the same geometry).
    geometryChanged = QtCore.Signal()

    def __init__(self, project_root: str | Path, *, backend='multigeometry', parent=None):
        super().__init__(parent)
        self.setWindowTitle('Stitch calibration')
        self.setObjectName('stitchCalibrationDialog')
        self.setModal(False)
        self.resize(940, 730)
        self.project_root = Path(project_root).resolve()
        self.session_dir = self.project_root / 'calibration' / f'session-{uuid.uuid4().hex[:12]}'
        self.frames = ()
        self.fit = None
        #: Predicted rings per (frame, calibrant, calibration): see _expected_rings.
        self._ring_cache = {}
        self._choices = []
        self._identity = None
        self._job = None
        self._row_identity = None
        self._close_pending = False
        self._distance_result = None
        #: The arm the staged frames read: (del, nu), or (2θ, None) for fourc/twoc.
        self._frame_motors = ('del', 'nu')
        #: The instrument the current fit recorded; any other value leaves it unsaveable.
        self._fit_instrument = None
        self.saved_source_motors = {}
        self.saved_instrument_geometry = None
        self.saved_image_rotation = 0
        self._owner = SingleWorkerOwner(_run_job)
        self._fields = {}
        #: key -> (form, row widget); a field sharing another's row is in _parts instead
        self._rows = {}
        self._parts = {}
        # The arm model the user chose; applied whenever it is available.
        self._preferred_arm_model = None
        outer = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel('Start from a rough geometry (a direct-beam scan, a typed distance, or skip), select '
                                   'representative frames, calibrate each image, then fit and save one geometry.')
        heading.setWordWrap(True)
        outer.addWidget(heading)
        self.tabs = QtWidgets.QTabWidget()
        self._tab_pages = {}
        outer.addWidget(self.tabs, 1)
        self._build_distance()  # a rough geometry first: it predicts every frame's rings
        self._build_source()
        self._build_frames()
        self._build_fit(backend)
        self._sync_rough_rows()
        self.status = QtWidgets.QLabel('Ready. Three to five frames are suggested; angle coverage and fit diagnostics determine sufficiency.')
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        footer = QtWidgets.QHBoxLayout()
        self.cancel_button = QtWidgets.QPushButton('Cancel operation')
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setEnabled(False)
        footer.addWidget(self.cancel_button)
        footer.addStretch()
        new_session = QtWidgets.QPushButton('New session')
        new_session.clicked.connect(self._new_session)
        footer.addWidget(new_session)
        resume = QtWidgets.QPushButton('Open saved session…')
        resume.clicked.connect(self._resume)
        footer.addWidget(resume)
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(self.close)
        footer.addWidget(close)
        outer.addLayout(footer)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)

    def _form(self, title, key):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        form = QtWidgets.QFormLayout()
        # The fields take the width; macOS would centre them at their size hints.
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addLayout(form)
        self.tabs.addTab(widget, title)
        self._tab_pages[key] = widget
        return layout, form

    def _show_tab(self, key):
        self.tabs.setCurrentWidget(self._tab_pages[key])

    @staticmethod
    def _tip(tip, *widgets):
        for widget in widgets:
            if tip and widget is not None:
                widget.setToolTip(tip)

    def _combo(self, form, label, items, *, tip=''):
        """A Controls combo (readable popup rows) of ``(text, data)`` items, as one form row."""
        combo = ControlsComboBox()
        for text, value in items:
            combo.addItem(text, value)
        form.addRow(label, combo)
        self._tip(tip, combo, form.labelForField(combo))
        return combo

    def _edit(self, form, key, label, default='', *, chooser=None, tip='', placeholder=''):
        """One text field, with the Controls 📁 button when ``chooser`` names a file filter or 'directory'."""
        edit = QtWidgets.QLineEdit(str(default))
        edit.setObjectName(key)
        edit.setPlaceholderText(placeholder)
        self._fields[key] = edit
        row = edit
        if chooser is not None:
            row = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(edit, 1)
            button = browse_button(label)
            button.clicked.connect(lambda: self._browse(edit, chooser))
            layout.addWidget(button)
        form.addRow(label, row)
        self._rows[key] = (form, row)
        self._tip(tip, edit, form.labelForField(row))
        return edit

    def _attach(self, key, host, label, default='', *, tip='', placeholder='', width=None):
        """A second field on a file row (the scan beside its SPEC file, the stem beside its folder)."""
        caption = QtWidgets.QLabel(label)
        edit = QtWidgets.QLineEdit(str(default))
        edit.setObjectName(key)
        edit.setPlaceholderText(placeholder)
        if width:
            edit.setMaximumWidth(width)
        layout = self._rows[host][1].layout()
        layout.addWidget(caption)
        layout.addWidget(edit, 0 if width else 1)
        self._fields[key] = edit
        self._parts[key] = (host, caption, edit)
        self._tip(tip, caption, edit)
        return edit

    def _show(self, key, shown):
        """Show or hide a field: its form row, or itself when it shares another field's row."""
        if key in self._parts:
            for widget in self._parts[key][1:]:
                widget.setVisible(shown)
        else:
            form, row = self._rows[key]
            form.setRowVisible(row, shown)

    def _row_visible(self, key):
        if key in self._parts:
            host, _caption, edit = self._parts[key]
            return self._row_visible(host) and not edit.isHidden()
        form, row = self._rows[key]
        return form.isRowVisible(row)

    def _label(self, key):
        form, row = self._rows[key]
        return form.labelForField(row)

    def _auto_placeholders(self, prefix=''):
        """The automatic image folder and stem for a blank field, as the Stitch page shows them."""
        spec = self._text(prefix + 'source')
        path = Path(spec) if spec else None
        first = re.match(r'\s*(\d+)', self._text(prefix + 'scan'))
        self._fields[prefix + 'images'].setPlaceholderText(
            f'auto · {path.parent}' if path else "auto · the SPEC file's folder")
        self._fields[prefix + 'stem'].setPlaceholderText(
            f"auto · {path.stem if path else '<SPEC name>'}_scan{first[1] if first else '<N>'}_")

    def _browse(self, edit, chooser):
        from xdart.gui.widgets.file_dialogs import browse
        folder = chooser == 'directory'
        picks = browse(self, 'Select images directory' if folder else 'Select file', folder=folder,
                       file_filter='All files (*)' if folder else chooser, current=edit.text(),
                       fallback=str(self.project_root))
        if picks and picks[0]:
            edit.setText(picks[0])

    def _set_field_text(self, key, value):
        self._fields[key].setText(str(value))

    _SCAN_TIP = 'The SPEC scan number, e.g. 14 (or 14.1 for a repeated number).'
    _FOLDER_TIP = "The folder holding this scan's image files. Blank: the SPEC file's folder."
    _STEM_TIP = ("Text every image file name of this scan contains. Blank: <SPEC name>_scan<N>_, "
                 "the beamline's naming.")

    def _build_source(self):
        layout, form = self._form('2 · Select frames', 'select')
        self.source_kind = self._combo(form, 'Source type', [('SPEC + scan + images', 'spec'),
                                                             ('NeXus images + metadata', 'nexus_stack')],
            tip='SPEC + images: a SPEC file lists each frame\'s angles and image files hold the frames. '
                'NeXus: one file holds both.')
        self._edit(form, 'source', 'SPEC file', chooser='All files (*)',
                   tip='The calibrant scan: its SPEC file, or its NeXus file.')
        self._attach('scan', 'source', 'Scan', '1', tip=self._SCAN_TIP, width=90)
        self._edit(form, 'images', 'Images folder', chooser='directory', tip=self._FOLDER_TIP)
        self._attach('stem', 'images', 'Stem', '', tip=self._STEM_TIP)
        self._edit(form, 'entry', 'NeXus entry', 'entry', tip='The NeXus entry (group) holding the frames and angles.')
        reader = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(reader)
        row.setContentsMargins(0, 0, 0, 0)
        raw_tip = 'Only for headerless .raw images: the frame\'s rows and columns, pixel type and header size.'
        for key, label, default in [('shape', 'rows, columns', '195, 1475'), ('dtype', 'type', 'int32'),
                                    ('header', 'header bytes', '0')]:
            caption = QtWidgets.QLabel(label)
            edit = QtWidgets.QLineEdit(default)
            edit.setObjectName(key)
            self._fields[key] = edit
            self._tip(raw_tip, caption, edit)
            row.addWidget(caption)
            row.addWidget(edit, 1)
        form.addRow('RAW images', reader)
        self._rows['raw'] = (form, reader)
        self._tip(raw_tip, form.labelForField(reader))
        self.rotation = self._combo(form, 'Image rotation (applied once)',
                                    [(f'{value}° counterclockwise', value) for value in (0, 90, 180, 270)],
                                    tip='Rotates every image once before calibration; use it when the stored '
                                        'images are turned relative to the detector view you calibrate in.')
        self.source_kind.currentIndexChanged.connect(self._sync_source_rows)
        for key in ('source', 'scan'):
            self._fields[key].textChanged.connect(lambda _text: self._auto_placeholders())
        self._sync_source_rows()
        self._auto_placeholders()
        self.list_button = QtWidgets.QPushButton('Read metadata and suggest 5 frames')
        self.list_button.setToolTip('Lists the scan\'s frames with their angles and ticks five that span them.')
        self.list_button.clicked.connect(self._enumerate)
        layout.addWidget(self.list_button)
        self.selection = QtWidgets.QTableWidget(0, 4)
        self.selection.setHorizontalHeaderLabels(['Use', 'Frame', 'Delta (°)', 'Nu (°)'])
        self.selection.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.selection, 1)
        self.prepare_button = QtWidgets.QPushButton('Prepare selected images')
        self.prepare_button.setToolTip('Copies the ticked frames into this session\'s folder, ready for pyFAI.')
        self.prepare_button.clicked.connect(self._prepare)
        layout.addWidget(self.prepare_button)

    def _sync_source_rows(self, *_args):
        spec = self.source_kind.currentData() == 'spec'
        for key, shown in (('scan', spec), ('images', spec), ('raw', spec), ('entry', not spec)):
            self._show(key, shown)
        self._label('source').setText('SPEC file' if spec else 'NeXus file')

    def _build_frames(self):
        layout, form = self._form('3 · Calibrate each frame', 'calibrate')
        hint = QtWidgets.QLabel('Calibrate each frame in pyFAI: pick its rings, then press Save as PONI-file. xdart keeps the PONI and its enabled control points, closes pyFAI and selects the next frame. Completed rows and files remain when another frame is cancelled.')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.frame_table = QtWidgets.QTableWidget(0, 6)
        self.frame_table.setHorizontalHeaderLabels(['Frame', 'Delta / Nu (°)', 'Image', 'PONI / points', 'Status',
                                                    'Expected rings'])
        self.frame_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.frame_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.frame_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.frame_table, 1)
        buttons = QtWidgets.QHBoxLayout()
        self.calibrate_button = QtWidgets.QPushButton('Calibrate selected frame…')
        self.calibrate_button.clicked.connect(self._calibrate)
        self.import_button = QtWidgets.QPushButton('Import matched PONI + NPT…')
        self.import_button.clicked.connect(self._import)
        buttons.addWidget(self.calibrate_button)
        buttons.addWidget(self.import_button)
        layout.addLayout(buttons)
        self.session_label = QtWidgets.QLabel('Persistent calibration session will be saved inside the project.')
        self.session_label.setWordWrap(True)
        layout.addWidget(self.session_label)

    def _build_fit(self, backend):
        layout, form = self._form('4 · Fit and save', 'fit')
        self.backend = ControlsComboBox()
        self.backend.addItem('pyFAI MultiGeometry', 'multigeometry')
        self.backend.addItem('xrayutilities · PSIC SURFACE', 'xu_hist')
        self.backend.setCurrentIndex(max(0, self.backend.findData(backend)))
        form.addRow('Geometry model', self.backend)
        self.scales = QtWidgets.QCheckBox('Fit MG motor scales when angle coverage identifies them')
        self.scales.setChecked(False)
        form.addRow(self.scales)
        self.arm_model = ControlsComboBox()
        self.arm_model.addItem('Stacked arm (5 parameters)', 'stacked')
        self.arm_model.addItem('Stacked arm + detector mount and sample offset', 'stacked_extended')
        form.addRow('Arm model', self.arm_model)
        self.arm_model.activated.connect(self._arm_model_chosen)
        self._edit(form, 'rms', 'Maximum RMS |q| (Å⁻¹)', '.03')
        self._edit(form, 'distance_seed', 'Distance seed (mm; optional)', '')
        self._edit(form, 'distance_bounds', 'Explicit distance bounds (min, max mm)', '')
        self._edit(form, 'del_domain', 'XU supported delta range (min, max °; optional)', '')
        self._edit(form, 'nu_domain', 'XU supported nu range (min, max °; optional)', '')
        note = QtWidgets.QLabel('A distance seed only initializes the MG fit. Entering bounds constrains it. XU uses the notebook SURFACE convention, bounded centres / ±5° offsets, and 0.5–1.5 × the seed distance unless explicit bounds are entered. Blank angular domains use control-frame coverage. Control coverage and declared supported range are saved separately.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.fit_button = QtWidgets.QPushButton('Fit completed frames')
        self.fit_button.clicked.connect(self._fit)
        layout.addWidget(self.fit_button)
        box = QtWidgets.QGroupBox('Automatic refinement from a calibrant scan (MultiGeometry)')
        auto = QtWidgets.QFormLayout(box)
        self.refine_calibrant = ControlsComboBox()
        self.refine_calibrant.setEditable(True)
        self.refine_calibrant.addItems(['LaB6', 'CeO2', 'Si', 'Al2O3', 'Cr2O3', 'AgBh'])
        auto.addRow('Calibrant', self.refine_calibrant)
        self.refine_confirm = QtWidgets.QCheckBox(
            'The scan in Select frames is this calibrant, not a sample')
        auto.addRow(self.refine_confirm)
        self.refine_seed = ControlsComboBox()
        self.refine_seed.addItem('Current fit (unsaved)', 'fit')
        self.refine_seed.addItem('Geometry file', 'file')
        auto.addRow('Start from', self.refine_seed)
        self._edit(auto, 'refine_seed_path', 'Seed geometry JSON', chooser='JSON (*.json);;All files (*)')
        # the same calibrant and calibration predict each frame's rings (Calibrate each frame)
        self._fields['refine_seed_path'].editingFinished.connect(self._render_frames)
        self.refine_calibrant.currentTextChanged.connect(self._render_frames)
        self._edit(auto, 'refine_frames', 'Frame budget', '48')
        self._edit(auto, 'refine_max_counts', 'Maximum valid counts (optional)', '')
        self.refine_button = QtWidgets.QPushButton('Refine automatically')
        self.refine_button.setToolTip('Finds ring points on frames across the scan, starting from the seed '
                                      'geometry, and keeps the detector mount and sample offset only when '
                                      'they predict held-out nu rows better.')
        self.refine_button.clicked.connect(self._refine)
        auto.addRow(self.refine_button)
        layout.addWidget(box)
        self.fit_summary = QtWidgets.QPlainTextEdit()
        self.fit_summary.setReadOnly(True)
        layout.addWidget(self.fit_summary, 1)
        self.save_button = QtWidgets.QPushButton('Save geometry JSON and use in Stitch')
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)
        layout.addWidget(self.save_button)
        for key in ('rms', 'distance_seed', 'distance_bounds', 'del_domain', 'nu_domain'):
            self._fields[key].textChanged.connect(self._invalidate_fit)
        self.backend.currentIndexChanged.connect(self._invalidate_fit)
        self.scales.toggled.connect(self._invalidate_fit)
        self.arm_model.currentIndexChanged.connect(self._invalidate_fit)
        self.backend.currentIndexChanged.connect(self._sync_arm_model)
        self.scales.toggled.connect(self._sync_arm_model)
        self._sync_arm_model()

    def _build_distance(self):
        layout, form = self._form('1 · Geometry', 'rough')
        self._beam_form = form
        from xdart.gui.widgets.geometry_editor import GeometryEditor
        diffractometer = QtWidgets.QGroupBox('Diffractometer')
        box = QtWidgets.QVBoxLayout(diffractometer)
        self.geometry_editor = GeometryEditor(used=('nu', 'del', 'tth'), used_by='Stitch')
        box.addWidget(self.geometry_editor)
        layout.insertWidget(0, diffractometer)
        self.geometry_editor.changed.connect(self._geometry_edited)
        self.geometry_editor.changed.connect(self.geometryChanged.emit)
        # the arm's motors are these circles' (read by name as before)
        self._fields['del_motor'] = self.geometry_editor.motor_edit('del')
        self._fields['nu_motor'] = self.geometry_editor.motor_edit('nu')
        self.rough_mode = self._combo(form, 'Rough geometry', [('Direct-beam scan', 'scan'),
                                                               ('Enter the distance', 'manual'), ('Skip', 'skip')],
            tip='A first guess of where the detector is, so pyFAI opens each frame with its rings roughly in '
                'place and the table lists which rings to expect. A direct-beam scan measures the distance; '
                'or type it. Rough values are enough.')
        self._edit(form, 'rough_distance', 'Sample–detector distance (mm)', '',
                   tip='Sample to detector, in millimetres, with the detector facing the beam squarely.')
        self.beam_source_kind = self._combo(form, 'Source', [('SPEC + image series', 'spec_series'),
                                                             ('SPEC + Eiger / NeXus', 'spec_container'),
                                                             ('NeXus images + metadata', 'nexus')],
            tip='Where the direct-beam scan\'s angles and images come from.')
        self._edit(form, 'beam_source', 'SPEC file', chooser='All files (*)',
                   tip='The direct-beam scan: its SPEC file (the angles), or its NeXus file.')
        self._attach('beam_scan', 'beam_source', 'Scan', '1', tip=self._SCAN_TIP, width=90)
        self._edit(form, 'beam_image_file', 'Eiger / NeXus images',
                   chooser='HDF5 / NeXus (*.h5 *.hdf5 *.nxs *.nexus);;All files (*)',
                   tip='The Eiger master or NeXus file holding this scan\'s images.')
        self._edit(form, 'beam_images', 'Images folder', chooser='directory', tip=self._FOLDER_TIP)
        self._attach('beam_stem', 'beam_images', 'Stem', '', tip=self._STEM_TIP)
        self._edit(form, 'beam_entry', 'NeXus entry', 'entry', tip='The NeXus entry (group) holding the images.')
        self.beam_source_kind.currentIndexChanged.connect(self._sync_rough_rows)
        for key in ('beam_source', 'beam_scan'):
            self._fields[key].textChanged.connect(lambda _text: self._auto_placeholders('beam_'))
        self._auto_placeholders('beam_')
        self._edit(form, 'beam_motor', 'Two-theta motor (degrees)', 'del',
                   tip='The motor that swings the detector in the direct-beam scan (del for psic\'s vertical arm).')
        self._edit(form, 'beam_zero', 'Known normal-incidence zero (degrees)', '0',
                   tip='That motor\'s reading when the detector faces the beam squarely; 0 unless it is offset.')
        self._edit(form, 'beam_pixel', 'Approximate beam pixel x, y (oriented image)', '',
                   placeholder='column, row — e.g. 737, 97',
                   tip='Column, row of the pixel the direct beam hits with the detector facing it squarely (the '
                       'motor at the zero above), in the image as displayed. Rough is fine.')
        self._edit(form, 'beam_detector', 'Detector factory name', 'Pilatus300kw',
                   tip='The detector model; it sets the pixel size and shape.')
        self.motion = self._combo(form, 'Beam motion', [('horizontal', None), ('vertical', None)],
                                  tip='The direction the beam spot moves across the image as the motor turns.')
        self._edit(form, 'beam_strip', 'Strip half width (pixels)', '2',
                   tip='The beam spot is searched in a strip this many pixels either side of its path.')
        self._edit(form, 'beam_roi', 'ROI x0, y0, x1, y1 (optional)', '',
                   tip='Search only this pixel box (columns x0–x1, rows y0–y1).')
        self._edit(form, 'beam_mask', 'Mask file (optional)', chooser='Masks (*.edf *.tif *.tiff *.npy);;All files (*)',
                   tip='Pixels to ignore when finding the beam spot.')
        self._edit(form, 'beam_saturation', 'Saturation level (optional)', '',
                   tip='Counts at or above this are saturated and ignored.')
        self.rough_note = QtWidgets.QLabel()
        self.rough_note.setWordWrap(True)
        layout.addWidget(self.rough_note)
        self.beam_button = QtWidgets.QPushButton('Fit direct-beam distance')
        self.beam_button.setToolTip('Finds the beam spot in each image and fits the distance from how far '
                                    'it moves per degree.')
        self.beam_button.clicked.connect(self._distance)
        layout.addWidget(self.beam_button)
        self.beam_summary = QtWidgets.QPlainTextEdit()
        self.beam_summary.setReadOnly(True)
        self.beam_summary.setMaximumHeight(self.beam_summary.fontMetrics().lineSpacing() * 9)
        layout.addWidget(self.beam_summary)
        self.apply_distance_button = QtWidgets.QPushButton('Use result as distance seed')
        self.apply_distance_button.setToolTip('Starts the final fit (4 · Fit and save) from this distance.')
        self.apply_distance_button.clicked.connect(self._apply_distance)
        layout.addWidget(self.apply_distance_button)
        layout.addStretch(1)
        self.rough_mode.currentIndexChanged.connect(self._sync_rough_rows)
        for key in ('rough_distance', 'beam_pixel', 'beam_zero', 'beam_motor', 'beam_detector'):
            self._fields[key].editingFinished.connect(self._render_frames)

    _ROUGH_NOTES = {
        'scan': 'A direct-beam scan: SPEC supplies the angles for an image series or an Eiger/NeXus container; '
                'standalone NeXus supplies both. Angle and image counts must match. Uses the image rotation and '
                'image-series reader settings from Select frames. Its distance, with the beam pixel and zero, '
                'predicts each frame\'s rings; the regression uncertainty excludes motor, zero and pitch systematics.',
        'manual': 'The distance, the pixel the direct beam hits at normal incidence and that zero predict each '
                  'frame\'s rings (the Geometry\'s arm, one degree per degree); the fit refines them.',
        'skip': 'No rough geometry: each frame\'s rings are predicted from a calibration file (Fit and save · Seed '
                'geometry JSON), or from this session\'s fit once three frames are saved.',
    }
    _GEOMETRY_ROWS = ('beam_motor', 'beam_zero', 'beam_pixel', 'beam_detector')

    def _sync_rough_rows(self, *_args):
        """Every Geometry row's visibility, from the rough mode and the direct-beam source together."""
        mode, kind = self.rough_mode.currentData(), self.beam_source_kind.currentData()
        scan = mode == 'scan'
        visible = {'rough_distance': mode == 'manual', 'beam_source': scan,
                   'beam_scan': scan and kind != 'nexus', 'beam_image_file': scan and kind == 'spec_container',
                   'beam_images': scan and kind == 'spec_series', 'beam_stem': scan and kind == 'spec_series',
                   'beam_entry': scan and kind != 'spec_series',
                   **dict.fromkeys(('beam_strip', 'beam_roi', 'beam_mask', 'beam_saturation'), scan),
                   **dict.fromkeys(self._GEOMETRY_ROWS, mode != 'skip')}
        form = self._beam_form
        for key, shown in visible.items():
            self._show(key, shown)
        form.setRowVisible(self.beam_source_kind, scan)
        form.setRowVisible(self.motion, scan)
        for widget in (self.beam_button, self.beam_summary, self.apply_distance_button):
            widget.setVisible(scan)
        self._label('beam_source').setText('NeXus file' if kind == 'nexus' else 'SPEC file')
        self.rough_note.setText(self._ROUGH_NOTES[mode])
        if hasattr(self, 'frame_table'):
            self._render_frames()

    def _rough_record(self):
        """Step 1 as a nominal arm, or None (skipped, not yet measured, or incomplete)."""
        mode = self.rough_mode.currentData()
        try:
            if mode == 'manual':
                distance = float(self._text('rough_distance')) / 1000
            elif mode == 'scan' and self._distance_result is not None:
                distance = self._distance_result.distance_m
            else:
                return None
            zero = dict(zip(('del', 'nu'), self._frame_motors)).get(self._beam_role()) or self._text('beam_motor')
            return rough_arm_record(
                distance_m=distance, detector=self._text('beam_detector'),
                beam_px=tuple(float(v) for v in self._text('beam_pixel').split(',')),
                zero_deg=float(self._text('beam_zero') or 0), zero_motor=zero,
                del_motor=self._frame_motors[0], nu_motor=self._frame_motors[1])
        except (ValueError, TypeError, KeyError, RuntimeError):
            return None

    def _beam_role(self):
        """The arm circle the direct-beam motor names ("del", "tth", or a motor), by the
        Geometry; a single 2θ arm has one circle, whatever the default text says."""
        geometry = self.geometry_editor.value()
        role = geometry.arm_role(self._text('beam_motor'))
        if role is None and None in geometry.stitch_arm():
            role = 'del' if geometry.stitch_arm()[0] else 'nu'
        return role

    def set_source_context(self, source: SourceSpec, *, image_rotation=0, del_motor=None, nu_motor=None,
                           geometry_path=None):
        if self._owner.busy:
            return
        if geometry_path:
            self._fields['refine_seed_path'].setText(str(geometry_path))
            if self.fit is None:
                self.refine_seed.setCurrentIndex(self.refine_seed.findData('file'))
        self.source_kind.setCurrentIndex(0 if source.kind is SourceKind.SPEC else 1)
        opts = source.options
        for key, value in [('source', source.uri), ('scan', opts.get('scan', '1')),
                           ('images', opts.get('image_dir', '')), ('stem', opts.get('image_stem', '')),
                           ('entry', source.entry or 'entry'), ('del_motor', del_motor), ('nu_motor', nu_motor)]:
            if value is not None:  # psic's del/nu motors, when given (set_instrument names them all)
                self._fields[key].setText(str(value))
        reader = opts.get('read_image_kwargs', {})
        for key, value in [('shape', ', '.join(map(str, reader.get('detector_shape', (195, 1475))))),
                           ('dtype', reader.get('raw_dtype', 'int32')), ('header', reader.get('raw_header_skip', 0))]:
            self._fields[key].setText(str(value))
        self.rotation.setCurrentIndex(max(0, self.rotation.findData(image_rotation)))

    #: Choices saved with the Stitch settings. The calibrant-scan and same-detector
    #: confirmations are per use and never saved.
    _SAVED_CHOICES = ('source_kind', 'rotation', 'backend', 'arm_model', 'refine_calibrant',
                      'refine_seed', 'beam_source_kind', 'motion', 'rough_mode')

    def settings(self):
        """This dialog's inputs as plain values, for the saved Stitch settings."""
        choices = {name: {'data': _plain_choice(getattr(self, name).currentData()),
                          'text': getattr(self, name).currentText()}
                   for name in self._SAVED_CHOICES}
        preferred = self.arm_model.findData(self._preferred_arm_model)
        if preferred >= 0:  # the choice, even while the rules show the five-parameter arm
            choices['arm_model'] = {'data': self._preferred_arm_model,
                                    'text': self.arm_model.itemText(preferred)}
        return {
            'fields': {key: self._text(key) for key in self._fields},
            'choices': choices,
            'fit_motor_scales': self.scales.isChecked(),
            'instrument_geometry': self.geometry_editor.value().to_record(),
        }

    def set_instrument(self, geometry: InstrumentGeometry):
        """The diffractometer and motor names (the page's, shared by Stitch)."""
        self.geometry_editor.set_value(geometry)

    def apply_settings(self, values):
        """Restore saved inputs; unknown entries are ignored."""
        if not isinstance(values, dict):
            return
        try:
            self.geometry_editor.set_value(InstrumentGeometry.from_record(values.get('instrument_geometry')))
        except (ValueError, TypeError):
            pass  # an older settings file: the del/nu fields below still name the arm
        for key, text in (values.get('fields') or {}).items():
            if key in self._fields and isinstance(text, str):
                self._set_field_text(key, text)
        for name, choice in (values.get('choices') or {}).items():
            combo = getattr(self, name, None) if name in self._SAVED_CHOICES else None
            if combo is None or not isinstance(choice, dict):
                continue
            data = choice.get('data')
            # A combo without item data holds None for every item: match its text.
            index = -1 if data is None else combo.findData(data)
            if index < 0:
                index = combo.findText(str(choice.get('text', '')))
            if index < 0:
                if combo.isEditable() and choice.get('text'):  # a typed-in entry, e.g. a calibrant
                    combo.setCurrentText(str(choice['text']))
                continue
            if combo is self.arm_model:
                self._preferred_arm_model = combo.itemData(index)
            else:
                combo.setCurrentIndex(index)
        if isinstance(values.get('fit_motor_scales'), bool):
            self.scales.setChecked(values['fit_motor_scales'])
        self._sync_arm_model()

    def _text(self, key):
        return self._fields[key].text().strip()

    def _reader(self):
        return {'detector_shape': tuple(int(v) for v in self._text('shape').split(',')),
                'raw_dtype': self._text('dtype'), 'raw_header_skip': int(self._text('header'))}

    def _spec(self, beam=False):
        prefix = 'beam_' if beam else ''
        path = self._text(prefix + 'source')
        if not path:
            raise ValueError('select a scan metadata source')
        if beam:
            kind = SourceKind.NEXUS_STACK if self.beam_source_kind.currentData() == 'nexus' else SourceKind.SPEC
        else:
            kind = SourceKind(self.source_kind.currentData())
        options = {}
        if kind is SourceKind.SPEC:
            if beam and self.beam_source_kind.currentData() == 'spec_container':
                if not self._text('beam_image_file'):
                    raise ValueError('select the Eiger or NeXus image container for this SPEC scan')
                return SourceSpec(Path(path).resolve(), kind, entry=self._text('beam_entry') or 'entry',
                    options={'scan': self._text('beam_scan'), 'image_file': self._text('beam_image_file')})
            # a blank folder is the SPEC file's, as in Stitch; a blank stem is the source's default
            options = {'scan': self._text(prefix + 'scan'),
                       'image_dir': self._text(prefix + 'images') or str(Path(path).resolve().parent),
                       'read_image_kwargs': self._reader()}
            if self._text(prefix + 'stem'):
                options['image_stem'] = self._text(prefix + 'stem')
        return SourceSpec(Path(path).resolve(), kind, entry=self._text('beam_entry' if beam else 'entry') or 'entry', options=options)

    def _start(self, kind, arguments):
        if self._owner.busy:
            return
        self._job = kind
        self._identity = self._owner.begin((kind, arguments))
        self.tabs.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText(f'{kind.capitalize()} in progress…')
        self._timer.start()

    def _guard(self, operation):
        try:
            operation()
        except Exception as error:
            self.status.setText(str(error))

    def _enumerate(self):
        def go():
            self._enumerated_spec = self._spec()
            self._start('enumerate', (self._enumerated_spec, self.geometry_editor.value().to_record()))
        self._guard(go)

    def _prepare(self):
        def go():
            if (not self._choices or self._spec() != self._enumerated_spec
                    or self._enumerated_motors != self.geometry_editor.value().stitch_arm()):
                raise ValueError('read metadata again after changing the source or the Geometry\'s motors')
            indices = tuple(self._choices[row][0] for row in range(self.selection.rowCount())
                            if self.selection.item(row, 0).checkState() == QtCore.Qt.CheckState.Checked)
            if self.frames:
                raise ValueError('this session already has staged frames; open another calibration window for a new selection')
            self._start('prepare', (self._enumerated_spec, indices, self.session_dir, self.rotation.currentData(),
                                   self._enumerated_motors))
        self._guard(go)

    def _selected_frame(self):
        row = self.frame_table.currentRow()
        if row < 0 or row >= len(self.frames):
            raise ValueError('select a calibration frame')
        frame = self.frames[row]
        self._row_identity = frame.identity
        return frame

    def _calibrate(self):
        self._guard(lambda: self._start('calibrate', (self._selected_frame(), self._seed_options())))

    def _seed_options(self):
        """What pyFAI opens a frame with (see ``_pyfai_arguments``)."""
        done = tuple(frame for frame in self.frames if frame.complete)
        path = self._text('refine_seed_path')
        detector = None
        for frame in done:  # a saved frame's detector (its PONI names it)
            try:
                lines = Path(frame.poni_path).read_text(encoding='utf-8').splitlines()
            except OSError:
                continue
            detector = next((line.split(':', 1)[1].strip() for line in lines if line.startswith('Detector:')), None)
            if detector:
                break
        return dict(calibrant=self.refine_calibrant.currentText().strip() or 'LaB6',
                    del_motor=self._frame_motors[0], nu_motor=self._frame_motors[1],
                    record=(self.fit.asset.get('calibration')
                            if self.fit is not None and self.fit.backend == 'multigeometry' else None),
                    frames=done, rough=self._rough_record(),
                    file=str(Path(path).resolve()) if path and Path(path).is_file() else None,
                    detector=detector)

    def _expected_rings(self, frame):
        """The calibrant rings predicted on this frame (None without a calibration)."""
        seed = self._seed_options()
        record = next((value for value in (seed['record'], seed['rough'], seed['file']) if value is not None), None)
        if record is None:
            return None
        import hashlib
        key = (frame.identity, seed['calibrant'], hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
               if isinstance(record, dict) else record)
        if key not in self._ring_cache:
            from xrd_tools.integrate.calibration_seed import predict_frame_seed
            try:
                self._ring_cache[key] = predict_frame_seed(
                    record, frame, calibrant=seed['calibrant'], del_motor=seed['del_motor'],
                    nu_motor=seed['nu_motor']).rings
            except (ValueError, KeyError, OSError):
                self._ring_cache[key] = None
        return self._ring_cache[key]

    def _import(self):
        def go():
            frame = self._selected_frame()
            from xdart.gui.widgets.file_dialogs import browse
            picks = browse(self, 'PONI for this exact frame', file_filter='PONI (*.poni)',
                           fallback=str(self.session_dir))
            if not picks:
                return
            poni = picks[0]
            picks = browse(self, 'Matching control points for this exact frame',
                           file_filter='Control points (*.npt)', current=poni)
            if picks:
                self._start('import', (frame, poni, picks[0]))
        self._guard(go)

    def _invalidate_fit(self):
        self.fit = None
        self._fit_instrument = None
        self.save_button.setEnabled(False)
        self.fit_summary.clear()

    def _geometry_edited(self):
        """A fit belongs to the instrument it recorded: any other value leaves it unsaveable."""
        if not hasattr(self, 'save_button'):  # still being built
            return
        if self.fit is not None and self.geometry_editor.value().to_record() != self._fit_instrument:
            self._invalidate_fit()
        self._sync_arm_model()

    def _staged_arm_message(self, arm):
        def described(pair):  # a single arm's one motor, with the plane that says which circle it is
            delta, nu = pair
            if delta and nu:
                return f'{delta} and {nu}'
            return f'{delta or nu} as a {"vertical" if delta else "horizontal"} 2θ arm'
        return (f'The frames were staged reading {described(self._frame_motors)}; the Geometry now names '
                f'{described(arm)}. Restore those motors under 1 · Geometry, or start a New session to '
                'stage frames with these.')

    def _sync_arm_model(self):
        arm = self._frame_motors if self.frames else self.geometry_editor.value().stitch_arm()
        nu = None if None in arm else arm[1]  # a single arm holds one circle: no nu rows
        rows = {round(float(frame.motors[nu]), 6)
                for frame in self.frames if frame.complete and nu in frame.motors}
        reason = ('xrayutilities has no detector mount or sample offset model'
                  if self.backend.currentData() == 'xu_hist'
                  else 'a single 2θ arm holds one circle at zero; the extension needs both' if nu is None
                  else 'fitting motor scales keeps the five-parameter arm' if self.scales.isChecked()
                  else 'needs completed frames on at least 3 nu positions' if len(rows) < 3 else '')
        if reason:
            self.arm_model.setCurrentIndex(0)
        elif self._preferred_arm_model is not None:
            index = self.arm_model.findData(self._preferred_arm_model)
            if index >= 0 and index != self.arm_model.currentIndex():
                self.arm_model.setCurrentIndex(index)
        self.arm_model.setEnabled(not reason)
        self.arm_model.setToolTip(reason or 'Also fits the detector mount (tilt, roll) on the arm and '
                                  'the sample offset from the rotation centre.')

    def _arm_model_chosen(self, _index):
        self._preferred_arm_model = self.arm_model.currentData()

    def _fit(self):
        def go():
            if self.frames and self.geometry_editor.value().stitch_arm() != self._frame_motors:
                raise ValueError(self._staged_arm_message(self.geometry_editor.value().stitch_arm()))
            frames = tuple(frame for frame in self.frames if frame.complete)
            bounds = tuple(float(v) / 1000 for v in self._text('distance_bounds').split(',')) if self._text('distance_bounds') else None
            seed = float(self._text('distance_seed')) / 1000 if self._text('distance_seed') else None
            domain = None
            if self._text('del_domain') or self._text('nu_domain'):
                domain = {motor: [float(v) for v in self._text(motor + '_domain').split(',')] for motor in ('del', 'nu')}
            options = dict(backend=self.backend.currentData(), del_motor=self._frame_motors[0],
                nu_motor=self._frame_motors[1], fit_scales=self.scales.isChecked(),
                instrument_geometry=self.geometry_editor.value().to_record(),
                arm_model=self.arm_model.currentData(),
                distance_seed_m=seed, distance_bounds_m=bounds, max_rms_q=float(self._text('rms')), domain=domain,
                distance_provenance=self._distance_provenance if hasattr(self, '_distance_provenance') else None)
            self._invalidate_fit()
            self._start('fit', (frames, options))
        self._guard(go)

    def _refine(self):
        def go():
            geometry = self.geometry_editor.value()
            delta, nu = geometry.stitch_arm()
            if not self.refine_confirm.isChecked():
                raise ValueError('confirm that the scan is a calibrant scan; never refine the geometry on a sample')
            spec = self._spec()
            if self.refine_seed.currentData() == 'fit':
                if self.fit is None or self.fit.backend != 'multigeometry':
                    raise ValueError('fit a MultiGeometry arm first, or start from a geometry file')
                seed = self.fit.asset['calibration']
            else:
                if not self._text('refine_seed_path'):
                    raise ValueError('choose the seed geometry JSON')
                seed = str(Path(self._text('refine_seed_path')).resolve())
            record = seed if isinstance(seed, dict) else json.loads(Path(seed).read_text(encoding='utf-8'))
            source_motors = tuple((position, motor) for position, motor in (('del_angle', delta), ('nu_angle', nu))
                                  if motor is not None)
            from xrd_tools.session.instrument import calibration_arm
            problem = geometry.arm_mismatch(*calibration_arm(record, source_motors))
            if problem:
                raise ValueError(problem.replace('This calibration', 'The seed geometry', 1))
            options = dict(source_motors=source_motors,
                           instrument_geometry=geometry.to_record(),
                           image_rotation=self.rotation.currentData(),
                           calibrant=self.refine_calibrant.currentText().strip() or 'LaB6',
                           max_frames=int(self._text('refine_frames')),
                           max_counts=(float(self._text('refine_max_counts'))
                                       if self._text('refine_max_counts') else None),
                           max_rms_q=float(self._text('rms')))
            self._invalidate_fit()
            self._start('refine', (spec, seed, options))
        self._guard(go)

    def _save(self):
        def go():
            if self.fit is None:
                raise ValueError('fit the current controls before saving')
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save calibrated geometry',
                str(self.project_root / 'calibration' / f'{self.fit.backend}-{uuid.uuid4().hex[:8]}.json'), 'JSON (*.json)')
            if not path:
                return
            selected = Path(path).resolve()
            if not selected.is_relative_to(self.project_root):
                raise ValueError('save the geometry inside the Stitch project')
            saved = save_calibration_fit(self.fit, selected)
            from xdart.utils.browse import remember_browse_path
            remember_browse_path(selected)
            self.saved_source_motors = self.fit.asset['provenance']['source_motors']
            self.saved_instrument_geometry = self.fit.asset['provenance'].get('instrument_geometry')
            self.saved_image_rotation = self.fit.asset['provenance']['image_rotation']
            self.status.setText(f'Saved geometry and provenance: {saved}')
            self.calibrationSaved.emit(saved, self.fit.backend)
        self._guard(go)

    def _distance(self):
        def go():
            options = dict(detector=self._text('beam_detector'),
                beam_pixel=tuple(float(v) for v in self._text('beam_pixel').split(',')),
                motion=self.motion.currentText(), strip_half_width=int(self._text('beam_strip')),
                model='tangent', angle_zero_deg=float(self._text('beam_zero')),
                rotation=self.rotation.currentData())
            if self._text('beam_roi'):
                options['roi'] = tuple(int(v) for v in self._text('beam_roi').split(','))
            if self._text('beam_mask'):
                from xrd_tools.io.image import load_mask
                options['mask'] = load_mask(self._text('beam_mask'))
            if self._text('beam_saturation'):
                options['saturation'] = float(self._text('beam_saturation'))
            motor = dict(zip(('del', 'nu'), self.geometry_editor.value().stitch_arm())).get(self._beam_role())
            self._start('distance', (self._spec(beam=True), motor or self._text('beam_motor'), options))
        self._guard(go)

    def _apply_distance(self):
        if self._distance_result is None:
            self.status.setText('Fit a direct-beam scan first.')
            return
        result = self._distance_result
        self._fields['distance_seed'].setText(f'{result.distance_mm:.12g}')
        self._distance_provenance = dict(distance_m=result.distance_m,
            regression_std_m=result.distance_std_m, angle_zero_deg=result.angle_zero_deg,
            pixel_pitch_m=result.pixel_pitch_m, image_shape=list(result.image_shape),
            model=result.model, frame_ids=[str(v) for v in result.frame_ids],
            point_ids=[tuple(str(value) if isinstance(value, Path) else value for value in point)
                       if isinstance(point, tuple) else point for point in result.point_ids],
            angles_deg=result.angles_deg.tolist(),
            used=result.used.tolist(), rejection_reasons=list(result.rejection_reasons),
            centres_px=[float(v) if math.isfinite(v) else None for v in result.centres_px],
            residuals_px=[float(v) if math.isfinite(v) else None for v in result.residuals_px],
            uncertainty_policy='conditional_regression_standard_error_not_total_accuracy')
        self._show_tab('fit')

    def _select_next_frame(self, saved):
        """After a saved frame, select the next one still needing a PONI (after it, then from the top)."""
        rows = [row for row, frame in enumerate(self.frames) if not frame.complete]
        after = next((row for row, frame in enumerate(self.frames) if frame.identity == saved.identity), -1)
        if rows:
            self.frame_table.selectRow(next((row for row in rows if row > after), rows[0]))

    def _render_frames(self):
        self.frame_table.setRowCount(len(self.frames))
        for row, frame in enumerate(self.frames):
            # the frame's own detector image (and its frame number inside a multi-frame file)
            image = Path(frame.raw_path).name + (f' #{frame.raw_index}' if frame.raw_index else '')
            values = [str(frame.index), ', '.join(f'{k}={v:g}' for k, v in frame.motors.items()),
                      image, Path(frame.poni_path).name if frame.complete else '—',
                      'Complete' if frame.complete else 'Needs PONI + points']
            rings = self._expected_rings(frame)
            values.append('—' if rings is None else ring_summary(rings) or 'none')
            ring_tip = ('No calibration to predict from yet: choose a calibration file, or save three '
                        'frames and their rings are predicted from this session\'s fit.' if rings is None
                        else '\n'.join(f'Ring {ring.number} · 2θ {ring.two_theta_deg:.2f}° · d {ring.d_spacing_A:.4f} Å'
                                        for ring in rings) or 'No calibrant ring reaches this frame.')
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(f'{frame.raw_path}\nstaged for pyFAI: {frame.image_path}' if column == 2
                                else ring_tip if column == 5 else f'{frame.poni_path}\n{frame.npt_path}')
                self.frame_table.setItem(row, column, item)
        if self.frames:
            self.frame_table.selectRow(0)
        self.session_label.setText(f'Session: {self.session_dir / "calibration-session.json"}')
        self._sync_arm_model()

    def _new_session(self):
        if self._owner.busy:
            return
        self.session_dir = self.project_root / 'calibration' / f'session-{uuid.uuid4().hex[:12]}'
        self.frames = ()
        self._invalidate_fit()
        self._render_frames()
        self._show_tab('select')
        self.status.setText('New session. Earlier saved session files remain available.')

    def _resume(self):
        if self._owner.busy:
            return
        def go():
            from xdart.gui.widgets.file_dialogs import browse
            picks = browse(self, 'Open calibration session', file_filter='Calibration session (*.json)',
                           fallback=str(self.project_root))
            path = picks[0] if picks else ''
            if path:
                self.frames, arm = load_calibration_session(path, with_arm=True)
                # an older session names no arm: del and nu in order, or a vertical 2θ arm
                self._frame_motors = arm or (tuple(self.frames[0].motors) + (None,))[:2]
                self.session_dir = Path(path).parent
                self._invalidate_fit()
                self._render_frames()
                self._show_tab('calibrate')
        self._guard(go)

    def _cancel(self):
        if self._identity is not None:
            self._owner.cancel(self._identity)
            self.status.setText('Cancelling; completed rows and saved files are retained.')

    def _poll(self):
        update = self._owner.poll(self._identity)
        if update is None:
            return
        if update.terminal is None:
            self.status.setText(f'{self._job.capitalize()}: {update.progress.stage}…')
            return
        terminal, kind = update.terminal, self._job
        self._identity = None
        self.tabs.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._timer.stop()
        if terminal.status is OperationTerminalStatus.RETURNED:
            result = terminal.payload
            try:
                if kind == 'enumerate':
                    result, names, record = result
                    self.geometry_editor.set_available_motors(names)
                    self.geometry_editor.set_value(InstrumentGeometry.from_record(record))
                    self._enumerated_motors = InstrumentGeometry.from_record(record).stitch_arm()
                    read = [motor for motor in self._enumerated_motors if motor]
                    self.selection.setColumnCount(2 + len(read))
                    self.selection.setHorizontalHeaderLabels(['Use', 'Frame', *(f'{motor} (°)' for motor in read)])
                    self._choices = result
                    suggested = suggested_frame_indices(result)
                    self.selection.setRowCount(len(result))
                    for row, (index, motors, _) in enumerate(result):
                        item = QtWidgets.QTableWidgetItem('')
                        item.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
                        item.setCheckState(QtCore.Qt.CheckState.Checked if index in suggested else QtCore.Qt.CheckState.Unchecked)
                        self.selection.setItem(row, 0, item)
                        for column, text in enumerate([str(index), *(f'{motors[key]:g}' for key in read)], 1):
                            self.selection.setItem(row, column, QtWidgets.QTableWidgetItem(text))
                elif kind == 'prepare':
                    self.frames = result
                    self._frame_motors = self._enumerated_motors
                    self._render_frames()
                    self._show_tab('calibrate')
                elif kind in ('calibrate', 'import') and result is not None:
                    self.frames = tuple(result if frame.identity == self._row_identity else frame for frame in self.frames)
                    save_calibration_session(self.session_dir, self.frames, arm=self._frame_motors)
                    self._invalidate_fit()
                    self._render_frames()
                    self._select_next_frame(result)
                elif kind in ('fit', 'refine'):
                    self.fit = result
                    self._fit_instrument = result.asset['provenance'].get('instrument_geometry')
                    self.fit_summary.setPlainText(fit_summary_text(result))
                    self.save_button.setEnabled(True)
                elif kind == 'distance':
                    self._distance_result = result
                    rows = [f'Distance: {result.distance_mm:.6g} mm; regression SE: {result.distance_std_m * 1000:.3g} mm',
                            'Frame · angle (°) · centre (px) · residual (px) · status']
                    rows += [f'{result.point_ids[i]} · {result.angles_deg[i]:.6g} · {result.centres_px[i]:.5g} · {result.residuals_px[i]:.4g} · {"used" if result.used[i] else result.rejection_reasons[i]}' for i in range(len(result.angles_deg))]
                    self.beam_summary.setPlainText('\n'.join(rows))
                    self._render_frames()  # the measured distance now predicts the rings
                if kind == 'calibrate' and result is not None:
                    pending = [frame for frame in self.frames if not frame.complete]
                    self.status.setText(f'Frame {result.index} saved. ' + (
                        'Next: Calibrate selected frame…' if pending else 'All frames calibrated: Fit and save.'))
                else:
                    self.status.setText('No PONI was saved; the frame remains unchanged.' if kind == 'calibrate' and result is None else f'{kind.capitalize()} complete.')
            except Exception as error:
                self.status.setText(str(error))
        elif terminal.status is OperationTerminalStatus.CANCELLED:
            self.status.setText('Cancelled. Previously completed frames and saved files are retained.')
        else:
            self.status.setText(terminal.failure_message)
        if self._close_pending:
            self._close_pending = False
            self.close()

    def closeEvent(self, event):
        if self._owner.busy:
            self._close_pending = True
            self._cancel()
            event.ignore()
        else:
            self._timer.stop()
            event.accept()

    def active(self):
        """Whether host navigation must retain this window's worker."""
        return self._owner.busy

    def shutdown(self):
        """Host Exit: cancel and return the existing retryable close receipt."""
        from xdart.gui.pages.values import PageCleanup
        self._timer.stop()
        receipt = self._owner.close()
        if receipt.status is PageCleanup.CLEAN:
            self.hide()
        return receipt
