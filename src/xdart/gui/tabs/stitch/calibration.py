"""Modeless guided Stitch calibration, using the shared child-process owner."""
from __future__ import annotations

from dataclasses import asdict, replace
from contextlib import ExitStack
import json
import math
from pathlib import Path
import uuid

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.pages.operation_owner import SingleWorkerOwner, OperationCancelled, OperationTerminalStatus
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.integrate.stitch_calibration import (
    adopt_calibration_pair, capture_calibration_frames, fit_calibration,
    load_calibration_session, save_calibration_fit, save_calibration_session,
    source_frame_choices, suggested_frame_indices, validate_frame_files,
)


def _run_job(request, cancelled, publish):
    kind, arguments = request
    if kind == 'enumerate':
        return source_frame_choices(*arguments)
    if kind == 'prepare':
        spec, indices, directory, rotation, motors = arguments
        return capture_calibration_frames(spec, indices, directory, rotation=rotation,
                                           motor_names=motors, cancelled=cancelled)
    if kind == 'calibrate':
        from xdart.gui.tabs.scattering.experiment_authoring import prepare_calibration_request, run_calibration
        from xdart.gui.tabs.scattering.operation_values import OperationIdentity, OperationTerminalStatus as ChildStatus
        frame = arguments
        validate_frame_files(frame, require_points=False)
        child_request = prepare_calibration_request(frame.image_path)
        child_request = replace(child_request, points_directory=str(Path(frame.image_path).parent))
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


class StitchCalibrationDialog(QtWidgets.QDialog):
    """Retained window. ``calibrationSaved(path, backend)`` adopts a fitted asset.

    ``set_source_context(SourceSpec, image_rotation=0, del_motor='del',
    nu_motor='nu')`` seeds the source controls without selecting frames or
    changing previously completed rows. Call ``show/raise_/activateWindow``.
    """
    calibrationSaved = QtCore.Signal(str, str)

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
        self._choices = []
        self._identity = None
        self._job = None
        self._row_identity = None
        self._close_pending = False
        self._distance_result = None
        self._frame_motors = ('del', 'nu')
        self.saved_source_motors = {}
        self.saved_image_rotation = 0
        self._owner = SingleWorkerOwner(_run_job)
        self._fields = {}
        self._path_rows = {}
        outer = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel('Select representative frames, calibrate each image, then fit and save one geometry.')
        heading.setWordWrap(True)
        outer.addWidget(heading)
        self.tabs = QtWidgets.QTabWidget()
        outer.addWidget(self.tabs, 1)
        self._build_source()
        self._build_frames()
        self._build_fit(backend)
        self._build_distance()
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

    def _form(self, title):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        self.tabs.addTab(widget, title)
        return layout, form

    def _edit(self, form, key, label, default='', *, chooser=None):
        if chooser is not None and key.startswith('beam_'):
            from xdart.gui.widgets.controls_panel import FormRow
            row = FormRow(label=label, path=('Calibration', key), value=default, browse=True)
            row.browseRequested.connect(lambda: self._browse(row.editor, chooser, key=key))
            self._path_rows[key] = row
            self._fields[key] = row.editor
            form.addRow(row)
            return row.editor
        edit = QtWidgets.QLineEdit(str(default))
        edit.setObjectName(key)
        self._fields[key] = edit
        if chooser is None:
            form.addRow(label, edit)
        else:
            row = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(edit)
            button = QtWidgets.QPushButton('Browse…')
            button.clicked.connect(lambda: self._browse(edit, chooser))
            layout.addWidget(button)
            form.addRow(label, row)
        return edit

    def _browse(self, edit, chooser, *, key=None):
        current = self._text(key) if key is not None else edit.text()
        if chooser == 'directory':
            path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select images directory', current)
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select file', current, chooser)
        if path:
            if key is not None:
                self._set_field_text(key, path)
            else:
                edit.setText(path)

    def _set_field_text(self, key, value):
        if key in self._path_rows:
            from xrd_tools.session.readiness import ControlFormField, SectionId
            row = self._path_rows[key]
            row.apply_field(ControlFormField(section=SectionId.EXPERIMENT, path=row.path,
                label=row.label.text(), value=str(value), browse=True))
        else:
            self._fields[key].setText(str(value))

    def _build_source(self):
        layout, form = self._form('1 · Select frames')
        self.source_kind = QtWidgets.QComboBox()
        self.source_kind.addItem('SPEC + scan + images', 'spec')
        self.source_kind.addItem('NeXus images + metadata', 'nexus_stack')
        form.addRow('Source type', self.source_kind)
        self._edit(form, 'source', 'Scan metadata file', chooser='All files (*)')
        self._edit(form, 'scan', 'SPEC scan', '1')
        self._edit(form, 'images', 'SPEC images directory', chooser='directory')
        self._edit(form, 'stem', 'Image filename stem', '')
        self._edit(form, 'entry', 'NeXus entry', 'entry')
        self._edit(form, 'del_motor', 'Detector delta column (degrees)', 'del')
        self._edit(form, 'nu_motor', 'Detector nu column (degrees)', 'nu')
        reader = QtWidgets.QHBoxLayout()
        for key, label, default in [('shape', 'RAW rows, columns', '195, 1475'), ('dtype', 'RAW dtype', 'int32'), ('header', 'Header bytes', '0')]:
            reader.addWidget(QtWidgets.QLabel(label))
            edit = QtWidgets.QLineEdit(default)
            self._fields[key] = edit
            reader.addWidget(edit)
        layout.addLayout(reader)
        self.rotation = QtWidgets.QComboBox()
        for value in (0, 90, 180, 270):
            self.rotation.addItem(f'{value}° counterclockwise', value)
        form.addRow('Image rotation (applied once)', self.rotation)
        self.list_button = QtWidgets.QPushButton('Read metadata and suggest 5 frames')
        self.list_button.clicked.connect(self._enumerate)
        layout.addWidget(self.list_button)
        self.selection = QtWidgets.QTableWidget(0, 4)
        self.selection.setHorizontalHeaderLabels(['Use', 'Frame', 'Delta (°)', 'Nu (°)'])
        self.selection.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.selection, 1)
        self.prepare_button = QtWidgets.QPushButton('Prepare selected images')
        self.prepare_button.clicked.connect(self._prepare)
        layout.addWidget(self.prepare_button)

    def _build_frames(self):
        layout, form = self._form('2 · Calibrate each frame')
        hint = QtWidgets.QLabel('Open each staged image in pyFAI, select rings and save the PONI. Its enabled control points are captured automatically. Completed rows and files remain when another frame is cancelled.')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.frame_table = QtWidgets.QTableWidget(0, 5)
        self.frame_table.setHorizontalHeaderLabels(['Frame', 'Delta / Nu (°)', 'Image', 'PONI / points', 'Status'])
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
        layout, form = self._form('3 · Fit and save')
        self.backend = QtWidgets.QComboBox()
        self.backend.addItem('pyFAI MultiGeometry', 'multigeometry')
        self.backend.addItem('xrayutilities · PSIC SURFACE', 'xu_hist')
        self.backend.setCurrentIndex(max(0, self.backend.findData(backend)))
        form.addRow('Geometry model', self.backend)
        self.scales = QtWidgets.QCheckBox('Fit MG motor scales when angle coverage identifies them')
        self.scales.setChecked(False)
        form.addRow(self.scales)
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

    def _build_distance(self):
        layout, form = self._form('Optional · Direct beam distance')
        self._beam_form = form
        self.beam_source_kind = QtWidgets.QComboBox()
        for label, value in [('SPEC + image series', 'spec_series'),
                             ('SPEC + Eiger / NeXus', 'spec_container'),
                             ('NeXus images + metadata', 'nexus')]:
            self.beam_source_kind.addItem(label, value)
        form.addRow('Source', self.beam_source_kind)
        self._edit(form, 'beam_source', 'Direct-beam scan file', chooser='All files (*)')
        self._edit(form, 'beam_scan', 'SPEC scan', '1')
        self._edit(form, 'beam_image_file', 'Eiger / NeXus images', chooser='HDF5 / NeXus (*.h5 *.hdf5 *.nxs *.nexus);;All files (*)')
        self._edit(form, 'beam_images', 'SPEC images directory', chooser='directory')
        self._edit(form, 'beam_stem', 'Image filename stem', '')
        self._edit(form, 'beam_entry', 'NeXus entry', 'entry')
        self.beam_source_kind.currentIndexChanged.connect(self._beam_source_changed)
        self._beam_source_changed()
        self._edit(form, 'beam_motor', 'Two-theta motor (degrees)', 'del')
        self._edit(form, 'beam_zero', 'Known normal-incidence zero (degrees)', '0')
        self._edit(form, 'beam_pixel', 'Approximate beam pixel x, y (oriented image)', '200, 100')
        self._edit(form, 'beam_detector', 'Detector factory name', 'Pilatus300kw')
        self.motion = QtWidgets.QComboBox()
        self.motion.addItems(['horizontal', 'vertical'])
        form.addRow('Beam motion', self.motion)
        self._edit(form, 'beam_strip', 'Strip half width (pixels)', '2')
        self._edit(form, 'beam_roi', 'ROI x0, y0, x1, y1 (optional)', '')
        self._edit(form, 'beam_mask', 'Mask file (optional)', chooser='Masks (*.edf *.tif *.tiff *.npy);;All files (*)')
        self._edit(form, 'beam_saturation', 'Saturation level (optional)', '')
        note = QtWidgets.QLabel('SPEC supplies the angles for an image series or an Eiger/NeXus container; standalone NeXus supplies both. Angle and image counts must match. Uses the image rotation and image-series reader settings from Select frames. Regression uncertainty excludes motor, zero and pitch systematics; it does not set bounds automatically.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.beam_button = QtWidgets.QPushButton('Fit direct-beam distance')
        self.beam_button.clicked.connect(self._distance)
        layout.addWidget(self.beam_button)
        self.beam_summary = QtWidgets.QPlainTextEdit()
        self.beam_summary.setReadOnly(True)
        layout.addWidget(self.beam_summary, 1)
        self.compatible = QtWidgets.QCheckBox('I confirm this is the same detector plane and orientation as the powder calibration')
        layout.addWidget(self.compatible)
        self.apply_distance_button = QtWidgets.QPushButton('Use result as distance seed')
        self.apply_distance_button.clicked.connect(self._apply_distance)
        layout.addWidget(self.apply_distance_button)

    def _beam_source_changed(self):
        kind = self.beam_source_kind.currentData()
        for key, visible in [('beam_scan', kind != 'nexus'),
                             ('beam_image_file', kind == 'spec_container'),
                             ('beam_images', kind == 'spec_series'),
                             ('beam_stem', kind == 'spec_series'),
                             ('beam_entry', kind != 'spec_series')]:
            self._beam_form.setRowVisible(self._path_rows.get(key, self._fields[key]), visible)

    def set_source_context(self, source: SourceSpec, *, image_rotation=0, del_motor='del', nu_motor='nu'):
        if self._owner.busy:
            return
        self.source_kind.setCurrentIndex(0 if source.kind is SourceKind.SPEC else 1)
        opts = source.options
        for key, value in [('source', source.uri), ('scan', opts.get('scan', '1')),
                           ('images', opts.get('image_dir', '')), ('stem', opts.get('image_stem', '')),
                           ('entry', source.entry or 'entry'), ('del_motor', del_motor), ('nu_motor', nu_motor)]:
            self._fields[key].setText(str(value))
        reader = opts.get('read_image_kwargs', {})
        for key, value in [('shape', ', '.join(map(str, reader.get('detector_shape', (195, 1475))))),
                           ('dtype', reader.get('raw_dtype', 'int32')), ('header', reader.get('raw_header_skip', 0))]:
            self._fields[key].setText(str(value))
        self.rotation.setCurrentIndex(max(0, self.rotation.findData(image_rotation)))

    def _text(self, key):
        if key in self._path_rows:
            return str(self._path_rows[key].current_value()).strip()
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
            if not self._text(prefix + 'images'):
                raise ValueError('SPEC calibration requires an image directory')
            options = {'scan': self._text(prefix + 'scan'), 'image_dir': self._text(prefix + 'images'),
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
            self._enumerated_motors = (self._text('del_motor'), self._text('nu_motor'))
            self._start('enumerate', (self._enumerated_spec, self._enumerated_motors))
        self._guard(go)

    def _prepare(self):
        def go():
            if not self._choices or self._spec() != self._enumerated_spec or self._enumerated_motors != (self._text('del_motor'), self._text('nu_motor')):
                raise ValueError('read metadata again after changing the source or angle columns')
            indices = tuple(self._choices[row][0] for row in range(self.selection.rowCount())
                            if self.selection.item(row, 0).checkState() == QtCore.Qt.CheckState.Checked)
            if self.frames:
                raise ValueError('this session already has staged frames; open another calibration window for a new selection')
            self._start('prepare', (self._enumerated_spec, indices, self.session_dir,
                                   self.rotation.currentData(), self._enumerated_motors))
        self._guard(go)

    def _selected_frame(self):
        row = self.frame_table.currentRow()
        if row < 0 or row >= len(self.frames):
            raise ValueError('select a calibration frame')
        frame = self.frames[row]
        self._row_identity = frame.identity
        return frame

    def _calibrate(self):
        self._guard(lambda: self._start('calibrate', self._selected_frame()))

    def _import(self):
        def go():
            frame = self._selected_frame()
            poni, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'PONI for this exact frame', str(self.session_dir), 'PONI (*.poni)')
            if not poni:
                return
            npt, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Matching control points for this exact frame', str(Path(poni).parent), 'Control points (*.npt)')
            if npt:
                self._start('import', (frame, poni, npt))
        self._guard(go)

    def _invalidate_fit(self):
        self.fit = None
        self.save_button.setEnabled(False)

    def _fit(self):
        def go():
            frames = tuple(frame for frame in self.frames if frame.complete)
            bounds = tuple(float(v) / 1000 for v in self._text('distance_bounds').split(',')) if self._text('distance_bounds') else None
            seed = float(self._text('distance_seed')) / 1000 if self._text('distance_seed') else None
            domain = None
            if self._text('del_domain') or self._text('nu_domain'):
                domain = {motor: [float(v) for v in self._text(motor + '_domain').split(',')] for motor in ('del', 'nu')}
            options = dict(backend=self.backend.currentData(), del_motor=self._frame_motors[0],
                nu_motor=self._frame_motors[1], fit_scales=self.scales.isChecked(),
                distance_seed_m=seed, distance_bounds_m=bounds, max_rms_q=float(self._text('rms')), domain=domain,
                distance_provenance=self._distance_provenance if hasattr(self, '_distance_provenance') else None)
            self._invalidate_fit()
            self._start('fit', (frames, options))
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
            self.saved_source_motors = self.fit.asset['provenance']['source_motors']
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
            self._start('distance', (self._spec(beam=True), self._text('beam_motor'), options))
        self._guard(go)

    def _apply_distance(self):
        if self._distance_result is None or not self.compatible.isChecked():
            self.status.setText('Fit a direct-beam scan and confirm compatible detector geometry first.')
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
            uncertainty_policy='conditional_regression_standard_error_not_total_accuracy',
            compatibility_confirmed=True)
        self.tabs.setCurrentIndex(2)

    def _render_frames(self):
        self.frame_table.setRowCount(len(self.frames))
        for row, frame in enumerate(self.frames):
            values = [str(frame.index), ', '.join(f'{k}={v:g}' for k, v in frame.motors.items()),
                      Path(frame.image_path).name, Path(frame.poni_path).name if frame.complete else '—',
                      'Complete' if frame.complete else 'Needs PONI + points']
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(frame.image_path if column == 2 else f'{frame.poni_path}\n{frame.npt_path}')
                self.frame_table.setItem(row, column, item)
        if self.frames:
            self.frame_table.selectRow(0)
        self.session_label.setText(f'Session: {self.session_dir / "calibration-session.json"}')

    def _new_session(self):
        if self._owner.busy:
            return
        self.session_dir = self.project_root / 'calibration' / f'session-{uuid.uuid4().hex[:12]}'
        self.frames = ()
        self._invalidate_fit()
        self._render_frames()
        self.tabs.setCurrentIndex(0)
        self.status.setText('New session. Earlier saved session files remain available.')

    def _resume(self):
        if self._owner.busy:
            return
        def go():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open calibration session', str(self.project_root), 'Calibration session (*.json)')
            if path:
                self.frames = load_calibration_session(path)
                self._frame_motors = tuple(self.frames[0].motors)
                self.session_dir = Path(path).parent
                self._invalidate_fit()
                self._render_frames()
                self.tabs.setCurrentIndex(1)
        self._guard(go)

    def _cancel(self):
        if self._identity is not None:
            self._owner.cancel(self._identity)
            self.status.setText('Cancelling; completed rows and saved files are retained.')

    def _poll(self):
        update = self._owner.poll(self._identity)
        if update is None or update.terminal is None:
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
                    self._choices = result
                    suggested = suggested_frame_indices(result)
                    self.selection.setRowCount(len(result))
                    for row, (index, motors, _) in enumerate(result):
                        item = QtWidgets.QTableWidgetItem('')
                        item.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
                        item.setCheckState(QtCore.Qt.CheckState.Checked if index in suggested else QtCore.Qt.CheckState.Unchecked)
                        self.selection.setItem(row, 0, item)
                        for column, text in enumerate([str(index), *(f'{motors[key]:g}' for key in self._enumerated_motors)], 1):
                            self.selection.setItem(row, column, QtWidgets.QTableWidgetItem(text))
                elif kind == 'prepare':
                    self.frames = result
                    self._frame_motors = self._enumerated_motors
                    self._render_frames()
                    self.tabs.setCurrentIndex(1)
                elif kind in ('calibrate', 'import') and result is not None:
                    self.frames = tuple(result if frame.identity == self._row_identity else frame for frame in self.frames)
                    save_calibration_session(self.session_dir, self.frames)
                    self._invalidate_fit()
                    self._render_frames()
                elif kind == 'fit':
                    self.fit = result
                    self.fit_summary.setPlainText(f'RMS |q|: {result.rms_q:.6g} Å⁻¹\nJacobian condition: {result.condition_number:.4g}\nControl points: {result.n_points}\n{result.message}\n\n' + json.dumps(result.asset['calibration'], indent=2))
                    self.save_button.setEnabled(True)
                elif kind == 'distance':
                    self._distance_result = result
                    rows = [f'Distance: {result.distance_mm:.6g} mm; regression SE: {result.distance_std_m * 1000:.3g} mm',
                            'Frame · angle (°) · centre (px) · residual (px) · status']
                    rows += [f'{result.point_ids[i]} · {result.angles_deg[i]:.6g} · {result.centres_px[i]:.5g} · {result.residuals_px[i]:.4g} · {"used" if result.used[i] else result.rejection_reasons[i]}' for i in range(len(result.angles_deg))]
                    self.beam_summary.setPlainText('\n'.join(rows))
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
