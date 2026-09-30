"""Modeless guided Stitch calibration, using the shared child-process owner."""
from __future__ import annotations

from dataclasses import asdict, replace
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re
import uuid

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.pages.operation_owner import SingleWorkerOwner, OperationCancelled, OperationTerminalStatus
from xdart.gui.widgets.controls_panel import ControlsComboBox, browse_button
from xdart.gui.widgets.file_dialogs import NEXUS_FILTER
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.integrate.calibration_seed import (
    DETECTOR_ORIENTATIONS, detector_choices, detector_shape, ring_summary, rough_arm_record,
)
from xrd_tools.session.instrument import InstrumentGeometry
from xrd_tools.integrate.stitch_calibration import (
    adopt_calibration_pair, automatic_arm_reason, capture_calibration_frames, fit_calibration,
    load_calibration_session, pair_validation_angles, save_calibration_pair, save_calibration_session,
    scan_motor_names, source_frame_choices, suggested_frame_indices, validate_frame_files, xu_partner_path,
)
from .direct_beam_fit import DirectBeamFitWindow, direct_beam_fit_text


#: Share of the progress bar per automatic-refinement stage (the owner needs one fixed total).
_REFINE_STAGES = {'Screening the scan': (0, 250), 'Finding ring points': (250, 700),
                  'Validating models': (700, 980), 'Fitting': (980, 1000)}
#: Save: the pyFAI file is quick; the xrayutilities transfer and its gates take seconds.
_TRANSFER_STAGES = {'Transferring to xrayutilities': (0, 500), 'Validating the transfer': (500, 1000)}


def _staged_progress(publish, stages):
    """One fixed-total progress bar over named stages (the owner refuses a changing total)."""
    def progress(stage, done, total):
        low, high = stages[stage]
        publish(f'{stage} {done}/{total}', low + (high - low) * done // max(total, 1), 1000)
    return progress


def _run_job(request, cancelled, publish):
    try:
        return _dispatch_job(request, cancelled, publish)
    except InterruptedError as error:  # the numerical owners' cancellation
        raise OperationCancelled(str(error)) from error


def _seed_record(seed, cancelled):
    """The calibration a frame's geometry is predicted from: this session's fit, else the arm
    fitted to the frames calibrated so far (one is enough), else the rough geometry or the
    page's calibration file (None without one)."""
    if seed.get('record') is not None:
        return seed['record']
    if seed.get('prediction') is not None:
        return seed['prediction']
    if seed.get('frames'):
        from xrd_tools.integrate.calibration_seed import session_prediction
        record = session_prediction(seed['frames'], del_motor=seed['del_motor'], nu_motor=seed['nu_motor'],
                                    cancelled=cancelled, bounds=seed.get('bounds'), **seed.get('distance', {}))
        if record is not None:
            return record  # else the saved frames do not fit yet: the rough geometry or the file
    return seed.get('rough') if seed.get('rough') is not None else seed.get('file')


def _prediction_after(seed, adopted, cancelled):
    """The arm fitted to the calibrated frames with this one: it predicts the rest (or None).
    A cancel while it is fitted drops only the prediction: the frame pyFAI saved is kept."""
    from xrd_tools.integrate.calibration_seed import session_prediction
    frames = tuple(frame for frame in seed.get('frames', ()) if frame.identity != adopted.identity)
    try:
        return session_prediction((*frames, adopted), del_motor=seed['del_motor'], nu_motor=seed['nu_motor'],
                                  cancelled=cancelled, bounds=seed.get('bounds'), **seed.get('distance', {}))
    except InterruptedError:
        return None


def _pyfai_arguments(frame, seed, cancelled=None):
    """pyFAI-calib2 options for one guided frame: its calibrant and energy, and the frame's
    predicted geometry when a calibration is at hand, so "extract rings" numbers the rings."""
    arguments = ['-c', seed['calibrant']]
    if seed.get('mask'):  # the Detector panel's mask, turned as the staged frame is
        arguments += ['-m', seed['mask']]
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
        return refine_arm_from_scan(spec, seed, cancelled=cancelled,
                                    progress=_staged_progress(publish, _REFINE_STAGES), **options)
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
        return adopted, _prediction_after(seed, adopted, cancelled)
    if kind == 'import':
        frame, poni, npt, seed = arguments
        adopted = adopt_calibration_pair(frame, poni, npt)
        return adopted, _prediction_after(seed, adopted, cancelled)
    if kind == 'predict':
        key, seed = arguments
        from xrd_tools.integrate.calibration_seed import session_prediction
        return key, session_prediction(seed['frames'], del_motor=seed['del_motor'],
                                       nu_motor=seed['nu_motor'], cancelled=cancelled, bounds=seed.get('bounds'),
                                       **seed.get('distance', {}))
    if kind == 'fit':
        frames, options = arguments
        return fit_calibration(frames, cancelled=cancelled,
                               progress=_staged_progress(publish, _TRANSFER_STAGES), **options)
    if kind == 'save':  # the pyFAI file, then its xrayutilities partner beside it
        fit, path, angles = arguments
        return save_calibration_pair(fit, path, validation_angles=angles, cancelled=cancelled,
                                     progress=_staged_progress(publish, _TRANSFER_STAGES))
    if kind == 'distance':
        from xrd_tools.sources import open_source
        from xrd_tools.integrate.direct_beam import calibrate_direct_beam, open_direct_beam_scan
        from xrd_tools.core.geometry import ImageOrientation
        spec, motor, options = arguments
        options = dict(options)
        if cancelled.is_set():
            raise OperationCancelled('direct-beam calibration cancelled')
        energy = None  # the scan's one recorded energy (SPEC image series), for the wavelength
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
                    energies = {value for _index, _motors, value in choices if value is not None}
                    energy = energies.pop() if len(energies) == 1 else None
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
            return result, energy
    raise ValueError('unknown calibration operation')


def fit_summary_text(fit):
    """Plain-text diagnostics for a fitted calibration (guided or automatic)."""
    provenance = fit.asset.get('provenance', {})
    lines = [f'RMS |q|: {fit.rms_q:.6g} Å⁻¹', f'Jacobian condition: {fit.condition_number:.4g}',
             f'Control points: {fit.n_points}']
    if fit.message:
        lines.append(fit.message)
    lines += fit_bounds_lines(provenance.get('fit_bounds'))
    errors = provenance.get('parameter_errors') or {}
    if provenance.get('arm_model') == 'stacked_extended' and errors:
        # An XU fit's calibration is the transferred asset; the fitted arm is in provenance.
        record = provenance.get('pyfai_extended_arm') or fit.asset['calibration']
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
    transfer = provenance.get('xu_transfer')
    if transfer:
        lines.append(f"Transferred to xrayutilities: max |Δq| {transfer['max_abs_dq_A^-1']:.3g} Å⁻¹ over every "
                     f"pixel of {transfer['validation_frames']} frames (gate {transfer['gate_A^-1']:.0e})")
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


_BOUND_LABELS = {'distance_mm': ('distance', 'mm'), 'poni1_mm': ('PONI 1', 'mm'), 'poni2_mm': ('PONI 2', 'mm'),
                 'del_zero_deg': ('del zero', '°'), 'nu_zero_deg': ('nu zero', '°'),
                 'del_held_deg': ('held del (fixed pitch)', '°'), 'nu_held_deg': ('held nu (fixed yaw)', '°'),
                 'mount_tilt_deg': ('mount tilt', '°'), 'mount_roll_deg': ('mount roll', '°'),
                 'sample_offset1_mm': ('sample offset 1', 'mm'), 'sample_offset2_mm': ('sample offset 2', 'mm'),
                 'sample_offset3_mm': ('sample offset 3', 'mm'), 'energy_eV': ('energy', 'eV')}


def fit_bounds_lines(record):
    """The fit's parameters as it reports them: initial → fitted (the correction), the bounds, and
    those it left at a bound, which are never widened (ARM-ZERO-BOUNDS-20260929)."""
    if not record or not record.get('parameters'):
        return []
    estimates = record.get('estimates') or {}
    sources = ', '.join(f'{circle} from {source}' for circle, source in (estimates.get('zero_sources') or {}).items())
    lines = ['Fit parameters (initial → fitted, correction; bounds)' + (f' — zeros: {sources}' if sources else '') + ':']
    for name, row in record['parameters'].items():
        label, unit = _BOUND_LABELS.get(name, (name, ''))
        bounds = ('free' if row['lower'] is None else f"{row['lower']:.6g} to {row['upper']:.6g}")
        lines.append(f"  {label}: {row['initial']:.6g} → {row['fitted']:.6g} {unit} "
                     f"({row['fitted'] - row['initial']:+.3g}; {bounds})" + (' — AT BOUND' if row['at_bound'] else ''))
    return lines


def saved_pair_text(saved):
    """What Save wrote: both files, or the pyFAI file and why it has no xrayutilities partner."""
    if saved.xu_path is None:
        return f'Saved {saved.pyfai_path}; {saved.xu_refusal}'
    transfer = saved.xu_transfer
    return (f"Saved {saved.pyfai_path} and its xrayutilities file {Path(saved.xu_path).name} "
            f"(|Δq| {transfer['stitch_path_max_abs_dq_A^-1']:.2g} Å⁻¹, polarization "
            f"{transfer['polarization_max_relative']:.2g} relative, over every pixel of "
            f"{transfer['validation_frames']} frames)")


def migrated_instrument(geometry: InstrumentGeometry, values) -> InstrumentGeometry:
    """The geometry with an older window's 'Known normal-incidence zero' (``fields.beam_zero``, the
    direct-beam motor's reading at normal incidence) as that motor's circle's motor zero, unless the
    circle already has one. The page's instrument reader and the window's settings both apply it, so
    the first opening keeps it (Codex review F1)."""
    fields = (values or {}).get('fields') or {} if isinstance(values, dict) else {}
    try:
        zero = float(fields.get('beam_zero') or 0.)
    except (TypeError, ValueError):
        return geometry
    role = geometry.arm_role(str(fields.get('beam_motor') or 'del'))
    if not zero or not math.isfinite(zero) or role is None:
        return geometry
    circle = role if geometry.geometry == 'psic' else 'tth'
    if geometry.zero(circle) is not None:  # a newer zero is kept
        return geometry
    return replace(geometry, zeros=(*geometry.zeros, (circle, zero)))


def _plain_choice(value):
    """A combo's data as a JSON value (None when it is not one)."""
    return value if value is None or isinstance(value, (bool, int, float, str)) else None


class StitchCalibrationDialog(QtWidgets.QDialog):
    """Retained window. ``calibrationSaved(path, backend)`` adopts a saved pyFAI calibration
    (``backend`` 'multigeometry'); its xrayutilities partner, when one was made, is beside it
    (``xu_partner_path``).

    ``set_source_context(SourceSpec, image_rotation=0, del_motor='del',
    nu_motor='nu', geometry_path=None)`` seeds the source controls (and the
    automatic refinement's seed geometry) without selecting frames or changing
    previously completed rows. Call ``show/raise_/activateWindow``.
    """
    calibrationSaved = QtCore.Signal(str, str)
    #: The diffractometer or a circle's motor changed (the page keeps the same geometry).
    geometryChanged = QtCore.Signal()

    def __init__(self, project_root: str | Path, *, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Stitch calibration')
        self.setObjectName('stitchCalibrationDialog')
        self.setModal(False)
        self.resize(940, 730)
        self.project_root = Path(project_root).resolve()
        self.session_dir = self.project_root / 'calibration' / f'session-{uuid.uuid4().hex[:12]}'
        self.frames = ()
        self.fit = None
        #: The arm fitted to the calibrated frames: (their identities, record); see _seed_options.
        self._prediction = None
        #: Predicted rings per (frame, calibrant, calibration): see _expected_rings.
        self._ring_cache = {}
        self._choices = []
        self._identity = None
        self._job = None
        self._row_identity = None
        self._close_pending = False
        self._distance_result = None
        self._distance_energy_eV = None  # the fitted direct-beam scan's recorded energy
        self._direct_beam_fit_window = None
        self._distance_key = self._distance_pending_key = None  # what the result was fitted from
        #: The arm the staged frames read: (del, nu), or (2θ, None) for fourc/twoc.
        self._frame_motors = ('del', 'nu')
        #: The instrument the current fit recorded; any other value leaves it unsaveable.
        self._fit_instrument = None
        self.saved_source_motors = {}
        self.saved_instrument_geometry = None
        self.saved_image_rotation = 0
        self.saved_mask = ''
        self._owner = SingleWorkerOwner(_run_job)
        self._fields = {}
        #: key -> (form, row widget); a field sharing another's row is in _parts instead
        self._rows = {}
        self._parts = {}
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
        self._build_fit()
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
        # A tab taller than the window scrolls rather than squeezing its rows (wrapped notes clip).
        page = QtWidgets.QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        page.setWidget(widget)
        self.tabs.addTab(page, title)
        self._tab_pages[key] = page
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

    def _choice(self, form, key, label, items, default, *, tip=''):
        """A Controls combo read and saved like a text field (its data: a factory name, a code)."""
        combo = self._combo(form, label, items, tip=tip)
        self._fields[key] = combo
        self._rows[key] = (form, combo)
        self._set_field_text(key, default)
        return combo

    def _edit(self, form, key, label, default='', *, chooser=None, tip='', placeholder=''):
        """One text field, with the Controls 📁 button when ``chooser`` names a file filter,
        'directory' or 'spec' (SPEC files only), or is a function returning one at the click."""
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
        from xdart.gui.widgets.file_dialogs import browse, browse_spec
        chooser = chooser() if callable(chooser) else chooser
        folder = chooser == 'directory'
        if chooser == 'spec':
            picks = browse_spec(self, 'Select SPEC file', current=edit.text(), fallback=str(self.project_root))
        else:
            picks = browse(self, 'Select images directory' if folder else 'Select file', folder=folder,
                           file_filter='All files (*)' if folder else chooser, current=edit.text(),
                           fallback=str(self.project_root))
        if picks and picks[0]:
            edit.setText(picks[0])

    def _set_field_text(self, key, value):
        field, value = self._fields[key], str(value)
        if isinstance(field, QtWidgets.QComboBox):
            index = next((row for row in range(field.count())
                          if value.casefold() in (str(field.itemData(row)).casefold(), field.itemText(row).casefold())), -1)
            if index < 0 and value:  # a name this pyFAI lists differently, or no longer: kept, not lost
                field.addItem(value, value)
                index = field.count() - 1
            if index >= 0:
                field.setCurrentIndex(index)
            return
        field.setText(value)

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
        self._edit(form, 'source', 'SPEC file',
                   chooser=lambda: 'spec' if self.source_kind.currentData() == 'spec' else NEXUS_FILTER,
                   tip='The calibrant scan: its SPEC file, or its NeXus file.')
        self._attach('scan', 'source', 'Scan', '1', tip=self._SCAN_TIP, width=90)
        self._edit(form, 'images', 'Images folder', chooser='directory', tip=self._FOLDER_TIP)
        self._attach('stem', 'images', 'Stem', '', tip=self._STEM_TIP)
        self._edit(form, 'entry', 'NeXus entry', 'entry', tip='The NeXus entry (group) holding the frames and angles.')
        self.refine_calibrant = self._combo(form, 'Calibrant', [(name, None) for name in (
            'LaB6', 'CeO2', 'Si', 'Al2O3', 'Cr2O3', 'AgBh')],
            tip='The powder standard in this scan: pyFAI opens each frame with it, the table predicts its '
                'rings, and the refinement fits them. Another pyFAI calibrant name may be typed.')
        self.refine_calibrant.setEditable(True)
        self.refine_calibrant.currentTextChanged.connect(self._render_frames)
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
        for key, shown in (('scan', spec), ('images', spec), ('entry', not spec)):
            self._show(key, shown)
        self._label('source').setText('SPEC file' if spec else 'NeXus file')

    def _build_frames(self):
        # Calibrating the frames and fitting them are one loop: one tab (maintainer, 2026-09-29)
        layout, form = self._form('3 · Calibrate and fit', 'calibrate')
        self._calibrate_layout = layout
        hint = QtWidgets.QLabel('Calibrate each frame in pyFAI: pick its rings, then press Save as PONI-file. xdart keeps the PONI and its enabled control points, closes pyFAI and selects the next frame. Completed rows and files remain when another frame is cancelled.')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.frame_table = QtWidgets.QTableWidget(0, 6)
        self.frame_table.setHorizontalHeaderLabels(['Frame', 'Delta / Nu (°)', 'Image', 'PONI / points', 'Status',
                                                    'Expected rings'])
        self.frame_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.frame_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.frame_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column, tip in ((1, 'The detector circles\' angles for this frame.'),
                            (3, 'The PONI and control points pyFAI saved for this frame.'),
                            (5, 'The calibrant rings predicted to reach this frame; hover a cell for each '
                                'ring\'s 2θ.')):
            self.frame_table.horizontalHeaderItem(column).setToolTip(tip)
        # the fit shares the tab: the table keeps room for about six frames, the tab scrolls
        self.frame_table.setMinimumHeight(self.frame_table.verticalHeader().defaultSectionSize() * 6
                                          + self.frame_table.horizontalHeader().sizeHint().height() + 4)
        layout.addWidget(self.frame_table, 1)
        buttons = QtWidgets.QHBoxLayout()
        self.calibrate_button = QtWidgets.QPushButton('Calibrate selected frame…')
        self.calibrate_button.setToolTip('Opens the selected frame in pyFAI with its predicted geometry: pick '
                                         'its rings there, then press Save as PONI-file.')
        self.calibrate_button.clicked.connect(self._calibrate)
        self.import_button = QtWidgets.QPushButton('Import matched PONI + NPT…')
        self.import_button.setToolTip('Adopts a PONI and its control points (.npt), made elsewhere, for '
                                      'exactly the selected frame.')
        self.import_button.clicked.connect(self._import)
        buttons.addWidget(self.calibrate_button)
        buttons.addWidget(self.import_button)
        layout.addLayout(buttons)
        self.session_label = QtWidgets.QLabel('Persistent calibration session will be saved inside the project.')
        self.session_label.setWordWrap(True)
        layout.addWidget(self.session_label)

    def _card(self, layout, title):
        """A collapsed Controls subsection holding one more form (options most runs leave alone)."""
        from xdart.gui.widgets.controls_panel import SubsectionCard
        card = SubsectionCard(title)
        body = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(body)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        card.add_row(body)
        card.set_collapsed(True)
        layout.addWidget(card)
        return card, form

    def _build_fit(self):
        layout = self._calibrate_layout  # under the frames it fits
        rule = QtWidgets.QFrame()
        rule.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        rule.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(rule)
        self.arm_note = QtWidgets.QLabel()
        self.arm_note.setWordWrap(True)
        self.arm_note.setToolTip('The arm model is chosen for you: the detector mount (tilt, roll) and the '
                                 'sample offset whenever the frames can determine them, else the '
                                 'five-parameter arm (the distance, the beam position and the circles\' '
                                 'zero offsets). The fit summary says which was fitted.')
        layout.addWidget(self.arm_note)
        # The energy, refined within ± a bound of the recorded one (maintainer, 2026-09-29: "add a
        # refine energy toggle. If selected there should be an option to set the constraint - just
        # one number for +-that number"); Fit and Refine both take it. On by default, ±100 eV
        # (maintainer, 2026-09-29, D1: "+-100 and on by default"): with the sample held near the
        # rotation centre, a fixed energy's error lands in the zeros and the sample offset.
        energy_row = QtWidgets.QWidget()
        energy_layout = QtWidgets.QHBoxLayout(energy_row)
        energy_layout.setContentsMargins(0, 0, 0, 0)
        self.refine_energy = QtWidgets.QCheckBox('Refine energy')
        energy_tip = ('Also fits the beam energy, within ± the bound of the frames\' recorded energy. The arm\'s '
                      'known angles give an absolute 2θ scale, so the calibrant rings can fix it; with frames on only '
                      'one side of zero it trades off against the zero offsets and the sample offset, and the fit '
                      'summary gives its uncertainty. The saved calibration keeps the recorded energy, which '
                      'Stitch checks the scans\' energies against.')
        self.refine_energy.setToolTip(energy_tip)
        self.refine_energy.setChecked(True)
        bound = QtWidgets.QLineEdit('100')
        bound.setObjectName('energy_bound')
        bound.setMaximumWidth(64)
        bound.setToolTip('The refined energy stays within this many eV of the recorded one.')
        self._fields['energy_bound'] = bound
        energy_layout.addWidget(self.refine_energy)
        energy_layout.addWidget(QtWidgets.QLabel('±'))
        energy_layout.addWidget(bound)
        energy_layout.addWidget(QtWidgets.QLabel('eV'))
        energy_layout.addStretch(1)
        layout.addWidget(energy_row)
        self.fit_button = QtWidgets.QPushButton('Fit the calibrated frames')
        self.fit_button.setToolTip('Fits one detector-arm geometry to the ring points picked on the frames '
                                   'marked Complete above. Needs three of them.')
        self.fit_button.clicked.connect(self._fit)
        layout.addWidget(self.fit_button)
        self._advanced_card, options = self._card(layout, 'Advanced')
        self.scales = QtWidgets.QCheckBox('Fit motor scales')
        self.scales.setChecked(False)
        self.scales.setToolTip('Also fits each motor\'s scale (degrees of arm rotation per motor degree), '
                               'where the frames span enough angle to tell. Off: exactly one degree per '
                               'degree, which is right for almost every diffractometer. A calibration with '
                               'fitted scales has no xrayutilities file: xrayutilities takes angles in degrees.')
        options.addRow(self.scales)
        self._edit(options, 'rms', 'Maximum RMS |q| (Å⁻¹)', '.03',
                   tip='A fit whose RMS ring-position error, in |q|, exceeds this is refused.')
        self._edit(options, 'distance_seed', 'Distance seed (mm)', '',
                   tip='Where the fit starts, e.g. the direct-beam distance. Blank: the first calibrated '
                       'frame\'s.')
        self._edit(options, 'distance_bounds', 'Distance bounds (min, max mm)', '',
                   tip='Keeps the fitted distance between these. Blank: within the distance freedom below.')
        # The freedom each fit keeps about its estimates (maintainer, 2026-09-29: "this is to make
        # sure the sample displacement in particular doesn't go to unrealistic values")
        for key, label, tip in self._BOUND_FIELDS:
            self._edit(options, key, label, self._BOUND_DEFAULTS[key], tip=tip)
        self.bounds_note = QtWidgets.QLabel()
        self.bounds_note.setWordWrap(True)
        options.addRow(self.bounds_note)
        self.reset_bounds_button = QtWidgets.QPushButton('Reset defaults')
        self.reset_bounds_button.setToolTip('The maintainer\'s defaults: ± 2° for the zeros and the mount, '
                                            '± 0.05 mm for each sample displacement, ± 5 % for the distance.')
        self.reset_bounds_button.clicked.connect(self._reset_bounds)
        options.addRow(self.reset_bounds_button)
        range_tip = ('The {} angles Stitch with xrayutilities may use this calibration for: its xrayutilities '
                     'file is checked on every pixel over this range (with the calibrant frames and the scan\'s '
                     'frames), and Stitch refuses frames outside what was checked. Blank: the calibrant '
                     'scan\'s span. pyFAI MultiGeometry has no such limit.')
        self._edit(options, 'del_domain', 'Supported del range (min, max °)', '', tip=range_tip.format('del'))
        self._edit(options, 'nu_domain', 'Supported nu range (min, max °)', '', tip=range_tip.format('nu'))
        self._refine_card, auto = self._card(layout, 'Refine from the whole scan (optional)')
        self.refine_seed = self._combo(auto, 'Start from', [('Current fit (unsaved)', 'fit'), ('Geometry file', 'file'),
                                                            ('Rough geometry (1 · Geometry)', 'rough')],
            tip='Current fit: the fit just made above. Geometry file: a geometry this window saved. Rough '
                'geometry: the distance and direct-beam pixel of 1 · Geometry, with no frame calibrated '
                '(on 17-2\'s LaB6 scan it refines about as well as from a ring fit).')
        self._edit(auto, 'refine_seed_path', 'Geometry file', chooser='JSON (*.json);;All files (*)',
                   tip='A geometry this window saved (a stacked or single detector arm). Until this session '
                       'has its own fit, it also predicts each frame\'s rings.')
        self.refine_seed_note = QtWidgets.QLabel()
        self.refine_seed_note.setWordWrap(True)
        auto.addRow(self.refine_seed_note)
        # the same calibration predicts each frame's rings (Calibrate each frame)
        self._fields['refine_seed_path'].editingFinished.connect(self._render_frames)
        self._fields['refine_seed_path'].textChanged.connect(self._sync_refine_rows)
        self.refine_seed.currentIndexChanged.connect(self._sync_refine_rows)
        self._edit(auto, 'refine_frames', 'Frame budget', '48',
                   tip='At most this many frames of the calibrant scan are searched for ring points, spread '
                       'across its angles. More: better coverage, slower.')
        self.refine_button = QtWidgets.QPushButton('Refine automatically')
        self.refine_button.setToolTip('Finds ring points on frames across the whole calibrant scan, starting '
                                      'from the geometry above, and keeps the detector mount and sample offset '
                                      'only when they predict held-out nu rows better.')
        self.refine_button.clicked.connect(self._refine)
        auto.addRow(self.refine_button)
        self.fit_summary = QtWidgets.QPlainTextEdit()
        self.fit_summary.setReadOnly(True)
        self.fit_summary.setMinimumHeight(self.fit_summary.fontMetrics().lineSpacing() * 8)
        layout.addWidget(self.fit_summary, 1)
        self.save_button = QtWidgets.QPushButton('Save geometry and use in Stitch')
        self.save_button.setToolTip('Saves the fitted geometry inside the project as a pyFAI goniometer JSON and, '
                                    'beside it, the same calibration for xrayutilities (<name>.xu.json, checked '
                                    'against pyFAI on every pixel), and makes it Stitch\'s calibration: the Backend '
                                    'under Processing chooses which of the two Stitch uses.')
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)
        layout.addWidget(self.save_button)
        for key in ('rms', 'distance_seed', 'distance_bounds', 'energy_bound', *self._BOUND_DEFAULTS):
            self._fields[key].textChanged.connect(self._invalidate_fit)
        for key in self._BOUND_DEFAULTS:
            self._fields[key].textChanged.connect(self._bounds_edited)
        self.scales.toggled.connect(self._invalidate_fit)
        self.refine_energy.toggled.connect(self._invalidate_fit)
        self.refine_energy.toggled.connect(self._fields['energy_bound'].setEnabled)
        self._fields['energy_bound'].setEnabled(self.refine_energy.isChecked())
        self.scales.toggled.connect(self._sync_arm_model)
        self._sync_arm_model()
        self._sync_refine_rows()
        self._bounds_edited()

    def _sync_refine_rows(self, *_args):
        """Say at once whether the geometry file can seed the refinement (a detector arm, not an
        older independent-angle calibration)."""
        from_file = self.refine_seed.currentData() == 'file'
        self._show('refine_seed_path', from_file)
        note = (self._seed_file_problem(self._text('refine_seed_path'))
                if from_file and self._text('refine_seed_path') else None) or ''
        self.refine_seed_note.setText(note)
        self.refine_seed_note.setVisible(bool(note))

    @staticmethod
    def _seed_file_problem(path):
        """Why this geometry file cannot start the refinement, or None (a detector arm)."""
        from xrd_tools.core.geometry.stacked_arm import is_arm_record
        try:
            record = json.loads(Path(path).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return 'This file cannot be read as a geometry JSON.'
        if not is_arm_record(record):
            return ('This file uses independent angles (an older MultiGeometry format), so it cannot start the '
                    'refinement. Fit the calibrated frames and start from the current fit.')
        return None

    def _build_distance(self):
        layout, form = self._form('1 · Geometry', 'rough')
        self._beam_form = form
        from xdart.gui.widgets.geometry_editor import GeometryEditor
        diffractometer = QtWidgets.QGroupBox('Diffractometer')
        box = QtWidgets.QVBoxLayout(diffractometer)
        self.geometry_editor = GeometryEditor(used=('nu', 'del', 'tth'), used_by='Stitch')
        box.addWidget(self.geometry_editor)
        top = QtWidgets.QHBoxLayout()  # the instrument: its circles, and the detector every step reads
        top.addWidget(diffractometer, 1)
        top.addWidget(self._build_detector(), 1)
        layout.insertLayout(0, top)
        self.geometry_editor.changed.connect(self._geometry_edited)
        self.geometry_editor.changed.connect(self.geometryChanged.emit)
        self.geometry_editor.changed.connect(self._bounds_edited)
        # the arm's motors are these circles' (read by name as before)
        self._fields['del_motor'] = self.geometry_editor.motor_edit('del')
        self._fields['nu_motor'] = self.geometry_editor.motor_edit('nu')
        self.rough_mode = self._combo(form, 'Rough geometry', [('Direct-beam scan', 'scan'),
                                                               ('Enter the distance', 'manual'), ('Skip', 'skip')],
            tip='A first guess of where the detector is, so pyFAI opens each frame with its rings roughly in '
                'place and the table lists which rings to expect. A direct-beam scan measures the distance; '
                'or type it. Rough values are enough: once one frame is calibrated, the others are predicted '
                'from it instead.')
        self._edit(form, 'rough_distance', 'Sample–detector distance (mm)', '',
                   tip='Sample to detector, in millimetres, with the detector facing the beam squarely.')
        self.beam_source_kind = self._combo(form, 'Source', [('SPEC + image series', 'spec_series'),
                                                             ('SPEC + Eiger / NeXus', 'spec_container'),
                                                             ('NeXus images + metadata', 'nexus')],
            tip='Where the direct-beam scan\'s angles and images come from.')
        self._edit(form, 'beam_source', 'SPEC file',
                   chooser=lambda: NEXUS_FILTER if self.beam_source_kind.currentData() == 'nexus' else 'spec',
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
        self._edit(form, 'beam_pixel', 'Approximate beam pixel x, y (oriented image)', '',
                   tip='Column, row of the pixel the direct beam hits with the detector facing it squarely (each '
                       'circle at its motor zero, beside its motor above), in the image as displayed; it is not '
                       'necessarily the centre. '
                       'Blank with a direct-beam scan: found on the scan\'s frame nearest the zero (the '
                       'brightest spot, a lone hot pixel ignored). Typed: along the beam\'s motion it can be '
                       'rough; across it (the row for horizontal motion, the column for vertical) it must be '
                       'within the strip half width, because the spot is searched only in that strip.')
        self.motion = self._combo(form, 'Beam motion', [('horizontal', None), ('vertical', None)],
                                  tip='The direction the beam spot moves across the image as the motor turns.')
        self._edit(form, 'rough_mount', 'Detector mount tilt, roll (°)', '0, 0',
                   tip='The detector mount\'s tilt and roll, held fixed (the arm\'s mount extension; a '
                       'single 2θ arm has none). 0, 0: the detector faces the beam squarely at every angle.')
        self._edit(form, 'rough_wavelength', 'Wavelength (Å)', '',
                   placeholder="blank: the calibrant scan's energy, else the direct-beam scan's",
                   tip='The X-ray wavelength of this geometry. Blank: the calibrant scan\'s recorded energy (2 · '
                       'Select frames), else the direct-beam scan\'s once it is fitted, when they record one. '
                       'Save as calibration and a refinement started from this geometry need it (Stitch checks '
                       'each scan\'s energy against it).')
        self._fit_options, options = self._card(layout, 'Fit options')
        self._edit(options, 'beam_strip', 'Strip half width (pixels)', '2',
                   tip='The beam spot is searched in a strip this many pixels either side of its path.')
        self._edit(options, 'beam_roi', 'ROI x0, y0, x1, y1 (optional)', '',
                   tip='Search only this pixel box (columns x0–x1, rows y0–y1).')
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
        self.show_fit_button = QtWidgets.QPushButton('Show the fit…')
        self.show_fit_button.setToolTip('The beam centre against the motor angle, the fitted model and its '
                                        'residuals, with the distance (it opens after every fit).')
        self.show_fit_button.clicked.connect(self._show_direct_beam_fit)
        self.show_fit_button.setEnabled(False)
        layout.addWidget(self.show_fit_button)
        self.apply_distance_button = QtWidgets.QPushButton('Use result as distance seed')
        self.apply_distance_button.setToolTip('Starts the final fit (3 · Calibrate and fit) from this distance.')
        self.apply_distance_button.clicked.connect(self._apply_distance)
        layout.addWidget(self.apply_distance_button)
        # A calibration from this geometry alone (maintainer, 2026-09-29: "a quick multigeometry
        # calibration json file from just the sample detector distance, and direct beam pixel")
        self.save_rough_button = QtWidgets.QPushButton('Save as calibration…')
        self.save_rough_button.setToolTip(
            'Saves this geometry, with no ring fit, as Stitch\'s calibration: the distance and beam pixel '
            'above, the rotations held as typed, one degree per degree. As Save does, it writes the pyFAI '
            'file and, beside it, the xrayutilities file, checked on the direct-beam scan\'s angles, the '
            'calibrant scan\'s and the supported ranges under 3 · Calibrate and fit ▸ Advanced. Useful when '
            'nu stays at 0; the file records that it is a direct-beam calibration.')
        self.save_rough_button.clicked.connect(self._save_rough)
        layout.addWidget(self.save_rough_button)
        layout.addStretch(1)
        self.rough_mode.currentIndexChanged.connect(self._sync_rough_rows)
        for key in ('rough_distance', 'beam_pixel', 'beam_motor', 'rough_mount', 'rough_wavelength'):
            self._fields[key].editingFinished.connect(self._render_frames)
        self._fields['rough_mount'].textChanged.connect(self._invalidate_fit)  # the fits' mount estimate
        self._fields['rough_mount'].textChanged.connect(self._bounds_edited)
        for circle in ('nu', 'del', 'tth'):  # the zeros place the rough arm and start every fit
            self.geometry_editor.zero_edit(circle).editingFinished.connect(self._render_frames)
            self.geometry_editor.zero_edit(circle).textChanged.connect(self._bounds_edited)
        for key in ('beam_detector', 'beam_orientation'):
            self._fields[key].currentIndexChanged.connect(self._render_frames)
        self._fields['beam_detector'].currentIndexChanged.connect(self._detector_chosen)
        self._fields['beam_detector'].currentIndexChanged.connect(self._update_image_size)
        self.rotation.currentIndexChanged.connect(self._update_image_size)  # the beam pixel's image turns
        self._update_image_size()

    def _build_detector(self):
        """The detector every step reads the images with: its model, the stored image's layout and
        turn, its saturation and one mask for the whole calibration (maintainer, 2026-09-29)."""
        panel = QtWidgets.QGroupBox('Detector')
        form = QtWidgets.QFormLayout(panel)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._choice(form, 'beam_detector', 'Model', [(label, name) for label, name in detector_choices()],
                     'Pilatus300kw', tip='The detector model (pyFAI); it sets the pixel size and shape.')
        reader = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(reader)
        row.setContentsMargins(0, 0, 0, 0)
        raw_tip = ('Only for headerless .raw images (the calibrant and direct-beam scans): the frame\'s rows '
                   'and columns (set from the model), pixel type and header size.')
        for key, label, default, stretch in [('shape', 'rows, cols', '195, 1475', 3), ('dtype', 'type', 'int32', 2),
                                             ('header', 'header', '0', 1)]:
            caption = QtWidgets.QLabel(label)
            edit = QtWidgets.QLineEdit(default)
            edit.setObjectName(key)
            edit.setMinimumWidth(edit.fontMetrics().horizontalAdvance(default + '0'))
            self._fields[key] = edit
            self._tip(raw_tip, caption, edit)
            row.addWidget(caption)
            row.addWidget(edit, stretch)
        form.addRow('RAW images', reader)
        self._rows['raw'] = (form, reader)
        self._tip(raw_tip, form.labelForField(reader))
        self.rotation = self._combo(form, 'Image rotation (applied once)',
                                    [(f'{value}° counterclockwise', value) for value in (0, 90, 180, 270)],
                                    tip='Rotates every image once before calibration; use it when the stored '
                                        'images are turned relative to the detector view you calibrate in.')
        self._image_size = QtWidgets.QLabel()
        self._image_size.setWordWrap(True)
        form.addRow('Image size', self._image_size)
        self._rows['beam_size'] = (form, self._image_size)
        self._tip('The detector image\'s width (x, columns) and height (y, rows) in pixels, as the beam '
                  'pixel is read: the detector\'s own layout, turned by the image rotation above (90° and '
                  '270° swap x and y). The first-pixel corner below does not change it.',
                  self._image_size, form.labelForField(self._image_size))
        corners = {3: 'bottom right (pyFAI default)', 1: 'top left', 2: 'top right', 4: 'bottom left'}
        self._choice(form, 'beam_orientation', 'First pixel (pyFAI orientation)',
                     [(f'{code} · first pixel {corners[code]}', code) for code, _name in DETECTOR_ORIENTATIONS], 3,
                     tip='pyFAI\'s detector orientation: the corner holding the image\'s first pixel, seen '
                         'from behind the detector looking at the sample. It is not a rotation: it never turns '
                         'the image or swaps x and y (the image rotation above does). It changes how the PONIs '
                         'are written, not the geometry: the arm\'s motor scales take its signs. Keep 3, '
                         'pyFAI\'s default, unless pyFAI shows this detector\'s images mirrored. It is saved '
                         'with every PONI and fit.')
        for combo in (self._fields['beam_detector'], self.rotation, self._fields['beam_orientation']):
            # half the window's width: long entries elide instead of widening the panel
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        self._edit(form, 'saturation', 'Saturation (counts)', '',
                   tip='Counts at or above this are saturated: a direct-beam frame with such a pixel in its '
                       'beam strip is left out, and the automatic refinement picks no ring points above it. '
                       'Blank: the image type\'s maximum.')
        self._edit(form, 'mask', 'Mask file', chooser='Masks (*.edf *.tif *.tiff *.npy);;All files (*)',
                   tip='Pixels to leave out everywhere in this calibration: the direct-beam fit, pyFAI-calib2 '
                       '(its mask), the automatic refinement\'s ring search and the expected rings. Drawn on '
                       'the stored image (before the image rotation), as the Stitch Mask File is; saving the '
                       'calibration makes it Stitch\'s Mask File.')
        self._fields['mask'].editingFinished.connect(self._render_frames)
        return panel

    _ROUGH_NOTES = {
        'scan': 'A direct-beam scan: SPEC supplies the angles for an image series or an Eiger/NeXus container; '
                'standalone NeXus supplies both. Angle and image counts must match. Uses the Detector panel\'s '
                'image rotation, RAW settings, saturation and mask. Its distance, with the beam pixel and zero, '
                'predicts each frame\'s rings; the regression uncertainty excludes motor, zero and pitch systematics.',
        'manual': 'The distance, the pixel the direct beam hits at normal incidence and that zero predict each '
                  'frame\'s rings (the Geometry\'s arm, one degree per degree) until the first frame is '
                  'calibrated; that frame then predicts the others, and the fit refines them all.',
        'skip': 'No rough geometry: each frame\'s rings are predicted from a geometry file (3 · Calibrate and fit), '
                'or from the first frame you calibrate.',
    }
    _GEOMETRY_ROWS = ('beam_motor', 'beam_pixel', 'rough_mount', 'rough_wavelength')

    def _update_image_size(self, *_args):
        """The detector image's x × y size, as the beam pixel is read (after the Select-frames rotation)."""
        shape = detector_shape(self._text('beam_detector'))
        if shape is None:
            self._image_size.setText('unknown detector')
            self._fields['beam_pixel'].setPlaceholderText('column, row')
            return
        rows, columns = shape
        rotation = int(self.rotation.currentData() or 0)
        width, height = (rows, columns) if rotation in (90, 270) else (columns, rows)
        self._image_size.setText(f'{width} × {height} px (x × y): x 0–{width - 1}, y 0–{height - 1} · '
                                 f'image rotation {rotation}°')
        self._beam_pixel_hint()

    def _beam_pixel_hint(self, *_args):
        """A direct-beam scan finds the beam pixel when it is blank; a typed distance needs it."""
        if not hasattr(self, 'rough_mode'):  # still being built
            return
        found = getattr(self._distance_result, 'beam_pixel', None)
        self._fields['beam_pixel'].setPlaceholderText(
            'column, row where the direct beam hits at the zero' if self.rough_mode.currentData() != 'scan'
            else f'auto · the direct beam found at {found[0]:.1f}, {found[1]:.1f}' if found is not None
            else 'auto · where the direct beam hits, found on the frame nearest the zero')

    def _detector_chosen(self, *_args):
        """The chosen detector's shape becomes the RAW reader's rows and columns (Select frames)."""
        shape = detector_shape(self._text('beam_detector'))
        if shape is not None and 'shape' in self._fields:
            self._fields['shape'].setText(f'{shape[0]}, {shape[1]}')

    def _sync_rough_rows(self, *_args):
        """Every Geometry row's visibility, from the rough mode and the direct-beam source together."""
        mode, kind = self.rough_mode.currentData(), self.beam_source_kind.currentData()
        scan = mode == 'scan'
        visible = {'rough_distance': mode == 'manual', 'beam_source': scan,
                   'beam_scan': scan and kind != 'nexus', 'beam_image_file': scan and kind == 'spec_container',
                   'beam_images': scan and kind == 'spec_series', 'beam_stem': scan and kind == 'spec_series',
                   'beam_entry': scan and kind != 'spec_series',
                   **dict.fromkeys(self._GEOMETRY_ROWS, mode != 'skip')}
        form = self._beam_form
        for key, shown in visible.items():
            self._show(key, shown)
        form.setRowVisible(self.beam_source_kind, scan)
        form.setRowVisible(self.motion, scan)
        for widget in (self._fit_options, self.beam_button, self.beam_summary, self.show_fit_button,
                       self.apply_distance_button):
            widget.setVisible(scan)
        self.save_rough_button.setVisible(mode != 'skip')
        self._label('beam_source').setText('NeXus file' if kind == 'nexus' else 'SPEC file')
        self.rough_note.setText(self._ROUGH_NOTES[mode])
        self._beam_pixel_hint()
        if hasattr(self, 'frame_table'):
            self._render_frames()

    def _rough_record(self):
        """Step 1 as a nominal arm, or None (skipped, not yet measured, or incomplete)."""
        try:
            return self._rough_arm()
        except (ValueError, TypeError, KeyError, RuntimeError):
            return None

    def _rough_arm(self):
        """Step 1 as a nominal arm, None when skipped or not yet measured; malformed input raises."""
        mode = self.rough_mode.currentData()
        if mode == 'manual' and self._text('rough_distance'):
            distance = float(self._text('rough_distance')) / 1000
        elif mode == 'scan' and self._distance_result is not None:
            if not self._distance_fresh():  # its zero, source or detector changed since (Codex review F5)
                raise ValueError(self._STALE_DISTANCE)
            distance = self._distance_result.distance_m
        else:
            return None
        # the Geometry shown owns tab 1's arm, not the arm frames were staged with (Codex review F4)
        geometry = self.geometry_editor.value()
        delta, nu = geometry.stitch_arm()
        # blank with a measured scan: the pixel the fit found
        # a fitted scan says where the beam hits at the zero (the fit's intercept along the motion);
        # a typed or found pixel only placed its strips (Codex review F1)
        beam = (self._distance_result.zero_beam_pixel if mode == 'scan'
                else tuple(float(v) for v in self._text('beam_pixel').split(',')) if self._text('beam_pixel')
                else None)
        if beam is None:
            return None
        mount = tuple(float(v) for v in (self._text('rough_mount') or '0, 0').split(','))
        if len(mount) != 2:
            raise ValueError('type the mount tilt and roll as two numbers, e.g. 0, 0')
        wavelength = self._rough_wavelength_A()
        return rough_arm_record(
            distance_m=distance, detector=self._text('beam_detector'),
            beam_px=beam, zeros_deg=self._zeros(),  # both moving circles at their zeros
            del_motor=delta, nu_motor=nu,
            orientation=int(self._text('beam_orientation')), mount_deg=mount,
            wavelength_m=wavelength * 1e-10 if wavelength is not None else 1e-10)

    def _rough_wavelength_A(self):
        """The typed wavelength (Å), else the calibrant scan's when every frame read records the same
        energy, else the direct-beam scan's (maintainer, 2026-09-29: "why doesn't it get it from the
        spec file?") while its fit is current, else None."""
        if self._text('rough_wavelength'):
            return float(self._text('rough_wavelength'))
        from xrd_tools.core.energy import energy_eV_to_wavelength_m
        energies = {energy for _index, _values, energy in self._choices if energy is not None}
        if len(energies) == 1:
            return float(energy_eV_to_wavelength_m(energies.pop())) * 1e10
        if (not energies and self.rough_mode.currentData() == 'scan' and self._distance_fresh()
                and self._distance_energy_eV is not None):
            return float(energy_eV_to_wavelength_m(self._distance_energy_eV)) * 1e10
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
            if self.fit is None and self._seed_file_problem(geometry_path) is None:  # an arm it can start from
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

    _STALE_DISTANCE = ('The direct-beam fit was made with other inputs (a motor zero, its motor, the source, the '
                       'detector or the search settings changed since): fit the direct-beam distance again.')

    #: The freedom every fit keeps about its estimates (maintainer, 2026-09-29), with defaults.
    _BOUND_DEFAULTS = {'zero_freedom': '2', 'mount_freedom': '2', 'displacement_limit': '0.05',
                       'distance_freedom': '5'}
    _BOUND_FIELDS = (
        ('zero_freedom', 'Zero freedom (± °)',
         "Each moving circle's motor zero is fitted within this many degrees of its estimate (beside its motor "
         "under 1 · Geometry ▸ Diffractometer; blank: 0, or the seed calibration's own when refining from "
         "one), and a single arm's held angle within as many of its own."),
        ('mount_freedom', 'Mount freedom (± °)',
         "The detector mount's tilt and roll are fitted within this many degrees of the mount under "
         "1 · Geometry (Detector mount tilt, roll)."),
        ('displacement_limit', 'Sample displacement limit (± mm)',
         "Each component of the sample's offset from the rotation centre is fitted within this of zero: "
         "0.05 mm by default, 0.02 mm for a sample known to be well centred. A displacement resting on "
         "its limit is reported, never widened."),
        ('distance_freedom', 'Distance freedom (± %)',
         'Without Distance bounds, the distance is fitted within this percentage of its seed.'),
    )

    #: Choices saved with the Stitch settings. The calibrant-scan and same-detector
    #: confirmations are per use and never saved.
    _SAVED_CHOICES = ('source_kind', 'rotation', 'refine_calibrant', 'refine_seed', 'beam_source_kind',
                      'motion', 'rough_mode')

    def settings(self):
        """This dialog's inputs as plain values, for the saved Stitch settings."""
        choices = {name: {'data': _plain_choice(getattr(self, name).currentData()),
                          'text': getattr(self, name).currentText()}
                   for name in self._SAVED_CHOICES}
        return {
            'fields': {key: self._text(key) for key in self._fields},
            'choices': choices,
            'fit_motor_scales': self.scales.isChecked(),
            'refine_energy': self.refine_energy.isChecked(),
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
        fields = dict(values.get('fields') or {})
        for old, new in (('beam_mask', 'mask'), ('beam_saturation', 'saturation'), ('refine_max_counts', 'saturation')):
            if fields.get(old) and not fields.get(new):  # an older window's field, now the Detector panel's
                fields[new] = fields[old]
        for key, text in fields.items():
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
            combo.setCurrentIndex(index)
        if isinstance(values.get('fit_motor_scales'), bool):
            self.scales.setChecked(values['fit_motor_scales'])
        if isinstance(values.get('refine_energy'), bool):
            self.refine_energy.setChecked(values['refine_energy'])
        # after the del/nu fields an older file names its arm with: its normal-incidence zero
        self.geometry_editor.set_value(migrated_instrument(self.geometry_editor.value(), values))
        self._sync_arm_model()

    def _zeros(self):
        """The Diffractometer panel's motor zeros by arm circle ('del', 'nu'); a malformed box is
        refused by its circle's name, never read as blank (Codex review F4)."""
        malformed = self.geometry_editor.invalid_zeros()
        if malformed:
            raise ValueError(f"type {malformed[0]}'s motor zero as a number of degrees, or leave it blank")
        return self.geometry_editor.value().arm_zeros()

    def _bounds(self):
        """The fits' estimates and freedom: each moving circle's zero from the Diffractometer panel
        (blank: 0, or the seed's when refining), the mount from 1 · Geometry, the freedom under
        Advanced. A malformed entry raises, naming it: a bound is never guessed."""
        from xrd_tools.integrate.refine import ArmFitBounds
        values = {}
        for key, label, _tip in self._BOUND_FIELDS:
            try:
                values[key] = float(self._text(key))
            except ValueError:
                raise ValueError(f'type {label} as a positive number') from None
        zeros = self._zeros()
        try:
            mount = tuple(float(v) for v in (self._text('rough_mount') or '0, 0').split(','))
        except ValueError:
            mount = ()
        if len(mount) != 2:
            raise ValueError('type the mount tilt and roll as two numbers, e.g. 0, 0')
        try:
            return ArmFitBounds(zeros_deg=zeros, mount_deg=mount,
                                zero_freedom_deg=values['zero_freedom'], mount_freedom_deg=values['mount_freedom'],
                                sample_offset_mm=values['displacement_limit'],
                                distance_fraction=values['distance_freedom'] / 100)
        except ValueError as error:
            raise ValueError(f'fit bounds: {error}') from None

    def _distance_options(self):
        """The Advanced distance seed and bounds (metres), for every fit and the prediction alike
        (Codex review F3); a malformed entry is refused by name."""
        try:
            seed = float(self._text('distance_seed')) / 1000 if self._text('distance_seed') else None
        except ValueError:
            raise ValueError('type the Distance seed in mm, or leave it blank') from None
        try:
            bounds = (tuple(float(v) / 1000 for v in self._text('distance_bounds').split(','))
                      if self._text('distance_bounds') else None)
        except ValueError:
            bounds = ()
        if bounds is not None and len(bounds) != 2:
            raise ValueError('type the Distance bounds as min, max in mm, or leave them blank')
        return {'distance_seed_m': seed, 'distance_bounds_m': bounds}

    def _bounds_or_none(self):
        """The fits' bounds, or None while an entry is malformed (or the window is being built)."""
        try:
            return self._bounds()
        except (ValueError, KeyError):
            return None

    def _fit_policy(self):
        """The zeros, bounds and distance the fits take now, as a session records them (None while an
        entry is malformed)."""
        bounds, distance = self._bounds_or_none(), self._distance_or_none()
        if bounds is None or not distance:
            return None
        limits = distance['distance_bounds_m']
        return {'bounds': bounds.to_record(), 'distance_seed_m': distance['distance_seed_m'],
                'distance_bounds_m': None if limits is None else list(limits)}

    def _apply_fit_policy(self, policy):
        """A session's recorded zeros, bounds, mount and distance, into the window."""
        from xrd_tools.integrate.refine import ArmFitBounds
        bounds = ArmFitBounds.from_record(policy['bounds'])
        geometry, zeros = self.geometry_editor.value(), dict(bounds.zeros_deg)
        for role, motor in zip(('del', 'nu'), geometry.stitch_arm()):
            if motor is not None:  # the arm's circles: psic's own, or a single arm's 2θ
                circle = role if geometry.geometry == 'psic' else 'tth'
                self.geometry_editor.zero_edit(circle).setText('' if role not in zeros else f'{zeros[role]:.12g}')
        self._set_field_text('zero_freedom', f'{bounds.zero_freedom_deg:g}')
        self._set_field_text('mount_freedom', f'{bounds.mount_freedom_deg:g}')
        self._set_field_text('displacement_limit', f'{bounds.sample_offset_mm:g}')
        self._set_field_text('distance_freedom', f'{bounds.distance_fraction * 100:g}')
        self._set_field_text('rough_mount', ', '.join(f'{value:g}' for value in bounds.mount_deg))
        seed, limits = policy['distance_seed_m'], policy['distance_bounds_m']
        self._set_field_text('distance_seed', '' if seed is None else f'{seed * 1000:.12g}')
        self._set_field_text('distance_bounds', '' if limits is None else ', '.join(f'{v * 1000:.12g}' for v in limits))

    def _distance_or_none(self):
        try:
            return self._distance_options()
        except (ValueError, KeyError):
            return {}

    def _bounds_edited(self, *_args):
        """Say which estimates the fits start from (the prediction's key follows the bounds)."""
        if not hasattr(self, 'bounds_note'):  # still being built
            return
        bounds = self._bounds_or_none()
        if bounds is None:
            self.bounds_note.setText('Fit bounds: a value above is not a number; Fit and Refine say which.')
            return
        geometry = self.geometry_editor.value()
        zeros = geometry.arm_zeros()
        moving = [circle for circle, motor in zip(('del', 'nu'), geometry.stitch_arm()) if motor]
        parts = [f'{circle} zero {zeros[circle]:g}°' if circle in zeros else f"{circle} zero blank (0, or the seed's)"
                 for circle in moving]
        self.bounds_note.setText('Fits start from: ' + ', '.join(parts)
                                 + f"; mount tilt, roll {bounds.mount_deg[0]:g}, {bounds.mount_deg[1]:g}° (1 · Geometry)")

    def _reset_bounds(self):
        for key, value in self._BOUND_DEFAULTS.items():
            self._fields[key].setText(value)

    def _text(self, key):
        field = self._fields[key]
        if isinstance(field, QtWidgets.QComboBox):
            data = field.currentData()
            return str(field.currentText() if data is None else data).strip()
        return field.text().strip()

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
        self._guard(lambda: self._start('calibrate', (self._selected_frame(),
                                                      {**self._seed_options(), 'mask': self._pyfai_mask()})))

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
        prediction = (self._prediction[1] if self._prediction is not None
                      and self._prediction[0] == self._calibrated_key() else None)
        return dict(calibrant=self.refine_calibrant.currentText().strip() or 'LaB6',
                    del_motor=self._frame_motors[0], nu_motor=self._frame_motors[1], prediction=prediction,
                    record=self.fit.asset.get('calibration') if self.fit is not None else None,
                    frames=done, rough=self._rough_record(), bounds=self._bounds_or_none(),
                    distance=self._distance_or_none(),
                    file=str(Path(path).resolve()) if path and Path(path).is_file() else None,
                    detector=detector)

    def _calibrated_key(self):
        """The calibrated frames and the bounds the prediction was fitted with: editing a zero or
        a bound leaves a prediction stale."""
        bounds = self._bounds_or_none()
        return (tuple((frame.identity, frame.poni_sha256) for frame in self.frames if frame.complete),
                json.dumps([None if bounds is None else bounds.to_record(),
                            self._text('distance_seed'), self._text('distance_bounds')], sort_keys=True))

    def _expected_rings(self, frame):
        """The calibrant rings predicted on this frame (None without a calibration)."""
        seed = self._seed_options()
        record = next((value for value in (seed['record'], seed['prediction'], seed['rough'], seed['file'])
                       if value is not None), None)
        if record is None:
            return None
        import hashlib
        key = (frame.identity, seed['calibrant'], hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
               if isinstance(record, dict) else record, self._mask_state())
        if key not in self._ring_cache:
            from xrd_tools.integrate.calibration_seed import predict_frame_seed
            try:
                self._ring_cache[key] = predict_frame_seed(
                    record, frame, calibrant=seed['calibrant'], del_motor=seed['del_motor'],
                    nu_motor=seed['nu_motor'], mask=self._loaded_mask()).rings
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
                self._start('import', (frame, poni, picks[0], self._seed_options()))
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
        """Say which arm model the next fit takes, and why (decision 40: automatic)."""
        delta, nu = self._frame_motors if self.frames else self.geometry_editor.value().stitch_arm()
        done = tuple(frame for frame in self.frames if frame.complete)
        reason = automatic_arm_reason(done, del_motor=delta, nu_motor=nu, fit_scales=self.scales.isChecked())
        if not reason:
            text = ('Arm model (automatic): the detector mount and sample offset, which the frames on '
                    f'{len({round(float(frame.motors[nu]), 6) for frame in done})} nu positions determine; the '
                    'five-parameter arm if that fit is refused.')
        else:
            text = f'Arm model (automatic): the five-parameter arm; {reason}.'
        if self.scales.isChecked():
            text += (' With fitted motor scales the energy stays at the recorded one (it is refined only with '
                     'fixed scales) and Save writes no xrayutilities file.')
        self.arm_note.setText(text)

    def _fit(self):
        def go():
            if self.frames and self.geometry_editor.value().stitch_arm() != self._frame_motors:
                raise ValueError(self._staged_arm_message(self.geometry_editor.value().stitch_arm()))
            frames = tuple(frame for frame in self.frames if frame.complete)
            options = dict(del_motor=self._frame_motors[0], nu_motor=self._frame_motors[1],
                fit_scales=self.scales.isChecked(), instrument_geometry=self.geometry_editor.value().to_record(),
                arm_model='auto', **self._distance_options(), max_rms_q=float(self._text('rms')),
                distance_provenance=(self._distance_provenance if getattr(self, '_distance_provenance_key', None)
                                     is not None and self._distance_provenance_key == self._direct_beam_key()
                                     else None),  # a direct-beam record only while its inputs hold
                # the solver refines the energy only with fixed scales: fitted ones hold it (arm note)
                energy_bound_eV=None if self.scales.isChecked() else self._energy_bound(), bounds=self._bounds())
            self._invalidate_fit()
            self._start('fit', (frames, options))
        self._guard(go)

    def _energy_bound(self):
        """The ± eV the energy is refined within, or None when it is not refined."""
        if not self.refine_energy.isChecked():
            return None
        try:
            bound = float(self._text('energy_bound'))
        except ValueError:
            bound = math.nan
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError('type the energy bound as a positive number of eV')
        return bound

    def _refine(self):
        def go():
            geometry = self.geometry_editor.value()
            delta, nu = geometry.stitch_arm()
            spec = self._spec()
            if self.refine_seed.currentData() == 'fit':
                if self.fit is None:
                    raise ValueError('Fit the calibrated frames first, or start from a geometry file')
                seed = self.fit.asset['calibration']
            elif self.refine_seed.currentData() == 'rough':  # no frame calibrated (maintainer, 2026-09-29)
                seed = self._rough_arm()
                if seed is None:
                    raise ValueError('measure or type the distance and the beam pixel first (1 · Geometry)')
                if self._rough_wavelength_A() is None:
                    raise ValueError('type the wavelength in 1 · Geometry: the rings are predicted with it')
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
                           max_counts=float(self._text('saturation')) if self._text('saturation') else None,
                           mask=self._oriented_mask() if self._text('mask') else None,
                           max_rms_q=float(self._text('rms')), energy_bound_eV=self._energy_bound(),
                           bounds=self._bounds(), **self._distance_options())
            self._invalidate_fit()
            self._start('refine', (spec, seed, options))
        self._guard(go)

    def _save(self):
        def go():
            if self.fit is None:
                raise ValueError('fit the current controls before saving')
            selected = self._save_path()
            if selected is not None:
                self._saving = self.fit
                self._start('save', (self.fit, selected, pair_validation_angles(
                    self.fit, scan_angles=self._scan_angles(self.fit), domain=self._typed_domain())))
        self._guard(go)

    def _save_path(self):
        """The new calibration file the person chooses inside the project, or None."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save calibrated geometry',
            str(self.project_root / 'calibration' / f'multigeometry-{uuid.uuid4().hex[:8]}.json'), 'JSON (*.json)')
        if not path:
            return None
        selected = Path(path).resolve()
        if not selected.is_relative_to(self.project_root):
            raise ValueError('save the geometry inside the Stitch project')
        if selected.exists() or xu_partner_path(selected).exists():
            raise ValueError(f'{selected.name} or {xu_partner_path(selected).name} exists: choose a new name')
        return selected

    def _save_rough(self):
        """Save step 1's geometry as the calibration pair, without a ring fit."""
        def go():
            fit = self._rough_fit()
            selected = self._save_path()
            if selected is not None:
                angles = self._scan_angles(fit)
                if self.rough_mode.currentData() == 'scan' and self._distance_result is not None:
                    role = self._beam_role()
                    angles += [(angle, 0.) if role == 'del' else (0., angle)
                               for angle in self._distance_result.angles_deg.tolist()]
                self._saving = fit
                self._start('save', (fit, selected, pair_validation_angles(
                    fit, scan_angles=angles, domain=self._typed_domain())))
        self._guard(go)

    def _rough_fit(self):
        """Step 1's geometry as a calibration the save job takes; its provenance says it is a
        direct-beam calibration, not ring-fitted."""
        from xrd_tools.integrate.stitch_calibration import CalibrationFit
        record = self._rough_arm()
        if record is None:
            raise ValueError('measure or type the distance and the beam pixel first (1 · Geometry)')
        if self._rough_wavelength_A() is None:
            raise ValueError('type the wavelength to save this geometry as a calibration')
        delta, nu = self.geometry_editor.value().stitch_arm()  # as _rough_arm
        direct = (self._distance_record(self._distance_result)
                  if self.rough_mode.currentData() == 'scan' and self._distance_result is not None else None)
        provenance = {
            'frames': [], 'rms_q_A^-1': None, 'condition_number': None, 'n_points': 0,
            'distance_seed_m': None, 'distance_bounds_m': None, 'fit_motor_scales': False,
            'source_motors': {name: motor for name, motor in (('nu_angle', nu), ('del_angle', delta))
                              if motor is not None},
            'image_rotation': self.rotation.currentData(), 'direct_beam': direct,
            'calibration_kind': 'direct_beam_not_ring_fitted',
            'fixed_mount_deg': [float(v) for v in (self._text('rough_mount') or '0, 0').split(',')],
            'motor_zeros_deg': self.geometry_editor.value().arm_zeros(),
            'instrument_geometry': self.geometry_editor.value().to_record(),
        }
        return CalibrationFit('multigeometry', {'calibration': record, 'provenance': provenance}, 0.0, 0.0, 0,
                              'Direct beam: the distance and beam pixel, rotations held; not ring-fitted')

    def _scan_angles(self, fit):
        """(del, nu) of every calibrant-scan frame whose metadata was read, by the fit's motors."""
        motors = fit.asset['provenance']['source_motors']
        del_motor, nu_motor = motors.get('del_angle'), motors.get('nu_angle')
        return [(values.get(del_motor, 0.) if del_motor else 0., values.get(nu_motor, 0.) if nu_motor else 0.)
                for _index, values, _energy in self._choices
                if all(motor is None or motor in values for motor in (del_motor, nu_motor))]

    def _typed_domain(self):
        """The supported del/nu ranges typed under Advanced ({} when blank: the scan's span)."""
        domain = {}
        for circle in ('del', 'nu'):
            text = self._text(circle + '_domain')
            if text:
                low, high = (float(value) for value in text.split(','))
                if not low <= high:
                    raise ValueError(f'the supported {circle} range needs min ≤ max')
                domain[circle] = [low, high]
        return domain

    def _oriented_mask(self):
        """The Detector panel's mask, drawn on the stored image, turned as the images are."""
        from xrd_tools.core.geometry import ImageOrientation
        from xrd_tools.io.image import load_mask
        return ImageOrientation(rotation=int(self.rotation.currentData() or 0)).apply(
            load_mask(self._text('mask'))).astype(bool)

    def _mask_state(self):
        """The mask as used: its file's state and the rotation (None without a mask)."""
        if not self._text('mask'):
            return None
        try:
            state = Path(self._text('mask')).stat()
        except OSError:
            return (self._text('mask'), None)
        return (self._text('mask'), state.st_mtime_ns, state.st_size, self.rotation.currentData())

    def _loaded_mask(self):
        """The turned mask for the expected rings, loaded once per mask state (None when it
        cannot be read: the rings are then predicted without it)."""
        state = self._mask_state()
        if state is None:
            return None
        if getattr(self, '_mask_cache', (None,))[0] != state:
            try:
                self._mask_cache = (state, self._oriented_mask())
            except (OSError, ValueError):
                self._mask_cache = (state, None)
        return self._mask_cache[1]

    def _pyfai_mask(self):
        """The turned mask as a file pyFAI-calib2 reads (non-zero: masked), in the session folder."""
        mask = self._loaded_mask()
        if mask is None:
            if self._text('mask'):
                raise ValueError(f"the Detector panel's mask cannot be read: {self._text('mask')}")
            return None
        import hashlib
        import fabio
        digest = hashlib.sha256(repr(mask.shape).encode() + np.packbits(mask).tobytes()).hexdigest()[:16]
        path = self.session_dir / 'masks' / f'mask-{digest}.edf'
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            fabio.edfimage.EdfImage(data=mask.astype(np.int8)).write(str(path))
        return str(path)

    def _direct_beam_request(self, *, with_mask=True):
        """The direct-beam fit's (source, motor, options) from the window as it is now; without the
        mask's pixels its state stands in (for the result's identity)."""
        orientation = int(self._text('beam_orientation'))
        options = dict(detector=self._text('beam_detector'),
            detector_config={'orientation': orientation} if orientation != 3 else None,
            beam_pixel=(tuple(float(v) for v in self._text('beam_pixel').split(','))
                        if self._text('beam_pixel') else None),  # blank: found at the zero
            motion=self.motion.currentText(), strip_half_width=int(self._text('beam_strip')),
            model='tangent', angle_zero_deg=self._zeros().get(self._beam_role(), 0.),
            rotation=self.rotation.currentData())
        if self._text('beam_roi'):
            options['roi'] = tuple(int(v) for v in self._text('beam_roi').split(','))
        if self._text('mask'):
            options['mask'] = self._oriented_mask() if with_mask else self._mask_state()
        if self._text('saturation'):
            options['saturation'] = float(self._text('saturation'))
        motor = dict(zip(('del', 'nu'), self.geometry_editor.value().stitch_arm())).get(self._beam_role())
        return self._spec(beam=True), motor or self._text('beam_motor'), options

    def _direct_beam_key(self):
        """What a direct-beam result was fitted from (source, motor, zero, detector, extraction), or
        None when the window cannot say: a result is used only while this is unchanged (Codex review F5)."""
        try:
            spec, motor, options = self._direct_beam_request(with_mask=False)
        except (ValueError, TypeError, KeyError, OSError):
            return None
        return json.dumps([str(spec.uri), spec.kind.value, spec.entry, dict(spec.options), motor, options],
                          sort_keys=True, default=str)

    def _distance_fresh(self):
        return (self._distance_result is not None and self._distance_key is not None
                and self._direct_beam_key() == self._distance_key)

    def _distance(self):
        def go():
            request = self._direct_beam_request()
            self._distance_pending_key = self._direct_beam_key()
            self._start('distance', request)
        self._guard(go)

    def _show_direct_beam_fit(self):
        """The direct-beam fit in its own window, with the distance (opened after every fit)."""
        if self._distance_result is None:
            self.status.setText('Fit a direct-beam scan first.')
            return
        if self._direct_beam_fit_window is None:
            self._direct_beam_fit_window = DirectBeamFitWindow(self)
        self._direct_beam_fit_window.show_result(self._distance_result, self._distance_energy_eV)
        self._direct_beam_fit_window.show()
        self._direct_beam_fit_window.raise_()
        self.show_fit_button.setEnabled(True)

    def hideEvent(self, event):
        """The fit window goes with this window (closed, hidden, or shut down by the host)."""
        if self._direct_beam_fit_window is not None:
            self._direct_beam_fit_window.close()
        super().hideEvent(event)

    def keyPressEvent(self, event):
        """Return or Enter commits the box being typed in and presses no button: with many boxes and
        buttons, the dialog's default button (Fit direct-beam distance) would start a job."""
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def _apply_distance(self):
        if self._distance_result is None:
            self.status.setText('Fit a direct-beam scan first.')
            return
        if not self._distance_fresh():
            self.status.setText(self._STALE_DISTANCE)
            return
        result = self._distance_result
        self._fields['distance_seed'].setText(f'{result.distance_mm:.12g}')
        self._distance_provenance = self._distance_record(result)
        self._distance_provenance_key = self._distance_key
        self._show_tab('calibrate')

    @staticmethod
    def _distance_record(result):
        """A direct-beam fit's provenance record."""
        return dict(distance_m=result.distance_m,
            regression_std_m=result.distance_std_m, angle_zero_deg=result.angle_zero_deg,
            pixel_pitch_m=result.pixel_pitch_m, image_shape=list(result.image_shape),
            beam_pixel=list(result.beam_pixel), zero_beam_pixel=list(result.zero_beam_pixel),
            model=result.model, frame_ids=[str(v) for v in result.frame_ids],
            point_ids=[tuple(str(value) if isinstance(value, Path) else value for value in point)
                       if isinstance(point, tuple) else point for point in result.point_ids],
            angles_deg=result.angles_deg.tolist(),
            used=result.used.tolist(), rejection_reasons=list(result.rejection_reasons),
            centres_px=[float(v) if math.isfinite(v) else None for v in result.centres_px],
            residuals_px=[float(v) if math.isfinite(v) else None for v in result.residuals_px],
            uncertainty_policy='conditional_regression_standard_error_not_total_accuracy')

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
            ring_tip = ('No calibration to predict from yet: enter a rough geometry (1 · Geometry) or a '
                        'geometry file, or calibrate one frame: the others are then predicted from it.'
                        if rings is None
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
        self._prediction = None
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
                self.frames, arm, policy = load_calibration_session(path, with_arm=True, with_policy=True)
                # an older session names no arm: del and nu in order, or a vertical 2θ arm
                self._frame_motors = arm or (tuple(self.frames[0].motors) + (None,))[:2]
                self.session_dir = Path(path).parent
                if policy is not None:  # its own zeros and bounds, before anything is predicted (Codex F6)
                    self._apply_fit_policy(policy)
                self._invalidate_fit()
                self._prediction = None
                self._render_frames()
                self._show_tab('calibrate')
                if any(frame.complete for frame in self.frames):  # its calibrated frames predict the rest
                    self._start('predict', (self._calibrated_key(), self._seed_options()))
                if policy is None:
                    self.status.setText('This session records no fit bounds: the window\'s zeros and bounds are used.')
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
                    result, prediction = result
                    self.frames = tuple(result if frame.identity == self._row_identity else frame for frame in self.frames)
                    self._prediction = (self._calibrated_key(), prediction)
                    save_calibration_session(self.session_dir, self.frames, arm=self._frame_motors,
                                             fit_policy=self._fit_policy())
                    self._invalidate_fit()
                    self._render_frames()
                    self._select_next_frame(result)
                elif kind == 'predict':
                    self._prediction = result
                    self._render_frames()
                elif kind in ('fit', 'refine'):
                    summary = fit_summary_text(result)  # before adopting: a failure leaves no fit
                    self.fit = result
                    self.refine_seed.setCurrentIndex(self.refine_seed.findData('fit'))  # it starts from this
                    self._fit_instrument = result.asset['provenance'].get('instrument_geometry')
                    self.fit_summary.setPlainText(summary)
                    self.save_button.setEnabled(True)
                elif kind == 'save':  # the ring fit, or the direct-beam geometry (_save_rough)
                    provenance = self._saving.asset['provenance']
                    self.saved_source_motors = provenance['source_motors']
                    self.saved_instrument_geometry = provenance.get('instrument_geometry')
                    self.saved_image_rotation = provenance['image_rotation']
                    self.saved_mask = self._text('mask')  # becomes Stitch's Mask File when set
                    from xdart.utils.browse import remember_browse_path
                    remember_browse_path(Path(result.pyfai_path))
                    self.fit_summary.appendPlainText('\n' + saved_pair_text(result))
                elif kind == 'distance':
                    result, self._distance_energy_eV = result
                    self._distance_result, self._distance_key = result, self._distance_pending_key
                    rows = [f'Distance: {result.distance_mm:.6g} mm; regression SE: {result.distance_std_m * 1000:.3g} mm',
                            f'Beam pixel at the zero: {result.zero_beam_pixel[0]:.2f}, {result.zero_beam_pixel[1]:.2f} '
                            f'(strips placed at {result.beam_pixel[0]:.1f}, {result.beam_pixel[1]:.1f}'
                            + (')' if self._text('beam_pixel') else ', found on the frame nearest the zero)'),
                            'Frame · angle (°) · centre (px) · residual (px) · status']
                    rows += [f'{result.point_ids[i]} · {result.angles_deg[i]:.6g} · {result.centres_px[i]:.5g} · {result.residuals_px[i]:.4g} · {"used" if result.used[i] else result.rejection_reasons[i]}' for i in range(len(result.angles_deg))]
                    self.beam_summary.setPlainText('\n'.join(rows))
                    self.beam_summary.verticalScrollBar().setValue(0)  # the distance, first
                    self._beam_pixel_hint()
                    self._render_frames()  # the measured distance now predicts the rings
                    self._show_direct_beam_fit()
                if kind == 'calibrate' and result is not None:
                    pending = [frame for frame in self.frames if not frame.complete]
                    self.status.setText(f'Frame {result.index} saved. ' + (
                        'Next: Calibrate selected frame…' if pending else 'All frames calibrated: fit them below, then save.'))
                elif kind == 'save':
                    self.status.setText(saved_pair_text(result))
                    self.calibrationSaved.emit(result.pyfai_path, 'multigeometry')  # once, with both files written
                elif kind == 'distance':
                    self.status.setText(direct_beam_fit_text(self._distance_result, self._distance_energy_eV))
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
