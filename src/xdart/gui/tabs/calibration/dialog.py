"""The modeless guided calibration window every page opens with its own context (Stitch, RSM), using the
shared child-process owner."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from contextlib import ExitStack
import json
import logging
import math
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING
import uuid

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.pages.operation_owner import SingleWorkerOwner, OperationCancelled, OperationTerminalStatus
from xdart.gui.widgets.controls_panel import FormRow, SectionCard
from xdart.gui.widgets.file_dialogs import NEXUS_FILTER
from xdart.gui.widgets.read_only_text import ReadOnlyTextPane
from xdart.gui.tabs.scattering.results_table import ResultsTable
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.sources.point_images import PointImagesRequired, PointImageDetection, inspect_point_images
from xdart.gui.tabs.scattering.point_images_editor import PointImagesSettings
from xrd_tools.integrate.calibration_seed import (
    DETECTOR_ORIENTATIONS, detector_choices, detector_shape, direct_beam_motion_mismatch, ring_summary,
    rough_arm_record,
)
from xrd_tools.session.instrument import InstrumentGeometry
from xrd_tools.session.threshold_editing import (
    ThresholdEditRefusal, ThresholdState, auto_ceiling_hint, threshold_transition,
)
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from xrd_tools.integrate.stitch_calibration import (
    adopt_calibration_pair, automatic_arm_reason, capture_calibration_frames, direct_beam_pose, fit_calibration,
    load_calibration_session, pair_validation_angles, save_calibration_pair, save_calibration_session,
    scan_motor_names, source_frame_choices, suggested_frame_indices, validate_frame_files, xu_partner_path,
    direct_beam_scan_motor, open_calibration_beam_scan,
)
from .fit_windows import (
    PARAMETER_LABELS, RECORD_LABELS, CalibrationFitWindow, DirectBeamFitWindow, calibration_fit_text, direct_beam_fit_text,
)

if TYPE_CHECKING:
    from xrd_tools.core.geometry.camera import DetectorCamera

_LOG = logging.getLogger(__name__)


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
        kind, arguments = request
        for spec in arguments:
            if type(spec) is SourceSpec and kind != 'fit':
                detection = inspect_point_images(spec)
                policy = spec.options.get('point_images')
                if detection is not None and (policy is None or
                        policy['images_per_point'] != detection.images_per_point):
                    return detection
        return _dispatch_job(request, cancelled, publish)
    except PointImagesRequired as error:
        return error.detection
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
                                    cancelled=cancelled, bounds=seed.get('bounds'),
                                    instrument_geometry=seed.get('instrument'), turned_from=seed.get('turned'),
                                    **seed.get('distance', {}))
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
                                  cancelled=cancelled, bounds=seed.get('bounds'),
                                  instrument_geometry=seed.get('instrument'), turned_from=seed.get('turned'),
                                  **seed.get('distance', {}))
    except InterruptedError:
        return None


def _pyfai_arguments(frame, seed, cancelled=None):
    """pyFAI-calib2 options for one guided frame: its calibrant and energy, and the frame's
    predicted geometry when a calibration is at hand, so "extract rings" numbers the rings.

    A frame whose scan records no energy takes the window's wavelength (typed, else the calibrant
    scan's, else the calibrated frames' PONI files', else the direct-beam scan's: ``seed['wavelength_A']``),
    else a seed calibration's; never pyFAI's default (the window refuses first: ``_frame_energy_known``)."""
    if not frame.energy_eV and seed.get('wavelength_A'):
        from xrd_tools.core.energy import wavelength_m_to_energy_eV
        frame = replace(frame, energy_eV=float(wavelength_m_to_energy_eV(seed['wavelength_A'] * 1e-10)))
    from xrd_tools.integrate.calibration import pyfai_calibrant_name
    arguments = ['-c', pyfai_calibrant_name(seed['calibrant'])]
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
    if seed.get('detector') and seed['detector'] != 'Detector':  # a generic one needs its pitches (item 8)
        arguments += ['-D', seed['detector']]
    return tuple(arguments)


def _frame_energy_known(frame, seed):
    """Whether pyFAI can calibrate the frame at a known wavelength: its scan's energy, the window's
    wavelength, or a real calibration's (the rough geometry's is a placeholder without one)."""
    return bool(frame.energy_eV or seed.get('wavelength_A') or any(
        seed.get(key) is not None for key in ('record', 'prediction', 'file')) or seed.get('frames'))


def _dispatch_job(request, cancelled, publish):
    kind, arguments = request
    if kind == 'orientation':
        from xrd_tools.geometry_figures import consistent_motion_candidates, diagnose_calibration_frames
        key, mode, options = arguments
        options = dict(options)
        if cancelled.is_set():
            raise OperationCancelled('orientation check cancelled')
        if mode == 'motion':
            finding = consistent_motion_candidates(**options)
            result = finding, None, None
        else:
            seed = options.pop('seed')
            # Use only an existing fixed arm. Checking never invokes session_prediction or a fit.
            record = seed.get('record') or seed.get('prediction') or seed.get('rough') or seed.get('file')
            if record is None:
                raise ValueError('Enter or load a fixed rough geometry before checking these frames.')
            result = diagnose_calibration_frames(record=record, cancelled=cancelled.is_set, **options)
        if cancelled.is_set():
            raise OperationCancelled('orientation check cancelled')
        return key, result
    if kind == 'source_details':
        from xrd_tools.integrate.stitch_calibration import calibration_source_energy_eV
        from xrd_tools.sources.spec import find_spec_image_files, spec_image_stem
        prefix, spec, folder, stem = arguments
        energy = calibration_source_energy_eV(spec) if not prefix else None
        frame_count, source_problem = None, ''
        if not prefix:
            try:
                frame_count = len(source_frame_choices(spec, ()))
            except (ValueError, KeyError, OSError) as error:
                frame_count, source_problem = 0, str(error)
                # Energy discovery is independent of missing or incomplete images.
        automatic = None
        if spec.kind is SourceKind.SPEC and not stem:
            files = find_spec_image_files(folder, Path(spec.uri).stem, spec.options['scan'])
            if files:
                automatic = spec_image_stem(files[0], Path(spec.uri).stem, spec.options['scan'])
        if cancelled.is_set():
            raise OperationCancelled('source metadata cancelled')
        return prefix, energy, automatic, frame_count, source_problem
    if kind == 'refine':
        from xrd_tools.integrate.stitch_autocalibration import refine_arm_from_scan
        spec, seed, options = arguments
        return refine_arm_from_scan(spec, seed, cancelled=cancelled,
                                    progress=_staged_progress(publish, _REFINE_STAGES), **options)
    if kind == 'stitch_preview':
        from xrd_tools.integrate.stitch_calibration import stitch_calibrant_preview
        from xrd_tools.analysis.plans import StitchCancelled
        fit, spec, options = arguments
        try:
            return stitch_calibrant_preview(fit, spec, cancelled=cancelled,
                progress=lambda done, total: publish(f'Stitching calibrant {done}/{total}', done, total), **options)
        except StitchCancelled as error:
            raise OperationCancelled(str(error)) from error
    if kind == 'enumerate':
        spec, record = arguments
        names = scan_motor_names(spec)
        geometry = InstrumentGeometry.from_record(record)
        if names:  # the scan's own motor for each circle (a choice the scan has stays)
            geometry = InstrumentGeometry.pick(geometry.geometry, names, keep=geometry)
        motors = tuple(motor for motor in geometry.stitch_arm() if motor)  # a 2θ arm reads one
        return source_frame_choices(spec, motors), names, geometry.to_record()
    if kind == 'prepare':
        spec, indices, directory, orientation, arm, conditioning, saturation = arguments
        return capture_calibration_frames(spec, indices, directory, rotation=orientation.rotation,
                                           flip=orientation.flip_horizontal, arm=arm, cancelled=cancelled,
                                           conditioning=conditioning, saturation=saturation)
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
                                       instrument_geometry=seed.get('instrument'), turned_from=seed.get('turned'),
                                       **seed.get('distance', {}))
    if kind == 'fit':
        frames, options = arguments
        return fit_calibration(frames, cancelled=cancelled,
                               progress=_staged_progress(publish, _TRANSFER_STAGES), **options)
    if kind == 'save':  # the pyFAI file, then its xrayutilities partner beside it
        fit, path, angles, *overwrite = arguments
        return save_calibration_pair(fit, path, validation_angles=angles, cancelled=cancelled,
                                     progress=_staged_progress(publish, _TRANSFER_STAGES),
                                     overwrite=overwrite[0] if overwrite else True)
    if kind in ('beam_detect', 'distance'):
        from xrd_tools.integrate.direct_beam import calibrate_direct_beam, infer_direct_beam_motion
        from xrd_tools.core.geometry import ImageOrientation
        if kind == 'beam_detect':
            spec, detector_motors, options = arguments
            motor = direct_beam_scan_motor(spec, detector_motors)
        else:
            spec, motor, options, held = arguments
        options = dict(options)
        orientation = ImageOrientation(rotation=options.pop('rotation'),
                                       flip_horizontal=options.pop('flip', False))
        options['reader_options'] = dict(spec.options.get('read_image_kwargs', {}))
        if cancelled.is_set():
            raise OperationCancelled('direct-beam operation cancelled')
        with open_calibration_beam_scan(spec, motor) as inputs:
            if kind == 'beam_detect':
                motion, delta, endpoints = infer_direct_beam_motion(list(inputs.images), orientation=orientation,
                    cancelled=cancelled, return_details=True, **options)
                return (motor, motion, delta, tuple(float(inputs.angles_deg[index]) for index in endpoints),
                        len(inputs.angles_deg))
            energy, held_readings = direct_beam_pose(spec, motor, held)
            def images():
                for frame in inputs.images:
                    if cancelled.is_set():
                        raise OperationCancelled('direct-beam calibration cancelled')
                    yield frame
            result = calibrate_direct_beam(inputs.angles_deg, images(), orientation=orientation,
                frame_ids=inputs.frame_ids, point_ids=inputs.point_ids, **options)
            if cancelled.is_set():
                raise OperationCancelled('direct-beam calibration cancelled')
            return result, energy, held_readings
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
        lines.append('Fitted arm parameters (± 1σ):')
        for name, value in zip(record['param_names'], record['param']):
            if name not in errors:
                continue
            scale, unit = ((1e3, 'mm') if name.startswith(('dist', 'poni', 'sample_offset'))
                           else (180 / math.pi, '°'))
            lines.append(f'  {RECORD_LABELS.get(name, name)}: {value * scale:.6g} ± {errors[name] * scale:.2g} {unit}')
        correlations = provenance.get('parameter_correlations') or {}
        names = [RECORD_LABELS.get(name, name) for name in correlations.get('names', [])]
        matrix = correlations.get('matrix', [])
        strong = [f'{names[i]} and {names[j]} {matrix[i][j]:+.2f}' for i in range(len(names))
                  for j in range(i + 1, len(names)) if abs(matrix[i][j]) > .9]
        if strong:
            lines.append('Strongly correlated, so each is less certain than its σ says (|r| > 0.9): '
                         + ', '.join(strong))
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


#: Where a fit's starting zero came from (ArmFitBounds.zero_sources), in the window's words.
_ZERO_SOURCES = {'typed': 'as typed under 1 · Instrument', 'seed': "the starting geometry's",
                 'default': '0, none typed'}


def fit_bounds_lines(record):
    """The fit's parameters as it reports them: where each started → where it ended (the change), the
    range it could move in, and those that stopped at a limit, which is never widened
    (ARM-ZERO-BOUNDS-20260929)."""
    if not record or not record.get('parameters'):
        return []
    estimates = record.get('estimates') or {}
    sources = '; '.join(f'{circle} zero {_ZERO_SOURCES.get(source, source)}'
                        for circle, source in (estimates.get('zero_sources') or {}).items())
    lines = ['Fitted values (start → fitted, change; allowed range)' + (f' — {sources}' if sources else '') + ':']
    for name, row in record['parameters'].items():
        label, unit = PARAMETER_LABELS.get(name, (name, ''))
        bounds = ('any' if row['lower'] is None else f"{row['lower']:.6g} to {row['upper']:.6g}")
        lines.append(f"  {label}: {row['initial']:.6g} → {row['fitted']:.6g} {unit} "
                     f"({row['fitted'] - row['initial']:+.3g}; {bounds})"
                     + (' — stopped at its limit' if row['at_bound'] else ''))
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


#: The two scans' source kinds are the page's one Data format's (LT-SMALL-FIXES-20261014, LT-004/LT-005).
_FROM_DATA_FORMAT = ("Set by the page's Data format (its Project card); the calibrant and direct-beam scans both "
                     "follow it.")
#: What the page a window serves does with what it saves, in the window's tips (G6 choice 4): the window is
#: shared, so each tip names its own page's use, and claims none for a page it does not know.
_PAGE_USE = {
    "Stitch": {
        "energy": " The saved calibration keeps the recorded energy, which Stitch checks the scans' energies against.",
        "energy_field": " (Stitch checks each scan's energy against it)",
        "backend": ": the Backend under Processing chooses which of the two Stitch uses",
    },
    "RSM": {
        "energy": (" The saved calibration keeps the recorded energy and the refined one: RSM adds their difference "
                   "to each scan's own energy when it applies the refinement, and warns of a scan recorded more "
                   "than 2 % from it."),
        "energy_field": " (RSM warns of a scan recorded more than 2 % from it)",
        "backend": ": RSM grids with the xrayutilities one",
    },
}


@dataclass(frozen=True)
class CalibrationContext:
    """The page a calibration window serves: its name, in the window's title and buttons ("Save geometry and
    use in RSM"), and its Project, where the window keeps its sessions and saves. Each page opens its own
    window with its own context, from its own Project and source (``set_source_context``), and adopts only
    its own window's saves. The page, not the window, keeps the window's settings.

    ``detector`` (a pyFAI detector name) and ``camera`` (a ``DetectorCamera``) are the page's own detector the
    window starts on, before any saved settings are applied (they win); None: the window's defaults, the
    Pilatus 300K-W in Stitch's frame (AUD-9 stage 2b item 10: RSM starts on its Pilatus 100K, camera z−, x−)."""

    page: str
    project_root: str | Path
    detector: str | None = None
    camera: DetectorCamera | None = None

    def __post_init__(self):
        if self.detector is not None and (type(self.detector) is not str or not self.detector.strip()):
            raise TypeError("a calibration window's starting detector is a pyFAI detector name")
        if self.camera is not None:
            from xrd_tools.core.geometry.camera import DetectorCamera
            if type(self.camera) is not DetectorCamera:
                raise TypeError("a calibration window's starting camera is a DetectorCamera")


class CalibrationDialog(QtWidgets.QDialog):
    """Retained window. ``calibrationSaved(path, backend)`` adopts a saved pyFAI calibration
    (``backend`` 'multigeometry'); its xrayutilities partner, when one was made, is beside it
    (``xu_partner_path``).

    ``set_source_context(SourceSpec, image_rotation=0, del_motor='del',
    nu_motor='nu', geometry_path=None)`` seeds the source controls (and the
    automatic refinement's seed geometry) without selecting frames or changing
    previously completed rows; ``image_rotation=None`` keeps the window's camera
    (a page with no turn of its own). Call ``show/raise_/activateWindow``.
    """
    calibrationSaved = QtCore.Signal(str, str)
    #: The diffractometer or a circle's motor changed (the page keeps the same geometry).
    geometryChanged = QtCore.Signal()
    #: A job started (True) or ended (False): the page locks its inputs, its Data format among them, meanwhile.
    busyChanged = QtCore.Signal(bool)

    def __init__(self, context: CalibrationContext, *, parent=None):
        super().__init__(parent)
        if type(context) is not CalibrationContext:
            raise TypeError('a calibration window needs its page\'s CalibrationContext')
        self.context = context
        self.setWindowTitle(f'{context.page} calibration')
        self.setObjectName('calibrationDialog')
        self.setModal(False)
        self.resize(940, 876)
        self.project_root = Path(context.project_root).resolve()
        self.session_dir = self.project_root / 'calibration' / f'session-{uuid.uuid4().hex[:12]}'
        self.frames = ()
        self.fit = None
        self._saved_ring_fit = None
        self._checklist_fit_outdated = False
        #: The arm fitted to the calibrated frames: (their identities, record); see _seed_options.
        self._prediction = None
        #: Predicted rings per (frame, calibrant, calibration): see _expected_rings.
        self._ring_cache = {}
        #: Calibration's threshold, Int's rule and Int's default: on, Auto leaving saturated pixels out (maintainer,
        #: 2026-10-08); shown by every row in _threshold_rows (Advanced's, the Quick card's).
        self._threshold_state = ThresholdState()
        self._threshold_rows = []
        self._choices = []
        self._identity = None
        self._job = None
        self._row_identity = None
        self._close_pending = False
        self.orientation_panel = None
        self.instrument_schematic = None
        self._orientation_generation = 0
        self._orientation_pending_key = None
        self._orientation_geometry = None
        self._orientation_figure = None
        self._schematic_readings = None  # None is an explicitly labelled zero-pose example, not metadata.
        self._schematic_pose = 'zero-readback example'
        #: the fields the checklist's Instrument and Orientation check rows show when not ready (LT-008, LT-009)
        self._instrument_field = self._orientation_field = None
        self._beam_detection = None
        self._beam_generation = 0
        self._beam_problem = 'Select a direct-beam source and scan'
        self._distance_result = None
        self._distance_energy_eV = None  # the fitted direct-beam scan's recorded energy
        self._distance_held = {}  # its other moving circle's recorded reading
        self._distance_warning = None  # its geometry moves the beam another way than it did (or None)
        self._job_policy = None  # the fit policy the running Fit or Refine took
        self._direct_beam_fit_window = None
        self._calibration_fit_window = None
        self._fit_kind = 'fit'  # the job the current fit came from: 'fit' or 'refine'
        self._fit_source = self._pending_fit_source = None
        self._preview_key = None
        self._distance_key = self._distance_pending_key = None  # what the result was fitted from
        self._source_energy_eV = None
        self._poni_wavelength = (None, None)  # (the calibrated frames' PONI files, the wavelength they agree on)
        self._pending_source_details = None
        self._calibrant_ready_key = None
        self._calibrant_frame_count = None
        self._calibrant_source_problem = ''
        self._restoring_fields = False
        self._loaded_calibration_identity = None
        self._saved_setup = None
        self._loaded_calibration_kind = None
        self._fit_setup = None
        self._close_after_save = False
        #: The arm the staged frames read: (del, nu), or (2θ, None) for fourc/twoc.
        self._frame_motors = ('del', 'nu')
        #: The instrument the current fit recorded; any other value leaves it unsaveable.
        self._fit_instrument = None
        self.saved_source_motors = {}
        self.saved_instrument_geometry = None
        self.saved_image_rotation = 0
        self.saved_image_flip = False  # with the rotation: the camera's transform into Stitch's frame
        self.saved_mask = ''
        #: The page's Data format once it has set this window's sources (None: a window no page has set)
        self._data_format = None
        self._shared_nexus_entry = None  # None: standalone editor; a page owns its Entry option.
        self._owner = SingleWorkerOwner(_run_job)
        #: Quick calibration: the step it is on (1-3; None when idle), the job it waits for, whether Cancel was
        #: pressed, and the inputs its direct-beam check last ran on (_quick_advance)
        self._quick_stage = self._quick_kind = None
        self._quick_cancelled = False
        self._quick_detected_generation = None
        #: canonical key -> the card's row showing it (beam_source, beam_scan, source, scan)
        self._quick_rows = {}
        self._quick_folder_buttons = {}
        #: key -> its file row's chooser, so another row can browse as it does (the Quick card's)
        self._choosers = {}
        self._fields = {}
        #: key -> its Int ``FormRow``: the field is the row's editor, its value read through the row
        self._formrows = {}
        #: a combo editor -> its FormRow (the form's row)
        self._row_of = {}
        #: key -> (form, row widget); a field sharing another's row is in _parts instead
        self._rows = {}
        self._parts = {}
        outer = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        from xdart.gui.widgets.controls_panel import ActionButton
        from xdart.gui.widgets.context_help import bind_topic, owner_for
        help_button = ActionButton.for_help(self, 'calibration.multi-step')
        self.tabs.setCornerWidget(help_button, QtCore.Qt.TopRightCorner)
        self._tab_pages = {}
        outer.addWidget(self.tabs, 1)
        # Four tabs (maintainer, 2026-09-30): the instrument, a rough geometry (it predicts every
        # frame's rings), the calibrant scan's frames, then calibrating and fitting them.
        self._build_instrument()
        self._build_distance()
        self._build_source()
        self._build_frames()
        self._build_fit()
        self._sync_rough_rows()
        self.status = QtWidgets.QLabel('Ready. Three to five frames are suggested; angle coverage and fit diagnostics determine sufficiency.')
        self.status.setWordWrap(True)
        bind_topic(self, 'calibration.multi-step', details=lambda:
                   self.status.text() + '\n' + self.fit_summary.toPlainText())
        self._help_owner = owner_for(self)
        outer.addWidget(self.status)
        footer = QtWidgets.QHBoxLayout()
        from .orientation_views import action
        self.orientation_button = action('Orientation check', 'Open the read-only staged-frame or direct-beam orientation check; no values are applied.', self)
        self.schematic_button = action('Instrument schematic', 'Show configured circle axes in three views; select a frame for its recorded motor readings.', self)
        self.orientation_button.clicked.connect(self.open_orientation_check)
        self.schematic_button.clicked.connect(self.show_instrument_schematic)
        footer.addWidget(self.orientation_button)
        footer.addWidget(self.schematic_button)
        self.cancel_button = QtWidgets.QPushButton('Cancel operation')
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setEnabled(False)
        footer.addWidget(self.cancel_button)
        footer.addStretch()
        self.new_session_button = QtWidgets.QPushButton('New session')
        self.new_session_button.setToolTip('Starts a new session: the staged frames and the fit are cleared; saved '
                                           'session files remain. Waits while an operation runs.')
        self.new_session_button.clicked.connect(self._new_session)
        footer.addWidget(self.new_session_button)
        self.open_session_button = QtWidgets.QPushButton('Open saved session…')
        self.open_session_button.setToolTip("Opens a saved session's frames and fit choices. Waits while an "
                                            'operation runs.')
        self.open_session_button.clicked.connect(self._resume)
        footer.addWidget(self.open_session_button)
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(self.close)
        footer.addWidget(close)
        outer.addLayout(footer)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)
        # the page's own detector and camera, when it names them (item 10); saved settings applied after win
        if context.detector is not None:
            self._set_field_text('beam_detector', context.detector)
        if context.camera is not None:
            self.camera.set_value(context.camera)
        self._pixel_detector = self._text('beam_detector')
        self._pixel_orientation = self._orientation()
        self._beam_probe_timer = QtCore.QTimer(self)
        self._beam_probe_timer.setSingleShot(True)
        self._beam_probe_timer.setInterval(150)
        self._beam_probe_timer.timeout.connect(self._detect_beam)
        for key in ('beam_source', 'beam_scan', 'beam_entry', 'beam_images', 'beam_stem',
                    'beam_image_file', 'beam_roi', 'mask', 'saturation', 'shape', 'dtype', 'header'):
            self._fields[key].textChanged.connect(self._invalidate_beam_detection)
        for key in ('beam_detector', 'beam_orientation'):
            self._fields[key].currentIndexChanged.connect(self._invalidate_beam_detection)
        self.camera.changed.connect(self._invalidate_beam_detection)
        self._beam_arm = self.geometry_editor.value().stitch_arm()
        self.geometry_editor.changed.connect(self._beam_geometry_changed)
        self.beam_source_kind.currentIndexChanged.connect(self._invalidate_beam_detection)
        self.rough_mode.currentIndexChanged.connect(self._invalidate_beam_detection)
        for combo in (self.source_kind, self.beam_source_kind):  # a RAW type is known only for an image series
            combo.currentIndexChanged.connect(self._caption_model)
        self._caption_model()
        # the motion advice follows every input of the fit's identity and of step 1's arm, after the handlers above
        for key in self._DIRECT_BEAM_ADVICE_FIELDS:
            field = self._fields[key]
            (field.currentIndexChanged if isinstance(field, QtWidgets.QComboBox)
             else field.textChanged).connect(self._render_direct_beam_advice)
        for signal in (self.geometry_editor.changed, self.camera.changed, self.point_images.changed,
                       self.beam_source_kind.currentIndexChanged, self.rough_mode.currentIndexChanged):
            signal.connect(self._render_direct_beam_advice)
        self.tabs.currentChanged.connect(self._tab_readiness)
        for field in dict.fromkeys(self._fields.values()):
            signal = (getattr(field, 'currentIndexChanged', None) if isinstance(field, QtWidgets.QComboBox)
                      else getattr(field, 'textChanged', None))
            if signal is not None:
                signal.connect(self._invalidate_orientation)
        for signal in (self.geometry_editor.changed, self.camera.changed, self.point_images.changed,
                       self.beam_source_kind.currentIndexChanged, self.rough_mode.currentIndexChanged,
                       self.refine_calibrant.currentIndexChanged):
            signal.connect(self._invalidate_orientation)
        self.geometry_editor.changed.connect(self._redraw_schematic)
        self.frame_table.itemSelectionChanged.connect(self._schematic_frame_selected)
        self._build_checklist()
        outer.insertWidget(0, self.checklist_card)
        self._build_quick()
        self._update_beam_readiness()

    def _build_checklist(self):
        self.checklist_card = SectionCard('Calibration checklist')
        tip = 'Read-only progress from the current calibration owners. Activate a row to visit its controls; no science or save is started.'
        self.checklist_card.setToolTip(tip)
        self.checklist_card.title.setToolTip(tip)
        self.checklist_card.status.setToolTip('Counts completed workflow steps; does not certify fit quality or physical orientation.')
        self.checklist_card.set_help_topic('calibration.multi-step')
        self.checklist_table = ResultsTable(('Step', 'State', 'What to do'), row_selection=True,
            minimum_rows=8, max_height=270, header_tooltips={
                0: 'Opening phrase of the teaching step; its tooltip carries the full step. Activate to navigate.',
                1: 'Readiness or provenance reported by the current owners; Done is not a fit-quality verdict.',
                2: 'Current owner diagnostic or the next workflow action.'})
        self.checklist_table.setToolTip(tip)
        self.checklist_table.setColumnWidth(0, 250)
        self.checklist_table.setColumnWidth(1, 100)
        self.checklist_table.cellActivated.connect(self._checklist_navigate)
        self.checklist_card.set_embedded_widget(self.checklist_table)
        self.checklist_card.set_collapsed(True)
        self._refresh_checklist()

    def _refresh_checklist(self):
        if not hasattr(self, 'checklist_table'):
            return
        from xrd_tools.integrate.calibration_checklist import ChecklistFacts, calibration_checklist
        instrument_ready, instrument_message = False, 'Describe the instrument, detector and camera.'
        # the field each check reads: the row's navigation shows it, opening a hidden Config (LT-009)
        field = self.geometry_editor.geometry
        try:
            geometry = self.geometry_editor.value()
            geometry.stitch_arm()
            field = self.geometry_editor.problem_field() or field
            self._zeros()
            field = self._fields['beam_detector']
            if detector_shape(self._text('beam_detector')) is None:
                raise ValueError('Select a known detector.')
            field = self._fields['beam_orientation']
            if int(self._text('beam_orientation')) not in dict(DETECTOR_ORIENTATIONS):
                raise ValueError('Select a supported first-pixel orientation.')
            field = self.camera.rotation
            self.camera.value()
            problem = self._detector_config_problem()
            if problem is not None:
                message, field = problem
                raise ValueError(message)
            instrument_ready = True
            instrument_message = f'{geometry.geometry} · {self._text("beam_detector")} · configured camera; orientation check is optional.'
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            instrument_message = str(error)
        self._instrument_field = None if instrument_ready else field
        ready, reason, target, self._orientation_field = self._orientation_readiness()
        waiting = self._owner.busy
        if waiting:
            ready, reason, target = False, ('Checking orientation…' if self._job == 'orientation' else
                                            'Wait for the running operation to finish, or cancel it.'), 'orientation'
        self._project_orientation(ready, reason)
        mode = self.rough_mode.currentData()
        stale = mode == 'scan' and self._distance_result is not None and not self._distance_fresh()
        rough_ready = False
        rough_message = self._ROUGH_NOTES[mode]
        try:
            rough_ready = self._rough_arm() is not None
            if stale:
                rough_message = self._STALE_DISTANCE
            elif mode == 'scan':
                rough_message = self.beam_summary.toPlainText() if rough_ready else self._beam_problem
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            rough_message = str(error)
        complete = sum(frame.complete for frame in self.frames)
        summary = self.fit_summary.toPlainText()
        facts = ChecklistFacts(instrument_ready=instrument_ready, instrument_message=instrument_message,
            rough_mode=mode, rough_ready=rough_ready, rough_stale=stale, rough_message=rough_message,
            source_ready=self._calibrant_source_ready(), source_message=self.active_source_row.editor.text(),
            frame_count=len(self.frames), complete_count=complete,
            frame_message=f'{complete} of {len(self.frames)} Complete' if self.frames else 'Select frames and calibrate their rings.',
            fit_present=self.fit is not None, fit_outdated=self._checklist_fit_outdated,
            fit_message=(summary.splitlines() or [''])[0] if self.fit is not None
            else 'Fit the calibrated frames or refine the scan.',
            fit_detail=summary if self.fit is not None else '',
            saved_current_fit=self.fit is not None and self._saved_ring_fit is self.fit,
            orientation_ready=ready or waiting, orientation_waiting=waiting, orientation_target=target,
            **({} if ready else {'orientation_message': reason}))
        self._checklist_rows = calibration_checklist(facts)
        tips = {(i, c): row.text if c == 0 else
                f'{row.state}: current owner readiness or provenance; no scientific quality verdict.' if c == 1
                else row.detail or row.message or 'Activate this row to visit the existing controls.'
                for i, row in enumerate(self._checklist_rows) for c in range(3)}
        self.checklist_table.show_rows(('Step', 'State', 'What to do'),
            tuple((row.step, row.state, row.message) for row in self._checklist_rows), cell_tooltips=tips)
        current = next((i for i, row in enumerate(self._checklist_rows[:7])
                        if row.state in ('Needs input', 'Out of date')), 6)
        step = self._checklist_rows[current]
        self.checklist_card.set_readiness(step.state not in ('Needs input', 'Out of date'),
            f'Step {current + 1} of 7: {step.message}')

    def _checklist_navigate(self, row, _column):
        if self._owner.busy or not 0 <= row < len(self._checklist_rows):
            return
        self._help_owner.release_draft()
        target = self._checklist_rows[row].target
        if target == 'orientation':
            self.open_orientation_check()
        elif row == len(self._checklist_rows) - 1:  # the check cannot run: its reason, where it is fixed (LT-008)
            self._show_tab(target)
            self._reveal(self._orientation_field)
            self.status.setText(self._checklist_rows[row].message)
        elif target == 'inspect' and self.fit is not None:
            self._show_calibration_fit()
        else:
            self._show_tab('calibrate' if target in ('fit', 'inspect', 'save') else target)
            if target in ('fit', 'inspect', 'save'):
                (self.save_button if target == 'save' else self.fit_button).setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
            elif target == 'instrument':
                self._reveal(self._instrument_field or self.geometry_editor.geometry)
            else:
                {'rough': self.rough_mode, 'select': self._fields['source'], 'calibrate': self.frame_table}[target].setFocus(
                     QtCore.Qt.FocusReason.OtherFocusReason)

    def _orientation_readiness(self):
        """Whether the Orientation check can run on the window's current inputs and, when not, why, the step that fixes
        it and the field to show there (LT-008): one fact for the checklist, the footer entry, the panel's Run, its
        finding pane and the checklist's navigation, read from the window's own owners with the run's own checks and
        no file read. Staged frames need the camera, motor mapping and source they were staged with, a fixed arm to
        predict their rings (the run reads its file) and a Mask file the window could read (its cached read); no
        frames, a current direct-beam fit. The run checks again (_start_orientation)."""
        if self.frames:
            message = self._staged_camera_message()
            if message:
                return False, message, 'instrument', self.camera.rotation
            try:
                geometry = self.geometry_editor.value()
                arm = geometry.stitch_arm()
            except (ValueError, TypeError, KeyError):
                arm = None
            if arm != self._frame_motors:
                field = self.geometry_editor.geometry
                if arm is not None:  # the motor box of the circle whose motor differs
                    played = 'del' if arm[0] != self._frame_motors[0] else 'nu'
                    circle = next((c for c in geometry.circles if geometry.as_psic().get(c, c) == played), None)
                    field = self.geometry_editor.motor_edit(circle) if circle is not None else field
                staged = ', '.join(f'{circle} → {motor}' for circle, motor in zip(('del', 'nu'), self._frame_motors)
                                   if motor)
                return (False, f'Restore the motor mapping the frames were staged with ({staged}) in '
                               '1 · Instrument ▸ Diffractometer Config, or start a New session for this mapping.',
                        'instrument', field)
            if self._staged_source_mismatch():
                return (False, 'Select the calibrant source and scan the frames were staged from (3 · Calibrant scan), '
                               'or start a New session for this source.', 'select', self._fields['source'])
            prediction = self._prediction is not None and self._prediction[0] == self._calibrated_key()
            seed_path = self._text('refine_seed_path')
            seed_state = getattr(self, '_refine_seed_state', None)
            seed_ready = bool(seed_path) and seed_state == (seed_path, '')
            if not ((self.fit is not None and self.fit.asset.get('calibration')) or prediction
                    or self._rough_record() is not None or seed_ready):
                if seed_path and self.refine_seed.currentData() == 'file':
                    problem = (seed_state[1] if seed_state is not None and seed_state[0] == seed_path else
                               'Select a readable geometry file.')
                    return (False, f'{problem} Choose a valid fixed geometry file (4 · Calibrate and fit).',
                            'calibrate', self._fields['refine_seed_path'])
                return (False, 'Enter or load a fixed rough geometry (2 · Rough geometry), or name a geometry file '
                               '(4 · Calibrate and fit), to check these frames against.', 'rough', self.rough_mode)
            cached = getattr(self, '_mask_cache', None)
            if (self._text('mask') and cached is not None and cached[1] is None and cached[0][0] == self._text('mask')
                    and (len(cached[0]) == 2 or cached[0][3] == self.camera.value().to_xu())):
                return (False, f"The Detector panel's Mask file cannot be read: {self._text('mask')}. Choose a readable "
                               'one (1 · Instrument).', 'instrument', self._fields['mask'])
            try:
                self._saturation_value()
            except ValueError:
                return (False, 'Type the saturation as a positive whole count, or leave it blank '
                               '(1 · Instrument ▸ Detector Config).', 'instrument', self._fields['saturation'])
        elif self._distance_result is None:
            return (False, 'Fit the direct-beam distance (2 · Rough geometry), or stage calibrant frames '
                           '(3 · Calibrant scan), to check the orientation.', 'rough', self.rough_mode)
        elif not self._distance_fresh():
            return False, self._STALE_DISTANCE, 'rough', self.rough_mode
        return True, '', 'orientation', None

    def _detector_config_problem(self):
        """The first malformed entry in Detector Config, as (refusal, field), or None: the Instrument row of the
        checklist names it and its navigation shows it (LT-009). The jobs read these as before."""
        def raw_shape():
            if len(tuple(int(value) for value in self._text('shape').split(','))) != 2:
                raise ValueError('two values')
        for key, check, refusal in (
                ('header', lambda: int(self._text('header')), 'Header (bytes) must be a whole number'),
                ('shape', raw_shape, 'RAW images rows, cols must be two whole numbers'),
                ('dtype', lambda: np.dtype(self._text('dtype')), 'RAW images type must be a NumPy type, such as int32'),
                ('saturation', self._saturation_value, 'Saturation (counts) must be blank or a positive whole count')):
            try:
                check()
            except (ValueError, TypeError):
                return f'Detector Config: {refusal}.', self._fields[key]
        return None

    def _project_orientation(self, ready, reason):
        """The readiness fact on the footer entry, the panel's Run and, while it shows no finding, its finding pane:
        disabled with the reason, never a stale message (LT-008)."""
        footer = self.orientation_button
        footer.apply_spec(replace(footer.spec, enabled=ready, reason='' if ready else reason))
        panel = self.orientation_panel
        if panel is not None:
            panel.set_readiness(ready, reason)
            if not ready and not self._owner.busy and panel.finding is None:
                panel.finding_pane.set_text(reason)

    def open_orientation_check(self):
        if self.orientation_panel is None:
            from .orientation_views import OrientationPanel
            self.orientation_panel = OrientationPanel(self, run=self._start_orientation, cancel=self._cancel,
                camera_focus=self._orientation_camera_focus, sign_focus=self._orientation_sign_focus,
                feature=self._orientation_feature)
            self._tab_pages['orientation'] = self.orientation_panel
            self.tabs.addTab(self.orientation_panel, 'Orientation check')
            self.tabs.setTabToolTip(self.tabs.indexOf(self.orientation_panel),
                                   'Read-only ring and motion evidence; edit geometry only in the existing controls.')
            self.orientation_panel.frame_row.editor.currentIndexChanged.connect(self._invalidate_orientation)
            self.orientation_panel.candidate_row.editor.currentIndexChanged.connect(self._orientation_candidate_changed)
        self.orientation_panel.set_frames(self.frames)
        self.orientation_panel.busy(self._owner.busy)
        self._refresh_checklist()  # its Run and finding pane take the readiness fact
        self.tabs.setCurrentWidget(self.orientation_panel)
        self.show(); self.raise_(); self.activateWindow()

    def _orientation_key(self):
        return (self._orientation_generation, tuple((f.identity, f.image_sha256, f.poni_sha256) for f in self.frames),
                self.camera.value(), self.geometry_editor.value(), id(self._distance_result), self._distance_key)

    def _invalidate_orientation(self, *_args):
        if os.environ.get('XDART_DEBUG_ORIENTATION'):
            field = next((k for k, v in self._fields.items() if v is self.sender()), None)
            _LOG.warning('orientation invalidate generation=%s field=%s job=%s busy=%s',
                         self._orientation_generation, field, self._job, self._owner.busy)
        self._orientation_generation += 1
        self._orientation_geometry = None
        self._orientation_figure = None
        if self.orientation_panel is not None:
            self.orientation_panel.invalidate()
        self._refresh_checklist()

    def _start_orientation(self, *_args):
        if self._owner.busy:
            return
        ready, reason, _target, _field = self._orientation_readiness()
        if not ready:  # Run is disabled then; the readiness fact refuses a run asked for anyway (LT-008)
            self.status.setText(reason)
            self.orientation_panel.finding_pane.set_text(reason)
            return

        def go():
            panel = self.orientation_panel
            preview_camera, preview_senses = self.camera.value(), (1, 1)
            if panel.finding is not None:
                index = panel.candidate_row.editor.currentData()
                if index is not None and 0 <= index < len(panel.finding.candidates):
                    candidate = panel.finding.candidates[index]
                    preview_camera, preview_senses = candidate.camera, candidate.senses
            if self.frames:
                if self._staged_camera_message():
                    raise ValueError(self._staged_camera_message())
                if self.geometry_editor.value().stitch_arm() != self._frame_motors or self._staged_source_mismatch():
                    raise ValueError('Restore the staged frames’ source and motor mapping before checking them.')
                selected = panel.frame_row.editor.currentData()
                frames = tuple(sorted(self.frames, key=lambda f: f.identity != selected))
                seed = self._seed_options()
                mode, options = 'rings', dict(frames=frames, seed=seed,
                    del_motor=self._frame_motors[0], nu_motor=self._frame_motors[1],
                    mask=self._loaded_mask(), conditioning=self._conditioning(), saturation=self._saturation_value(),
                    calibrant=seed['calibrant'], preview_camera=preview_camera, preview_senses=preview_senses)
                if self._text('mask') and options['mask'] is None:
                    raise ValueError('The configured static mask cannot be read.')
            else:
                if not self._distance_fresh():
                    raise ValueError(self._STALE_DISTANCE)
                from xrd_tools.core.geometry import ImageOrientation
                from xrd_tools.core.geometry.camera import stitch_camera
                fitted_options = json.loads(self._distance_key)[5]
                measured = stitch_camera(ImageOrientation(rotation=fitted_options['rotation'],
                    flip_horizontal=fitted_options.get('flip', False)))
                role = self._beam_role()
                motor_names = self.geometry_editor.value().stitch_arm()
                held = {circle: self._distance_held[name] for circle, name in zip(('del', 'nu'), motor_names)
                        if name in self._distance_held}
                mode, options = 'motion', dict(result=self._distance_result, measured_camera=measured,
                    detector=self._text('beam_detector'), motor=role, held=held,
                    first_pixel=int(self._text('beam_orientation')))
            self._orientation_pending_key = self._orientation_key()
            panel.invalidate()
            self._start('orientation', (self._orientation_pending_key, mode, options))
        try:
            go()
        except Exception as error:  # the run's own checks: its refusal, not another status message, in the pane
            self.status.setText(str(error))
            self.orientation_panel.finding_pane.set_text(str(error))

    def _orientation_candidate_changed(self, *_args):
        panel = self.orientation_panel
        if (panel is None or panel.finding is None or self._orientation_geometry is None
                or self._orientation_figure is None or self._owner.busy):
            return
        index = panel.candidate_row.editor.currentData()
        if index is None or not 0 <= index < len(panel.finding.candidates):
            return
        from xrd_tools.geometry_figures import redraw_orientation_display
        from .orientation_views import show_figures
        candidate = panel.finding.candidates[index]
        self._orientation_figure = redraw_orientation_display(self._orientation_figure,
            self._orientation_geometry, candidate.camera, candidate.senses)
        show_figures(panel.figure_view, (self._orientation_figure,))

    def _orientation_camera_focus(self):
        self._help_owner.release_draft()
        self._show_tab('instrument')
        self.activateWindow()
        self.camera.rotation.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _orientation_sign_focus(self):
        self._help_owner.release_draft()
        self._show_tab('instrument')
        self.activateWindow()
        geometry = self.geometry_editor.value()
        role = self._beam_role() or 'del'
        arm = geometry.stitch_arm()
        motor = arm[0 if role == 'del' else 1]
        circle = next((c for c, name in geometry.motors if name == motor and c in geometry.circles), None)
        if circle is None:
            self.status.setText('Select a measured detector circle to show its motor direction.')
            return
        self._reveal(self.geometry_editor.axis_box(circle))  # inside Diffractometer Config (LT-009)

    def _orientation_feature(self, _key, column, row):
        panel = self.orientation_panel
        if panel is None or panel.step_row.editor.currentIndex() != 2:
            return
        if self._orientation_geometry is None or panel.finding is None:
            panel.finding_pane.set_text('Run on a staged calibrant frame before selecting a known feature.')
            return
        from xrd_tools.geometry_figures import feature_azimuth
        index = panel.candidate_row.editor.currentData()
        camera = panel.finding.candidates[index].camera if index is not None else self.camera.value()
        direction = panel.feature_row.editor.currentText()
        projection = panel.figure_view.keyed_projection
        mapping = projection.panels[0].figure.display_binning if projection else None
        pixel = mapping.display_to_native((column, row)) if mapping else (column, row)
        result = feature_azimuth(self._orientation_geometry, camera, pixel,
                                  expected_lab=None if direction == 'Undeclared' else direction)
        panel.finding_pane.set_text(result.reason)

    def show_instrument_schematic(self):
        if self.instrument_schematic is None:
            from .orientation_views import InstrumentSchematicWindow
            self.instrument_schematic = InstrumentSchematicWindow(self)
        self._redraw_schematic()
        self.instrument_schematic.show(); self.instrument_schematic.raise_()

    def set_schematic_readbacks(self, readings, pose='selected readbacks'):
        """Draw the schematic at these motor readbacks; ``pose`` names them in its titles (LT-007)."""
        self._schematic_readings = dict(readings)
        self._schematic_pose = pose
        self._redraw_schematic()

    def _schematic_frame_selected(self):
        row = self.frame_table.currentRow()
        if 0 <= row < len(self.frames):
            self.set_schematic_readbacks(self.frames[row].motors, f'frame {self.frames[row].index} readbacks')

    def _redraw_schematic(self, *_args):
        if self.instrument_schematic is None:
            return
        geometry = self.geometry_editor.value()
        readings = self._schematic_readings
        if readings is None:  # an explicit example, named as such in the titles and the message pane
            readings = {motor: 0. for _, motor in geometry.motors}
        self.instrument_schematic.show_geometry(geometry, readings, self._schematic_pose)

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
        from xdart.gui.widgets.context_help import bind_topic
        bind_topic(page, {'instrument': 'geometry.orientation', 'rough': 'geometry.orientation',
                         'select': 'calibration.frame-choice', 'calibrate': 'calibration.diagnostics'}[key])
        return layout, form

    def _show_tab(self, key):
        self.tabs.setCurrentWidget(self._tab_pages[key])

    @staticmethod
    def _tip(tip, *widgets):
        for widget in widgets:
            if tip and widget is not None:
                widget.setToolTip(tip)

    def _combo(self, form, label, items, *, tip='', host=None):
        """One Controls combo row (Int's ``FormRow``) of ``(text, data)`` items; returns its editor. The
        items are added here with their data (a factory name, a code), which FormRow's text choices do
        not carry: this window drives its rows directly, never through ``apply_field``. The row's label
        sits in the form's label column, so every row aligns."""
        row = FormRow(label=label, path=('Calibration', label), value=None, kind='combo')
        combo = row.editor
        for text, value in items:
            combo.addItem(text, value)
        if host is None:
            form.addRow(row.label, row)
        else:
            row.label.setMinimumWidth(0)
            self._formrows[host].add_trailing_widget(row, 1)
        self._row_of[combo] = row
        self._tip(tip, combo, row.label)
        return combo

    def _choice(self, form, key, label, items, default, *, tip=''):
        """A Controls combo read and saved like a text field (its data: a factory name, a code)."""
        combo = self._combo(form, label, items, tip=tip)
        self._fields[key] = combo
        self._formrows[key] = self._row_of[combo]
        self._rows[key] = (form, self._row_of[combo])
        self._set_field_text(key, default)
        return combo

    def _edit(self, form, key, label, default='', *, chooser=None, tip='', placeholder=''):
        """One text row (Int's ``FormRow``), with its 📁 button when ``chooser`` names a file filter,
        'directory' or 'spec' (SPEC files only), or is a function returning one at the click. A file row
        shows its file's name, the full path on hover and while editing, as Int's Controls do; its value is
        read through the row (``_text``). The row's label sits in the form's label column."""
        row = FormRow(label=label, path=('Calibration', key), value=str(default), browse=chooser is not None)
        edit = row.editor
        edit.setPlaceholderText(placeholder)
        if chooser is not None:
            row.browseRequested.connect(lambda _path: self._browse(key, chooser))
            self._choosers[key] = chooser
        form.addRow(row.label, row)
        self._fields[key] = edit
        self._formrows[key] = row
        self._rows[key] = (form, row)
        self._tip(tip, edit, row.label)
        return edit

    def _attach(self, key, host, label, default='', *, tip='', placeholder='', width=None):
        """A second field on a file row (the scan beside its SPEC file, the stem beside its folder): a compact
        ``FormRow``, its label a caption, after the host row's editor."""
        row = FormRow(label=label, path=('Calibration', key), value=str(default))
        row.label.setMinimumWidth(0)
        edit = row.editor
        edit.setPlaceholderText(placeholder)
        if width:
            edit.setMaximumWidth(width)
        self._formrows[host].add_trailing_widget(row, 0 if width else 1)
        self._fields[key] = edit
        self._formrows[key] = row
        self._parts[key] = (host, row)
        self._tip(tip, row.label, edit)
        return edit

    def _show(self, key, shown):
        """Show or hide a field: its form row, or its compact row when it shares another field's row. The
        editor itself follows too, so the field says whether it is shown."""
        if key in self._parts:
            self._parts[key][1].setVisible(shown)
        else:
            form, row = self._rows[key]
            form.setRowVisible(row, shown)
        if key in self._formrows:
            self._fields[key].setVisible(shown)

    def _row_visible(self, key):
        if key in self._parts:
            host, row = self._parts[key]
            return self._row_visible(host) and not row.isHidden()
        form, row = self._rows[key]
        return form.isRowVisible(row)

    def _label(self, key):
        form, row = self._rows[key]
        return form.labelForField(row)

    def _auto_placeholders(self, prefix=''):
        """The automatic image folder and stem for a blank field, as the page shows them."""
        spec = self._text(prefix + 'source')
        path = Path(spec) if spec else None
        first = re.match(r'\s*(\d+)', self._text(prefix + 'scan'))
        self._fields[prefix + 'images'].setPlaceholderText(
            f'auto · {path.parent}' if path else "auto · the SPEC file's folder")
        self._fields[prefix + 'stem'].setPlaceholderText(
            f"auto · {path.stem if path else '<SPEC name>'}_scan{first[1] if first else '<N>'}_")
        for key in (prefix + 'images', prefix + 'stem'):
            self._quick_mirror(key)

    def _browse(self, key, chooser):
        from xdart.gui.widgets.file_dialogs import browse, browse_spec
        chooser = chooser() if callable(chooser) else chooser
        folder = chooser == 'directory'
        current = self._text(key)
        if chooser == 'spec':
            picks = browse_spec(self, 'Select SPEC file', current=current, fallback=str(self.project_root))
        else:
            picks = browse(self, 'Select images directory' if folder else 'Select file', folder=folder,
                           file_filter='All files (*)' if folder else chooser, current=current,
                           fallback=str(self.project_root))
        if picks and picks[0]:
            self._set_field_text(key, picks[0])
            if key in ('beam_source', 'beam_images', 'source', 'images'):
                if key == 'beam_source':
                    self._set_field_text('beam_image_file', '')
                self._source_details('beam_' if key.startswith('beam_') else '')

    def _source_details(self, prefix=''):
        """Committed source fields only: reuse the worker for pixel-free metadata and naming."""
        if self._restoring_fields:
            return
        if self._owner.busy:
            self._pending_source_details = prefix
            return
        self._source_details_orientation_inputs = (self._orientation_key(), self._source_energy_eV)
        if not prefix:
            self._source_energy_eV = None
            self._calibrant_ready_key = None
            self._calibrant_frame_count = None
            self._calibrant_source_problem = ''
            self._refresh_energy_hint()
            self._source_details_key = self._energy_source_key()
        path = self._text(prefix + 'source')
        if not path:
            return
        kind = (SourceKind.NEXUS_STACK if self.beam_source_kind.currentData() == 'nexus'
                else SourceKind.SPEC) if prefix else (
                    SourceKind.NEXUS_STACK if self.source_kind.currentData() == 'nexus_stack' else SourceKind.SPEC)
        spec = self._spec(beam=bool(prefix))
        folder = self._image_folder(prefix)
        self._auto_placeholders(prefix)
        self._source_details_inputs_key = self._source_details_input_key(prefix)
        self._start('source_details', (prefix, spec, folder, self._text(prefix + 'stem')))

    def _source_details_input_key(self, prefix):
        return tuple(self._text(prefix + key) for key in ('source', 'scan', 'entry', 'images', 'stem')) + (
            self.beam_source_kind.currentData() if prefix else self.source_kind.currentData(),)

    def _energy_source_key(self):
        return (self._text('source'), self._text('scan'), self._text('entry'),
                self._text('images'), self._text('stem'), self.source_kind.currentData())

    def _invalidate_source_energy(self, *_args):
        if self._restoring_fields:
            return
        self._source_energy_eV = None
        self._calibrant_ready_key = None
        self._calibrant_frame_count = None
        self._calibrant_source_problem = ''
        self._choices = []
        self._refresh_energy_hint()
        self._update_fit_source_label()

    def _calibrant_source_changed(self, *_args):
        if self._restoring_fields:
            return
        # A new file has its own scan selection and image naming, never the previous file's overrides.
        self._set_field_text('scan', '1')
        self._set_field_text('images', '')
        self._set_field_text('stem', '')
        self._set_field_text('rough_energy', '')
        self._invalidate_source_energy()

    def _show_source_energy(self, energy):
        self._source_energy_eV = energy
        self._refresh_energy_hint()

    def _refresh_energy_hint(self):
        """The Energy field's grey Auto hint: the energy a blank field takes and where it is from (the scan's
        recorded one, else the calibrated frames' PONI files'; the direct-beam scan's is said by its fit)."""
        from xrd_tools.core.energy import wavelength_m_to_energy_eV
        energy, origin = self._source_energy_eV, ('SPEC' if self.source_kind.currentData() != 'nexus_stack'
                                                  else 'NeXus')
        if energy is None and not any(value is not None for _index, _motors, value in self._choices):
            wavelength = self._frames_poni_wavelength_m()
            if wavelength is not None:
                energy, origin = float(wavelength_m_to_energy_eV(wavelength)), "the calibrated frames' PONI files"
        self._fields['rough_energy'].setPlaceholderText(
            f'Auto · {energy / 1000:.4f} keV from {origin}' if energy is not None else 'Auto · scan energy unavailable')

    def _frames_poni_wavelength_m(self):
        """The wavelength (m) the calibrated frames' PONI files agree on, None when there are none, they disagree
        or the frames were staged from another source (maintainer, 2026-10-09: "if there is no typed in energy but
        there is a poni file, it should use the energy from the poni file"). Read as Fit reads them; cached by file."""
        files = tuple((frame.poni_path, frame.poni_sha256) for frame in self.frames if frame.complete)
        if not files or self._staged_source_mismatch():
            return None
        if self._poni_wavelength[0] != files:
            from xrd_tools.integrate.calibration import load_detector_calibration
            values = []
            for path, _sha in files:
                try:
                    value = float(load_detector_calibration(path).poni.wavelength)
                except Exception:  # a file Fit would refuse says nothing here; drawing the table never fails on it
                    continue
                if math.isfinite(value) and value > 0:
                    values.append(value)
            agreed = values[0] if values and all(math.isclose(value, values[0], rel_tol=1e-6)
                                                 for value in values) else None
            self._poni_wavelength = (files, agreed)
        return self._poni_wavelength[1]

    def _calibrant_source_ready(self):
        return self._calibrant_ready_key is not None and self._calibrant_ready_key == self._energy_source_key()

    def _staged_source_mismatch(self):
        if not self.frames or not self._text('source'):
            return False
        frame = self.frames[0]
        try:
            source = self._spec()
            scan = str(source.options.get('scan', '1'))
            staged_scan = str(frame.source_options.get('options', {}).get('scan', '1'))
            if '.' not in scan:
                scan += '.1'
            if '.' not in staged_scan:
                staged_scan += '.1'
            return (str(source.uri) != frame.source_uri or source.kind.value != frame.source_kind
                    or (source.kind is SourceKind.SPEC and scan != staged_scan)
                    or source.entry != frame.source_options.get('entry', 'entry'))
        except (ValueError, KeyError, OSError):
            return True

    def _update_fit_source_label(self):
        if not hasattr(self, 'active_source_row'):
            return
        if self.frames:
            frame = self.frames[0]
            scan = frame.source_options.get('options', {}).get('scan')
            text = f'Prepared: {frame.source_uri}' + (f' · scan {scan}' if scan is not None else '')
            if self._staged_source_mismatch():
                text += ' · Selected source differs; prepare matching frames in a new session'
        elif self._calibrant_source_ready():
            text = self._text('source') + (f' · scan {self._text("scan")}'
                    if self.source_kind.currentData() != 'nexus_stack' else '')
        else:
            text = 'Select a valid calibrant source and read its metadata (3 · Calibrant scan)'
        if self._calibrant_frame_count is not None and not (self.frames and self._staged_source_mismatch()):
            text += f' · {self._calibrant_frame_count} images'
        if self._calibrant_source_problem:
            text += f' · {self._calibrant_source_problem}'
        self.active_source_row.editor.setText(text)
        self.active_source_row.editor.setToolTip(text)
        self._refresh_checklist()

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
        layout, form = self._form('3 · Calibrant scan', 'select')
        self.source_kind = self._combo(form, 'Source type', [('SPEC + image series', 'spec'),
                                                             ('SPEC + Eiger / NeXus', 'spec_container'),
                                                             ('NeXus images + metadata', 'nexus_stack')],
            tip='SPEC supplies each frame\'s angles; an image series or an Eiger/NeXus container supplies '
                'the frames. Standalone NeXus holds both images and metadata. ' + _FROM_DATA_FORMAT)
        self.source_kind.setEnabled(False)  # the page's Data format sets it (LT-005)
        self._edit(form, 'source', 'SPEC file',
                   chooser=lambda: NEXUS_FILTER if self.source_kind.currentData() == 'nexus_stack' else 'spec',
                   tip='The calibrant scan: its SPEC file, or its NeXus file.')
        self._attach('scan', 'source', 'Scan', '1', tip=self._SCAN_TIP, width=90)
        self._edit(form, 'images', 'Images folder', chooser='directory', tip=self._FOLDER_TIP)
        self._attach('stem', 'images', 'Stem', '', tip=self._STEM_TIP)
        self._formrows['source_kind'] = self._row_of[self.source_kind]
        self._rows['source_kind'] = (form, self._row_of[self.source_kind])
        self._attach('entry', 'source_kind', 'NeXus entry', 'entry', width=120,
                     tip='The NeXus entry (group) holding the frames and angles.')
        from xrd_tools.integrate.calibration import CALIBRANT_NAMES
        self.refine_calibrant = self._combo(form, 'Calibrant', [(name, None) for name in CALIBRANT_NAMES],
            tip='The powder standard in this scan: pyFAI opens each frame with it, the table predicts its '
                'rings, and the refinement fits them. Another pyFAI calibrant name may be typed.')
        self.refine_calibrant.setEditable(True)
        self.refine_calibrant.currentTextChanged.connect(self._render_frames)
        self._formrows['calibrant'] = self._row_of[self.refine_calibrant]
        self._rows['calibrant'] = (form, self._row_of[self.refine_calibrant])
        self._attach('rough_energy', 'calibrant', 'Energy (keV)', '', width=300,
                     placeholder='Auto · scan energy unavailable',
                     tip='Blank uses the calibrant scan\'s recorded energy or wavelength; when it records none (or '
                         'an impossible one), the wavelength of the calibrated frames\' PONI files, when they agree. '
                         'The grey Auto hint shows that energy in keV and where it is from. Type a value to '
                         'override it; clear it to restore Auto. Wavelength is computed from this value; the '
                         'direct-beam distance fit needs no energy. Saving a rough calibration needs an energy: '
                         'this, else the one a blank field uses, else the direct-beam scan\'s recorded energy'
                         + _PAGE_USE.get(self.context.page, {}).get('energy_field', '') + '.')
        self._fields['rough_energy'].setMinimumWidth(300)
        self._fields['rough_energy'].editingFinished.connect(self._render_frames)
        self.source_kind.currentIndexChanged.connect(self._sync_source_rows)
        self._fields['source'].textChanged.connect(self._calibrant_source_changed)
        for key in ('scan', 'entry', 'images', 'stem'):
            self._fields[key].textChanged.connect(self._invalidate_source_energy)
        self.source_kind.currentIndexChanged.connect(self._invalidate_source_energy)
        for key in ('source', 'scan'):
            self._fields[key].textChanged.connect(lambda _text: self._auto_placeholders())
        for key in ('source', 'scan', 'entry', 'images', 'stem'):
            self._fields[key].editingFinished.connect(lambda: self._source_details())
        self._sync_source_rows()
        self._auto_placeholders()
        self.list_button = QtWidgets.QPushButton('Read metadata and suggest 5 frames')
        self.list_button.setToolTip('Lists the scan\'s frames with their angles and ticks five that span them.')
        self.list_button.clicked.connect(self._enumerate)
        layout.addWidget(self.list_button)
        self.selection = ResultsTable(('Use', 'Frame', 'Delta (°)', 'Nu (°)'),
                                      row_numbers=True, stretch_columns=True,
                                      header_tooltips={0: 'Prepare the checked frames.',
                                                       1: 'Frame index in the detector scan.'})
        self.selection.setToolTip('Check representative frames to prepare for calibration; angles come from the scan.')
        layout.addWidget(self.selection, 1)
        self.prepare_button = QtWidgets.QPushButton('Prepare selected images')
        self.prepare_button.setToolTip('Copies the ticked frames into this session\'s folder, ready for pyFAI.')
        self.prepare_button.clicked.connect(self._prepare)
        layout.addWidget(self.prepare_button)

    def _sync_source_rows(self, *_args):
        kind = self.source_kind.currentData()
        spec = kind != 'nexus_stack'
        for key, shown in (('scan', spec), ('images', spec),
                           ('entry', kind != 'spec' and self._shared_nexus_entry is None)):
            self._show(key, shown)
        self._label('source').setText('SPEC file' if spec else 'NeXus file')
        self._sync_quick_rows()

    def _conditioning(self):
        state = self._threshold_state
        return (StitchConditioning('manual', state.threshold_min, state.threshold_max) if state.apply_threshold
                else StitchConditioning('auto' if state.mask_saturation else 'off'))

    def _threshold_row(self):
        """Int's shared threshold row on this window's one threshold state: every row shows that state
        (_show_threshold) and edits it through the shared transitions (_threshold_edited); its tooltips are the
        shared registry's for these paths."""
        from xdart.gui.widgets.controls_panel import RangeRow
        row = RangeRow(label='Threshold',
                       low={'path': ('Calibration', 'min'), 'value': 0},
                       high={'path': ('Calibration', 'max'), 'value': None, 'placeholder': 'Auto'},
                       toggle={'path': ('Calibration', 'enabled'), 'value': True},
                       display_decimals=0)
        row.label.setToolTip('Which counts calibrant ring finding and the automatic refinement use, by Int\'s '
                             'Threshold rule: on by default, Auto leaving saturated pixels out. The switch and the '
                             'bounds say more.')
        row.valueChanged.connect(self._threshold_edited)
        self._threshold_rows.append(row)
        return row

    def _show_threshold(self):
        if not self._threshold_rows:
            return
        from xrd_tools.session.readiness import ControlFormField, SectionId
        try:
            ceiling = self._saturation_value()
            if ceiling is not None and (not math.isfinite(ceiling) or ceiling <= 0):
                ceiling = None
        except ValueError:
            ceiling = None
        self._threshold_default_max, hint = auto_ceiling_hint(
            [None if ceiling is None else (ceiling, 'saturation')])
        state = self._threshold_state
        automatic = state.mask_saturation and not state.apply_threshold
        values = {'min': 0 if automatic or state.threshold_min is None else state.threshold_min,
                  'max': None if automatic else state.threshold_max,
                  'enabled': state.apply_threshold or state.mask_saturation}
        fields = {('Calibration', key): ControlFormField(
            SectionId.PROCESSING, 'Threshold', ('Calibration', key), value=value,
            placeholder=hint if key == 'max' else None) for key, value in values.items()}
        for row in self._threshold_rows:
            row.apply_fields(fields)

    def _threshold_edited(self, path, value):
        result = threshold_transition(self._threshold_state, path[-1], value, self._threshold_default_max)
        if result is None or isinstance(result, ThresholdEditRefusal):
            self._show_threshold()
        if isinstance(result, ThresholdEditRefusal):
            self.status.setText(result.reason)
        elif result is not None:
            self._threshold_state = result
            self._show_threshold()
            self._invalidate_beam_detection()
            self._ring_cache.clear()
            self._invalidate_fit()
            self._render_frames()
            self._save_session_quietly()

    def _build_frames(self):
        # Calibrating the frames and fitting them are one loop: one tab (maintainer, 2026-09-29)
        layout, form = self._form('4 · Calibrate and fit', 'calibrate')
        self._calibrate_layout = layout
        self.active_source_row = FormRow(label='Calibrant source', path=('Calibration', 'active_source'), value='')
        self.active_source_row.editor.setReadOnly(True)
        form.addRow(self.active_source_row.label, self.active_source_row)
        hint = QtWidgets.QLabel('Calibrate each frame in pyFAI: pick its rings, then press Save as PONI-file. xdart keeps the PONI and its enabled control points, closes pyFAI and selects the next frame. Completed rows and files remain when another frame is cancelled.')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        # the fit shares the tab: the table keeps room for about six frames, the tab scrolls
        self.frame_table = ResultsTable(
            ('Frame', 'Delta / Nu (°)', 'Image', 'PONI / points', 'Status', 'Expected rings'),
            row_numbers=True, stretch_columns=True, row_selection=True, minimum_rows=6,
            header_tooltips={0: 'Frame index in the detector scan.',
                             1: 'The detector circles\' angles for this frame.',
                             2: 'Original detector image and staged pyFAI image; hover for their paths.',
                             3: 'The PONI and control points pyFAI saved for this frame.',
                             4: 'Whether this frame has a matched PONI and control points.',
                             5: 'The calibrant rings predicted to reach this frame; hover a cell for each ring\'s 2θ.'})
        self.frame_table.setToolTip('Select a frame to calibrate it or import its matched PONI and control points.')
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
        # Above both buttons, as both fits take its options; collapsed (maintainer, 2026-09-30:
        # "Advanced should Move above 'Fit the Calibrated frames'", the energy option inside it).
        self._advanced_card, options = self._card(layout, 'Advanced')
        self.point_images = PointImagesSettings()
        options.addRow(self.point_images)
        self.point_images.changed.connect(lambda: self._source_details())
        self.threshold_range = self._threshold_row()
        options.addRow(self.threshold_range.label, self.threshold_range)
        self._fields['saturation'].editingFinished.connect(self._show_threshold)
        self._show_threshold()
        # The energy, refined within ± a bound of the recorded one (maintainer, 2026-09-29: "add a
        # refine energy toggle. If selected there should be an option to set the constraint - just
        # one number for +-that number"); Fit and Refine both take it. On by default, ±100 eV
        # (maintainer, 2026-09-29, D1: "+-100 and on by default"): with the sample held near the
        # rotation centre, a fixed energy's error lands in the zeros and the sample offset.
        energy_row = QtWidgets.QWidget()
        energy_layout = QtWidgets.QHBoxLayout(energy_row)
        energy_layout.setContentsMargins(0, 0, 0, 0)
        refine_row = FormRow(label='Refine energy', path=('Calibration', 'refine_energy'), value=True, kind='bool')
        self.refine_energy = refine_row.editor
        use = _PAGE_USE.get(self.context.page, {})
        energy_tip = ('Also fits the beam energy, within ± the bound of the frames\' recorded energy. The arm\'s '
                      'known angles give an absolute 2θ scale, so the calibrant rings can fix it; with frames on only '
                      'one side of zero it trades off against the zero offsets and the sample offset, and the fit '
                      'summary gives its uncertainty.' + use.get('energy', ''))
        self.refine_energy.setToolTip(energy_tip)
        bound_row = FormRow(label='±', path=('Calibration', 'energy_bound'), value='100')
        bound_row.label.setMinimumWidth(0)
        bound = bound_row.editor
        bound.setMaximumWidth(64)
        bound.setToolTip('The refined energy stays within this many eV of the recorded one.')
        self._fields['energy_bound'] = bound
        self._formrows['energy_bound'] = bound_row
        energy_layout.addWidget(refine_row)
        energy_layout.addWidget(bound_row)
        energy_layout.addWidget(QtWidgets.QLabel('eV'))
        energy_layout.addStretch(1)
        options.addRow(energy_row)
        scales_row = FormRow(label='Fit motor scales', path=('Calibration', 'fit_motor_scales'), value=False,
                             kind='bool')
        self.scales = scales_row.editor
        self.scales.setToolTip('Also fits each motor\'s scale (degrees of arm rotation per motor degree), '
                               'where the frames span enough angle to tell. Off: exactly one degree per '
                               'degree, which is right for almost every diffractometer. A calibration with '
                               'fitted scales has no xrayutilities file: xrayutilities takes angles in degrees.')
        options.addRow(scales_row)
        self._edit(options, 'rms', 'Largest accepted RMS |q| (Å⁻¹)', '.03',
                   tip='A fit whose RMS ring-position error, in |q|, exceeds this is refused.')
        self._edit(options, 'distance_seed', 'Starting distance (mm)', '',
                   tip='Where the fit starts, e.g. the direct-beam distance. Blank: the first calibrated '
                       'frame\'s.')
        self._edit(options, 'distance_bounds', 'Distance range (min, max mm)', '',
                   tip='Keeps the fitted distance between these. Blank: within ± the Distance percentage below.')
        # How far each fit may move from where it starts (maintainer, 2026-09-29: "this is to make
        # sure the sample displacement in particular doesn't go to unrealistic values")
        heading = QtWidgets.QLabel('How far the fit may move each value from where it starts:')
        heading.setWordWrap(True)
        options.addRow(heading)
        for key, label, tip in self._BOUND_FIELDS:
            self._edit(options, key, label, self._BOUND_DEFAULTS[key], tip=tip)
        self.bounds_note = QtWidgets.QLabel()
        self.bounds_note.setWordWrap(True)
        options.addRow(self.bounds_note)
        self.reset_bounds_button = QtWidgets.QPushButton('Reset to defaults')
        self.reset_bounds_button.setToolTip('The maintainer\'s defaults: ± 2° for the zeros and the mount, '
                                            '± 0.05 mm for each sample displacement, ± 5 % for the distance.')
        self.reset_bounds_button.clicked.connect(self._reset_bounds)
        options.addRow(self.reset_bounds_button)
        self.fit_button = QtWidgets.QPushButton('Global fit of calibrated frames')
        self.fit_button.setToolTip('Fits one detector-arm geometry to the ring points picked on the frames '
                                   'marked Complete above. Needs three of them.')
        self.fit_button.clicked.connect(self._fit)
        layout.addWidget(self.fit_button)
        # Open (maintainer, 2026-09-30: "the Refine from the whole scan tab should be expanded by default")
        self._refine_card, auto = self._card(layout, 'Refine from the whole scan (optional)')
        self._refine_card.set_collapsed(False)
        self.refine_seed = self._combo(auto, 'Start from', [('Current fit (unsaved)', 'fit'), ('Geometry file', 'file'),
                                                            ('Rough geometry (tab 2)', 'rough')],
            tip='Current fit: the fit just made above. Geometry file: a geometry this window saved. Rough '
                'geometry: the distance and direct-beam pixel of 2 · Rough geometry, with no frame calibrated '
                '(on 17-2\'s LaB6 scan it refines about as well as from a ring fit).')
        self._formrows['refine_seed'] = self._row_of[self.refine_seed]
        self._attach('refine_frames', 'refine_seed', 'Frame budget', '48', width=90,
                     tip='At most this many frames of the calibrant scan are searched for ring points, '
                         'spread across its angles. More: better coverage, slower.')
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
        self.refine_button = QtWidgets.QPushButton('Refine automatically')
        self.refine_button.setToolTip('Finds ring points on frames across the whole calibrant scan, starting '
                                      'from the geometry above, and keeps the detector mount and sample offset '
                                      'only when they predict held-out nu rows better.')
        self.refine_button.clicked.connect(self._refine)
        auto.addRow(self.refine_button)
        self.fit_summary = ReadOnlyTextPane()  # the shared read-only text (Codex's stage-2b review F2)
        self.fit_summary.setMinimumHeight(self.fit_summary.fontMetrics().lineSpacing() * 8)
        layout.addWidget(self.fit_summary, 1)
        self.show_calibration_fit_button = QtWidgets.QPushButton('Show the fit…')
        self.show_calibration_fit_button.setToolTip('The fit\'s control points in their own window: each '
                                                    'one\'s |q| residual across the arm\'s angles and against '
                                                    'its ring (opened after every Fit or Refine).')
        self.show_calibration_fit_button.clicked.connect(self._show_calibration_fit)
        self.show_calibration_fit_button.setEnabled(False)
        use = _PAGE_USE.get(self.context.page, {})
        self.save_button = QtWidgets.QPushButton(f'Save geometry and use in {self.context.page}')
        self.save_button.setToolTip('Saves the fitted geometry inside the project as a pyFAI goniometer JSON and, '
                                    'beside it, the same calibration for xrayutilities (<name>.xu.json, checked '
                                    'against pyFAI on every pixel), and makes it '
                                    f'{self.context.page}\'s calibration{use.get("backend", "")}.')
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.show_calibration_fit_button, 1)
        buttons.addWidget(self.save_button, 1)
        layout.addLayout(buttons)
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
        # Reuse this existing validation on readiness projections; the path alone is not fixed geometry, and
        # Orientation must not open/read the seed again on each edit or checklist refresh.
        self._refine_seed_state = (self._text('refine_seed_path'), note) if from_file else None
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

    def _build_instrument(self):
        """The instrument every step reads: the diffractometer's circles and their motors, then the
        detector, one above the other (maintainer, 2026-09-30)."""
        layout, _ = self._form('1 · Instrument', 'instrument')
        self._instrument_layout = layout
        from xdart.gui.widgets.geometry_editor import GeometryEditor
        self._diffractometer_box = QtWidgets.QGroupBox('Diffractometer')
        box = QtWidgets.QVBoxLayout(self._diffractometer_box)
        self.geometry_editor = GeometryEditor(used=('nu', 'del', 'tth'), used_by=self.context.page)
        box.addWidget(self.geometry_editor)
        self._detector_box = self._build_detector()
        layout.insertWidget(0, self._diffractometer_box)
        layout.insertWidget(1, self._detector_box)
        layout.addStretch(1)
        self.geometry_editor.changed.connect(self._diffractometer_title)
        self._diffractometer_title()
        self.geometry_editor.changed.connect(self._geometry_edited)
        self.geometry_editor.changed.connect(self.geometryChanged.emit)
        self.geometry_editor.changed.connect(self._bounds_edited)
        self.geometry_editor.changed.connect(self._setup_edited)
        self._fields['beam_detector'].currentIndexChanged.connect(self._setup_edited)
        self._fields['beam_orientation'].currentIndexChanged.connect(self._setup_edited)
        # the arm's motors are these circles' (read by name as before)
        self._fields['del_motor'] = self.geometry_editor.motor_edit('del')
        self._fields['nu_motor'] = self.geometry_editor.motor_edit('nu')
        for circle in ('nu', 'del', 'tth'):  # the zeros and senses place the rough arm and start every fit
            # GeometryEditor.changed already invalidates the finding as the zero changes. Focus-out while
            # folding Config only refreshes the view; it must not invalidate the unchanged scientific key again.
            self.geometry_editor.zero_edit(circle).editingFinished.connect(
                lambda: self._render_frames(invalidate_orientation=False))
            self.geometry_editor.zero_edit(circle).textChanged.connect(self._bounds_edited)
            self.geometry_editor.axis_box(circle).currentIndexChanged.connect(self._render_frames)

    def _diffractometer_title(self, *_args):
        """The Diffractometer panel's title summarises the selected geometry and whether its motor names and senses
        are its standard ones (LT-009), visible with Config hidden; the tooltip names what differs."""
        summary, tip = self.geometry_editor.summary()
        self._diffractometer_box.setTitle(f'Diffractometer · {summary}')
        self._diffractometer_box.setToolTip(tip)

    def _reveal(self, widget):
        """Focus ``widget``, first showing the Config that holds it when hidden (LT-009: a field named as a problem is
        shown where it is fixed)."""
        for card in (self.geometry_editor.config, self._detector_config):
            if card.isAncestorOf(widget):
                card.set_collapsed(False)
        widget.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _build_quick(self):
        """Quick calibration (M6, maintainer 2026-10-08): a card in 1 · Instrument, below the Diffractometer and
        Detector panels (LT-009), with the two scans that identify a calibration, their file and scan number (the
        values 2 · Rough geometry and 3 · Calibrant scan use, edited in either place; no second copy, nothing saved of
        its own), the window's Threshold (Advanced's, likewise; maintainer, 2026-10-08) and one action that runs the
        window's own direct-beam fit, metadata read and Refine from the rough geometry (_quick_advance). Its mask is
        the Detector panel's one Mask file, exposed just above (LT-009: no duplicate row)."""
        from .orientation_views import action
        self.quick_card = SectionCard('Quick calibration')
        tip = ('Fit direct-beam distance (2 · Rough geometry), Read metadata (3 · Calibrant scan), then Refine '
               'automatically from the rough geometry (4 · Calibrate and fit), one after another. It stops at the '
               'first problem and shows that step; inspect the fit and save it by hand.')
        self.quick_card.setToolTip(tip)
        self.quick_card.title.setToolTip(tip)
        self.quick_card.status.setToolTip('The step Quick calibration is on, where it stopped, or that it finished.')
        body = QtWidgets.QWidget()
        form = self._quick_form = QtWidgets.QFormLayout(body)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        modes = {self.rough_mode.itemData(i): self.rough_mode.itemText(i) for i in range(self.rough_mode.count())}
        self.quick_rough_mode = FormRow(label='Rough geometry', path=('Calibration', 'Quick', 'rough_mode'),
            value=self.rough_mode.currentData(), kind='combo', choices=tuple(modes), choice_labels=modes)
        self.quick_rough_mode.set_purpose_tooltip(
            'The same selection as 2 · Rough geometry. Quick calibration requires Direct-beam scan; '
            'select it to show and choose that scan. Enter the distance and Skip remain available for manual calibration.')
        self.quick_rough_mode.valueChanged.connect(
            lambda _path, value: self.rough_mode.setCurrentIndex(self.rough_mode.findData(value)))
        form.addRow(self.quick_rough_mode.label, self.quick_rough_mode)
        for scan, prefix, tab in (('Direct-beam', 'beam_', '2 · Rough geometry'), ('Calibrant', '', '3 · Calibrant scan')):
            file_row = FormRow(label=f'{scan} SPEC file', path=('Calibration', 'Quick', prefix + 'source'), value='',
                               browse=True)
            scan_row = FormRow(label='Scan', path=('Calibration', 'Quick', prefix + 'scan'), value='')
            scan_row.label.setMinimumWidth(0)
            scan_row.editor.setMaximumWidth(90)
            file_row.add_trailing_widget(scan_row, 0)
            # the page's Data format shows each role's own fields here, as its tab does (LT-005): a NeXus or container
            # scan's entry, and a SPEC scan's images folder and stem
            entry_row = FormRow(label='Entry', path=('Calibration', 'Quick', prefix + 'entry'), value='')
            entry_row.label.setMinimumWidth(0)
            entry_row.editor.setMaximumWidth(120)
            file_row.add_trailing_widget(entry_row, 0)
            form.addRow(file_row.label, file_row)
            images_row = FormRow(label='Image Path', path=('Calibration', 'Quick', prefix + 'images'),
                                 value='', browse=True)  # under its scan's file row, which names the role
            copy_folder = action('Use for both',
                f"Copy this {scan.lower()} scan's image folder once to the other scan. Auto uses this SPEC "
                "file's folder. Each scan keeps its own Image Stem; clear a field to restore Auto.", self)
            self._quick_folder_buttons[prefix] = copy_folder
            copy_folder.clicked.connect(lambda _checked=False, prefix=prefix: self._quick_copy_folder(prefix))
            images_row.add_trailing_widget(copy_folder)
            stem_row = FormRow(label='Image Stem', path=('Calibration', 'Quick', prefix + 'stem'), value='')
            stem_row.label.setMinimumWidth(0)
            images_row.add_trailing_widget(stem_row, 1)
            form.addRow(images_row.label, images_row)
            for key, row, what in ((prefix + 'source', file_row, 'file'), (prefix + 'scan', scan_row, 'scan number'),
                                   (prefix + 'entry', entry_row, 'NeXus entry'),
                                   (prefix + 'images', images_row, "images folder (blank: the SPEC file's folder)"),
                                   (prefix + 'stem', stem_row, 'image name stem (blank: <SPEC name>_scan<N>_)')):
                self._quick_project(key, row, f"The {scan.lower()} scan's {what}: the same value as {tab}'s, edited "
                                              'here or there.')
            for key, row in ((prefix + 'source', file_row), (prefix + 'images', images_row)):
                row.browseRequested.connect(lambda _path, key=key: self._browse(key, self._choosers[key]))
                row.draftChanged.connect(self._sync_quick_folder_buttons)
                self._fields[key].textChanged.connect(self._sync_quick_folder_buttons)
        self.quick_threshold = self._threshold_row()  # Advanced's threshold: one state, shown on both rows
        form.addRow(self.quick_threshold.label, self.quick_threshold)
        self._show_threshold()
        self.quick_button = QtWidgets.QPushButton('Run quick calibration')
        self.quick_button.clicked.connect(self._quick_start)
        form.addRow(self.quick_button)
        self.quick_card.add_row(body)
        self._instrument_layout.insertWidget(self._instrument_layout.indexOf(self._detector_box) + 1, self.quick_card)
        self._sync_quick_rows()
        self._sync_quick_button()

    def _quick_project(self, key, row, tip):
        """The card's ``row`` for the window's field ``key``: one value, edited on the card or in its tab."""
        row.set_purpose_tooltip(tip)
        self._quick_rows[key] = row
        row.valueChanged.connect(lambda _path, value: self._quick_commit(key, value))
        self._fields[key].textChanged.connect(lambda _text: self._quick_mirror(key))
        self._formrows[key].valueChanged.connect(lambda _path, _value: self._quick_mirror(key))
        self._quick_mirror(key)

    def _quick_mirror(self, key):
        """The card's row shows the tab's value, a draft typed there included. It never overwrites a draft typed on the
        card, nor a focused file row (a file row takes no set while it holds the focus); a focused scan box that holds
        no draft does follow (Tab from a new file to its scan: the reset scan, not the old file's)."""
        row = self._quick_rows.get(key)
        if row is None:
            return
        row.editor.setPlaceholderText(self._fields[key].placeholderText())
        if row.editor.hasFocus() and (row.editor.isModified() or key.endswith(('source', 'images'))):
            return
        value = self._text(key)
        if row.current_value() != value:
            row.editor.setText(value)

    def _quick_commit(self, key, value):
        """A scan edited on the card is the tab's value, set there and committed as that row commits, so the tab's
        own handlers run (a new calibrant file resets its scan, images, stem and energy; source details refresh).
        An unchanged value writes nothing: re-setting a file would reset them too (review A1). Committed, the card's
        row holds no draft, so its modified flag (Quick's and the mirror's draft test) is cleared."""
        self._quick_rows[key].editor.setModified(False)
        value = '' if value is None else str(value).strip()
        if value == self._text(key):
            return
        field = self._fields[key]
        field.setText(value)
        field.editingFinished.emit()

    def _sync_quick_rows(self, *_args):
        """The card's rows follow the tabs' rows: no direct-beam scan unless Rough geometry measures one, no scan
        number for a NeXus file, and the tab's word for the kind of file."""
        if not self._quick_rows:
            return
        combo = self.quick_rough_mode.editor
        combo.setCurrentIndex(combo.findData(self.rough_mode.currentData()))
        measured = self.rough_mode.currentData() == 'scan'
        for prefix, role, kind, nexus_kind, shown in (
                ('beam_', 'Direct-beam', self.beam_source_kind.currentData(), 'nexus', measured),
                ('', 'Calibrant', self.source_kind.currentData(), 'nexus_stack', True)):
            nexus, series = kind == nexus_kind, kind in ('spec', 'spec_series')
            rows = self._quick_rows
            self._quick_form.setRowVisible(rows[prefix + 'source'], shown)
            self._quick_form.setRowVisible(rows[prefix + 'images'], shown and not nexus)
            rows[prefix + 'source'].label.setText(f"{role} {'NeXus' if nexus else 'SPEC'} file")
            rows[prefix + 'scan'].setVisible(not nexus)
            rows[prefix + 'entry'].setVisible(not series and self._shared_nexus_entry is None)
        self._sync_quick_folder_buttons()

    def _sync_quick_folder_buttons(self, *_args):
        for prefix, button in self._quick_folder_buttons.items():
            rows = self._quick_rows
            reason = ''
            if self._quick_stage is not None or (self._owner.busy and self._job not in ('beam_detect', 'source_details')):
                reason = 'Available when the current calibration operation ends.'
            elif self.rough_mode.currentData() != 'scan':
                reason = 'Select Direct-beam scan in Rough geometry to use both scan folders.'
            elif self.beam_source_kind.currentData() == 'nexus' or self.source_kind.currentData() == 'nexus_stack':
                reason = 'Both scans must use SPEC files with image folders.'
            elif not (str(rows[prefix + 'images'].current_value()).strip()
                      or str(rows[prefix + 'source'].current_value()).strip()):
                reason = "Select this scan's SPEC file or enter its Image Path first."
            button.apply_spec(replace(button.spec, enabled=not reason, reason=reason))

    def _quick_copy_folder(self, prefix):
        """One explicit folder copy through the existing commits; stems remain independent."""
        if not self._quick_folder_buttons[prefix].isEnabled():
            return
        for key in (prefix + 'source', prefix + 'images'):
            self._quick_commit(key, self._quick_rows[key].current_value())
        self._quick_commit(('' if prefix else 'beam_') + 'images', self._image_folder(prefix))

    def _image_folder(self, prefix=''):
        path = self._text(prefix + 'source')
        return self._text(prefix + 'images') or (str(Path(path).resolve().parent) if path else '')

    def _build_distance(self):
        layout, form = self._form('2 · Rough geometry', 'rough')
        self._beam_form = form
        self.rough_mode = self._combo(form, 'Rough geometry', [('Direct-beam scan', 'scan'),
                                                               ('Enter the distance', 'manual'), ('Skip', 'skip')],
            tip='A first guess of where the detector is, so pyFAI opens each frame with its rings roughly in '
                'place and the table lists which rings to expect. A direct-beam scan measures the distance; '
                'or type it. Rough values are enough: once one frame is calibrated, the others are predicted '
                'from it instead.')
        self._row_of[self.rough_mode].set_help_topic('geometry.orientation', button=True)
        self._edit(form, 'rough_distance', 'Sample–detector distance (mm)', '',
                   tip='Sample to detector, in millimetres, with the detector facing the beam squarely.')
        self.beam_source_kind = self._combo(form, 'Source', [('SPEC + image series', 'spec_series'),
                                                             ('SPEC + Eiger / NeXus', 'spec_container'),
                                                             ('NeXus images + metadata', 'nexus')],
            tip='Where the direct-beam scan\'s angles and images come from. ' + _FROM_DATA_FORMAT)
        self.beam_source_kind.setEnabled(False)  # the page's Data format sets it (LT-005)
        self._formrows['beam_source_kind'] = self._row_of[self.beam_source_kind]
        self._rows['beam_source_kind'] = (form, self._row_of[self.beam_source_kind])
        self._attach('beam_entry', 'beam_source_kind', 'NeXus entry', 'entry',
                     tip='The NeXus entry (group) holding the images.')
        self._edit(form, 'beam_source', 'SPEC file',
                   chooser=lambda: NEXUS_FILTER if self.beam_source_kind.currentData() == 'nexus' else 'spec',
                   tip='The direct-beam scan: its SPEC file (the angles), or its NeXus file.')
        self._attach('beam_scan', 'beam_source', 'Scan', '1', tip=self._SCAN_TIP, width=90)
        self._edit(form, 'beam_image_file', 'Eiger / NeXus images',
                   chooser='HDF5 / NeXus (*.h5 *.hdf5 *.nxs *.nexus);;All files (*)',
                   tip='The Eiger master or NeXus file holding this scan\'s images.')
        self._edit(form, 'beam_images', 'Images folder', chooser='directory', tip=self._FOLDER_TIP)
        self._attach('beam_stem', 'beam_images', 'Stem', '', tip=self._STEM_TIP)
        self.beam_source_kind.currentIndexChanged.connect(self._sync_rough_rows)
        for key in ('beam_source', 'beam_scan'):
            self._fields[key].textChanged.connect(lambda _text: self._auto_placeholders('beam_'))
        self._auto_placeholders('beam_')
        for key in ('beam_source', 'beam_scan', 'beam_entry', 'beam_images', 'beam_stem'):
            self._fields[key].editingFinished.connect(lambda: self._source_details('beam_'))
        self._edit(form, 'beam_pixel', 'Direct beam pixel (x, y)', '',
                   tip='Column, row of the pixel the direct beam hits with the detector facing it squarely (each '
                       'circle at its motor zero, beside its motor under 1 · Instrument), in the image as '
                       'displayed; it is not necessarily the centre. '
                       'Blank with a direct-beam scan: found on the scan\'s frame nearest the zero (the '
                       'brightest spot, a lone hot pixel ignored). Typed: along the beam\'s motion it can be '
                       'rough; across it (the row for horizontal motion, the column for vertical) it must be '
                       'within the strip half width, because the spot is searched only in that strip.')
        self._attach('rough_mount', 'beam_pixel', 'Detector mount tilt, roll (°)', '0, 0', width=180,
                   tip='The detector mount\'s tilt and roll, held fixed (the arm\'s mount extension; a '
                       'single 2θ arm has none). 0, 0: the detector faces the beam squarely at every angle.')
        self._fit_options, options = self._card(layout, 'Fit options')
        self._edit(options, 'beam_strip', 'Strip half width (pixels)', '2',
                   tip='The beam spot is searched in a strip this many pixels either side of its path.')
        self._edit(options, 'beam_roi', 'ROI x0, y0, x1, y1 (optional)', '',
                   tip='Search only this pixel box (columns x0–x1, rows y0–y1).')
        self.beam_readiness = SectionCard('', collapsible=False, compact=True)
        layout.addWidget(self.beam_readiness)
        self.beam_button = QtWidgets.QPushButton('Fit direct-beam distance')
        self.beam_button.setToolTip('Finds the beam spot in each image and fits the distance from how far '
                                    'it moves per degree.')
        self.beam_button.clicked.connect(self._distance)
        layout.addWidget(self.beam_button)
        self.beam_summary = ReadOnlyTextPane()
        self.beam_summary.setMaximumHeight(self.beam_summary.fontMetrics().lineSpacing() * 9)
        layout.addWidget(self.beam_summary)
        self.show_fit_button = QtWidgets.QPushButton('Show the fit…')
        self.show_fit_button.setToolTip('The beam centre against the motor angle, the fitted model and its '
                                        'residuals, with the distance (it opens after every fit).')
        self.show_fit_button.clicked.connect(self._show_direct_beam_fit)
        self.show_fit_button.setEnabled(False)
        self.apply_distance_button = QtWidgets.QPushButton('Use result as distance seed')
        self.apply_distance_button.setToolTip('Starts the final fit from this distance: it becomes 4 · Calibrate and '
                                              'fit ▸ Advanced ▸ Starting distance.')
        self.apply_distance_button.clicked.connect(self._apply_distance)
        # A calibration from this geometry alone (maintainer, 2026-09-29: "a quick multigeometry
        # calibration json file from just the sample detector distance, and direct beam pixel")
        self.save_rough_button = QtWidgets.QPushButton('Save as calibration…')
        self.save_rough_button.setToolTip(
            f'Saves this geometry, with no ring fit, as {self.context.page}\'s calibration: the distance and beam '
            'pixel '
            'above, the rotations held as typed, one degree per degree. As Save does, it writes the pyFAI '
            'file and, beside it, the xrayutilities file, checked on the direct-beam scan\'s angles, the '
            'calibrant scan\'s angles. With no scan, it checks a small grid around zero. Useful when '
            'nu stays at 0; the file records that it is a direct-beam calibration.')
        self.save_rough_button.clicked.connect(self._save_rough)
        actions = QtWidgets.QHBoxLayout()
        for button in (self.show_fit_button, self.apply_distance_button, self.save_rough_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        layout.addStretch(1)
        self.rough_mode.currentIndexChanged.connect(self._sync_rough_rows)
        for key in ('rough_distance', 'beam_pixel', 'rough_mount'):
            self._fields[key].editingFinished.connect(self._render_frames)
        self._fields['rough_mount'].textChanged.connect(self._invalidate_fit)  # the fits' mount estimate
        self._fields['rough_mount'].textChanged.connect(self._bounds_edited)
        for key in ('beam_detector', 'beam_orientation'):
            self._fields[key].currentIndexChanged.connect(self._render_frames)
        self._fields['beam_detector'].currentIndexChanged.connect(self._detector_chosen)
        self._fields['beam_detector'].currentIndexChanged.connect(self._update_image_size)
        self.camera.changed.connect(self._update_image_size)  # the beam pixel's image turns
        self.camera.changed.connect(self._beam_camera_changed)
        self.camera.changed.connect(self._camera_edited)
        self.camera.changed.connect(self._setup_edited)
        self._update_image_size()

    def _build_detector(self):
        """The detector every step reads the images with: its model, the stored image's layout and
        turn, its saturation and one mask for the whole calibration (maintainer, 2026-09-29)."""
        panel = self._detector_panel = QtWidgets.QGroupBox('Detector')
        form = QtWidgets.QFormLayout(panel)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        # one primary row, Model · Rotation · Config, then the Mask file; the rest in Config, collapsed at first
        # (LT-009; maintainer, 2026-10-14: "Inline expansion")
        from xdart.gui.widgets.controls_panel import SubsectionCard
        self._model_captions = dict((name, label) for label, name in detector_choices(shapes=True))
        self._model_labels = dict((name, label) for label, name in detector_choices())
        self._choice(form, 'beam_detector', 'Model', list(detector_choices(shapes=True)), 'Pilatus300kw',
                     tip="The detector model (pyFAI), its native rows × columns and, when Config's RAW layout is "
                         'Manual, the pixel type typed there; it sets the pixel size and shape. The detector default '
                         'reads headerless RAW frames as int32; TIFF images and Eiger or NeXus files carry their own '
                         'type.')
        from xdart.gui.widgets.camera_editor import CameraEditor
        self.camera = CameraEditor(rotation_apart=True)
        config = self._detector_config = SubsectionCard('')
        config_button = config.external_trigger(
            'Config', 'Show or hide the RAW layout and header, the camera flips and reset, the first-pixel corner and '
                      'the saturation. Showing or hiding changes no value and keeps any draft.')
        self._formrows['beam_detector'].add_trailing_widget(self.camera.rotation_row, 0)
        self._formrows['beam_detector'].add_trailing_widget(config_button, 0)
        self._edit(form, 'mask', 'Mask file', chooser='Masks (*.edf *.tif *.tiff *.npy);;All files (*)',
                   tip='Pixels to leave out everywhere in this calibration: the direct-beam fit, pyFAI-calib2 '
                       '(its mask), the automatic refinement\'s ring search and the expected rings. Drawn on '
                       f'the stored image (before the camera\'s turn), as {self.context.page}\'s Mask File is; saving '
                       f'the calibration makes it {self.context.page}\'s Mask File.')
        self._fields['mask'].editingFinished.connect(self._render_frames)
        form.addRow(config)
        body = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(body)  # Config's own rows
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        config.add_row(body)
        self.raw_layout = self._combo(form, 'RAW layout', [('Detector default', 'factory'), ('Manual', 'manual')],
            tip='Detector default: its rows and columns, with the RAW reader default int32. '
                'Manual: override the stored RAW layout; the physical detector model and pixel pitch stay selected. '
                'TIFF, Eiger and NeXus carry their own layout.')
        self._formrows['raw_layout'] = self._row_of[self.raw_layout]
        self._rows['raw_layout'] = (form, self._row_of[self.raw_layout])
        self._attach('header', 'raw_layout', 'Header (bytes)', '0', width=75,
                     tip='Bytes to skip before pixels in a headerless RAW file; unused for TIFF, Eiger and NeXus.')
        reader = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(reader)
        row.setContentsMargins(0, 0, 0, 0)
        raw_tip = ('Only for headerless .raw images (the calibrant and direct-beam scans): the frame\'s rows '
                   'and columns (set from the model), pixel type and header size.')
        for key, label, default, stretch in [('shape', 'rows, cols', '195, 1475', 3), ('dtype', 'type', 'int32', 2)]:
            part = FormRow(label=label, path=('Calibration', key), value=default)
            part.label.setMinimumWidth(0)
            edit = part.editor
            edit.setMinimumWidth(edit.fontMetrics().horizontalAdvance(default + '0'))
            self._fields[key] = edit
            self._formrows[key] = part
            self._tip(raw_tip, part.label, edit)
            row.addWidget(part, stretch)
        form.addRow('RAW images', reader)
        self._rows['raw'] = (form, reader)
        self._tip(raw_tip, form.labelForField(reader))
        self.raw_layout.currentIndexChanged.connect(self._sync_raw_layout)
        self._sync_raw_layout()
        # The shared mounting editor keeps signed lab axes internally (maintainer, 2026-10-02); its rotation sits on
        # the Model row, its flips, reset and image size here
        form.addRow('Camera orientation (lab frame)', self.camera)
        self._tip(self._CAMERA_TIP, self.camera.rotation, self.camera.rotation_label, self.camera.fliplr,
                  self.camera.flipud, form.labelForField(self.camera))
        self._image_size = self.camera.image_size
        corners = {3: 'bottom right (pyFAI default)', 1: 'top left', 2: 'top right', 4: 'bottom left'}
        self._choice(form, 'beam_orientation', 'First pixel (pyFAI orientation)',
                     [(f'{code} · first pixel {corners[code]}', code) for code, _name in DETECTOR_ORIENTATIONS], 3,
                     tip='pyFAI\'s detector orientation: the corner holding the image\'s first pixel, seen '
                         'from behind the detector looking at the sample. It is not a rotation: it never turns '
                         'the image or swaps x and y (the camera above does). It changes how the PONIs '
                         'are written and how pixel indices map to physical positions, including χ. '
                         'The arm\'s motor scales take its signs. Keep 3, '
                         'pyFAI\'s default, unless pyFAI shows this detector\'s images mirrored. It is saved '
                         'with every PONI and fit.')
        for combo in (self._fields['beam_detector'], self._fields['beam_orientation']):
            # long entries elide instead of widening the window
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        self._edit(form, 'saturation', 'Saturation (counts)', '',
                   tip='Counts at or above this are saturated: a direct-beam frame with such a pixel in its '
                       'beam strip is left out, and the automatic refinement picks no ring points above it. '
                       'Blank keeps Auto: the detector\'s documented counter limit when known, otherwise '
                       'the image type\'s maximum. Eiger\'s physical cutoff depends on acquisition settings; '
                       'type the file\'s cutoff to use it. A typed value stays when the detector changes.')
        self._show_saturation_hint()
        for signal in (self._fields['beam_detector'].currentIndexChanged, self.raw_layout.currentIndexChanged,
                       self._fields['dtype'].textChanged):
            signal.connect(self._caption_model)
        self._caption_model()
        return panel

    def _caption_model(self, *_args):
        """The Model box names each detector with its native rows × columns; the chosen one adds the pixel type when
        it is known, typed under a Manual RAW layout while a scan reads an image series (LT-009: no "RAW" and no
        invented type; the detector default's int32 is the RAW reader's default, and TIFF, Eiger and NeXus files carry
        their own)."""
        combo = self._fields['beam_detector']
        kinds = (getattr(self, 'source_kind', None), getattr(self, 'beam_source_kind', None))  # None while built
        series = None in kinds or kinds[0].currentData() == 'spec' or kinds[1].currentData() == 'spec_series'
        dtype = self._text('dtype') if series and self.raw_layout.currentData() == 'manual' else ''
        for index in range(combo.count()):
            caption = self._model_captions.get(combo.itemData(index), combo.itemText(index))
            caption += f' · {dtype}' if dtype and index == combo.currentIndex() else ''
            if combo.itemText(index) != caption:
                combo.setItemText(index, caption)

    _ROUGH_NOTES = {
        'scan': 'A direct-beam scan: SPEC supplies the angles for an image series or an Eiger/NeXus container; '
                'standalone NeXus supplies both. Angle and image counts must match. Uses the Detector panel\'s '
                'camera, RAW settings, saturation and mask (1 · Instrument). Its distance, with the beam '
                'pixel and zero, predicts each frame\'s rings; the regression uncertainty excludes motor, zero and '
                'pitch systematics.',
        'manual': 'The distance, the pixel the direct beam hits at normal incidence and that zero predict each '
                  'frame\'s rings (the Diffractometer\'s arm, one degree per degree) until the first frame is '
                  'calibrated; that frame then predicts the others, and the fit refines them all.',
        'skip': 'No rough geometry: each frame\'s rings are predicted from a geometry file (4 · Calibrate and fit '
                '▸ Refine from the whole scan), or from the first frame you calibrate.',
    }
    #: Hidden by Skip. Calibrant scan owns the energy field for frames without recorded energy.
    _GEOMETRY_ROWS = ('beam_pixel', 'rough_mount')

    _CAMERA_TIP = ('Detector mounting looking downstream, with every circle at zero. Rotation is clockwise '
                   'from upright, then Flip ↔ (left/right) reverses stored columns and Flip ↕ (up/down) reverses stored rows. '
                   'Both flips equal a 180° turn; the controls cover eight distinct orientations. Upright is '
                   'stored rows x−, columns z+, as in the upright Pilatus 300K-W and Eiger2 500K datasets. '
                   'The lab axes are those in the Diffractometer section: x up, z to the left looking '
                   'downstream. Every image is converted once into pyFAI\'s frame, rows x+, columns z−. '
                   'Reset chooses upright; it does not determine an unknown detector\'s physical mounting.')

    def _orientation(self):
        """The camera's transform into Stitch's frame: a turn, then a left-right mirror."""
        from xrd_tools.core.geometry.camera import stitch_orientation
        return stitch_orientation(self.camera.value())

    def _turned_detector(self):
        """The Detector panel's model as the camera turns it by a quarter into the window's frame
        (``TURNED_DETECTOR_KEY``'s value), or None: a camera keeping the axes, or a detector pyFAI does not
        list (AUD-9 stage 2b item 8)."""
        from xrd_tools.integrate.turned_detector import turned_detector_record
        orientation, model = self._orientation(), self._text('beam_detector')
        if not orientation.swaps_axes or detector_shape(model) is None:
            return None
        return turned_detector_record(model, orientation)

    def _apply_camera(self, values):
        """The saved camera, else an older window's image rotation (its unmirrored camera)."""
        from xrd_tools.core.geometry import ImageOrientation
        from xrd_tools.core.geometry.camera import DetectorCamera, stitch_camera
        camera = values.get('camera')
        rotation = ((values.get('choices') or {}).get('rotation') or {}).get('data')
        try:
            if isinstance(camera, list) and len(camera) == 2:
                self.camera.set_value(DetectorCamera(*map(str, camera)))
            elif rotation in (0, 90, 180, 270):
                self.camera.set_value(stitch_camera(ImageOrientation(rotation=rotation)))
        except (ValueError, TypeError):
            pass  # an unknown camera: the default stays

    def _update_image_size(self, *_args):
        """The detector image's width × height, as the beam pixel is read (turned by the Detector panel's camera); a
        quarter turn also says the stored image's size and the turn."""
        model = self._model_labels.get(self._text('beam_detector'), self._text('beam_detector'))
        shape = detector_shape(self._text('beam_detector'))
        if shape is None:
            self._detector_panel.setTitle(f'Detector · {model} · orientation: '
                                          f'{self.camera.transform_description()}')
            self._image_size.setText('unknown detector')
            self._fields['beam_pixel'].setPlaceholderText('column, row')
            return
        rows, columns = shape
        orientation = self._orientation()
        width, height = (rows, columns) if orientation.swaps_axes else (columns, rows)
        self._detector_panel.setTitle(f'Detector · {model} · {width} × {height} px (W × H) · '
                                      f'orientation: {self.camera.transform_description()}')
        # a quarter turn: the window's image is the stored one turned, and so is its detector (item 8)
        stored = (f' in the window (the stored image is {columns} × {rows}, turned {orientation.rotation}°'
                  f'{" and mirrored left-right" if orientation.flip_horizontal else ""})'
                  if orientation.swaps_axes else '')
        self._image_size.setText(f'{width} × {height} px (W × H)')
        self._image_size.setToolTip(f'Image size: Width × Height in pixels{stored}. '
                                   'The first-pixel corner below does not change the image size.')
        self._beam_pixel_hint()

    def _beam_pixel_hint(self, *_args):
        """A direct-beam scan finds the beam pixel when it is blank; a typed distance needs it."""
        if not hasattr(self, 'rough_mode'):  # still being built
            return
        found = getattr(self._distance_result, 'beam_pixel', None) if self._distance_fresh() else None
        self._fields['beam_pixel'].setPlaceholderText(
            'column, row where the direct beam hits at the zero' if self.rough_mode.currentData() != 'scan'
            else f'auto · the direct beam found at {found[0]:.1f}, {found[1]:.1f}' if found is not None
            else 'auto · where the direct beam hits, found on the frame nearest the zero')

    def _detector_chosen(self, *_args):
        """The chosen detector's shape becomes the RAW reader's rows and columns."""
        model = self._text('beam_detector')
        if not self._restoring_fields and model != getattr(self, '_pixel_detector', model):
            self._set_field_text('beam_pixel', '')  # a pixel belongs to one detector, not another model
        self._pixel_detector = model
        if not self._restoring_fields:
            self.raw_layout.setCurrentIndex(self.raw_layout.findData('factory'))
        self._sync_raw_layout()
        self._show_saturation_hint()

    def _sync_raw_layout(self, *_args):
        manual = self.raw_layout.currentData() == 'manual'
        if not manual and not self._restoring_fields:
            shape = detector_shape(self._text('beam_detector'))
            if shape is not None:
                self._set_field_text('shape', f'{shape[0]}, {shape[1]}')
            self._set_field_text('dtype', 'int32')
        for key in ('shape', 'dtype'):
            self._fields[key].setReadOnly(not manual)
        self._show('raw', manual)

    def _custom_raw_layout(self):
        try:
            shape = tuple(int(v) for v in self._text('shape').split(','))
            return shape != detector_shape(self._text('beam_detector')) or self._text('dtype') != 'int32'
        except ValueError:
            return True  # preserve a draft; the action validates the reader

    def _show_saturation_hint(self):
        """The shared documented limit without disk reads; the grey hint keeps Auto, like Int."""
        from xrd_tools.core.invalid import resolve_detector_saturation
        ceiling = resolve_detector_saturation(detector=self._text('beam_detector'))
        self._fields['saturation'].setPlaceholderText(
            f'Auto · {ceiling.value} (detector)' if ceiling is not None else
            'Auto · image type maximum; acquisition cutoff varies')
        self._show_threshold()

    def _saturation_value(self):
        """An explicit cutoff, else the existing detector limit; unknown models keep dtype fallback."""
        from xrd_tools.core.invalid import resolve_detector_saturation, saturation_count
        if self._text('saturation'):
            value = saturation_count(float(self._text('saturation')))
            if value is None:
                raise ValueError('saturation must be a positive whole count')
            return value
        ceiling = resolve_detector_saturation(detector=self._text('beam_detector'))
        return None if ceiling is None else ceiling.value

    def _beam_camera_changed(self):
        orientation, old = self._orientation(), getattr(self, '_pixel_orientation', self._orientation())
        shape = detector_shape(self._text('beam_detector'))
        if not self._restoring_fields and shape and self._text('beam_pixel'):
            try:
                pixel = tuple(map(float, self._text('beam_pixel').split(',')))
                pixel = orientation.apply_pixel(old.unapply_pixel(pixel, shape), shape)
                self._set_field_text('beam_pixel', ', '.join(f'{value:.12g}' for value in pixel))
            except ValueError:
                self._set_field_text('beam_pixel', '')
        self._pixel_orientation = orientation

    def _sync_rough_rows(self, *_args):
        """Every Rough geometry row's visibility, from the rough mode and the direct-beam source together."""
        mode, kind = self.rough_mode.currentData(), self.beam_source_kind.currentData()
        scan = mode == 'scan'
        visible = {'rough_distance': mode == 'manual', 'beam_source': scan,
                   'beam_scan': scan and kind != 'nexus', 'beam_image_file': False,
                   'beam_images': scan and kind != 'nexus', 'beam_stem': scan and kind != 'nexus',
                   'beam_entry': scan and kind != 'spec_series' and self._shared_nexus_entry is None,
                   **dict.fromkeys(self._GEOMETRY_ROWS, mode != 'skip')}
        form = self._beam_form
        for key, shown in visible.items():
            self._show(key, shown)
        form.setRowVisible(self._row_of[self.beam_source_kind], scan)
        self.beam_readiness.setVisible(scan)
        for widget in (self._fit_options, self.beam_button, self.beam_summary, self.show_fit_button,
                       self.apply_distance_button):
            widget.setVisible(scan)
        self.save_rough_button.setVisible(mode != 'skip')
        self._label('beam_source').setText('NeXus file' if kind == 'nexus' else 'SPEC file')
        self.rough_mode.setToolTip(self._ROUGH_NOTES[mode])
        self._beam_pixel_hint()
        if hasattr(self, 'frame_table'):
            self._render_frames()
        self._sync_quick_rows()

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
        # a fitted scan saw the beam with its own circle at its zero and the other held where the scan
        # read it (17-2's scan 3: del at 0, its zero 2.3°); a typed pixel is at both zeros
        readings = dict(self._distance_held) if mode == 'scan' else None
        return rough_arm_record(
            distance_m=distance, detector=self._text('beam_detector'),
            beam_px=beam, zeros_deg=self._zeros(),
            del_motor=delta, nu_motor=nu,
            orientation=int(self._text('beam_orientation')), mount_deg=mount,
            wavelength_m=wavelength * 1e-10 if wavelength is not None else 1e-10,
            beam_readings=readings or None, senses=geometry.arm_senses(),
            camera=self._orientation())  # a quarter turn: the detector turned with the images (item 8)

    def _rough_wavelength_A(self):
        """Wavelength from the Energy field, else the calibrant scan's when every frame records the same
        energy, else, when it records none (or an impossible one), the calibrated frames' PONI files' when they
        agree (maintainer, 2026-10-09: "if there is no typed in energy but there is a poni file, it should use
        the energy from the poni file"), else the direct-beam scan's (maintainer, 2026-09-29: "why doesn't it
        get it from the spec file?") while its fit is current, else None."""
        from xrd_tools.core.energy import energy_eV_to_wavelength_m
        if self._text('rough_energy'):
            energy = float(self._text('rough_energy')) * 1000
            if not math.isfinite(energy) or energy <= 0:
                raise ValueError('Energy must be finite and positive, in keV')
            return float(energy_eV_to_wavelength_m(energy)) * 1e10
        if self._source_energy_eV is not None:
            return float(energy_eV_to_wavelength_m(self._source_energy_eV)) * 1e10
        energies = {energy for _index, _values, energy in self._choices if energy is not None}
        if len(energies) == 1:
            return float(energy_eV_to_wavelength_m(energies.pop())) * 1e10
        wavelength = None if energies else self._frames_poni_wavelength_m()
        if wavelength is not None:
            return wavelength * 1e10
        if (not energies and self.rough_mode.currentData() == 'scan' and self._distance_fresh()
                and self._distance_energy_eV is not None):
            return float(energy_eV_to_wavelength_m(self._distance_energy_eV)) * 1e10
        return None

    def _beam_role(self):
        motor = self._beam_detection[1] if self._beam_detection is not None else ''
        return self.geometry_editor.value().arm_role(motor)

    def set_data_format(self, data_format):
        """The page's one Data format (LT-SMALL-FIXES-20261014): the calibrant and direct-beam source kinds follow
        it, each through its own rows, file chooser and Quick rows; their inputs stay their own. False, changing
        nothing, while a job runs (the page's inputs are locked then as well)."""
        if self._owner.busy:
            return False
        from xrd_tools.sources.selection import calibration_source_kinds
        self._data_format = data_format
        for combo, kind in zip((self.source_kind, self.beam_source_kind), calibration_source_kinds(data_format)):
            index = combo.findData(kind)
            if index >= 0 and index != combo.currentIndex():
                combo.setCurrentIndex(index)  # its own handlers: rows, energy and detection invalidation
        return True

    def set_nexus_entry(self, value):
        """Project's optional shared override; Auto preserves imported role defaults until explicitly changed."""
        if self._owner.busy:
            return False
        value = str(value or '').strip()
        previous = self._shared_nexus_entry
        self._shared_nexus_entry = value
        changed = []
        if value or previous:
            for key in ('entry', 'beam_entry'):
                target = value or 'entry'
                if self._text(key) != target:
                    self._set_field_text(key, target)
                    changed.append(key)
        for key in changed:  # commit both roles before either metadata request starts
            self._fields[key].editingFinished.emit()
        self._sync_source_rows()
        self._sync_rough_rows()
        return True

    def set_source_context(self, source: SourceSpec, *, image_rotation=0, del_motor=None, nu_motor=None,
                           geometry_path=None, image_flip=False):
        if self._owner.busy:
            return
        if geometry_path:
            self._fields['refine_seed_path'].setText(str(geometry_path))
            if self.fit is None and self._seed_file_problem(geometry_path) is None:  # an arm it can start from
                self.refine_seed.setCurrentIndex(self.refine_seed.findData('file'))
        if self._data_format is None:  # a page's Data format, once set, decides the kind (set_data_format)
            self.source_kind.setCurrentIndex(self.source_kind.findData(
                'spec' if source.kind is SourceKind.SPEC else 'nexus_stack'))
        opts = source.options
        for key, value in [('source', source.uri), ('scan', opts.get('scan', '1')),
                           ('images', opts.get('image_dir', '')), ('stem', opts.get('image_stem', '')),
                           ('entry', self._shared_nexus_entry or source.entry or 'entry'),
                           ('del_motor', del_motor), ('nu_motor', nu_motor)]:
            if value is not None:  # psic's del/nu motors, when given (set_instrument names them all)
                self._fields[key].setText(str(value))
        reader = opts.get('read_image_kwargs', {})
        for key, value in [('shape', ', '.join(map(str, reader.get('detector_shape',
                                     detector_shape(self._text('beam_detector')) or (195, 1475))))),
                           ('dtype', reader.get('raw_dtype', 'int32')), ('header', reader.get('raw_header_skip', 0))]:
            self._fields[key].setText(str(value))
        self.raw_layout.setCurrentIndex(self.raw_layout.findData('manual' if self._custom_raw_layout() else 'factory'))
        self._source_details()
        if image_rotation is None:  # the page turns no images of its own: the window's camera stays (item 10)
            return
        from xrd_tools.core.geometry import ImageOrientation
        from xrd_tools.core.geometry.camera import stitch_camera
        self.camera.set_value(stitch_camera(ImageOrientation(rotation=image_rotation, flip_horizontal=image_flip)))

    #: Every field the direct-beam fit's identity (``_direct_beam_key``: its source, reader and extraction options)
    #: and step 1's arm (``_rough_arm``) read; each refreshes the motion advice (Codex B1 re-review: a hand-picked
    #: list missed the beam pixel, strip and mount). test_direct_beam_readiness records the reads against it.
    _DIRECT_BEAM_ADVICE_FIELDS = (
        'beam_source', 'beam_entry', 'beam_scan', 'beam_image_file', 'beam_images', 'beam_stem', 'shape', 'dtype',
        'header', 'beam_detector', 'beam_orientation', 'beam_pixel', 'beam_strip', 'beam_roi', 'mask', 'saturation',
        'rough_distance', 'rough_mount', 'rough_energy')

    _STALE_DISTANCE = ('The direct-beam fit was made with other inputs (a motor zero, its motor, the source, the '
                       'detector or the search settings changed since): fit the direct-beam distance again.')
    _NO_ENERGY = 'type Energy (keV) in 3 · Calibrant scan: the rings are predicted with it'

    #: Quick calibration's steps: their names and the tabs that hold each step's controls
    _QUICK_STEPS = {1: ('1 · Direct-beam fit', 'rough'), 2: ('2 · Calibrant metadata', 'select'),
                    3: ('3 · Refine', 'calibrate')}
    _QUICK_NEEDS_SCAN = 'Quick calibration fits a direct-beam scan: choose "Direct-beam scan" in 2 · Rough geometry.'
    _QUICK_METADATA_STALE = "Reading the calibrant scan's metadata changed the instrument. "
    _QUICK_CANCELLED = 'Cancelled; the steps completed before it are kept.'
    _QUICK_NO_FRAMES = 'the calibrant scan lists no frames (3 · Calibrant scan).'
    _QUICK_TIP = ('Fits the direct-beam scan, reads the calibrant scan\'s metadata, then refines automatically from the '
                  'rough geometry, with these scans and every other setting as the tabs have them. It stops at the '
                  'first problem, never saves, and changes no camera, motor sense or zero.')

    #: The freedom every fit keeps about its estimates (maintainer, 2026-09-29), with defaults.
    _BOUND_DEFAULTS = {'zero_freedom': '2', 'mount_freedom': '2', 'displacement_limit': '0.05',
                       'distance_freedom': '5'}
    _BOUND_FIELDS = (
        ('zero_freedom', 'Motor zeros (± °)',
         "Each moving circle's motor zero is fitted within this many degrees of where it starts: the zero "
         "typed beside its motor under 1 · Instrument ▸ Diffractometer (none typed: 0°, or, when refining "
         "from a geometry, that geometry's). A single arm's held angle likewise."),
        ('mount_freedom', 'Detector tilt and roll (± °)',
         "The detector's tilt and roll on the arm are fitted within this many degrees of the values under "
         "2 · Rough geometry (Detector mount tilt, roll)."),
        ('displacement_limit', 'Sample offset from the rotation centre (± mm)',
         "Each component of the sample's offset from the rotation centre is fitted within this of zero: "
         "0.05 mm by default, 0.02 mm for a sample known to be well centred. An offset stopped at this "
         "limit is reported, never widened."),
        ('distance_freedom', 'Distance (± %)',
         'Without a Distance range, the distance is fitted within this percentage of where it starts.'),
    )

    #: Choices saved with the page's settings (Stitch's, RSM's). The calibrant-scan and same-detector
    #: confirmations are per use and never saved; the two scans' source kinds are the page's Data format's.
    _SAVED_CHOICES = ('refine_calibrant', 'refine_seed', 'rough_mode', 'raw_layout')

    def settings(self):
        """This dialog's inputs as plain values, for the page's saved settings."""
        choices = {name: {'data': _plain_choice(getattr(self, name).currentData()),
                          'text': getattr(self, name).currentText()}
                   for name in self._SAVED_CHOICES}
        return {
            'fields': {key: self._text(key) for key in self._fields},
            'choices': choices,
            'fit_motor_scales': self.scales.isChecked(),
            'refine_energy': self.refine_energy.isChecked(),
            'instrument_geometry': self.geometry_editor.value().to_record(),
            'camera': list(self.camera.value().to_xu()),
            'calibration_threshold': asdict(self._threshold_state),
            'point_images': self.point_images.document(),
            'loaded_calibration': self._loaded_calibration_identity,
        }

    def set_instrument(self, geometry: InstrumentGeometry):
        """The diffractometer and motor names (the page's own)."""
        self.geometry_editor.set_value(geometry)

    def apply_settings(self, values):
        """Restore saved inputs; unknown entries are ignored."""
        if not isinstance(values, dict):
            return
        loaded = values.get('loaded_calibration')
        self._loaded_calibration_identity = (tuple(loaded) if isinstance(loaded, (list, tuple))
                                            and len(loaded) == 3 and all(isinstance(v, str) for v in loaded)
                                            else None)
        if 'point_images' in values:
            self.point_images.restore(values['point_images'])
        try:
            self.geometry_editor.set_value(InstrumentGeometry.from_record(values.get('instrument_geometry')))
        except (ValueError, TypeError):
            pass  # an older settings file: the del/nu fields below still name the arm
        fields = dict(values.get('fields') or {})
        if self._shared_nexus_entry:
            fields.update(entry=self._shared_nexus_entry, beam_entry=self._shared_nexus_entry)
        from xrd_tools.core.energy import wavelength_m_to_energy_eV
        if fields.get('rough_wavelength') and not fields.get('rough_energy'):
            try:
                fields['rough_energy'] = f"{float(wavelength_m_to_energy_eV(float(fields['rough_wavelength']) * 1e-10)) / 1000:.4f}"
            except (TypeError, ValueError):
                pass
        if fields.get('beam_image_file') and not fields.get('beam_images') and not fields.get('beam_stem'):
            old = Path(fields['beam_image_file'])
            fields.update(beam_images=str(old.parent), beam_stem=old.stem, beam_image_file='')
        self._restoring_fields = True
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
        self._apply_camera(values)
        self._threshold_state = ThresholdState()  # no saved choice: on; a saved one, on or off, stays
        if isinstance(values.get('calibration_threshold'), dict):
            try:
                state = ThresholdState(**values['calibration_threshold'])
                if type(state.apply_threshold) is not bool or type(state.mask_saturation) is not bool:
                    raise ValueError('invalid calibration threshold flags')
                if state.apply_threshold or state.threshold_max is not None:
                    StitchConditioning('manual', state.threshold_min, state.threshold_max)
                self._threshold_state = state
            except (TypeError, ValueError):
                pass
        self._show_threshold()
        self._restoring_fields = False
        if 'raw_layout' not in (values.get('choices') or {}) and self._custom_raw_layout():
            self.raw_layout.setCurrentIndex(self.raw_layout.findData('manual'))  # legacy reader overrides
        self._sync_raw_layout()
        # after the del/nu fields an older file names its arm with: its normal-incidence zero
        self.geometry_editor.set_value(migrated_instrument(self.geometry_editor.value(), values))
        self._sync_arm_model()
        self._pending_source_details = ''
        self._beam_probe_timer.start()

    def load_calibration(self, path):
        """Link the selected committed geometry to these shared controls, preserving same-file drafts.

        Staged frames and their saved session remain available; changing geometry
        invalidates derived fits rather than presenting the imported file as a fit.
        """
        from xrd_tools.integrate.calibration_seed import calibration_file_identity, calibration_seed_inputs
        if self.active() or not path:
            return
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        try:
            identity = calibration_file_identity(path)
            if identity == self._loaded_calibration_identity:
                if self._saved_setup is None:
                    seed = calibration_seed_inputs(path, fallback_instrument=self.geometry_editor.value())
                    self._saved_setup = (seed['instrument'], seed['camera'], seed['detector'], str(seed['orientation']))
                    self._loaded_calibration_kind = seed['calibration_kind']
                return
            seed = calibration_seed_inputs(path, fallback_instrument=self.geometry_editor.value())
        except (OSError, ValueError, TypeError, KeyError, NotImplementedError) as error:
            self.status.setText(f'Cannot load calibration {path.name}: {error}')
            return
        self._invalidate_fit()
        self._distance_result = self._distance_key = self._distance_warning = None
        self._distance_energy_eV, self._distance_held = None, {}
        self.beam_summary.clear()
        self.show_fit_button.setEnabled(False)
        if self._direct_beam_fit_window is not None:
            self._direct_beam_fit_window.close()
        values = self.settings()
        values['fields'].update(beam_detector=seed['detector'], beam_orientation=str(seed['orientation']),
            rough_distance=f"{seed['distance_m'] * 1000:.12g}",
            beam_pixel=', '.join(f'{v:.12g}' for v in seed['beam_px']),
            rough_mount=', '.join(f'{v:.12g}' for v in seed['mount_deg']),
            rough_energy=f"{seed['energy_eV'] / 1000:.12g}",
            refine_seed_path=identity[0])
        values.update(instrument_geometry=seed['instrument'].to_record(), camera=list(seed['camera'].to_xu()),
                      loaded_calibration=identity)
        values['choices'].update(rough_mode={'data': 'manual'}, refine_seed={'data': 'file'})
        self.apply_settings(values)
        self._prediction = None
        self._ring_cache.clear()
        self._saved_setup = self._setup_state()
        self._loaded_calibration_kind = seed['calibration_kind']
        self.setWindowTitle(f'{self.context.page} calibration · {path.name}')
        self.setToolTip(f'Loaded calibration: {identity[0]}. The file is the refinement seed; edits are inputs for a new fit.')
        self.status.setText(f'Loaded calibration: {path.name}. Calibrant source and fitting preferences retained.')

    def _zeros(self):
        """The Diffractometer panel's motor zeros by arm circle ('del', 'nu'); a malformed box is
        refused by its circle's name, never read as blank (Codex review F4)."""
        malformed = self.geometry_editor.invalid_zeros()
        if malformed:
            raise ValueError(f"type {malformed[0]}'s motor zero as a number of degrees, or leave it blank")
        return self.geometry_editor.value().arm_zeros()

    def _bounds(self):
        """The fits' estimates and freedom: each moving circle's zero from the Diffractometer panel
        (blank: 0, or the seed's when refining), the mount from 2 · Rough geometry, the freedom under
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
            raise ValueError('type the Starting distance in mm, or leave it blank') from None
        try:
            bounds = (tuple(float(v) / 1000 for v in self._text('distance_bounds').split(','))
                      if self._text('distance_bounds') else None)
        except ValueError:
            bounds = ()
        if bounds is not None and len(bounds) != 2:
            raise ValueError('type the Distance range as min, max in mm, or leave it blank')
        return {'distance_seed_m': seed, 'distance_bounds_m': bounds}

    def _bounds_or_none(self):
        """The fits' bounds, or None while an entry is malformed (or the window is being built)."""
        try:
            return self._bounds()
        except (ValueError, KeyError):
            return None

    def _fit_policy(self):
        """What the fits take now, as a session records them: the zeros, bounds and distance, the energy
        choice and bound, the motor-scale choice, and the Diffractometer the zeros are read by (None
        while an entry is malformed)."""
        bounds, distance = self._bounds_or_none(), self._distance_or_none()
        if bounds is None or not distance:
            return None
        try:
            energy = float(self._text('energy_bound'))
        except ValueError:
            energy = math.nan
        if not (math.isfinite(energy) and energy > 0):
            if self.refine_energy.isChecked():
                return None
            energy = None
        limits = distance['distance_bounds_m']
        return {'bounds': bounds.to_record(), 'distance_seed_m': distance['distance_seed_m'],
                'distance_bounds_m': None if limits is None else list(limits),
                'refine_energy': self.refine_energy.isChecked(), 'energy_bound_eV': energy,
                'fit_motor_scales': self.scales.isChecked(),
                'instrument_geometry': self.geometry_editor.value().to_record(),
                **({'conditioning': self._conditioning().to_dict()} if self._conditioning().mode != 'off' else {})}

    def _save_session(self, policy=None):
        """Record the session's frames with the fit policy (combined Stitch review F2): when frames are
        prepared, a row calibrated or imported, a Fit or Refine done (the policy that job took), and
        at close. A malformed entry leaves the file as it was."""
        policy = self._fit_policy() if policy is None else policy
        if not self.frames or self._frame_motors is None or policy is None:
            return
        save_calibration_session(self.session_dir, self.frames, arm=self._frame_motors, fit_policy=policy)

    def _apply_fit_policy(self, policy):
        """A session's recorded choices into the window: its Diffractometer first, since the zeros are
        read by its circles (a horizontal 2θ arm's is nu's; combined Stitch review F3), then the zeros,
        bounds, mount and distance, the energy choice and bound and the motor-scale choice."""
        from xrd_tools.integrate.refine import ArmFitBounds
        if 'instrument_geometry' in policy:
            self.geometry_editor.set_value(InstrumentGeometry.from_record(policy['instrument_geometry']))
        band = StitchConditioning(**policy.get('conditioning', {'mode': 'off'}))
        self._threshold_state = ThresholdState(band.mode == 'manual', band.threshold_min, band.threshold_max,
                                               band.mode == 'auto')
        self._show_threshold()
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
        if 'refine_energy' in policy:  # version 5 (combined Stitch review F1)
            self.refine_energy.setChecked(policy['refine_energy'])
            if policy['energy_bound_eV'] is not None:
                self._set_field_text('energy_bound', f"{policy['energy_bound_eV']:g}")
            self.scales.setChecked(policy['fit_motor_scales'])

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
        parts = [f'{circle} zero {zeros[circle]:g}°' for circle in moving if circle in zeros]
        untyped = [circle for circle in moving if circle not in zeros]
        if untyped:
            parts.append(f"{' and '.join(untyped)} zero{'s' if len(untyped) > 1 else ''} 0° (none typed; a Refine "
                         "from a geometry file starts from its own)")
        self.bounds_note.setText('Fits start from: ' + ', '.join(parts)
                                 + f"; detector tilt {bounds.mount_deg[0]:g}°, roll {bounds.mount_deg[1]:g}° "
                                   "(both under 2 · Rough geometry)")

    def _reset_bounds(self):
        for key, value in self._BOUND_DEFAULTS.items():
            self._fields[key].setText(value)

    def _text(self, key):
        field = self._fields[key]
        if isinstance(field, QtWidgets.QComboBox):
            data = field.currentData()
            return str(field.currentText() if data is None else data).strip()
        row = self._formrows.get(key)  # a file row shows its file's name: its value is the row's
        return str(field.text() if row is None else row.current_value()).strip()

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
            kind = SourceKind.NEXUS_STACK if self.source_kind.currentData() == 'nexus_stack' else SourceKind.SPEC
        options = {}
        entry = self._shared_nexus_entry or self._text(prefix + 'entry') or 'entry'
        if kind is SourceKind.SPEC:
            if (beam and self.beam_source_kind.currentData() == 'spec_container'
                    and self._text('beam_image_file') and not self._text('beam_images') and not self._text('beam_stem')):
                return SourceSpec(Path(path).resolve(), kind, entry=entry,
                    options={'scan': self._text('beam_scan'), 'image_file': self._text('beam_image_file'),
                             **({'nexus_entry': self._shared_nexus_entry} if self._shared_nexus_entry else {})})
            # a blank folder is the SPEC file's, as in Stitch; a blank stem is the source's default
            options = {'scan': self._text(prefix + 'scan'),
                       'image_dir': self._image_folder(prefix),
                       'read_image_kwargs': self._reader()}
            if self._text(prefix + 'stem'):
                options['image_stem'] = self._text(prefix + 'stem')
        if self._shared_nexus_entry and self._data_format != 'spec_series':
            options['nexus_entry'] = self._shared_nexus_entry
        return self.point_images.options(SourceSpec(Path(path).resolve(), kind,
            entry=entry, options=options))

    def _start(self, kind, arguments):
        if os.environ.get('XDART_DEBUG_ORIENTATION'):
            _LOG.warning('calibration start kind=%s busy=%s pending_metadata=%r orientation_generation=%s',
                         kind, self._owner.busy, self._pending_source_details, self._orientation_generation)
        if self._owner.busy:
            return
        self._job = kind
        self._point_request = (kind, arguments)
        if self._calibration_fit_window is not None:
            self._calibration_fit_window.preview_busy(True)
        self._identity = self._owner.begin((kind, arguments))
        # the two background reads keep typing open; the job buttons wait for them (maintainer, 2026-10-09)
        self._set_tabs_enabled(kind in ('beam_detect', 'source_details'))
        if self.orientation_panel is not None:
            self.orientation_panel.busy(True)
        self.cancel_button.setEnabled(True)
        self._job_status('Stitching calibrant preview…' if kind == 'stitch_preview' else
                            'Checking direct-beam scan…' if kind == 'beam_detect'
                            else f'{kind.capitalize()} in progress…')
        self._timer.start()
        self._sync_quick_button()
        self.busyChanged.emit(True)
        self._refresh_checklist()  # the Orientation check waits: the checklist, footer and Run say so (LT-008)

    def _job_status(self, detail):
        """A calibrant preview reports in its panel; the message box keeps the fit result."""
        if self._job == 'stitch_preview':
            if self._calibration_fit_window is not None:
                self._calibration_fit_window.preview_busy(self._owner.busy, detail)
        else:
            self.status.setText(detail)

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
            conditioning = self._conditioning()
            self._start('prepare', (self._enumerated_spec, indices, self.session_dir, self._orientation(),
                                   self._enumerated_motors, conditioning,
                                   self._saturation_value() if conditioning.mode != 'off' else None))
        self._guard(go)

    def _selected_frame(self):
        row = self.frame_table.currentRow()
        if row < 0 or row >= len(self.frames):
            raise ValueError('select a calibration frame')
        frame = self.frames[row]
        self._row_identity = frame.identity
        return frame

    def _calibrate(self):
        def start():
            if self._staged_camera_message():  # the mask would be turned for this camera, the frame for its own
                raise ValueError(self._staged_camera_message())  # (Codex round-2 review F3; the Fit's refusal)
            frame = self._selected_frame()
            seed = {**self._seed_options(), 'mask': self._pyfai_mask(frame)}
            if not _frame_energy_known(frame, seed):
                raise ValueError(f'frame {frame.identity}: its scan records no energy; type Energy (keV) '
                                 '(3 · Calibrant scan) so pyFAI calibrates it at the right one')
            self._start('calibrate', (frame, seed))
        self._guard(start)

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
        wavelength = self._rough_wavelength_A()
        return dict(calibrant=self.refine_calibrant.currentText().strip() or 'LaB6',
                    del_motor=self._frame_motors[0], nu_motor=self._frame_motors[1], prediction=prediction,
                    record=self.fit.asset.get('calibration') if self.fit is not None else None,
                    frames=done, rough=self._rough_record(), bounds=self._bounds_or_none(),
                    distance=self._distance_or_none(),
                    file=str(Path(path).resolve()) if path and Path(path).is_file() else None,
                    detector=detector, wavelength_A=wavelength,
                    instrument=self.geometry_editor.value().to_record(),  # its circles' senses
                    turned=self._turned_detector())  # a quarter turn's model (item 8)

    def _calibrated_key(self):
        """The calibrated frames and the bounds and senses the prediction was fitted with: editing a
        zero, a bound or a circle's sense leaves a prediction stale."""
        bounds = self._bounds_or_none()
        key = (tuple((frame.identity, frame.poni_sha256) for frame in self.frames if frame.complete),
               json.dumps([None if bounds is None else bounds.to_record(),
                           self._text('distance_seed'), self._text('distance_bounds'),
                           self.geometry_editor.value().arm_senses()], sort_keys=True))
        turned = self._turned_detector()  # a quarter turn: and the model it was fitted turned (item 8)
        return key if turned is None else (*key, json.dumps(turned, sort_keys=True))

    def _expected_rings(self, frame):
        """The calibrant rings predicted on this frame (None without a calibration)."""
        seed = self._seed_options()
        record = next((value for value in (seed['record'], seed['prediction'], seed['rough'], seed['file'])
                       if value is not None), None)
        if record is None:
            return None
        band = self._conditioning()
        try:
            saturation = self._saturation_value() if band.mode != 'off' else None
        except ValueError:
            return None  # a draft saturation is not a predicted pixel range
        import hashlib
        key = (frame.identity, seed['calibrant'], hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
               if isinstance(record, dict) else record, self._mask_state(), band, saturation)
        if key not in self._ring_cache:
            from xrd_tools.integrate.calibration_seed import predict_frame_seed
            try:
                self._ring_cache[key] = predict_frame_seed(
                    record, frame, calibrant=seed['calibrant'], del_motor=seed['del_motor'],
                    nu_motor=seed['nu_motor'], mask=self._loaded_mask(), conditioning=band,
                    saturation=saturation).rings
            except (ValueError, KeyError, OSError):
                self._ring_cache[key] = None
        return self._ring_cache[key]

    def _import(self):
        def go():
            if self._owner.busy:  # before the file dialogs: _start would drop the import once both are chosen
                raise ValueError(self._BUSY_WAIT)
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
        self._checklist_fit_outdated = self.fit is not None or self._checklist_fit_outdated
        self._saved_ring_fit = None
        self._cancel_preview()
        self._fit_source = None
        self.fit = None
        self._fit_instrument = None
        self._fit_setup = None
        self.save_button.setEnabled(False)
        self.fit_summary.clear()
        self.show_calibration_fit_button.setEnabled(False)
        if self._calibration_fit_window is not None:  # it showed the fit that is gone
            self._calibration_fit_window.clear_preview()
            self._calibration_fit_window.close()
        self._refresh_checklist()

    def _geometry_edited(self):
        """A fit belongs to the instrument it recorded: any other value leaves it unsaveable."""
        if not hasattr(self, 'save_button'):  # still being built
            return
        if self.fit is not None and self.geometry_editor.value().to_record() != self._fit_instrument:
            self._invalidate_fit()
        self._sync_arm_model()

    def _staged_camera_message(self):
        """Why the staged frames no longer match the Detector panel's camera, or None: each was turned and
        mirrored as the camera said when it was staged, and its points are in that image."""
        from xrd_tools.core.geometry import ImageOrientation
        from xrd_tools.core.geometry.camera import stitch_camera
        orientation = self._orientation()
        if not self.frames or (self.frames[0].rotation, self.frames[0].flip) == (orientation.rotation,
                                                                                orientation.flip_horizontal):
            return None
        staged = stitch_camera(ImageOrientation(rotation=self.frames[0].rotation,
                                                flip_horizontal=self.frames[0].flip))
        now = self.camera.value()
        return (f'The frames were staged with the camera rows {staged.rows}, columns {staged.columns}; the '
                f'Detector panel now says rows {now.rows}, columns {now.columns}. Restore it, or start a New '
                'session to stage and calibrate frames with this camera.')

    def _camera_edited(self, *_args):
        """A fit belongs to the camera its frames were staged with: another leaves it unsaveable."""
        if not hasattr(self, 'save_button') or self.fit is None:  # still being built, or nothing to keep
            return
        provenance, orientation = self.fit.asset['provenance'], self._orientation()
        if (provenance.get('image_rotation', 0), provenance.get('image_flip', False)) != (
                orientation.rotation, orientation.flip_horizontal):
            self._invalidate_fit()

    def _staged_arm_message(self, arm):
        def described(pair):  # a single arm's one motor, with the plane that says which circle it is
            delta, nu = pair
            if delta and nu:
                return f'{delta} and {nu}'
            return f'{delta or nu} as a {"vertical" if delta else "horizontal"} 2θ arm'
        return (f'The frames were staged reading {described(self._frame_motors)}; the Diffractometer now names '
                f'{described(arm)}. Restore those motors under 1 · Instrument, or start a New session to '
                'stage frames with these.')

    def _sync_arm_model(self):
        """Say which arm model the next fit takes, and why (decision 40: automatic)."""
        delta, nu = self._frame_motors if self.frames else self.geometry_editor.value().stitch_arm()
        done = tuple(frame for frame in self.frames if frame.complete)
        reason = automatic_arm_reason(done, del_motor=delta, nu_motor=nu, fit_scales=self.scales.isChecked())
        if not reason:
            text = ('Model (chosen for you): the five-parameter arm (distance, beam centre, motor zeros) with '
                    "the extension (the detector's tilt and roll on the arm, the sample's offset from the rotation "
                    f'centre), which the frames on {len({round(float(frame.motors[nu]), 6) for frame in done})} '
                    'nu positions can determine; without it if that fit is refused.')
        else:
            text = ('Model (chosen for you): the five-parameter arm (distance, beam centre, motor zeros), without '
                    "the extension (the detector's tilt and roll on the arm, the sample's offset from the rotation "
                    f'centre): {reason}.')
        if self.scales.isChecked():
            text += (' With fitted motor scales the energy stays at the recorded one (it is refined only with '
                     'fixed scales) and Save writes no xrayutilities file.')
        self.arm_note.setText(text)

    def _fit(self):
        def go():
            if self._owner.busy:
                raise ValueError(self._BUSY_WAIT)
            if not self.frames or not all(frame.source_uri for frame in self.frames):
                raise ValueError('Select a calibrant source and prepare its frames first (3 · Calibrant scan).')
            if self._staged_source_mismatch():
                raise ValueError('The prepared frames belong to another calibrant source; select their source '
                                 'or start a new session and prepare the selected scan.')
            if self.frames and self.geometry_editor.value().stitch_arm() != self._frame_motors:
                raise ValueError(self._staged_arm_message(self.geometry_editor.value().stitch_arm()))
            if self._staged_camera_message():
                raise ValueError(self._staged_camera_message())
            frames = tuple(frame for frame in self.frames if frame.complete)
            options = dict(del_motor=self._frame_motors[0], nu_motor=self._frame_motors[1],
                fit_scales=self.scales.isChecked(), instrument_geometry=self.geometry_editor.value().to_record(),
                arm_model='auto', **self._distance_options(), max_rms_q=float(self._text('rms')),
                distance_provenance=(self._distance_provenance if getattr(self, '_distance_provenance_key', None)
                                     is not None and self._distance_provenance_key == self._direct_beam_key()
                                     else None),  # a direct-beam record only while its inputs hold
                # the solver refines the energy only with fixed scales: fitted ones hold it (arm note)
                energy_bound_eV=None if self.scales.isChecked() else self._energy_bound(), bounds=self._bounds(),
                turned_from=self._turned_detector())  # a quarter turn's model: the record carries it (item 8)
            self._invalidate_fit()
            self._job_policy = self._fit_policy()
            self._pending_fit_source = self._spec()
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
            if self._owner.busy:
                raise ValueError(self._BUSY_WAIT)
            if not self._calibrant_source_ready():
                raise ValueError('Select a valid calibrant source and read its metadata first (3 · Calibrant scan).')
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
                    raise ValueError('measure or type the distance and the beam pixel first (2 · Rough geometry)')
                if self._rough_wavelength_A() is None:
                    raise ValueError(self._NO_ENERGY)
            else:
                if not self._text('refine_seed_path'):
                    raise ValueError('choose the geometry file to start from')
                seed = str(Path(self._text('refine_seed_path')).resolve())
            record = seed if isinstance(seed, dict) else json.loads(Path(seed).read_text(encoding='utf-8'))
            source_motors = tuple((position, motor) for position, motor in (('del_angle', delta), ('nu_angle', nu))
                                  if motor is not None)
            from xrd_tools.session.instrument import calibration_arm
            problem = geometry.arm_mismatch(*calibration_arm(record, source_motors))
            if problem:
                raise ValueError(problem.replace('This calibration', 'The seed geometry', 1))
            max_counts = self._saturation_value()
            if max_counts is not None and not self._text('saturation'):
                max_counts -= 1  # the ring finder keeps <= max_counts; Auto rejects the ceiling itself
            options = dict(source_motors=source_motors,
                           instrument_geometry=geometry.to_record(),
                           image_rotation=self._orientation().rotation,
                           image_flip=self._orientation().flip_horizontal,
                           calibrant=self.refine_calibrant.currentText().strip() or 'LaB6',
                           max_frames=int(self._text('refine_frames')),
                           max_counts=max_counts,
                           conditioning=None if self._conditioning().mode == 'off' else self._conditioning(),
                           mask=self._oriented_mask() if self._text('mask') else None,
                           max_rms_q=float(self._text('rms')), energy_bound_eV=self._energy_bound(),
                           bounds=self._bounds(), **self._distance_options())
            self._invalidate_fit()
            self._job_policy = self._fit_policy()
            self._pending_fit_source = spec
            self._start('refine', (spec, seed, options))
        self._guard(go)

    def _save(self):
        def go():
            if self._owner.busy:  # before the file dialog: _start would drop the save once a name is chosen
                raise ValueError(self._BUSY_WAIT)
            if self.fit is None:
                raise ValueError('fit the current controls before saving')
            selected = self._save_path(beam=False)
            if selected is not None:
                self._saving = self.fit
                self._start('save', (self.fit, selected, pair_validation_angles(
                    self.fit, scan_angles=self._scan_angles(self.fit))))
        self._guard(go)

    def _setup_state(self):
        """The instrument/detector setup that changes the meaning of a saved calibration."""
        return (self.geometry_editor.value(), self.camera.value(),
                self._text('beam_detector'), self._text('beam_orientation'))

    def _setup_edited(self, *_args):
        if self.fit is not None and self._fit_setup is not None and self._setup_state() != self._fit_setup:
            self._invalidate_fit()

    def _default_name(self, beam):
        """The calibration's default file name, as Stitch names its results (maintainer, 2026-09-29:
        "spec_name_scan_no.json"): the calibrant scan's (the direct-beam scan's for a direct-beam
        geometry) SPEC file and scan, or its NeXus file; the first free name, since a saved calibration
        is never overwritten."""
        prefix = 'beam_' if beam else ''
        source_text = self._text(prefix + 'source')
        source = Path(source_text or 'calibration')
        scan = re.match(r'\d+', self._text(prefix + 'scan') or '')
        stem = (f'{source.stem}_scan{scan[0]}' if scan and not (beam and self.beam_source_kind.currentData() == 'nexus')
                else source.stem)
        folder = ((source if source.is_absolute() else self.project_root / source).parent
                  if source_text else self.project_root)
        for count in range(1, 1000):
            name = folder / (f'{stem}.json' if count == 1 else f'{stem}_{count}.json')
            if not name.exists() and not xu_partner_path(name).exists():
                return name
        return folder / f'{stem}-{uuid.uuid4().hex[:8]}.json'

    def _save_path(self, *, beam=False, new_only=False):
        """The new calibration file the person chooses inside the project, or None."""
        from xdart.gui.widgets import file_dialogs
        # The shared last-folder rule precedes the source folder; checks keep it inside the Project and new.
        suggested = self._default_name(beam)
        picks = file_dialogs.browse(self, 'Save calibrated geometry', save=True, file_filter='JSON (*.json)',
                                    current=str(suggested), fallback=str(self.project_root),
                                    default_name=suggested.name)
        path = picks[0] if picks else ''
        if not path:
            return None
        selected = Path(path).resolve()
        if not selected.is_relative_to(self.project_root):
            raise ValueError(f'save the geometry inside the {self.context.page} project')
        partner = xu_partner_path(selected)
        companions = (partner, selected.with_suffix(selected.suffix + '.provenance.json'),
                      partner.with_suffix(partner.suffix + '.provenance.json'))
        if new_only and (selected.exists() or any(path.exists() for path in companions)):
            raise ValueError('Choose a new calibration filename; Save on close preserves the original pair')
        if not selected.exists() and any(companion.exists() for companion in companions):
            # Qt confirms an existing primary. Orphan companions need the same explicit choice.
            answer = QtWidgets.QMessageBox.question(self, 'Replace calibration files?',
                f'Replace the existing companion files for {selected.name}?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return None
        return selected

    def _save_rough(self):
        """Save step 1's geometry as the calibration pair, without a ring fit."""
        def go():
            if self._owner.busy:  # before the file dialog: _start would drop the save once a name is chosen
                raise ValueError(self._BUSY_WAIT)
            self._render_direct_beam_advice()  # what is saved is the arm the advice is about
            fit = self._rough_fit()
            selected = self._save_path(beam=self.rough_mode.currentData() == 'scan')
            if selected is not None:
                angles = self._scan_angles(fit)
                if self.rough_mode.currentData() == 'scan' and self._distance_result is not None:
                    role = self._beam_role()
                    angles += [(angle, 0.) if role == 'del' else (0., angle)
                               for angle in self._distance_result.angles_deg.tolist()]
                self._saving = fit
                self._start('save', (fit, selected, pair_validation_angles(
                    fit, scan_angles=angles)))
        self._guard(go)

    def _rough_fit(self):
        """Step 1's geometry as a calibration the save job takes; its provenance says it is a
        direct-beam calibration, not ring-fitted."""
        from xrd_tools.integrate.stitch_calibration import CalibrationFit
        record = self._rough_arm()
        if record is None:
            raise ValueError('measure or type the distance and the beam pixel first (2 · Rough geometry)')
        if self._rough_wavelength_A() is None:
            raise ValueError('type Energy (keV) to save this geometry as a calibration')
        delta, nu = self.geometry_editor.value().stitch_arm()  # as _rough_arm
        direct = ({**self._distance_record(self._distance_result),
                   'held_motor_readings': dict(self._distance_held)}  # the pose the beam was seen at
                  if self.rough_mode.currentData() == 'scan' and self._distance_result is not None else None)
        provenance = {
            'frames': [], 'rms_q_A^-1': None, 'condition_number': None, 'n_points': 0,
            'distance_seed_m': None, 'distance_bounds_m': None, 'fit_motor_scales': False,
            'source_motors': {name: motor for name, motor in (('nu_angle', nu), ('del_angle', delta))
                              if motor is not None},
            'image_rotation': self._orientation().rotation,
            **({'image_flip': True} if self._orientation().flip_horizontal else {}), 'direct_beam': direct,
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

    def _oriented_mask(self):
        """The Detector panel's mask, drawn on the stored image, turned and mirrored as the images are."""
        from xrd_tools.io.image import load_mask
        return self._orientation().apply(load_mask(self._text('mask'))).astype(bool)

    def _mask_state(self):
        """The mask as used: its file's state and the camera (None without a mask)."""
        if not self._text('mask'):
            return None
        try:
            state = Path(self._text('mask')).stat()
        except OSError:
            return (self._text('mask'), None)
        return (self._text('mask'), state.st_mtime_ns, state.st_size, self.camera.value().to_xu())

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

    def _pyfai_mask(self, frame=None):
        """The turned mask as a file pyFAI-calib2 reads (non-zero: masked), in the session folder; under a quarter
        turn with the model's own gaps turned too, which pyFAI's generic detector there does not hold (item 8).
        ``frame``'s staged pixels add the detector's invalid-pixel value whatever the threshold and, with it on, the
        band (calibration_frame_mask); without a frame, the threshold off, the Detector panel's mask alone."""
        mask = self._loaded_mask()
        if mask is None and self._text('mask'):
            raise ValueError(f"the Detector panel's mask cannot be read: {self._text('mask')}")
        if frame is not None or self._conditioning().mode != 'off':
            from xrd_tools.integrate.stitch_calibration import calibration_frame_mask
            mask = calibration_frame_mask(frame or self._selected_frame(), mask,
                                          conditioning=self._conditioning(), saturation=self._saturation_value())
        turned = self._turned_detector()
        if turned is not None:
            from xrd_tools.integrate.turned_detector import turned_model_mask
            gaps = turned_model_mask(turned['detector'], self._orientation())
            if gaps is not None and mask is not None and mask.shape != gaps.shape:
                raise ValueError(f"the Detector panel's mask is {mask.shape[1]} × {mask.shape[0]} px turned, the "
                                 f"{turned['detector']}'s image {gaps.shape[1]} × {gaps.shape[0]}")
            if gaps is not None:
                mask = gaps if mask is None else mask | gaps
        if mask is None:
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
        if self._beam_detection is None or self._beam_detection[0] != self._beam_generation:
            raise ValueError(self._beam_problem)
        _, motor, motion, *_diagnostics = self._beam_detection
        options = self._beam_fit_options(with_mask=with_mask)
        options.update(motion=motion, angle_zero_deg=self._zeros().get(self._beam_role(), 0.))
        arm = dict(zip(('del', 'nu'), self.geometry_editor.value().stitch_arm()))
        held = tuple(name for circle, name in arm.items() if name and circle != self._beam_role())
        return self._spec(beam=True), motor, options, held

    def _beam_fit_options(self, *, with_mask=True):
        """Detector and extraction settings, independent of source readiness."""
        orientation = int(self._text('beam_orientation'))
        options = dict(detector=self._text('beam_detector'),
            detector_config={'orientation': orientation} if orientation != 3 else None,
            beam_pixel=(tuple(float(v) for v in self._text('beam_pixel').split(','))
                        if self._text('beam_pixel') else None),  # blank: found at the zero
            strip_half_width=int(self._text('beam_strip')), model='tangent',
            rotation=self._orientation().rotation, flip=self._orientation().flip_horizontal)
        if self._text('beam_roi'):
            options['roi'] = tuple(int(v) for v in self._text('beam_roi').split(','))
        if self._text('mask'):
            options['mask'] = self._oriented_mask() if with_mask else self._mask_state()
        saturation = self._saturation_value()
        if saturation is not None:
            options['saturation'] = saturation
        return options

    def _invalidate_beam_detection(self, *_args):
        if not hasattr(self, '_beam_probe_timer'):
            return
        self._beam_generation += 1
        self._beam_detection = None
        self._beam_problem = ('Checking direct-beam scan…' if self._text('beam_source')
                              else 'Select a direct-beam source and scan')
        if self._job == 'beam_detect' and self._identity is not None:
            self._owner.cancel(self._identity)
        self._update_beam_readiness()
        self._beam_probe_timer.start()
        self._render_direct_beam_advice()  # the camera or an extraction input changed: the fit may be stale

    def _beam_geometry_changed(self, *_args):
        arm = self.geometry_editor.value().stitch_arm()
        if arm != self._beam_arm:
            self._beam_arm = arm
            self._invalidate_beam_detection()

    def _update_beam_readiness(self):
        ready = self._beam_detection is not None and self._beam_detection[0] == self._beam_generation
        text = self._beam_problem
        if ready:
            dx, row_delta = self._beam_detection[3]
            dy = -row_delta  # Int's raw viewer mirrors detector Y once; +Y is up.
            first, last = self._beam_detection[4]
            text = (f'{self._beam_detection[1]} scan ({first:g}, {last:g}) · '
                    f'{self._beam_detection[5]} pts · Δx {dx:+.2f} px, Δy {dy:+.2f} px'
                    f' ({"right" if dx > 0 else "left"}, {"up" if dy > 0 else "down"})')
        self.beam_button.setEnabled(ready and not self._owner.busy)
        self.beam_button.setToolTip(text)
        self.beam_readiness.set_readiness(ready, text)
        if hasattr(self, 'status') and self.tabs.currentWidget() is self._tab_pages['rough']:
            self.status.setText(text)
        self._refresh_checklist()

    def _tab_readiness(self, *_args):
        if self.tabs.currentWidget() is self._tab_pages['rough']:
            self._update_beam_readiness()
        elif self.tabs.currentWidget() is self._tab_pages['select']:
            self.status.setText('Select a calibrant scan; three to five frames are suggested.')
        elif self.tabs.currentWidget() is self._tab_pages['instrument']:
            self.status.setText('Describe the diffractometer, detector and camera.')
        elif self.tabs.currentWidget() is self._tab_pages['calibrate']:
            self.status.setText(calibration_fit_text(self.fit, self._fit_kind) if self.fit is not None
                                else 'Calibrate selected frames or refine the calibrant scan.')

    def _detect_beam(self):
        if self._owner.busy:
            self._beam_probe_timer.start()
            return
        if self._pending_source_details is not None:
            prefix, self._pending_source_details = self._pending_source_details, None
            self._source_details(prefix)
            if self._owner.busy:
                return
        if self._restoring_fields or self.rough_mode.currentData() != 'scan' or not self._text('beam_source'):
            return
        if self._beam_detection is not None:
            self._update_beam_readiness()
            return
        try:
            options = dict(detector=self._text('beam_detector'),
                detector_config=({'orientation': int(self._text('beam_orientation'))}
                                 if int(self._text('beam_orientation')) != 3 else None),
                rotation=self._orientation().rotation, flip=self._orientation().flip_horizontal)
            if self._text('mask'):
                options['mask'] = self._oriented_mask()
            if self._text('beam_roi'):
                options['roi'] = tuple(int(v) for v in self._text('beam_roi').split(','))
                if len(options['roi']) != 4:
                    raise ValueError('ROI requires x0, y0, x1, y1')
            saturation = self._saturation_value()
            if saturation is not None:
                options['saturation'] = saturation
            self._beam_probe_generation = self._beam_generation
            self._start('beam_detect', (self._spec(beam=True),
                tuple(name for name in self.geometry_editor.value().stitch_arm() if name), options))
            self._update_beam_readiness()
        except (ValueError, TypeError, KeyError, OSError) as error:
            self._beam_problem = str(error)
            self._update_beam_readiness()

    def _direct_beam_key(self):
        """What a direct-beam result was fitted from (source, motor, zero, detector, extraction), or
        None when the window cannot say: a result is used only while this is unchanged (Codex review F5)."""
        try:
            spec, motor, options, held = self._direct_beam_request(with_mask=False)
        except (ValueError, TypeError, KeyError, OSError):
            return None
        return json.dumps([str(spec.uri), spec.kind.value, spec.entry, dict(spec.options), motor, options, held],
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
        self._render_direct_beam_advice()  # about the controls as they are now
        if self._direct_beam_fit_window is None:
            self._direct_beam_fit_window = DirectBeamFitWindow(self)
        self._direct_beam_fit_window.show_result(self._distance_result, self._distance_energy_eV,
                                                 self._distance_held, self._distance_warning)
        self._direct_beam_fit_window.show()
        self._direct_beam_fit_window.raise_()
        self.show_fit_button.setEnabled(True)

    def _direct_beam_motion_warning(self):
        """What disagrees, and the mounting to try, when step 1's geometry moves the direct beam another way than
        the fitted scan did; None when it agrees or cannot tell. The fit keeps |slope|, so a detector mounted the
        other way round fits as well (maintainer, 2026-10-06: "why 90 degrees doesn't work")."""
        from xrd_tools.core.geometry.camera import DetectorCamera
        from xdart.gui.widgets.camera_editor import mounting_description
        try:
            delta, nu = self.geometry_editor.value().stitch_arm()
            motor = {'del': delta, 'nu': nu}.get(self._beam_role())
            record = None if motor is None else self._rough_arm()
            if record is None:
                return None
            mismatch = direct_beam_motion_mismatch(record, self._distance_result, motor=motor, del_motor=delta,
                                                   nu_motor=nu, held=self._distance_held)
        except ValueError:  # step 1 has no arm to compare yet (stale, incomplete), or one it cannot place
            return None
        except Exception:  # noqa: BLE001 - an advisory: the fit and its display never depend on it
            _LOG.warning('direct-beam motion check failed', exc_info=True)
            return None
        if mismatch is None:
            return None
        camera = self.camera.value()
        if mismatch.quarter_turn:
            return f'{mismatch} The camera is probably a quarter turn off from {mounting_description(camera)}.'
        turned = DetectorCamera(*(axis[0] + ('-' if axis[1] == '+' else '+') for axis in camera.to_xu()))
        unmeasured = 'nu' if self._beam_role() == 'del' else 'del'
        return (f'{mismatch} The detector is probably mounted the other way round: try '
                f"{mounting_description(turned)} rather than {mounting_description(camera)}, or check {motor}'s "
                'sense in the Diffractometer section. A single-axis reflection as well as a 180° turn can '
                f'reverse this motion, with a corresponding sense hypothesis; {unmeasured} is unmeasured. '
                'This one-axis scan cannot identify the mounting or both motor senses.')

    def _render_direct_beam_advice(self, *_args):
        """The motion advice for the controls as they are now, on its three surfaces together: the distance rows,
        the direct-beam fit window and the status line (Codex review B1: a sense or camera edit after the fit
        changes the geometry the advice is about). A fit that no longer matches its inputs keeps no advice, and
        the status says to fit again. Nothing changes while the advice does not."""
        if not hasattr(self, 'beam_summary') or self._distance_result is None:
            return
        advice = self._direct_beam_motion_warning()
        if advice == self._distance_warning:
            return
        self._distance_warning = advice
        rows = self.beam_summary.toPlainText().splitlines()
        rows = rows[1:] if rows and rows[0].startswith('⚠ ') else rows  # the fit's own rows stay as fitted
        self.beam_summary.setPlainText('\n'.join(([f'⚠ {advice}'] if advice else []) + rows))
        self.beam_summary.verticalScrollBar().setValue(0)
        if self._direct_beam_fit_window is not None:
            self._direct_beam_fit_window.show_result(self._distance_result, self._distance_energy_eV,
                                                     self._distance_held, advice)
        self.status.setText(direct_beam_fit_text(self._distance_result, self._distance_energy_eV,
                                                 self._distance_held, advice)
                            if self._distance_fresh() else self._STALE_DISTANCE)

    def _show_calibration_fit(self):
        """The ring fit (Fit or Refine) in its own window, with what it found (opened after every fit)."""
        if self.fit is None:
            self.status.setText('Fit the calibrated frames or refine first.')
            return
        if self._calibration_fit_window is None:
            self._calibration_fit_window = CalibrationFitWindow(self)
            self._calibration_fit_window.previewRequested.connect(self._stitch_preview)
            self._calibration_fit_window.previewClosed.connect(self._cancel_preview)
        self._calibration_fit_window.show_result(self.fit, self._fit_kind)
        self._calibration_fit_window.show()
        self._calibration_fit_window.raise_()
        self.show_calibration_fit_button.setEnabled(True)

    def _preview_inputs(self):
        return (id(self.fit), self._spec(), self._orientation(), self._text('mask'),
                self._conditioning(), self._saturation_value())

    def _stitch_preview(self):
        def go():
            if self._owner.busy or self.fit is None:
                return
            if self._fit_source is None or self._spec() != self._fit_source:
                raise ValueError('Select the calibrant source used for this fit before stitching its preview.')
            self._preview_key = self._preview_inputs()
            self._start('stitch_preview', (self.fit, self._fit_source,
                dict(mask=self._oriented_mask() if self._text('mask') else None,
                     conditioning=self._conditioning(), saturation=self._saturation_value())))
        try:
            go()
        except Exception as error:
            if self._calibration_fit_window is not None:
                self._calibration_fit_window.preview_busy(False, str(error))

    def _cancel_preview(self):
        self._preview_key = None
        if self._owner.busy and self._job == 'stitch_preview':
            self._cancel()

    def hideEvent(self, event):
        """The fit windows go with this window (closed, hidden, or shut down by the host)."""
        for window in (self._direct_beam_fit_window, self._calibration_fit_window, self.instrument_schematic):
            if window is not None:
                window.close()
        super().hideEvent(event)

    def keyPressEvent(self, event):
        """Return or Enter commits the box being typed in and presses no button: with many boxes and
        buttons, the dialog's default button (Fit direct-beam distance) would start a job.
        Escape returning from a file chooser must not reject this retained window."""
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Escape):
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
        self._advanced_card.set_collapsed(False)  # where the seed is: Advanced ▸ Starting distance
        self._show_tab('calibrate' if self._calibrant_source_ready() else 'select')

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

    def _render_frames(self, *, invalidate_orientation=True):
        if invalidate_orientation:
            self._invalidate_orientation()
        if self.orientation_panel is not None:
            self.orientation_panel.set_frames(self.frames)
        self._update_fit_source_label()
        self._refresh_energy_hint()  # a calibrated frame's PONI file can supply the energy
        rows, tips = [], {}
        for row, frame in enumerate(self.frames):
            # the frame's own detector image (and its frame number inside a multi-frame file)
            image = Path(frame.raw_path).name + (f' #{frame.raw_index}' if frame.raw_index else '')
            values = [str(frame.index), ', '.join(f'{k}={v:g}' for k, v in frame.motors.items()),
                      image, Path(frame.poni_path).name if frame.complete else '—',
                      'Complete' if frame.complete else 'Needs PONI + points']
            rings = self._expected_rings(frame)
            values.append('—' if rings is None else ring_summary(rings) or 'none')
            ring_tip = ('No calibration to predict from yet: enter a rough geometry (2 · Rough geometry) or a '
                        'geometry file, or calibrate one frame: the others are then predicted from it.'
                        if rings is None
                        else '\n'.join(f'Ring {ring.number} · 2θ {ring.two_theta_deg:.2f}° · d {ring.d_spacing_A:.4f} Å'
                                        for ring in rings) or 'No calibrant ring reaches this frame.')
            for column, value in enumerate(values):
                tips[row, column] = (f'{frame.raw_path}\nstaged for pyFAI: {frame.image_path}' if column == 2
                                     else ring_tip if column == 5 else f'{frame.poni_path}\n{frame.npt_path}')
            rows.append(values)
        self.frame_table.show_rows(('Frame', 'Delta / Nu (°)', 'Image', 'PONI / points', 'Status',
                                    'Expected rings'), rows, cell_tooltips=tips)
        if self.frames:
            self.frame_table.selectRow(0)
        self.session_label.setText(f'Session: {self.session_dir / "calibration-session.json"}')
        self._sync_arm_model()
        self._refresh_checklist()  # the expected rings read the mask once: the Orientation check's readiness knows it

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
                frames, arm, policy = load_calibration_session(path, with_arm=True, with_policy=True)
                # an older session names no arm: del and nu in order, or a vertical 2θ arm
                motors = arm or (tuple(frames[0].motors) + (None,))[:2]
                if (policy is not None and 'instrument_geometry' not in policy
                        and tuple(self.geometry_editor.value().stitch_arm()) != tuple(motors)):
                    # a version-4 session's zeros are by circle of its arm: never read by another
                    raise ValueError(f"this session's arm is del = {motors[0] or 'held'}, nu = {motors[1] or 'held'}: "
                                     'choose that Diffractometer under 1 · Instrument, then open it again')
                self.frames, self._frame_motors = frames, motors
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
                elif 'refine_energy' not in policy:
                    self.status.setText('This session records no energy, motor-scale or Diffractometer choice: '
                                        'the window\'s are used.')
        self._guard(go)

    #: The buttons that start a job wait while one runs (maintainer, 2026-10-09): the background direct-beam probe
    #: keeps the tabs open for typing, and a press then was taken by Qt and dropped by _start, Fit and Refine
    #: discarding the current fit first (CAL-BUSY-ACTION-DISCARDS-FIT). The session buttons wait too: they replace the
    #: frames a job works on, and return while one runs (the metadata read keeps the tabs open since CAL-HELD-BATCH).
    _JOB_BUTTONS = ('list_button', 'prepare_button', 'calibrate_button', 'import_button', 'fit_button',
                    'refine_button', 'save_rough_button', 'new_session_button', 'open_session_button')
    _BUSY_WAIT = 'Wait for the running operation to end (Cancel operation stops it); the current fit is kept.'

    def _sync_job_buttons(self):
        for name in self._JOB_BUTTONS:
            button = getattr(self, name, None)
            if button is not None:
                button.setEnabled(not self._owner.busy)
        if hasattr(self, 'save_button'):  # it saves the adopted fit: enabled with one, as adoption sets it
            self.save_button.setEnabled(self.fit is not None and not self._owner.busy)

    def _sync_quick_button(self):
        """Every start and end of a job comes here: the Quick button and the job buttons follow it."""
        self._sync_job_buttons()
        self._sync_quick_folder_buttons()
        if not hasattr(self, 'quick_button'):
            return
        active = self._quick_stage is not None
        ready = not active and not self._owner.busy
        self.quick_button.setEnabled(ready)
        self.quick_button.setToolTip(self._QUICK_TIP if ready else self._QUICK_TIP + (
            '\nRunning now: Cancel operation stops it.' if active else '\nAvailable when the current operation ends.'))

    def _quick_start(self):
        """Quick calibration: the window's own Fit direct-beam distance, Read metadata and Refine automatically from
        the rough geometry, one after another, stopping at the first problem (_quick_advance)."""
        if self._quick_stage is not None or self._owner.busy:
            return
        for key, row in self._quick_rows.items():  # a scan typed on the card and not yet committed is the one used
            if row.editor.isModified() and row.current_value() != self._text(key):
                row.editor.editingFinished.emit()
        if self.rough_mode.currentData() != 'scan':
            self._quick_stop(1, self._QUICK_NEEDS_SCAN)
            return
        self._quick_cancelled, self._quick_detected_generation = False, None
        self._quick_stage, self._quick_kind = 1, None
        self._quick_progress()
        self._quick_advance()

    def _quick_advance(self):
        """Start the current step's job once the worker is free. A refresh or check the window queued (restoring
        its settings, a source edit) runs first, and its end brings Quick back here (review A3); a job that is
        refused at once stops Quick with its reason."""
        if self._quick_stage is None or self._close_pending or self._owner.busy:
            return
        if self._quick_stage == 1:
            if self.rough_mode.currentData() != 'scan':  # switched while its direct-beam check ran (tabs enabled)
                self._quick_stop(1, self._QUICK_NEEDS_SCAN)
                return
            if self._beam_detection is None or self._beam_detection[0] != self._beam_generation:
                if self._quick_detected_generation == self._beam_generation:  # its check ran on these inputs
                    self._quick_stop(1, self._beam_problem)
                    return
                self._beam_probe_timer.stop()
                self._detect_beam()  # a queued source refresh may start first
                if self._owner.busy:
                    if self._job == 'beam_detect':  # the inputs that check runs on
                        self._quick_detected_generation = self._beam_probe_generation
                    return
                if self._beam_detection is None or self._beam_detection[0] != self._beam_generation:
                    self._quick_stop(1, self._beam_problem)
                    return
            self._quick_start_job('distance', self._distance)
        elif self._quick_stage == 2:
            self._quick_start_job('enumerate', self._enumerate)
        else:
            self.refine_seed.setCurrentIndex(self.refine_seed.findData('rough'))
            self._quick_start_job('refine', self._refine)

    def _quick_start_job(self, kind, action):
        self._quick_kind = kind
        action()
        if not (self._owner.busy and self._job == kind):  # refused at once: the reason is on the status line
            self._quick_kind = None
            self._quick_stop(self._quick_stage, self.status.text())

    def _quick_after_terminal(self, kind, status):
        """A job ended while Quick calibration runs: its next step, or a stop at the first problem. A job it did not
        start (a refresh, the direct-beam check) only lets it go on."""
        if self._quick_stage is None:
            return
        if self._close_pending:  # the window is closing: no later step
            self._quick_clear()
            return
        step = self._quick_stage
        if self._quick_cancelled:
            self._quick_stop(step, self._QUICK_CANCELLED)
            return
        if kind != self._quick_kind:
            self._quick_advance()
            return
        self._quick_kind = None
        if status is not OperationTerminalStatus.RETURNED:  # the job's own reason is on the status line
            self._quick_stop(step, self.status.text())
            return
        if kind == 'distance':
            if self._distance_warning:  # the camera or a motor's sense, as the fit's advice says
                self._quick_stop(1, self._distance_warning)
                return
            if not self._distance_fresh():
                self._quick_stop(1, self._STALE_DISTANCE)
                return
        elif kind == 'enumerate':
            if not self._calibrant_source_ready() or not self._calibrant_frame_count:
                self._quick_stop(2, self._QUICK_NO_FRAMES)
                return
            try:
                wavelength = self._rough_wavelength_A()
            except ValueError as error:
                self._quick_stop(2, str(error))
                return
            if wavelength is None:
                self._quick_stop(2, self._NO_ENERGY)
                return
            # the metadata picks the scan's own motors for the arm, which can make step 1's fit stale (review A4)
            if not self._distance_fresh():
                self._quick_stop(1, self._QUICK_METADATA_STALE + self._STALE_DISTANCE)
                return
            advice = self._direct_beam_motion_warning()
            if advice:
                self._quick_stop(1, advice)
                return
        elif self.fit is None:
            self._quick_stop(3, self.status.text())
            return
        else:  # Refine's summary and window are shown; its stitched preview reports in its own panel
            self._quick_done()
            return
        self._quick_stage = step + 1
        self._quick_progress()
        self._quick_advance()

    def _quick_stop(self, step, message):
        """Stop at ``step``: its tab shown first (switching tabs rewrites the status line, review A5), then the step
        and its reason on the status line and the card. Completed steps stay; every action stays available."""
        name, tab = self._QUICK_STEPS[step]
        self._quick_clear()
        self._show_tab(tab)
        reason = str(message or '').strip() or "see this step's controls."
        self.status.setText(f'Quick calibration stopped at {name}: {reason}')
        self.quick_card.set_readiness(False, f'Stopped at {name}: {reason}')

    def _quick_done(self):
        self._quick_clear()
        self._show_tab('calibrate')
        self.status.setText(calibration_fit_text(self.fit, 'refine'))
        self.quick_card.set_readiness(True, 'Done: refined from the rough geometry; inspect the fit, then save it')

    def _quick_progress(self):
        name, _ = self._QUICK_STEPS[self._quick_stage]
        self.quick_card.set_valid_marker(False)
        self.quick_card.set_status_text(f'Step {self._quick_stage} of 3: {name}…')
        self._sync_quick_button()

    def _quick_clear(self):
        self._quick_stage = self._quick_kind = None
        self._quick_cancelled = False
        self._sync_quick_button()

    def _cancel(self):
        if self._quick_stage is not None:  # Cancel operation stops Quick calibration at its step
            self._quick_cancelled = True
        self.point_images.finish_run()
        if self._identity is not None:
            self._owner.cancel(self._identity)
            self._job_status('Cancelling; completed rows and saved files are retained.')

    def _poll(self):
        update = self._owner.poll(self._identity)
        if update is None:
            return
        if update.terminal is None:
            self._job_status(f'{self._job.capitalize()}: {update.progress.stage}…')
            return
        terminal, kind = update.terminal, self._job
        if os.environ.get('XDART_DEBUG_ORIENTATION'):
            _LOG.warning('calibration terminal kind=%s status=%s pending_metadata=%r orientation_generation=%s',
                         kind, terminal.status, self._pending_source_details, self._orientation_generation)
        self._identity = None
        self._set_tabs_enabled(True)
        if self.orientation_panel is not None:
            self.orientation_panel.busy(False)
        self.cancel_button.setEnabled(False)
        self._timer.stop()
        if self._calibration_fit_window is not None:
            self._calibration_fit_window.preview_busy(False)
        self.busyChanged.emit(False)
        self._refresh_checklist()  # the Orientation check's Run follows its readiness, never busy alone (LT-008)
        if terminal.status is OperationTerminalStatus.RETURNED:
            result = terminal.payload
            if type(result) is PointImageDetection:
                if self._close_pending:
                    self._quick_clear()
                    self.point_images.finish_run()
                    self.close()
                    return
                if self._quick_cancelled:  # a returned preflight must not open a new prompt after Quick's Cancel
                    self.point_images.finish_run()
                    self._quick_after_terminal(kind, terminal.status)
                    self._sync_quick_button()
                    return
                if self.point_images.confirm((result,), self):
                    _, arguments = self._point_request
                    arguments = tuple(self.point_images.options(value) if type(value) is SourceSpec else value
                                      for value in arguments)
                    if kind == 'enumerate':
                        self._enumerated_spec = arguments[0]
                    self.point_images.finish_run()
                    self._start(kind, arguments)
                else:
                    self.status.setText('Grouping not confirmed; scan unprocessed')
                    self._quick_after_terminal(kind, None)
                    self._sync_quick_button()
                return
            self.point_images.finish_run()
            try:
                if kind == 'orientation':
                    key, (finding, figure, geometry) = result
                    if not self._close_pending and key == self._orientation_key():
                        self._orientation_geometry = geometry
                        self._orientation_figure = figure
                        self.orientation_panel.show_finding(finding, figure,
                            configured_senses=self.geometry_editor.value().arm_senses())
                        # The job drew the hypothesis selected at its start; draw the one selected now.
                        self._orientation_candidate_changed()
                elif kind == 'beam_detect':
                    if self._beam_probe_generation == self._beam_generation:
                        self._beam_detection = (self._beam_generation, *result)
                        self._beam_problem = ''
                    self._update_beam_readiness()
                elif kind == 'source_details':
                    prefix, energy, automatic, frame_count, source_problem = result
                    if self._source_details_inputs_key == self._source_details_input_key(prefix):
                        if not prefix and self._source_details_key == self._energy_source_key():
                            self._show_source_energy(energy)
                            self._calibrant_ready_key = self._source_details_key if frame_count else None
                            self._calibrant_frame_count = frame_count
                            self._calibrant_source_problem = source_problem
                            self._update_fit_source_label()
                        if automatic is not None:
                            self._fields[prefix + 'stem'].setPlaceholderText(f'auto · {automatic}')
                            self._quick_mirror(prefix + 'stem')
                        unchanged = self._source_details_orientation_inputs == (
                            self._orientation_key(), self._source_energy_eV)
                        self._render_frames(invalidate_orientation=not unchanged)
                elif kind == 'enumerate':
                    result, names, record = result
                    self.geometry_editor.set_available_motors(names)
                    self.geometry_editor.set_value(InstrumentGeometry.from_record(record))
                    self._enumerated_motors = InstrumentGeometry.from_record(record).stitch_arm()
                    read = [motor for motor in self._enumerated_motors if motor]
                    self._choices = result
                    self._calibrant_frame_count = len(result)
                    self._calibrant_source_problem = ''
                    self._calibrant_ready_key = self._energy_source_key()
                    energies = {energy for _index, _values, energy in result if energy is not None}
                    if len(energies) == 1:
                        self._show_source_energy(energies.pop())
                    suggested = suggested_frame_indices(result)
                    self.selection.show_rows(
                        ['Use', 'Frame', *(f'{motor} (°)' for motor in read)],
                        [('', str(index), *(f'{motors[key]:g}' for key in read)) for index, motors, _ in result],
                        check_states={(row, 0): index in suggested for row, (index, _, _) in enumerate(result)})
                elif kind == 'prepare':
                    self.frames = result
                    self._frame_motors = self._enumerated_motors
                    self._save_session()  # the policy from the start (combined Stitch review F2)
                    self._render_frames()
                    self._show_tab('calibrate')
                elif kind in ('calibrate', 'import') and result is not None:
                    result, prediction = result
                    self.frames = tuple(result if frame.identity == self._row_identity else frame for frame in self.frames)
                    self._prediction = (self._calibrated_key(), prediction)
                    self._save_session()
                    self._invalidate_fit()
                    self._render_frames()
                    self._select_next_frame(result)
                elif kind == 'predict':
                    self._prediction = result
                    self._render_frames()
                elif kind in ('fit', 'refine'):
                    if self._job_policy and 'conditioning' in self._job_policy:
                        result.asset['provenance']['calibration_conditioning'] = self._job_policy['conditioning']
                    summary = fit_summary_text(result)  # before adopting: a failure leaves no fit
                    self.fit = result  # Start from keeps the person's choice (maintainer, 2026-09-29)
                    self._checklist_fit_outdated = False
                    self._fit_source = self._pending_fit_source
                    self._fit_kind = kind
                    self._fit_instrument = result.asset['provenance'].get('instrument_geometry')
                    self._fit_setup = self._setup_state()
                    self.fit_summary.setPlainText(summary)
                    self.save_button.setEnabled(True)
                    self._save_session_quietly(self._job_policy)  # the policy it took; never unadopts it
                    self._show_calibration_fit()
                elif kind == 'stitch_preview':
                    window = self._calibration_fit_window
                    if (not self._close_pending and window is not None and window.isVisible()
                            and self._preview_key == self._preview_inputs()):
                        window.show_preview(result)
                    self._preview_key = None
                elif kind == 'save':  # the ring fit, or the direct-beam geometry (_save_rough)
                    if self._close_after_save:
                        from xrd_tools.integrate.stitch_calibration import paired_xu_calibration
                        paired_xu_calibration(result.pyfai_path)
                    from xrd_tools.integrate.calibration_seed import calibration_file_identity
                    self._loaded_calibration_identity = calibration_file_identity(result.pyfai_path)
                    provenance = self._saving.asset['provenance']
                    if self._saving is self.fit and provenance.get('calibration_kind') != 'direct_beam_not_ring_fitted':
                        self._saved_ring_fit = self.fit
                    self.saved_source_motors = provenance['source_motors']
                    self.saved_instrument_geometry = provenance.get('instrument_geometry')
                    self.saved_image_rotation = provenance['image_rotation']
                    self.saved_image_flip = provenance.get('image_flip', False)
                    self.saved_mask = self._text('mask')  # becomes Stitch's Mask File when set
                    self._saved_setup = self._setup_state()
                    self._loaded_calibration_kind = provenance.get('calibration_kind')
                    from xdart.utils.browse import remember_browse_path
                    remember_browse_path(Path(result.pyfai_path))
                    self.fit_summary.appendPlainText('\n' + saved_pair_text(result))
                elif kind == 'distance':
                    result, self._distance_energy_eV, self._distance_held = result
                    self._distance_result, self._distance_key = result, self._distance_pending_key
                    self._distance_warning = self._direct_beam_motion_warning()
                    rows = [f'⚠ {self._distance_warning}'] if self._distance_warning else []
                    rows += [f'Distance: {result.distance_mm:.6g} mm; regression SE: {result.distance_std_m * 1000:.3g} mm',
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
                elif kind == 'save':  # a direct-beam geometry saved against its scan's motion still says so
                    advice = (self._distance_warning
                              if self._loaded_calibration_kind == 'direct_beam_not_ring_fitted' else None)
                    self.status.setText(saved_pair_text(result) if advice is None
                                        else f'⚠ {advice}\n{saved_pair_text(result)}')
                    self.calibrationSaved.emit(result.pyfai_path, 'multigeometry')  # once, with both files written
                    if self._close_after_save:
                        self._close_after_save = False
                        self.close()
                elif kind == 'beam_detect':
                    self._update_beam_readiness()
                    self._render_direct_beam_advice()  # inputs restored: the fit, and its advice, apply again
                elif kind == 'distance':
                    self.status.setText(direct_beam_fit_text(self._distance_result, self._distance_energy_eV,
                                                             self._distance_held, self._distance_warning))
                elif kind in ('fit', 'refine'):  # what it found, as the direct-beam fit says its distance
                    self.status.setText(calibration_fit_text(result, kind))
                elif kind == 'stitch_preview':
                    self._job_status('Calibrant preview complete.')
                else:
                    self.status.setText('No PONI was saved; the frame remains unchanged.' if kind == 'calibrate' and result is None else f'{kind.capitalize()} complete.')
            except Exception as error:
                self._close_after_save = False
                self._job_status(str(error))
        elif terminal.status is OperationTerminalStatus.CANCELLED:
            self._close_after_save = False
            self._job_status('Cancelled. Previously completed frames and saved files are retained.')
        else:
            self._close_after_save = False
            self._job_status(terminal.failure_message)
        if kind == 'orientation' and self.orientation_panel is not None:
            if terminal.status is OperationTerminalStatus.CANCELLED:
                self.orientation_panel.finding_pane.set_text('Cancelled. No finding was adopted.')
            elif terminal.status is not OperationTerminalStatus.RETURNED:
                self.orientation_panel.finding_pane.set_text(terminal.failure_message)
            elif self._orientation_pending_key != self._orientation_key() and not self._close_pending:
                self.orientation_panel.finding_pane.set_text('Inputs changed during the check; run again. No stale finding was adopted.')
        if kind == 'beam_detect':
            if (terminal.status is not OperationTerminalStatus.RETURNED
                    and self._beam_probe_generation == self._beam_generation):
                self._beam_problem = terminal.failure_message or 'Direct-beam detection cancelled'
            self._update_beam_readiness()
            if (terminal.status is OperationTerminalStatus.RETURNED
                    and self._beam_detection is not None
                    and self._beam_detection[0] == self._beam_generation):
                self._tab_readiness()
        elif self._beam_detection is None and not self._close_pending:
            self._beam_probe_timer.start()
        if self._pending_source_details is not None and not self._close_pending:
            prefix, self._pending_source_details = self._pending_source_details, None
            self._source_details(prefix)
        self._refresh_checklist()
        # after the queued refresh and the direct-beam check have their turn, so Refine never starts with one queued
        self._quick_after_terminal(kind, terminal.status)
        self._sync_quick_button()
        if self._close_pending:
            self._close_pending = False
            self.close()

    def _set_tabs_enabled(self, enabled):
        if hasattr(self, 'checklist_table'):
            self.checklist_table.setEnabled(enabled)
        # Preserve the science/editor lock while the corner help action stays available.
        self.tabs.tabBar().setEnabled(enabled)
        for key, page in self._tab_pages.items():
            page.setEnabled(enabled or key == 'orientation')

    def closeEvent(self, event):
        if (not self._owner.busy and self._saved_setup is not None
                and self._setup_state() != self._saved_setup):
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle('Unsaved calibration setup')
            box.setText('Motor directions or detector orientation changed. RSM and Stitch still use the saved calibration.')
            can_save = self.fit is not None or self._loaded_calibration_kind == 'direct_beam_not_ring_fitted'
            box.setInformativeText('Save a new calibration pair and select it for this page, discard these setup edits, '
                'or keep editing.' if can_save else 'The previous fit is invalidated. Fit the changed setup before '
                'saving a new calibration pair, or discard these setup edits.')
            save = box.addButton('Save new calibration…', QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            save.setEnabled(can_save)
            save.setToolTip('Save a new valid pyFAI/xrayutilities pair; preserve the original calibration files.')
            discard = box.addButton('Discard setup changes', QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
            discard.setToolTip('Restore the selected saved calibration’s instrument and detector setup, then close.')
            cancel = box.addButton('Cancel', QtWidgets.QMessageBox.ButtonRole.RejectRole)
            cancel.setToolTip('Keep this window open to inspect or fit the edited setup.')
            box.setDefaultButton(cancel)
            box.exec()
            if box.clickedButton() is save:
                event.ignore()
                self._save_setup_on_close()
                return
            if box.clickedButton() is not discard:
                event.ignore()
                return
            path = self._loaded_calibration_identity[0]
            self._loaded_calibration_identity = None
            self.load_calibration(path)
            if self._setup_state() != self._saved_setup:
                event.ignore()
                return
        self._beam_probe_timer.stop()
        if self._owner.busy:
            self._close_pending = True
            self._cancel()
            event.ignore()
        else:
            self._timer.stop()
            self._save_session_quietly()
            self._help_owner.shutdown()
            event.accept()

    def _save_setup_on_close(self):
        """Use the existing pair writer, requiring a new filename and a current fit/rough geometry."""
        def go():
            fit = self.fit
            if fit is None:
                if self._loaded_calibration_kind != 'direct_beam_not_ring_fitted':
                    raise ValueError('fit the changed setup before saving')
                fit = self._rough_fit()
            selected = self._save_path(new_only=True)
            if selected is not None:
                self._saving = fit
                self._close_after_save = True
                self._start('save', (fit, selected, pair_validation_angles(
                    fit, scan_angles=self._scan_angles(fit)), False))
        self._guard(go)

    def _save_session_quietly(self, policy=None):
        """After a Fit or Refine (the policy it took) and at close (the policy last shown; combined Stitch
        review F2): a failure to write the session never takes back a fit, keeps the window open or the
        page from closing."""
        try:
            self._save_session(policy)
        except Exception:  # noqa: BLE001 - a close-time record; the log keeps what went wrong
            _LOG.warning('Could not record the calibration session', exc_info=True)

    def active(self):
        """Whether host navigation must retain this window's worker."""
        return self._owner.busy

    def shutdown(self):
        """Host Exit: cancel and return the existing retryable close receipt."""
        from xdart.gui.pages.values import PageCleanup
        self._quick_stage = self._quick_kind = None
        self._beam_probe_timer.stop()
        self._timer.stop()
        receipt = self._owner.close()
        if receipt.status is PageCleanup.CLEAN:
            self._save_session_quietly()
            self._help_owner.shutdown()
            self.hide()
        return receipt
