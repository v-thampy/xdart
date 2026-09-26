"""Guided powder calibration values and notebook-derived fitting adapters.

Frames are decoded once through the registered source owner and retained as
exact TIFF inputs. The manifest records the original locator, metadata and
image/PONI/NPT hashes; fitting never guesses a filename-to-angle association.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.core.geometry import ImageOrientation
from xrd_tools.integrate.refine import ControlFrame, refine_goniometer


def _sha(path: str | Path, cap: int = 512 << 20) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError(f"invalid calibration file: {path}")
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def calibration_image_identity(image):
    """Shape and content of the actual calibration image, independent of file encoding.

    The guided input is a decoded float64 TIFF. Use canonical little-endian
    float64 pixels, including canonical NaNs, for the child/model comparison.
    Shape participates in the digest so a reshape cannot preserve identity.
    """
    values = np.asarray(image)
    if (values.ndim != 2 or not all(values.shape) or values.size > 1 << 25
            or values.dtype.kind not in 'iuf'):
        raise ValueError('calibration image must be a bounded real two-dimensional array')
    values = np.array(values, dtype='<f8', order='C', copy=True)
    values[np.isnan(values)] = np.nan
    shape = tuple(int(v) for v in values.shape)
    digest = hashlib.sha256(b'xdart-calibration-image-float64-v1\0')
    digest.update(np.asarray(shape, dtype='<u8').tobytes())
    digest.update(values.tobytes())
    return shape, digest.hexdigest()


def _staged_image_identity(frame):
    from xrd_tools.io.image import read_image
    shape, digest = calibration_image_identity(read_image(frame.image_path, preserve_dtype=True))
    if shape != frame.shape:
        raise ValueError(f'frame {frame.index}: staged image shape changed')
    return digest


@dataclass(frozen=True)
class CalibrationFrame:
    identity: str
    source_uri: str
    source_kind: str
    source_options: dict
    index: int
    raw_path: str
    raw_index: int | None
    image_path: str
    image_sha256: str
    shape: tuple[int, int]
    rotation: int
    motors: dict[str, float]
    energy_eV: float | None
    poni_path: str = ''
    poni_sha256: str = ''
    npt_path: str = ''
    npt_sha256: str = ''
    pairing_policy: str = ''
    control_image_sha256: str = ''

    @property
    def complete(self) -> bool:
        return bool(self.poni_path and self.npt_path and self.control_image_sha256
                    and self.pairing_policy in {'pyfai-model-image-v1', 'explicit-import-v1'})


@contextmanager
def _metadata_reader(source):
    # A NeXus source's presentation metadata_for excludes scanned motors.
    # The cursor provider is the existing complete-row authority.
    opener = getattr(source, 'open_cursor', None)
    if opener is not None:
        with opener() as cursor:
            yield cursor.metadata_provider().complete_metadata_for
    else:
        yield source.metadata_for


def _exact_source(spec):
    if spec.kind is SourceKind.SPEC:
        options = dict(spec.options)
        options['require_complete_images'] = True
        reader = dict(options.get('read_image_kwargs', {}))
        reader['rotation'] = 0
        options['read_image_kwargs'] = reader
        return replace(spec, options=options)
    return spec


def source_frame_choices(spec: SourceSpec, motor_names=('del', 'nu')):
    """Metadata-only enumeration; requested SPEC selectors never fall back."""
    from xrd_tools.sources import open_source
    from xrd_tools.io.spec import list_spec_scans
    if len(set(motor_names)) != len(motor_names) or any(not name for name in motor_names):
        raise ValueError('choose distinct, nonempty angle metadata columns')
    if spec.kind is SourceKind.SPEC:
        requested = str(spec.options.get('scan', ''))
        keys = list_spec_scans(spec.uri)
        if requested not in keys and f'{requested}.1' not in keys:
            raise ValueError('select an existing SPEC scan explicitly')
    spec = _exact_source(spec)
    source = open_source(spec)
    try:
        indices = source.frame_indices
        if len(indices) > 100000:
            raise ValueError('calibration scan exceeds the frame selection limit')
        if spec.kind is SourceKind.SPEC:
            from xrd_tools.io.spec import read_spec_scan_columns
            _, _, expected = read_spec_scan_columns(spec.uri, source.scan_key)
            if len(indices) != expected:
                raise ValueError('SPEC image count differs from metadata; exact frame association is required')
        rows = []
        with _metadata_reader(source) as read_metadata:
            for index in indices:
                metadata = read_metadata(index)
                try:
                    motors = {name: float(metadata[name]) for name in motor_names}
                except KeyError as error:
                    raise ValueError(f'frame {index}: required angle column {error.args[0]!r} is absent') from error
                if not all(math.isfinite(v) for v in motors.values()):
                    raise ValueError(f'frame {index}: diffraction angles must be finite')
                energy = metadata.get('energy')
                if energy is not None:
                    energy = float(energy)
                    if not math.isfinite(energy) or energy <= 0:
                        raise ValueError(f'frame {index}: invalid energy in eV')
                rows.append((int(index), motors, energy))
        return rows
    finally:
        close = getattr(source, 'close', None)
        if close is not None:
            close()


def scan_motor_names(spec: SourceSpec) -> tuple[str, ...]:
    """The scan's motor names (SPEC #O/#P positioners, NeXus positioners), for choosing
    which motor drives which circle; () when the source does not list them."""
    from xrd_tools.sources import open_source
    source = open_source(_exact_source(spec))
    try:
        motors = getattr(source, 'motors', None)
        if isinstance(motors, dict):
            return tuple(str(name) for name in motors)
        describe = getattr(source, 'container_descriptor', None)
        names = getattr(describe(), 'motor_names', None) if describe is not None else None
        return tuple(str(name) for name in names or ())
    finally:
        close = getattr(source, 'close', None)
        if close is not None:
            close()


def suggested_frame_indices(rows, count=5):
    """Greedy angular coverage suggestions; count is not fit sufficiency."""
    if not rows:
        return ()
    values = np.asarray([list(row[1].values()) for row in rows], dtype=float)
    spans = np.ptp(values, axis=0)
    values = (values - values.min(axis=0)) / np.where(spans > 0, spans, 1)
    selected = [0]
    while len(selected) < min(count, len(rows)):
        distance = np.min(np.sum((values[:, None] - values[selected]) ** 2, axis=2), axis=1)
        distance[selected] = -1
        selected.append(int(np.argmax(distance)))
    return tuple(rows[i][0] for i in sorted(selected))


def capture_calibration_frames(spec: SourceSpec, indices, directory, *,
                               rotation=0, motor_names=('del', 'nu'), cancelled=None, arm=None):
    """Persist the exact selected detector pixels and their source angles.

    ``arm`` is the detector arm's (del, nu) motors, ``None`` for a circle a single arm
    holds at zero; it names the angles read and is recorded with the session.
    """
    if arm is not None:
        motor_names = tuple(motor for motor in arm if motor is not None)
    from xrd_tools.sources import open_source
    from fabio.edfimage import EdfImage
    from xrd_tools.io.image import read_image
    spec = _exact_source(spec)
    graph = None
    if spec.kind in (SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER):
        from xrd_tools.sources.execution_graph import (
            qualify_source_execution_graph, requalify_source_execution_graph,
        )
        graph = qualify_source_execution_graph(spec,
            cancelled=None if cancelled is None else cancelled.is_set)
        spec = graph.execution_source
    source_state = Path(spec.uri).stat()
    choices = {row[0]: row for row in source_frame_choices(spec, motor_names)}
    indices = tuple(indices)
    if not 1 <= len(indices) <= 32 or len(set(indices)) != len(indices):
        raise ValueError('select 1–32 distinct calibration frames (3–5 suggested)')
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = open_source(spec)
    frames = []
    source_options = {'entry': spec.entry, 'options': dict(spec.options)}
    descriptor = getattr(source, 'container_descriptor', None)
    if descriptor is not None:
        source_options['dataset_path'] = descriptor().dataset_path
    try:
        for index in indices:
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError('calibration preparation cancelled')
            if index not in choices:
                raise ValueError(f'frame {index} is not in the selected scan')
            _, motors, energy = choices[index]
            with _metadata_reader(source) as read_metadata:
                metadata = read_metadata(index)
            if any(float(metadata[name]) != value for name, value in motors.items()):
                raise ValueError('source angle metadata changed during preparation')
            original = source.frame_for(index)
            before = Path(original.source_path or spec.uri).stat()
            loader = getattr(source, 'load_native_frame', source.load_frame)
            pixels = ImageOrientation(rotation=rotation).apply(loader(index))
            if graph is not None:
                requalify_source_execution_graph(spec, graph,
                    cancelled=None if cancelled is None else cancelled.is_set)
            after = Path(original.source_path or spec.uri).stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError('selected raw image changed while preparing calibration')
            if pixels.ndim != 2 or pixels.size > 1 << 25 or pixels.dtype.kind not in 'iuf':
                raise ValueError('calibration requires a bounded real two-dimensional image')
            row_dir = root / f'frame-{index}-{uuid.uuid4().hex[:12]}'
            row_dir.mkdir()
            # Named after its source image, so pyFAI's window and the saved PONI's Image
            # line name the real frame (with its number inside a multi-frame file).
            raw = Path(original.source_path or spec.uri)
            within = original.source_frame_index
            image = row_dir / f'{raw.stem}{f"_{within}" if within else ""}.edf'
            # float64 on disk (EDF; fabio's TIFF writer stores float32): pyFAI's ring
            # extraction converts to float32 "as a copy" and, given float32, fills the
            # detector gaps in the loaded image itself, which the save check refuses.
            EdfImage(data=np.asarray(pixels, dtype=np.float64)).write(str(image))
            staged = read_image(image, preserve_dtype=True)
            if staged.dtype != np.float64 or not np.array_equal(staged, pixels, equal_nan=True):
                raise ValueError('staged calibration pixels differ from the selected image')
            frames.append(CalibrationFrame(
                row_dir.name, str(Path(spec.uri).resolve()), spec.kind.value,
                json.loads(json.dumps(source_options, default=str)), int(index),
                str(Path(original.source_path or spec.uri).resolve()),
                original.source_frame_index, str(image), _sha(image), tuple(pixels.shape),
                int(rotation), dict(motors), energy))
            current = Path(spec.uri).stat()
            if (current.st_ino, current.st_size, current.st_mtime_ns) != (source_state.st_ino, source_state.st_size, source_state.st_mtime_ns):
                raise ValueError('scan metadata changed during preparation')
            save_calibration_session(root, frames, arm=arm)
    finally:
        close = getattr(source, 'close', None)
        if close is not None:
            close()
    save_calibration_session(root, frames, arm=arm)
    return tuple(frames)


def _session_arm(value, frames):
    """A version-3 session's arm (del, nu), checked against its frames' angles."""
    arm = value.get('arm')
    if (type(arm) is not dict or set(arm) != {'del', 'nu'}
            or any(motor is not None and (type(motor) is not str or not motor) for motor in arm.values())
            or arm['del'] is None and arm['nu'] is None or arm['del'] == arm['nu']):
        raise ValueError('invalid calibration session arm')
    if any(set(frame.motors) != {motor for motor in arm.values() if motor is not None} for frame in frames):
        raise ValueError('calibration session frames do not read its arm')
    return arm['del'], arm['nu']


def save_calibration_session(directory, frames, *, arm=None):
    """``arm`` (del, nu motors; None for a held circle) makes it version 3: a resumed
    session then knows which circle a single arm's one motor is (vertical or horizontal)."""
    path = Path(directory) / 'calibration-session.json'
    payload = {'schema': 'xdart.stitch_calibration_session', 'version': 2,
               'frames': [asdict(frame) for frame in frames]}
    if arm is not None:
        payload.update(version=3, arm={'del': arm[0], 'nu': arm[1]})
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)
    return path


def load_calibration_session(path, *, with_arm=False):
    """The session's frames; with ``with_arm``, also its arm (None before version 3)."""
    path = Path(path)
    if path.stat().st_size > 1 << 20:
        raise ValueError('calibration session exceeds the size limit')
    value = json.loads(path.read_text())
    keys = {'schema', 'version', 'frames'} | ({'arm'} if value.get('version') == 3 else set())
    if set(value) != keys or value['schema'] != 'xdart.stitch_calibration_session' or value['version'] not in (1, 2, 3):
        raise ValueError('unsupported calibration session')
    if not 1 <= len(value['frames']) <= 32:
        raise ValueError('invalid calibration frame count')
    frames = tuple(CalibrationFrame(**{**row, 'shape': tuple(row['shape'])}) for row in value['frames'])
    for frame in frames:
        # Legacy rows retain their files, but absent model/image binding means
        # they are incomplete until the user recalibrates or explicitly imports.
        validate_frame_files(frame, require_points=frame.complete)
    if not with_arm:
        return frames
    return frames, (_session_arm(value, frames) if value['version'] == 3 else None)


def validate_frame_files(frame, *, require_points=True):
    if _sha(frame.image_path) != frame.image_sha256:
        raise ValueError(f'frame {frame.index}: staged image changed')
    if require_points:
        if not frame.complete:
            raise ValueError(f'frame {frame.index}: matched PONI and NPT are required')
        if _staged_image_identity(frame) != frame.control_image_sha256:
            raise ValueError(f'frame {frame.index}: control points belong to a different calibration image')
        if _sha(frame.poni_path, 1 << 20) != frame.poni_sha256 or _sha(frame.npt_path, 4 << 20) != frame.npt_sha256:
            raise ValueError(f'frame {frame.index}: saved PONI or control points changed')


def load_control_frame(frame):
    from pyFAI.control_points import ControlPoints
    from xrd_tools.integrate.calibration import load_detector_calibration
    validate_frame_files(frame)
    calibration = load_detector_calibration(frame.poni_path)
    points = ControlPoints(frame.npt_path)
    values = [(y, x, points.get(lbl=label).ring)
              for label in points.get_labels() for y, x in points.get(lbl=label).points]
    if not values:
        raise ValueError(f'frame {frame.index}: no control points')
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values[:, 0] >= frame.shape[0]) or np.any(values[:, 1] >= frame.shape[1]):
        raise ValueError(f'frame {frame.index}: control points are outside the staged image')
    if not np.all(values[:, 2] == values[:, 2].astype(int)):
        raise ValueError('control point rings must be integer indices')
    wl = float(points.calibrant.wavelength or 0)
    if not math.isclose(wl, calibration.poni.wavelength, rel_tol=1e-6):
        raise ValueError(f'frame {frame.index}: PONI and NPT wavelengths differ')
    if frame.energy_eV is not None:
        from xrd_tools.core.energy import energy_eV_to_wavelength_m
        if not math.isclose(float(energy_eV_to_wavelength_m(frame.energy_eV)), wl, rel_tol=1e-3):
            raise ValueError(f'frame {frame.index}: source energy and calibration wavelength differ')
    control = ControlFrame(values[:, 0], values[:, 1], values[:, 2], frame.motors)
    if control.rings.max() >= len(points.calibrant.dspacing):
        raise ValueError('control point ring has no calibrant d-spacing')
    return control, points.calibrant, calibration


def adopt_calibration_pair(frame, poni_path, npt_path, *, model_image_sha256=None):
    """Import an explicitly matched pair, retaining independent local copies."""
    import shutil
    validate_frame_files(frame, require_points=False)
    image_sha = _staged_image_identity(frame)
    if model_image_sha256 is not None and model_image_sha256 != image_sha:
        raise ValueError('loaded pyFAI image differs from selected calibration frame')
    row = Path(frame.image_path).parent
    token = uuid.uuid4().hex
    poni = row / f'calibration-{token}.poni'
    points = row / f'calibration-{token}.npt'
    _sha(poni_path, 1 << 20); _sha(npt_path, 4 << 20)
    shutil.copyfile(poni_path, poni)
    shutil.copyfile(npt_path, points)
    result = replace(frame, poni_path=str(poni), poni_sha256=_sha(poni),
                     npt_path=str(points), npt_sha256=_sha(points),
                     pairing_policy='explicit-import-v1' if model_image_sha256 is None else 'pyfai-model-image-v1',
                     control_image_sha256=image_sha)
    load_control_frame(result)
    return result


@dataclass(frozen=True)
class CalibrationFit:
    backend: str
    asset: dict
    rms_q: float
    condition_number: float
    n_points: int
    message: str


def _mg_detector_config(calibration):
    """Use the existing Stitch JSON config without dropping detector geometry.

    pyFAI saves fixed detector pitches redundantly in PONI files. They can be
    omitted from a goniometer record only when the registered detector with
    the admitted orientation/binning reproduces the exact full configuration.
    Generic/custom pitches, shapes and sensor configurations stay unsupported.
    """
    config = dict(calibration.detector_config)
    admitted = {key: value for key, value in config.items() if key in {'orientation', 'binning'}}
    extra = config.keys() - admitted.keys()
    if not extra:
        return admitted
    if extra - {'pixel1', 'pixel2', 'max_shape'}:
        raise ValueError('MG Stitch JSON does not support this detector configuration')
    from pyFAI.detectors import detector_factory
    full = detector_factory(calibration.poni.detector, config=config)
    compact = detector_factory(calibration.poni.detector, config=admitted)
    if (type(full) is not type(compact) or full.shape != compact.shape
            or full.get_config() != compact.get_config()):
        raise ValueError('MG Stitch JSON requires the registered detector pitch and shape')
    return admitted


def fit_calibration(frames, *, backend='multigeometry', del_motor='del', nu_motor='nu',
                    fit_scales=False, distance_seed_m=None, distance_bounds_m=None,
                    max_rms_q=0.03, domain=None, distance_provenance=None, cancelled=None,
                    arm_model='stacked', validation_angles=None, progress=None,
                    instrument_geometry=None, min_frames=3):
    """Fit the matched rows and refuse unconverged, degenerate or poor fits.

    ``arm_model='stacked_extended'`` also fits the detector mount on the arm and
    the sample offset from the rotation centre; it needs frames from at least
    three nu positions and fixed motor scales. For the XU backend the extended
    pyFAI arm is fitted, then transferred to xrayutilities (asset version 3) and
    checked on every pixel of ``validation_angles`` ((del, nu) pairs; default: a
    grid over ``domain``, else over the control frames).
    ``nu_motor=None`` or ``del_motor=None`` is a single detector arm (a fourc or
    twoc 2θ): psic's arm with that circle held at zero (vertical: del moves;
    horizontal: nu moves).
    ``min_frames=1`` is for predicting the other frames' rings from those calibrated so
    far (calibration_seed.session_prediction); a saved calibration needs three.
    """
    if not max(1, int(min_frames)) <= len(frames) <= 32:
        raise ValueError('fit requires at least three completed frames spanning the angles'
                         if min_frames == 3 else f'fit requires at least {min_frames} completed frames')
    if arm_model not in ('stacked', 'stacked_extended'):
        raise ValueError('unknown arm model')
    if nu_motor is None and del_motor is None:
        raise ValueError('a detector arm needs at least one moving circle')
    held = 'nu' if nu_motor is None else 'del' if del_motor is None else None
    single = held is not None
    if single and arm_model == 'stacked_extended':
        raise ValueError('a single detector arm has no nu positions; the detector mount and '
                         'sample offset refinement needs frames from several')
    if single and backend == 'xu_hist':
        raise ValueError('the xrayutilities backend fits psic\'s two detector circles; fit a single '
                         '2θ arm with the MultiGeometry backend')
    if arm_model == 'stacked_extended' and backend == 'xu_hist':
        return _fit_xu_extended(frames, del_motor=del_motor, nu_motor=nu_motor,
                                distance_seed_m=distance_seed_m, distance_bounds_m=distance_bounds_m,
                                max_rms_q=max_rms_q, domain=domain, distance_provenance=distance_provenance,
                                cancelled=cancelled, validation_angles=validation_angles, progress=progress,
                                instrument_geometry=instrument_geometry)
    loaded = [load_control_frame(frame) for frame in frames]
    controls, calibrants, calibrations = zip(*loaded)
    base = calibrations[0]
    for frame, cal, calibrant in zip(frames, calibrations, calibrants):
        if frame.shape != frames[0].shape or frame.rotation != frames[0].rotation:
            raise ValueError('all calibration frames must share shape and orientation')
        if dict(cal.detector_config) != dict(base.detector_config) or cal.poni.detector != base.poni.detector:
            raise ValueError('calibration frames have different detector configurations')
        if not math.isclose(cal.poni.wavelength, base.poni.wavelength, rel_tol=1e-6) or not np.array_equal(calibrant.dspacing, calibrants[0].dspacing):
            raise ValueError('calibration frames must share wavelength and calibrant')
    if not math.isfinite(max_rms_q) or max_rms_q <= 0:
        raise ValueError('maximum RMS must be finite and positive')
    if cancelled is not None and cancelled.is_set():
        raise InterruptedError('calibration fit cancelled')
    if backend == 'multigeometry':
        exported_config = _mg_detector_config(base)
        result = refine_goniometer(frames[0].poni_path, controls, rot1_motor=nu_motor,
                                  rot2_motor=del_motor, calibrant=calibrants[0],
                                  fit_scales=fit_scales, distance_seed_m=distance_seed_m,
                                  distance_bounds_m=distance_bounds_m, cancelled=cancelled,
                                  model='stacked_del_on_nu' if arm_model == 'stacked'
                                  else 'stacked_del_on_nu_extended')
        if not result.success or result.rms_q > max_rms_q:
            raise ValueError(result.message or f'fit RMS {result.rms_q:.5g} exceeds {max_rms_q:g} Å⁻¹')
        asset = result.diffractometer.stacked_arm.to_pyfai_goniometer(
            result.diffractometer.calibration)
        asset['detector_config'] = exported_config
        rms, condition, count = result.rms_q, result.condition_number, result.n_points
        arm = f'one detector arm, {held} held at zero' if single else 'del mounted on nu'
        message = (f'{arm[0].upper()}{arm[1:]}; frozen motor scales: ' + ', '.join(result.frozen_scales)
                   if result.frozen_scales else f'Converged; {arm}')
        if not fit_scales:
            message += '; motor scales fixed at degrees'
        if arm_model == 'stacked_extended':
            message += '; detector mount and sample offset refined'
    elif backend == 'xu_hist':
        asset, rms, condition, count = _fit_xu(frames, loaded, del_motor, nu_motor,
            distance_seed_m, distance_bounds_m, max_rms_q, domain, cancelled)
        message = 'Converged; fixed SURFACE camera x−, z+ and PSIC circle convention'
    else:
        raise ValueError('unknown calibration backend')
    provenance = {'frames': [asdict(frame) for frame in frames], 'rms_q_A^-1': rms,
                  'condition_number': condition, 'n_points': count,
                  'distance_seed_m': distance_seed_m, 'distance_bounds_m': distance_bounds_m,
                  'fit_motor_scales': bool(fit_scales) if backend == 'multigeometry' else False,
                  'source_motors': ({name: motor for name, motor in (('nu_angle', nu_motor), ('del_angle', del_motor))
                                     if motor is not None}
                                    if backend == 'multigeometry' else {'nu': nu_motor, 'del': del_motor}),
                  'image_rotation': frames[0].rotation,
                  'direct_beam': distance_provenance}
    if instrument_geometry is not None:  # which motor drives which circle, for RSM and later reads
        provenance['instrument_geometry'] = dict(instrument_geometry)
    if arm_model == 'stacked_extended':
        provenance['arm_model'] = arm_model
        provenance['parameter_errors'] = dict(result.param_errors)
        provenance['parameter_correlations'] = dict(result.param_correlations)
    if backend == 'xu_hist':
        seed = base.poni.dist if distance_seed_m is None else distance_seed_m
        provenance['effective_distance_bounds_m'] = distance_bounds_m or (max(1e-5, seed * .5), seed * 1.5)
    if cancelled is not None and cancelled.is_set():
        raise InterruptedError('calibration fit cancelled')
    # Provenance is a separate adjacent document for standard pyFAI files.
    return CalibrationFit(backend, {'calibration': asset, 'provenance': provenance}, rms, condition, count, message)


def _fit_xu(frames, loaded, del_motor, nu_motor, distance_seed_m, distance_bounds_m,
            max_rms_q, domain, cancelled):
    from scipy.optimize import least_squares
    from pyFAI.detectors import detector_factory
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    base = loaded[0][2]
    detector = detector_factory(base.poni.detector, config=dict(base.detector_config))
    if int(dict(base.detector_config).get('orientation', 3)) != 3:  # the SURFACE camera assumes it
        raise ValueError("the xrayutilities model takes pyFAI's default detector orientation (3); fit "
                         'this calibration with pyFAI MultiGeometry')
    if detector.sensor is not None or base.parallax is True or not detector.IS_FLAT or not detector.uniform_pixel:
        raise ValueError('authored SURFACE calibration requires a flat uniform detector without sensor/parallax')
    shape = frames[0].shape
    if tuple(detector.shape) != shape:
        raise ValueError('PONI detector shape differs from the selected oriented image')
    energy = float(wavelength_m_to_energy_eV(base.poni.wavelength))
    pitch1, pitch2 = float(detector.pixel1), float(detector.pixel2)
    seed_distance = base.poni.dist if distance_seed_m is None else float(distance_seed_m)
    bounds = (max(1e-5, seed_distance * .5), seed_distance * 1.5) if distance_bounds_m is None else tuple(distance_bounds_m)
    if not (0 < bounds[0] < bounds[1] and bounds[0] <= seed_distance <= bounds[1]):
        raise ValueError('distance bounds require 0 < low < high and an enclosed seed')
    seed = [base.poni.poni1 / pitch1, base.poni.poni2 / pitch2, seed_distance * 1e3, 0., 0.]
    # Notebook SURFACE cell 11: five parameters, fixed circle/camera convention,
    # physical bounds and direct ring-q residuals. Bounds scale with this panel.
    lower = [-shape[0], -shape[1], bounds[0] * 1e3, -5., -5.]
    upper = [2 * shape[0], 2 * shape[1], bounds[1] * 1e3, 5., 5.]
    with xu_runtime_session() as runtime:
        xu = runtime.xu
        def residual(params):
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError('calibration fit cancelled')
            c1, c2, dist, do, no = params
            qc = xu.QConversion(['x+', 'z-', 'y+', 'z-'], ['x+', 'z-'], [0, 1, 0])
            hx = xu.HXRD([0, 1, 0], [0, 0, 1], geometry='real', en=energy, qconv=qc)
            hx.Ang2Q.init_area('x-', 'z+', cch1=c1, cch2=c2, Nch1=shape[0], Nch2=shape[1],
                              pwidth1=pitch1 * 1e3, pwidth2=pitch2 * 1e3, distance=dist)
            out = []
            for control, calibrant, _ in loaded:
                qx, qy, qz = hx.Ang2Q.area(0., 0., 0., 0., control.motors[nu_motor] - no,
                                          control.motors[del_motor] - do)
                q = np.hypot(np.hypot(qx, qy), qz)
                reference = 2 * np.pi / np.asarray(calibrant.dspacing)[control.rings]
                out.append(q[control.rows, control.cols] - reference)
            return np.concatenate(out)
        result = least_squares(residual, seed, bounds=(lower, upper), method='trf',
                               max_nfev=300, xtol=1e-12, x_scale='jac')
        xu_wavelength = float(xu.en2lam(energy))
    rms = float(np.sqrt(np.mean(result.fun ** 2)))
    singular = np.linalg.svd(result.jac, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float('inf')
    if not result.success or not math.isfinite(condition) or condition > 1e8 or rms > max_rms_q:
        raise ValueError(f'XU fit refused: {result.message}; RMS {rms:.5g} Å⁻¹; condition {condition:.3g}')
    from xrd_tools.analysis.xu_stitch_calibration import authored_surface_calibration
    params = [float(v) for v in result.x]
    control_domain = {key: [min(f.motors[motor] for f in frames), max(f.motors[motor] for f in frames)]
                      for key, motor in [('del', del_motor), ('nu', nu_motor)]}
    asset = authored_surface_calibration(
        detector=detector, cch1=params[0], cch2=params[1], distance_m=params[2] / 1e3,
        del_offset_deg=params[3], nu_offset_deg=params[4], energy_eV=energy,
        xu_wavelength_A=xu_wavelength, source_motors={'del': del_motor, 'nu': nu_motor},
        rotation=frames[0].rotation, control_domain=control_domain,
        domain=domain or control_domain, rms_q=rms, condition=condition,
        n_frames=len(frames), n_points=len(result.fun))
    return asset, rms, condition, len(result.fun)


def _fit_xu_extended(frames, *, del_motor, nu_motor, distance_seed_m, distance_bounds_m, max_rms_q,
                     domain, distance_provenance, cancelled, validation_angles, progress,
                     instrument_geometry=None):
    """The extended pyFAI arm, transferred to an XU SURFACE v3 asset."""
    from xrd_tools.integrate.xu_transfer import transfer_extended_arm_to_xu
    extended = fit_calibration(frames, backend='multigeometry', del_motor=del_motor, nu_motor=nu_motor,
                               distance_seed_m=distance_seed_m, distance_bounds_m=distance_bounds_m,
                               max_rms_q=max_rms_q, distance_provenance=distance_provenance,
                               cancelled=cancelled, arm_model='stacked_extended',
                               instrument_geometry=instrument_geometry)
    if validation_angles is None:
        limits = domain or {key: [min(f.motors[motor] for f in frames), max(f.motors[motor] for f in frames)]
                            for key, motor in (('del', del_motor), ('nu', nu_motor))}
        validation_angles = [(d, n) for d in np.linspace(*limits['del'], 7) for n in np.linspace(*limits['nu'], 7)]
    asset, transfer = transfer_extended_arm_to_xu(
        extended.asset['calibration'], validation_angles=validation_angles,
        source_motors=(del_motor, nu_motor), image_rotation=frames[0].rotation,
        cancelled=cancelled, progress=progress)
    provenance = {**extended.asset['provenance'], 'source_motors': {'nu': nu_motor, 'del': del_motor},
                  'xu_transfer': transfer, 'pyfai_extended_arm': extended.asset['calibration']}
    message = (f"{extended.message}; transferred to xrayutilities: max |Δq| {transfer['max_abs_dq_A^-1']:.2g} Å⁻¹ "
               f"over every pixel of {transfer['validation_frames']} frames (gate {transfer['gate_A^-1']:.0e})")
    return CalibrationFit('xu_hist', {'calibration': asset, 'provenance': provenance},
                          extended.rms_q, extended.condition_number, extended.n_points, message)


def save_calibration_fit(fit: CalibrationFit, path):
    """Validate the fit through its backend before saving the adopted asset."""
    from xrd_tools.core.geometry import Diffractometer
    path = Path(path).resolve()
    provenance = path.with_suffix(path.suffix + '.provenance.json')
    if path.exists() or provenance.exists():
        raise FileExistsError('choose a new calibration filename; existing files are preserved')
    calibration = fit.asset['calibration']
    raw = json.dumps(calibration, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if fit.backend == 'xu_hist':
        from tempfile import TemporaryDirectory
        from xrd_tools.analysis.xu_stitch_calibration import (
            XuStitchCalibrationInput, capture_xu_stitch_calibration,
        )
        from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
        from xrd_tools.integrate.xu_stitch import resolve_xu_stitch_effective_geometry
        # The backend requires a file-backed receipt. Stage the exact bytes in
        # private scratch so refusal cannot leave an unusable destination asset.
        with TemporaryDirectory(prefix='xdart-xu-calibration-') as scratch:
            staged = Path(scratch).resolve() / 'calibration.json'
            staged.write_bytes(raw)
            receipt = capture_xu_stitch_calibration(
                XuStitchCalibrationInput(staged), project_root=staged.parent)
            with xu_runtime_session() as runtime:
                resolve_xu_stitch_effective_geometry(receipt, runtime)
    else:
        Diffractometer.from_pyfai_goniometer(calibration, source_motors={'nu_angle': 'nu', 'del_angle': 'del'})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
    if fit.backend == 'multigeometry':
        from xrd_tools.analysis.stitch_operation import (
            StitchGeometryInput, StitchGeometryKind, capture_stitch_geometry,
        )
        capture_stitch_geometry(StitchGeometryInput(path, StitchGeometryKind.PYFAI_GONIOMETER_JSON,
            source_motors=tuple(sorted(fit.asset['provenance']['source_motors'].items())),
            image_rotation=fit.asset['provenance']['image_rotation']))
    with provenance.open('x') as stream:
        json.dump(fit.asset['provenance'], stream, indent=2, allow_nan=False)
    return str(path)
