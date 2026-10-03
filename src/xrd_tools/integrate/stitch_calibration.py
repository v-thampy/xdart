"""Guided powder calibration values and notebook-derived fitting adapters.

Frames are decoded once through the registered source owner and retained as
exact TIFF inputs. The manifest records the original locator, metadata and
image/PONI/NPT hashes; fitting never guesses a filename-to-angle association.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from contextlib import contextmanager, ExitStack
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
    #: the staged image is also mirrored left-right (after ``rotation``): the detector camera's
    #: transform into Stitch's frame (``camera.stitch_orientation``); saved only when True
    flip: bool = False

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


def calibration_source_energy_eV(spec: SourceSpec) -> float | None:
    """A scan's common recorded energy, without reading detector pixels.

    SPEC energy columns/positioners are eV; its explicit wavelength_m/wavelength_A keys declare
    their units. NeXus uses the existing metadata reader's keV/angstrom projection. A missing,
    invalid or varying energy is not a default energy, and an absent SPEC selector never falls back.
    """
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    if spec.kind is SourceKind.SPEC:
        from xrd_tools.io.spec import list_spec_scans, read_spec_scan_columns
        key = str(spec.options.get('scan', ''))
        keys = list_spec_scans(spec.uri)
        if key not in keys:
            key += '.1'
        if key not in keys:
            raise ValueError('select an existing SPEC scan explicitly')
        columns, fixed, _ = read_spec_scan_columns(spec.uri, key)
        metadata = {name.casefold(): np.asarray(value, dtype=float) for name, value in (*fixed, *columns)}
        values = metadata.get('energy')
        if values is None:
            for name, factor in (('wavelength_m', 1.), ('wavelength_a', 1e-10)):
                if name in metadata:
                    values = wavelength_m_to_energy_eV(metadata[name] * factor)
                    break
        if values is None:
            return None
        values = np.asarray(values, dtype=float).reshape(-1)
        if not len(values) or not np.all(np.isfinite(values) & (values > 0)):
            return None
        return float(values[0]) if np.all(values == values[0]) else None
    from xrd_tools.io.nexus import read_nexus
    metadata = read_nexus(spec.uri, spec.entry or 'entry', motor_names=['energy'], counter_names=[])
    values = np.asarray(metadata.angles.get('energy', ()), dtype=float)
    if values.size:
        return (float(values.flat[0]) if np.all(np.isfinite(values) & (values > 0))
                and np.all(values == values.flat[0]) else None)
    if math.isfinite(metadata.energy) and metadata.energy > 0:
        return float(metadata.energy) * 1000
    if math.isfinite(metadata.wavelength) and metadata.wavelength > 0:
        return float(wavelength_m_to_energy_eV(metadata.wavelength * 1e-10))
    return None


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
        if not source.capabilities.has_raw_references:
            raise ValueError('No detector images found; check the images folder and stem for the selected scan')
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


#: A held circle's readings within this many degrees of each other (encoder noise) are one pose.
_HELD_TOLERANCE_DEG = 1e-3


def direct_beam_pose(spec: SourceSpec, motor: str, held=()) -> tuple[float | None, dict[str, float]]:
    """A direct-beam scan's recorded energy (eV) and its other moving circles as it held them, from its
    metadata, for every source kind the direct-beam fit reads (a SPEC image series or image container,
    a NeXus stack): the one energy the scan records (None when it records none, or several), and each
    ``held`` circle's reading. Kept apart from the fit, which uses the images alone (Codex testing-fixes
    review F2). A circle the scan does not record is left out (it reads its zero); one recorded for
    only some frames, not a number, or moving during the scan is refused, never read as zero."""
    if spec.kind is SourceKind.SPEC:
        from xrd_tools.sources import open_source
        rows = source_frame_choices(spec, (motor,))  # the scan's admission and each frame's energy
        count, energies = len(rows), [energy for _index, _motors, energy in rows]
        seen: dict[str, list] = {name: [] for name in held}
        if held:
            source = open_source(_exact_source(spec))
            try:
                with _metadata_reader(source) as read_metadata:
                    for index in source.frame_indices:
                        metadata = read_metadata(index)
                        for name in held:
                            if name in metadata:
                                seen[name].append(metadata[name])
            finally:
                close = getattr(source, 'close', None)
                if close is not None:
                    close()
    else:  # as open_direct_beam_scan reads a NeXus stack's angles
        from xrd_tools.io.nexus import read_nexus
        metadata = read_nexus(spec.uri, spec.entry or 'entry', motor_names=[motor, *held, 'energy'],
                              counter_names=[])
        if motor not in metadata.angles:
            raise ValueError(f'NeXus has no {motor!r} angle column')
        count = len(metadata.angles[motor])
        seen = {name: list(metadata.angles[name]) for name in held if name in metadata.angles}
        energies = list(metadata.angles.get('energy', ()))
        if not energies and metadata.energy and math.isfinite(metadata.energy) and metadata.energy > 0:
            energies = [float(metadata.energy) * 1000.0]  # its instrument energy, in keV
    readings: dict[str, float] = {}
    for name, values in seen.items():
        if not values:
            continue
        try:
            values = [float(value) for value in values]
        except (TypeError, ValueError):
            raise ValueError(f'the direct-beam scan\'s {name} readings are not numbers') from None
        if len(values) != count or not all(math.isfinite(value) for value in values):
            raise ValueError(f'the direct-beam scan records {name} for only some frames, or not as a number')
        if max(values) - min(values) > _HELD_TOLERANCE_DEG:
            raise ValueError(f'{name} moves during the direct-beam scan ({min(values):g} to {max(values):g}°): '
                             f'the scan should move {motor} alone')
        readings[name] = float(np.mean(values))
    energies = {float(value) for value in energies if value is not None}
    energies = {value for value in energies if math.isfinite(value) and value > 0}
    return (energies.pop() if len(energies) == 1 else None), readings


def direct_beam_scan_motor(spec: SourceSpec, detector_motors) -> str:
    """One actual scanned detector axis, never a counter or a held motor."""
    if spec.kind is SourceKind.SPEC:
        from xrd_tools.io.spec import get_spec_scanned_axes
        key = str(spec.options.get('scan', ''))
        if '.' not in key:
            key += '.1'
        axes = get_spec_scanned_axes(spec.uri, key)
    else:
        from xrd_tools.io.nexus import read_nexus
        metadata = read_nexus(spec.uri, spec.entry or 'entry')
        axes = tuple(name for name, values in metadata.angles.items()
                     if name != 'energy' and np.asarray(values).size > 1
                     and np.isfinite(values).all() and np.ptp(values) > _HELD_TOLERANCE_DEG)
    if not axes:
        raise ValueError('No scanned motor in the selected direct-beam scan')
    if len(axes) != 1:
        raise ValueError('Ambiguous direct-beam scan: multiple scanned motors (' + ', '.join(axes) + ')')
    if axes[0] not in detector_motors:
        raise ValueError(f'Unsupported direct-beam motor {axes[0]}: select a detector arm scan')
    return axes[0]


@contextmanager
def open_calibration_beam_scan(spec: SourceSpec, motor: str):
    """Exact lazy SPEC/container association shared by detection and distance fitting."""
    from xrd_tools.sources import open_source
    from xrd_tools.integrate.direct_beam import open_direct_beam_scan
    with ExitStack() as resources:
        options = dict(motor=motor, entry=spec.entry or 'entry')
        if spec.kind is SourceKind.SPEC:
            options['scan'] = spec.options['scan']
            if spec.options.get('image_file'):
                options['image_file'] = spec.options['image_file']
            else:
                choices = source_frame_choices(spec, (motor,))
                source = open_source(_exact_source(spec))
                if getattr(source, 'close', None) is not None:
                    resources.callback(source.close)
                frames = (source.frame_for(index) for index, _, _ in choices)
                options['images'] = ((frame.source_path, frame.source_frame_index or 0)
                                    if frame.source_path is not None else None for frame in frames)
        yield resources.enter_context(open_direct_beam_scan(spec.uri, **options))


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
                               rotation=0, motor_names=('del', 'nu'), cancelled=None, arm=None, flip=False):
    """Persist the exact selected detector pixels and their source angles, turned by ``rotation`` and
    mirrored left-right by ``flip`` (the detector camera's transform into Stitch's frame).

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
            pixels = ImageOrientation(rotation=rotation, flip_horizontal=bool(flip)).apply(loader(index))
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
                int(rotation), dict(motors), energy, flip=bool(flip)))
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


def _session_row(frame):
    """A frame as the session saves it: an unmirrored frame as before (no 'flip')."""
    row = asdict(frame)
    if not row['flip']:
        del row['flip']
    return row


def calibration_frame_mask(frame, mask=None, *, conditioning=None, saturation=None):
    """Combine the calibration's static mask with its Int threshold band, in staged pixel coordinates.

    The staged image stays unchanged. None/off preserves the existing static mask without reading pixels.
    """
    if conditioning is None or conditioning.mode == 'off':
        return mask
    from xrd_tools.io.image import read_image
    from xrd_tools.integrate.stitch_conditioning import condition_stitch_frame
    image = read_image(frame.image_path)
    if tuple(image.shape) != tuple(frame.shape):
        raise ValueError('the staged calibration image changed shape')
    bad = np.isnan(condition_stitch_frame(image, conditioning, saturation))
    if mask is not None:
        if np.shape(mask) != bad.shape:
            raise ValueError('the calibration mask does not match the staged image')
        bad |= np.asarray(mask, dtype=bool)
    return bad


def save_calibration_session(directory, frames, *, arm=None, fit_policy=None):
    """``arm`` (del, nu motors; None for a held circle) makes it version 3: a resumed
    session then knows which circle a single arm's one motor is (vertical or horizontal).
    ``fit_policy`` ({'bounds': ArmFitBounds record, 'distance_seed_m', 'distance_bounds_m'}) makes it
    version 4: a resumed session fits and predicts from its own estimates and bounds (Codex review F6).
    With the energy choice ('refine_energy', 'energy_bound_eV'), the motor scales ('fit_motor_scales')
    and the Diffractometer the zeros are read by ('instrument_geometry') it is version 5: a resumed
    session fits with its own choices and circles (combined Stitch review F1, F3).
    With calibration-only ``conditioning`` it is version 6; older policies retain their version."""
    path = Path(directory) / 'calibration-session.json'
    payload = {'schema': 'xdart.stitch_calibration_session', 'version': 2,
               'frames': [_session_row(frame) for frame in frames]}
    if arm is not None:
        payload.update(version=3, arm={'del': arm[0], 'nu': arm[1]})
    if fit_policy is not None:
        if arm is None:
            raise ValueError('a session records its fit policy with its arm')
        policy = _session_policy(fit_policy)
        payload.update(version=6 if 'conditioning' in policy else 5 if 'refine_energy' in policy else 4,
                       fit_policy=policy)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)
    return path


_POLICY_V4 = frozenset({'bounds', 'distance_seed_m', 'distance_bounds_m'})
_POLICY_V5 = _POLICY_V4 | {'refine_energy', 'energy_bound_eV', 'fit_motor_scales', 'instrument_geometry'}
_POLICY_V6 = _POLICY_V5 | {'conditioning'}


def _session_policy(policy):
    """A session's fit policy, checked: an ArmFitBounds record and the distance seed and bounds; from
    version 5 also the energy choice and bound, the motor-scale choice and the instrument;
    version 6 also the calibration-only pixel conditioning."""
    from xrd_tools.integrate.refine import ArmFitBounds
    from xrd_tools.session.instrument import InstrumentGeometry
    if not isinstance(policy, dict) or set(policy) not in (_POLICY_V4, _POLICY_V5, _POLICY_V6):
        raise ValueError('unsupported calibration session fit policy')
    extra = {}
    if set(policy) in (_POLICY_V5, _POLICY_V6):
        energy = policy['energy_bound_eV']
        if (type(policy['refine_energy']) is not bool or type(policy['fit_motor_scales']) is not bool
                or energy is not None and (isinstance(energy, bool) or not isinstance(energy, (int, float))
                                           or not 0 < energy < math.inf)):
            raise ValueError('unsupported calibration session fit policy')
        extra = {'refine_energy': policy['refine_energy'],
                 'energy_bound_eV': None if energy is None else float(energy),
                 'fit_motor_scales': policy['fit_motor_scales'],
                 'instrument_geometry': InstrumentGeometry.from_record(policy['instrument_geometry']).to_record()}
    if 'conditioning' in policy:
        from xrd_tools.integrate.stitch_conditioning import StitchConditioning
        extra['conditioning'] = StitchConditioning(**policy['conditioning']).to_dict()
    bounds = ArmFitBounds.from_record(policy['bounds']).to_record()
    seed, limits = policy['distance_seed_m'], policy['distance_bounds_m']
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, float)) or not 0 < seed < math.inf):
        raise ValueError('unsupported calibration session fit policy')
    if limits is not None and (not isinstance(limits, (list, tuple)) or len(limits) != 2
                               or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in limits)):
        raise ValueError('unsupported calibration session fit policy')
    return {'bounds': bounds, 'distance_seed_m': None if seed is None else float(seed),
            'distance_bounds_m': None if limits is None else [float(v) for v in limits], **extra}


def load_calibration_session(path, *, with_arm=False, with_policy=False):
    """The session's frames; with ``with_arm``, also its arm (None before version 3); with
    ``with_policy``, also its fit policy (None before version 4; the energy, scale and instrument
    choices from version 5, calibration-only conditioning from version 6)."""
    path = Path(path)
    if path.stat().st_size > 1 << 20:
        raise ValueError('calibration session exceeds the size limit')
    value = json.loads(path.read_text())
    version = value.get('version')
    keys = ({'schema', 'version', 'frames'} | ({'arm'} if version in (3, 4, 5, 6) else set())
            | ({'fit_policy'} if version in (4, 5, 6) else set()))
    if set(value) != keys or value['schema'] != 'xdart.stitch_calibration_session' or version not in (1, 2, 3, 4, 5, 6):
        raise ValueError('unsupported calibration session')
    policy = _session_policy(value['fit_policy']) if version in (4, 5, 6) else None
    if policy is not None and (('refine_energy' in policy) != (version in (5, 6))
                               or ('conditioning' in policy) != (version == 6)):
        raise ValueError('unsupported calibration session fit policy')
    if not 1 <= len(value['frames']) <= 32:
        raise ValueError('invalid calibration frame count')
    frames = tuple(CalibrationFrame(**{**row, 'shape': tuple(row['shape'])}) for row in value['frames'])
    for frame in frames:
        # Legacy rows retain their files, but absent model/image binding means
        # they are incomplete until the user recalibrates or explicitly imports.
        validate_frame_files(frame, require_points=frame.complete)
    arm = _session_arm(value, frames) if version in (3, 4, 5, 6) else None
    if with_policy:
        return frames, arm, policy
    return (frames, arm) if with_arm else frames


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
    #: the fit's control points, for display only (never saved): ``RefineResult.point_residuals``
    #: and "frames", each control frame's scan index and circle readings
    point_residuals: dict = field(default_factory=dict, compare=False, repr=False)


def stitch_calibrant_preview(fit: CalibrationFit, spec: SourceSpec, *, mask=None, conditioning=None,
                            saturation=None, npt_rad=500, npt_azim=180, npt_1d=1500,
                            radial_range=None, cancelled=None, progress=None):
    """Stitch the complete calibrant scan at its fitted geometry, without writing a result.

    The existing streaming Stitch owner produces the map and profile in one pass. Images and
    masks are in the fitted camera frame; ``mask`` is already oriented. The source's recorded
    energy is checked against the calibration's recorded energy, while integration keeps the
    fitted wavelength. No monitor normalization is inferred. Standard Stitch solid-angle
    correction is applied; polarization is off.
    """
    from xrd_tools.analysis.plans import StitchCancelled, StitchPlan, run_stitch
    from xrd_tools.analysis.stitch_operation import _OrientedSource, _detector_mask
    from xrd_tools.core.geometry import Diffractometer
    from xrd_tools.core.energy import wavelength_m_to_energy_eV
    from xrd_tools.sources import open_source
    if cancelled is not None and cancelled.is_set():
        raise StitchCancelled('calibrant preview cancelled')
    if fit.backend != 'multigeometry':
        raise ValueError('calibrant preview requires a fitted pyFAI arm')
    record, provenance = fit.asset['calibration'], fit.asset['provenance']
    diff = Diffractometer.from_pyfai_goniometer(record, source_motors=provenance['source_motors'])
    expected_energy = record.get('source_energy_eV') or float(wavelength_m_to_energy_eV(diff.calibration.poni.wavelength))
    rows = source_frame_choices(spec, tuple(provenance['source_motors'].values()))
    if not rows:
        raise ValueError('calibrant source has no detector images')
    for index, _motors, energy in rows:
        if energy is not None and not math.isclose(energy, expected_energy, rel_tol=1e-3):
            raise ValueError(f'frame {index}: source energy and recorded calibration energy differ')
    geometric = _detector_mask(diff)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if geometric is not None and mask.shape != geometric.shape:
            raise ValueError('calibrant mask shape differs from the fitted detector')
        mask = mask if geometric is None else mask | geometric
    else:
        mask = geometric
    indices = tuple(row[0] for row in rows)
    plan = StitchPlan(diffractometer=diff, backend='multigeometry', streaming_multigeometry=True,
        mode='2d', stitch_profile=True, unit='q_A^-1', npt_rad_2d=npt_rad,
        npt_azim_2d=npt_azim, npt_1d=npt_1d, radial_range=radial_range,
        azimuth_range=(-180., 180.), mask=mask, conditioning=conditioning,
        saturation=None if saturation is None else {i: saturation for i in indices})
    source = open_source(_exact_source(spec))
    try:
        oriented = _OrientedSource(source, ImageOrientation(rotation=provenance.get('image_rotation', 0),
            flip_horizontal=provenance.get('image_flip', False)), 512 << 20)
        result = run_stitch(plan, oriented, frame_indices=indices, workers=1,
            collect_frame_records=False, cancel_token=cancelled, progress_callback=progress)
        result.provenance['calibrant_preview'] = {'frame_count': len(indices),
            'source': str(spec.uri), 'scan': spec.options.get('scan'),
            'label': spec.uri.name + (f" · scan {spec.options['scan']}" if 'scan' in spec.options else ''),
            'wavelength_m': diff.calibration.poni.wavelength, 'normalization': 'None',
            'solid_angle': True, 'polarization': False}
        return result
    finally:
        close = getattr(source, 'close', None)
        if close is not None:
            close()


def fit_point_residuals(result, controls, indices) -> dict:
    """``CalibrationFit.point_residuals`` from a refinement over ``controls``, the scan frames ``indices``."""
    if not result.point_residuals:
        return {}
    return {**result.point_residuals,
            'frames': tuple((int(index), {str(name): float(value) for name, value in control.motors.items()})
                            for index, control in zip(indices, controls))}


def _mg_detector_config(calibration, turned_from=None):
    """Use the existing Stitch JSON config without dropping detector geometry.

    pyFAI saves fixed detector pitches redundantly in PONI files. They can be
    omitted from a goniometer record only when the registered detector with
    the admitted orientation/binning reproduces the exact full configuration.
    Generic/custom pitches, shapes and sensor configurations stay unsupported,
    but for a registered model turned by a quarter with the images (``turned_from``,
    else the calibration's own; AUD-9 stage 2b item 8): its generic detector of the
    turned shape is kept exactly as the calibration window writes it.
    """
    turned = turned_from if turned_from is not None else getattr(calibration, 'turned_from', None)
    if turned is not None:
        from xrd_tools.core.geometry.stacked_arm import checked_turned_detector
        from xrd_tools.integrate.turned_detector import is_turned_window_record
        turned = checked_turned_detector(turned)
        if not is_turned_window_record(calibration.poni.detector, calibration.detector_config, turned):
            raise ValueError(f"the frames' detector is not the {turned['detector']} turned with the images "
                             f"(turned {turned['image_rotation']}°)")
        return dict(calibration.detector_config)
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
                    instrument_geometry=None, min_frames=3, energy_bound_eV=None, bounds=None,
                    turned_from=None):
    """Fit the matched rows and refuse unconverged, degenerate or poor fits.

    ``bounds`` (:class:`~xrd_tools.integrate.refine.ArmFitBounds`; None: its defaults) gives the
    moving circles' motor-zero estimates, the mount's and the freedom about them (maintainer,
    2026-09-29): every solve starts at the estimates and stays within ±2° of each zero and of the
    mount, ±0.05 mm of each sample displacement and ±5 % of the distance seed unless
    ``distance_bounds_m`` is given. The provenance's ``fit_bounds`` records the estimates (and
    where each zero came from) and each parameter's initial and fitted value, bounds and contact.

    ``energy_bound_eV`` also refines the energy within that many eV of the frames' recorded one
    (maintainer, 2026-09-29: "add a refine energy toggle … just one number for +-that number"):
    the saved calibration takes the refined wavelength and keeps the recorded energy as
    ``source_energy_eV``, which the scans' recorded energies are checked against.

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
    ``arm_model='auto'`` fits the extended arm whenever it applies
    (:func:`automatic_arm_reason`) and the five-parameter arm otherwise, or when the
    extension is refused; the message and provenance say which, and why.
    ``instrument_geometry`` (an ``InstrumentGeometry`` record) also gives the circles' senses:
    a detector circle turning opposite to psic's fits with its motor scale reversed.
    ``turned_from`` (AUD-9 stage 2b item 8): the model the frames' camera turns by a quarter into their frame
    (``TURNED_DETECTOR_KEY``'s value): their PONIs name it turned (pyFAI's generic detector of the turned shape,
    as the window seeds pyFAI with it), and the saved record carries it, so its gaps are rebuilt. Quarter-turned
    frames that pyFAI calibrated with the model as it stores its images are refused by name.
    """
    from xrd_tools.integrate.refine import ArmFitBounds
    from xrd_tools.session.instrument import InstrumentGeometry
    senses = ({} if instrument_geometry is None
              else InstrumentGeometry.from_record(dict(instrument_geometry)).arm_senses())
    bounds = (ArmFitBounds() if bounds is None else bounds).resolved(
        [circle for circle, motor in (('del', del_motor), ('nu', nu_motor)) if motor is not None])
    if arm_model == 'auto':
        options = dict(backend=backend, del_motor=del_motor, nu_motor=nu_motor, fit_scales=fit_scales,
                       distance_seed_m=distance_seed_m, distance_bounds_m=distance_bounds_m, max_rms_q=max_rms_q,
                       domain=domain, distance_provenance=distance_provenance, cancelled=cancelled,
                       validation_angles=validation_angles, progress=progress,
                       instrument_geometry=instrument_geometry, min_frames=min_frames,
                       energy_bound_eV=energy_bound_eV, bounds=bounds, turned_from=turned_from)
        reason = automatic_arm_reason(frames, del_motor=del_motor, nu_motor=nu_motor, fit_scales=fit_scales)
        if not reason:
            try:
                return fit_calibration(frames, arm_model='stacked_extended', **options)
            except ValueError as error:  # implausible or under-determined: the five parameters
                reason = f'the detector mount and sample offset were refused: {error}'
        fit = fit_calibration(frames, arm_model='stacked', **options)
        fit.asset['provenance'].update(arm_model='stacked', arm_model_reason=reason)
        return replace(fit, message=f'{fit.message}; five-parameter arm: {reason}')
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
    if arm_model == 'stacked_extended' and backend == 'xu_hist':
        return _fit_xu_extended(frames, del_motor=del_motor, nu_motor=nu_motor,
                                distance_seed_m=distance_seed_m, distance_bounds_m=distance_bounds_m,
                                max_rms_q=max_rms_q, domain=domain, distance_provenance=distance_provenance,
                                cancelled=cancelled, validation_angles=validation_angles, progress=progress,
                                instrument_geometry=instrument_geometry, energy_bound_eV=energy_bound_eV,
                                bounds=bounds)
    if backend == 'xu_hist' and frames and frames[0].flip:
        raise ValueError('the xrayutilities fit takes an unmirrored camera; fit with pyFAI MultiGeometry, '
                         'whose xrayutilities file takes a mirrored one')
    if backend == 'xu_hist' and -1 in senses.values():
        raise ValueError("the xrayutilities fit takes psic's circle senses; fit with pyFAI MultiGeometry, "
                         'whose xrayutilities file takes a reversed circle')
    if energy_bound_eV is not None and backend == 'xu_hist':
        raise ValueError('the energy is refined on the pyFAI arm (and carried to its xrayutilities '
                         'partner); fit with pyFAI MultiGeometry')
    loaded = [load_control_frame(frame) for frame in frames]
    controls, calibrants, calibrations = zip(*loaded)
    base = calibrations[0]
    for frame, cal, calibrant in zip(frames, calibrations, calibrants):
        if (frame.shape, frame.rotation, frame.flip) != (frames[0].shape, frames[0].rotation, frames[0].flip):
            raise ValueError('all calibration frames must share shape and orientation')
        if dict(cal.detector_config) != dict(base.detector_config) or cal.poni.detector != base.poni.detector:
            raise ValueError('calibration frames have different detector configurations')
        if not math.isclose(cal.poni.wavelength, base.poni.wavelength, rel_tol=1e-6) or not np.array_equal(calibrant.dspacing, calibrants[0].dspacing):
            raise ValueError('calibration frames must share wavelength and calibrant')
    if not math.isfinite(max_rms_q) or max_rms_q <= 0:
        raise ValueError('maximum RMS must be finite and positive')
    if cancelled is not None and cancelled.is_set():
        raise InterruptedError('calibration fit cancelled')
    energy = {}
    if backend == 'multigeometry':
        turned = _turned_frames_detector(frames, base, turned_from)
        exported_config = _mg_detector_config(base, turned_from=turned)
        result = refine_goniometer(frames[0].poni_path, controls, rot1_motor=nu_motor,
                                  rot2_motor=del_motor, calibrant=calibrants[0],
                                  fit_scales=fit_scales, distance_seed_m=distance_seed_m,
                                  distance_bounds_m=distance_bounds_m, cancelled=cancelled,
                                  model='stacked_del_on_nu' if arm_model == 'stacked'
                                  else 'stacked_del_on_nu_extended',
                                  beam_rotation=0.,  # the design geometry; |q| cannot measure it
                                  energy_bound_eV=energy_bound_eV, bounds=bounds, senses=senses)
        if not result.success or result.rms_q > max_rms_q:
            raise ValueError(result.message or f'fit RMS {result.rms_q:.5g} exceeds {max_rms_q:g} Å⁻¹')
        calibration = result.diffractometer.calibration
        if turned is not None:  # the model and its turn travel with the record (AUD-9 item 8)
            calibration = replace(calibration, turned_from=turned)
        asset = result.diffractometer.stacked_arm.to_pyfai_goniometer(
            calibration)
        asset['detector_config'] = exported_config
        energy = dict(result.energy_refinement)
        if energy:  # the refined wavelength; the recorded energy the scans are checked against
            asset['source_energy_eV'] = energy['recorded_eV']
        rms, condition, count = result.rms_q, result.condition_number, result.n_points
        residuals = fit_point_residuals(result, controls, [frame.index for frame in frames])
        arm = f'one detector arm, {held} held at zero' if single else 'del mounted on nu'
        message = (f'{arm[0].upper()}{arm[1:]}; frozen motor scales: ' + ', '.join(result.frozen_scales)
                   if result.frozen_scales else f'Converged; {arm}')
        if not fit_scales:
            message += '; motor scales fixed at degrees'
        if arm_model == 'stacked_extended':
            message += '; detector mount and sample offset refined'
        if energy:
            message += (f"; energy {energy['refined_eV']:.1f} ± {energy['sigma_eV']:.1f} eV (recorded "
                        f"{energy['recorded_eV']:.1f}" + (f", held at ±{energy['bound_eV']:g} eV)"
                                                           if energy['at_bound'] else ")"))
        message += bound_contact_note(result.fit_bounds)
    elif backend == 'xu_hist':
        asset, rms, condition, count = _fit_xu(frames, loaded, del_motor, nu_motor,
            distance_seed_m, distance_bounds_m, max_rms_q, domain, cancelled)
        residuals = {}
        message = 'Converged; fixed SURFACE camera x−, z+ and PSIC circle convention'
    else:
        raise ValueError('unknown calibration backend')
    provenance = {'frames': [_session_row(frame) for frame in frames], 'rms_q_A^-1': rms,
                  'condition_number': condition, 'n_points': count,
                  'distance_seed_m': distance_seed_m, 'distance_bounds_m': distance_bounds_m,
                  'fit_motor_scales': bool(fit_scales) if backend == 'multigeometry' else False,
                  'source_motors': ({name: motor for name, motor in (('nu_angle', nu_motor), ('del_angle', del_motor))
                                     if motor is not None}
                                    if backend == 'multigeometry' else {'nu': nu_motor, 'del': del_motor}),
                  'image_rotation': frames[0].rotation,
                  **({'image_flip': True} if frames[0].flip else {}),  # older provenance: not mirrored
                  'direct_beam': distance_provenance}
    if instrument_geometry is not None:  # which motor drives which circle, for RSM and later reads
        provenance['instrument_geometry'] = dict(instrument_geometry)
    # the estimates and freedom the fit took, and where each parameter ended (the xrayutilities
    # five-parameter fit takes none)
    provenance['fit_bounds'] = ({'estimates': bounds.to_record(), 'parameters': dict(result.fit_bounds)}
                                if backend == 'multigeometry' else None)
    if arm_model == 'stacked_extended' or (backend == 'multigeometry' and energy):
        provenance['arm_model'] = arm_model
        provenance['parameter_errors'] = dict(result.param_errors)
        provenance['parameter_correlations'] = dict(result.param_correlations)
    if backend == 'multigeometry' and energy:
        provenance['energy_refinement'] = energy
    if backend == 'xu_hist':
        seed = base.poni.dist if distance_seed_m is None else distance_seed_m
        provenance['effective_distance_bounds_m'] = distance_bounds_m or (max(1e-5, seed * .5), seed * 1.5)
    if cancelled is not None and cancelled.is_set():
        raise InterruptedError('calibration fit cancelled')
    # Provenance is a separate adjacent document for standard pyFAI files.
    return CalibrationFit(backend, {'calibration': asset, 'provenance': provenance}, rms, condition, count, message,
                          point_residuals=residuals)


def _turned_frames_detector(frames, base, turned_from):
    """The turned-detector record a fit of ``frames`` carries: ``turned_from`` when their camera turns them by a
    quarter (checked against their turn; their PONIs' detector is checked by ``_mg_detector_config``), else None.
    Quarter-turned frames that pyFAI calibrated with a model as it stores its images (its shape the frames'
    transposed: pyFAI-calib2 given the model, not the window's seed) are refused by name (AUD-9 item 8)."""
    from xrd_tools.core.geometry.stacked_arm import checked_turned_detector
    turned_from = checked_turned_detector(turned_from)
    orientation = ImageOrientation(rotation=frames[0].rotation, flip_horizontal=frames[0].flip)
    if not orientation.swaps_axes:
        return None
    if base.poni.detector != 'Detector':
        from pyFAI.detectors import detector_factory
        config = dict(base.detector_config)
        rows, columns = (int(value) for value in (detector_factory(base.poni.detector, config=config) if config
                                                  else detector_factory(base.poni.detector)).shape)
        if (rows, columns) != tuple(frames[0].shape) and (columns, rows) == tuple(frames[0].shape):
            raise ValueError(
                f'pyFAI calibrated these frames with the {base.poni.detector} as it stores its images '
                f'({columns} × {rows} px), not turned with them as the window shows them ({rows} × {columns} px): '
                'calibrate them from 2 · Rough geometry or a calibration file, which give pyFAI the turned detector')
        return None
    if turned_from is not None and (turned_from['image_rotation'], turned_from['image_flip']) != (
            orientation.rotation, orientation.flip_horizontal):
        raise ValueError("the turned detector's turn is not the frames' camera's")
    return turned_from


def bound_contact_note(table) -> str:
    """'; at a fit bound: …' naming the parameters a fit left at a bound ('' when none)."""
    names = [name for name, row in (table or {}).items() if row.get('at_bound')]
    return f"; at a fit bound: {', '.join(names)}" if names else ''


def automatic_arm_reason(frames, *, del_motor='del', nu_motor='nu', fit_scales=False) -> str:
    """Why the automatic arm model keeps five parameters ('' when the extension applies).

    The detector mount and sample offset need both circles, fixed motor scales and
    completed frames on at least three nu positions (maintainer decision 40).
    """
    if del_motor is None or nu_motor is None:
        return 'a single 2θ arm holds one circle at zero; the extension needs both'
    if fit_scales:
        return 'motor scales are fitted, which the extension does not take'
    rows = {round(float(frame.motors[nu_motor]), 6) for frame in frames if nu_motor in frame.motors}
    if len(rows) < 3:
        return f'the extension needs frames on at least 3 nu positions ({len(rows)} so far)'
    return ''


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
                # a single 2θ arm holds its other circle at zero
                nu = 0. if nu_motor is None else control.motors[nu_motor]
                delta = 0. if del_motor is None else control.motors[del_motor]
                qx, qy, qz = hx.Ang2Q.area(0., 0., 0., 0., nu - no, delta - do)
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
                      if motor is not None else [0., 0.] for key, motor in [('del', del_motor), ('nu', nu_motor)]}
    if domain is not None:  # a single 2θ arm's asset holds its other circle at zero
        domain = {**domain, **{circle: [0., 0.] for circle, motor in (('del', del_motor), ('nu', nu_motor))
                               if motor is None}}
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
                     instrument_geometry=None, energy_bound_eV=None, bounds=None):
    """The extended pyFAI arm, transferred to an XU SURFACE v3 asset."""
    from xrd_tools.integrate.xu_transfer import transfer_arm_to_xu
    extended = fit_calibration(frames, backend='multigeometry', del_motor=del_motor, nu_motor=nu_motor,
                               distance_seed_m=distance_seed_m, distance_bounds_m=distance_bounds_m,
                               max_rms_q=max_rms_q, distance_provenance=distance_provenance,
                               cancelled=cancelled, arm_model='stacked_extended',
                               instrument_geometry=instrument_geometry, energy_bound_eV=energy_bound_eV,
                               bounds=bounds)
    if validation_angles is None:
        limits = domain or {key: [min(f.motors[motor] for f in frames), max(f.motors[motor] for f in frames)]
                            for key, motor in (('del', del_motor), ('nu', nu_motor))}
        validation_angles = [(d, n) for d in np.linspace(*limits['del'], 7) for n in np.linspace(*limits['nu'], 7)]
    asset, transfer = transfer_arm_to_xu(
        extended.asset['calibration'], validation_angles=validation_angles,
        source_motors=(del_motor, nu_motor), image_rotation=frames[0].rotation, image_flip=frames[0].flip,
        cancelled=cancelled, progress=progress)
    provenance = {**extended.asset['provenance'], 'source_motors': {'nu': nu_motor, 'del': del_motor},
                  'xu_transfer': transfer, 'pyfai_extended_arm': extended.asset['calibration']}
    message = (f"{extended.message}; transferred to xrayutilities: max |Δq| {transfer['max_abs_dq_A^-1']:.2g} Å⁻¹ "
               f"over every pixel of {transfer['validation_frames']} frames (gate {transfer['gate_A^-1']:.0e})")
    return CalibrationFit('xu_hist', {'calibration': asset, 'provenance': provenance},
                          extended.rms_q, extended.condition_number, extended.n_points, message)


def _calibration_bytes(calibration) -> bytes:
    """The exact bytes a saved calibration file holds."""
    return json.dumps(calibration, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def save_calibration_fit(fit: CalibrationFit, path):
    """Validate the fit through its backend before saving the adopted asset."""
    from xrd_tools.core.geometry import Diffractometer
    path = Path(path).resolve()
    provenance = path.with_suffix(path.suffix + '.provenance.json')
    if path.exists() or provenance.exists():
        raise FileExistsError('choose a new calibration filename; existing files are preserved')
    calibration = fit.asset['calibration']
    raw = _calibration_bytes(calibration)
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
            image_rotation=fit.asset['provenance']['image_rotation'],
            image_flip=fit.asset['provenance'].get('image_flip', False)))
    with provenance.open('x') as stream:
        json.dump(fit.asset['provenance'], stream, indent=2, allow_nan=False)
    return str(path)


def xu_partner_path(path) -> Path:
    """Where a pyFAI calibration's xrayutilities partner is saved: ``<name>.xu.json`` beside it."""
    path = Path(path)
    return path.with_name(f'{path.stem}.xu{path.suffix or ".json"}')


def pair_validation_angles(fit: CalibrationFit, *, scan_angles=(), domain=None, grid=7):
    """The (del, nu) circle angles an xrayutilities partner is checked on, a held circle at 0.

    The calibrant frames the fit used, every scan frame whose angles are given (``scan_angles``,
    (del, nu) pairs) and a ``grid`` × ``grid`` grid over ``domain`` ({'del': [lo, hi], 'nu': [lo, hi]},
    the range the calibration is to serve; a circle left out spans the other angles). The partner
    is admitted only where it was checked.
    """
    motors = fit.asset['provenance']['source_motors']
    del_motor, nu_motor = motors.get('del_angle'), motors.get('nu_angle')

    def circle(values, motor):
        return 0.0 if motor is None else float(values[motor])
    angles = [(circle(frame['motors'], del_motor), circle(frame['motors'], nu_motor))
              for frame in fit.asset['provenance']['frames']]
    angles += [(0.0 if del_motor is None else float(d), 0.0 if nu_motor is None else float(n)) for d, n in scan_angles]
    if not angles and not domain:
        # A manually typed rough geometry has no scan coverage. Check its analytical transfer
        # near zero; use the run's explicit extrapolation option for unvalidated poses.
        angles = [(d, n) for d in ((-1., 0., 1.) if del_motor else (0.,))
                  for n in ((-1., 0., 1.) if nu_motor else (0.,))]
    if domain:
        steps = {}
        for column, (name, motor) in enumerate((('del', del_motor), ('nu', nu_motor))):
            # a circle with no range and no frames (a direct-beam calibration with nu at 0) is held at 0
            limits = domain.get(name) or ([min(a[column] for a in angles), max(a[column] for a in angles)]
                                          if angles else None)
            steps[name] = [0.0] if motor is None or limits is None else np.linspace(*map(float, limits), grid).tolist()
        angles += [(d, n) for d in steps['del'] for n in steps['nu']]
    return list(dict.fromkeys(angles))


@dataclass(frozen=True)
class SavedCalibration:
    """A saved pyFAI calibration and its xrayutilities partner, or why it has none."""
    pyfai_path: str
    xu_path: str | None
    xu_refusal: str = ''
    xu_transfer: dict | None = None


def save_calibration_pair(fit: CalibrationFit, path, *, validation_angles, cancelled=None,
                          progress=None) -> SavedCalibration:
    """Save a pyFAI fit and its xrayutilities partner beside it (``xu_partner_path``).

    The pyFAI file is saved first and stays saved when the transfer is refused (fitted
    motor scales, another detector orientation, parallax or a sensor model), fails its
    gates or is cancelled: ``xu_refusal`` says why. The partner's provenance records the
    pyFAI file's SHA-256, which binds the pair (``paired_xu_calibration``).
    """
    from xrd_tools.integrate.xu_transfer import XuTransferRefused, transfer_arm_to_xu
    if fit.backend != 'multigeometry':
        raise ValueError('a calibration pair starts from a pyFAI fit')
    partner = xu_partner_path(Path(path).resolve())
    if partner.exists() or partner.with_suffix(partner.suffix + '.provenance.json').exists():
        raise FileExistsError('choose a new calibration filename; existing files are preserved')
    saved = save_calibration_fit(fit, path)
    provenance = fit.asset['provenance']
    del_motor, nu_motor = (provenance['source_motors'].get(name) for name in ('del_angle', 'nu_angle'))
    try:
        asset, transfer = transfer_arm_to_xu(
            fit.asset['calibration'], validation_angles=validation_angles, source_motors=(del_motor, nu_motor),
            image_rotation=provenance['image_rotation'], image_flip=provenance.get('image_flip', False),
            cancelled=cancelled, progress=progress)
    except XuTransferRefused as refusal:
        return SavedCalibration(saved, None, f'no xrayutilities file: {refusal}')
    except InterruptedError:
        return SavedCalibration(saved, None, 'no xrayutilities file: cancelled; save again to make it')
    # both files' bytes: the pair is the pyFAI file and exactly this partner (Codex review F5)
    xu_provenance = {**provenance, 'source_motors': {'del': del_motor, 'nu': nu_motor}, 'xu_transfer': transfer,
                     'pyfai_calibration': {'file': Path(saved).name, 'sha256': _sha(saved)},
                     'xu_calibration': {'file': partner.name,
                                        'sha256': hashlib.sha256(_calibration_bytes(asset)).hexdigest()}}
    try:  # the pyFAI file is saved and adopted whatever happens to its partner
        xu_saved = save_calibration_fit(CalibrationFit('xu_hist', {'calibration': asset, 'provenance': xu_provenance},
                                                       fit.rms_q, fit.condition_number, fit.n_points, fit.message),
                                        partner)
    except (OSError, ValueError) as error:
        return SavedCalibration(saved, None, f'no xrayutilities file: it could not be written ({error})')
    return SavedCalibration(saved, xu_saved, '', transfer)


def paired_xu_calibration(path) -> Path:
    """The xrayutilities partner saved with this pyFAI calibration, checked against its bytes.

    Raises ValueError, saying what to do, when there is none or it was made from other bytes.
    """
    path = Path(path).resolve()
    partner = xu_partner_path(path)
    advice = ('save the calibration again from the Calibration window, which writes both files '
              '(a fit with motor scales, detector orientation 1/2/4 or a sensor model has no xrayutilities file)')
    try:
        provenance = json.loads(partner.with_suffix(partner.suffix + '.provenance.json').read_text(encoding='utf-8'))
        recorded = provenance['pyfai_calibration']['sha256']
        written = provenance['xu_calibration']['sha256']
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError(f'{path.name} has no xrayutilities partner ({partner.name}): {advice}') from None
    if not partner.is_file() or recorded != _sha(path):
        raise ValueError(f'{partner.name} was not made from this {path.name}: {advice}')
    if written != _sha(partner):
        raise ValueError(f'{partner.name} is not the xrayutilities file saved with {path.name}: {advice}')
    return partner


def xu_calibration_path(path, project_root=None):
    """What Stitch with xrayutilities reads for this Calibration: the file itself when it is an
    xrayutilities asset (an XU-only calibration), else the partner saved beside a pyFAI goniometer
    JSON (``paired_xu_calibration``, which says what to do when there is none). A relative path is
    the project's."""
    given = Path(path)
    file = given if given.is_absolute() or project_root is None else Path(project_root) / given
    try:
        if file.stat().st_size > 1 << 20:
            return given  # no calibration JSON: the XU admission refuses it with its reason
        record = json.loads(file.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return given
    if isinstance(record, dict) and 'trans_function' in record:  # a pyFAI goniometer JSON
        return paired_xu_calibration(file)
    return given
