"""Scientific selections shared by fit hosts and exported notebook recipes.

Selections keep native bin centres and linear intensity. No display Log, colour
map, current GUI source, or guessed uncertainty enters a fit.
"""
from dataclasses import replace
import json
import numpy as np

from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
from xrd_tools.core.containers import convert_radial_axis


def phase_fit_snapshot(snapshot: FitInputSnapshot) -> FitInputSnapshot:
    selection = dict(snapshot.key.selection)
    if len(snapshot.axes) != 1 or selection.get('meaning', 'radial') != 'radial':
        raise ValueError('PHASE_RADIAL_INTENSITY_REQUIRED: phase fitting needs radial intensity')
    if snapshot.wavelength_angstrom is None:
        raise ValueError('PHASE_WAVELENGTH_REQUIRED: the result has no recorded wavelength')
    axis = snapshot.axes[0]
    if axis.unit in {'A^-1', 'Å⁻¹', 'q_A^-1', '1/angstrom'}:
        q = axis.values
    elif axis.unit in {'2th_deg', '2th_rad'}:
        q = convert_radial_axis(axis.values, axis.unit, 'q_A^-1', snapshot.wavelength_angstrom)
    else:
        raise ValueError('PHASE_RADIAL_INTENSITY_REQUIRED: unsupported radial coordinate')
    return replace(snapshot, axes=(FitAxis('q', 'q_A^-1', q),))


def stitch_fit_snapshot(payload, key: FitInputKey, *, coordinate='Q', phase=False, diagnostic='Intensity'):
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind
    if diagnostic != 'Intensity':
        raise ValueError('FIT_INTENSITY_REQUIRED: coverage and normalization are diagnostics')
    cut = dict(payload.gi).get(coordinate)
    pattern = cut or (payload.profile if payload.kind is AnalysisArtifactKind.STITCH_2D else payload)
    (name, axis), = pattern.axes
    unit = dict(pattern.axis_units if cut is not None or pattern is not payload else payload.inspection.axis_units)[name]
    meaning = coordinate if cut is not None else 'radial'
    target = {'Q': 'q_A^-1', '2theta': '2th_deg'}.get(coordinate)
    if target is not None and unit != target:
        if payload.wavelength_A is None:
            raise ValueError('FIT_WAVELENGTH_REQUIRED: radial conversion needs the recorded wavelength')
        axis = convert_radial_axis(axis, unit, target, payload.wavelength_A)
        unit = target
    selection = dict(key.selection)
    selection.update(coordinate=coordinate, diagnostic=diagnostic, meaning=meaning)
    key = replace(key, selection=tuple(sorted(selection.items())))
    snapshot = FitInputSnapshot(key, (FitAxis(coordinate, unit or '', axis),), pattern.intensity,
                               wavelength_angstrom=payload.wavelength_A, sigma=pattern.sigma)
    return phase_fit_snapshot(snapshot) if phase else snapshot


def _rsm_volume(payload):
    from xrd_tools.rsm.coordinate_frame import rsm_coordinate_frame_from_axes
    from xrd_tools.rsm.volume import RSMVolume
    axes = payload.axes
    units = payload.inspection.axis_units
    frame = rsm_coordinate_frame_from_axes(tuple(n for n, _ in axes), tuple(u for _, u in units))
    return RSMVolume.from_axes(frame, axes, payload.intensity)


def rsm_lineout_snapshot(payload, key, *, axis, centre, width):
    from xrd_tools.rsm.lineout import axis_lineouts
    cut = axis_lineouts(_rsm_volume(payload), centre, width, axes=(axis,))[axis]
    return FitInputSnapshot(key, (FitAxis(axis, cut.unit or 'r.l.u.', cut.coordinates),), cut.intensity,
                            cut.available)


def rsm_region_snapshot(payload, key, *, bounds, plane=None, index=None):
    """A native box, or a native axis plane cropped by the same box. No volume copy."""
    names = tuple(n for n, _ in payload.axes)
    units = dict(payload.inspection.axis_units)
    values = payload.intensity
    axes = payload.axes
    if plane is not None:
        k = names.index(plane)
        if type(index) is not int or not 0 <= index < values.shape[k]:
            raise ValueError('FIT_SLICE_INDEX_INVALID')
        slices = [slice(None)] * 3
        slices[k] = index
        values = values[tuple(slices)]
        axes = tuple(item for i, item in enumerate(axes) if i != k)
    selectors, selected = [], []
    for name, coords in axes:
        low, high = dict(bounds)[name]
        found = np.flatnonzero((coords >= low) & (coords <= high))
        if found.size < 2:
            raise ValueError(f'FIT_REGION_EMPTY: {name} needs at least two bins')
        selector = slice(int(found[0]), int(found[-1]) + 1)
        selectors.append(selector)
        selected.append(FitAxis(name, units[name] or 'r.l.u.', coords[selector]))
    shape = tuple(axis.values.size for axis in selected)
    if np.prod(shape) > (100_000 if len(shape) == 3 else 1_000_000):
        raise ValueError('FIT_REGION_POINT_LIMIT: narrow the native ranges')
    return FitInputSnapshot(key, tuple(selected), values[tuple(selectors)])


def rsm_reciprocal_transform(payload):
    """Physical widths require a persisted metric; absence is explicit, never inferred."""
    from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame, rsm_coordinate_matrix
    volume = _rsm_volume(payload)
    if volume.coordinate_frame is RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU:
        return np.eye(3)
    provenance = json.loads(payload.provenance_json)
    recorded = provenance.get('geometry', {}).get('ub')
    matrices = ([recorded] if recorded is not None else
                [item.get('ub') for item in provenance.get('members', ()) if item.get('ub') is not None])
    if not matrices or any(not np.array_equal(matrices[0], matrix) for matrix in matrices[1:]):
        return None
    # RSM's persisted UB is in the Q convention consumed by rsm_coordinate_matrix.
    return rsm_coordinate_matrix(volume.coordinate_frame, matrices[0])


# ---- the fit windows' file Source (NEXT-128-WINDOW-POLISH item 11) -------------------------------------------------
# The maintainer, 2026-11-03: "I think it will be good to have an option to choose a separate source than the nexus
# file selected in the main window in the fit windows ... This also fits well with the export jupyter idea"; his
# "Stored rows; fit the substrate (Recommended)". A source frame is its stored 1-D row, as written: Int's χ slice,
# monitor normalization and Set BG are display-only and never applied here (a substrate is the fit's template
# background). One reader serves the GUI's Reload, Fit and Batch and an exported notebook (``SourceReference``
# "int_1d"), so they read the same values.

#: frames read per ``read_1d_rows`` call in a batch; one chunk is held at a time
_INT_ROW_CHUNK = 64
#: the radial units Int's plot converts between (``scientific_axes._present_radial_axis``)
_DISPLAY_UNITS = frozenset({'q_A^-1', '2th_deg'})
_MAX_XYE_BYTES = 64 * 1024 * 1024
_UNREAD = object()


def _source_refused(code: str, detail: str) -> ValueError:
    return ValueError(f'FIT_SOURCE_{code}: {detail}')


def _row_digest(mode, unit, axis, values, sigma, *, wavelength=None) -> str:
    """One scientific fit input: mode, unit, arrays and wavelength (including unconverted q rows)."""
    import hashlib
    digest = hashlib.sha256()
    parts = (json.dumps({'mode': mode, 'unit': unit, 'wavelength_angstrom': wavelength},
                       sort_keys=True).encode('utf-8'),
             np.ascontiguousarray(axis, dtype='<f8').tobytes(), np.ascontiguousarray(values, dtype='<f8').tobytes(),
             b'' if sigma is None else np.ascontiguousarray(sigma, dtype='<f8').tobytes())
    for part in parts:
        digest.update(len(part).to_bytes(8, 'big'))
        digest.update(part)
    return digest.hexdigest()


class IntResultRows:
    """An Int NeXus result's stored 1-D rows in one ``mode`` as fit inputs (``FitInputSnapshot``), read in chunks of
    ``chunk`` frames with one chunk held, so a batch holds a bounded number of rows whatever the file's length.

    Each input is the frame's row on its stored axis, with its σ when stored, at the file's recorded wavelength (the
    monochromator's, as Int's Browse reads it; None when not recorded). ``unit`` places the rows on a displayed unit:
    q (Å⁻¹) and 2θ (°) convert with ``convert_radial_axis`` at that wavelength, as Int's plot converts them; any other
    difference is refused. A missing file, mode or frame is refused by name (``FIT_SOURCE_...``). Blocking: hosts call
    it on their fit worker."""

    def __init__(self, path, *, mode, entry='entry', unit=None, chunk=_INT_ROW_CHUNK):
        from pathlib import Path
        self.path = Path(path)
        self.mode, self.entry, self.unit = str(mode), str(entry), None if unit is None else str(unit)
        self._chunk = max(1, int(chunk))
        self._labels = None
        self._rows = None
        self._wavelength = _UNREAD
        self._file_facts = None

    def _check_file(self):
        """One loader binds one file generation, including cached rows and its separately read wavelength."""
        try:
            facts = self.path.stat()
        except OSError as error:
            raise _source_refused('UNREADABLE', f'the source file cannot be read: {self.path} ({error})') from None
        current = (facts.st_dev, facts.st_ino, facts.st_size, facts.st_mtime_ns, facts.st_ctime_ns)
        if self._file_facts is None:
            self._file_facts = current
        elif current != self._file_facts:
            raise _source_refused('CHANGED', f'{self.path.name} changed while its fit inputs were being read; '
                                            'choose or reload the source again')

    def _open(self):
        from xrd_tools.io.frame_view import FrameViewReader
        self._check_file()
        if not self.path.is_file():
            raise _source_refused('UNREADABLE', f'the source file is missing: {self.path}')
        return FrameViewReader(self.path, entry=self.entry, include_thumbnail=False, resolve_source=False)

    def _unreadable(self, error) -> ValueError:
        return _source_refused('UNREADABLE', f'{self.path.name} is not a readable xdart NeXus result: {error}')

    def _inventory(self, reader) -> tuple[int, ...]:
        if self._labels is None:
            modes = tuple(reader.modes_1d())
            if self.mode not in modes:
                raise _source_refused('MODE_MISSING', f'{self.path.name} has no 1-D mode {self.mode!r}; its modes are '
                                                      f'{", ".join(modes) or "none"}')
            self._labels = reader.labels_1d(self.mode)
        return self._labels

    def labels(self) -> tuple[int, ...]:
        """The frames with a stored row in the mode, ascending."""
        self._check_file()
        if self._labels is None:
            try:
                with self._open() as reader:
                    self._inventory(reader)
                    self._check_file()
            except ValueError as error:
                if str(error).startswith('FIT_SOURCE_'):
                    raise
                raise self._unreadable(error) from None
            except (OSError, RuntimeError, KeyError, TypeError) as error:
                raise self._unreadable(error) from None
        return self._labels

    def wavelength_angstrom(self) -> float | None:
        """The file's recorded wavelength in Å (None: not recorded), read once."""
        self._check_file()
        if self._wavelength is _UNREAD:
            from xrd_tools.core.energy import wavelength_m_to_angstrom
            from xrd_tools.io.browse_presentation import read_browse_presentation
            try:
                presentation, _mask = read_browse_presentation(self.path, entry=self.entry)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                raise self._unreadable(error) from None
            value = presentation.get('wavelength_m')
            self._wavelength = None if value is None else wavelength_m_to_angstrom(value, allow_default_sentinel=True)
            self._check_file()
        return self._wavelength

    def _rows_with(self, frame: int):
        self._check_file()
        if self._rows is not None and frame in self._rows.labels:
            return self._rows
        from bisect import bisect_left
        self._rows = None  # the held chunk goes before the next is read
        while True:
            try:
                with self._open() as reader:
                    labels = self._inventory(reader)
                    index = bisect_left(labels, frame)
                    if index == len(labels) or labels[index] != frame:
                        raise _source_refused('FRAME_MISSING', f'{self.path.name} has no frame {frame} in 1-D mode '
                                                               f'{self.mode!r}')
                    rows = reader.read_1d_rows(tuple(labels[index:index + self._chunk]), modes=(self.mode,))
                    self._check_file()
                break
            except ValueError as error:
                # Only the reader's explicit byte preflight refusal shrinks a chunk. Its scientific and
                # single-row limits stay strict, and the rejected bundle/reader closes before another read.
                if str(error) == 'Frame 1-D result byte projection exceeds limit' and self._chunk > 1:
                    self._chunk = max(1, self._chunk // 2)
                    continue
                if str(error).startswith('FIT_SOURCE_'):
                    raise
                raise self._unreadable(error) from None
            except (OSError, RuntimeError, KeyError, TypeError) as error:
                raise self._unreadable(error) from None
        self._rows = next(mode for mode in rows.modes if mode.mode == self.mode)
        return self._rows

    def snapshot(self, frame) -> FitInputSnapshot:
        """Frame ``frame``'s stored row as a fit input (the class docstring)."""
        from xrd_tools.core import axis_from_unit
        if type(frame) is not int or frame < 0:
            raise _source_refused('INVALID', f'a source frame is a whole number, not {frame!r}')
        rows = self._rows_with(frame)
        values, sigma = rows.row(frame)
        axis = rows.axis
        x, label, unit = axis.values, axis.label, axis.unit
        wavelength = self.wavelength_angstrom()
        if self.unit is not None and self.unit != unit:
            if not {unit, self.unit} <= _DISPLAY_UNITS:
                raise _source_refused('AXIS_MISMATCH', f'{self.path.name} stores mode {self.mode!r} in {unit}, which '
                                                       f'does not convert to the displayed {self.unit}')
            if wavelength is None:
                raise _source_refused('WAVELENGTH_REQUIRED', f'{self.path.name} records no wavelength, which '
                                                             f'converting {unit} to {self.unit} needs')
            x = convert_radial_axis(np.asarray(x, dtype=float), unit, self.unit, wavelength)
            label, unit = axis_from_unit(self.unit, x).label, self.unit
        selection = (('meaning', 'radial'), ('frame', frame), ('mode', self.mode), ('entry', self.entry))
        if self.unit is not None:
            selection += (('unit', self.unit),)
        key = FitInputKey(f'{self.path.stem} frame {frame}', str(self.path),
                          _row_digest(self.mode, unit, x, values, sigma, wavelength=wavelength), 0, selection)
        snapshot = FitInputSnapshot(key, (FitAxis(label, unit, x),), values,
                                    wavelength_angstrom=wavelength, sigma=sigma)
        self._check_file()
        return snapshot


def int_frame_snapshot(path, frame, *, mode, entry='entry', unit=None) -> FitInputSnapshot:
    """One Int result frame's stored 1-D row as a fit input (``IntResultRows``, one frame read)."""
    return IntResultRows(path, mode=mode, entry=entry, unit=unit, chunk=1).snapshot(frame)


def xye_fit_snapshot(path, *, unit) -> FitInputSnapshot:
    """An .xye file's columns as a fit input on its chosen x ``unit`` (a file records none; its name only hints at it),
    with its σ column when present and no wavelength (``phase_fit_snapshot`` refuses it by name)."""
    import os
    from pathlib import Path
    from xrd_tools.core import axis_from_unit
    from xrd_tools.io.export import read_xye
    path = Path(path)
    try:
        size = os.stat(path).st_size
    except OSError as error:
        raise _source_refused('UNREADABLE', f'the source file cannot be read: {path} ({error.strerror or error})') \
            from None
    if size > _MAX_XYE_BYTES:
        raise _source_refused('UNREADABLE', f'{path.name} is over 64 MiB')
    try:
        x, y, sigma = read_xye(path, strict=True)
    except Exception as error:  # noqa: BLE001 - a source file is user input
        raise _source_refused('UNREADABLE', f'{path.name} is not a two- or three-column .xye table: {error}') from None
    x = np.asarray(x, dtype=float)
    key = FitInputKey(path.stem, str(path), _row_digest('xye', str(unit), x, y, sigma), 0,
                      (('meaning', 'radial'), ('unit', str(unit))))
    return FitInputSnapshot(key, (FitAxis(axis_from_unit(str(unit), x).label, str(unit), x),), y, sigma=sigma)
