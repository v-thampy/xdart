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
