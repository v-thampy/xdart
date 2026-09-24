"""Automatic stacked-arm refinement from a calibrant scan (MultiGeometry Stitch).

The refinement starts from an existing stacked (del-on-nu) calibration. Ring
points are extracted from frames spread over the scan's angle grid with pyFAI's
own ring finder (``SingleGeometry.extract_cp``, the machinery pyFAI-calib2
uses), seeded by that calibration. Dropped frames are excluded with the shared
screen (:mod:`xrd_tools.integrate.frame_screen`): the beam monitor does not see a
frame whose detector counts collapsed.

The arm is refined with and without the detector-mount/sample-offset
extension. Leave-one-nu-row-out validation decides whether the extension is
kept: each nu row is predicted from a fit to the other rows, and the extension
must reduce the held-out error by ``min_improvement``. The geometry is fitted
on the calibrant only; never run this on sample data, whose peaks can move and
broaden with angle for physical reasons.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from xrd_tools.core.geometry import Diffractometer, ImageOrientation
from xrd_tools.integrate.frame_screen import FrameScreenPolicy, screen_totals
from xrd_tools.integrate.refine import ControlFrame, refine_stacked_arm
from xrd_tools.integrate.stitch_calibration import (
    CalibrationFit, _exact_source, _mg_detector_config, calibration_image_identity,
    source_frame_choices,
)

_SOURCE_MOTORS = (('del_angle', 'del'), ('nu_angle', 'nu'))
#: Calibration excludes partial frames as well as empty ones.
_CALIBRATION_SCREEN = FrameScreenPolicy(exclude='partial')


@dataclass(frozen=True)
class ScanRingPoints:
    """Automatically extracted control points and the frames behind them."""

    controls: tuple[ControlFrame, ...]
    frames: tuple[dict, ...]
    excluded: tuple[dict, ...]
    calibrant: object
    coverage: dict


def stratified_frame_indices(rows, count, nu_motor, del_motor):
    """Choose frames from every nu row first, then spread the rest over the angles.

    Each nu row gets an equal share of the budget, centred and evenly spaced in
    del; the remainder goes to the frames farthest from those already chosen.
    When the budget is smaller than the number of rows, evenly spaced rows get
    one frame each, and the returned coverage says how many rows were reached.
    """
    groups = {}
    for index, motors, _energy in rows:
        groups.setdefault(round(float(motors[nu_motor]), 6), []).append(
            (float(motors[del_motor]), int(index)))
    keys = sorted(groups)
    count = int(count)
    if count < 1 or not keys:
        raise ValueError('choose at least one frame from a nonempty scan')
    kept = keys if count >= len(keys) else sorted(
        {keys[int(i)] for i in np.round(np.linspace(0, len(keys) - 1, count))})
    share = max(1, count // len(kept))
    chosen = []
    for key in kept:
        group = sorted(groups[key])
        k = min(share, len(group))
        picks = np.unique(np.round((np.arange(k) + 0.5) * len(group) / k - 0.5).astype(int))
        chosen += [group[i][1] for i in picks]
    order = [int(row[0]) for row in rows]
    values = np.array([[float(row[1][del_motor]), float(row[1][nu_motor])] for row in rows])
    spans = np.ptp(values, axis=0)
    values = (values - values.min(axis=0)) / np.where(spans > 0, spans, 1.0)
    position = {index: k for k, index in enumerate(order)}
    distance = np.min(np.sum((values[:, None] - values[[position[i] for i in chosen]][None]) ** 2,
                             axis=2), axis=1)
    distance[[position[i] for i in chosen]] = -1.0
    while len(chosen) < min(count, len(order)):
        k = int(np.argmax(distance))
        chosen.append(order[k])
        distance = np.minimum(distance, np.sum((values - values[k]) ** 2, axis=1))
        distance[[position[i] for i in chosen]] = -1.0
    return tuple(sorted(chosen)), {'nu_rows_available': len(keys), 'nu_rows_selected': len(kept)}


def _seed(geometry, source_motors):
    if isinstance(geometry, Mapping):
        record = json.loads(json.dumps(geometry))
    else:
        record = json.loads(Path(geometry).read_text())
    diff = Diffractometer.from_pyfai_goniometer(record, source_motors=dict(source_motors))
    if diff.stacked_arm is None or diff.calibration is None:
        raise ValueError('automatic refinement starts from a stacked (del-on-nu) calibration')
    if diff.calibration.parallax is True:
        raise ValueError('automatic refinement does not yet support parallax-corrected calibrations')
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return record, diff, digest


def _calibrant(name, wavelength_m):
    from pyFAI.calibrant import get_calibrant  # noqa: PLC0415
    calibrant = get_calibrant(name) if isinstance(name, str) else name
    calibrant.wavelength = float(wavelength_m)
    return calibrant


def _ring_indices(calibrant, wavelength_m):
    """Map extract_cp ring numbers (unique 2-theta order) to d-spacing indices."""
    tth = np.unique(np.array([t for t in calibrant.get_2th() if t is not None], dtype=float))
    q_ring = 4e-10 * math.pi / wavelength_m * np.sin(tth / 2)
    q_d = 2 * math.pi / np.asarray(calibrant.dSpacing, dtype=float)
    index = np.array([int(np.argmin(np.abs(q_d - q))) for q in q_ring], dtype=int)
    if index.size and np.max(np.abs(q_d[index] - q_ring) / q_ring) > 1e-9:
        raise ValueError('calibrant ring order is inconsistent with its d-spacings')
    return index


def extract_scan_ring_points(spec, geometry, *, source_motors=_SOURCE_MOTORS, image_rotation=0,
                             calibrant='LaB6', frame_indices=None, max_frames=48,
                             points_per_degree=2.0, max_counts=None, frame_screen=_CALIBRATION_SCREEN,
                             cancelled=None) -> ScanRingPoints:
    """Extract calibrant ring points from frames across the scan's angle grid."""
    from pyFAI.goniometer import SingleGeometry  # noqa: PLC0415
    from xrd_tools.integrate.multi import create_multigeometry_integrator_series_from_geometry
    from xrd_tools.sources import open_source  # noqa: PLC0415

    _record, diff, _digest = _seed(geometry, source_motors)
    arm = diff.stacked_arm
    wavelength = float(diff.calibration.poni.wavelength)
    rows = source_frame_choices(spec, (arm.del_motor, arm.nu_motor))
    by_index = {row[0]: row for row in rows}
    if frame_indices is None:
        selected, coverage = stratified_frame_indices(
            rows, min(int(max_frames), len(rows)), arm.nu_motor, arm.del_motor)
    else:
        selected = tuple(int(i) for i in frame_indices)
        if len(set(selected)) != len(selected) or any(i not in by_index for i in selected):
            raise ValueError('choose distinct frames from the selected scan')
        coverage = {'nu_rows_available': len({round(row[1][arm.nu_motor], 6) for row in rows}),
                    'nu_rows_selected': len({round(by_index[i][1][arm.nu_motor], 6) for i in selected})}
    if not 3 <= len(selected) <= 256:
        raise ValueError('automatic refinement needs 3-256 frames')
    for index in selected:
        energy = by_index[index][2]
        if energy is not None:
            from xrd_tools.core.energy import energy_eV_to_wavelength_m  # noqa: PLC0415
            if not math.isclose(float(energy_eV_to_wavelength_m(energy)), wavelength, rel_tol=1e-3):
                raise ValueError(f'frame {index}: source energy and calibration wavelength differ')
    cal = _calibrant(calibrant, wavelength)
    ring_index = _ring_indices(cal, wavelength)
    motors = {arm.del_motor: np.array([by_index[i][1][arm.del_motor] for i in selected]),
              arm.nu_motor: np.array([by_index[i][1][arm.nu_motor] for i in selected])}
    orientation = ImageOrientation(rotation=image_rotation)
    exact = _exact_source(spec)
    source = open_source(exact)
    try:
        loader = getattr(source, 'load_native_frame', source.load_frame)

        def pixels(index, integrator):
            image = np.asarray(orientation.apply(loader(index)), dtype=np.float64)
            if image.shape != tuple(integrator.detector.shape):
                raise ValueError(f'frame {index}: image shape differs from the calibration detector')
            valid = np.isfinite(image) & (image >= 0)
            if integrator.detector.mask is not None:
                valid &= ~np.asarray(integrator.detector.mask, dtype=bool)
            if max_counts is not None:
                valid &= image <= float(max_counts)
            return image, valid

        integrators = list(create_multigeometry_integrator_series_from_geometry(diff, motors))
        totals = []
        for index, integrator in zip(selected, integrators):
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError('automatic refinement cancelled')
            image, valid = pixels(index, integrator)
            totals.append(float(image[valid].sum()))
        verdicts = screen_totals(totals, [int(i) for i in selected],
                                 np.round(motors[arm.nu_motor], 6).tolist(), frame_screen)
        excluded, keep = [], []
        for k, verdict in enumerate(verdicts):
            if verdict.excluded:
                excluded.append({'index': int(verdict.label), 'count_ratio': verdict.ratio,
                                 'reason': f'{verdict.status} frame: counts far below its nu-row neighbours'})
            else:
                keep.append(k)
        controls, frames = [], []
        for k in keep:
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError('automatic refinement cancelled')
            index, integrator = selected[k], integrators[k]
            image, valid = pixels(index, integrator)
            geometry_ = SingleGeometry(int(index), image=np.where(valid, image, 0.0), calibrant=cal,
                                       geometry=integrator)
            points = np.asarray(geometry_.extract_cp(pts_per_deg=float(points_per_degree)).getList(),
                                dtype=float).reshape(-1, 3)
            inside = ((points[:, 0] >= 0) & (points[:, 0] < image.shape[0])
                      & (points[:, 1] >= 0) & (points[:, 1] < image.shape[1]))
            points = points[inside]
            if not len(points):
                excluded.append({'index': int(index), 'reason': 'no ring points found', 'count_ratio': None})
                continue
            frame_motors = {arm.nu_motor: float(motors[arm.nu_motor][k]),
                            arm.del_motor: float(motors[arm.del_motor][k])}
            controls.append(ControlFrame(points[:, 0], points[:, 1],
                                         ring_index[points[:, 2].astype(int)], frame_motors))
            frames.append({'index': int(index), 'motors': frame_motors, 'points': int(len(points)),
                           'image_sha256': calibration_image_identity(image)[1]})
    finally:
        close = getattr(source, 'close', None)
        if close is not None:
            close()
    if len(controls) < 3:
        raise ValueError('automatic refinement found ring points on fewer than three frames')
    return ScanRingPoints(tuple(controls), tuple(frames), tuple(excluded), cal, coverage)


def _held_out(result, controls, calibrant, nu_motor, del_motor):
    """Residuals of a fitted arm on frames it did not see (sub-pixel positions)."""
    from xrd_tools.integrate.multi import create_multigeometry_integrator_series_from_geometry
    motors = {nu_motor: np.array([c.motors[nu_motor] for c in controls]),
              del_motor: np.array([c.motors[del_motor] for c in controls])}
    q_d = 2 * math.pi / np.asarray(calibrant.dSpacing, dtype=float)
    out = []
    for control, integrator in zip(controls, create_multigeometry_integrator_series_from_geometry(
            result.diffractometer, motors)):
        q = integrator.qFunction(control.row_positions, control.col_positions) / 10.0
        out.append(np.asarray(q, dtype=float) - q_d[control.rings])
    return np.concatenate(out)


def _robust_sigma(values):
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def refine_arm_from_scan(spec, geometry, *, source_motors=_SOURCE_MOTORS, image_rotation=0,
                         calibrant='LaB6', frame_indices=None, max_frames=48, points_per_degree=2.0,
                         max_counts=None, frame_screen=_CALIBRATION_SCREEN, reject_outliers=4.5,
                         min_improvement=0.2, max_rms_q=0.03, cancelled=None) -> CalibrationFit:
    """Refine a stacked-arm calibration from its calibrant scan, automatically.

    Returns a MultiGeometry :class:`CalibrationFit` for :func:`save_calibration_fit`.
    Its provenance records the extracted frames, the excluded frames and the
    held-out validation that chose between the five-parameter arm and the
    detector-mount/sample-offset extension.
    """
    record, diff, digest = _seed(geometry, source_motors)
    arm = diff.stacked_arm
    points = extract_scan_ring_points(
        spec, record, source_motors=source_motors, image_rotation=image_rotation,
        calibrant=calibrant, frame_indices=frame_indices, max_frames=max_frames,
        points_per_degree=points_per_degree, max_counts=max_counts,
        frame_screen=frame_screen, cancelled=cancelled)
    poni = diff.calibration.poni
    config = dict(diff.calibration.detector_config)

    def fit(controls, extended):
        return refine_stacked_arm(poni, controls, nu_motor=arm.nu_motor, del_motor=arm.del_motor,
                                  calibrant=points.calibrant, detector_config=config,
                                  extended=extended, reject_outliers=reject_outliers,
                                  beam_rotation=arm.beam_rotation, cancelled=cancelled)

    controls = points.controls
    rows = sorted({round(c.motors[arm.nu_motor], 6) for c in controls})
    validation = {'method': 'leave-one-nu-row-out', 'nu_rows': rows, 'min_improvement': min_improvement,
                  'coverage': dict(points.coverage)}
    chosen, reason = 'stacked', ''
    if len(rows) < 4:
        reason = 'fewer than four nu rows: the extension cannot be validated'
        if points.coverage['nu_rows_selected'] < points.coverage['nu_rows_available']:
            reason += (f" (the frame budget reached {points.coverage['nu_rows_selected']} of "
                       f"{points.coverage['nu_rows_available']} nu rows)")
    else:
        held = {'stacked': [], 'stacked_extended': []}
        folds = []
        for row in rows:
            train = [c for c in controls if round(c.motors[arm.nu_motor], 6) != row]
            test = [c for c in controls if round(c.motors[arm.nu_motor], 6) == row]
            fold = {'nu': row, 'points': int(sum(len(c.rows) for c in test))}
            for name, extended in (('stacked', False), ('stacked_extended', True)):
                trained = fit(train, extended)
                residual = _held_out(trained, test, points.calibrant, arm.nu_motor, arm.del_motor)
                held[name].append(residual)
                fold[name] = {'held_out_sigma_A^-1': _robust_sigma(residual), 'success': trained.success}
            folds.append(fold)
        sigma = {name: _robust_sigma(np.concatenate(parts)) for name, parts in held.items()}
        improvement = 1.0 - sigma['stacked_extended'] / sigma['stacked'] if sigma['stacked'] > 0 else 0.0
        validation.update({'held_out_sigma_A^-1': sigma, 'improvement': improvement, 'folds': folds})
        if improvement >= min_improvement:
            chosen = 'stacked_extended'
        else:
            reason = f'the extension improved held-out error by only {improvement:.0%}'
    result = fit(controls, chosen == 'stacked_extended')
    if chosen == 'stacked_extended' and not result.success:
        reason = f'the extended fit was refused ({result.message})'
        chosen, result = 'stacked', fit(controls, False)
    validation.update({'chosen': chosen, 'reason': reason})
    if not result.success or result.rms_q > max_rms_q:
        raise ValueError(result.message or f'fit RMS {result.rms_q:.5g} exceeds {max_rms_q:g} Å⁻¹')
    asset = result.diffractometer.stacked_arm.to_pyfai_goniometer(result.diffractometer.calibration)
    asset['detector_config'] = _mg_detector_config(diff.calibration)
    source = {'uri': str(Path(spec.uri).resolve()), 'kind': spec.kind.value, 'entry': spec.entry,
              'options': json.loads(json.dumps(dict(spec.options), default=str))}
    provenance = {'frames': list(points.frames), 'rms_q_A^-1': result.rms_q,
                  'condition_number': result.condition_number, 'n_points': result.n_points,
                  'distance_seed_m': None, 'distance_bounds_m': None, 'fit_motor_scales': False,
                  'source_motors': {'nu_angle': arm.nu_motor, 'del_angle': arm.del_motor},
                  'image_rotation': int(image_rotation), 'direct_beam': None,
                  'arm_model': chosen, 'parameter_errors': dict(result.param_errors),
                  'parameter_correlations': dict(result.param_correlations),
                  'automatic_refinement': {
                      'method': 'pyFAI SingleGeometry.extract_cp seeded by a stacked calibration',
                      'seed_sha256': digest, 'source': source, 'calibrant': str(calibrant),
                      'points_per_degree': float(points_per_degree), 'max_counts': max_counts,
                      'frame_screen': frame_screen.to_provenance(), 'excluded_frames': list(points.excluded),
                      'outlier_threshold_sigma': reject_outliers, 'rejected_points': result.rejected_points,
                      'validation': validation}}
    provenance = json.loads(json.dumps(provenance, default=float))
    message = ('Automatic refinement: detector mount and sample offset refined'
               if chosen == 'stacked_extended' else f'Automatic refinement: five-parameter arm kept ({reason})')
    return CalibrationFit('multigeometry', {'calibration': asset, 'provenance': provenance},
                          result.rms_q, result.condition_number, result.n_points, message)


__all__ = ['ScanRingPoints', 'extract_scan_ring_points', 'refine_arm_from_scan',
           'stratified_frame_indices']
