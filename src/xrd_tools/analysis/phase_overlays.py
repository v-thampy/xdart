"""Plain phase references computed on a tool's single worker, with one bounded cache.

The cache belongs to the callable, never to a request retained by the GUI owner.
It holds at most the last selection's line and detector contexts, each bounded by
the shared provider's 8/2048/4096 reflection and 16 MiB source limits. No source,
frame, image, integrator or structure is retained. A floor-zero listing normally
serves every floor edit. If that listing exceeds a cap, the requested filtered
listing is cached instead and serves equal or higher floors without relisting.
A phase still over its own cap is refused alone; the other phases keep their lines.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading

import numpy as np

from .phase_references import PhaseEntry


@dataclass(frozen=True, slots=True)
class PhaseOverlayRequest:
    entries: tuple[PhaseEntry, ...]
    floor: float
    line_unit: str
    line_range: tuple[float, float]
    wavelength_angstrom: float | None
    ring_snapshot: tuple[tuple[str, object], ...] = ()
    ring_shape: tuple[int, int] | None = None
    step_px: float = 2.
    line_reason: str = ''

    def __post_init__(self):
        object.__setattr__(self, 'entries', tuple(self.entries))
        object.__setattr__(self, 'line_range', tuple(self.line_range))
        object.__setattr__(self, 'ring_snapshot', tuple(tuple(p) for p in self.ring_snapshot))
        if self.ring_shape is not None:
            object.__setattr__(self, 'ring_shape', tuple(self.ring_shape))
        if any(not isinstance(v, (str, int, float, bool, type(None))) for _, v in self.ring_snapshot):
            raise TypeError('phase ring geometry must contain detached scalar values')


@dataclass(frozen=True, slots=True)
class PhaseAxisPositions:
    values: tuple[float, ...] = ()
    reason: str = ''


@dataclass(frozen=True, slots=True)
class PhaseOverlayFamily:
    id: str
    name: str
    source_kind: str
    q_lines: tuple[float, ...] = ()
    lines: tuple[float, ...] = ()
    rings: tuple = ()
    omitted_count: int = 0
    reason: str = ''


def positions_for_axis(qs, unit, wavelength_angstrom):
    """Convert canonical positions using only the adapter's displayed wavelength."""
    values = np.asarray(qs, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        return PhaseAxisPositions(reason='Reference positions must be finite, non-negative q values.')
    if unit == 'q':
        return PhaseAxisPositions(tuple(map(float, values)))
    if unit != '2theta':
        return PhaseAxisPositions(reason='Reference phases require a q or 2θ axis.')
    if wavelength_angstrom is None or not math.isfinite(wavelength_angstrom) or wavelength_angstrom <= 0:
        return PhaseAxisPositions(reason='2θ reference lines require one displayed wavelength.')
    arguments = values*wavelength_angstrom/(4.*math.pi)
    return PhaseAxisPositions(tuple(map(float, 2.*np.rad2deg(np.arcsin(arguments[arguments <= 1.])))))


class PhaseOverlayJob:
    """A SingleWorkerOwner callable; clear_cache fences an in-flight cache commit."""

    def __init__(self):
        self._lock = threading.Lock()
        self._epoch, self._cache = 0, None

    def clear_cache(self):
        with self._lock:
            self._epoch += 1
            self._cache = None

    @staticmethod
    def _check(cancel):
        if cancel is not None and cancel.is_set():
            raise InterruptedError('Reference phase computation cancelled.')

    def _listed(self, request, wavelength, bounds, cancel):
        # Import late: the shared provider remains the observable production seam.
        from . import phase_references as provider
        def listing(entries, floor):
            self._check(cancel)
            return provider.build_phase_references(*entries,
                wavelength_angstrom=wavelength, min_intensity=floor, **bounds)
        result, floor, refused = listing(request.entries, 0.), 0., {}
        if result.reason and 'reflection limit' in result.reason and request.floor > 0:
            floor = request.floor
            result = listing(request.entries, floor)
        if result.reason and 'reflection limit' in result.reason and len(request.entries) > 1:
            # One phase over its cap must not blank the others: keep each phase
            # that lists alone within the shared total; refuse only the rest.
            phases, total = [], 0
            for entry in request.entries:
                alone = listing((entry,), floor)
                count = sum(len(phase.reflections) for phase in alone.phases)
                if alone.reason or total+count > provider.MAX_REFLECTIONS_TOTAL:
                    refused[entry.id] = alone.reason or 'cumulative reflection limit exceeded'
                else:
                    phases.extend(alone.phases)
                    total += count
            result = provider.PhaseReferences(tuple(phases))
        self._check(cancel)
        return result, floor, refused

    def _ring_context(self, request, cancel):
        from xrd_tools import geometry_figures as geometry
        from xrd_tools.core.containers import PONI
        from xrd_tools.core.geometry.diffractometer import DetectorCalibration
        from xrd_tools.integrate.calibration import detector_calibration_to_integrator
        import json
        snapshot = dict(request.ring_snapshot)
        reason = geometry.detector_ring_unavailable(snapshot, request.ring_shape)
        if reason:
            return None, reason
        ai = detector_calibration_to_integrator(DetectorCalibration(PONI.from_dict(snapshot),
            json.loads(snapshot['detector_config_json']), parallax=snapshot.get('parallax')))
        if not ai.detector.IS_FLAT or not ai.detector.uniform_pixel or ai.detector.spline is not None:
            return None, 'Predicted rings require a uniform, flat, undistorted detector.'
        bounds = tuple(map(math.degrees, geometry._detector_angle_extrema(ai, request.ring_shape)))
        listed, floor, refused = self._listed(request, ai.wavelength*1e10, {'two_theta_range': bounds}, cancel)
        points, rings = geometry.cone_plane(ai), {}
        for phase in listed.phases:
            curves = []
            for reflection in phase.reflections:
                self._check(cancel)
                theta = 2.*math.asin(min(1., reflection.q*ai.wavelength*1e10/(4.*math.pi)))
                arcs = geometry.display_arcs(points, theta, request.ring_shape, request.step_px/.4)
                for arc in arcs:
                    curves.append((reflection, geometry.Curve(phase.name, arc, 'predicted',
                        f'{phase.name} reference: q {reflection.q:.5g} Å⁻¹. Display only.', math.degrees(theta))))
            rings[phase.id] = tuple(curves)
        return (listed, floor, rings, refused), listed.reason

    def __call__(self, request, cancel, progress):
        self._check(cancel)
        if not request.entries:
            self.clear_cache()
            return ()
        key = (tuple((e.id,e.name,e.source_kind,e.lattice) for e in request.entries),
               request.line_unit, request.line_range, request.wavelength_angstrom,
               request.ring_snapshot, request.ring_shape, request.step_px, request.line_reason)
        with self._lock:
            epoch = self._epoch
            cached = self._cache[1] if self._cache is not None and self._cache[0] == key else {}
        contexts, reasons = {}, []
        if not math.isfinite(request.floor) or not 0 <= request.floor <= 100:
            return tuple(PhaseOverlayFamily(e.id,e.name,e.source_kind,reason='Minimum intensity must be between 0 and 100 percent.') for e in request.entries)
        if not math.isfinite(request.step_px) or request.step_px <= 0:
            return tuple(PhaseOverlayFamily(e.id,e.name,e.source_kind,reason='Ring sampling step must be positive and finite.') for e in request.entries)
        line_reason = request.line_reason
        bounds = request.line_range
        if not line_reason and (len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or not 0 <= bounds[0] <= bounds[1]):
            line_reason = 'Reference phases require a finite q or 2θ range.'
        wavelength = request.wavelength_angstrom
        if not line_reason and request.line_unit == '2theta' and (wavelength is None or not math.isfinite(wavelength) or wavelength <= 0):
            line_reason = '2θ reference lines require one displayed wavelength.'
        if not line_reason and request.line_unit not in ('q', '2theta'):
            line_reason = 'Reference phases require a q or 2θ axis.'
        if not line_reason:
            if wavelength is None or not math.isfinite(wavelength) or wavelength <= 0:
                wavelength = 1.5406
                reasons.append('No record wavelength: reference intensities use the existing 1.5406 Å default; q listing ends at 4π/1.5406 Å⁻¹.')
            previous = cached.get('line')
            if previous is not None and previous[1] <= request.floor:
                contexts['line'] = previous
            else:
                listed, floor, refused = self._listed(request, wavelength,
                    {'q_range' if request.line_unit == 'q' else 'two_theta_range': bounds}, cancel)
                contexts['line'] = (listed, floor, {}, refused)
        else:
            reasons.append(line_reason)
        if request.ring_snapshot:
            previous = cached.get('ring')
            if previous is not None and previous[1] <= request.floor:
                contexts['ring'] = previous
            else:
                context, reason = self._ring_context(request, cancel)
                if context is not None:
                    contexts['ring'] = context
                elif reason:
                    reasons.append(reason)
        self._check(cancel)
        results = []
        for entry in request.entries:
            qs, rings, omitted, errors = (), (), 0, list(reasons)
            for kind, (listed, floor, arcs, refused) in contexts.items():
                if listed.reason:
                    errors.append(listed.reason)
                if entry.id in refused:
                    errors.append(refused[entry.id])
                phase = next((p for p in listed.phases if p.id == entry.id), None)
                if phase is None:
                    continue
                accepted = tuple(r for r in phase.reflections if r.intensity is None or r.intensity >= request.floor)
                omitted = max(omitted, phase.omitted_count+len(phase.reflections)-len(accepted))
                if kind == 'line':
                    qs = tuple(r.q for r in accepted)
                else:
                    rings = tuple(curve for r,curve in arcs.get(entry.id, ()) if r.intensity is None or r.intensity >= request.floor)
            positions = positions_for_axis(qs, request.line_unit, request.wavelength_angstrom)
            if positions.reason and qs:
                errors.append(positions.reason)
            results.append(PhaseOverlayFamily(entry.id,entry.name,entry.source_kind,qs,
                positions.values,rings,omitted,' '.join(dict.fromkeys(errors))))
        with self._lock:
            if epoch == self._epoch and not (cancel is not None and cancel.is_set()) and all(not c[0].reason for c in contexts.values()):
                self._cache = (key, contexts)
        return tuple(results)
