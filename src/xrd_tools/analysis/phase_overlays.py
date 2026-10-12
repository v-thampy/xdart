"""Plain phase references computed on a tool's single worker, with one bounded per-phase cache.

The cache belongs to the callable, never to a request retained by the GUI owner (REFERENCE-PHASES-20261011). It
holds at most ``MAX_PHASES`` phase slots, the least recently used leaving first, and each slot at most one line
context and one detector-ring context. A request is admitted whole first (phase count and CIF bytes). Each phase
is then listed alone at floor zero, or at its own floor when floor zero is over its cap, so a floor edit is
served without listing and one phase's change never lists another. Caps apply after filtering: a phase over its
own cap, or past the shared total in list order, is refused alone and the others keep their lines. A phase whose
entry, effective floor (``phase_floor``: none for a calibrant or lattice) and contexts are unchanged is returned as
the identical family object. No source, frame, image, integrator or structure is retained beyond those listings.
"""
from __future__ import annotations

from collections import OrderedDict
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
    #: per-phase minimum intensities, ``((entry id, percent), ...)``; a phase without one uses ``floor``
    floors: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'entries', tuple(self.entries))
        object.__setattr__(self, 'floors', tuple(tuple(pair) for pair in self.floors))
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


#: The P3a caveat every predicted calibrant ring keeps (the maintainer's wording).
RING_CAVEAT = 'Display only; radial agreement alone does not identify detector mounting.'


@dataclass(frozen=True, slots=True)
class _Detector:
    wavelength_angstrom: float
    bounds: tuple[float, float]
    points: object


def _valid_floor(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and 0 <= value <= 100)


def phase_floor(request, entry):
    """The minimum intensity that can change ``entry``'s family: a CIF phase's own value, else the request's. A
    calibrant or lattice lists positions without intensities, so no minimum applies to it (None)."""
    return dict(request.floors).get(entry.id, request.floor) if entry.source_kind == 'cif' else None


class PhaseOverlayJob:
    """A SingleWorkerOwner callable; clear_cache fences an in-flight cache commit."""

    def __init__(self):
        self._lock = threading.Lock()
        self._epoch, self._slots, self._detector = 0, OrderedDict(), None

    def clear_cache(self):
        with self._lock:
            self._epoch += 1
            self._slots = OrderedDict()
            self._detector = None

    @staticmethod
    def _check(cancel):
        if cancel is not None and cancel.is_set():
            raise InterruptedError('Reference phase computation cancelled.')

    def _list(self, entry, floor, wavelength, bounds, cancel):
        """One phase alone: at floor zero, or at its own floor when floor zero is over its cap."""
        # Import late: the shared provider remains the observable production seam.
        from . import phase_references as provider

        def listing(at):
            self._check(cancel)
            return provider.build_phase_references(entry, wavelength_angstrom=wavelength, min_intensity=at, **bounds)
        listed, at = listing(0.), 0.
        if listed.reason and 'reflection limit' in listed.reason and floor > 0:
            listed, at = listing(floor), floor
        return (at, None, listed.reason) if listed.reason else (at, listed.phases[0], '')

    @staticmethod
    def _served(cached, key, floor):
        if cached is None or cached[0] != key:
            return False
        return cached[2] is not None and cached[1] <= floor or cached[2] is None and cached[1] == floor

    def _detector_context(self, request, key, cancel):
        from xrd_tools import geometry_figures as geometry
        from xrd_tools.core.containers import PONI
        from xrd_tools.core.geometry.diffractometer import DetectorCalibration
        from xrd_tools.integrate.calibration import detector_calibration_to_integrator
        import json
        with self._lock:
            cached = self._detector
        if cached is not None and cached[0] == key:
            return cached[1], ''
        snapshot = dict(request.ring_snapshot)
        reason = geometry.detector_ring_unavailable(snapshot, request.ring_shape)
        if reason:
            return None, reason
        self._check(cancel)
        ai = detector_calibration_to_integrator(DetectorCalibration(PONI.from_dict(snapshot),
            json.loads(snapshot['detector_config_json']), parallax=snapshot.get('parallax')))
        if not ai.detector.IS_FLAT or not ai.detector.uniform_pixel or ai.detector.spline is not None:
            return None, 'Predicted rings require a uniform, flat, undistorted detector.'
        bounds = tuple(map(math.degrees, geometry._detector_angle_extrema(ai, request.ring_shape)))
        return _Detector(ai.wavelength*1e10, bounds, geometry.cone_plane(ai)), ''

    def _arcs(self, entry, reference, detector, request, cancel):
        from xrd_tools import geometry_figures as geometry
        calibrant = entry.source_kind == 'calibrant'
        curves = []
        for reflection in reference.reflections:
            self._check(cancel)
            theta = 2.*math.asin(min(1., reflection.q*detector.wavelength_angstrom/(4.*math.pi)))
            for arc in geometry.display_arcs(detector.points, theta, request.ring_shape, request.step_px/.4):
                tip = (f'Predicted {entry.name} ring: 2θ {math.degrees(theta):.4g}°. {RING_CAVEAT}' if calibrant
                       else f'{entry.name} reference: q {reflection.q:.5g} Å⁻¹. Display only.')
                curves.append((reflection, geometry.Curve(entry.name, arc, 'predicted', tip, math.degrees(theta))))
        return tuple(curves)

    def __call__(self, request, cancel, progress):
        self._check(cancel)
        if not request.entries:
            self.clear_cache()
            return ()
        overrides = dict(request.floors)
        if not all(_valid_floor(value) for value in (request.floor, *overrides.values())):
            return tuple(PhaseOverlayFamily(e.id,e.name,e.source_kind,reason='Minimum intensity must be between 0 and 100 percent.') for e in request.entries)
        if not math.isfinite(request.step_px) or request.step_px <= 0:
            return tuple(PhaseOverlayFamily(e.id,e.name,e.source_kind,reason='Ring sampling step must be positive and finite.') for e in request.entries)
        from . import phase_references as provider
        try:
            provider.admit_phase_entries(request.entries)
        except ValueError as exc:
            return tuple(PhaseOverlayFamily(e.id, e.name, e.source_kind, reason=str(exc)) for e in request.entries)
        reasons = []
        line_reason = request.line_reason
        bounds = request.line_range
        if not line_reason and (len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or not 0 <= bounds[0] <= bounds[1]):
            line_reason = 'Reference phases require a finite q or 2θ range.'
        wavelength = request.wavelength_angstrom
        if not line_reason and request.line_unit == '2theta' and (wavelength is None or not math.isfinite(wavelength) or wavelength <= 0):
            line_reason = '2θ reference lines require one displayed wavelength.'
        if not line_reason and request.line_unit not in ('q', '2theta'):
            line_reason = 'Reference phases require a q or 2θ axis.'
        line_key = None
        if not line_reason:
            if wavelength is None or not math.isfinite(wavelength) or wavelength <= 0:
                wavelength = 1.5406
                reasons.append('No record wavelength: reference intensities use the existing 1.5406 Å default; q listing ends at 4π/1.5406 Å⁻¹.')
            line_key = (request.line_unit, tuple(bounds), wavelength)
        else:
            reasons.append(line_reason)
        ring_key = (request.ring_snapshot, request.ring_shape, request.step_px) if request.ring_snapshot else None
        with self._lock:
            epoch = self._epoch
            known = {identity: dict(slot) for identity, slot in self._slots.items()}
        detector, slots = None, {}
        for entry in request.entries:
            identity = (entry.id, entry.name, entry.source_kind, entry.lattice)
            floor = overrides.get(entry.id, request.floor)
            slot = slots[identity] = known.get(identity, {})
            if line_key is not None and not self._served(slot.get('line'), line_key, floor):
                slot['line'] = (line_key, *self._list(entry, floor, wavelength,
                    {'q_range' if request.line_unit == 'q' else 'two_theta_range': tuple(bounds)}, cancel))
            if ring_key is not None and not self._served(slot.get('ring'), ring_key, floor):
                if detector is None:
                    detector, ring_reason = self._detector_context(request, ring_key, cancel)
                    if ring_reason:
                        reasons.append(ring_reason)
                        ring_key = None
                        continue
                at, reference, reason = self._list(entry, floor, detector.wavelength_angstrom,
                                                   {'two_theta_range': detector.bounds}, cancel)
                arcs = self._arcs(entry, reference, detector, request, cancel) if reference is not None else ()
                slot['ring'] = (ring_key, at, reference, reason, arcs)
        self._check(cancel)
        accepted, refusals = {}, {}
        for kind, key in (('line', line_key), ('ring', ring_key)):
            if key is None:
                continue
            total = 0
            for entry in request.entries:
                identity = (entry.id, entry.name, entry.source_kind, entry.lattice)
                cached = slots[identity].get(kind)
                floor = overrides.get(entry.id, request.floor)
                if cached is None or cached[2] is None:
                    refusals.setdefault(identity, []).append(cached[3] if cached is not None else '')
                    continue
                kept = tuple(r for r in cached[2].reflections if r.intensity is None or r.intensity >= floor)
                if total + len(kept) > provider.MAX_REFLECTIONS_TOTAL:
                    refusals.setdefault(identity, []).append('cumulative reflection limit exceeded')
                    continue
                total += len(kept)
                accepted[identity, kind] = kept
        results = []
        for entry in request.entries:
            identity = (entry.id, entry.name, entry.source_kind, entry.lattice)
            slot, floor = slots[identity], overrides.get(entry.id, request.floor)
            errors = list(reasons) + [reason for reason in refusals.get(identity, ()) if reason]
            qs = tuple(r.q for r in accepted.get((identity, 'line'), ()))
            kept = set(map(id, accepted.get((identity, 'ring'), ())))
            ring = slot.get('ring') if ring_key is not None else None
            rings = tuple(curve for reflection, curve in ring[4] if id(reflection) in kept) if ring else ()
            omitted = 0
            for kind in ('line', 'ring'):
                cached = slot.get(kind) if (line_key if kind == 'line' else ring_key) is not None else None
                if cached is not None and cached[2] is not None:
                    omitted = max(omitted, cached[2].omitted_count + len(cached[2].reflections)
                                  - len(accepted.get((identity, kind), cached[2].reflections)))
            positions = positions_for_axis(qs, request.line_unit, request.wavelength_angstrom)
            if positions.reason and qs:
                errors.append(positions.reason)
            reason = ' '.join(dict.fromkeys(errors))
            scalars = (phase_floor(request, entry), request.line_unit, request.wavelength_angstrom, line_key,
                       ring_key, reason)
            line = slot.get('line') if line_key is not None else None
            previous = slot.get('family')
            if previous is not None and previous[0] == scalars and previous[1] is line and previous[2] is ring:
                family = previous[3]
            else:
                family = PhaseOverlayFamily(entry.id, entry.name, entry.source_kind, qs, positions.values, rings,
                                            omitted, reason)
                slot['family'] = (scalars, line, ring, family)
            results.append(family)
        with self._lock:
            if epoch == self._epoch and not (cancel is not None and cancel.is_set()):
                for identity, slot in slots.items():
                    self._slots[identity] = slot
                    self._slots.move_to_end(identity)
                while len(self._slots) > provider.MAX_PHASES:
                    self._slots.popitem(last=False)
                if detector is not None:
                    self._detector = (ring_key, detector)
        return tuple(results)
