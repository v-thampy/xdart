"""Typed, fail-closed reload of current and historical Experiment records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from xrd_tools.core.provenance import read_provenance_from_handle
from xrd_tools.session.experiment_state import (
    CalibrationState, EnergySource, EnergyState, ExperimentState, FactStatus,
    GeometryState, IncidenceKind, IncidenceState, MaskState, OrientationState,
    PoniValues, SampleState,
)


class ReloadStatus(str, Enum):
    EXACT = "exact"
    PARTIAL = "partial"
    LEGACY = "legacy"
    ABSENT = "absent"
    OPEN_FAILURE = "open_failure"


@dataclass(frozen=True, slots=True)
class PersistedExperimentFacts:
    status: ReloadStatus
    experiment: ExperimentState | None = None
    calibration: CalibrationState | None = None
    geometry: GeometryState | None = None
    energy: EnergyState | None = None
    sample: SampleState | None = None
    content_fingerprint: str | None = None
    reason: str = ""


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _number(value: Any, *, positive: bool = False) -> float | None:
    if type(value) is bool:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and (not positive or result > 0.0) else None


def _scalar(group: Any, path: str) -> float | None:
    if path not in group:
        return None
    try:
        raw = group[path][()]
        if getattr(raw, "shape", ()):
            raw = raw.reshape(-1)[0]
        return _number(raw, positive=True)
    except Exception:
        return None


def _energy(entry: Any, calibration: CalibrationState | None) -> EnergyState | None:
    evidence: dict[str, float] = {}
    for name, path in (("headless", "instrument/monochromator/wavelength"),
                       ("legacy_source", "instrument/source/wavelength_A")):
        value = _scalar(entry, path)
        if value is not None:
            evidence[name] = value * 1e-10
    imported = None if calibration is None else calibration.imported_wavelength_m
    if imported is not None:
        evidence[EnergySource.CALIBRATION.value] = imported
    if not evidence:
        return None
    values = tuple(evidence.values())
    if any(not math.isclose(values[0], value, rel_tol=1e-3, abs_tol=0.0)
           for value in values[1:]):
        return EnergyState(evidence=evidence, status=FactStatus.CONFLICT)
    metadata = "headless" in evidence or "legacy_source" in evidence
    source = EnergySource.SOURCE_METADATA if metadata else EnergySource.CALIBRATION
    return EnergyState(values[0], source, evidence, FactStatus.PRESENT)


def _calibration(run: Mapping[str, Any]) -> CalibrationState | None:
    signature = _mapping(run.get("scientific_signature")) or {}
    assets = _mapping(signature.get("accepted_scientific_assets")) or {}
    raw = _mapping(assets.get("poni_values"))
    if raw is None:
        return None
    names = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")
    numbers = tuple(_number(raw.get(name)) for name in names)
    wavelength = _number(raw.get("wavelength_m"), positive=True)
    wavelength = wavelength or _number(raw.get("wavelength"), positive=True)
    if wavelength is None or any(value is None for value in numbers):
        return None
    digest, source = str(assets.get("mask_sha256") or ""), str(run.get("mask_file") or "")
    mask = (MaskState(source_uri=source, sha256=digest, status=FactStatus.PRESENT)
            if source or digest else MaskState.absent())
    return CalibrationState(
        values=PoniValues(*numbers, wavelength),
        detector_id=str(raw.get("detector") or ""),
        value_fingerprint=str(signature.get("fingerprint") or ""),
        source_sha256=str(assets.get("poni_sha256") or ""),
        source_uri=str(run.get("poni_file") or ""), mask=mask,
        status=FactStatus.PRESENT,
    )


def _geometry(value: Any) -> GeometryState | None:
    gi = _mapping(value)
    if gi is None: return None
    enabled = gi.get("enabled", False)
    motor = gi.get("resolved_motor") or gi.get("incidence_motor") or ""
    if type(enabled) is not bool or not isinstance(motor, str): return None
    angle = _number(gi.get("th_val"))
    if motor and motor.lower() != "manual":
        incidence = IncidenceState(IncidenceKind.MOTOR, motor,
                                   bool(gi.get("resolved_motor")))
    elif angle is not None:
        incidence = IncidenceState(IncidenceKind.MANUAL, manual_angle_deg=angle)
    else:
        incidence = IncidenceState()
    return GeometryState(enabled, incidence, _number(gi.get("tilt_angle")) or 0.0,
                         "pyfai-fiber-v1" if enabled else "")


def _sample(value: Any) -> SampleState | None:
    gi = _mapping(value)
    code = None if gi is None else gi.get("sample_orientation")
    if type(code) is not int or not 1 <= code <= 8:
        return None
    return SampleState(orientation=OrientationState(
        code, "pyfai-fiber-sample-orientation-v1"
    ))


class LegacyRecordAdapter:
    """The sole interpreter for Experiment facts in persisted NeXus records."""

    def read(self, path: str | Path, *, entry: str = "entry") -> PersistedExperimentFacts:
        import h5py

        try:
            handle = h5py.File(Path(path), "r")
        except OSError as exc:
            return PersistedExperimentFacts(
                ReloadStatus.OPEN_FAILURE, reason=f"{type(exc).__name__}: {exc}"
            )
        try:
            with handle:
                return self._read_handle(handle, entry)
        except Exception as exc:
            return PersistedExperimentFacts(
                ReloadStatus.ABSENT,
                reason=f"record decode failed closed: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _read_handle(handle: Any, entry: str) -> PersistedExperimentFacts:
        if entry not in handle:
            return PersistedExperimentFacts(ReloadStatus.ABSENT,
                                            reason=f"entry {entry!r} is absent")
        persisted = read_provenance_from_handle(handle, entry=entry)
        if not persisted:
            return PersistedExperimentFacts(ReloadStatus.ABSENT,
                                            reason="reduction provenance is absent")
        config = _mapping(persisted.get("config"))
        if config is None:
            return PersistedExperimentFacts(ReloadStatus.ABSENT,
                                            reason="reduction config is not a mapping")
        exact = _mapping(config.get("experiment"))
        if exact is not None:
            state = ExperimentState.from_provenance(exact)
            return PersistedExperimentFacts(
                ReloadStatus.EXACT, state, state.calibration, state.geometry,
                state.energy, state.sample, state.content_fingerprint,
            )
        run = _mapping(config.get("run_configuration"))
        if run is not None:
            calibration, geometry = _calibration(run), _geometry(run.get("gi"))
            return PersistedExperimentFacts(
                ReloadStatus.PARTIAL, calibration=calibration, geometry=geometry,
                energy=_energy(handle[entry], calibration), sample=_sample(run.get("gi")),
                content_fingerprint=str(run.get("fingerprint") or "") or None,
            )
        gi = config.get("gi_config")
        geometry, energy, sample = _geometry(gi), _energy(handle[entry], None), _sample(gi)
        if geometry is None and energy is None and sample is None:
            return PersistedExperimentFacts(ReloadStatus.ABSENT,
                                            reason="legacy experiment facts are absent")
        return PersistedExperimentFacts(
            ReloadStatus.LEGACY, geometry=geometry, energy=energy, sample=sample,
        )


__all__ = ["LegacyRecordAdapter", "PersistedExperimentFacts", "ReloadStatus"]
