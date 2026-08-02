"""Finite Experiment values and their revision-qualified editor."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from xrd_tools.core import energy as _energy

class FactStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    CONFLICT = "conflict"

class EnergySource(str, Enum):
    NONE = "none"
    CALIBRATION = "calibration"
    SOURCE_METADATA = "source_metadata"
    OPERATOR = "operator"

class IncidenceKind(str, Enum):
    ABSENT = "absent"
    MOTOR = "motor"
    MANUAL = "manual"
def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if type(value) is bool:
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and " if positive else ""
        raise ValueError(f"{name} must be {qualifier}finite")
    return result
def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value
def _finite_map(values: Mapping[str, Any], name: str, *, positive: bool = False) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = {
        _text(key, f"{name} key"): _finite(value, f"{name}[{key!r}]", positive=positive)
        for key, value in values.items()
    }
    if "" in result:
        raise ValueError(f"{name} keys must be non-empty")
    return MappingProxyType(dict(sorted(result.items())))
def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is missing or malformed")
    return value

@dataclass(frozen=True, slots=True)
class PoniValues:
    dist: float
    poni1: float
    poni2: float
    rot1: float
    rot2: float
    rot3: float
    wavelength_m: float
    def __post_init__(self) -> None:
        for name in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.dist <= 0.0:
            raise ValueError("dist must be > 0")
        object.__setattr__(self, "wavelength_m", _finite(
            self.wavelength_m, "wavelength_m", positive=True
        ))

@dataclass(frozen=True, slots=True)
class MaskState:
    source_uri: str = ""
    sha256: str = ""
    dtype: str = ""
    shape: tuple[int, ...] = ()
    status: FactStatus = FactStatus.ABSENT
    def __post_init__(self) -> None:
        object.__setattr__(self, "status", FactStatus(self.status))
        for name in ("source_uri", "sha256", "dtype"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        shape = tuple(self.shape)
        if any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError("mask shape entries must be positive integers")
        object.__setattr__(self, "shape", shape)
        if self.status is FactStatus.PRESENT and not self.sha256:
            raise ValueError("a present mask needs a content digest")
    @classmethod
    def absent(cls) -> MaskState:
        return cls()

@dataclass(frozen=True, slots=True)
class CalibrationState:
    values: PoniValues | None = None
    detector_id: str = ""
    detector_config: Mapping[str, float] = field(default_factory=dict)
    value_fingerprint: str = ""
    source_sha256: str = ""
    source_uri: str = ""
    mask: MaskState = field(default_factory=MaskState.absent)
    status: FactStatus = FactStatus.ABSENT
    def __post_init__(self) -> None:
        object.__setattr__(self, "status", FactStatus(self.status))
        for name in ("detector_id", "value_fingerprint", "source_sha256", "source_uri"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "detector_config", _finite_map(
            self.detector_config, "detector_config"
        ))
        if self.values is not None and not isinstance(self.values, PoniValues):
            raise TypeError("values must be PoniValues or None")
        if not isinstance(self.mask, MaskState):
            raise TypeError("mask must be MaskState")
        if (self.status is FactStatus.PRESENT) != (self.values is not None):
            raise ValueError("calibration status and PONI values disagree")
    @property
    def imported_wavelength_m(self) -> float | None:
        return None if self.values is None else self.values.wavelength_m

@dataclass(frozen=True, slots=True)
class IncidenceState:
    kind: IncidenceKind = IncidenceKind.ABSENT
    motor_name: str = ""
    motor_resolved: bool = False
    manual_angle_deg: float | None = None
    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", IncidenceKind(self.kind))
        object.__setattr__(self, "motor_name", _text(self.motor_name, "motor_name"))
        if type(self.motor_resolved) is not bool:
            raise TypeError("motor_resolved must be bool")
        if self.manual_angle_deg is not None:
            object.__setattr__(self, "manual_angle_deg", _finite(
                self.manual_angle_deg, "manual_angle_deg"
            ))
        if self.kind is IncidenceKind.MOTOR:
            valid = bool(self.motor_name) and self.manual_angle_deg is None
        elif self.kind is IncidenceKind.MANUAL:
            valid = self.manual_angle_deg is not None and not self.motor_name and not self.motor_resolved
        else:
            valid = not self.motor_name and not self.motor_resolved and self.manual_angle_deg is None
        if not valid:
            raise ValueError("incidence fields disagree with incidence kind")

@dataclass(frozen=True, slots=True)
class GeometryState:
    gi_enabled: bool = False
    incidence: IncidenceState = field(default_factory=IncidenceState)
    tilt_deg: float = 0.0
    convention_id: str = ""
    diffractometer_ref: str | None = None
    def __post_init__(self) -> None:
        if type(self.gi_enabled) is not bool or not isinstance(self.incidence, IncidenceState):
            raise TypeError("geometry enablement and incidence must be typed")
        object.__setattr__(self, "tilt_deg", _finite(self.tilt_deg, "tilt_deg"))
        object.__setattr__(self, "convention_id", _text(self.convention_id, "convention_id"))
        if self.diffractometer_ref is not None:
            object.__setattr__(self, "diffractometer_ref", _text(
                self.diffractometer_ref, "diffractometer_ref"
            ))

@dataclass(frozen=True, slots=True)
class EnergyState:
    wavelength_m: float | None = None
    source: EnergySource = EnergySource.NONE
    evidence: Mapping[str, float] = field(default_factory=dict)
    status: FactStatus = FactStatus.ABSENT
    def __post_init__(self) -> None:
        object.__setattr__(self, "source", EnergySource(self.source))
        object.__setattr__(self, "status", FactStatus(self.status))
        object.__setattr__(self, "evidence", _finite_map(self.evidence, "evidence", positive=True))
        if self.wavelength_m is not None:
            object.__setattr__(self, "wavelength_m", _finite(
                self.wavelength_m, "wavelength_m", positive=True
            ))
        if self.status is FactStatus.PRESENT:
            valid = self.wavelength_m is not None and self.source is not EnergySource.NONE and all(math.isclose(self.wavelength_m, value, rel_tol=1e-3, abs_tol=0.0) for value in self.evidence.values())
        elif self.status is FactStatus.ABSENT:
            valid = (self.wavelength_m is None and self.source is EnergySource.NONE
                     and not self.evidence)
        else:
            valid = bool(self.evidence) and (
                (self.wavelength_m is None) == (self.source is EnergySource.NONE)
            )
        if not valid:
            raise ValueError("energy selection disagrees with status or source")
    @classmethod
    def from_energy_eV(cls, energy_eV: float, *, source: EnergySource,
                       evidence: Mapping[str, float] | None = None) -> EnergyState:
        wavelength = _finite(_energy.energy_eV_to_wavelength_m(
            _finite(energy_eV, "energy_eV", positive=True)
        ), "converted wavelength", positive=True)
        return cls(wavelength, source, evidence or {EnergySource(source).value: wavelength},
                   FactStatus.PRESENT)
    @property
    def energy_eV(self) -> float | None:
        if self.wavelength_m is None:
            return None
        return _finite(_energy.wavelength_m_to_energy_eV(self.wavelength_m),
                       "converted energy", positive=True)
    def with_calibration_evidence(self, value: float | None) -> EnergyState:
        evidence = dict(self.evidence)
        if value is None:
            evidence.pop(EnergySource.CALIBRATION.value, None)
        else:
            evidence[EnergySource.CALIBRATION.value] = _finite(
                value, "calibration wavelength", positive=True
            )
        if self.source is EnergySource.CALIBRATION:
            if value is None:
                return EnergyState(evidence=evidence,
                                   status=FactStatus.CONFLICT if evidence else FactStatus.ABSENT)
            selected, source = value, EnergySource.CALIBRATION
        else:
            selected, source = self.wavelength_m, self.source
        if selected is None:
            return EnergyState(evidence=evidence, status=(
                FactStatus.CONFLICT if evidence else FactStatus.ABSENT
            ))
        conflict = any(not math.isclose(selected, item, rel_tol=1e-3, abs_tol=0.0)
                       for item in evidence.values())
        return EnergyState(selected, source, evidence,
                           FactStatus.CONFLICT if conflict else FactStatus.PRESENT)

@dataclass(frozen=True, slots=True)
class OrientationState:
    code: int = 1
    convention_id: str = ""
    def __post_init__(self) -> None:
        if type(self.code) is not int or not 1 <= self.code <= 8:
            raise ValueError("orientation code must be an integer in 1..8")
        object.__setattr__(self, "convention_id", _text(self.convention_id, "convention_id"))

@dataclass(frozen=True, slots=True)
class SampleState:
    sample_id: str = ""
    display_name: str = ""
    orientation: OrientationState = field(default_factory=OrientationState)
    tags: tuple[str, ...] = ()
    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _text(self.sample_id, "sample_id"))
        object.__setattr__(self, "display_name", _text(self.display_name, "display_name"))
        if not isinstance(self.orientation, OrientationState):
            raise TypeError("orientation must be OrientationState")
        object.__setattr__(self, "tags", tuple(_text(tag, "tag") for tag in self.tags))

@dataclass(frozen=True, slots=True)
class ExperimentProvenance:
    source_kind: str = ""
    source_uri: str = ""
    digest: str = ""
    imported_at: str = ""
    operator: str = ""
    notes: tuple[str, ...] = ()
    def __post_init__(self) -> None:
        for name in ("source_kind", "source_uri", "digest", "imported_at", "operator"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "notes", tuple(_text(note, "note") for note in self.notes))

@dataclass(frozen=True, slots=True)
class ExperimentState:
    experiment_id: str
    revision: int
    calibration: CalibrationState
    geometry: GeometryState
    energy: EnergyState
    sample: SampleState
    provenance: ExperimentProvenance = field(default_factory=ExperimentProvenance)
    schema_version: int = 1
    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", _text(self.experiment_id, "experiment_id"))
        if not self.experiment_id:
            raise ValueError("experiment_id must be non-empty")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision must be a non-negative integer")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported ExperimentState schema_version")
        expected = ((self.calibration, CalibrationState), (self.geometry, GeometryState),
                    (self.energy, EnergyState), (self.sample, SampleState),
                    (self.provenance, ExperimentProvenance))
        if any(not isinstance(value, kind) for value, kind in expected):
            raise TypeError("ExperimentState components must be typed")
        if self.energy.wavelength_m is None or not self.sample.sample_id:
            raise ValueError("ExperimentState needs selected energy and sample identity")
    def scientific_content(self) -> dict[str, Any]:
        calibration, geometry = self.calibration, self.geometry
        energy, sample, provenance = self.energy, self.sample, self.provenance
        values, mask, incidence = calibration.values, calibration.mask, geometry.incidence
        return {
            "schema_version": self.schema_version,
            "calibration": {
                "values": None if values is None else {
                    "dist": values.dist, "poni1": values.poni1, "poni2": values.poni2,
                    "rot1": values.rot1, "rot2": values.rot2, "rot3": values.rot3,
                    "wavelength_m": values.wavelength_m,
                },
                "detector_id": calibration.detector_id,
                "detector_config": dict(calibration.detector_config),
                "value_fingerprint": calibration.value_fingerprint,
                "source_sha256": calibration.source_sha256,
                "source_uri": calibration.source_uri,
                "mask": {"source_uri": mask.source_uri, "sha256": mask.sha256,
                         "dtype": mask.dtype, "shape": list(mask.shape),
                         "status": mask.status.value},
                "status": calibration.status.value,
            },
            "geometry": {
                "gi_enabled": geometry.gi_enabled,
                "incidence": {"kind": incidence.kind.value,
                              "motor_name": incidence.motor_name,
                              "motor_resolved": incidence.motor_resolved,
                              "manual_angle_deg": incidence.manual_angle_deg},
                "tilt_deg": geometry.tilt_deg,
                "convention_id": geometry.convention_id,
                "diffractometer_ref": geometry.diffractometer_ref,
            },
            "energy": {"wavelength_m": energy.wavelength_m, "source": energy.source.value,
                       "evidence": dict(energy.evidence), "status": energy.status.value},
            "sample": {"sample_id": sample.sample_id, "display_name": sample.display_name,
                       "orientation": {"code": sample.orientation.code,
                                       "convention_id": sample.orientation.convention_id},
                       "tags": list(sample.tags)},
            "provenance": {"source_kind": provenance.source_kind,
                           "source_uri": provenance.source_uri, "digest": provenance.digest,
                           "imported_at": provenance.imported_at, "operator": provenance.operator,
                           "notes": list(provenance.notes)},
        }
    @property
    def content_fingerprint(self) -> str:
        payload = json.dumps(self.scientific_content(), sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(payload).hexdigest()
    def to_provenance(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id, "revision": self.revision,
                "content_fingerprint": self.content_fingerprint,
                "scientific_content": self.scientific_content()}
    @classmethod
    def from_provenance(cls, payload: Mapping[str, Any]) -> ExperimentState:
        payload = _mapping(payload, "experiment provenance")
        content = _mapping(payload.get("scientific_content"), "scientific_content")
        calibration = dict(_mapping(content.get("calibration"), "calibration"))
        geometry = dict(_mapping(content.get("geometry"), "geometry"))
        energy = dict(_mapping(content.get("energy"), "energy"))
        sample = dict(_mapping(content.get("sample"), "sample"))
        provenance = dict(_mapping(content.get("provenance"), "provenance"))
        raw_values = calibration.get("values")
        calibration["values"] = None if raw_values is None else PoniValues(
            **dict(_mapping(raw_values, "calibration values"))
        )
        calibration["mask"] = MaskState(**dict(_mapping(
            calibration.get("mask"), "calibration mask"
        )))
        geometry["incidence"] = IncidenceState(**dict(_mapping(
            geometry.get("incidence"), "geometry incidence"
        )))
        sample["orientation"] = OrientationState(**dict(_mapping(
            sample.get("orientation"), "sample orientation"
        )))
        state = cls(
            experiment_id=payload.get("experiment_id"), revision=payload.get("revision"),
            schema_version=content.get("schema_version"),
            calibration=CalibrationState(**calibration), geometry=GeometryState(**geometry),
            energy=EnergyState(**energy), sample=SampleState(**sample),
            provenance=ExperimentProvenance(**provenance),
        )
        if payload.get("content_fingerprint") != state.content_fingerprint:
            raise ValueError("experiment content fingerprint mismatch")
        return state

class CasStatus(str, Enum):
    ACCEPTED = "accepted"
    NO_CHANGE = "no_change"
    FOREIGN_IDENTITY = "foreign_identity"
    STALE_REVISION = "stale_revision"

@dataclass(frozen=True, slots=True)
class CasOutcome:
    status: CasStatus
    state: ExperimentState
    reason: str = ""
    @property
    def accepted(self) -> bool:
        return self.status is CasStatus.ACCEPTED

@runtime_checkable
class ExperimentEditorPort(Protocol):
    def current(self) -> ExperimentState: ...
    def propose_calibration(self, candidate: CalibrationState, *, experiment_id: str,
                            expected_revision: int) -> CasOutcome: ...
    def propose_mask(self, candidate: MaskState, *, experiment_id: str,
                     expected_revision: int) -> CasOutcome: ...
class ExperimentEditor:
    """One locked owner for identity-and-revision-qualified proposals."""
    def __init__(self, initial: ExperimentState):
        if not isinstance(initial, ExperimentState):
            raise TypeError("initial must be ExperimentState")
        self._state, self._lock = initial, threading.Lock()
    def current(self) -> ExperimentState:
        with self._lock:
            return self._state
    def _refusal(self, experiment_id: str, revision: int) -> CasOutcome | None:
        if not isinstance(experiment_id, str) or experiment_id != self._state.experiment_id:
            return CasOutcome(CasStatus.FOREIGN_IDENTITY, self._state,
                              "foreign experiment identity")
        if type(revision) is not int or revision != self._state.revision:
            return CasOutcome(CasStatus.STALE_REVISION, self._state,
                              "stale experiment revision")
        return None
    def propose_calibration(self, candidate: CalibrationState, *, experiment_id: str,
                            expected_revision: int) -> CasOutcome:
        if not isinstance(candidate, CalibrationState):
            raise TypeError("candidate must be CalibrationState")
        with self._lock:
            refusal = self._refusal(experiment_id, expected_revision)
            if refusal:
                return refusal
            if candidate == self._state.calibration:
                return CasOutcome(CasStatus.NO_CHANGE, self._state)
            self._state = replace(
                self._state, revision=self._state.revision + 1, calibration=candidate,
                energy=self._state.energy.with_calibration_evidence(
                    candidate.imported_wavelength_m
                ),
            )
            return CasOutcome(CasStatus.ACCEPTED, self._state)
    def propose_mask(self, candidate: MaskState, *, experiment_id: str,
                     expected_revision: int) -> CasOutcome:
        if not isinstance(candidate, MaskState):
            raise TypeError("candidate must be MaskState")
        with self._lock:
            refusal = self._refusal(experiment_id, expected_revision)
            if refusal:
                return refusal
            if candidate == self._state.calibration.mask:
                return CasOutcome(CasStatus.NO_CHANGE, self._state)
            self._state = replace(
                self._state, revision=self._state.revision + 1,
                calibration=replace(self._state.calibration, mask=candidate),
            )
            return CasOutcome(CasStatus.ACCEPTED, self._state)
