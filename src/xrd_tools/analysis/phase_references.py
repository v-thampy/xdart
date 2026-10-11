"""Bounded, Qt-free phase sources and their shared reachable reflection list."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

MAX_PHASES = 8
MAX_CIF_BYTES = 4*1024*1024
MAX_TOTAL_CIF_BYTES = 16*1024*1024
MAX_REFLECTIONS_PER_PHASE = 2048
MAX_REFLECTIONS_TOTAL = 4096


@dataclass(frozen=True, slots=True)
class PhaseEntry:
    id: str
    name: str
    source_kind: str
    cif_bytes: bytes = b""
    lattice: tuple[tuple[str, object], ...] = ()
    cif_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PhaseLibraryLookup:
    entry: PhaseEntry | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PhaseReflection:
    q: float
    d_spacing: float
    intensity: float | None
    hkl: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class PhaseReference:
    id: str
    name: str
    source_kind: str
    reflections: tuple[PhaseReflection, ...]
    omitted_count: int = 0


@dataclass(frozen=True, slots=True)
class PhaseReferences:
    phases: tuple[PhaseReference, ...] = ()
    reason: str = ""


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _name(value) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > 256:
        raise ValueError("phase name must be nonempty and at most 256 bytes")
    return value


def _parse_cif(raw, structure_class=None):
    if len(raw) > MAX_CIF_BYTES:
        raise ValueError("CIF byte limit exceeded (4 MiB)")
    from .phase import HAS_PYMATGEN, Structure
    if not HAS_PYMATGEN:
        raise ValueError("pymatgen is required to parse CIF phases")
    try:
        return (Structure if structure_class is None else structure_class).from_str(raw.decode("utf-8-sig"), fmt="cif")
    except (ValueError, UnicodeError, KeyError, IndexError) as exc:
        raise ValueError("CIF could not be parsed") from exc


class PhaseLibrary:
    """Per-user content-addressed CIF bytes and explicit manual cell records."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("XDART_PHASE_LIBRARY", Path.home()/".xdart"/"phases"))

    def add_cif(self, path: str | Path, *, name: str | None = None) -> PhaseEntry:
        path = Path(path)
        with path.open("rb") as handle:
            raw = handle.read(MAX_CIF_BYTES+1)
        _parse_cif(raw)
        identity = hashlib.sha256(raw).hexdigest()
        name = _name(name or path.stem)
        target = self.path/(identity+".cif")
        _atomic_write(target, raw)
        _atomic_write(self.path/(identity+".json"), json.dumps({
            "id": identity, "name": name, "source_kind": "cif",
        }, sort_keys=True).encode())
        return PhaseEntry(identity, name, "cif", raw, cif_path=target)

    def add_lattice(self, name: str, *, a: float, b: float | None = None,
                    c: float | None = None, alpha: float = 90., beta: float = 90.,
                    gamma: float = 90., space_group: int | str = 1) -> PhaseEntry:
        from .phase import HAS_PYMATGEN, Lattice
        if not HAS_PYMATGEN:
            raise ValueError("pymatgen is required for lattice phases")
        from pymatgen.symmetry.groups import SpaceGroup
        values = dict(a=float(a), b=float(a if b is None else b), c=float(a if c is None else c),
                      alpha=float(alpha), beta=float(beta), gamma=float(gamma))
        if any(not math.isfinite(v) or v <= 0 for v in values.values()) or any(values[k] >= 180 for k in ("alpha", "beta", "gamma")):
            raise ValueError("invalid lattice parameters")
        if isinstance(space_group, str) and space_group.strip().isdigit():
            space_group = int(space_group)
        try:
            sg = SpaceGroup(space_group) if isinstance(space_group, str) else SpaceGroup.from_int_number(int(space_group))
            lattice = Lattice.from_parameters(**values)
            if not math.isfinite(lattice.volume) or lattice.volume <= 0:
                raise ValueError("invalid lattice volume")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("invalid lattice or space group") from exc
        values["space_group"] = sg.symbol
        name = _name(name)
        document = {"name": name, "source_kind": "lattice", "lattice": values}
        identity = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
        document["id"] = identity
        _atomic_write(self.path/(identity+".json"), json.dumps(document, sort_keys=True).encode())
        return PhaseEntry(identity, name, "lattice", lattice=tuple(sorted(values.items())))

    def resolve(self, identity: str) -> PhaseLibraryLookup:
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            return PhaseLibraryLookup(reason="Phase library entry is missing")
        try:
            document = json.loads((self.path/(identity+".json")).read_text())
            name, kind = _name(document["name"]), document["source_kind"]
            if document["id"] != identity:
                raise ValueError("phase identity mismatch")
            if kind == "cif":
                path = self.path/(identity+".cif")
                with path.open("rb") as handle:
                    raw = handle.read(MAX_CIF_BYTES+1)
                if len(raw) > MAX_CIF_BYTES or hashlib.sha256(raw).hexdigest() != identity:
                    raise ValueError("CIF bytes changed")
                return PhaseLibraryLookup(PhaseEntry(identity, name, kind, raw, cif_path=path))
            if kind == "lattice":
                return PhaseLibraryLookup(PhaseEntry(identity, name, kind, lattice=tuple(sorted(document["lattice"].items()))))
            raise ValueError("unknown phase source")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return PhaseLibraryLookup(reason=f"Phase library entry is missing or invalid: {exc}")

    def entries(self) -> tuple[PhaseEntry, ...]:
        return tuple(lookup.entry for path in sorted(self.path.glob("*.json"))
                     if (lookup := self.resolve(path.stem)).entry is not None)


def prepare_phase_models(*entries: PhaseEntry, wavelength_angstrom: float,
                         q_range=None, two_theta_range=None, min_intensity=5,
                         model_class=None, structure_class=None):
    """The sole source/listing path shared by overlays and the phase fit."""
    from .phase import PhaseModel
    model_class = PhaseModel if model_class is None else model_class
    if not 1 <= len(entries) <= MAX_PHASES:
        raise ValueError("phase limit exceeded (at most 8)")
    if not math.isfinite(float(min_intensity)) or not 0 <= min_intensity <= 100:
        raise ValueError("minimum intensity must be between 0 and 100 percent")
    if sum(len(e.cif_bytes) for e in entries) > MAX_TOTAL_CIF_BYTES:
        raise ValueError("cumulative CIF byte limit exceeded")
    models, references, total = [], [], 0
    for entry in entries:
        if entry.source_kind == "cif":
            model = model_class(entry.name)
            model.structure = _parse_cif(entry.cif_bytes, structure_class)
            model.source_kind = "cif"
        elif entry.source_kind == "lattice":
            # Validate the cell without first listing the full default sphere.
            from .phase import Lattice
            from pymatgen.symmetry.groups import SpaceGroup
            values = dict(entry.lattice)
            group = values.pop("space_group")
            model = model_class(entry.name)
            model.lattice = Lattice.from_parameters(**values)
            model.space_group = SpaceGroup(group) if isinstance(group, str) else SpaceGroup.from_int_number(group)
            model.source_kind = "lattice"
        else:
            raise ValueError("unknown phase source")
        model.calculate_peaks(wavelength=wavelength_angstrom,
                              two_theta_range=(0, 180) if two_theta_range is None else two_theta_range,
                              q_range=q_range, max_reflections=MAX_REFLECTIONS_PER_PHASE if entry.source_kind == "lattice" else None)
        admitted = tuple(p for p in model.peaks if entry.source_kind == "lattice" or p.intensity >= min_intensity)
        if len(admitted) > MAX_REFLECTIONS_PER_PHASE:
            raise ValueError("per-phase reflection limit exceeded")
        total += len(admitted)
        if total > MAX_REFLECTIONS_TOTAL:
            raise ValueError("cumulative reflection limit exceeded")
        references.append(PhaseReference(entry.id, entry.name, entry.source_kind, tuple(
            PhaseReflection(p.q, p.d_spacing, None if entry.source_kind == "lattice" else p.intensity, p.hkl)
            for p in admitted), len(model.peaks)-len(admitted)))
        model.peaks = list(admitted)
        models.append(model)
    return tuple(models), tuple(references)


def build_phase_references(*entries: PhaseEntry, wavelength_angstrom: float,
                           q_range=None, two_theta_range=None, min_intensity=5) -> PhaseReferences:
    try:
        _, phases = prepare_phase_models(*entries, wavelength_angstrom=wavelength_angstrom,
                                         q_range=q_range, two_theta_range=two_theta_range,
                                         min_intensity=min_intensity)
        return PhaseReferences(phases)
    except (ValueError, TypeError, OSError, ImportError) as exc:
        return PhaseReferences(reason=str(exc))


def phase_model(entry: PhaseEntry, *, wavelength_angstrom=1.5406,
                q_range=None, two_theta_range=None):
    """Build a real fit model from the same source/listing as the overlays."""
    models, _ = prepare_phase_models(entry, wavelength_angstrom=wavelength_angstrom,
                                     q_range=q_range, two_theta_range=two_theta_range,
                                     min_intensity=0.)
    return models[0]
