"""Lossless transport for the selected derived input, never a raw detector image or full RSM volume.

The small NPZ has numeric arrays and JSON bytes, loaded with ``allow_pickle=False``. Its receipt checks these
captured bytes, not a raw source tree. GUI generations are deliberately excluded from the scientific receipt.
The caller publishes the sidecar alongside the notebook; this module does no filesystem writes.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot

_MAX_BYTES = 32 * 1024 * 1024
_SCHEMA = "fit-notebook-input-v1"


class NotebookInputError(ValueError):
    """An exported scientific input cannot be reconstructed exactly (named reason in the message)."""


@dataclass(frozen=True, slots=True)
class SnapshotReference:
    path: str
    sha256: str
    label: str

    def __post_init__(self):
        if any(type(item) is not str for item in (self.path, self.sha256, self.label)):
            raise TypeError("snapshot reference fields are text")
        if not self.path or Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ValueError("a captured input path is relative to its explicit bundle root")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("a captured input receipt is a sha256 of its derived sidecar bytes")


@dataclass(frozen=True, slots=True)
class SourceReference:
    """A cheap durable extraction descriptor using the admitted artifact's existing scientific receipts.

    ``path`` may be project-relative (resolved against DATA_ROOT) or an explicitly external absolute path.
    ``selection`` is the shared fit_sources owner's arguments, never an executable callback or array. Recorded
    size/mtime are a cheap guard at the original path. The public artifact reader retains its recorded-path
    admission and can refuse a relocated container; the bounded captured NPZ transport is portable.
    """
    path: str
    reader: str
    label: str
    result_fingerprint: str
    request_fingerprint: str
    selection: tuple[tuple[str, object], ...] = ()
    file_facts: tuple[int, int] | None = None
    original_path: str = ""
    background: str = "none"

    def __post_init__(self):
        if self.reader not in {"stitch", "rsm_lineout", "rsm_region"}:
            raise ValueError("source reader is stitch, rsm_lineout or rsm_region")
        if any(type(item) is not str for item in (self.path, self.label, self.result_fingerprint,
                                                  self.request_fingerprint, self.original_path)):
            raise TypeError("source reference identity fields are text")
        # FitInputKey is the shared primitive selection/background validator; no new metadata schema parser.
        key = FitInputKey(self.label, self.path, self.result_fingerprint, 0, self.selection, self.background)
        if key.background != "none":
            raise NotebookInputError("SOURCE_DERIVED: capture a sidecar for presentation-background adjusted input")
        if self.file_facts is not None and (type(self.file_facts) is not tuple or len(self.file_facts) != 2
                                          or any(type(v) is not int for v in self.file_facts)):
            raise TypeError("source file facts are the recorded size and mtime_ns")


@dataclass(frozen=True, slots=True)
class CapturedInput:
    reference: SnapshotReference
    content: bytes


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def capture_snapshot(snapshot: FitInputSnapshot, path: str) -> CapturedInput:
    """Prepare one bounded, selected input sidecar without writing or requiring a successful fit.

    A rank-three input must be the selected fitting box (the same 100,000-voxel cap as the public solver).
    A whole volume is not an export input. Optional admitted sigma remains absent when absent.
    """
    if type(snapshot) is not FitInputSnapshot:
        raise TypeError("export captures a detached FitInputSnapshot")
    if snapshot.values.ndim > 3 or snapshot.values.size > (100_000 if snapshot.values.ndim == 3 else 1_000_000):
        raise NotebookInputError("INPUT_LIMIT: export a bounded trace, slice or fitting box")
    key = snapshot.key
    from xrd_tools import __version__
    metadata = {"schema": _SCHEMA, "key": {"result_identity": key.result_identity, "path": key.path,
                "fingerprint": key.fingerprint, "selection": key.selection, "background": key.background,
                "settings": key.settings}, "axes": [(axis.name, axis.unit) for axis in snapshot.axes],
                "wavelength_angstrom": snapshot.wavelength_angstrom, "xrd_tools_version": __version__}
    raw = json.dumps(metadata, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    arrays = {"values": snapshot.values, "metadata": np.frombuffer(raw, dtype=np.uint8)}
    arrays.update((f"axis_{i}", axis.values) for i, axis in enumerate(snapshot.axes))
    if snapshot.mask is not None:
        arrays["mask"] = snapshot.mask
    sigma = getattr(snapshot, "sigma", None)
    if sigma is not None:
        arrays["sigma"] = sigma
    if sum(value.nbytes for value in arrays.values()) > _MAX_BYTES:
        raise NotebookInputError("INPUT_LIMIT: selected input exceeds the notebook byte limit")
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    content = stream.getvalue()
    return CapturedInput(SnapshotReference(path, hashlib.sha256(content).hexdigest(), key.result_identity), content)


def load_snapshot(reference: SnapshotReference | SourceReference, *, bundle_root: str | Path,
                  data_root: str | Path | None = None) -> FitInputSnapshot:
    """Read the exact captured input using an explicit bundle root (independent of the kernel's cwd)."""
    if type(reference) is SourceReference:
        return _load_source(reference, data_root=bundle_root if data_root is None else data_root)
    path = Path(bundle_root) / reference.path
    try:
        if path.stat().st_size > _MAX_BYTES:
            raise NotebookInputError("INPUT_LIMIT: captured sidecar exceeds its byte limit")
        content = path.read_bytes()
    except OSError as error:
        raise NotebookInputError(f"INPUT_MISSING: {path}") from error
    if hashlib.sha256(content).hexdigest() != reference.sha256:
        raise NotebookInputError(f"INPUT_CHANGED: {path}; this is a new input, not the captured replay")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if sum(item.file_size for item in archive.infolist()) > _MAX_BYTES:
                raise NotebookInputError("INPUT_LIMIT: expanded sidecar exceeds its byte limit")
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            meta = json.loads(data["metadata"].tobytes().decode("utf-8"))
            if meta["schema"] != _SCHEMA or meta["key"]["result_identity"] != reference.label:
                raise NotebookInputError("INPUT_RECEIPT: captured scientific identity differs")
            key = dict(meta["key"])
            key["generation"] = 0
            for name in ("selection", "settings"):
                key[name] = _tuples(key[name])
            axes = tuple(FitAxis(name, unit, data[f"axis_{i}"]) for i, (name, unit) in enumerate(meta["axes"]))
            keywords = {}
            if "sigma" in data.files:
                keywords["sigma"] = data["sigma"]
            return FitInputSnapshot(FitInputKey(**key), axes, data["values"],
                                    data["mask"] if "mask" in data.files else None,
                                    meta["wavelength_angstrom"], **keywords)
    except NotebookInputError:
        raise
    except (ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        raise NotebookInputError(f"INPUT_INVALID: {path}: {error}") from error


def _load_source(reference: SourceReference, *, data_root):
    from xrd_tools.io.analysis_artifact import read_analysis_artifact, AnalysisArtifactError
    from xrd_tools.analysis import fit_sources
    path = Path(reference.path)
    if not path.is_absolute():
        path = Path(data_root) / path
    try:
        before = path.stat()
    except OSError as error:
        raise NotebookInputError(f"INPUT_MISSING: {path}") from error
    facts = (before.st_size, before.st_mtime_ns)
    if reference.original_path and path.absolute() == Path(reference.original_path).absolute():
        if reference.file_facts is not None and facts != reference.file_facts:
            raise NotebookInputError(f"INPUT_CHANGED: {path}; source is a new admitted input")
    try:
        payload = read_analysis_artifact(path)
    except (AnalysisArtifactError, ValueError, OSError) as error:
        raise NotebookInputError(f"INPUT_SOURCE_INVALID: {path}: {error}") from error
    after = path.stat()
    if facts != (after.st_size, after.st_mtime_ns) or payload.result_fingerprint != reference.result_fingerprint:
        raise NotebookInputError(f"INPUT_CHANGED: {path}; source scientific receipt differs")
    if payload.inspection.request_fingerprint != reference.request_fingerprint:
        raise NotebookInputError(f"INPUT_CHANGED: {path}; source request receipt differs")
    key = FitInputKey(reference.label, str(path), reference.result_fingerprint, 0, reference.selection)
    selection = dict(reference.selection)
    function = {"stitch": fit_sources.stitch_fit_snapshot, "rsm_lineout": fit_sources.rsm_lineout_snapshot,
                "rsm_region": fit_sources.rsm_region_snapshot}[reference.reader]
    fields = {"stitch": ("coordinate", "diagnostic", "phase"), "rsm_lineout": ("axis", "centre", "width"),
              "rsm_region": ("bounds", "plane", "index")}[reference.reader]
    return function(payload, key, **{name: selection[name] for name in fields if name in selection})
