# -*- coding: utf-8 -*-
"""Headless store/provider read-authority projections (X1-1/2/3 foundation).

One Qt-free place to read the per-frame facts the GUI (and notebooks, reduction,
services) need — metadata rows, normalization channels, wavelength evidence, and
typed display capabilities — from the authoritative
:class:`~xrd_tools.session.frame_record_store.FrameRecordStore` and the
:class:`~xrd_tools.sources.metadata_provider.MetadataProvider`, WITHOUT opening a
file, building a whole-scan table, or consulting ``PublicationStore`` / GUI
``scan_data``.

This module is a projection, not a store: it holds no state, performs no I/O,
mutates nothing, and adds no authoritative cache.  Everything here is a pure
function of already-produced value types.

Four surfaces:

* **A — metadata row** (:class:`MetadataRow`, :func:`metadata_row_from_view` /
  :func:`metadata_row_from_provider` / :func:`metadata_row_from_record`): one
  immutable per-frame metadata value.  Provider composition combines counters,
  constants, and scanned motors with the exact precedence
  ``scanned per-frame motor > per-frame table/counter > constant``; frame
  identities are bounds-checked and negative indices never wrap.
* **B — normalization** (:func:`normalization_channels` /
  :func:`normalization_value`): the candidate monitor-normalization channels of a
  row and the guarded value for a selected channel, reusing
  :func:`~xrd_tools.core.metadata.numeric_metadata` /
  :func:`~xrd_tools.core.metadata.resolve_monitor_norm` unchanged.
* **C — wavelength + capabilities** (:func:`wavelength_evidence`,
  :func:`display_capabilities`): value-only wavelength evidence (present / absent
  / conflict, no file I/O, no optional-field warning) and typed display
  capabilities (metadata, integrated 1D, integrated 2D, raw pixels,
  thumbnail/source-fallback) with explicit states, identity qualification, and a
  reason; resident data is distinguished from recoverable/hydratable data.
* **D — one store lookup** (:func:`project_frame`): a single typed projection
  entry point over :class:`FrameRecordStore` resident/hydration semantics.
  Hydration is synchronous and explicit; the caller owns any worker thread.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np

from xrd_tools.core import FrameRecord, FrameView
from xrd_tools.core.metadata import numeric_metadata, resolve_monitor_norm

__all__ = [
    "MetadataRow",
    "CapabilityState",
    "Capability",
    "DisplayCapabilities",
    "WavelengthEvidence",
    "FrameProjection",
    "MetadataConflictError",
    "metadata_row_from_view",
    "metadata_row_from_provider",
    "metadata_row_from_record",
    "normalization_channels",
    "normalization_value",
    "wavelength_evidence",
    "display_capabilities",
    "project_frame",
]


class MetadataConflictError(ValueError):
    """Two views / sources for one frame carry contradictory metadata or
    source identity — the projection refuses to silently pick one."""


# ── helpers ────────────────────────────────────────────────────────────────

def _scalar(value: Any) -> Any:
    """A 0-d numpy value → plain python scalar; other values pass through."""
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        return item
    return value


def _freeze_metadata_value(value: Any) -> Any:
    """Detach mutable metadata containers from their producer.

    ``FrameView`` makes its top-level metadata mapping read-only, but legacy
    metadata can still contain mutable arrays or containers.  A projection is a
    value boundary: callers must not be able to mutate a stored record through
    it.
    """
    if isinstance(value, np.ndarray):
        frozen = np.array(value, copy=True)
        frozen.setflags(write=False)
        return frozen
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_metadata_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata_value(item) for item in value)
    return value


def _safe_len(value: Any) -> int | None:
    """``len(value)`` (O(1) for list/ndarray) or ``None`` if it has no length."""
    try:
        return len(value)
    except TypeError:
        return None


def _finite_positive(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if np.isfinite(num) and num > 0.0 else None


def _values_conflict(a: Any, b: Any) -> bool:
    """True iff two raw metadata values for the SAME key genuinely disagree.

    Scalar-numeric pairs compare with a tight tolerance (a re-derived float is
    not a conflict); array-valued pairs compare element-wise (equal arrays are
    NOT a conflict); a scalar vs an array is a conflict; everything else
    compares by equality, and an unorderable comparison failure is treated as a
    conflict (fail closed)."""
    fa, fb = _as_float(a), _as_float(b)
    if fa is not None and fb is not None:
        if np.isnan(fa) and np.isnan(fb):
            return False
        return not np.isclose(fa, fb, rtol=1e-9, atol=0.0, equal_nan=True)
    aa, ba = _as_ndarray(a), _as_ndarray(b)
    if aa is not None and ba is not None:
        if aa.shape != ba.shape:
            return True
        try:
            if aa.dtype.kind == "f" and ba.dtype.kind == "f":
                return not np.array_equal(aa, ba, equal_nan=True)
            return not np.array_equal(aa, ba)
        except Exception:
            return True
    if (aa is None) != (ba is None):
        return True                       # scalar vs array
    try:
        return bool(a != b)
    except Exception:
        return True


def _as_float(value: Any) -> float | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        arr = np.asarray(value)
        if arr.shape != ():
            return None
        return float(arr)
    except (TypeError, ValueError):
        return None


def _as_ndarray(value: Any) -> np.ndarray | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    return arr if arr.ndim >= 1 else None


def _assert_no_metadata_conflict(views: tuple[FrameView, ...]) -> None:
    """Every key present in more than one view must agree (X1-A: fail clearly
    on a contradictory metadata value across a frame's modes)."""
    seen: dict[str, tuple[Any, int]] = {}
    for i, view in enumerate(views):
        for key, value in (view.metadata_raw or {}).items():
            k = str(key)
            if k in seen:
                prior, j = seen[k]
                if _values_conflict(prior, value):
                    raise MetadataConflictError(
                        f"frame carries contradictory metadata for {k!r}: "
                        f"view[{j}]={prior!r} != view[{i}]={value!r}")
            else:
                seen[k] = (value, i)


def _record_source_identity(record: FrameRecord) -> str:
    """The single source identity of a record's views, or "" if none.

    A view with a source path but an UNKNOWN (``None``) frame index reconciles
    with a same-path view that knows the index — mirroring the codebase's own
    ``_merge_views`` rule (``v1 if not None else v2``); it is NOT a conflict.
    Raises :class:`MetadataConflictError` only on a genuine disagreement: two
    different source paths, or two different CONCRETE frame indices for one path
    (X1-A/C)."""
    views = tuple(record.results_1d.values()) + tuple(record.results_2d.values())
    by_path: dict[str, set[int]] = {}
    for v in views:
        if v.source_path is None:
            continue
        indices = by_path.setdefault(str(v.source_path), set())
        if v.source_frame_index is not None:
            indices.add(int(v.source_frame_index))
    if len(by_path) > 1:
        raise MetadataConflictError(
            f"frame record {record.label!r} carries conflicting source paths: "
            f"{sorted(by_path)!r}")
    if not by_path:
        return ""
    path, indices = next(iter(by_path.items()))
    if len(indices) > 1:
        raise MetadataConflictError(
            f"frame record {record.label!r} carries conflicting source frame "
            f"indices for {path!r}: {sorted(indices)!r}")
    idx = next(iter(indices), None)
    return f"{path}#{'' if idx is None else idx}"


# ── A — immutable metadata-row projection ────────────────────────────────────

@dataclass(frozen=True, slots=True)
class MetadataRow:
    """An immutable per-frame metadata value: heterogeneous ``raw`` preserved,
    plus a finite scalar ``numeric`` view (the existing metadata rules)."""

    raw: Mapping[str, Any] = field(default_factory=dict)
    numeric: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = MappingProxyType({
            str(key): _freeze_metadata_value(value)
            for key, value in (self.raw or {}).items()
        })
        numeric = dict(self.numeric) if self.numeric else numeric_metadata(raw)
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "numeric", MappingProxyType(numeric))

    def __bool__(self) -> bool:
        return bool(self.raw)


_EMPTY_ROW = MetadataRow()


def metadata_row_from_view(view: FrameView) -> MetadataRow:
    """The metadata row of a resident/hydrated :class:`FrameView` — it already
    carries ``metadata_raw`` (+ a numeric view) populated once at
    ingestion/browse; this consumes those values, it does not rebuild a table."""
    return MetadataRow(view.metadata_raw, view.metadata_numeric)


def metadata_row_from_provider(provider: Any, frame_index: int) -> MetadataRow:
    """Compose one frame's row from an owner-scoped ``MetadataProvider``.

    Precedence ``scanned per-frame motor > per-frame table/counter > constant``:
    the provider's ``metadata_for(index)`` already overlays per-frame counters on
    constants (counter wins) and excludes scanned motors, so the scanned motor
    value at ``frame_index`` is overlaid last (it wins).  The frame identity is
    bounded by the provider's AUTHORITATIVE ``frame_count()`` (falling back to
    per-frame array lengths only when the provider does not report a count), so
    the row agrees with what ``metadata_for`` considers a valid frame; a NEGATIVE
    index is rejected — never wrapped.  A scanned-motor column shorter than the
    frame count (a baseline/aborted point) is simply absent for that frame,
    mirroring the provider's own dual guard.  Only the single element at
    ``frame_index`` is read, so a single-frame row is O(1) in scan length.
    """
    idx = int(frame_index)
    if idx < 0:
        raise IndexError(
            f"frame index {idx} is negative; refusing to wrap to the scan tail")

    frame_count = _provider_call(provider, "frame_count")
    if frame_count is not None:
        if idx >= int(frame_count):
            raise IndexError(
                f"frame index {idx} is out of range for frame count "
                f"{int(frame_count)}")
    complete_row = getattr(provider, "complete_metadata_for", None)
    motors: Mapping[str, Any] = {}
    if frame_count is None:
        motors = _provider_call(provider, "motors") or {}
        lengths = [n for n in (
            _safe_len(v) for v in list(motors.values())
            + list((_provider_call(provider, "scan_table") or {}).values()))
            if n is not None]
        if lengths and idx >= max(lengths):
            raise IndexError(
                f"frame index {idx} is out of range for scan length {max(lengths)}")
    elif callable(complete_row):
        return MetadataRow(complete_row(idx) or {})

    if not motors:
        motors = _provider_call(provider, "motors") or {}

    raw: dict[str, Any] = dict(_provider_call(provider, "metadata_for", idx) or {})
    for name, arr in motors.items():          # scanned motor wins (overlaid last)
        n = _safe_len(arr)
        if n is None or idx >= n:              # baseline-short column → absent here
            continue
        try:
            raw[str(name)] = _scalar(arr[idx])  # index ONE element (O(1))
        except (TypeError, ValueError, IndexError, KeyError):
            pass
    return MetadataRow(raw, numeric_metadata(raw))


def _provider_call(provider: Any, name: str, *args: Any) -> Any:
    fn = getattr(provider, name, None)
    if fn is None:
        return None if name in ("wavelength", "frame_count") else {}
    return fn(*args)


def metadata_row_from_record(
    record: FrameRecord,
    *,
    mode_1d: str | None = None,
    mode_2d: str | None = None,
) -> MetadataRow:
    """One frame record's metadata row, deterministically.

    Contract when multiple modes carry metadata: prefer the EXPLICITLY requested
    view (``mode_1d``/``mode_2d``), else the active 1D view, else the active 2D
    view.  Every mode on the frame is checked for a contradictory metadata value
    or source identity first; a real conflict raises
    :class:`MetadataConflictError` rather than silently taking whichever dict
    iterates first.
    """
    views = tuple(record.results_1d.values()) + tuple(record.results_2d.values())
    _assert_no_metadata_conflict(views)
    _record_source_identity(record)           # raises on conflicting identity

    # Fallback chain: prefer the explicitly requested view(s), then the active
    # 1D, then the active 2D.  An absent explicit mode falls THROUGH to the
    # active view (metadata is frame-level and shared across modes) rather than
    # silently dropping a frame's real metadata.
    candidates: list[FrameView | None] = []
    if mode_1d is not None:
        candidates.append(record.view_1d(mode_1d))
    if mode_2d is not None:
        candidates.append(record.view_2d(mode_2d))
    candidates.append(record.view_1d())       # active 1D
    candidates.append(record.view_2d())       # active 2D
    primary = next((v for v in candidates if v is not None), None)
    if primary is None:
        return _EMPTY_ROW
    return metadata_row_from_view(primary)


# ── B — store-only normalization facts ───────────────────────────────────────

def normalization_channels(row: MetadataRow) -> tuple[str, ...]:
    """Candidate monitor-normalization channel names of a row — its finite
    scalar-numeric keys (the existing ``numeric_metadata`` rule), sorted for a
    stable order.  Whether a selected channel yields a usable value is guarded
    separately by :func:`normalization_value`."""
    return tuple(sorted(row.numeric))


def normalization_value(row: MetadataRow, channel: str | None) -> float | None:
    """The guarded monitor-normalization value for ``channel`` in ``row``.

    Delegates to :func:`~xrd_tools.core.metadata.resolve_monitor_norm` unchanged
    (case-insensitive; ``None`` for a missing / nonnumeric / non-finite / zero /
    negative value) so normalization semantics are not forked."""
    return resolve_monitor_norm(row.raw, channel)


# ── C — wavelength + capability projection ───────────────────────────────────

class _WavelengthStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class WavelengthEvidence:
    """Value-only wavelength evidence available WITHOUT opening a file.

    ``value`` is the agreed wavelength when at least one source reports a finite
    positive value and all reporting sources agree; ``status`` is
    ``present`` / ``absent`` / ``conflict``; ``sources`` maps each contributing
    source label to its value.  A missing wavelength is ``absent`` — it is not a
    warning here (final processing readiness owns the operator-facing error after
    the selected PONI/Experiment authority is considered).  Unit reconciliation
    is the caller's responsibility; this reports values as provided.
    """

    value: float | None
    status: str
    sources: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))

    @property
    def present(self) -> bool:
        return self.status == _WavelengthStatus.PRESENT.value

    @property
    def conflict(self) -> bool:
        return self.status == _WavelengthStatus.CONFLICT.value


def wavelength_evidence(
    *,
    view: FrameView | None = None,
    provider: Any | None = None,
    row: MetadataRow | None = None,
) -> WavelengthEvidence:
    """Collect wavelength evidence from a view/provider/row without I/O.

    Sources: the provider's ``wavelength()``, and a finite-positive numeric
    ``wavelength`` key in the frame metadata (from ``row`` or ``view``).  No PONI
    (meters, H20-owned) source is consulted here.  Agreement (within a tight
    tolerance) → ``present``; disagreement → ``conflict``; none → ``absent``.
    """
    sources: dict[str, float] = {}
    if provider is not None:
        pw = _finite_positive(_provider_call(provider, "wavelength"))
        if pw is not None:
            sources["provider"] = pw
    meta = row if row is not None else (
        metadata_row_from_view(view) if view is not None else None)
    if meta is not None:
        for key, value in meta.numeric.items():
            if str(key).lower() == "wavelength":
                mw = _finite_positive(value)
                if mw is not None:
                    sources["metadata"] = mw
                break

    values = list(sources.values())
    if not values:
        return WavelengthEvidence(None, _WavelengthStatus.ABSENT.value, sources)
    # Agreement tolerance is generous (1e-3 relative) so a rounded header value
    # (e.g. 1.5406) agrees with a full-precision one (1.540598); genuinely
    # different values still surface as a conflict.  Cross-UNIT sources are the
    # caller's responsibility (documented) — X1 does not own unit/PONI ownership.
    if all(np.isclose(values[0], v, rtol=1e-3, atol=0.0) for v in values):
        return WavelengthEvidence(
            values[0], _WavelengthStatus.PRESENT.value, sources)
    return WavelengthEvidence(None, _WavelengthStatus.CONFLICT.value, sources)


class CapabilityState(str, Enum):
    """State of one typed display-capability fact."""

    AVAILABLE = "available"        # resident right now
    PENDING = "pending"           # not resident but recoverable / hydratable
    UNAVAILABLE = "unavailable"   # no such data and not recoverable
    ERROR = "error"               # a typed error (e.g. conflicting identity)


@dataclass(frozen=True, slots=True)
class Capability:
    """One capability fact: a :class:`CapabilityState` plus a reason."""

    state: CapabilityState
    reason: str = ""


@dataclass(frozen=True, slots=True)
class DisplayCapabilities:
    """Typed display-capability facts for one selected record.

    Each fact distinguishes resident data (``available``) from
    recoverable/hydratable data (``pending``); ``identity`` qualifies the
    evidence (the record's single source identity, or "" / a conflict reason).
    """

    identity: str
    metadata: Capability
    integrated_1d: Capability
    integrated_2d: Capability
    raw: Capability
    thumbnail: Capability


def _integrated_capability(
    record: FrameRecord, dim: str, mode: str | None,
    hydratable_modes: frozenset[tuple[str, str]] | None,
) -> Capability:
    views = record.results_1d if dim == "1d" else record.results_2d
    if not views:
        return Capability(CapabilityState.UNAVAILABLE, f"no {dim} result mode")
    resolved_mode = mode if mode is not None else (
        record.active_mode_1d if dim == "1d" else record.active_mode_2d)
    view = record.view_1d(mode) if dim == "1d" else record.view_2d(mode)
    if view is None:
        return Capability(
            CapabilityState.UNAVAILABLE, f"{dim} mode {mode!r} not present")
    resident = (view.intensity_1d if dim == "1d" else view.intensity_2d) is not None
    if resident:
        return Capability(CapabilityState.AVAILABLE, "resident")
    # Thinned: recoverable ONLY if this mode is persisted on disk.  A
    # consciously-dropped mode (mark_dropped / MEM-1b) is not persisted and not
    # hydratable → UNAVAILABLE, not a false PENDING.  Without store context
    # (hydratable_modes None) recoverability is unknown → best-effort PENDING.
    if hydratable_modes is None:
        return Capability(
            CapabilityState.PENDING, "thinned; hydratable if persisted")
    if (dim, resolved_mode) in hydratable_modes:
        return Capability(CapabilityState.PENDING, "thinned; hydratable from store")
    return Capability(
        CapabilityState.UNAVAILABLE, "thinned and not persisted (dropped)")


def display_capabilities(
    record: FrameRecord,
    *,
    source_identity: str | None = None,
    mode_1d: str | None = None,
    mode_2d: str | None = None,
    hydratable_modes: frozenset[tuple[str, str]] | None = None,
) -> DisplayCapabilities:
    """Typed, immutable display-capability facts for ``record`` — a PURE
    projection: no I/O, no store mutation, no hydration.

    A capability is inferred ONLY from actual resident/recoverable evidence,
    never from the mere presence of a label/title.  ``hydratable_modes`` is the
    store's per-mode persisted set (``FrameRecordStore.persisted_modes``): a
    thinned mode is ``pending`` only when its ``(dim, mode)`` key is hydratable,
    else ``unavailable`` (a consciously dropped mode is not recoverable).  With
    ``hydratable_modes=None`` (no store context) a thinned mode is best-effort
    ``pending``.  A record whose views carry conflicting source identities is an
    ERROR record (every fact ``error``).
    """
    try:
        identity = _record_source_identity(record)
    except MetadataConflictError as exc:
        err = Capability(CapabilityState.ERROR, str(exc))
        return DisplayCapabilities("<conflict>", err, err, err, err, err)
    if source_identity and identity and source_identity != identity:
        err = Capability(
            CapabilityState.ERROR,
            "store/source identity disagrees with the selected record: "
            f"{source_identity!r} != {identity!r}",
        )
        return DisplayCapabilities("<conflict>", err, err, err, err, err)
    identity = source_identity if source_identity else identity

    views = tuple(record.results_1d.values()) + tuple(record.results_2d.values())
    row = _EMPTY_ROW
    try:
        row = metadata_row_from_record(record, mode_1d=mode_1d, mode_2d=mode_2d)
    except MetadataConflictError as exc:
        err = Capability(CapabilityState.ERROR, str(exc))
        return DisplayCapabilities(identity or "<conflict>", err, err, err, err, err)
    metadata = (Capability(CapabilityState.AVAILABLE, "resident") if row
                else Capability(CapabilityState.UNAVAILABLE, "no metadata"))

    integrated_1d = _integrated_capability(record, "1d", mode_1d, hydratable_modes)
    integrated_2d = _integrated_capability(record, "2d", mode_2d, hydratable_modes)

    raw_resident = any(v.raw is not None for v in views)
    source_eligible = any(v.source_path is not None for v in views)
    if raw_resident:
        raw = Capability(CapabilityState.AVAILABLE, "resident raw pixels")
    elif source_eligible:
        raw = Capability(
            CapabilityState.PENDING, "source-fallback eligible (not store-hydratable)")
    else:
        raw = Capability(CapabilityState.UNAVAILABLE, "no raw pixels, no source")

    thumb_resident = any(v.thumbnail is not None for v in views)
    if thumb_resident:
        thumbnail = Capability(CapabilityState.AVAILABLE, "resident thumbnail")
    elif source_eligible:
        thumbnail = Capability(
            CapabilityState.PENDING, "source-fallback eligible")
    else:
        thumbnail = Capability(CapabilityState.UNAVAILABLE, "no thumbnail evidence")

    return DisplayCapabilities(
        identity, metadata, integrated_1d, integrated_2d, raw, thumbnail)


# ── D — one public store lookup boundary ─────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FrameProjection:
    """The single typed projection of one frame from a :class:`FrameRecordStore`."""

    label: int | str
    present: bool
    metadata: MetadataRow
    capabilities: DisplayCapabilities
    wavelength: WavelengthEvidence
    normalization_channels: tuple[str, ...]


def project_frame(
    store: Any,
    label: int | str,
    *,
    mode_1d: str | None = None,
    mode_2d: str | None = None,
    hydrate: bool = False,
    provider: Any | None = None,
    provider_frame_index: int | None = None,
) -> FrameProjection:
    """Project one frame's read-authority facts from ``store``.

    This is the single lookup boundary (X1-D): it uses only the
    :class:`FrameRecordStore` resident (:meth:`get`) or, when ``hydrate=True``,
    synchronous hydration (:meth:`get_or_hydrate`) — never ``PublicationStore``.
    Hydration is synchronous and explicit; a disk-backed hydrator must be driven
    from the caller's worker thread, not here.  A single-frame projection reads
    ONE record and its selected view; it does not build a whole-scan table or
    trigger O(scan) hydration.
    """
    record = store.get_or_hydrate(label) if hydrate else store.get(label)
    if record is None:
        absent = Capability(CapabilityState.UNAVAILABLE, "record not present")
        return FrameProjection(
            label=label, present=False, metadata=_EMPTY_ROW,
            capabilities=DisplayCapabilities(
                "", absent, absent, absent, absent, absent),
            wavelength=WavelengthEvidence(None, _WavelengthStatus.ABSENT.value),
            normalization_channels=())

    identity = ""
    try:
        identity = store.source_identity(label)
    except Exception:
        identity = ""
    hydratable: frozenset[tuple[str, str]] | None = None
    getter = getattr(store, "hydratable_modes", None)
    if not callable(getter):
        getter = getattr(store, "persisted_modes", None)
    if callable(getter):
        try:
            hydratable = frozenset(getter(label))
        except Exception:
            hydratable = None
    # A conflicting record surfaces as a typed ERROR projection (via
    # display_capabilities), never as a raised exception at this boundary.
    row_error: MetadataConflictError | None = None
    try:
        row = metadata_row_from_record(record, mode_1d=mode_1d, mode_2d=mode_2d)
        if provider is not None:
            selected_view = _selected_record_view(record, mode_1d, mode_2d)
            index = provider_frame_index
            if index is None and selected_view is not None:
                index = selected_view.source_frame_index
            if index is not None:
                provider_row = metadata_row_from_provider(provider, index)
                row = _merge_metadata_rows(row, provider_row)
    except MetadataConflictError as exc:
        row_error = exc
        row = _EMPTY_ROW
    caps = display_capabilities(
        record, source_identity=identity, mode_1d=mode_1d, mode_2d=mode_2d,
        hydratable_modes=hydratable)
    if caps.metadata.state is CapabilityState.ERROR:
        row = _EMPTY_ROW
    if row_error is not None and caps.metadata.state is not CapabilityState.ERROR:
        caps = replace(
            caps,
            metadata=Capability(CapabilityState.ERROR, str(row_error)),
        )
    elif row and caps.metadata.state is CapabilityState.UNAVAILABLE:
        caps = replace(
            caps,
            metadata=Capability(
                CapabilityState.AVAILABLE, "owner-scoped provider metadata"),
        )
    view = _selected_record_view(record, mode_1d, mode_2d)
    wavelength = wavelength_evidence(view=view, provider=provider, row=row)
    return FrameProjection(
        label=label, present=True, metadata=row, capabilities=caps,
        wavelength=wavelength,
        normalization_channels=normalization_channels(row))


def _selected_record_view(
    record: FrameRecord,
    mode_1d: str | None,
    mode_2d: str | None,
) -> FrameView | None:
    candidates = []
    if mode_1d is not None:
        candidates.append(record.view_1d(mode_1d))
    if mode_2d is not None:
        candidates.append(record.view_2d(mode_2d))
    candidates.extend((record.view_1d(), record.view_2d()))
    return next((view for view in candidates if view is not None), None)


def _merge_metadata_rows(primary: MetadataRow, supplemental: MetadataRow) -> MetadataRow:
    """Merge a stored row with owner-scoped provider facts, failing closed.

    The stored row is authoritative for already-ingested values; the provider
    may fill missing counters/motors but may not silently replace a conflicting
    value for the same frame identity.
    """
    merged = dict(primary.raw)
    for key, value in supplemental.raw.items():
        if key in merged and _values_conflict(merged[key], value):
            raise MetadataConflictError(
                f"stored/provider metadata disagree for {key!r}: "
                f"{merged[key]!r} != {value!r}"
            )
        merged.setdefault(key, value)
    return MetadataRow(merged)
