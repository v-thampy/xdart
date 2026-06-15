"""Qt-free frame publication contract for xdart displays.

``FramePublication`` is the GUI-side envelope around
``xrd_tools.core.FrameView``.  It is deliberately separate from
``LiveFrame``: live frames carry locks, caches, lazy loaders, and mutable
runtime state, while publications are snapshots the display can validate and
store without reaching back through widget state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import RLock
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from xrd_tools.core import (
    DEFAULT_MODE_KEY,
    FrameRecord,
    FrameView,
    TwoDKind,
    numeric_metadata,
)


# Legacy GUI dict keys (``frame.gi_1d`` / ``frame.gi_2d``, written by
# ``ewald.frame``) -> canonical on-disk mode_keys (== GI*Mode.value, the same
# vocabulary the io layer + FrameEvent use).  DIMENSION-SCOPED: ``polar`` means
# q_total in a 1D context but q_chi in a 2D context, so the two maps must stay
# separate — never flatten them.  These legacy spellings are GUI-side and never
# reach disk; the canonical keys do.  ``exit2d`` is handled here because the ssrl
# GI2D mode coercer has no alias for that literal 2D key.
_LEGACY_TO_CANONICAL_1D = {
    "qtotal": "q_total", "qip": "q_ip", "qoop": "q_oop", "exit": "exit_angle",
}
_LEGACY_TO_CANONICAL_2D = {
    "gi2d": "qip_qoop", "polar": "q_chi", "exit2d": "exit_angles",
}


def legacy_to_canonical_1d(key: str) -> str:
    """Map a ``frame.gi_1d`` dict key to its canonical 1D mode_key (passthrough
    if already canonical)."""
    return _LEGACY_TO_CANONICAL_1D.get(key, key)


def legacy_to_canonical_2d(key: str) -> str:
    """Map a ``frame.gi_2d`` dict key to its canonical 2D mode_key (passthrough
    if already canonical)."""
    return _LEGACY_TO_CANONICAL_2D.get(key, key)


def _resolve_active_mode(passed, modes, active_result):
    """The canonical active mode_key for a per-dimension mode map.

    Identity is authoritative: the publication's ``.view`` is built from
    ``int_1d``/``int_2d``, so the record's active mode MUST be the one whose
    result IS that object (the live integrator assigns the same result to
    ``gi_*[key]`` and ``int_*``) — otherwise ``record.active_view()`` would
    diverge from ``.view``.  The explicit ``passed`` key is only a fallback hint
    for when there is no active result to match (then the first computed mode,
    else ``DEFAULT_MODE_KEY``)."""
    if active_result is not None:
        for mode, result in modes.items():
            if result is active_result:
                return mode
    if passed is not None and passed in modes:
        return passed
    return next(iter(modes), DEFAULT_MODE_KEY)


def _readonly_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not value:
        return MappingProxyType({})
    return MappingProxyType(dict(value))


def _finite_fraction(arr: np.ndarray | None) -> float | None:
    if arr is None:
        return None
    data = np.asarray(arr, dtype=float)
    if data.size == 0:
        return 0.0
    return float(np.isfinite(data).sum() / data.size)


def _dummy_fraction(arr: np.ndarray | None, *, dummy_value: float = -1.0) -> float | None:
    if arr is None:
        return None
    data = np.asarray(arr, dtype=float)
    if data.size == 0:
        return 0.0
    return float(np.isclose(data, dummy_value, equal_nan=False).sum() / data.size)


def _axis_range(axis) -> tuple[float, float] | None:
    values = getattr(axis, "values", None)
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.nanmin(finite)), float(np.nanmax(finite))


@dataclass(frozen=True, slots=True)
class PublicationDiagnostics:
    """Health checks computed before a frame reaches display or disk."""

    finite_fraction_1d: float | None = None
    finite_fraction_2d: float | None = None
    dummy_fraction_2d: float | None = None
    axis_ranges: Mapping[str, tuple[float, float] | None] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    errors_1d: tuple[str, ...] = ()
    errors_2d: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_ranges", _readonly_mapping(self.axis_ranges))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors_1d", tuple(self.errors_1d))
        object.__setattr__(self, "errors_2d", tuple(self.errors_2d))
        errors = tuple(self.errors) or self.errors_1d + self.errors_2d
        object.__setattr__(self, "errors", tuple(errors))

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class FramePublication:
    """Display publication snapshot for one frame.

    ``view`` is the ACTIVE-mode projection — the display surface every consumer
    reads (always supplied by the builders, so the 30+ ``publication.view.*``
    consumers and the ``slots=True`` layout are untouched).  ``record`` is the
    multi-result backing (every GI mode computed for this frame, ADR-0003), a
    verdict-free :class:`FrameRecord`; when omitted it defaults to the
    single-mode ``FrameRecord.from_view(view)``.
    """

    view: FrameView
    record: FrameRecord | None = None
    source_identity: str = ""
    generation: int = 0
    raw_ref: Any | None = None
    raw_status: str = "unknown"
    metadata_raw: Mapping[str, Any] = field(default_factory=dict)
    metadata_numeric: Mapping[str, float] = field(default_factory=dict)
    diagnostics: PublicationDiagnostics = field(default_factory=PublicationDiagnostics)

    def __post_init__(self) -> None:
        if self.record is None:
            object.__setattr__(self, "record", FrameRecord.from_view(self.view))
        raw = self.metadata_raw or self.view.metadata_raw
        numeric = self.metadata_numeric or self.view.metadata_numeric or numeric_metadata(raw)
        object.__setattr__(self, "metadata_raw", _readonly_mapping(raw))
        object.__setattr__(self, "metadata_numeric", _readonly_mapping(numeric))

    @property
    def label(self) -> int | str:
        return self.view.label


def validate_publication(
    publication: FramePublication,
    *,
    allow_dummy_2d: bool = False,
    raise_on_error: bool = False,
) -> PublicationDiagnostics:
    """Validate display-critical frame publication invariants.

    This is an early GUI/display gate.  It does not replace or relax the
    strict ssrl NeXus writer validators.
    """

    view = publication.view
    warnings: list[str] = []
    errors_1d: list[str] = []
    errors_2d: list[str] = []
    finite_1d = _finite_fraction(view.intensity_1d)
    finite_2d = _finite_fraction(view.intensity_2d)
    dummy_2d = _dummy_fraction(view.intensity_2d)
    axis_ranges = {
        "axis_1d": _axis_range(view.axis_1d),
        "axis_2d_x": _axis_range(view.axis_2d_x),
        "axis_2d_y": _axis_range(view.axis_2d_y),
    }

    if view.has_1d and (finite_1d is None or finite_1d == 0.0):
        errors_1d.append("1D intensity contains no finite values")
    if view.has_2d:
        if finite_2d is None or finite_2d == 0.0:
            errors_2d.append("2D intensity contains no finite values")
        if dummy_2d is not None and dummy_2d >= 0.95 and not allow_dummy_2d:
            errors_2d.append("2D intensity is almost entirely dummy pixels")
        if view.two_d_kind is not TwoDKind.Q_CHI and view.incident_angle is None:
            warnings.append("GI 2D publication has no resolved incident angle")
        for name in ("axis_2d_x", "axis_2d_y"):
            if axis_ranges[name] is None:
                errors_2d.append(f"{name} has no finite range")

    diagnostics = PublicationDiagnostics(
        finite_fraction_1d=finite_1d,
        finite_fraction_2d=finite_2d,
        dummy_fraction_2d=dummy_2d,
        axis_ranges=axis_ranges,
        warnings=tuple(warnings),
        errors_1d=tuple(errors_1d),
        errors_2d=tuple(errors_2d),
    )
    if raise_on_error and diagnostics.errors:
        raise ValueError("; ".join(diagnostics.errors))
    return diagnostics


def publication_has_1d_errors(publication: FramePublication) -> bool:
    return bool(publication.diagnostics.errors_1d)


def publication_has_2d_errors(publication: FramePublication) -> bool:
    return bool(publication.diagnostics.errors_2d)


def publication_error_details(publication: FramePublication, output: str) -> str:
    if output == "1d":
        errors = publication.diagnostics.errors_1d
    elif output == "2d":
        errors = publication.diagnostics.errors_2d
    else:
        errors = publication.diagnostics.errors
    return "; ".join(errors)


def _record_from_live_frame(frame, view, metadata_raw, incident_angle,
                            active_mode_1d, active_mode_2d) -> FrameRecord:
    """Build the multi-result :class:`FrameRecord` from a live frame's per-mode
    GI dicts.  Non-GI frames (empty ``gi_1d``/``gi_2d``) collapse to a single
    ``DEFAULT_MODE_KEY`` record equivalent to ``view``."""
    gi_1d = getattr(frame, "gi_1d", None) or {}
    gi_2d = getattr(frame, "gi_2d", None) or {}
    if not gi_1d and not gi_2d:
        return FrameRecord.from_view(view)

    label = view.label
    common = dict(
        metadata_raw=metadata_raw,
        metadata_numeric=numeric_metadata(metadata_raw),
        incident_angle=incident_angle,
        source_path=getattr(frame, "source_file", None) or None,
        source_frame_index=getattr(frame, "source_frame_idx", None),
    )
    thumbnail = getattr(frame, "thumbnail", None)
    modes_1d = {legacy_to_canonical_1d(k): r for k, r in gi_1d.items()}
    modes_2d = {legacy_to_canonical_2d(k): r for k, r in gi_2d.items()}
    results_1d = {
        m: FrameView.from_results(label=label, result_1d=r, **common)
        for m, r in modes_1d.items()
    }
    results_2d = {
        m: FrameView.from_results(
            label=label, result_2d=r,
            thumbnail=thumbnail, mask_baked=thumbnail is not None, **common,
        )
        for m, r in modes_2d.items()
    }
    am1 = _resolve_active_mode(active_mode_1d, modes_1d, getattr(frame, "int_1d", None))
    am2 = _resolve_active_mode(active_mode_2d, modes_2d, getattr(frame, "int_2d", None))
    return FrameRecord(
        label=label,
        results_1d=results_1d,
        results_2d=results_2d,
        active_mode_1d=am1 if results_1d else DEFAULT_MODE_KEY,
        active_mode_2d=am2 if results_2d else DEFAULT_MODE_KEY,
    )


def publication_from_live_frame(
    frame: Any,
    *,
    generation: int = 0,
    source_identity: str | None = None,
    include_raw: bool = False,
    validate: bool = True,
    active_mode_1d: str | None = None,
    active_mode_2d: str | None = None,
) -> FramePublication:
    """Adapt a current xdart ``LiveFrame``-like object into a publication.

    Carries every computed GI mode in ``publication.record`` (ADR-0003); the
    active mode is the explicit ``active_mode_*`` if given, else inferred from
    which ``gi_*`` entry IS ``int_*`` (identity).  ``view`` is unchanged."""

    metadata_raw = dict(getattr(frame, "scan_info", None) or {})
    result_2d = getattr(frame, "int_2d", None)
    incident_angle = None
    if getattr(frame, "gi", False):
        try:
            incident_angle = float(frame._get_incident_angle())
        except Exception:
            incident_angle = None

    view = FrameView.from_results(
        label=getattr(frame, "idx", ""),
        result_1d=getattr(frame, "int_1d", None),
        result_2d=result_2d,
        raw=(getattr(frame, "map_raw", None) if include_raw else None),
        thumbnail=getattr(frame, "thumbnail", None),
        mask_baked=getattr(frame, "thumbnail", None) is not None,
        metadata_raw=metadata_raw,
        metadata_numeric=numeric_metadata(metadata_raw),
        incident_angle=incident_angle,
        source_path=getattr(frame, "source_file", None) or None,
        source_frame_index=getattr(frame, "source_frame_idx", None),
    )
    record = _record_from_live_frame(
        frame, view, metadata_raw, incident_angle, active_mode_1d, active_mode_2d,
    )
    publication = FramePublication(
        view=view,
        record=record,
        source_identity=(
            source_identity
            if source_identity is not None
            else str(getattr(frame, "source_file", "") or getattr(frame, "idx", ""))
        ),
        generation=generation,
        raw_ref=frame,
        raw_status=("ready" if getattr(frame, "map_raw", None) is not None else "missing"),
        metadata_raw=metadata_raw,
        metadata_numeric=numeric_metadata(metadata_raw),
    )
    if validate:
        diagnostics = validate_publication(publication)
        publication = replace(publication, diagnostics=diagnostics)
    return publication


def publication_from_frame_view(
    view: FrameView,
    *,
    record: FrameRecord | None = None,
    generation: int = 0,
    source_identity: str = "",
    raw_ref: Any | None = None,
    raw_status: str = "unknown",
    validate: bool = True,
) -> FramePublication:
    """Wrap a headless :class:`FrameView` in the xdart publication envelope.

    ``record`` carries every persisted GI mode (from the mode-aware reader);
    when omitted it is the single-mode ``FrameRecord.from_view(view)``."""

    publication = FramePublication(
        view=view,
        record=record if record is not None else FrameRecord.from_view(view),
        source_identity=source_identity or str(view.source_path or view.label),
        generation=generation,
        raw_ref=raw_ref,
        raw_status=raw_status,
        metadata_raw=view.metadata_raw,
        metadata_numeric=view.metadata_numeric,
    )
    if validate:
        publication = replace(
            publication,
            diagnostics=validate_publication(publication),
        )
    return publication


def publication_from_nexus_frame(
    scan_file: str,
    frame: int,
    *,
    generation: int = 0,
    entry: str = "entry",
    include_thumbnail: bool = True,
    validate: bool = True,
) -> FramePublication:
    """Read a saved processed frame and publish it through the same contract."""

    from xrd_tools.io import read_frame_record

    record = read_frame_record(
        scan_file,
        frame,
        entry=entry,
        include_thumbnail=include_thumbnail,
    )
    view = record.active_view()
    return publication_from_frame_view(
        view,
        record=record,
        generation=generation,
        source_identity=str(scan_file),
        raw_status=("thumbnail" if view.thumbnail is not None else "missing"),
        validate=validate,
    )


def _view_has_heavy_arrays(view: FrameView) -> bool:
    return any(
        value is not None
        for value in (
            view.intensity_1d, view.sigma_1d,
            view.intensity_2d, view.sigma_2d,
            view.raw, view.thumbnail,
        )
    )


def _publication_has_heavy_payload(publication: FramePublication) -> bool:
    if publication.raw_ref is not None:
        return True
    if _view_has_heavy_arrays(publication.view):
        return True
    # Record-aware: a multi-result record holds the NON-active modes' arrays,
    # invisible to the active .view — they must count toward the heavy bound so
    # eviction actually frees them (and is triggered) rather than letting a
    # record-backed publication defeat max_heavy_items.
    record = publication.record
    if record is not None:
        for mode_view in (*record.results_1d.values(), *record.results_2d.values()):
            if _view_has_heavy_arrays(mode_view):
                return True
    return False


def _semilight_publication(publication: FramePublication) -> FramePublication:
    """Tier-1 eviction (D2): drop the heavy arrays but KEEP the thumbnail.

    A ~256 KB thumbnail per frame keeps scroll-back instantly paintable
    (the Image Viewer falls back to ``view.thumbnail``) while the full
    payload rehydrates in the background; thumbnails have their own,
    much larger bound (tier 2)."""
    view = publication.view
    thumb_view = FrameView(
        label=view.label,
        two_d_kind=view.two_d_kind,
        thumbnail=view.thumbnail,
        mask_baked=view.mask_baked,
        metadata_raw=view.metadata_raw,
        metadata_numeric=view.metadata_numeric,
        incident_angle=view.incident_angle,
        geometry=view.geometry,
        source_path=view.source_path,
        source_frame_index=view.source_frame_index,
        extra=view.extra,
    )
    return replace(
        publication,
        view=thumb_view,
        # Thin the record too, else the non-active modes' arrays survive
        # eviction (view-record drift + memory leak).  The active thumbnail
        # slot is all an evicted publication retains; full per-mode data
        # rehydrates from disk via read_frame_record.
        record=FrameRecord.from_view(thumb_view),
        raw_ref=None,
        raw_status="thumbnail" if view.thumbnail is not None else "evicted",
        metadata_raw=view.metadata_raw,
        metadata_numeric=view.metadata_numeric,
    )


def _lightweight_publication(publication: FramePublication) -> FramePublication:
    """Tier-2 eviction: metadata/diagnostics-only (no arrays at all)."""
    view = publication.view
    light_view = FrameView(
        label=view.label,
        two_d_kind=view.two_d_kind,
        mask_baked=False,
        metadata_raw=view.metadata_raw,
        metadata_numeric=view.metadata_numeric,
        incident_angle=view.incident_angle,
        geometry=view.geometry,
        source_path=view.source_path,
        source_frame_index=view.source_frame_index,
        extra=view.extra,
    )
    return replace(
        publication,
        view=light_view,
        record=FrameRecord.from_view(light_view),  # no arrays linger in the record
        raw_ref=None,
        raw_status="evicted",
        metadata_raw=view.metadata_raw,
        metadata_numeric=view.metadata_numeric,
    )


class PublicationStore:
    """Small generation-aware store for frame publications.

    ``max_heavy_items`` bounds display-heavy arrays while keeping the frame's
    label, metadata, source identity, and diagnostics in the store.  Full
    source/NeXus rehydration is intentionally deferred; this protects long live
    scans from unbounded memory growth without changing the publication API.
    """

    def __init__(
        self,
        *,
        max_items: int | None = None,
        max_heavy_items: int | None = 64,
        max_thumbnail_items: int | None = 512,
    ) -> None:
        if max_items is not None and max_items < 1:
            raise ValueError("max_items must be positive or None")
        if max_heavy_items is not None and max_heavy_items < 0:
            raise ValueError("max_heavy_items must be non-negative or None")
        if max_thumbnail_items is not None and max_thumbnail_items < 0:
            raise ValueError("max_thumbnail_items must be non-negative or None")
        self._lock = RLock()
        self._generation = 0
        self._max_items = max_items
        self._max_heavy_items = max_heavy_items
        self._max_thumbnail_items = max_thumbnail_items
        self._items: dict[int | str, FramePublication] = {}
        self._heavy_labels: list[int | str] = []
        self._thumb_labels: list[int | str] = []
        # D2: optional rehydration source (label -> FramePublication|None).
        # A SYNCHRONOUS loader — register a cheap one, or call
        # get_or_hydrate from a background worker (never blocking h5py
        # reads on the GUI thread; the thumbnail tier keeps scroll-back
        # paintable meanwhile).
        self._hydrator = None

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def clear(self) -> None:
        with self._lock:
            self._generation += 1
            self._items.clear()
            self._heavy_labels.clear()
            self._thumb_labels.clear()

    def set_hydrator(self, hydrator) -> None:
        """Register the rehydration source for :meth:`get_or_hydrate`."""
        with self._lock:
            self._hydrator = hydrator

    def get_or_hydrate(self, label: int | str) -> FramePublication | None:
        """Return the publication, rehydrating an evicted payload via the
        registered hydrator (synchronous — call from a background worker
        for disk-backed hydrators)."""
        with self._lock:
            publication = self._items.get(label)
            hydrator = self._hydrator
        if publication is not None and (
                _publication_has_heavy_payload(publication)
                or publication.raw_status not in ("evicted", "thumbnail")):
            return publication
        if hydrator is None:
            return publication
        try:
            fresh = hydrator(label)
        except Exception:
            return publication
        if fresh is None:
            return publication
        return self.upsert(fresh)

    def upsert(self, publication: FramePublication) -> FramePublication:
        with self._lock:
            if publication.generation != self._generation:
                publication = replace(publication, generation=self._generation)
            label = publication.label
            if label in self._items:
                self._items.pop(label)
                self._drop_heavy_label_locked(label)
                self._drop_thumb_label_locked(label)
            self._items[publication.label] = publication
            if _publication_has_heavy_payload(publication):
                self._heavy_labels.append(label)
            if publication.view.thumbnail is not None:
                self._thumb_labels.append(label)
            self._enforce_bounds_locked()
            return publication

    def extend(self, publications: Iterable[FramePublication]) -> tuple[FramePublication, ...]:
        with self._lock:
            return tuple(self.upsert(publication) for publication in publications)

    def get(self, label: int | str) -> FramePublication | None:
        with self._lock:
            return self._items.get(label)

    def labels(self) -> tuple[int | str, ...]:
        with self._lock:
            return tuple(self._items)

    def snapshot(self) -> Mapping[int | str, FramePublication]:
        with self._lock:
            return MappingProxyType(dict(self._items))

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def _drop_heavy_label_locked(self, label: int | str) -> None:
        try:
            self._heavy_labels.remove(label)
        except ValueError:
            pass

    def _drop_thumb_label_locked(self, label: int | str) -> None:
        try:
            self._thumb_labels.remove(label)
        except ValueError:
            pass

    def _enforce_bounds_locked(self) -> None:
        if self._max_items is not None:
            while len(self._items) > self._max_items:
                label = next(iter(self._items))
                self._items.pop(label, None)
                self._drop_heavy_label_locked(label)
                self._drop_thumb_label_locked(label)

        # tier 1 (D2): over the heavy bound -> drop arrays, KEEP thumbnail
        if self._max_heavy_items is not None:
            while len(self._heavy_labels) > self._max_heavy_items:
                label = self._heavy_labels.pop(0)
                publication = self._items.get(label)
                if publication is None:
                    continue
                self._items[label] = _semilight_publication(publication)

        # tier 2: thumbnails have their own, larger bound
        if self._max_thumbnail_items is not None:
            while len(self._thumb_labels) > self._max_thumbnail_items:
                label = self._thumb_labels.pop(0)
                publication = self._items.get(label)
                if publication is None:
                    continue
                self._items[label] = _lightweight_publication(publication)
                self._drop_heavy_label_locked(label)


__all__ = [
    "FramePublication",
    "PublicationDiagnostics",
    "PublicationStore",
    "legacy_to_canonical_1d",
    "legacy_to_canonical_2d",
    "publication_from_frame_view",
    "publication_from_live_frame",
    "publication_from_nexus_frame",
    "publication_error_details",
    "publication_has_1d_errors",
    "publication_has_2d_errors",
    "validate_publication",
]
