"""Qt-free frame publication contract for xdart displays.

``FramePublication`` is the GUI-side envelope around
``xrd_tools.core.FrameView``.  It is deliberately separate from
``LiveFrame``: live frames carry locks, caches, lazy loaders, and mutable
runtime state, while publications are snapshots the display can validate and
store without reaching back through widget state.
"""

from __future__ import annotations

import logging
import os
import posixpath
from dataclasses import dataclass, field, fields, is_dataclass, replace
from threading import RLock
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from xrd_tools.core import (
    Axis,
    FrameRecord,
    FrameView,
    TwoDKind,
    numeric_metadata,
)
from xrd_tools.io.nexus_record import (
    frame_record_from_live_frame as _shared_frame_record_from_live_frame,
    legacy_to_canonical_1d,
    legacy_to_canonical_2d,
)

logger = logging.getLogger(__name__)

DEFAULT_PUBLICATION_MAX_ITEMS = 512


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
    #: X1 Slice 3c (S3-OR1): the IMMUTABLE owning-scan identity — display
    #: publication provenance stamped by the production publish sites from the
    #: already-authoritative current/run/browse scan key (never inferred from a
    #: source filename, never via I/O).  ``None`` = legacy/viewer-only or
    #: genuinely ownerless; the scan-qualified projection then falls back to
    #: the positive source-identity proof only.  Preserved through replace,
    #: thinning, hydration, merge, and store carryover.
    scan_key: str | None = None

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


@dataclass(frozen=True, slots=True)
class Light1DPublicationShell:
    """Array-free scalar receipt for one privately guarded lease row."""

    label: int | str
    generation: int
    source_identity: str


@dataclass(frozen=True, slots=True)
class _Light1DModeTemplate:
    mode: Any
    axis_label: str
    axis_unit: str
    axis_log: bool


@dataclass(frozen=True, slots=True)
class _Light1DPair:
    shell: Light1DPublicationShell
    store_generation: int
    scan_key: str | None
    active_mode: Any
    modes: tuple[_Light1DModeTemplate, ...]
    guard: Any


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


def publication_from_live_frame(
    frame: Any,
    *,
    generation: int = 0,
    source_identity: str | None = None,
    include_raw: bool = False,
    include_2d: bool = True,
    include_thumbnail: bool = True,
    retain_raw_ref: bool = True,
    validate: bool = True,
    active_mode_1d: str | None = None,
    active_mode_2d: str | None = None,
    scan_key: str | None = None,
) -> FramePublication:
    """Adapt a current xdart ``LiveFrame``-like object into a publication.

    Carries every computed GI mode in ``publication.record`` (ADR-0003); the
    active mode is the explicit ``active_mode_*`` if given, else inferred from
    which ``gi_*`` entry IS ``int_*`` (identity).  ``view`` is unchanged."""

    metadata_raw = dict(getattr(frame, "scan_info", None) or {})
    result_2d = getattr(frame, "int_2d", None) if include_2d else None
    thumbnail = getattr(frame, "thumbnail", None) if include_thumbnail else None
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
        thumbnail=thumbnail,
        mask_baked=thumbnail is not None,
        metadata_raw=metadata_raw,
        metadata_numeric=numeric_metadata(metadata_raw),
        incident_angle=incident_angle,
        source_path=getattr(frame, "source_file", None) or None,
        source_frame_index=getattr(frame, "source_frame_idx", None),
    )
    record = _shared_frame_record_from_live_frame(
        frame,
        active_mode_1d=active_mode_1d,
        active_mode_2d=active_mode_2d,
        include_raw=include_raw,
        include_2d=include_2d,
        include_thumbnail=include_thumbnail,
    )
    raw_ref = frame if retain_raw_ref else None
    if retain_raw_ref:
        raw_status = (
            "ready" if getattr(frame, "map_raw", None) is not None else "missing"
        )
    elif not include_2d and getattr(frame, "int_1d", None) is not None:
        raw_status = "1d-only"
    else:
        raw_status = "missing"
    publication = FramePublication(
        view=view,
        record=record,
        source_identity=(
            source_identity
            if source_identity is not None
            else str(getattr(frame, "source_file", "") or getattr(frame, "idx", ""))
        ),
        generation=generation,
        raw_ref=raw_ref,
        raw_status=raw_status,
        metadata_raw=metadata_raw,
        metadata_numeric=numeric_metadata(metadata_raw),
        scan_key=scan_key,
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
    scan_key: str | None = None,
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
        scan_key=scan_key,
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
    scan_key: str | None = None,
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
        scan_key=scan_key,
    )


def _view_has_heavy_arrays(view: FrameView) -> bool:
    # 1D rows are tiny compared with raw detector images and 2D cakes.  The
    # overlay/waterfall fast path hydrates many of them in one batch, so they
    # must not be governed by the 64-item raw/cake heavy window.
    return any(
        value is not None
        for value in (
            view.intensity_2d, view.sigma_2d,
            view.raw, view.thumbnail,
        )
    )


def _raw_ref_has_heavy_arrays(raw_ref: Any | None) -> bool:
    if raw_ref is None:
        return False
    for name in ("int_2d", "map_raw", "thumbnail"):
        if getattr(raw_ref, name, None) is not None:
            return True
    bg_raw = getattr(raw_ref, "bg_raw", None)
    if bg_raw is not None and np.asarray(bg_raw).ndim > 0:
        return True
    return False


def _publication_has_heavy_payload(publication: FramePublication) -> bool:
    if _raw_ref_has_heavy_arrays(publication.raw_ref):
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


def _view_has_data_arrays(view: FrameView) -> bool:
    """True if the view carries real DATA arrays (1D/2D intensity or raw) — the
    thumbnail does NOT count.  Distinct from _view_has_heavy_arrays (which counts
    the thumbnail, for the eviction bound): a tier-1 (semilight) view keeps only
    the thumbnail, so it has no data and should rehydrate."""
    return any(
        value is not None
        for value in (view.intensity_1d, view.intensity_2d, view.raw)
    )


def publication_raw_recoverable(publication: FramePublication) -> bool:
    """True when the FULL raw is absent but can be made view-resident.

    PF-1e: an old Append row may carry only a valid ``source`` reference — no
    stored thumbnail (``raw_status`` ``"missing"``/``"unknown"``).  A live
    publication can also borrow pixels from its mutable ``LiveFrame`` through
    ``raw_ref.map_raw`` while its immutable view still has ``raw=None``.  That
    borrowed array can be released by the live-memory bound between eligibility
    and render, so it must be promoted into the view just like a source-only
    row.  If it was already released, the registered hydrator falls back to the
    same lazy source route used by the headless readers.  Thumbnail backfill
    stays optional.
    """
    view = publication.view
    raw_ref = publication.raw_ref
    raw_ref_value = (
        getattr(raw_ref, "map_raw", None) if raw_ref is not None else None
    )
    raw_ref_source = (
        getattr(raw_ref, "source_file", None) if raw_ref is not None else None
    )
    source_path = (
        getattr(view, "source_path", None)
        or raw_ref_source
    )
    return (
        view.raw is None
        and (raw_ref_value is not None or bool(source_path))
    )


def _publication_has_full_payload(publication: FramePublication) -> bool:
    """True if the publication has its data payload (so there is nothing to
    rehydrate): a live raw_ref, real data arrays in the active view, or any
    per-mode record view with data arrays.  Used by get_or_hydrate INSTEAD of
    _publication_has_heavy_payload — the latter counts the thumbnail as heavy, so
    a tier-1 (thumbnail-only) publication wrongly looked 'already loaded' and
    never rehydrated."""
    if publication.raw_status == "1d-only":
        return False
    if publication.raw_ref is not None:
        return True
    if _view_has_data_arrays(publication.view):
        return True
    record = publication.record
    if record is not None:
        for mode_view in (*record.results_1d.values(), *record.results_2d.values()):
            if _view_has_data_arrays(mode_view):
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


def _publication_has_1d_arrays(publication: FramePublication) -> bool:
    def view_has_1d_array(view: FrameView) -> bool:
        axis = view.axis_1d
        return any(value is not None for value in (
            None if axis is None else axis.values,
            view.intensity_1d,
            view.sigma_1d,
        ))

    return view_has_1d_array(publication.view) or any(
        view_has_1d_array(view)
        for view in publication.record.results_1d.values()
    )


def _without_1d_arrays(publication: FramePublication) -> FramePublication:
    view = replace(
        publication.view,
        axis_1d=None,
        intensity_1d=None,
        sigma_1d=None,
    )
    return replace(
        publication,
        view=view,
        record=FrameRecord(
            label=publication.label,
            results_1d={
                mode: replace(
                    mode_view,
                    axis_1d=None,
                    intensity_1d=None,
                    sigma_1d=None,
                )
                for mode, mode_view in publication.record.results_1d.items()
            },
            results_2d=publication.record.results_2d,
            active_mode_1d=publication.record.active_mode_1d,
            active_mode_2d=publication.record.active_mode_2d,
        ),
        raw_ref=None,
    )


def _merge_records(existing: FrameRecord, incoming: FrameRecord) -> FrameRecord:
    """Accumulate the incoming record's modes into the existing one (ADR-0003
    fork B): the store keeps ONE FrameRecord per frame that grows as GI modes
    are recomputed.

    Each incoming 1D/2D mode is folded in via ``with_result_1d/2d`` (immutable
    upsert-by-key — an incoming mode already present is overwritten with the
    fresher view).  The INCOMING active modes win (the merged ``active_mode_1d/2d``
    equal the incoming ones), and the caller keeps the publication's ``.view``
    unchanged, so display consumers (which read ``.view`` today) are unaffected.

    NOTE the merged record may be a SUPERSET of ``.view``: for a same-dimension
    re-publish ``record.active_view()`` equals ``.view``, but a cross-dimension
    accumulation (e.g. a 1D-only publish then a 2D-only publish for one frame)
    leaves ``.view`` as the latest single projection while ``record.active_view()``
    carries BOTH dimensions — that richer union is the whole point (the record is
    what a future multi-mode consumer reads; ``.view`` is the current display
    surface).  An incoming record with empty result maps folds as a no-op.
    """
    acc = existing
    for mode, view in incoming.results_1d.items():
        acc = acc.with_result_1d(
            mode, view, make_active=(mode == incoming.active_mode_1d))
    for mode, view in incoming.results_2d.items():
        acc = acc.with_result_2d(
            mode, view, make_active=(mode == incoming.active_mode_2d))
    return acc


def _same_source_id(sa, sb) -> bool:
    """True unless the two source identities are genuinely DIFFERENT files.

    The SAME physical frame gets different spellings on different code paths — a
    live frame carries an absolute ``source_file`` while a disk-reloaded frame
    carries the relative ``source/path`` — so an exact string compare would
    wrongly skip a legitimate same-frame accumulation.  Compared by PATH-COMPONENT
    SUFFIX: equal after ``normpath``, or the shorter path's components are a tail
    of the longer's (abs vs rel of the same file).  Unlike a bare-basename
    compare this still REJECTS two different directories that share a filename
    (e.g. ``run1/frame_0001.tif`` vs ``run2/frame_0001.tif``).  Missing source
    IDs merge only when BOTH sides are missing; one missing + one known source is
    not enough evidence to splice records.

    KNOWN LIMITATION (transitional compat — review #3, accepted for v1.0): if one
    side is a SINGLE-component id (a bare filename, no directory), the suffix
    match degrades to basename matching for that pair — a bare ``frame_0001.tif``
    would match ANY path ending in it.  This is tolerated because source IDs are
    not yet reliably canonical (a live frame carries an absolute ``source_file``,
    a reload the relative ``source/path``), and a real bare-filename relative path
    (``source_base`` == the file's own directory) must still merge with its
    absolute spelling.  Tighten to exact canonical identity once IDs are
    canonicalised at the source seam (then abs/rel normalise to one form and this
    whole suffix dance — and the hole — goes away)."""
    if not sa or not sb:
        return not sa and not sb
    na, nb = os.path.normpath(sa), os.path.normpath(sb)
    if na == nb:
        return True
    pa, pb = na.split(os.sep), nb.split(os.sep)
    short, long_ = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
    return long_[-len(short):] == short


def _same_source(a: FramePublication, b: FramePublication) -> bool:
    return _same_source_id(a.source_identity, b.source_identity)


def _scan_owners_compatible(a, b) -> bool:
    """Whether two publications may contribute to one accumulated record.

    An unstamped legacy publication can still merge through the source guard.
    When both publications carry explicit owners, those owners must identify
    the same scan; a matching source file alone is not sufficient evidence.
    """
    if a in (None, "") or b in (None, ""):
        return True

    def normalized(value) -> str:
        source = str(value).strip().rsplit("#", 1)[0].replace("\\", "/")
        return posixpath.normpath(source).casefold()

    return normalized(a) == normalized(b)


# MEM-2: sentinel so an unspecified heavy cap resolves to the RAM-aware window
# at construction (frame shape is unknown here → the coarse RAM tier; the
# wrangler resizes staging + record to the frame-precise window at run start).
_AUTO_HEAVY_WINDOW = object()


class PublicationStore:
    """Small generation-aware store for frame publications.

    ``max_heavy_items`` bounds display-heavy arrays while keeping the frame's
    label, metadata, source identity, and diagnostics in the store.  Full
    source/NeXus rehydration is intentionally deferred; this protects long live
    scans from unbounded memory growth without changing the publication API.
    Left unspecified, it defaults to the RAM-aware :func:`heavy_window` (MEM-2).
    """

    def __init__(
        self,
        *,
        max_items: int | None = DEFAULT_PUBLICATION_MAX_ITEMS,
        max_heavy_items=_AUTO_HEAVY_WINDOW,
        max_thumbnail_items: int | None = 512,
    ) -> None:
        if max_heavy_items is _AUTO_HEAVY_WINDOW:
            from xrd_tools.core import heavy_window
            max_heavy_items = heavy_window()
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
        self.allocation: Any = None
        self._light_1d = None
        self._items: dict[int | str, FramePublication] = {}
        self._light_1d_items: dict[int | str, _Light1DPair] = {}
        self._heavy_labels: list[int | str] = []
        self._thumb_labels: list[int | str] = []
        # D2: optional rehydration source (label -> FramePublication|None).
        # A SYNCHRONOUS loader — register a cheap one, or call
        # get_or_hydrate from a background worker (never blocking h5py
        # reads on the GUI thread; the thumbnail tier keeps scroll-back
        # paintable meanwhile).
        self._hydrator = None
        self._hydrator_1d_many = None
        # MEM1-15: optional evictability probe (label -> bool).  When set,
        # tier-0 eviction honors the persist-before-evict invariant the other
        # stores already keep: an unpersisted ("owed") publication PINS memory
        # instead of being dropped — dropping it would blank the frame on
        # scroll-back because this store is the cake's only render source.
        # The live widget wires this to FrameRecordStore.is_persisted; the
        # probe must be cheap, lock-safe to call under this store's lock, and
        # must never call back into this store.
        self._evictable = None
        # Step 6: prior-pass (record, source_identity) carried across a same-scan
        # reintegrate so a re-upsert MERGES the recomputed mode into the frame's
        # accumulated record (begin_reintegrate populates it; upsert consumes per
        # label).  Only the record + source are carried — NOT the full
        # publication — so the heavy raw_ref (the frame holding map_raw) is not
        # pinned for the duration of the pass.
        self._carryover: dict[int | str, tuple] = {}

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def bind_allocation(self, allocation) -> None:
        """Bind the exact H10 allocation before any publication exists."""
        with self._lock:
            if self.allocation is not None:
                if self.allocation is not allocation:
                    raise ValueError("PublicationStore is bound to another allocation")
                return
            if self._items:
                raise ValueError("cannot bind an allocation to a populated store")
            self.allocation = allocation
            self._max_items = int(allocation.publication_items)
            self._max_heavy_items = int(allocation.publication_heavy_items)
            self._max_thumbnail_items = int(allocation.thumbnail_items)

    def bind_light_1d(self, lease) -> None:
        """Bind the allocation's sole byte-native light-1D lease."""
        from xrd_tools.session import Light1DRetentionLease

        if type(lease) is not Light1DRetentionLease:
            raise TypeError("publication light-1D owner must be an exact lease")
        with self._lock:
            if self.allocation is None:
                raise RuntimeError("light-1D publication requires allocation binding")
            if lease.authority.parent_allocation is not self.allocation:
                raise ValueError("light-1D lease belongs to another allocation")
            if self._light_1d is not None and self._light_1d is not lease:
                raise ValueError("PublicationStore already has a light-1D lease")
            self._light_1d = lease

    def publish_light_1d(self, record, *, source_identity: str):
        """Retain a heterogeneous row behind an array-free scalar shell."""
        from xrd_tools.session import Light1DStaleGeneration, Light1DUnavailable

        with self._lock:
            lease = self._light_1d
            if lease is None:
                raise RuntimeError("PublicationStore has no light-1D lease")
            if int(record.generation) != int(lease.generation):
                raise Light1DStaleGeneration("light-1D publication generation is stale")
            label = record.row_identity
            keys = lease.keys()
            victims = [label] if label in self._light_1d_items else []
            if label not in keys and keys and len(keys) >= lease.row_cap:
                oldest = keys[0]
                if oldest not in self._light_1d_items:
                    raise RuntimeError(f"light-1D resident {oldest!r} has no publication shell")
                victims.append(oldest)
            retiring = tuple((victim, self._light_1d_items[victim].guard)
                             for victim in victims)

            def commit(retain_candidate):
                for victim, guard in retiring:
                    self._retire_light_pair_locked(victim, qualified_guard=guard)
                try:
                    retain_candidate()
                except Light1DUnavailable as exc:
                    return exc
                shell = Light1DPublicationShell(label, lease.generation, str(source_identity))
                guard = lease.borrow(label)
                if guard is None:
                    raise RuntimeError("retained light-1D row has no private guard")
                self._light_1d_items[label] = _Light1DPair(
                    shell, self._generation, None, record.active_mode, (), guard,
                )
                return shell

            outcome = lease._preflight_store_record(record=record, retiring=retiring, commit=commit)
            if isinstance(outcome, Light1DUnavailable): raise outcome
            return outcome

    def get_light_1d_shell(self, label: int | str):
        with self._lock:
            pair = self._light_1d_items.get(label)
            return None if pair is None else pair.shell

    def _retire_light_pair_locked(self, label: int | str, *, qualified_guard=None) -> bool:
        pair = self._light_1d_items.get(label)
        if pair is None:
            return False
        lease = self._light_1d
        if lease is None:
            raise RuntimeError("light-1D pair has no bound lease")
        if qualified_guard is None:
            lease._preflight_store_record(retiring=((label, pair.guard),))
        elif pair.guard is not qualified_guard:
            raise RuntimeError("qualified light-1D pair guard changed")
        pair.guard.close()
        try:
            retired = lease.retire(
                label, grant_id=lease.grant_id, generation=lease.generation,
            )
        except BaseException:
            guard = lease.borrow(label)
            if guard is None:
                raise RuntimeError("light-1D private guard restoration failed")
            self._light_1d_items[label] = replace(pair, guard=guard)
            raise
        if not retired:
            guard = lease.borrow(label)
            if guard is None:
                raise RuntimeError("retired light-1D pair is inconsistent")
            self._light_1d_items[label] = replace(pair, guard=guard)
            raise RuntimeError("light-1D pair and lease key are inconsistent")
        self._light_1d_items.pop(label, None)
        return True

    def _compose_locked(
        self, label: int | str, publication: FramePublication | None,
    ) -> FramePublication | None:
        pair = self._light_1d_items.get(label)
        if publication is None or pair is None or not pair.modes:
            return publication
        views = {}
        for template in pair.modes:
            values = pair.guard.modes[template.mode]
            mode_base = publication.record.results_1d.get(
                template.mode, publication.view,
            )
            views[template.mode] = replace(
                mode_base,
                axis_1d=Axis(
                    template.axis_label,
                    template.axis_unit,
                    template.axis_log,
                    values.coordinate,
                ),
                intensity_1d=values.intensity,
                sigma_1d=values.uncertainty,
            )
        active = views[pair.active_mode]
        return replace(
            publication,
            view=replace(
                publication.view,
                axis_1d=active.axis_1d,
                intensity_1d=active.intensity_1d,
                sigma_1d=active.sigma_1d,
            ),
            record=FrameRecord(
                label=publication.label,
                results_1d=views,
                results_2d=publication.record.results_2d,
                active_mode_1d=pair.active_mode,
                active_mode_2d=publication.record.active_mode_2d,
            ),
        )

    @staticmethod
    def _exact_gui_array(array, spec, role: str) -> None:
        dtype = getattr(array, "dtype", None)
        if (
            type(array) is not np.ndarray
            or array.ndim != 1
            or array.shape != (spec.length,)
            or dtype != np.dtype(np.float64)
            or not dtype.isnative
            or spec.itemsize != np.dtype(np.float64).itemsize
            or np.dtype(spec.dtype) != np.dtype(np.float64)
            or spec.shared
        ):
            raise ValueError(f"unsupported GUI light-1D layout for {role}")

    def _validate_gui_light_1d_locked(self, publication, record):
        from xrd_tools.session import (
            Light1DLeaseState, Light1DRecord, Light1DStaleGeneration,
        )

        lease = self._light_1d
        if type(publication) is not FramePublication or type(record) is not Light1DRecord:
            raise TypeError("GUI light-1D publication requires exact values")
        if self.allocation is None or lease is None:
            raise RuntimeError("GUI light-1D publication requires exact bindings")
        if lease.authority.parent_allocation is not self.allocation:
            raise ValueError("GUI light-1D allocation authority is foreign")
        if lease.state is not Light1DLeaseState.ACTIVE:
            raise Light1DStaleGeneration("GUI light-1D lease is not active")
        if publication.generation != self._generation:
            raise Light1DStaleGeneration("GUI publication store generation is stale")
        if record.generation != lease.generation:
            raise Light1DStaleGeneration("GUI light-1D lease generation is stale")
        if publication.label != record.row_identity or publication.record.label != record.row_identity:
            raise ValueError("GUI light-1D label identity mismatch")
        source = record.provenance.get("source_identity")
        scan_key = record.provenance.get("scan_key")
        if (
            type(publication.source_identity) is not str
            or not publication.source_identity
            or source != publication.source_identity
        ):
            raise ValueError("GUI light-1D source identity mismatch")
        if (
            type(publication.scan_key) is not str
            or not publication.scan_key
            or scan_key != publication.scan_key
        ):
            raise ValueError("GUI light-1D scan identity mismatch")
        if publication.raw_ref is not None:
            raise ValueError("GUI light-1D publication cannot retain raw_ref")
        expected = tuple(mode.mode for mode in lease.layout.modes)
        if (
            tuple(record.modes) != expected
            or record.active_mode != lease.layout.active_mode
            or tuple(publication.record.results_1d) != expected
            or publication.record.active_mode_1d != lease.layout.active_mode
        ):
            raise ValueError("GUI light-1D mode topology mismatch")
        templates = []
        for mode_layout in lease.layout.modes:
            mode = mode_layout.mode
            values = record.modes[mode]
            view = publication.record.results_1d[mode]
            if view.axis_1d is None or view.axis_1d.values is None:
                raise ValueError("GUI light-1D mode has no coordinate")
            triples = (
                (values.coordinate, view.axis_1d.values,
                 mode_layout.coordinate, "coordinate"),
                (values.intensity, view.intensity_1d,
                 mode_layout.intensity, "intensity"),
                (values.uncertainty, view.sigma_1d,
                 mode_layout.uncertainty, "uncertainty"),
            )
            for supplied, displayed, spec, role in triples:
                if spec is None:
                    if supplied is not None or displayed is not None:
                        raise ValueError("GUI light-1D uncertainty topology mismatch")
                    continue
                if supplied is None or displayed is None:
                    raise ValueError("GUI light-1D uncertainty topology mismatch")
                self._exact_gui_array(supplied, spec, f"{mode!r}.{role}")
                self._exact_gui_array(displayed, spec, f"display {mode!r}.{role}")
                if not np.array_equal(supplied, displayed, equal_nan=True):
                    raise ValueError("GUI light-1D view and record disagree")
            templates.append(_Light1DModeTemplate(
                mode, view.axis_1d.label, view.axis_1d.unit, view.axis_1d.log,
            ))
        active = publication.record.results_1d[lease.layout.active_mode]
        if not (
            np.array_equal(publication.view.axis_1d.values,
                           active.axis_1d.values, equal_nan=True)
            and np.array_equal(publication.view.intensity_1d,
                               active.intensity_1d, equal_nan=True)
            and (
                publication.view.sigma_1d is active.sigma_1d
                or np.array_equal(publication.view.sigma_1d,
                                  active.sigma_1d, equal_nan=True)
            )
        ):
            raise ValueError("GUI light-1D active projection disagrees")
        return tuple(templates)

    @staticmethod
    def _pair_values_match(pair: _Light1DPair, record) -> bool:
        try:
            return all(
                np.array_equal(left, right, equal_nan=True)
                for mode, values in record.modes.items()
                for left, right in (
                    (pair.guard.modes[mode].coordinate, values.coordinate),
                    (pair.guard.modes[mode].intensity, values.intensity),
                    (pair.guard.modes[mode].uncertainty, values.uncertainty),
                )
                if left is not None or right is not None
            )
        except (KeyError, ValueError, TypeError):
            return False

    def _project_total_victims_locked(self, label: int | str) -> tuple:
        order = tuple(key for key in self._items if key != label) + (label,)
        over = 0 if self._max_items is None else max(0, len(order) - self._max_items)
        victims = []
        for candidate in order:
            if len(victims) == over: break
            if self._label_evictable_locked(candidate): victims.append(candidate)
        return tuple(victims)

    def _install_base_locked(self, publication: FramePublication, *, enforce_total=True) -> None:
        label = publication.label
        self._items.pop(label, None)
        self._drop_heavy_label_locked(label)
        self._drop_thumb_label_locked(label)
        self._items[label] = publication
        if _publication_has_heavy_payload(publication):
            self._heavy_labels.append(label)
        if publication.view.thumbnail is not None:
            self._thumb_labels.append(label)
        self._enforce_bounds_locked(enforce_total=enforce_total)

    def publish_gui_light_1d(self, publication, light_record):
        from xrd_tools.session import Light1DUnavailable

        with self._lock:
            templates = self._validate_gui_light_1d_locked(
                publication, light_record,
            )
            lease = self._light_1d
            label = publication.label
            base = _without_1d_arrays(publication)
            prior = self._light_1d_items.get(label)
            reuse_pair = False
            if prior is not None:
                identity = (
                    prior.store_generation == self._generation
                    and prior.shell.source_identity == publication.source_identity
                    and prior.scan_key == publication.scan_key
                    and prior.active_mode == light_record.active_mode
                    and prior.modes == templates
                )
                if identity and self._pair_values_match(prior, light_record):
                    reuse_pair = True
                elif not identity:
                    raise ValueError("GUI light-1D pair identity changed")
            total_victims = self._project_total_victims_locked(label)
            if label in total_victims: raise Light1DUnavailable("incoming GUI publication is pinned")
            keys = lease.keys()
            victims = list(dict.fromkeys((() if reuse_pair else ((label,) if prior else ())) + total_victims))
            if not reuse_pair and label not in keys and len(keys) >= lease.row_cap > 0 and not any(victim in keys for victim in victims):
                victims.append(keys[0])
            retiring = tuple((victim, getattr(self._light_1d_items.get(victim), "guard", None))
                             for victim in victims if victim in keys or victim in self._light_1d_items)
            def commit(retain_candidate):
                for victim in victims:
                    pair = self._light_1d_items.get(victim)
                    if pair is not None:
                        self._retire_light_pair_locked(victim, qualified_guard=pair.guard)
                    if victim != label:
                        self._items.pop(victim, None)
                        self._drop_heavy_label_locked(victim)
                        self._drop_thumb_label_locked(victim)
                if not reuse_pair:
                    try:
                        retain_candidate()
                    except Light1DUnavailable:
                        pass
                    else:
                        guard = lease.borrow(label)
                        if guard is None:
                            lease.retire(label, grant_id=lease.grant_id, generation=lease.generation)
                            raise RuntimeError("retained GUI light-1D row has no guard")
                        self._light_1d_items[label] = _Light1DPair(
                            Light1DPublicationShell(label, lease.generation,
                                                    publication.source_identity),
                            self._generation, publication.scan_key,
                            light_record.active_mode, templates, guard,
                        )
                self._install_base_locked(base, enforce_total=False)
                return self._compose_locked(label, self._items.get(label))

            return lease._preflight_store_record(record=light_record, retiring=retiring, commit=commit, needs_retain=not reuse_pair)

    def light_1d_cleanup_hooks(
        self, exact_lease, *, cancel=None, drain=None, verify=None, release=None,
    ):
        from xrd_tools.session import (
            Light1DCleanupHooks, Light1DLeaseState, Light1DRetentionLease,
        )

        if type(exact_lease) is not Light1DRetentionLease:
            raise TypeError("cleanup hooks require an exact light-1D lease")
        with self._lock:
            if (
                self._light_1d is not exact_lease
                or self.allocation is None
                or exact_lease.authority.parent_allocation is not self.allocation
                or exact_lease.state is not Light1DLeaseState.ACTIVE
            ):
                raise RuntimeError("cleanup hooks require the exact active binding")

        hooks = None

        def clear_pairs():
            exact_lease._assert_cleanup_callback(hooks, "clear", clear_pairs, 2)
            with self._lock:
                if self._light_1d is not exact_lease:
                    raise RuntimeError("cleanup clear has a foreign store binding")
                for label in tuple(self._light_1d_items):
                    pair = self._light_1d_items[label]
                    pair.guard.close()
                    self._light_1d_items.pop(label)

        def detach_store():
            exact_lease._assert_cleanup_callback(hooks, "detach", detach_store, 3)
            with self._lock:
                if (
                    self._light_1d is not exact_lease
                    or self.allocation is None
                    or exact_lease.authority.parent_allocation is not self.allocation
                ):
                    raise RuntimeError("cleanup detach has a foreign store binding")
                if (
                    self._light_1d_items
                    or exact_lease.keys()
                    or exact_lease.pending_hydration_count
                    or any(item.raw_ref is not None
                           or _publication_has_1d_arrays(item)
                           for item in self._items.values())
                ):
                    raise RuntimeError("cleanup detach found retained light-1D state")
                self._items.clear()
                self._heavy_labels.clear()
                self._thumb_labels.clear()
                self._carryover.clear()
                self._generation += 1
                self.allocation = None
                self._light_1d = None

        hooks = Light1DCleanupHooks(
            cancel=cancel,
            drain=drain,
            clear=clear_pairs,
            detach=detach_store,
            verify=verify,
            release=release,
        )
        return hooks

    def ndarray_owner_census(self):
        """Public split census derived from the store's actual owned state."""
        with self._lock:
            lease = self._light_1d
            return {
                "lease": (
                    frozenset() if lease is None else lease.owned_buffer_ids
                ),
                "publication": self._publication_ndarray_roots_locked(),
            }

    @staticmethod
    def _ndarray_root(value: np.ndarray) -> np.ndarray:
        root = value
        while isinstance(root.base, np.ndarray):
            root = root.base
        return root

    def _publication_ndarray_roots_locked(self) -> frozenset[int]:
        roots: set[int] = set()
        seen: set[int] = set()

        def visit(value) -> None:
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            if isinstance(value, np.ndarray):
                roots.add(id(self._ndarray_root(value)))
                return
            if isinstance(value, Light1DPublicationShell):
                # The borrow is a non-owning projection into the lease.  Its
                # arrays belong exclusively to the lease side of the census.
                return
            if isinstance(value, Mapping):
                for key, item in value.items():
                    visit(key)
                    visit(item)
                return
            if isinstance(value, (tuple, list, set, frozenset)):
                for item in value:
                    visit(item)
                return
            if is_dataclass(value) and not isinstance(value, type):
                for descriptor in fields(value):
                    visit(getattr(value, descriptor.name))

        excluded = {
            "_lock", "_light_1d", "allocation", "_hydrator",
            "_hydrator_1d_many", "_evictable",
        }
        for name, value in self.__dict__.items():
            if name not in excluded:
                visit(value)
        return frozenset(roots)

    def clear(self) -> None:
        """Full reset (a scan boundary): empty everything + bump generation."""
        from xrd_tools.session import Light1DLeaseState

        with self._lock:
            lease = self._light_1d
            if lease is not None and lease.state is not Light1DLeaseState.ACTIVE:
                raise RuntimeError("non-active bound PublicationStore cannot clear")
            for label in tuple(self._items):
                self._retire_light_pair_locked(label)
                self._items.pop(label, None)
                self._drop_heavy_label_locked(label)
                self._drop_thumb_label_locked(label)
                self._carryover.pop(label, None)
            for label in tuple(self._light_1d_items):
                self._retire_light_pair_locked(label)
            self._generation += 1
            self._items.clear()
            self._heavy_labels.clear()
            self._thumb_labels.clear()
            self._carryover.clear()
            if lease is None:
                self.allocation = None

    def begin_reintegrate(self) -> None:
        """Reset for a SAME-SCAN reintegrate pass (Step 6).

        Empties ``_items`` and bumps the generation exactly like :meth:`clear`,
        so the mid-pass display is byte-identical to today (a partial Overall
        view blanks, a single frame re-renders fresh as it is republished).  But
        each frame's prior publication is CARRIED OVER, so when that frame is
        re-upserted this pass, :meth:`upsert` merges the recomputed mode into its
        accumulated record (the prior GI modes survive instead of being wiped).
        Eviction is respected: an evicted frame carries a thinned record, so its
        dropped modes are not resurrected (they rehydrate from disk)."""
        with self._lock:
            if self._light_1d is not None:
                raise RuntimeError(
                    "bound light-1D reintegration requires a new generation"
                )
            self._carryover = {
                # X1 3c: carry the scan owner alongside the record so the
                # reintegrate republish keeps the stamp.
                label: (pub.record, pub.source_identity, pub.scan_key)
                for label, pub in self._items.items()
                if pub.record is not None
            }
            self._generation += 1
            self._items.clear()
            self._heavy_labels.clear()
            self._thumb_labels.clear()

    def end_reintegrate(self) -> None:
        """Drop any carry-over NOT consumed during the pass.

        ``upsert`` pops a carry-over entry only when its frame is republished, so
        a reintegrate that stopped early or skipped a failed frame would otherwise
        leave stale records pinned — and a later scroll-back rehydration of such a
        frame would merge that stale record.  The reintegrate wrappers call this
        in a ``finally`` so it runs on stop/exception too.  Idempotent."""
        with self._lock:
            self._carryover.clear()

    def set_max_heavy_items(self, max_heavy_items: int | None) -> None:
        """Resize the heavy-array retention window and enforce it immediately."""
        if max_heavy_items is not None and max_heavy_items < 0:
            raise ValueError("max_heavy_items must be non-negative or None")
        with self._lock:
            if (
                self.allocation is not None
                and max_heavy_items != self.allocation.publication_heavy_items
            ):
                raise ValueError(
                    "allocation-bound PublicationStore cap cannot diverge"
                )
            self._max_heavy_items = max_heavy_items
            self._enforce_bounds_locked()

    def invalidate(self, labels) -> None:
        """Drop store entries for ``labels`` so display re-hydrates from disk.

        Used when a reintegrate shadow is DISCARDED (Stop / abort): the
        recomputed publications staged this pass are no longer authoritative --
        the canonical (prior) row must win.  Popping the entry (and its
        carry-over, so the stale record isn't merged into the next pass) makes
        the next render re-hydrate the prior row from disk, keeping display and
        lazy-load in agreement.  Generation bumps so in-flight renders re-resolve.
        """
        with self._lock:
            changed = False
            for label in labels:
                pair_removed = self._retire_light_pair_locked(label)
                if self._items.pop(label, None) is not None or pair_removed:
                    self._drop_heavy_label_locked(label)
                    self._drop_thumb_label_locked(label)
                    self._carryover.pop(label, None)
                    changed = True
            if changed:
                self._generation += 1

    def set_hydrator(self, hydrator) -> None:
        """Register the rehydration source for :meth:`get_or_hydrate`."""
        with self._lock:
            self._hydrator = hydrator

    def set_1d_hydrator(self, hydrator) -> None:
        """Register a batch 1D-only rehydration source.

        Overlay/Waterfall selections need only the integrated 1D rows.  This
        hook lets the GUI hydrate those rows in one stacked disk read without
        materializing raw detector images or 2D cakes.
        """
        with self._lock:
            self._hydrator_1d_many = hydrator

    def get_or_hydrate(self, label: int | str, *, commit_gate=None,
                       commit_epoch=None) -> FramePublication | None:
        """Return the publication, rehydrating an evicted payload via the
        registered hydrator (synchronous — call from a background worker
        for disk-backed hydrators).

        X1 O-3 (c3R-b §9.2.4): ``commit_gate`` is the REQUESTING context's
        commit authority.  The hydrator's disk read runs unlocked and off the
        GUI thread as before; only the final insertion is taken through the
        gate, so an invalidation, a rescope or a release that lands mid-read
        linearizes cleanly and the payload is inserted NOWHERE.  Without a gate
        the behaviour is unchanged — that is the idle/legacy path."""
        with self._lock:
            publication = self._compose_locked(label, self._items.get(label))
            hydrator = self._hydrator
        # Rehydrate when the payload is GONE (tier-1 thumbnail-only or tier-2
        # evicted), keyed on real DATA arrays — NOT _publication_has_heavy_payload,
        # which counts the thumbnail as heavy and so wrongly short-circuited a
        # tier-1 (semilight) frame, leaving it stuck on the thumbnail forever.
        # PF-1e: ALSO rehydrate when the full raw is absent but recoverable
        # from the row's source reference — a source-only row (no stored
        # thumbnail -> raw_status "missing") otherwise short-circuited here
        # forever: the worker reported success (a resident lighter item),
        # the full-purpose residency check scored it false, and the raw
        # panel stayed blank while the headless source fallback could read
        # the image the whole time.
        if publication is not None:
            payload_gone = (
                not _publication_has_full_payload(publication)
                and publication.raw_status in ("evicted", "thumbnail", "1d-only")
            )
            if not payload_gone and not publication_raw_recoverable(publication):
                return publication
        if hydrator is None:
            return publication
        try:
            fresh = hydrator(label)
        except Exception:
            return publication
        if fresh is None:
            return publication
        with self._lock:
            if self._light_1d is not None:
                fresh = _without_1d_arrays(fresh)
        if commit_gate is None:
            return self.upsert(fresh)
        if not commit_gate.enter(commit_epoch):
            # The requesting context lost its commit authority while the read
            # was in flight.  The read is simply discarded; nothing is stored.
            return None
        try:
            return self.upsert(fresh)
        finally:
            commit_gate.leave()

    def get_1d_many_or_hydrate(
        self, labels: Iterable[int | str], *, commit_gate=None,
        commit_epoch=None
    ) -> dict[int | str, FramePublication]:
        """Return publications with 1D payloads, batch hydrating misses.

        This deliberately bypasses :meth:`get_or_hydrate`: the full hydrator
        rebuilds raw + cake payloads, which is far too much work for a bottom
        1D plot selection sweep.
        """
        requested = tuple(labels)
        if not requested:
            return {}
        with self._lock:
            if self._light_1d is not None:
                missing = tuple(
                    label for label in requested
                    if label not in self._light_1d_items
                    or label not in self._items
                )
                if missing:
                    raise RuntimeError(
                        "bound light-1D misses require worker hydration tokens"
                    )
                return {
                    label: self._compose_locked(label, self._items[label])
                    for label in requested
                }
            missing = []
            for label in requested:
                publication = self._items.get(label)
                if publication is None or not publication.view.has_1d:
                    missing.append(label)
            hydrator = self._hydrator_1d_many
        if missing and hydrator is not None:
            try:
                # The batch read runs UNLOCKED, exactly like the full hydrator;
                # only the insertions go through the requesting context's
                # commit gate (§9.2.4), so an invalidation mid-read inserts
                # nothing rather than half a batch.
                fresh = tuple(hydrator(tuple(missing)) or ())
                if commit_gate is None:
                    for publication in fresh:
                        if publication is not None:
                            self.upsert(publication)
                elif commit_gate.enter(commit_epoch):
                    try:
                        for publication in fresh:
                            if publication is not None:
                                self.upsert(publication)
                    finally:
                        commit_gate.leave()
            except Exception:
                logger.debug("batch 1D publication hydration failed", exc_info=True)
        return self.get_many(requested)

    def upsert(self, publication: FramePublication) -> FramePublication:
        from xrd_tools.session import Light1DLeaseState

        with self._lock:
            incoming_generation = publication.generation
            label = publication.label
            existing = self._items.get(label)
            foreign_pair = False
            if self._light_1d is not None:
                if self._light_1d.state is not Light1DLeaseState.ACTIVE:
                    raise RuntimeError("non-active bound PublicationStore cannot upsert")
                if publication.raw_ref is not None:
                    raise ValueError("bound PublicationStore refuses raw_ref")
                if _publication_has_1d_arrays(publication):
                    raise ValueError(
                        "bound PublicationStore generic upsert refuses 1-D arrays"
                    )
                pair = self._light_1d_items.get(label)
                if pair is not None and not (
                    incoming_generation == self._generation
                    and publication.source_identity == pair.shell.source_identity
                    and publication.scan_key == pair.scan_key
                ):
                    self._retire_light_pair_locked(label)
                    foreign_pair = True
            # A STALE incoming (queued before a clear()/generation bump) for a
            # frame ALREADY present is from a superseded epoch — DROP it and keep
            # the current entry, so old-scan data can neither replace nor splice
            # into the live frame (codex follow-up review).  Rehydration stamps
            # the CURRENT store generation (display_data._rehydrate_publication),
            # so it is never seen as stale.  A stale incoming for a NEW label is
            # still stored below, for legacy/sessionless callers.
            if (
                existing is not None
                and incoming_generation != self._generation
                and not foreign_pair
            ):
                return existing
            if incoming_generation != self._generation:
                publication = replace(publication, generation=self._generation)
            if existing is not None:
                # Fork B accumulation, hardened: same-epoch (guaranteed here) +
                # same-source -> merge the incoming record's modes into the
                # existing record, so the store carries every GI mode computed
                # for this frame and the stored .view stays the latest active
                # projection (display consumers unchanged).  A DIFFERENT
                # non-empty source_identity (a label reused within one epoch)
                # must NOT splice -> plain wholesale replace.
                if (
                    existing.generation == self._generation
                    and _same_source(existing, publication)
                    and _scan_owners_compatible(
                        existing.scan_key, publication.scan_key)
                ):
                    publication = replace(
                        publication,
                        record=_merge_records(existing.record, publication.record),
                        # X1 3c: the same-source merge preserves the explicit
                        # scan owner — incl. when a legacy UNSTAMPED republish
                        # (or an unstamped hydrator result) folds onto a
                        # stamped entry.
                        scan_key=(publication.scan_key
                                  if publication.scan_key is not None
                                  else existing.scan_key),
                    )
                self._items.pop(label)
                self._drop_heavy_label_locked(label)
                self._drop_thumb_label_locked(label)
            else:
                # Step 6: first re-upsert of this frame in a reintegrate pass —
                # merge the recomputed mode into the record carried over from the
                # previous pass (begin_reintegrate), so the frame's accumulated
                # GI modes survive.  Source-guarded like the in-store merge; the
                # carried entry is consumed (popped) so it merges only once.
                carried = self._carryover.pop(label, None)
                if carried is not None:
                    carried_record, carried_source, carried_scan_key = carried
                    if (
                        _same_source_id(
                            carried_source, publication.source_identity)
                        and _scan_owners_compatible(
                            carried_scan_key, publication.scan_key)
                    ):
                        publication = replace(
                            publication,
                            record=_merge_records(carried_record, publication.record),
                            # X1 3c: restore the carried scan owner for a
                            # legacy unstamped republish of the same source.
                            scan_key=(publication.scan_key
                                      if publication.scan_key is not None
                                      else carried_scan_key),
                        )
            self._items[publication.label] = publication
            if _publication_has_heavy_payload(publication):
                self._heavy_labels.append(label)
            if publication.view.thumbnail is not None:
                self._thumb_labels.append(label)
            self._enforce_bounds_locked()
            if self._light_1d is None:
                return publication
            return self._compose_locked(label, self._items.get(label)) or publication

    def extend(self, publications: Iterable[FramePublication]) -> tuple[FramePublication, ...]:
        with self._lock:
            return tuple(self.upsert(publication) for publication in publications)

    def get(self, label: int | str) -> FramePublication | None:
        with self._lock:
            return self._compose_locked(label, self._items.get(label))

    def has_heavy_payload(self, label: int | str) -> bool:
        with self._lock:
            publication = self._items.get(label)
            return bool(
                publication is not None
                and _publication_has_heavy_payload(publication)
            )

    def has_thumbnail(self, label: int | str) -> bool:
        with self._lock:
            publication = self._items.get(label)
            return bool(
                publication is not None
                and publication.view.thumbnail is not None
            )

    def evict_heavy(self, label: int | str) -> bool:
        """Drop full arrays for one evictable publication, retaining thumbnail."""
        with self._lock:
            publication = self._items.get(label)
            if (
                publication is None
                or not _publication_has_heavy_payload(publication)
                or not self._label_evictable_locked(label)
            ):
                return False
            self._items[label] = _semilight_publication(publication)
            self._drop_heavy_label_locked(label)
            return True

    def evict_thumbnail(self, label: int | str) -> bool:
        """Drop the thumbnail for one evictable publication."""
        with self._lock:
            publication = self._items.get(label)
            if (
                publication is None
                or publication.view.thumbnail is None
                or not self._label_evictable_locked(label)
            ):
                return False
            self._items[label] = _lightweight_publication(publication)
            self._drop_heavy_label_locked(label)
            self._drop_thumb_label_locked(label)
            return True

    def discard(self, label: int | str) -> bool:
        """Remove one evictable publication from the resident lookup tier."""
        with self._lock:
            if label not in self._items or not self._label_evictable_locked(label):
                return False
            self._retire_light_pair_locked(label)
            self._items.pop(label, None)
            self._drop_heavy_label_locked(label)
            self._drop_thumb_label_locked(label)
            self._carryover.pop(label, None)
            return True

    def get_many(
        self, labels: Iterable[int | str]
    ) -> dict[int | str, FramePublication]:
        """Return stored publications for ``labels`` under one lock.

        Display code uses this for the common selected-frame render path.  It
        avoids copying the full publication store on every frame update while
        preserving the existing immutable-publication contract.
        """
        with self._lock:
            return {
                label: self._compose_locked(label, publication)
                for label in labels
                if (publication := self._items.get(label)) is not None
            }

    def labels(self) -> tuple[int | str, ...]:
        with self._lock:
            return tuple(self._items)

    def snapshot(self) -> Mapping[int | str, FramePublication]:
        with self._lock:
            return MappingProxyType({
                label: self._compose_locked(label, publication)
                for label, publication in self._items.items()
            })

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

    def set_evictable_probe(self, probe) -> None:
        """Register the MEM1-15 persist gate: ``probe(label) -> bool``.

        ``True`` means the label is safe to drop from tier 0 (its record is
        persisted — rehydratable from disk).  ``False`` pins it ("owed",
        never lost).  ``None`` restores the ungated legacy behavior (viewer
        stores, whose publications come FROM disk, keep it unset)."""
        with self._lock:
            self._evictable = probe

    def _label_evictable_locked(self, label: int | str) -> bool:
        probe = self._evictable
        if probe is None:
            return True
        try:
            return bool(probe(label))
        except Exception:
            return False

    def _enforce_bounds_locked(self, *, enforce_total=True) -> None:
        if enforce_total and self._max_items is not None:
            probe = self._evictable
            if probe is None:
                while len(self._items) > self._max_items:
                    label = next(iter(self._items))
                    self._retire_light_pair_locked(label)
                    self._items.pop(label, None)
                    self._drop_heavy_label_locked(label)
                    self._drop_thumb_label_locked(label)
            else:
                # Persist-gated (MEM1-15): evict the oldest EVICTABLE labels.
                # Unpersisted publications pin memory rather than being
                # dropped — the store may transiently exceed max_items while
                # a save is in flight (the writer's flush marks persisted
                # every ~150 ms, so pinning is short-lived).
                over = len(self._items) - self._max_items
                if over > 0:
                    for label in list(self._items):
                        if over <= 0:
                            break
                        try:
                            if not probe(label):
                                continue
                        except Exception:
                            continue
                        self._retire_light_pair_locked(label)
                        self._items.pop(label, None)
                        self._drop_heavy_label_locked(label)
                        self._drop_thumb_label_locked(label)
                        over -= 1

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
