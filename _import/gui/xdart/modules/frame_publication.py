"""Qt-free frame publication contract for xdart displays.

``FramePublication`` is the GUI-side envelope around
``ssrl_xrd_tools.core.FrameView``.  It is deliberately separate from
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

from ssrl_xrd_tools.core import (
    FrameView,
    TwoDKind,
    numeric_metadata,
)


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
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_ranges", _readonly_mapping(self.axis_ranges))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class FramePublication:
    """Display publication snapshot for one frame."""

    view: FrameView
    source_identity: str = ""
    generation: int = 0
    raw_ref: Any | None = None
    raw_status: str = "unknown"
    metadata_raw: Mapping[str, Any] = field(default_factory=dict)
    metadata_numeric: Mapping[str, float] = field(default_factory=dict)
    diagnostics: PublicationDiagnostics = field(default_factory=PublicationDiagnostics)

    def __post_init__(self) -> None:
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
    errors: list[str] = []
    finite_1d = _finite_fraction(view.intensity_1d)
    finite_2d = _finite_fraction(view.intensity_2d)
    dummy_2d = _dummy_fraction(view.intensity_2d)
    axis_ranges = {
        "axis_1d": _axis_range(view.axis_1d),
        "axis_2d_x": _axis_range(view.axis_2d_x),
        "axis_2d_y": _axis_range(view.axis_2d_y),
    }

    if view.has_1d and (finite_1d is None or finite_1d == 0.0):
        errors.append("1D intensity contains no finite values")
    if view.has_2d:
        if finite_2d is None or finite_2d == 0.0:
            errors.append("2D intensity contains no finite values")
        if dummy_2d is not None and dummy_2d >= 0.95 and not allow_dummy_2d:
            errors.append("2D intensity is almost entirely dummy pixels")
        if view.two_d_kind is not TwoDKind.Q_CHI and view.incident_angle is None:
            warnings.append("GI 2D publication has no resolved incident angle")
        for name in ("axis_2d_x", "axis_2d_y"):
            if axis_ranges[name] is None:
                errors.append(f"{name} has no finite range")

    diagnostics = PublicationDiagnostics(
        finite_fraction_1d=finite_1d,
        finite_fraction_2d=finite_2d,
        dummy_fraction_2d=dummy_2d,
        axis_ranges=axis_ranges,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )
    if raise_on_error and diagnostics.errors:
        raise ValueError("; ".join(diagnostics.errors))
    return diagnostics


def publication_has_1d_errors(publication: FramePublication) -> bool:
    return any(msg.startswith("1D") for msg in publication.diagnostics.errors)


def publication_has_2d_errors(publication: FramePublication) -> bool:
    return any(
        msg.startswith("2D") or msg.startswith("axis_2d")
        for msg in publication.diagnostics.errors
    )


def publication_from_live_frame(
    frame: Any,
    *,
    generation: int = 0,
    source_identity: str | None = None,
    include_raw: bool = False,
    validate: bool = True,
) -> FramePublication:
    """Adapt a current xdart ``LiveFrame``-like object into a publication."""

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
    publication = FramePublication(
        view=view,
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
    generation: int = 0,
    source_identity: str = "",
    raw_ref: Any | None = None,
    raw_status: str = "unknown",
    validate: bool = True,
) -> FramePublication:
    """Wrap a headless :class:`FrameView` in the xdart publication envelope."""

    publication = FramePublication(
        view=view,
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

    from ssrl_xrd_tools.io import read_frame_view

    view = read_frame_view(
        scan_file,
        frame,
        entry=entry,
        include_thumbnail=include_thumbnail,
    )
    return publication_from_frame_view(
        view,
        generation=generation,
        source_identity=str(scan_file),
        raw_status=("thumbnail" if view.thumbnail is not None else "missing"),
        validate=validate,
    )


class PublicationStore:
    """Small generation-aware store for frame publications."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._generation = 0
        self._items: dict[int | str, FramePublication] = {}

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def clear(self) -> None:
        with self._lock:
            self._generation += 1
            self._items.clear()

    def upsert(self, publication: FramePublication) -> FramePublication:
        with self._lock:
            if publication.generation != self._generation:
                publication = replace(publication, generation=self._generation)
            self._items[publication.label] = publication
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


__all__ = [
    "FramePublication",
    "PublicationDiagnostics",
    "PublicationStore",
    "publication_from_frame_view",
    "publication_from_live_frame",
    "publication_from_nexus_frame",
    "publication_has_1d_errors",
    "publication_has_2d_errors",
    "validate_publication",
]
