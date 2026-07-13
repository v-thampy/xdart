# -*- coding: utf-8 -*-
"""Qt-light analysis context for popup tools.

Popup analysis dialogs should not reach through ``staticWidget`` into
wranglers, integrators, display widgets, or HDF5 viewers.  They receive this
small context instead: a set of provider callables owned by the main tab.  The
providers may be backed by live publications, reloaded NeXus rows, or future
session stores, but the dialogs only see analysis-ready data.

The context deliberately keeps live fitting intact.  ``current_pattern_tuple``
is called whenever a live frame is published; the peak fitter can keep using a
latest-wins worker while future strain/texture tools share the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

PatternTuple = tuple[Any, Any, str]
PatternProvider = Callable[[], PatternTuple | None]
FramePatternProvider = Callable[[int], PatternTuple | None]
ScanUriProvider = Callable[[], str | None]
MaskProvider = Callable[[str | None], Any]
ReadLockProvider = Callable[[str | None], Any]
FrameLabelsProvider = Callable[[], Sequence[Any]]
MetadataProvider = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class PatternData:
    """One analysis-ready 1D pattern.

    ``x`` and ``y`` are intentionally typed as ``Any`` so the context stays
    independent of NumPy at import time.  Consumers convert to arrays at the
    edge where they already know the analysis they are running.
    """

    x: Any
    y: Any
    x_label: str = "q"
    frame_label: str | None = None

    def as_tuple(self) -> PatternTuple:
        return self.x, self.y, self.x_label


def _coerce_pattern(data: PatternTuple | PatternData | None,
                    frame_label: str | None = None) -> PatternData | None:
    if data is None:
        return None
    if isinstance(data, PatternData):
        if frame_label is not None and data.frame_label is None:
            return PatternData(data.x, data.y, data.x_label, frame_label)
        return data
    try:
        x, y, label = data
    except (TypeError, ValueError):
        return None
    return PatternData(x=x, y=y, x_label=str(label or "q"),
                       frame_label=frame_label)


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    """Stable data contract for analysis popups.

    The owner supplies the providers.  Dialogs and tools should use these
    methods rather than reaching into GUI implementation details.
    """

    current_pattern_provider: PatternProvider = lambda: None
    frame_pattern_provider: FramePatternProvider = lambda _idx: None
    scan_uri_provider: ScanUriProvider = lambda: None
    mask_provider: MaskProvider = lambda _uri: None
    read_lock_provider: ReadLockProvider = lambda _uri: None
    frame_labels_provider: FrameLabelsProvider = tuple
    metadata_provider: MetadataProvider = dict
    extras: Mapping[str, Any] = field(default_factory=dict)

    def current_pattern(self) -> PatternData | None:
        labels = self.frame_labels()
        frame_label = str(labels[0]) if labels else None
        return _coerce_pattern(self.current_pattern_provider(), frame_label)

    def current_pattern_tuple(self) -> PatternTuple | None:
        data = self.current_pattern()
        return None if data is None else data.as_tuple()

    def pattern_for_frame(self, frame_index: int) -> PatternData | None:
        return _coerce_pattern(self.frame_pattern_provider(int(frame_index)),
                               str(frame_index))

    def pattern_tuple_for_frame(self, frame_index: int) -> PatternTuple | None:
        data = self.pattern_for_frame(frame_index)
        return None if data is None else data.as_tuple()

    def current_scan_uri(self) -> str | None:
        return self.scan_uri_provider()

    def mask_for_scan_uri(self, uri: str | None = None) -> Any:
        return self.mask_provider(uri if uri is not None else self.current_scan_uri())

    def read_lock_for_uri(self, uri: str | None = None) -> Any:
        """Writer-coordinating lock for reading ``uri``, or None when no
        in-process writer shares that file (dialogs then read unlocked)."""
        return self.read_lock_provider(
            uri if uri is not None else self.current_scan_uri())

    def frame_labels(self) -> tuple[Any, ...]:
        try:
            return tuple(self.frame_labels_provider() or ())
        except Exception:
            return ()

    def metadata(self) -> dict[str, Any]:
        try:
            return dict(self.metadata_provider() or {})
        except Exception:
            return {}
