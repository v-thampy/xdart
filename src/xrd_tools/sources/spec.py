# -*- coding: utf-8 -*-
"""Metadata-only SPEC FrameSource.

A SPEC file records a scan's per-point counters + scanned motor (the ``#L``
columns) and the scan-start motor positions (``#O``/``#P``), but **no detector
images**.  :class:`SpecSource` surfaces one scan's points as "frames" carrying
that metadata so the Scan Plot tool can plot any column vs any column — while
``load_frame`` raises, so ROI stats stay correctly disabled (there is no raw to
reduce).  Built on :mod:`silx.io.specfile` (imported lazily, like the rest of
the SPEC metadata path)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from xrd_tools.core.scan import SourceCapabilities, SourceKind, SourceSpec
from xrd_tools.sources.base import BaseFrameSource


class SpecSource(BaseFrameSource):
    """FrameSource over ONE scan of a SPEC file — metadata only, no images.

    ``scan`` selects the scan (``"1.1"`` / ``1`` / ``"1"``); the default is the
    first scan in the file.  Each frame is a scan point; ``metadata_for`` merges
    the per-point columns with the (constant) scan-start motor positions, the
    per-point values winning when a name appears in both."""

    kind = SourceKind.SPEC

    def __init__(self, path: str | Path, *, scan: Any = None,
                 name: str | None = None) -> None:
        from silx.io.specfile import SpecFile  # noqa: PLC0415  (lazy, like io.metadata)

        self.path = Path(path)
        sf = SpecFile(str(self.path))
        keys = list(sf.keys())
        scan_key = self._resolve_scan_key(keys, scan)
        self.scan_key = scan_key

        columns: dict[str, np.ndarray] = {}
        motors: dict[str, float] = {}
        npts = 0
        if scan_key is not None:
            sc = sf[scan_key]
            data = np.asarray(sc.data)
            npts = int(data.shape[1]) if data.ndim == 2 else 0
            for label in sc.labels:
                try:
                    columns[str(label)] = np.asarray(
                        sc.data_column_by_name(label), dtype=float)
                except Exception:
                    pass
            for mname in sc.motor_names:
                try:
                    motors[str(mname)] = float(sc.motor_position_by_name(mname))
                except Exception:
                    pass
        self._columns = columns
        self._motors = motors

        super().__init__(
            name=name or f"{self.path.stem} [{scan_key}]" if scan_key
            else self.path.stem,
            frame_indices=list(range(npts)),
            spec=SourceSpec(self.path, SourceKind.SPEC),
            capabilities=SourceCapabilities(
                supports_random_access=True,
                supports_chunks=False,
                has_metadata=True,
                has_raw_references=False,
                has_scan_manifest=True,
            ),
        )

    @staticmethod
    def _resolve_scan_key(keys, scan):
        """Pick the scan key (``"N.1"``) — ``scan`` matched exactly or as ``N`` →
        ``N.1``; default the first scan.  ``None`` when the file has no scans."""
        if not keys:
            return None
        if scan is None:
            return keys[0]
        s = str(scan)
        if s in keys:
            return s
        if f"{s}.1" in keys:
            return f"{s}.1"
        return keys[0]

    @property
    def motors(self) -> dict[str, np.ndarray]:
        """The per-point ``#L`` columns as whole arrays (one value per frame)."""
        return dict(self._columns)

    def metadata_for(self, index: int) -> Mapping[str, Any]:
        i = int(index)
        md: dict[str, Any] = dict(self._motors)        # constant scan-start motors
        md.update({k: v[i] for k, v in self._columns.items() if i < len(v)})
        return md

    def load_frame(self, index: int) -> np.ndarray:
        raise NotImplementedError(
            "SPEC sources are metadata-only (no detector images); "
            "ROI stats need a source with raw frames")


__all__ = ["SpecSource"]
