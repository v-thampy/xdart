# -*- coding: utf-8 -*-
"""SPEC FrameSource — the primary scan-definition source.

A scan is ``(SPEC file, scan number)``: the SPEC file gives the per-point
metadata (the ``#L`` columns + the constant ``#O``/``#P`` motors), and — when an
image directory is supplied — the matching detector image files give the raw
frames (the RSM ``ScanInfo(spec_path, img_dir)`` pattern, stitching design §5.4).

* **Metadata only** (no ``image_dir``): ``load_frame`` raises, so ROI stats stay
  disabled — but every column plots.
* **With images** (``image_dir`` + the scan's filename stem): ``load_frame``
  reads the raw frame, so ROI / stitch / RSM can reduce it.

All SPEC parsing + image I/O is reused from the headless ``xrd_tools.io`` layer
(``io.spec`` / ``io.image``); this is the thin :class:`FrameSource` wrapper.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from xrd_tools.core.scan import SourceCapabilities, SourceKind, SourceSpec
from xrd_tools.sources.base import BaseFrameSource


class SpecSource(BaseFrameSource):
    """FrameSource over ONE scan of a SPEC file (metadata; raw images optional).

    Parameters
    ----------
    path
        The (typically extensionless) SPEC file.
    scan
        Scan selector — ``"1.1"`` / ``1`` / ``"1"``; default = the first scan.
    image_dir
        Directory of the detector image files.  When given, the scan's frames
        become loadable raw images (ROI/stitch/RSM); omitted ⇒ metadata only.
    image_stem
        Filename substring selecting this scan's images (default the SSRL
        convention ``_{spec_stem}_scan{N}_``).
    read_image_kwargs
        Passed to :func:`xrd_tools.io.image.read_image` (e.g. ``detector_shape``
        / ``raw_dtype`` / ``raw_header_skip`` for headerless raw binaries).
    """

    kind = SourceKind.SPEC

    def __init__(self, path: str | Path, *, scan: Any = None,
                 name: str | None = None, image_dir: str | Path | None = None,
                 image_stem: str | None = None,
                 read_image_kwargs: Mapping[str, Any] | None = None) -> None:
        from xrd_tools.io.spec import list_spec_scans, read_spec_scan_table

        self.path = Path(path)
        self.available_scans = list_spec_scans(self.path)
        self.scan_key = self._resolve_scan_key(self.available_scans, scan)

        columns: dict[str, np.ndarray] = {}
        motors: dict[str, float] = {}
        npts = 0
        if self.scan_key is not None:
            columns, motors, npts = read_spec_scan_table(self.path, self.scan_key)
        self._columns = columns
        self._motors = motors

        # Optional raw images — located once by stem in image_dir.  The matched
        # files may be one-file-per-frame (raw / tiff / edf / cbf) OR a single
        # multi-frame container (Eiger master h5), so build an explicit
        # (file, frame-in-file) map and read via io.image.read_image (format-
        # agnostic).
        self.image_dir = Path(image_dir) if image_dir else None
        self._read_image_kwargs = dict(read_image_kwargs or {})
        self._frame_map: list[tuple[Path, int]] = []
        if self.image_dir is not None and self.scan_key is not None:
            from xrd_tools.io.image import find_image_files
            stem = image_stem or self._default_image_stem(self.scan_key)
            files = list(find_image_files(self.image_dir, stem=stem))
            if len(files) == 1 and npts and npts > 1:
                # one multi-frame container -> frame index within it
                self._frame_map = [(files[0], i) for i in range(npts)]
            else:
                self._frame_map = [(f, 0) for f in files]
        has_raw = bool(self._frame_map)

        # Frame count: the metadata point count is authoritative; with one file
        # per frame, clamp to the images actually found.
        if has_raw and npts:
            n_frames = (npts if len(self._frame_map) >= npts
                        else len(self._frame_map))
        elif has_raw:
            n_frames = len(self._frame_map)
        else:
            n_frames = npts

        super().__init__(
            name=name or self._default_name(),
            frame_indices=list(range(n_frames)),
            spec=SourceSpec(self.path, SourceKind.SPEC),
            capabilities=SourceCapabilities(
                supports_random_access=True,
                supports_chunks=has_raw,
                has_metadata=True,
                has_raw_references=has_raw,
                has_scan_manifest=True,
            ),
        )

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _resolve_scan_key(keys, scan):
        """Pick the scan key — ``scan`` matched exactly or as ``N`` → ``N.1``;
        default the first scan.  ``None`` when the file has no scans."""
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

    def _default_name(self):
        return (f"{self.path.stem} [{self.scan_key}]"
                if self.scan_key else self.path.stem)

    def _default_image_stem(self, scan_key):
        """The image-filename substring for this scan: ``{spec_stem}_scan{N}_``.
        The trailing ``_`` anchors the scan number so ``scan5`` does not also
        match ``scan50``; override via ``image_stem`` for other conventions."""
        scan_n = str(scan_key).split(".")[0]
        return f"{self.path.stem}_scan{scan_n}_"

    # ---- FrameSource API ------------------------------------------------
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
        if not self._frame_map:
            raise NotImplementedError(
                "SPEC source has no image directory — metadata only; pass "
                "image_dir to enable raw frames (ROI/stitch/RSM)")
        path, frame = self._frame_map[int(index)]
        from xrd_tools.io.image import read_image
        return np.asarray(
            read_image(path, frame=frame, **self._read_image_kwargs),
            dtype=float)


__all__ = ["SpecSource"]
