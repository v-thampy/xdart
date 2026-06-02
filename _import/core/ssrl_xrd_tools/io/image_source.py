"""Image-source classification + loading for the display layer.

A single, headless, tested boundary for the question xdart's Image Viewer
keeps getting wrong: *what kind of file is this, and how do I get a
displayable 2D image for a frame?*  It folds together the
``_is_xdart_processed`` guess and the raw-master-vs-thumbnail fallback
chain that previously lived (and broke) inside the GUI.

Three entry points, built on the existing :mod:`ssrl_xrd_tools.io.read`
primitives (:func:`get_raw_frame`, :func:`get_thumbnail`) and
:func:`ssrl_xrd_tools.io.image.read_image`:

* :func:`classify_image_source` — what is this file? (raw detector master /
  processed-xdart / thumbnail-only / unknown), its frame labels, and whether
  a resolvable raw source or a thumbnail exists.
* :func:`load_image_frame` — a genuine raw detector frame (master / tiff /
  eiger), by 0-based index.
* :func:`load_processed_raw_or_thumbnail` — for a processed ``.nxs``: the
  full-resolution raw via the per-frame source pointer, else the dequantized
  thumbnail; the result records which one it returned so the caller never has
  to re-open the file to find out (and never re-applies a flat detector mask
  to a thumbnail).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ImageSourceKind",
    "ImageSourceInfo",
    "RawFrameResult",
    "classify_image_source",
    "load_image_frame",
    "load_processed_raw_or_thumbnail",
]

_HDF5_SUFFIXES = (".h5", ".hdf5", ".nxs", ".cxi")

# Raw-detector dataset candidates (mirrors the GUI's old _is_xdart_processed).
_RAW_DATASET_CANDIDATES = (
    "entry/instrument/detector/data",
    "entry/instrument/detector/data_000001",
    "entry/data/data",
    "entry/measurement/data",
    "entry/data",
)


class ImageSourceKind(str, Enum):
    """What a file is, for image display."""
    RAW_MASTER = "raw_master"            # genuine raw detector data (master/tiff/eiger)
    PROCESSED_XDART = "processed_xdart"  # processed v2 .nxs with a resolvable raw source
    THUMBNAIL_ONLY = "thumbnail_only"    # processed v2 .nxs, only dequantizable thumbnails
    UNKNOWN = "unknown"                  # can't tell / unreadable


@dataclass(frozen=True)
class ImageSourceInfo:
    """Result of :func:`classify_image_source`."""
    kind: ImageSourceKind
    path: str
    frame_labels: tuple = ()   # tuple[int, ...] — frame labels available to display
    has_raw: bool = False      # a full-resolution raw image is resolvable
    has_thumbnail: bool = False  # a stored thumbnail is available

    @property
    def n_frames(self) -> int:
        return len(self.frame_labels)


@dataclass(frozen=True)
class RawFrameResult:
    """Result of :func:`load_processed_raw_or_thumbnail`.

    ``source`` is ``"raw"`` (full-resolution detector image — a detector/flat
    mask may be applied), ``"thumbnail"`` (mask already baked in; do NOT
    re-apply a flat mask) or ``"none"`` (nothing available → render blank)."""
    image: "np.ndarray | None"
    source: str   # "raw" | "thumbnail" | "none"
    frame: int


# ── helpers ───────────────────────────────────────────────────────────

def _is_hdf5(path: Path) -> bool:
    return path.suffix.lower() in _HDF5_SUFFIXES


def _frame_labels_from_groups(entry) -> list:
    """*Displayable* frame labels from ``entry/frames/frame_NNNN`` groups,
    sorted.  A group counts only if it carries a thumbnail or a source
    pointer — i.e. something :func:`load_processed_raw_or_thumbnail` can
    actually return an image for.  This is deliberately NOT the union of an
    integrated dataset's ``frame_index`` (which can list labels that have no
    frame group, e.g. gapped/offset eiger scans) — using that union is what
    left the Image Viewer blank when its first label had no group."""
    frames = entry.get("frames")
    labels = []
    if frames is not None:
        for name in frames:
            if not name.startswith("frame_"):
                continue
            fg = frames.get(name)
            if fg is None:
                continue
            if fg.get("thumbnail") is None and fg.get("source") is None:
                continue   # nothing loadable for this frame
            try:
                labels.append(int(name.split("frame_")[1]))
            except (ValueError, IndexError):
                continue
    return sorted(labels)


def _resolve_source_master(scan_file: Path, src) -> "Path | None":
    """Resolve a frame's ``source`` group to an existing master path, using
    the same candidate order as :func:`get_raw_frame`.  Returns None if no
    candidate exists."""
    from ssrl_xrd_tools.io.read import _decode

    if src is None or "path" not in src:
        return None
    rel = _decode(src["path"][()])
    if not rel:
        return None
    rel_path = Path(str(rel)).expanduser()
    candidates = []
    if rel_path.is_absolute():
        candidates.append(rel_path)
    candidates.extend([
        scan_file.parent / rel_path,
        scan_file.parent / rel_path.name,
        rel_path,
    ])
    seen = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            resolved = cand
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


# ── public API ────────────────────────────────────────────────────────

def classify_image_source(path) -> ImageSourceInfo:
    """Classify ``path`` for image display without the caller guessing.

    Non-HDF5 files (tiff / raw / eiger master) are RAW_MASTER.  An HDF5 file
    is PROCESSED_XDART when it carries integrated data / per-frame groups and
    at least one frame resolves to a raw source master; THUMBNAIL_ONLY when it
    is processed but only thumbnails are reachable; RAW_MASTER when it holds a
    raw detector dataset; UNKNOWN when it can't be read or classified.
    """
    import h5py
    from ssrl_xrd_tools.io.image import count_frames

    p = Path(path)

    if not _is_hdf5(p):
        # tiff / raw / eiger / non-HDF master — a genuine raw detector file.
        try:
            n = int(count_frames(p))
        except Exception:
            logger.debug("classify_image_source: count_frames failed for %s", p,
                         exc_info=True)
            n = 1
        n = max(n, 1)
        return ImageSourceInfo(
            kind=ImageSourceKind.RAW_MASTER, path=str(p),
            frame_labels=tuple(range(n)), has_raw=True, has_thumbnail=False)

    try:
        with h5py.File(p, "r") as f:
            entry = f.get("entry")
            processed = (
                "entry/integrated_1d" in f
                or "entry/integrated_2d" in f
                or "entry/frames" in f
            )
            if not processed:
                # Raw detector dataset present? -> raw master.
                for cand in _RAW_DATASET_CANDIDATES:
                    obj = f.get(cand)
                    if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                        n = obj.shape[0] if obj.ndim >= 3 else 1
                        return ImageSourceInfo(
                            kind=ImageSourceKind.RAW_MASTER, path=str(p),
                            frame_labels=tuple(range(int(n))),
                            has_raw=True, has_thumbnail=False)
                if "entry/reduction" not in f:
                    return ImageSourceInfo(
                        kind=ImageSourceKind.UNKNOWN, path=str(p))
                # reduction group with no raw dataset -> treat as processed.

            labels = _frame_labels_from_groups(entry) if entry is not None else []

            has_raw = False
            has_thumbnail = False
            frames = entry.get("frames") if entry is not None else None
            if frames is not None:
                for name in frames:
                    fg = frames.get(name)
                    if fg is None:
                        continue
                    if not has_thumbnail and fg.get("thumbnail") is not None:
                        has_thumbnail = True
                    if not has_raw and _resolve_source_master(p, fg.get("source")) is not None:
                        has_raw = True
                    if has_raw and has_thumbnail:
                        break

            if has_raw:
                kind = ImageSourceKind.PROCESSED_XDART
            elif has_thumbnail:
                kind = ImageSourceKind.THUMBNAIL_ONLY
            else:
                # Processed file we can't draw from (no master, no thumbnail).
                kind = ImageSourceKind.PROCESSED_XDART
            return ImageSourceInfo(
                kind=kind, path=str(p), frame_labels=tuple(labels),
                has_raw=has_raw, has_thumbnail=has_thumbnail)
    except Exception:
        logger.debug("classify_image_source failed for %s", p, exc_info=True)
        return ImageSourceInfo(kind=ImageSourceKind.UNKNOWN, path=str(p))


def load_image_frame(path, frame) -> np.ndarray:
    """Load a genuine raw detector frame (master / tiff / eiger) by 0-based
    index via :func:`ssrl_xrd_tools.io.image.read_image`."""
    from ssrl_xrd_tools.io.image import read_image
    return np.asarray(read_image(Path(path), frame=int(frame)), dtype=float)


def _read_thumbnail_direct(path, frame) -> "np.ndarray | None":
    """Open the file and dequantize a stored thumbnail directly — a
    last-resort belt-and-suspenders if :func:`get_raw_frame` itself errors
    (not just a clean 'no master' fallthrough)."""
    import h5py
    from ssrl_xrd_tools.io.read import _dequantize_thumbnail
    try:
        with h5py.File(Path(path), "r") as f:
            key = f"entry/frames/frame_{int(frame):04d}/thumbnail"
            if key not in f:
                return None
            return np.asarray(_dequantize_thumbnail(f[key]), dtype=float)
    except Exception:
        logger.debug("_read_thumbnail_direct failed for frame %s of %s",
                     frame, path, exc_info=True)
        return None


def load_processed_raw_or_thumbnail(path, frame) -> RawFrameResult:
    """For a processed ``.nxs``: return the full-resolution raw image for a
    frame **label** if the per-frame source master resolves, else the
    dequantized thumbnail, else nothing — recording which in ``source``.

    Mirrors (and replaces) the GUI's strict-raw → thumbnail → direct-read
    fallback chain.
    """
    from ssrl_xrd_tools.io.read import get_raw_frame

    frame = int(frame)
    # Strict raw first (no thumbnail) so we know it's genuinely full-res.
    try:
        img = get_raw_frame(path, frame, allow_thumbnail=False)
        return RawFrameResult(image=np.asarray(img, dtype=float),
                              source="raw", frame=frame)
    except Exception:
        logger.debug("load_processed_raw_or_thumbnail: no raw master for "
                     "frame %s of %s; trying thumbnail", frame, path,
                     exc_info=True)
    # Raw unavailable -> dequantized thumbnail (get_raw_frame returns it when
    # the master is missing and allow_thumbnail is True).
    try:
        img = get_raw_frame(path, frame, allow_thumbnail=True)
        return RawFrameResult(image=np.asarray(img, dtype=float),
                              source="thumbnail", frame=frame)
    except Exception:
        logger.debug("load_processed_raw_or_thumbnail: get_raw_frame failed "
                     "for frame %s of %s; trying direct thumbnail read",
                     frame, path, exc_info=True)
    # Last resort: read the stored thumbnail directly (guards a get_raw_frame
    # that errors outright while a thumbnail is still present).
    thumb = _read_thumbnail_direct(path, frame)
    if thumb is not None:
        return RawFrameResult(image=thumb, source="thumbnail", frame=frame)
    return RawFrameResult(image=None, source="none", frame=frame)
