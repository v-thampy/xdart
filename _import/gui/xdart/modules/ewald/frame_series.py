# -*- coding: utf-8 -*-
"""
LiveFrame storage indexed by frame id, backed by an xdart v2 NeXus file.

v2 schema (xdart 0.37+) is the only file shape we support:

* Integrated 1D/2D arrays live as **stacked** datasets under
  ``/entry/integrated_1d`` and ``/entry/integrated_2d``.
* Per-frame thumbnails (uncompressed uint8) live under
  ``/entry/frames/frame_NNNN/thumbnail``.
* Raw motor positioners live under ``/entry/{sample,instrument/detector}/
  positioners/``.

``LiveFrameSeries.__setitem__`` is **in-memory only** — it appends the new
frame index to ``self.index`` but does no disk I/O.  Persistence happens
once per batch via :func:`xdart.modules.ewald.nexus_writer.save_scan_to_nexus`.

``LiveFrameSeries.__getitem__`` lazy-loads a :class:`LiveFrame` from the v2
stacked arrays + per-frame thumbnail group.
"""

import logging
import os

from pandas import Series
import numpy as np

logger = logging.getLogger(__name__)


class _IndexedList(list):
    """A ``list`` with O(1) ``in`` membership via a parallel set.

    Drop-in replacement: every reading operation (iteration, slicing,
    indexing, ``len``, ``sort``) is inherited from ``list`` unchanged,
    so any code that already does ``for i in series.index``,
    ``series.index[0]``, ``series.index[-1]``, etc. keeps working.
    Every *mutating* operation maintains a parallel ``set`` so the
    common hot-path check ``idx in series.frames.index`` runs in
    O(1) instead of O(N) — important during live mode where it's
    called for every incoming frame.

    Caveat: slicing returns a plain ``list``, not an ``_IndexedList``
    (consistent with ``list`` semantics).  Callers that want to keep
    the indexed behavior on the copy should rewrap explicitly
    (``_IndexedList(some_slice)``).
    """

    __slots__ = ("_set",)

    def __init__(self, items=()):
        super().__init__(items)
        self._set: set = set(self)

    def __contains__(self, x) -> bool:
        return x in self._set

    def append(self, x) -> None:
        super().append(x)
        self._set.add(x)

    def extend(self, xs) -> None:
        for x in xs:
            super().append(x)
            self._set.add(x)

    def insert(self, i, x) -> None:
        super().insert(i, x)
        self._set.add(x)

    def remove(self, x) -> None:
        super().remove(x)
        if super().count(x) == 0:
            self._set.discard(x)

    def pop(self, i=-1):
        x = super().pop(i)
        if super().count(x) == 0:
            self._set.discard(x)
        return x

    def clear(self) -> None:
        super().clear()
        self._set.clear()

    def __setitem__(self, k, v):
        if isinstance(k, slice):
            super().__setitem__(k, v)
            self._set = set(self)
        else:
            old = super().__getitem__(k)
            super().__setitem__(k, v)
            if super().count(old) == 0:
                self._set.discard(old)
            self._set.add(v)

    def __delitem__(self, k):
        if isinstance(k, slice):
            super().__delitem__(k)
            self._set = set(self)
        else:
            old = super().__getitem__(k)
            super().__delitem__(k)
            if super().count(old) == 0:
                self._set.discard(old)

# xdart imports
from xdart.utils import catch_h5py_file as catch

# This module imports
from .frame import LiveFrame


def _ensure_frames_group(h5file):
    """Create ``entry/frames`` group hierarchy if it doesn't exist."""
    entry = h5file.require_group("entry")
    entry.attrs.setdefault("NX_class", "NXentry")
    frames = entry.require_group("frames")
    frames.attrs.setdefault("NX_class", "NXcollection")
    return frames


# Cache of the label→row map for the integrated_1d frame_index, keyed by
# id(h5file).  Each entry is ``(fingerprint, {label: row})`` where the
# fingerprint is ``(length, first_label, last_label)`` — read O(1) from the
# dataset on every call.  We rebuild the full map only when the fingerprint
# changes, which catches BOTH growth (append) and a same-length relabel
# (reintegration that reorders/renumbers frames).  The fingerprint guards
# against id() recycling too: a recycled handle pointing at a different file
# will almost always have a different fingerprint, and if it doesn't the map
# is still valid for that content.  Without this cache each lookup scanned
# the whole frame_index, making a K-frame multi-select O(K·N).
_FRAME_POS_CACHE: dict[int, tuple[tuple, dict[int, int]]] = {}


def _frame_position(h5file, idx: int) -> int | None:
    """Return the row of ``idx`` inside the stacked ``frame_index`` array.

    Returns ``None`` when the file has no integrated_1d group yet
    (i.e. the batch flush hasn't happened) or when idx isn't present.

    Uses a fingerprint-validated label→row cache so repeated lookups
    against the same open file are O(1) instead of an O(N) array scan each.
    """
    if "entry/integrated_1d/frame_index" not in h5file:
        return None
    ds = h5file["entry/integrated_1d/frame_index"]
    n = int(ds.shape[0])
    if n == 0:
        return None
    # Cheap fingerprint: length + endpoints.  Catches append (length) and
    # relabel (endpoints) without reading the whole array.
    fp = (n, int(ds[0]), int(ds[n - 1]))
    key = id(h5file)
    cached = _FRAME_POS_CACHE.get(key)
    if cached is None or cached[0] != fp:
        if len(_FRAME_POS_CACHE) > 64:
            _FRAME_POS_CACHE.clear()
        fi = np.asarray(ds[()])
        lookup = {int(lbl): row for row, lbl in enumerate(fi)}
        _FRAME_POS_CACHE[key] = (fp, lookup)
    else:
        lookup = cached[1]
    return lookup.get(int(idx))


def _load_frame_v2(h5file, idx: int, *, static: bool, gi: bool,
                  source_root: str | None = None) -> LiveFrame:
    """Build a :class:`LiveFrame` for ``idx`` from the v2 stacked arrays.

    Reads:

    * 1D: ``intensity_1d[i]``, ``sigma_1d[i]`` (if present), ``q``
    * 2D: ``intensity_2d[i]``, ``q`` (= ``q_2d``), ``chi``
    * thumbnail: ``frames/frame_NNNN/thumbnail`` (optional)

    Falls back to a minimal frame (just ``idx`` set) if any section is
    missing — callers should still get a usable object.
    """
    from ssrl_xrd_tools.core.containers import (
        IntegrationResult1D, IntegrationResult2D,
    )

    frame = LiveFrame(idx, static=static, gi=gi)

    pos = _frame_position(h5file, idx)

    # ── 1D ────────────────────────────────────────────────────────
    g1 = h5file.get("entry/integrated_1d") if pos is not None else None
    if g1 is not None and "intensity" in g1:
        q = np.asarray(g1["q"][()], dtype=float)
        intensity = np.asarray(g1["intensity"][pos], dtype=float)
        sigma = (
            np.asarray(g1["sigma"][pos], dtype=float)
            if "sigma" in g1 else None
        )
        unit_attr = g1["q"].attrs.get("units", b"") if "q" in g1 else b""
        if isinstance(unit_attr, bytes):
            unit_attr = unit_attr.decode("utf-8", errors="replace")
        unit = (
            "q_A^-1" if "angstrom" in (unit_attr or "")
            else "q_nm^-1" if "nm" in (unit_attr or "")
            else (unit_attr or "q_A^-1")
        )
        frame.int_1d = IntegrationResult1D(
            radial=q, intensity=intensity, sigma=sigma, unit=unit,
        )

    # ── 2D ────────────────────────────────────────────────────────
    g2 = h5file.get("entry/integrated_2d") if pos is not None else None
    if g2 is not None and "intensity" in g2:
        # File layout: (frame, chi, q).  xdart frame convention: (nq, nchi).
        slab = np.asarray(g2["intensity"][pos], dtype=float)  # (chi, q)
        slab_xdart = slab.T  # (q, chi)
        q2 = np.asarray(g2["q"][()], dtype=float)
        chi = np.asarray(g2["chi"][()], dtype=float)
        chi_unit_attr = g2["chi"].attrs.get("units", b"") if "chi" in g2 else b""
        if isinstance(chi_unit_attr, bytes):
            chi_unit_attr = chi_unit_attr.decode("utf-8", errors="replace")
        frame.int_2d = IntegrationResult2D(
            radial=q2, azimuthal=chi, intensity=slab_xdart,
            sigma=None,
            unit=getattr(frame.int_1d, "unit", "q_A^-1") if frame.int_1d else "q_A^-1",
            azimuthal_unit=chi_unit_attr or "deg",
        )

    # ── per-frame thumbnail + source ref ──────────────────────────
    fg_key = f"entry/frames/frame_{idx:04d}"
    fg = h5file.get(fg_key)
    if fg is not None:
        if "thumbnail" in fg:
            try:
                frame.thumbnail = np.asarray(fg["thumbnail"][()])
            except (KeyError, ValueError, TypeError, OSError) as e:
                # Thumbnail read errors are non-fatal — the displayframe
                # can fall back to map_raw if it's around, and the rest
                # of the frame state is still valid.
                logger.debug("thumbnail read failed for frame %d: %s",
                             idx, e)
        _load_source_ref(frame, fg)

    # L1 lazy raw load setup + R3 guardrail.
    # Stash the source-root for ``LiveFrame._lazy_load_raw`` to
    # resolve relative paths against.  Then decide whether
    # re-integration is feasible by checking that the source file
    # exists on disk — if it does, lazy load can recover map_raw;
    # if it doesn't, the GUI guardrail should still fire.
    if source_root:
        frame._source_root = source_root
    if frame.source_file and frame._lazy_load_resolvable():
        frame.is_reload_only = False
    else:
        frame.is_reload_only = True
    return frame


def _load_source_ref(frame: LiveFrame, fg) -> None:
    """Populate ``frame.source_file`` and ``frame.source_frame_idx`` from a
    per-frame :class:`NXcollection`.

    R2 schema lives under ``<frame_group>/source/{path, frame_index}``.
    A legacy ``source_ref`` dict (never actually written by the v2
    writer prior to R2 — the attribute-name mismatch silenced it) is
    also supported for forward-compat with any one-off files that
    might carry it.
    """
    # Narrow except set on every read: h5py raises KeyError on missing
    # fields, ValueError/OSError on corrupt data, TypeError on weird
    # dtypes.  Losing a source ref is non-fatal (the GUI guardrail
    # falls back to reload-only mode) — but a *silent* loss is exactly
    # the bug R2 fixed, so log at debug level so it's at least visible
    # under XDART_LOG_LEVEL=DEBUG.
    _SRC_READ_ERRORS = (KeyError, ValueError, TypeError, OSError)

    src_grp = fg.get("source") if "source" in fg else None
    if src_grp is not None:
        try:
            path = src_grp["path"][()]
            if isinstance(path, bytes):
                path = path.decode("utf-8", errors="replace")
            frame.source_file = str(path)
        except _SRC_READ_ERRORS as e:
            logger.debug("source/path read failed for frame %s: %s",
                         frame.idx, e)
        try:
            frame.source_frame_idx = int(src_grp["frame_index"][()])
        except _SRC_READ_ERRORS as e:
            logger.debug("source/frame_index read failed for frame %s: %s",
                         frame.idx, e)
        return

    # Legacy support: dict-shaped source_ref subgroup.
    legacy = fg.get("source_ref") if "source_ref" in fg else None
    if legacy is None:
        return
    path = None
    if "path" in legacy:
        path = legacy["path"]
    elif "file" in legacy:
        path = legacy["file"]
    if path is not None:
        try:
            v = path[()]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            frame.source_file = str(v)
        except _SRC_READ_ERRORS as e:
            logger.debug("legacy source_ref path read failed for frame %s: %s",
                         frame.idx, e)
    if "frame_index" in legacy:
        try:
            frame.source_frame_idx = int(legacy["frame_index"][()])
        except _SRC_READ_ERRORS as e:
            logger.debug("legacy source_ref frame_index read failed for frame %s: %s",
                         frame.idx, e)


class LiveFrameSeries:
    """Index-keyed container for :class:`LiveFrame` objects.

    See module docstring for the storage contract.
    """

    def __init__(self, data_file, file_lock, frames=None,
                 static=False, gi=False, h5file=None):
        if frames is None:
            frames = []
        self.data_file = data_file
        self.file_lock = file_lock
        # O(1) ``in`` membership in hot paths (live wrangler, GUI).
        # Behaves like ``list[int]`` for everything else.
        self.index: _IndexedList = _IndexedList()
        self.static = static
        self.gi = gi
        # Hot-cache of fully-populated LiveFrame objects.  Used by the
        # writer to access freshly-integrated frames before they hit
        # disk, and by viewer code to avoid re-loading recently-touched
        # frames.  Bounded so long scans don't blow memory; see
        # ``_in_memory_cap``.
        self._in_memory: dict[int, LiveFrame] = {}
        self._in_memory_cap = 64
        if frames:
            for a in frames:
                self.__setitem__(a.idx, a, h5file=h5file)
        self._i = 0

    def stash(self, frame):
        """Keep ``frame`` in memory so the writer can read it next flush.

        Called from :meth:`LiveScan.add_frame` after integrating a
        fresh frame, so the v2 NeXus writer can pull
        ``int_1d``/``int_2d``/``thumbnail`` straight off the live object
        instead of re-loading from disk (which would fail for the
        first-ever frame, before any stacked dataset has been written).

        Older entries beyond ``_in_memory_cap`` are evicted in FIFO
        order to keep memory bounded on long scans.
        """
        self._in_memory[frame.idx] = frame
        if len(self._in_memory) > self._in_memory_cap:
            # FIFO eviction — drop the oldest key
            oldest = next(iter(self._in_memory))
            self._in_memory.pop(oldest, None)

    def __getitem__(self, idx):
        """Return LiveFrame for ``idx``: in-memory hit, else lazy-load."""
        if idx not in self.index:
            raise KeyError(f"Frame not found with {idx} index")
        if idx in self._in_memory:
            return self._in_memory[idx]
        # Resolve the source-root (scan data_file directory) once
        # per load so reloaded frames can lazy-load raw frames via
        # ``frame._source_root``-relative source_file paths.
        source_root = (
            os.path.dirname(self.data_file) if self.data_file else None
        )
        with self.file_lock:
            with catch(self.data_file, 'r') as f:
                return _load_frame_v2(f, idx, static=self.static, gi=self.gi,
                                     source_root=source_root)

    def iloc(self, idx):
        """Location-based retrieval of frames (returns by position in index)."""
        return self.__getitem__(self.index[idx])

    def __setitem__(self, idx, frame, h5file=None, global_mask=None):
        """In-memory append + stash.  No disk I/O.

        Persistence is the v2 writer's job
        (:func:`xdart.modules.ewald.nexus_writer.save_scan_to_nexus`);
        this method just keeps the index ordered and the live frame
        cached so the writer can find its integration results.
        """
        if idx != frame.idx:
            frame.idx = idx
        if frame.idx not in self.index:
            self.index.append(frame.idx)
        self.stash(frame)

    def append(self, frame, h5file=None, global_mask=None):
        """Add a new frame (or extract from a pandas Series) to the index."""
        frames = LiveFrameSeries(self.data_file, self.file_lock,
                                 static=self.static, gi=self.gi)
        # Preserve _IndexedList semantics (list[:] would degrade it).
        frames.index = _IndexedList(self.index)
        # Preserve any in-memory cache on the new LiveFrameSeries — losing it
        # would force the v2 writer to re-load every frame from disk.
        frames._in_memory = dict(self._in_memory)
        frames._in_memory_cap = self._in_memory_cap
        if isinstance(frame, Series):
            _frame = frame.iloc[0]
        else:
            _frame = frame
        frames.__setitem__(_frame.idx, _frame, h5file=h5file,
                           global_mask=global_mask)
        return frames

    def sort_index(self, inplace=False):
        """Sort the index in place or return a sorted copy."""
        if inplace:
            self.index.sort()
            return None
        frames = LiveFrameSeries(self.data_file, self.file_lock,
                                 static=self.static, gi=self.gi)
        frames.index = _IndexedList(sorted(self.index))
        frames._in_memory = dict(self._in_memory)
        frames._in_memory_cap = self._in_memory_cap
        return frames

    def __next__(self):
        if self._i < len(self.index):
            frame = self.iloc(self._i)
            self._i += 1
            return frame
        raise StopIteration

    def __iter__(self):
        self._i = 0
        return self


__all__ = ["LiveFrameSeries"]
