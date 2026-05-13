# -*- coding: utf-8 -*-
"""
EwaldArch storage indexed by frame id, backed by an xdart v2 NeXus file.

v2 schema (xdart 0.37+) is the only file shape we support:

* Integrated 1D/2D arrays live as **stacked** datasets under
  ``/entry/integrated_1d`` and ``/entry/integrated_2d``.
* Per-frame thumbnails (uncompressed uint8) live under
  ``/entry/frames/frame_NNNN/thumbnail``.
* Raw motor positioners live under ``/entry/{sample,instrument/detector}/
  positioners/``.

``ArchSeries.__setitem__`` is **in-memory only** — it appends the new
frame index to ``self.index`` but does no disk I/O.  Persistence happens
once per batch via :func:`xdart.modules.ewald.nexus_writer.save_sphere_to_nexus`.

``ArchSeries.__getitem__`` lazy-loads an :class:`EwaldArch` from the v2
stacked arrays + per-frame thumbnail group.
"""

from pandas import Series
import numpy as np

# xdart imports
from xdart.utils import catch_h5py_file as catch

# This module imports
from .arch import EwaldArch


def _ensure_frames_group(h5file):
    """Create ``entry/frames`` group hierarchy if it doesn't exist."""
    entry = h5file.require_group("entry")
    entry.attrs.setdefault("NX_class", "NXentry")
    frames = entry.require_group("frames")
    frames.attrs.setdefault("NX_class", "NXcollection")
    return frames


def _frame_position(h5file, idx: int) -> int | None:
    """Return the row of ``idx`` inside the stacked ``frame_index`` array.

    Returns ``None`` when the file has no integrated_1d group yet
    (i.e. the batch flush hasn't happened) or when idx isn't present.
    """
    if "entry/integrated_1d/frame_index" not in h5file:
        return None
    fi = np.asarray(h5file["entry/integrated_1d/frame_index"][()])
    where = np.where(fi == idx)[0]
    if where.size == 0:
        return None
    return int(where[0])


def _load_arch_v2(h5file, idx: int, *, static: bool, gi: bool) -> EwaldArch:
    """Build an :class:`EwaldArch` for ``idx`` from the v2 stacked arrays.

    Reads:

    * 1D: ``intensity_1d[i]``, ``sigma_1d[i]`` (if present), ``q``
    * 2D: ``intensity_2d[i]``, ``q`` (= ``q_2d``), ``chi``
    * thumbnail: ``frames/frame_NNNN/thumbnail`` (optional)

    Falls back to a minimal arch (just ``idx`` set) if any section is
    missing — callers should still get a usable object.
    """
    from ssrl_xrd_tools.core.containers import (
        IntegrationResult1D, IntegrationResult2D,
    )

    arch = EwaldArch(idx, static=static, gi=gi)

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
        arch.int_1d = IntegrationResult1D(
            radial=q, intensity=intensity, sigma=sigma, unit=unit,
        )

    # ── 2D ────────────────────────────────────────────────────────
    g2 = h5file.get("entry/integrated_2d") if pos is not None else None
    if g2 is not None and "intensity" in g2:
        # File layout: (frame, chi, q).  xdart arch convention: (nq, nchi).
        slab = np.asarray(g2["intensity"][pos], dtype=float)  # (chi, q)
        slab_xdart = slab.T  # (q, chi)
        q2 = np.asarray(g2["q"][()], dtype=float)
        chi = np.asarray(g2["chi"][()], dtype=float)
        chi_unit_attr = g2["chi"].attrs.get("units", b"") if "chi" in g2 else b""
        if isinstance(chi_unit_attr, bytes):
            chi_unit_attr = chi_unit_attr.decode("utf-8", errors="replace")
        arch.int_2d = IntegrationResult2D(
            radial=q2, azimuthal=chi, intensity=slab_xdart,
            sigma=None,
            unit=getattr(arch.int_1d, "unit", "q_A^-1") if arch.int_1d else "q_A^-1",
            azimuthal_unit=chi_unit_attr or "deg",
        )

    # ── per-frame thumbnail + source ref ──────────────────────────
    fg_key = f"entry/frames/frame_{idx:04d}"
    fg = h5file.get(fg_key)
    if fg is not None:
        if "thumbnail" in fg:
            try:
                arch.thumbnail = np.asarray(fg["thumbnail"][()])
            except Exception:
                pass
        _load_source_ref(arch, fg)

    # R3 guardrail: this arch was reconstructed from disk, not handed
    # over fresh from the wrangler.  ``map_raw`` is None and won't be
    # populated until a lazy raw loader is wired up — so the GUI
    # mustn't pretend re-integration is possible.
    arch.is_reload_only = True
    return arch


def _load_source_ref(arch: EwaldArch, fg) -> None:
    """Populate ``arch.source_file`` and ``arch.source_frame_idx`` from a
    per-frame :class:`NXcollection`.

    R2 schema lives under ``<frame_group>/source/{path, frame_index}``.
    A legacy ``source_ref`` dict (never actually written by the v2
    writer prior to R2 — the attribute-name mismatch silenced it) is
    also supported for forward-compat with any one-off files that
    might carry it.
    """
    src_grp = fg.get("source") if "source" in fg else None
    if src_grp is not None:
        try:
            path = src_grp["path"][()]
            if isinstance(path, bytes):
                path = path.decode("utf-8", errors="replace")
            arch.source_file = str(path)
        except Exception:
            pass
        try:
            arch.source_frame_idx = int(src_grp["frame_index"][()])
        except Exception:
            pass
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
            arch.source_file = str(v)
        except Exception:
            pass
    if "frame_index" in legacy:
        try:
            arch.source_frame_idx = int(legacy["frame_index"][()])
        except Exception:
            pass


class ArchSeries:
    """Index-keyed container for :class:`EwaldArch` objects.

    See module docstring for the storage contract.
    """

    def __init__(self, data_file, file_lock, arches=None,
                 static=False, gi=False, h5file=None):
        if arches is None:
            arches = []
        self.data_file = data_file
        self.file_lock = file_lock
        self.index: list[int] = []
        self.static = static
        self.gi = gi
        # Hot-cache of fully-populated EwaldArch objects.  Used by the
        # writer to access freshly-integrated arches before they hit
        # disk, and by viewer code to avoid re-loading recently-touched
        # arches.  Bounded so long scans don't blow memory; see
        # ``_in_memory_cap``.
        self._in_memory: dict[int, EwaldArch] = {}
        self._in_memory_cap = 64
        if arches:
            for a in arches:
                self.__setitem__(a.idx, a, h5file=h5file)
        self._i = 0

    def stash(self, arch):
        """Keep ``arch`` in memory so the writer can read it next flush.

        Called from :meth:`EwaldSphere.add_arch` after integrating a
        fresh frame, so the v2 NeXus writer can pull
        ``int_1d``/``int_2d``/``thumbnail`` straight off the live object
        instead of re-loading from disk (which would fail for the
        first-ever frame, before any stacked dataset has been written).

        Older entries beyond ``_in_memory_cap`` are evicted in FIFO
        order to keep memory bounded on long scans.
        """
        self._in_memory[arch.idx] = arch
        if len(self._in_memory) > self._in_memory_cap:
            # FIFO eviction — drop the oldest key
            oldest = next(iter(self._in_memory))
            self._in_memory.pop(oldest, None)

    def __getitem__(self, idx):
        """Return EwaldArch for ``idx``: in-memory hit, else lazy-load."""
        if idx not in self.index:
            raise KeyError(f"Arch not found with {idx} index")
        if idx in self._in_memory:
            return self._in_memory[idx]
        with self.file_lock:
            with catch(self.data_file, 'r') as f:
                return _load_arch_v2(f, idx, static=self.static, gi=self.gi)

    def iloc(self, idx):
        """Location-based retrieval of arches (returns by position in index)."""
        return self.__getitem__(self.index[idx])

    def __setitem__(self, idx, arch, h5file=None, global_mask=None):
        """In-memory append + stash.  No disk I/O.

        Persistence is the v2 writer's job
        (:func:`xdart.modules.ewald.nexus_writer.save_sphere_to_nexus`);
        this method just keeps the index ordered and the live arch
        cached so the writer can find its integration results.
        """
        if idx != arch.idx:
            arch.idx = idx
        if arch.idx not in self.index:
            self.index.append(arch.idx)
        self.stash(arch)

    def append(self, arch, h5file=None, global_mask=None):
        """Add a new arch (or extract from a pandas Series) to the index."""
        arches = ArchSeries(self.data_file, self.file_lock,
                            static=self.static, gi=self.gi)
        arches.index = self.index[:]
        # Preserve any in-memory cache on the new ArchSeries — losing it
        # would force the v2 writer to re-load every arch from disk.
        arches._in_memory = dict(self._in_memory)
        arches._in_memory_cap = self._in_memory_cap
        if isinstance(arch, Series):
            _arch = arch.iloc[0]
        else:
            _arch = arch
        arches.__setitem__(_arch.idx, _arch, h5file=h5file,
                           global_mask=global_mask)
        return arches

    def sort_index(self, inplace=False):
        """Sort the index in place or return a sorted copy."""
        if inplace:
            self.index.sort()
            return None
        arches = ArchSeries(self.data_file, self.file_lock,
                            static=self.static, gi=self.gi)
        arches.index = sorted(self.index)
        arches._in_memory = dict(self._in_memory)
        arches._in_memory_cap = self._in_memory_cap
        return arches

    def __next__(self):
        if self._i < len(self.index):
            arch = self.iloc(self._i)
            self._i += 1
            return arch
        raise StopIteration

    def __iter__(self):
        self._i = 0
        return self
