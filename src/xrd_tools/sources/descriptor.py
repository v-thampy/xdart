# -*- coding: utf-8 -*-
"""``ContainerDescriptor`` — small immutable NeXus/HDF5 container facts (R2).

A descriptor is everything about one finalized (or provisional) detector
container that can be resolved WITHOUT decoding a single detector pixel: its R1
candidate identity, probe/readiness state, source kind, canonical scan name,
resolved NXentry, detector dataset path(s), logical frame count / frame shape /
native dtype / dataset shape, native chunk shape + compression/filter summary,
whether it is one 2-D exposure or a stack, and its cheap wavelength +
finalized/recovery facts.

The descriptor is a plain frozen value: it holds NO open ``h5py`` object,
dataset view, fabio object, detector array, per-frame metadata table, Qt object,
sink, or reduction resource.  It is cheap to serialize/log (:meth:`to_dict`) and
safe to retain after the cursor that produced it has closed.

There is ONE handle-aware implementation, :func:`describe_container_from_open`,
which inspects an already-open container.  :func:`describe_container` is a thin
path convenience wrapper that opens a short-lived handle and delegates to it
(the R2 :class:`~xrd_tools.sources.cursor.ContainerCursor` reuses the same
handle-aware builder over its own sustained handle — no duplicated dataset or
readiness logic).

All dataset/entry resolution reuses the accepted ``xrd_tools.io`` handle-taking
helpers (``find_nexus_image_dataset_in_open_file``,
``_find_eiger_external_link_paths``, ``resolve_nxentry``, ``is_bluesky_nxwriter``,
the ``_read_energy``/``_read_wavelength`` resolvers, and the processed-file
classifier) so a descriptor agrees with R1 discovery/probe by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xrd_tools.core.scan import SourceKind
from xrd_tools.sources.probe import ProbeState

__all__ = [
    "ContainerDescriptor",
    "describe_container",
    "describe_container_from_open",
    "resolve_stack_paths",
]


@dataclass(frozen=True, slots=True)
class ContainerDescriptor:
    """Immutable, pixel-free facts about one NeXus/HDF5 detector container."""

    # -- R1 candidate identity ------------------------------------------------
    path: Path
    size: int | None = None
    mtime_ns: int | None = None
    adapter_id: str | None = None
    # -- probe / readiness ----------------------------------------------------
    state: ProbeState = ProbeState.READY
    reason: str = ""
    # -- source classification ------------------------------------------------
    kind: SourceKind = SourceKind.NEXUS_STACK
    scan_name: str = ""
    requested_entry: str = "entry"
    resolved_entry: str | None = None
    # -- detector dataset layout (no pixels decoded) --------------------------
    dataset_path: str | None = None
    #: Multiple internal paths for an external-link Eiger master (segments
    #: concatenated along the frame axis); empty for a single-dataset stack.
    segment_paths: tuple[str, ...] = ()
    frame_count: int = 0
    frame_shape: tuple[int, ...] | None = None
    dtype: np.dtype | None = None
    dataset_shape: tuple[int, ...] | None = None
    chunks: tuple[int, ...] | None = None
    compression: str | None = None
    is_2d: bool = False
    # -- cheap physics + lifecycle -------------------------------------------
    wavelength: float | None = None
    is_bluesky: bool = False
    finalized: bool = True

    @property
    def version_stamp(self) -> tuple[int, int] | None:
        """The R1 ``(size, mtime_ns)`` stamp, or ``None`` when unknown."""
        if self.size is None or self.mtime_ns is None:
            return None
        return (int(self.size), int(self.mtime_ns))

    @property
    def frame_bytes(self) -> int | None:
        """Native bytes of one frame, or ``None`` when layout is unknown."""
        if self.frame_shape is None or self.dtype is None:
            return None
        n = 1
        for dim in self.frame_shape:
            n *= int(dim)
        return n * int(self.dtype.itemsize)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly view (``numpy.dtype`` -> its string, ``Path`` ->
        ``str``) suitable for logging and metrics."""
        return {
            "path": str(self.path),
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "adapter_id": self.adapter_id,
            "state": self.state.value if isinstance(self.state, ProbeState) else str(self.state),
            "reason": self.reason,
            "kind": self.kind.value if isinstance(self.kind, SourceKind) else str(self.kind),
            "scan_name": self.scan_name,
            "requested_entry": self.requested_entry,
            "resolved_entry": self.resolved_entry,
            "dataset_path": self.dataset_path,
            "segment_paths": list(self.segment_paths),
            "frame_count": self.frame_count,
            "frame_shape": list(self.frame_shape) if self.frame_shape is not None else None,
            "dtype": None if self.dtype is None else str(self.dtype),
            "dataset_shape": list(self.dataset_shape) if self.dataset_shape is not None else None,
            "chunks": list(self.chunks) if self.chunks is not None else None,
            "compression": self.compression,
            "is_2d": self.is_2d,
            "wavelength": self.wavelength,
            "is_bluesky": self.is_bluesky,
            "finalized": self.finalized,
        }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _scan_name(path: Path) -> str:
    """Container full-stem naming (numeric suffix is scan identity; an Eiger
    ``_master`` tag is stripped).  Mirrors the R1 nexus adapter's
    ``_nexus_scan_name`` (kept in agreement by inspection)."""
    stem = Path(path).stem
    if stem.lower().endswith("_master"):
        return stem[: -len("_master")]
    return stem


def _filter_summary(dataset: Any) -> str | None:
    """A compact compression/filter summary for one dataset: HDF5 filter ids
    (e.g. ``1`` gzip, ``2`` shuffle, ``32004`` lz4, ``32008`` bitshuffle) joined
    by ``,``, or ``None`` when the dataset is stored uncompressed.

    Uses the create-property-list filter pipeline so external-link Eiger data
    filtered by a plugin (which reports ``dataset.compression is None``) is still
    summarized.  Best-effort: any failure falls back to ``dataset.compression``.
    """
    try:
        plist = dataset.id.get_create_plist()
        n = int(plist.get_nfilters())
        if n <= 0:
            return None
        ids = []
        for i in range(n):
            filt = plist.get_filter(i)
            ids.append(int(filt[0]))
        return ",".join(str(i) for i in ids) if ids else None
    except Exception:
        comp = getattr(dataset, "compression", None)
        return str(comp) if comp else None


def resolve_stack_paths(h5f: Any, entry: str) -> tuple[list[str], bool]:
    """Resolve the detector dataset internal path(s) from an OPEN container.

    Returns ``(paths, is_2d)``: an external-link Eiger master resolves to its
    ordered ``/{entry}/data/data_NNNNNN`` link paths (``is_2d`` False); otherwise
    the single detector dataset path from
    :func:`find_nexus_image_dataset_in_open_file`, with ``is_2d`` True when that
    dataset is a lone 2-D exposure.  ``([], False)`` when no detector dataset is
    present.  May raise ``ProcessedXdartInputError`` for a processed xdart file
    (the caller treats that as ``PROCESSED_OUTPUT``).
    """
    from xrd_tools.io.nexus import (
        _find_eiger_external_link_paths,
        find_nexus_image_dataset_in_open_file,
    )

    links = _find_eiger_external_link_paths(h5f, entry)
    if links:
        return list(links), False
    path = find_nexus_image_dataset_in_open_file(h5f, entry)
    if path is None:
        # The entry-scoped reader found nothing.  Fall back to R1's WHOLE-FILE
        # resolver (the one the authoritative _nexus_probe uses) so a detector
        # at a non-entry location (e.g. a root ``/data``) is still found and the
        # descriptor's readiness agrees with R1 by construction.  Use the
        # dataset's own path so the cursor's NexusImageStack can open it.
        # ProcessedXdartInputError propagates (processed output); a genuine
        # "no detector dataset" ValueError resolves to IMAGELESS.
        from xrd_tools.io.image import _find_hdf5_image_dataset
        from xrd_tools.io.nexus import UnsupportedDetectorRankError
        try:
            ds = _find_hdf5_image_dataset(h5f)
        except UnsupportedDetectorRankError:
            raise  # rank violation is INVALID, never IMAGELESS
        except ValueError:
            return [], False
        path = ds.name
    ds = h5f[path]
    return [path], bool(getattr(ds, "ndim", 0) == 2)


def _stack_facts(h5f: Any, paths: list[str]) -> dict[str, Any]:
    """Layout facts (dtype/chunks/compression/shape/frame_count) from the OPEN
    handle, reading only dataset METADATA — never a detector pixel."""
    first = h5f[paths[0]]
    dtype = np.dtype(first.dtype)
    chunks = tuple(int(c) for c in first.chunks) if first.chunks else None
    compression = _filter_summary(first)

    from xrd_tools.io.nexus import UnsupportedDetectorRankError

    if len(paths) == 1:
        ds = first
        if ds.ndim == 2:
            frame_shape = tuple(int(d) for d in ds.shape)
            return {
                "frame_count": 1, "frame_shape": frame_shape,
                "dataset_shape": frame_shape, "dtype": dtype, "chunks": chunks,
                "compression": compression, "is_2d": True,
            }
        if ds.ndim != 3:
            # NXS-DIM-1: never fabricate 3-D facts from an unsupported rank.
            raise UnsupportedDetectorRankError(
                f"{ds.name} has rank {ds.ndim}; detector data must be 2-D "
                f"(one frame) or 3-D (a stack)"
            )
        frame_shape = tuple(int(d) for d in ds.shape[1:])
        return {
            "frame_count": int(ds.shape[0]), "frame_shape": frame_shape,
            "dataset_shape": tuple(int(d) for d in ds.shape), "dtype": dtype,
            "chunks": chunks, "compression": compression, "is_2d": False,
        }

    # Multiple external-link Eiger segments: concatenate along the frame axis
    # (strict-3D, the same invariant NexusImageStack enforces).
    total = 0
    frame_shape: tuple[int, ...] | None = None
    for p in paths:
        d = h5f[p]
        if d.ndim != 3:
            raise UnsupportedDetectorRankError(
                f"{p} has rank {d.ndim}; Eiger segments must be 3-D"
            )
        total += int(d.shape[0])
        fs = tuple(int(x) for x in d.shape[1:])
        if frame_shape is None:
            frame_shape = fs
    return {
        "frame_count": total, "frame_shape": frame_shape,
        "dataset_shape": (total, *(frame_shape or ())), "dtype": dtype,
        "chunks": chunks, "compression": compression, "is_2d": False,
    }


def describe_container_from_open(
    h5f: Any,
    *,
    path: str | Path,
    entry: str = "entry",
    candidate: Any = None,
    size: int | None = None,
    mtime_ns: int | None = None,
    adapter_id: str | None = None,
) -> ContainerDescriptor:
    """Build a :class:`ContainerDescriptor` from an ALREADY-OPEN container.

    This is the single implementation; it never opens or closes a file and never
    decodes a detector pixel.  When *candidate* (an R1
    :class:`~xrd_tools.sources.discover.Candidate`) is given, its
    path/size/mtime/adapter identity is recorded on the descriptor.
    """
    from xrd_tools.io.bluesky_nexus import is_bluesky_nxwriter, resolve_nxentry
    from xrd_tools.io.image import _is_eiger_master
    from xrd_tools.io.nexus import (
        UnsupportedDetectorRankError,
        _read_energy,
        _read_wavelength,
    )
    from xrd_tools.io.processed_scan_id import (
        ProcessedXdartInputError,
        is_processed_xdart_file,
    )

    path = Path(path)
    if candidate is not None:
        size = candidate.size
        mtime_ns = candidate.mtime_ns
        adapter_id = candidate.adapter_id

    entry_grp = None
    try:
        entry_grp = resolve_nxentry(h5f, entry)
    except Exception:
        entry_grp = None
    resolved_entry = None
    if entry_grp is not None:
        resolved_entry = entry_grp.name.strip("/").split("/")[-1] or entry
    entry_name = resolved_entry or entry

    is_bluesky = False
    has_end = False
    if entry_grp is not None:
        try:
            is_bluesky = bool(is_bluesky_nxwriter(entry_grp))
        except Exception:
            is_bluesky = False
        try:
            has_end = "end_time" in entry_grp
        except Exception:
            has_end = False
    else:
        # No NXentry resolved yet.  A still-writing NXWriter run stamps the root
        # ``creator='NXWriter'`` attribute BEFORE its entry tree exists, so a
        # nascent shell must be detected at the FILE level — otherwise it would
        # read as finalized/IMAGELESS instead of IN_PROGRESS and a consumer
        # could prematurely consume-and-retire a still-writing run (agree with
        # R1 is_unfinalized_nxwriter).
        try:
            is_bluesky = bool(is_bluesky_nxwriter(h5f))
        except Exception:
            is_bluesky = False
        has_end = False  # no entry -> no end_time yet
    finalized = (not is_bluesky) or has_end
    scan_name = _scan_name(path)

    common: dict[str, Any] = dict(
        path=path, size=size, mtime_ns=mtime_ns, adapter_id=adapter_id,
        scan_name=scan_name, requested_entry=entry, resolved_entry=resolved_entry,
        is_bluesky=is_bluesky, finalized=finalized,
    )

    # Processed xdart output is a decisive terminal skip — never raw input.
    processed = False
    try:
        processed = bool(is_processed_xdart_file(h5f, entry_name))
    except Exception:
        processed = False
    if processed:
        return ContainerDescriptor(
            state=ProbeState.PROCESSED_OUTPUT, kind=SourceKind.PROCESSED_NEXUS,
            reason="processed xdart record (integrated_1d/2d or schema stamp)",
            **common)

    kind = SourceKind.EIGER_MASTER if _is_eiger_master(path) else SourceKind.NEXUS_STACK

    try:
        paths, _ = resolve_stack_paths(h5f, entry_name)
    except ProcessedXdartInputError:
        return ContainerDescriptor(
            state=ProbeState.PROCESSED_OUTPUT, kind=SourceKind.PROCESSED_NEXUS,
            reason="processed xdart record", **common)
    except UnsupportedDetectorRankError as exc:
        return ContainerDescriptor(
            state=ProbeState.INVALID, kind=kind,
            reason=f"unsupported detector rank: {exc}", **common)
    except (KeyError, OSError) as exc:
        return ContainerDescriptor(
            state=ProbeState.IN_PROGRESS,
            kind=kind,
            reason=("detector dataset link target is not yet available; "
                    f"retry later: {exc}"),
            **(common | {"finalized": False}),
        )

    # Eiger masters are commonly visible before their externally linked data
    # files land.  Resolving such a link raises KeyError from h5py even though
    # the master itself is valid and should remain eligible for a later retry.
    # Keep that normal acquisition state distinct from an imageless, finalized
    # container.  This is deliberately at the descriptor boundary; the older
    # low-level finder keeps its existing behavior.
    try:
        if not paths:
            facts = None
        else:
            facts = _stack_facts(h5f, paths)
    except UnsupportedDetectorRankError as exc:
        return ContainerDescriptor(
            state=ProbeState.INVALID, kind=kind,
            reason=f"unsupported detector rank: {exc}", **common)
    except (KeyError, OSError) as exc:
        return ContainerDescriptor(
            state=ProbeState.IN_PROGRESS,
            kind=kind,
            reason=("detector dataset link target is not yet available; "
                    f"retry later: {exc}"),
            **(common | {"finalized": False}),
        )

    if not paths:
        if is_bluesky and not has_end:
            state, reason = ProbeState.IN_PROGRESS, (
                "NXWriter run not yet finalized (no end_time)")
        else:
            state, reason = ProbeState.IMAGELESS, "no 2-D+ detector dataset found"
        return ContainerDescriptor(state=state, kind=kind, reason=reason, **common)

    assert facts is not None
    wavelength: float | None = None
    if entry_grp is not None:
        try:
            wl = _read_wavelength(entry_grp, _read_energy(entry_grp))
            wavelength = float(wl) if _finite(wl) else None
        except Exception:
            wavelength = None

    if is_bluesky and not has_end:
        state, reason = ProbeState.IN_PROGRESS, "NXWriter run not yet finalized (no end_time)"
    elif facts["frame_count"] <= 0:
        state, reason = ProbeState.IMAGELESS, "zero-frame detector dataset"
    else:
        state, reason = ProbeState.READY, "detector dataset present"

    return ContainerDescriptor(
        state=state, kind=kind, reason=reason,
        dataset_path=paths[0], segment_paths=tuple(paths) if len(paths) > 1 else (),
        frame_count=facts["frame_count"], frame_shape=facts["frame_shape"],
        dtype=facts["dtype"], dataset_shape=facts["dataset_shape"],
        chunks=facts["chunks"], compression=facts["compression"], is_2d=facts["is_2d"],
        wavelength=wavelength, **common)


def describe_container(
    path: str | Path,
    *,
    entry: str = "entry",
    candidate: Any = None,
) -> ContainerDescriptor:
    """Path convenience wrapper: open a short-lived handle and delegate to
    :func:`describe_container_from_open` (the single implementation).

    An unreadable / still-being-written container yields an ``IN_PROGRESS``
    descriptor (mirroring the R1 probe's "treat unreadable as still-writing"
    rule) rather than raising, so a caller polling a young file is not forced to
    special-case ``OSError``.  A genuinely missing path raises
    ``FileNotFoundError``.
    """
    import h5py

    path = Path(path)
    size = mtime_ns = None
    if candidate is not None:
        size, mtime_ns = candidate.size, candidate.mtime_ns
    else:
        try:
            st = path.stat()
            size, mtime_ns = st.st_size, st.st_mtime_ns
        except FileNotFoundError:
            raise
        except OSError:
            size = mtime_ns = None

    try:
        with h5py.File(path, "r") as h5f:
            return describe_container_from_open(
                h5f, path=path, entry=entry, candidate=candidate,
                size=size, mtime_ns=mtime_ns)
    except FileNotFoundError:
        raise
    except OSError as exc:
        return ContainerDescriptor(
            path=path, size=size, mtime_ns=mtime_ns,
            adapter_id=(candidate.adapter_id if candidate is not None else None),
            state=ProbeState.IN_PROGRESS,
            reason=f"unreadable HDF5 (treated as still-writing): {exc}",
            scan_name=_scan_name(path), requested_entry=entry, finalized=False)
