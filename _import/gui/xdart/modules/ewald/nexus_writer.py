"""xdart v2 NeXus writer (xdart 0.37+ schema).

This module produces files conforming to the layout described in
``xdart/docs/nexus_stitch_refactor_plan.md`` §2.  The single public
entry point is :func:`save_scan_to_nexus`, called from
:meth:`LiveScan._save_to_nexus`.

**Keep-xdart-thin (#18):** the on-disk layout for the stacked
``integrated_1d``/``integrated_2d`` groups, the ``stitched_*`` groups,
the motor ``positioners``, and ``per_frame_geometry`` is owned by the
shared, headless-reusable primitives in
:mod:`ssrl_xrd_tools.io.nexus` (``write_integrated_stack``,
``write_stitched``, ``write_positioners``, ``write_per_frame_geometry``).
This module is now a thin GUI-side adapter: it gathers the LiveScan's
in-memory state (frames, scan_data, geometry, PONI, thumbnails), decides
*which* frames to hand the stacked-write primitive (the O(K) append
cursor + the "rewrite from all frames on a shape change" guard live
here), and keeps the things that are genuinely xdart-specific —
NFS-retry file open, NXprocess provenance, per-frame thumbnails, the
detector/source instrument stamp.

Key invariants of the v2 schema:

1. ``/entry/integrated_1d`` and ``/entry/integrated_2d`` are **stacked**
   datasets shape ``(N, nq)`` and ``(N, nchi, nq)`` respectively — never
   per-frame NXdata groups.
2. ``/entry/frames/frame_NNNN/`` carries *only* per-frame non-array
   metadata + thumbnail.  No duplicated integrated arrays.
3. Thumbnails are **uncompressed uint8 or uint16**, not gzip-compressed.
4. Raw motor positioners live verbatim under
   ``/entry/instrument/detector/positioners/`` and
   ``/entry/sample/positioners/``.
5. Derived pyFAI rotations + GI incidence angle live in
   ``/entry/per_frame_geometry/``, recomputed from the raw positioners
   via the :class:`DiffractometerGeometry` config blob stored in
   ``/entry/reduction/config/geometry/``.
6. Provenance (NXprocess) is written via
   :func:`ssrl_xrd_tools.core.provenance.write_provenance` — versions
   are pulled from ``importlib.metadata`` and never hard-coded.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import h5py
import nexusformat.nexus as nx
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ssrl_xrd_tools.core.geometry import DiffractometerGeometry

    from xdart.modules.live import LiveScan


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File-opening helper — mirrors ssrl_xrd_tools.core.hdf5.catch_h5py_file
# semantics (NFS retry on transient OSError) but goes through
# ``nx.nxopen`` so the returned object is an ``NXroot`` view rather
# than a raw h5py.File.  Underlying h5py.File still reachable via
# ``root.nxfile.file`` for the (h5py-based) primitives we delegate to.
# ---------------------------------------------------------------------------

def _open_with_retry(path: Union[str, "Path"], mode: str,
                     tries: int = 100, sleep_s: float = 0.05):
    """Open a NeXus file via ``nx.nxopen``, retrying transient OSErrors.

    Beamline NFS mounts sometimes briefly refuse to open a file while
    another process is still releasing its lock.  Retry the same way
    ``ssrl_xrd_tools.core.hdf5.catch_h5py_file`` does so the writer
    behaves identically to the previous code path.

    ``nx.nxopen`` accepts the same mode strings as ``h5py.File``
    (``'r'``, ``'rw'``, ``'r+'``, ``'w'``, ``'w-'``, ``'a'``).
    """
    last_exc: Exception | None = None
    for _ in range(tries):
        try:
            return nx.nxopen(os.fspath(path), mode)
        except OSError as exc:
            last_exc = exc
            time.sleep(sleep_s)
    # Final attempt — let it propagate naturally if it still fails
    if last_exc is not None:
        return nx.nxopen(os.fspath(path), mode)
    raise RuntimeError("unreachable")


def _h5(f) -> h5py.File:
    """Reach the underlying ``h5py.File`` from an ``NXroot`` returned by
    :func:`_open_with_retry`.

    The stacked-write / positioner / geometry / stitched primitives in
    ``ssrl_xrd_tools.io.nexus`` operate on a raw :class:`h5py.Group`, so
    every section that delegates to them grabs the live h5py file here
    and passes ``h5f.require_group(entry)``.  Only ``_ensure_nxentry``
    and the per-frame thumbnail writer still go through nexusformat.
    """
    return f.nxfile.file


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def save_scan_to_nexus(
    scan: "LiveScan",
    path: Union[str, "Path"],
    *,
    mode: str = "a",
    entry: str = "entry",
    finalize: bool = False,
    replace_frame_indices=None,
) -> None:
    """Write ``scan``'s state into the file at ``path`` as a v2 NXroot.

    Two write modes:

    * **Append (default, ``replace_frame_indices=None``)** —
      acquisition flow.  Stacked integrated_1d/2d datasets are
      extended with the frames added since the last save; per-frame
      metadata groups are append-only; the reduction group is written
      once (or on finalize).

    * **Replace** — ``replace_frame_indices`` is an iterable of
      frame indices whose recomputed ``int_1d`` / ``int_2d`` should
      overwrite their existing rows in place.  Used by GUI
      reintegration (``scan_threads.bai_1d_all``).  In this mode the
      per-frame metadata + positioners + geometry are left alone (they
      don't change on reintegration), but the reduction group is
      re-written so the persisted ``bai_*_args`` reflect the new run.

    Parameters
    ----------
    scan
        :class:`LiveScan` carrying the in-memory state.  Must expose
        ``frames`` (ordered), ``scan_data`` (pandas DataFrame),
        ``bai_1d_args``, ``bai_2d_args``, optionally ``geometry``
        (:class:`DiffractometerGeometry`) and ``incidence_motor``.
    path
        Filesystem path to the ``.nxs`` file.  The writer opens and
        closes its own file handle (NFS-retry semantics included), so
        callers should NOT hold an h5py.File on the same path during
        this call.
    mode
        HDF5 open mode (default ``"a"`` — open existing or create).
        Pass ``"w"`` to truncate.
    entry
        NXentry group name (default ``"entry"``).
    finalize
        If ``True``, this is the last write of the scan — additional
        write-once items (PONI, stitched outputs) are flushed.  Safe to
        call with ``finalize=False`` repeatedly during a scan.
    replace_frame_indices
        See "Replace" mode above.  ``None`` (default) for append mode.
    """
    _logger = logging.getLogger(__name__)
    _verbose = _logger.isEnabledFor(logging.DEBUG)

    def _tick(label, t0):
        if _verbose:
            _logger.debug("save_scan_to_nexus[%s]: %.3fs",
                          label, time.time() - t0)
        return time.time()

    is_replace = replace_frame_indices is not None

    _t_total = time.time()
    _t0 = time.time()
    with _open_with_retry(path, mode) as f:
        _t0 = _tick("open", _t0)
        _ensure_nxentry(f, entry)
        _t0 = _tick("entry", _t0)
        h5f = _h5(f)

        # 1. Provenance — append mode: only on first save or finalize.
        # Replace mode: always rewrite so the persisted ``bai_*_args``
        # reflect whatever parameters the reintegration used (this is
        # the whole reason the user kicked off a re-integration in the
        # first place).
        if is_replace or finalize or "reduction" not in h5f.get(entry, {}):
            _write_reduction(h5f, scan, entry=entry)
        _t0 = _tick("reduction", _t0)

        # 2. Stacked integrated_1d and integrated_2d (delegated to
        #    ssrl_xrd_tools.io.nexus.write_integrated_stack).
        _write_integrated_1d(f, scan, entry=entry,
                             replace_frame_indices=replace_frame_indices)
        _t0 = _tick("integrated_1d", _t0)
        _write_integrated_2d(f, scan, entry=entry,
                             replace_frame_indices=replace_frame_indices)
        _t0 = _tick("integrated_2d", _t0)

        # 3-6: per-frame metadata, positioners, derived geometry and
        # instrument are *write-once* values (raw motor positions
        # don't change on reintegration; neither do PONI, thumbnail,
        # mask).  Skip in replace mode for a faster save.
        if not is_replace:
            # Per-frame metadata already has its own cursor (R4) —
            # cheap on every save.
            _write_per_frame_metadata(f, scan, entry=entry)
            _t0 = _tick("per_frame_metadata", _t0)

            # H1: positioners and per_frame_geometry rebuild full-scan
            # arrays on every call (the ssrl primitives reindex the
            # whole scan_data to the frame set), so gate them on
            # first-save or finalize.  Intermediate periodic saves don't
            # need them — live viewers index by frame_index from the
            # stacked integrated_* datasets, and the scan motor columns
            # are still inspectable via the source NeXus / SPEC file.
            instr_path = f"{entry}/instrument"
            first_instr = instr_path not in h5f

            # Per-frame metadata tables (scan_data, positioners,
            # per_frame_geometry) must stay the SAME length as the stacked
            # integrated_* rows.  Live mode never passes finalize=True and
            # saves per-frame, so a once-only "first save" gate froze these
            # at their first length while integrated rows kept growing — a
            # reloaded file then has e.g. 5 integrated frames but 2 metadata
            # rows (read_scan drops the short columns).  Rewrite each
            # whenever it's stale (on-disk length != current frame count).
            # They're small metadata arrays (not images), so the full-table
            # rewrite is cheap even per-frame.
            n_frames = len(scan.frames.index)
            sd_stale = _per_frame_len(h5f, f"{entry}/scan_data/frame_index") != n_frames
            geom_stale = (
                _per_frame_len(h5f, f"{entry}/per_frame_geometry/frame_index")
                != n_frames
            )
            pos_stale = _positioners_len(h5f, entry) != n_frames

            if finalize or sd_stale:
                _write_scan_metadata(f, scan, entry=entry)
                _t0 = _tick("scan_metadata", _t0)
            if finalize or pos_stale:
                _write_positioners(f, scan, entry=entry)
                _t0 = _tick("positioners", _t0)
            if finalize or geom_stale:
                _write_per_frame_geometry(f, scan, entry=entry)
                _t0 = _tick("per_frame_geometry", _t0)
            if finalize or first_instr:
                _write_instrument(f, scan, entry=entry)
                _t0 = _tick("instrument", _t0)

        # 7. Stitched outputs (if present on the scan) — finalize only.
        if finalize:
            _write_stitched(f, scan, entry=entry)
            _t0 = _tick("stitched", _t0)

    if _verbose:
        _logger.debug("save_scan_to_nexus[close+TOTAL]: %.3fs",
                      time.time() - _t_total)


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def _ensure_nxentry(f, entry: str) -> None:
    """Ensure ``/<entry>`` exists as an :class:`NXentry`.

    Uses nx assignment so the resulting group lives in nexusformat's
    in-memory tree and can be navigated (e.g. ``f[entry]``) by the
    per-frame thumbnail writer, which still uses nexusformat.
    """
    if entry not in f:
        f[entry] = nx.NXentry()
    # Ensure NX_class is correctly set (idempotent on rewrites).
    f[entry].attrs["NX_class"] = "NXentry"
    if "default" not in f[entry].attrs:
        f[entry].attrs["default"] = "integrated_1d"


def _write_reduction(h5f, scan, *, entry: str) -> None:
    """Write /entry/reduction/ via ssrl_xrd_tools provenance."""
    from ssrl_xrd_tools.core.provenance import write_provenance

    config: dict[str, Any] = {
        "bai_1d_args": dict(scan.bai_1d_args),
        "bai_2d_args": dict(scan.bai_2d_args),
    }
    if hasattr(scan, "gi_config") and scan.gi_config:
        config["gi_config"] = dict(scan.gi_config)

    # Geometry: stored as a structured subgroup (handled specially in
    # write_provenance), so the convention is human-inspectable in HDF5.
    geom = getattr(scan, "geometry", None)
    if geom is not None:
        config["geometry"] = {
            "convention": geom.convention,
            "mapping_json": geom.to_json(),
            "motor_sources": {
                m: m for m in geom.all_referenced_motors()
            },
        }

    inputs: dict[str, Any] = {}
    if hasattr(scan, "raw_files") and scan.raw_files:
        inputs["raw_files"] = list(scan.raw_files)
    if hasattr(scan, "meta_file") and scan.meta_file:
        inputs["meta_file"] = str(scan.meta_file)

    write_provenance(
        h5f,
        entry=entry,
        program="xdart",
        config=config,
        inputs=inputs or None,
    )


# ---------------------------------------------------------------------------
# Frame-selection for the stacked-write primitive
# ---------------------------------------------------------------------------
#
# ``write_integrated_stack`` rewrites the group from whatever batch it's
# handed when the incoming row size differs from disk (C3 shape change),
# so the GUI adapter must guarantee that batch is *complete* in that
# case — otherwise a mid-scan numpoints change would silently drop the
# earlier frames.  For the normal same-shape append we hand it only the
# frames added since the last save (O(K)).  This selection logic is the
# one piece that has to stay xdart-side; the actual write is shared.

def _existing_dataset_n(h5f, path: str) -> int:
    """Return on-disk frame count for an integrated_* group, or 0 — O(1)."""
    if path not in h5f:
        return 0
    g = h5f[path]
    if "intensity" not in g:
        return 0
    return int(g["intensity"].shape[0])


def _per_frame_len(h5f, ds_path: str) -> int:
    """First-axis length of a per-frame dataset (e.g. a ``frame_index``), or
    -1 if absent — used to detect a stale per-frame metadata table."""
    if ds_path in h5f:
        try:
            return int(h5f[ds_path].shape[0])
        except (TypeError, IndexError):
            return -1
    return -1


def _positioners_len(h5f, entry: str) -> int:
    """Length of any one NXpositioner ``value`` array (sample or detector),
    or -1 if no positioners exist — staleness probe for the positioner
    tables, which have no single frame_index of their own."""
    for grp_path in (f"{entry}/sample/positioners",
                     f"{entry}/instrument/detector/positioners"):
        if grp_path not in h5f:
            continue
        grp = h5f[grp_path]
        for key in grp:
            val = grp[key].get("value") if hasattr(grp[key], "get") else None
            if val is not None:
                try:
                    return int(val.shape[0])
                except (TypeError, IndexError):
                    return -1
    return -1


def _new_frames_for_write(scan, h5f, group_path: str) -> tuple[list, int]:
    """Return ``(new_frames, existing_n)`` for an incremental append.

    ``existing_n`` is the on-disk row count for this group's
    ``intensity`` dataset (0 if the group / dataset doesn't exist yet).
    ``new_frames`` is the slice of in-memory frames whose data needs to
    be appended — frames at index ``[existing_n:total_n]`` of
    ``scan.frames.index``.  For normal append workflows those frames are
    in ``LiveFrameSeries._in_memory`` (the wrangler just stashed them),
    so this materialises without disk reads.

    If on-disk has *more* frames than in-memory (rare; scan reloaded with
    fewer frames after a partial save) we return ``existing_n=-1`` so the
    caller falls back to a full rewrite.
    """
    existing_n = _existing_dataset_n(h5f, group_path)
    total_n = len(scan.frames.index)
    if existing_n > total_n:
        return [], -1
    # Select frames whose *label* isn't already on disk, not a positional
    # tail slice.  A late/out-of-order frame (e.g. frame 1 arriving after
    # [0, 2] are saved) sits at a position inside the slice the tail would
    # skip, so the positional approach dropped it — its per-frame group
    # got written but its integrated row never did.  Comparing labels
    # writes exactly the rows missing from the stack.
    on_disk: set = set()
    if group_path in h5f and "frame_index" in h5f[group_path]:
        on_disk = {
            int(x) for x in np.asarray(
                h5f[group_path]["frame_index"][()]
            ).ravel()
        }
    new_indices = [i for i in scan.frames.index if int(i) not in on_disk]
    new_frames = [scan.frames[i] for i in new_indices]
    return new_frames, existing_n


def _disk_row_shape(h5f, group_path: str) -> tuple | None:
    """Trailing (per-frame) shape of an integrated_* intensity stack, or None."""
    if group_path in h5f and "intensity" in h5f[group_path]:
        return tuple(h5f[group_path]["intensity"].shape[1:])
    return None


def _select_frames_to_write(scan, h5f, group_path, replace_frame_indices,
                            row_shape_fn) -> tuple[list, list]:
    """Choose ``(frames, frame_indices)`` to pass to ``write_integrated_stack``.

    ``row_shape_fn(frame)`` returns the on-disk row shape that frame's
    result would occupy (``None`` if the frame has no result), so a
    mismatch against the existing stack can be detected and widened to a
    full rewrite.
    """
    all_ids = list(scan.frames.index)

    def _all_frames():
        return [scan.frames[i] for i in all_ids], list(all_ids)

    # ── Replace (reintegration) ──────────────────────────────────────
    # Hand the recomputed frames; the primitive upserts each row in
    # place.  If the row size changed (numpoints/unit), the primitive
    # rewrites from the batch — so widen to *all* frames to avoid
    # dropping the ones not in ``replace_frame_indices``.
    if replace_frame_indices is not None and group_path in h5f:
        ids = [i for i in replace_frame_indices if i in scan.frames.index]
        if not ids:
            return [], []
        frames = [scan.frames[i] for i in ids]
        disk = _disk_row_shape(h5f, group_path)
        new_shape = row_shape_fn(frames[0])
        if disk is not None and new_shape is not None and disk != new_shape:
            return _all_frames()
        return frames, ids

    # ── Append (default; also replace-with-no-existing-group) ────────
    new_frames, existing_n = _new_frames_for_write(scan, h5f, group_path)
    if existing_n == -1:
        if group_path in h5f:
            del h5f[group_path]
        return _all_frames()
    if not new_frames:
        return [], []
    disk = _disk_row_shape(h5f, group_path)
    new_shape = row_shape_fn(new_frames[0])
    if disk is not None and new_shape is not None and disk != new_shape:
        # Mid-scan parameter change: the primitive rewrites the group, so
        # it must see every frame, not just the new tail.
        return _all_frames()
    return new_frames, [int(getattr(fr, "idx", i)) for i, fr in
                        zip(range(existing_n, existing_n + len(new_frames)),
                            new_frames)]


def _row_shape_1d(frame) -> tuple | None:
    r = getattr(frame, "int_1d", None)
    if r is None or getattr(r, "intensity", None) is None:
        return None
    return (int(np.asarray(r.intensity).shape[0]),)


def _row_shape_2d(frame) -> tuple | None:
    r = getattr(frame, "int_2d", None)
    if r is None or getattr(r, "intensity", None) is None:
        return None
    # int_2d.intensity is xdart-shape (nq, nchi); on disk it's (nchi, nq).
    return tuple(np.asarray(r.intensity).T.shape)


def _write_integrated_1d(f, scan, *, entry: str,
                         replace_frame_indices=None) -> None:
    """Write/extend ``/entry/integrated_1d`` via the shared stacked-write
    primitive.  See :func:`_select_frames_to_write` for the append-cursor
    and shape-change handling that stays GUI-side."""
    if not scan.frames.index:
        return
    from ssrl_xrd_tools.io.nexus import write_integrated_stack

    h5f = _h5(f)
    group_path = f"{entry}/integrated_1d"
    frames, indices = _select_frames_to_write(
        scan, h5f, group_path, replace_frame_indices, _row_shape_1d,
    )
    if not frames:
        return
    results = [getattr(fr, "int_1d", None) for fr in frames]
    if any(r is None for r in results):
        return
    write_integrated_stack(
        h5f.require_group(entry), frame_indices=indices, results_1d=results,
    )


def _write_integrated_2d(f, scan, *, entry: str,
                         replace_frame_indices=None) -> None:
    """Write/extend ``/entry/integrated_2d`` via the shared stacked-write
    primitive.  Per-frame ``int_2d`` is xdart-shape ``(nq, nchi)``; the
    primitive transposes it to ``(nchi, nq)`` on disk."""
    if not scan.frames.index:
        return
    from ssrl_xrd_tools.io.nexus import write_integrated_stack

    h5f = _h5(f)
    group_path = f"{entry}/integrated_2d"
    frames, indices = _select_frames_to_write(
        scan, h5f, group_path, replace_frame_indices, _row_shape_2d,
    )
    if not frames:
        return
    results = [getattr(fr, "int_2d", None) for fr in frames]
    if any(r is None for r in results):
        return
    write_integrated_stack(
        h5f.require_group(entry), frame_indices=indices, results_2d=results,
    )


def _write_per_frame_metadata(f, scan, *, entry: str) -> None:
    """Per-frame thumbnails + source refs as :class:`NXcollection` groups.

    Layout::

        /entry/frames/             NXcollection
            frame_NNNN/            NXcollection
                thumbnail          uint8 (with @vmin, @vmax, @dtype)
                timestamp          str, optional
                source/            NXcollection, optional
                    path           str (relpath to the raw source file)
                    frame_index    int  (index within the source file)

    This stays nexusformat-based (and xdart-side): thumbnails are a
    viewer concern, not part of the headless reduction schema.

    Performance note: per-frame groups are *append-only* during a
    scan — once a frame's thumbnail/metadata is on disk it doesn't
    change.  We pull the already-written frame keys from h5py and only
    materialise a :class:`LiveFrame` for indices *not* yet on disk —
    those hit the in-memory cache the wrangler populated moments
    earlier, so a single save costs O(new frames) and zero lazy-loads.
    """
    if not scan.frames.index:
        return

    # The top-level /entry/frames container needs to exist as an
    # NXcollection.  Re-creating it would clobber per-frame groups
    # written by previous batches, so we create-if-missing instead of
    # del-and-replace.
    frames_path = f"{entry}/frames"
    if frames_path not in f:
        f[frames_path] = nx.NXcollection()
    frames = f[frames_path]

    # Use the underlying h5py group for the existence check.  nx's
    # in-memory tree may not have refreshed since the last save, but
    # h5py reads off the open file directly — authoritative.
    h5_frames = _h5(f)[frames_path]
    existing_frame_keys = set(h5_frames.keys())

    # Filter the index *before* touching any frame object.  This is
    # the whole point of the cursor: we never lazy-load a frame we'd
    # immediately skip.
    new_indices = [
        idx for idx in scan.frames.index
        if f"frame_{idx:04d}" not in existing_frame_keys
    ]
    if not new_indices:
        return

    for idx in new_indices:
        frame = scan.frames[idx]
        frame_key = f"frame_{idx:04d}"

        # Build a fresh per-frame NXcollection for new frames.
        fg = nx.NXcollection()

        thumb = getattr(frame, "thumbnail", None)
        if thumb is not None:
            arr, lut = _quantize_thumbnail(thumb)
            # ``dtype`` collides with NXfield's reserved kwarg (which
            # sets the array dtype) — pass attrs as a dict so the
            # *attribute* named "dtype" goes through.
            fg["thumbnail"] = nx.NXfield(
                arr,
                attrs={"vmin": lut[0], "vmax": lut[1], "dtype": lut[2]},
            )

        # NOTE: per the v2 schema (module docstring §2), per-frame
        # groups carry *only* metadata + thumbnail — never the full
        # raw image.  An earlier version of this helper wrote
        # frame.map_raw verbatim, which silently dumped 18 MB per
        # Eiger frame into the .nxs.  Don't bring that back.

        _write_source_ref(fg, frame)

        ts = getattr(frame, "timestamp", None)
        if ts is not None:
            fg["timestamp"] = nx.NXfield(str(ts))

        frames[frame_key] = fg


def _write_source_ref(fg, frame) -> None:
    """Attach an ``NXcollection`` carrying the raw-source pointer.

    Writes ``source/path`` and ``source/frame_index`` under the
    per-frame group ``fg`` when the frame has a non-empty
    ``source_file`` attribute.  ``frame_index`` defaults to the
    frame's own ``idx`` when the frame doesn't carry an explicit
    ``source_frame_idx`` (typical for the image wrangler, where each
    image is a single-frame file; for Eiger / multi-frame sources the
    wrangler should set ``source_frame_idx`` to the index *within*
    the source data file).
    """
    src_path = getattr(frame, "source_file", "") or ""
    if not src_path:
        return
    src_frame_idx = getattr(frame, "source_frame_idx", None)
    if src_frame_idx is None:
        src_frame_idx = getattr(frame, "idx", 0)
    sub = nx.NXcollection()
    sub["path"] = nx.NXfield(str(src_path))
    sub["frame_index"] = nx.NXfield(int(src_frame_idx))
    fg["source"] = sub


def _write_positioners(f, scan, *, entry: str) -> None:
    """Write motor positioners under ``NXsample`` / ``NXdetector``.

    Delegates the layout to
    :func:`ssrl_xrd_tools.io.nexus.write_positioners`, which reindexes
    ``scan_data`` to the integrated-frame set (so the per-frame
    dimension matches ``integrated_1d``/``2d``) and splits sample- vs
    detector-axis motors via the geometry.  No geometry → no-op.
    """
    from ssrl_xrd_tools.io.nexus import write_positioners as _wp

    scan_data = getattr(scan, "scan_data", None)
    geom = getattr(scan, "geometry", None)
    if geom is None or scan_data is None or len(scan_data) == 0:
        return
    frame_index = list(getattr(getattr(scan, "frames", None), "index", []) or [])
    _wp(_h5(f).require_group(entry), scan_data, frame_index, geom)


def _write_scan_metadata(f, scan, *, entry: str) -> None:
    """Persist the full per-frame scan metadata table (delegates to
    :func:`ssrl_xrd_tools.io.nexus.write_scan_metadata`).

    Unlike positioners (geometry motors only), this stores every column the
    wrangler recorded in ``scan.scan_data`` so a reload restores the same
    metadata the live in-memory scan had — fixes the metadata panel showing
    only the incidence motor after a batch run reloads from disk.
    """
    from ssrl_xrd_tools.io.nexus import write_scan_metadata as _wsm

    scan_data = getattr(scan, "scan_data", None)
    if scan_data is None or len(scan_data) == 0:
        return
    frame_index = list(getattr(getattr(scan, "frames", None), "index", []) or [])
    _wsm(_h5(f).require_group(entry), scan_data, frame_index)


def _write_per_frame_geometry(f, scan, *, entry: str) -> None:
    """Write derived per-frame pyFAI rotations + incidence angle.

    Delegates to
    :func:`ssrl_xrd_tools.io.nexus.write_per_frame_geometry`, which
    reindexes ``scan_data`` to the frame set, derives rot1/2/3 +
    incident_angle via ``geometry.derive_per_frame``, and labels the
    rows with the actual frame ids (so a downstream join-by-frame_index
    lines up with ``integrated_1d``).  No geometry / no usable motor
    columns → no-op.
    """
    from ssrl_xrd_tools.io.nexus import write_per_frame_geometry as _wg

    scan_data = getattr(scan, "scan_data", None)
    geom = getattr(scan, "geometry", None)
    if geom is None or scan_data is None or len(scan_data) == 0:
        return
    frame_index = list(getattr(getattr(scan, "frames", None), "index", []) or [])
    _wg(_h5(f).require_group(entry), scan_data, frame_index, geom)


def _representative_poni(scan):
    """Return any PONI that represents the scan's geometry, without iteration.

    Beam-line geometry is constant across a scan (the wrangler holds
    a single :class:`AzimuthalIntegrator` on ``scan._cached_integrator``
    and copies it into each frame).  The instrument-metadata writer
    only needs *one* PONI to stamp the .nxs file.

    Resolution order:

    1. Any frame in ``LiveFrameSeries._in_memory`` — the wrangler always
       leaves at least the most recent batch's frames here, so this
       is the zero-disk path.
    2. The scan's cached pyFAI integrator (if attached by the
       wrangler), reconstituted into a PONI-shaped object — useful
       if ``_in_memory`` was somehow drained.

    Returns ``None`` only when both sources are absent (e.g. an
    empty scan serialised before any frame was integrated, or a
    unit test that never set up an integrator).
    """
    in_mem = getattr(scan.frames, "_in_memory", None)
    if in_mem:
        any_frame = next(iter(in_mem.values()))
        poni = getattr(any_frame, "poni", None)
        if poni is not None:
            return poni
    ai = getattr(scan, "_cached_integrator", None)
    if ai is None:
        return None
    # Lazy import — avoids circular dep via xdart.modules.ewald in
    # the test fixtures, and the PONI class isn't pulled in by the
    # ``frame.py`` import chain.
    try:
        from xdart.utils.containers import PONI  # type: ignore
    except Exception:  # pragma: no cover
        try:
            from ssrl_xrd_tools.integrate.calibration import PONI  # type: ignore
        except Exception:
            return None
    return PONI(
        dist=float(getattr(ai, "dist", 0.0)),
        poni1=float(getattr(ai, "poni1", 0.0)),
        poni2=float(getattr(ai, "poni2", 0.0)),
        rot1=float(getattr(ai, "rot1", 0.0)),
        rot2=float(getattr(ai, "rot2", 0.0)),
        rot3=float(getattr(ai, "rot3", 0.0)),
    )


def _write_instrument(f, scan, *, entry: str) -> None:
    """Write :class:`NXinstrument` with :class:`NXsource` + :class:`NXdetector`.

    Operates on the raw h5py file (not nexusformat) so it coexists
    cleanly with :func:`_write_positioners`, which writes
    ``/instrument/detector/positioners`` via the h5py-based ssrl
    primitive.  Mixing a nexusformat write here against those h5py
    writes risked nx's cached tree clobbering the just-written
    positioners — using h5py for both keeps the detector group
    consistent.  We create-or-require groups and replace only the
    specific scalar datasets, never the positioners subgroup.

    Layout::

        /entry/instrument/              NXinstrument
            source/                     NXsource
                wavelength_A            float, scalar
            detector/                   NXdetector
                dist, poni1, poni2,     float, scalar (pyFAI geometry)
                  rot1, rot2, rot3
                mask                    int64 (N,)  flat pixel indices
                  @description
                positioners/            NXcollection (written by
                                        _write_positioners; preserved)
    """
    h5f = _h5(f)
    instr = h5f.require_group(f"{entry}/instrument")
    instr.attrs["NX_class"] = "NXinstrument"

    # ── source (NXsource) ─────────────────────────────────────────
    wavelength = scan.mg_args.get("wavelength")
    if wavelength is not None:
        if "source" in instr:
            del instr["source"]
        src = instr.create_group("source")
        src.attrs["NX_class"] = "NXsource"
        src.create_dataset("wavelength_A", data=float(wavelength) * 1e10)

    # ── detector (NXdetector) ─────────────────────────────────────
    det = instr.require_group("detector")
    det.attrs["NX_class"] = "NXdetector"

    # PONI scalars — read from the representative source (see helper).
    poni = _representative_poni(scan)
    if poni is not None:
        for k in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3"):
            v = getattr(poni, k, None)
            if v is not None:
                if k in det:
                    del det[k]
                det.create_dataset(k, data=float(v))

    # Global mask — flat indices of masked pixels (detector mask + the
    # user-supplied Mask File, combined via the wrangler).  Stored so
    # the viewer can overlay the mask without the original mask file.
    gmask = getattr(scan, "global_mask", None)
    arr = None
    if gmask is not None:
        try:
            arr = np.asarray(gmask, dtype=np.int64)
        except (TypeError, ValueError):
            arr = None
    # Always drop any prior mask first so CLEARING the mask (gmask None or
    # empty) actually removes it on disk — otherwise a rewrite leaves the
    # old mask in place and a reload restores a mask the user cleared.
    if "mask" in det:
        del det["mask"]
    if arr is not None and arr.size > 0:
        ds = det.create_dataset("mask", data=arr)
        ds.attrs["description"] = "flat pixel indices, shape (N,)"


def _write_stitched(f, scan, *, entry: str) -> None:
    """Write stitched 1D / 2D outputs via the shared primitive.

    Delegates to :func:`ssrl_xrd_tools.io.nexus.write_stitched` (the
    symmetric counterpart to ``read_stitched``).  Only invoked when
    ``finalize=True`` (typically end-of-scan).

    Note the orientation owned by the primitive: ``stitched_2d`` is
    stored **as-is** ``(n_q, n_chi)`` with dims ``(q, chi)`` — unlike
    the per-frame ``integrated_2d`` stack ``(frame, chi, q)``.
    """
    from ssrl_xrd_tools.io.nexus import write_stitched as _ws

    s1 = getattr(scan, "stitched_1d", None)
    s2 = getattr(scan, "stitched_2d", None)
    if s1 is None and s2 is None:
        return
    _ws(_h5(f).require_group(entry), stitched_1d=s1, stitched_2d=s2)


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def _quantize_thumbnail(
    arr: np.ndarray,
    dtype: str = "uint8",
) -> tuple[np.ndarray, tuple[float, float, str]]:
    """Linear-quantize a 2-D thumbnail array to uint8 or uint16.

    Returns the quantized array + the LUT triple ``(vmin, vmax, dtype)``
    for storage as attributes so viewers can invert.
    """
    finite = np.isfinite(arr)
    if not finite.any():
        # All NaN/inf — produce a flat zero thumbnail
        quant = np.zeros(arr.shape, dtype=np.uint8 if dtype == "uint8" else np.uint16)
        return quant, (0.0, 1.0, dtype)
    vmin, vmax = np.percentile(arr[finite], [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1e-12
    # Replace NaN/inf (typically from masked pixels) with 0 BEFORE the
    # clip so they don't propagate through (arr - vmin) / range → NaN →
    # (NaN * 255).astype(uint8) which raises "invalid value in cast".
    arr_clean = np.where(finite, arr, vmin)
    norm = np.clip((arr_clean - vmin) / (vmax - vmin), 0, 1)
    if dtype == "uint16":
        return (norm * 65535).astype(np.uint16), (float(vmin), float(vmax), "uint16")
    return (norm * 255).astype(np.uint8), (float(vmin), float(vmax), "uint8")


__all__ = ["save_scan_to_nexus"]
