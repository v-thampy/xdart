"""xdart v2 NeXus writer (xdart 0.37+ schema).

This module produces files conforming to the layout described in
``xdart/docs/nexus_stitch_refactor_plan.md`` §2.  The single public
entry point is :func:`save_scan_to_nexus`, called from
:meth:`LiveScan._save_to_nexus`.

**Keep-xdart-thin (#18):** the on-disk layout for the stacked
``integrated_1d``/``integrated_2d`` groups, the ``stitched_*`` groups,
the motor ``positioners``, and ``per_frame_geometry`` is owned by the
shared, headless-reusable primitives in
:mod:`xrd_tools.io.nexus` (``write_integrated_stack``,
``write_stitched``, ``write_positioners``, ``write_per_frame_geometry``).
This module is now a thin GUI-side adapter: it gathers the LiveScan's
in-memory state (frames, scan_data, geometry, PONI, thumbnails), decides
*which* frames to hand the stacked-write primitive (the O(K) append
cursor + the "require an explicit full rewrite on axis change" guard live
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
   :func:`xrd_tools.core.provenance.write_provenance` — versions
   are pulled from ``importlib.metadata`` and never hard-coded.
"""

from __future__ import annotations

import logging
import os
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import h5py
import nexusformat.nexus as nx
import numpy as np

from xrd_tools.io.nexus import resolve_stack_compression

if TYPE_CHECKING:  # pragma: no cover
    from xrd_tools.core.geometry import DiffractometerGeometry

    from xdart.modules.live import LiveScan


logger = logging.getLogger(__name__)

# Compression for the GUI writer's integrated 1D/2D stacks.  Resolved by the
# shared core helper (xrd_tools.io.nexus.resolve_stack_compression) so the GUI
# writer and the headless reduction honor the SAME XDART_INTEGRATED_COMPRESSION
# env var (read ONCE at import; set it in the shell BEFORE launching xdart):
#     lz4  -> hdf5plugin LZ4+shuffle (DEFAULT; fast, reader needs hdf5plugin which
#             is a base dep; falls back to gzip only if hdf5plugin is missing)
#     gzip -> gzip+shuffle (portable; stock-h5py readable, no hdf5plugin needed)
#     none -> uncompressed (biggest files)
# gzip/lzf both map to portable gzip+shuffle; raw lzf is never emitted (ARM64-macOS
# bus error).  lz4 round-trips on arm64-macOS (verified); the old bus error was
# h5py's bundled LZF, not hdf5plugin's LZ4.  On the live streaming pipeline the
# write cost is overlapped.  _resolve_integrated_compression kept as a thin alias.
_resolve_integrated_compression = resolve_stack_compression
INTEGRATED_STACK_COMPRESSION = _resolve_integrated_compression()
logger.info("Integrated-stack compression = %r", INTEGRATED_STACK_COMPRESSION)


@dataclass
class NexusWriteCursor:
    """Trusted append position for one live scan output file."""

    path: str
    groups: dict[str, tuple[int, int | None, tuple]] = field(default_factory=dict)
    metadata: tuple[int, int | None, tuple] | None = None
    instrument: tuple | None = None
    # H1: labels whose row for a given group is permanently unwritable
    # (publication-gate-rejected, or lazy-reloaded with no result).  Excluded
    # from append selection so they aren't re-lazy-loaded inside the open
    # writer handle and re-skipped (with a warning) on every save.
    # {group_path: set(labels)}
    dropped: dict = field(default_factory=dict)


def _write_cursor(scan, h5f) -> NexusWriteCursor:
    path = os.fspath(h5f.filename)
    cursor = getattr(scan, "_nexus_write_cursor", None)
    if cursor is None or cursor.path != path:
        cursor = NexusWriteCursor(path=path)
        try:
            scan._nexus_write_cursor = cursor
        except AttributeError:
            pass
    return cursor


def _index_structure_signature(index, n: int) -> tuple:
    """Cheaply identify whether the saved prefix can still be trusted."""
    version = getattr(index, "_structure_version", None)
    if version is not None:
        return ("version", int(version))
    prefix = [int(x) for x in list(index)[:n]]
    return ("fingerprint", len(prefix), hash(tuple(prefix)))


def _array_digest(arr) -> tuple:
    if arr is None:
        return ("none",)
    try:
        a = np.asarray(arr)
    except (TypeError, ValueError):
        return ("invalid", repr(arr))
    if a.size == 0:
        return ("empty",)
    ac = np.ascontiguousarray(a)
    digest = hashlib.blake2b(ac.view(np.uint8), digest_size=16).hexdigest()
    return (tuple(ac.shape), str(ac.dtype), digest)


# ---------------------------------------------------------------------------
# File-opening helper — mirrors xrd_tools.core.hdf5.catch_h5py_file
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
    ``xrd_tools.core.hdf5.catch_h5py_file`` does so the writer
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
    ``xrd_tools.io.nexus`` operate on a raw :class:`h5py.Group`, so
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
    _atomic_write: bool = True,
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
    path = Path(path)
    if _atomic_write and mode == "w":
        tmp_path = path.with_name(
            f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
        )
        try:
            save_scan_to_nexus(
                scan,
                tmp_path,
                mode=mode,
                entry=entry,
                finalize=finalize,
                replace_frame_indices=replace_frame_indices,
                _atomic_write=False,
            )
            os.replace(tmp_path, path)
        except Exception:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise
        return

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
        cursor = _write_cursor(scan, h5f)

        # 1. Stacked integrated_1d and integrated_2d (delegated to
        #    xrd_tools.io.nexus.write_integrated_stack).  Select and
        # validate both outputs before either is written so a 2D mismatch
        # cannot leave 1D one frame ahead — or even refresh provenance.
        prepared_1d = _prepare_integrated_1d(
            f, scan, entry=entry,
            replace_frame_indices=replace_frame_indices,
            cursor=cursor,
        )
        prepared_2d = _prepare_integrated_2d(
            f, scan, entry=entry,
            replace_frame_indices=replace_frame_indices,
            cursor=cursor,
        )
        prepared_1d, prepared_2d = _filter_prepared_frame_publications(
            prepared_1d, prepared_2d,
        )
        # Drop the publication-rejected replace rows from disk BEFORE the
        # uniform-stack validation: with a changed row shape/axis the
        # coverage check (_require_batch_covers_existing) would otherwise
        # see the rejected frame's STALE row still on disk, uncovered by
        # the incoming batch, and abort the WHOLE reintegration save over
        # one per-frame drop -- violating the publication-gate contract
        # (reject per frame, never abort whole-scan).  The dropped rows are
        # publication-invalid; removing them is correct even if a later
        # validation step still fails.
        _drop_filtered_replace_rows(h5f, prepared_1d, prepared_2d)
        _validate_prepared_integrated(h5f.require_group(entry), prepared_1d, prepared_2d)

        # 2. Provenance — append mode: only on first save or finalize.
        # Replace mode: always rewrite so the persisted ``bai_*_args``
        # reflect whatever parameters the reintegration used (this is
        # the whole reason the user kicked off a re-integration in the
        # first place).
        # P2 (codex): the GI first-chunk-freeze diagnostic is stamped on the
        # scan AFTER initialize_scan's first save wrote the reduction group,
        # and the GUI never passes finalize=True -- without this re-fire the
        # persisted-provenance disclosure never landed on the real path.
        # Cursor-deduped so steady-state saves stay write-free.
        _gi_diag = getattr(scan, "gi_freeze_diagnostic", None)
        _cur = _write_cursor(scan, h5f)
        _diag_pending = bool(_gi_diag) and getattr(
            _cur, "gi_diag_written", None) != _gi_diag
        if (is_replace or finalize or _diag_pending
                or "reduction" not in h5f.get(entry, {})):
            _write_reduction(h5f, scan, entry=entry)
            if _gi_diag:
                _cur.gi_diag_written = _gi_diag
        _t0 = _tick("reduction", _t0)

        _commit_integrated_1d(f, prepared_1d)
        _t0 = _tick("integrated_1d", _t0)
        _commit_integrated_2d(f, prepared_2d)
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
            # Per-frame metadata tables (scan_data, positioners,
            # per_frame_geometry) must stay the SAME length as the stacked
            # integrated_* rows.  Live mode never passes finalize=True and
            # saves per-frame, so a once-only "first save" gate froze these
            # at their first length while integrated rows kept growing — a
            # reloaded file then has e.g. 5 integrated frames but 2 metadata
            # rows (read_scan drops the short columns).  Rewrite each
            # whenever it's stale (on-disk length != current frame count).
            # Use tail upserts during ordered acquisition; reconcile the full
            # metadata tables only after reload, reorder, or finalization.
            _write_incremental_metadata(
                f, scan, entry=entry, cursor=cursor, finalize=finalize,
            )
            _t0 = _tick("frame_metadata_tables", _t0)
        instr_path = f"{entry}/instrument"
        first_instr = instr_path not in h5f
        instrument_sig = _instrument_signature(scan)
        if finalize or first_instr or cursor.instrument != instrument_sig:
            _write_instrument(f, scan, entry=entry)
            cursor.instrument = instrument_sig
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
    """Write /entry/reduction/ via xrd_tools provenance."""
    from xrd_tools.core.provenance import write_provenance

    config: dict[str, Any] = {
        "bai_1d_args": dict(scan.bai_1d_args),
        "bai_2d_args": dict(scan.bai_2d_args),
    }
    if hasattr(scan, "gi_config") and scan.gi_config:
        config["gi_config"] = dict(scan.gi_config)
    # T0-4 disclosure: when the GI grid was frozen from the first chunk
    # because the whole-scan incidence range couldn't be verified, persist
    # the advisory in the output file — not just a transient GUI label.
    _gi_diag = getattr(scan, "gi_freeze_diagnostic", None)
    if _gi_diag:
        config["gi_freeze_diagnostic"] = str(_gi_diag)

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

    from xdart import __version__ as _xdart_version

    write_provenance(
        h5f,
        entry=entry,
        program="xdart",
        # explicit: the dist is "xrd-tools" now, so write_provenance's
        # importlib lookup of "xdart" would silently record '' (or a stale
        # legacy install's version).
        program_version=_xdart_version,
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


def _new_frames_for_write(scan, h5f, group_path: str,
                          cursor: NexusWriteCursor | None = None) -> tuple[list, int]:
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
    if cursor is not None and group_path in cursor.groups:
        cached_n, cached_last, cached_sig = cursor.groups[group_path]
        disk_last = (
            int(h5f[group_path]["frame_index"][existing_n - 1])
            if existing_n and group_path in h5f and "frame_index" in h5f[group_path]
            else None
        )
        memory_last = int(scan.frames.index[existing_n - 1]) if existing_n else None
        memory_sig = _index_structure_signature(scan.frames.index, existing_n)
        if (cached_n == existing_n and cached_last == disk_last
                and cached_sig == memory_sig and memory_last == disk_last):
            _drop = cursor.dropped.get(group_path, ()) if cursor else ()
            new_indices = [i for i in scan.frames.index[existing_n:]
                           if int(i) not in _drop]
            return [scan.frames[i] for i in new_indices], existing_n
    # Select frames whose *label* isn't already on disk, not a positional
    # tail slice.  A late/out-of-order frame (e.g. frame 1 arriving after
    # [0, 2] are saved) sits at a position inside the slice the tail would
    # skip, so the positional approach dropped it — its per-frame group
    # got written but its integrated row never did.  Comparing labels
    # writes exactly the rows missing from the stack.
    on_disk: set = set()
    if group_path in h5f and "frame_index" in h5f[group_path]:
        disk_ids = [
            int(x) for x in np.asarray(
                h5f[group_path]["frame_index"][()]
            ).ravel()
        ]
        on_disk = set(disk_ids)
        live_ids = {int(x) for x in scan.frames.index}
        if not on_disk.issubset(live_ids):
            return [], -2
    _drop = cursor.dropped.get(group_path, ()) if cursor else ()
    new_indices = [i for i in scan.frames.index
                   if int(i) not in on_disk and int(i) not in _drop]
    new_frames = [scan.frames[i] for i in new_indices]
    return new_frames, existing_n


def _refresh_group_cursor(
    cursor: NexusWriteCursor | None,
    h5f,
    group_path: str,
    scan_index=None,
) -> None:
    if cursor is None or group_path not in h5f or "frame_index" not in h5f[group_path]:
        return
    labels = h5f[group_path]["frame_index"]
    n = int(labels.shape[0])
    if scan_index is None:
        scan_index = [int(x) for x in np.asarray(labels[()]).ravel()]
    cursor.groups[group_path] = (
        n,
        int(labels[n - 1]) if n else None,
        _index_structure_signature(scan_index, n),
    )


def _disk_row_shape(h5f, group_path: str) -> tuple | None:
    """Trailing (per-frame) shape of an integrated_* intensity stack, or None."""
    if group_path in h5f and "intensity" in h5f[group_path]:
        return tuple(h5f[group_path]["intensity"].shape[1:])
    return None


def _select_frames_to_write(scan, h5f, group_path, replace_frame_indices,
                            row_shape_fn, axis_signature_fn,
                            cursor: NexusWriteCursor | None = None) -> tuple[list, list]:
    """Choose ``(frames, frame_indices)`` to pass to ``write_integrated_stack``.

    ``row_shape_fn(frame)`` returns the on-disk row shape that frame's
    result would occupy (``None`` if the frame has no result), so a
    mismatch against the existing stack can require an explicit full rewrite.
    """
    all_ids = list(scan.frames.index)

    # A replace save (reintegration) supplies FRESH results for its frames:
    # clear their stale ``dropped`` bookkeeping up front.  Without this, a
    # group whose every row was publication-rejected during the run (group
    # never created on disk) fell through ``group_path in h5f`` into the
    # append branch below, where the dropped set silently excluded every
    # recomputed frame -- the designed recovery path wrote nothing.  A row
    # that is STILL invalid after recomputation is re-dropped by the
    # publication gate in this same save.
    if replace_frame_indices is not None and cursor is not None:
        stale = cursor.dropped.get(group_path)
        if stale:
            stale.difference_update(int(i) for i in replace_frame_indices)

    def _all_frames():
        return [scan.frames[i] for i in all_ids], list(all_ids)

    # ── Replace (reintegration) ──────────────────────────────────────
    # Hand the recomputed frames; the primitive upserts each row in
    # place.  If its axis or shape changed, require the caller to have
    # recomputed every frame; never widen a partial batch with stale rows.
    if replace_frame_indices is not None and group_path in h5f:
        ids = [i for i in replace_frame_indices if i in scan.frames.index]
        if not ids:
            return [], []
        frames = [scan.frames[i] for i in ids]
        disk = _disk_row_shape(h5f, group_path)
        new_shape = row_shape_fn(frames[0])
        axis_changed = not _axis_signatures_equal(
            _disk_axis_signature(h5f, group_path), axis_signature_fn(frames[0]),
        )
        if disk is not None and new_shape is not None and (disk != new_shape or axis_changed):
            if set(map(int, ids)) != set(map(int, all_ids)):
                raise ValueError(
                    "Reintegration changed the output axis, unit, or row shape. "
                    "Recompute and save every frame together; a partial rewrite "
                    "would mix fresh rows with stale rows."
                )
            return frames, ids
        return frames, ids

    # ── Append (default; also replace-with-no-existing-group) ────────
    new_frames, existing_n = _new_frames_for_write(scan, h5f, group_path, cursor)
    if existing_n == -1:
        raise ValueError(
            f"{group_path} has more persisted rows than the live scan; "
            "reload or perform an explicit full reintegration."
        )
    if existing_n == -2:
        raise ValueError(
            f"{group_path} contains persisted frame labels that are no longer "
            "present in the live scan; reload or perform an explicit full "
            "reintegration."
        )
    if not new_frames:
        return [], []
    disk = _disk_row_shape(h5f, group_path)
    new_shape = row_shape_fn(new_frames[0])
    if disk is not None and new_shape is not None and (
        disk != new_shape
        or not _axis_signatures_equal(
            _disk_axis_signature(h5f, group_path), axis_signature_fn(new_frames[0]),
        )
    ):
        raise ValueError(
            "Integration settings changed during append. Reintegrate and save "
            "the complete scan so persisted rows share one axis."
        )
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


def _decode_unit(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")


def _axis_signature_1d(frame):
    result = getattr(frame, "int_1d", None)
    if result is None:
        return None
    return (np.asarray(result.radial), str(getattr(result, "unit", "") or ""))


def _axis_signature_2d(frame):
    result = getattr(frame, "int_2d", None)
    if result is None:
        return None
    return (
        np.asarray(result.radial),
        np.asarray(result.azimuthal),
        str(getattr(result, "unit", "") or ""),
        str(getattr(result, "azimuthal_unit", "") or ""),
    )


def _disk_axis_signature(h5f, group_path: str):
    if group_path not in h5f:
        return None
    group = h5f[group_path]
    if "q" not in group:
        return None
    if group_path.endswith("integrated_1d"):
        return (np.asarray(group["q"][()]), _decode_unit(group["q"].attrs.get("units", "")))
    if "chi" not in group:
        return None
    return (
        np.asarray(group["q"][()]),
        np.asarray(group["chi"][()]),
        _decode_unit(group["q"].attrs.get("units", "")),
        _decode_unit(group["chi"].attrs.get("units", "")),
    )


def _axis_signatures_equal(left, right) -> bool:
    if left is None or right is None or len(left) != len(right):
        return left is right
    for a, b in zip(left, right):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            aa, bb = np.asarray(a), np.asarray(b)
            if aa.shape != bb.shape or not np.allclose(aa, bb, rtol=1e-6, atol=1e-7,
                               equal_nan=True):
                return False
        elif a != b:
            return False
    return True


def _validate_prepared_integrated(entry_grp, prepared_1d, prepared_2d) -> None:
    """Preflight every selected output before committing either one."""
    from xrd_tools.io.nexus import validate_integrated_stack_write
    if prepared_1d is not None and prepared_1d["results"]:
        validate_integrated_stack_write(
            entry_grp,
            frame_indices=prepared_1d["indices"],
            results_1d=prepared_1d["results"],
        )
    if prepared_2d is not None and prepared_2d["results"]:
        validate_integrated_stack_write(
            entry_grp,
            frame_indices=prepared_2d["indices"],
            results_2d=prepared_2d["results"],
        )


def _filter_prepared_frame_publications(prepared_1d, prepared_2d):
    """Drop invalid rows per output before mutating disk.

    This complements the strict ssrl stack validators above: those ensure
    rows can be stacked consistently, while this gate catches frame-level
    diagnostics such as all-dummy GI cakes before they reach display or disk.
    It filters 1D and 2D independently so one bad frame/output does not lose
    the rest of the scan.
    """
    from xdart.modules.frame_publication import (
        publication_error_details,
        publication_from_live_frame,
        publication_has_1d_errors,
        publication_has_2d_errors,
    )

    checked: dict[int, object] = {}
    filtered_1d = _filter_prepared_output(
        prepared_1d,
        checked,
        has_errors=publication_has_1d_errors,
        error_details=lambda pub: publication_error_details(pub, "1d"),
        label="1D",
        publication_from_live_frame=publication_from_live_frame,
    )
    filtered_2d = _filter_prepared_output(
        prepared_2d,
        checked,
        has_errors=publication_has_2d_errors,
        error_details=lambda pub: publication_error_details(pub, "2d"),
        label="2D",
        publication_from_live_frame=publication_from_live_frame,
    )
    return filtered_1d, filtered_2d


def _filter_prepared_output(
    prepared,
    checked: dict[int, object],
    *,
    has_errors,
    error_details,
    label: str,
    publication_from_live_frame,
):
    if prepared is None:
        return None

    kept_frames = []
    kept_indices = []
    kept_results = []
    dropped_indices = []
    for frame, idx, result in zip(
        prepared["frames"], prepared["indices"], prepared["results"],
    ):
        key = id(frame)
        publication = checked.get(key)
        if publication is None:
            publication = publication_from_live_frame(frame, validate=True)
            checked[key] = publication
        if has_errors(publication):
            logger.warning(
                "Skipping frame %s %s write: %s",
                idx,
                label,
                error_details(publication),
            )
            dropped_indices.append(idx)
            _cur = prepared.get("cursor")
            if _cur is not None and not prepared.get("is_replace"):
                _cur.dropped.setdefault(
                    prepared["group_path"], set()).add(int(idx))
            continue
        kept_frames.append(frame)
        kept_indices.append(idx)
        kept_results.append(result)

    if not kept_frames and not (dropped_indices and prepared.get("is_replace")):
        logger.warning(
            "Skipping %s integrated stack write: every selected row failed "
            "publication validation.",
            label,
        )
        return None
    if len(kept_frames) == len(prepared["frames"]):
        return prepared
    filtered = dict(prepared)
    filtered["frames"] = kept_frames
    filtered["indices"] = kept_indices
    filtered["results"] = kept_results
    filtered["dropped_indices"] = dropped_indices
    return filtered


def _drop_filtered_replace_rows(h5f, *prepared_outputs) -> None:
    for prepared in prepared_outputs:
        if not prepared or not prepared.get("is_replace"):
            continue
        dropped = prepared.get("dropped_indices") or []
        if not dropped:
            continue
        _drop_integrated_rows(h5f, prepared["group_path"], dropped)
        _refresh_group_cursor(
            prepared["cursor"], h5f, prepared["group_path"], prepared["scan_index"],
        )
        from xdart.modules.ewald.frame_series import clear_frame_position_cache
        clear_frame_position_cache(h5f.filename)


def _drop_integrated_rows(h5f, group_path: str, frame_indices) -> None:
    """Row surgery moved to the core (6a): xrd_tools.io.nexus_record."""
    from xrd_tools.io.nexus_record import drop_integrated_rows
    drop_integrated_rows(h5f, group_path, frame_indices)


def _prepare_integrated_1d(f, scan, *, entry: str,
                           replace_frame_indices=None,
                           cursor: NexusWriteCursor | None = None):
    """Select the 1D rows that would be written, without mutating disk."""
    if not scan.frames.index:
        return None

    h5f = _h5(f)
    group_path = f"{entry}/integrated_1d"
    frames, indices = _select_frames_to_write(
        scan, h5f, group_path, replace_frame_indices, _row_shape_1d,
        _axis_signature_1d, cursor,
    )
    if not frames:
        return
    results = [getattr(fr, "int_1d", None) for fr in frames]
    if any(r is None for r in results):
        # H1: drop result-less rows PER FRAME, never the whole save (a frame
        # lazy-reloaded after its row was publication-dropped has int_1d=None
        # forever; the old all-or-nothing skip silently truncated every
        # LATER frame's write too).  Remember the labels on the cursor so
        # they aren't re-selected (and re-lazy-loaded inside the open
        # writer) on every subsequent save.
        kept = [(fr, i, r) for fr, i, r in zip(frames, indices, results)
                if r is not None]
        missing = [int(i) for fr, i, r in zip(frames, indices, results)
                   if r is None]
        if not kept:
            # EVERY selected row lacks a result: structural (this output was
            # never computed for these frames), not the H1 mixed-drop
            # pathology -- skip silently and leave the cursor alone (a later
            # reintegrate may fill them in).
            return None
        if cursor is not None and replace_frame_indices is None:
            cursor.dropped.setdefault(group_path, set()).update(missing)
        logger.warning(
            "integrated_%s: skipping %d frame(s) with no result (%s%s); "
            "writing the remaining %d.", "1d", len(missing), missing[:8],
            "..." if len(missing) > 8 else "", len(kept),
        )
        frames = [fr for fr, _i, _r in kept]
        indices = [i for _fr, i, _r in kept]
        results = [r for _fr, _i, r in kept]
    return {
        "entry": entry,
        "group_path": group_path,
        "frames": frames,
        "indices": indices,
        "results": results,
        "cursor": cursor,
        "scan_index": scan.frames.index,
        "is_replace": replace_frame_indices is not None and group_path in h5f,
    }


def _commit_integrated_1d(f, prepared) -> None:
    if prepared is None or not prepared["results"]:
        return
    from xrd_tools.io.nexus import write_integrated_stack
    h5f = _h5(f)
    write_integrated_stack(
        h5f.require_group(prepared["entry"]),
        frame_indices=prepared["indices"],
        results_1d=prepared["results"],
        compression=INTEGRATED_STACK_COMPRESSION,
    )
    _refresh_group_cursor(
        prepared["cursor"], h5f, prepared["group_path"], prepared["scan_index"],
    )
    from xdart.modules.ewald.frame_series import clear_frame_position_cache
    clear_frame_position_cache(h5f.filename)


def _prepare_integrated_2d(f, scan, *, entry: str,
                           replace_frame_indices=None,
                           cursor: NexusWriteCursor | None = None):
    """Select the 2D rows that would be written, without mutating disk."""
    if not scan.frames.index:
        return None
    if getattr(scan, "skip_2d", False):
        # Int 1D mode: 2D is intentionally not computed -- nothing to select,
        # and the per-frame no-result warning below would be pure noise.
        return None

    h5f = _h5(f)
    group_path = f"{entry}/integrated_2d"
    frames, indices = _select_frames_to_write(
        scan, h5f, group_path, replace_frame_indices, _row_shape_2d,
        _axis_signature_2d, cursor,
    )
    if not frames:
        return
    results = [getattr(fr, "int_2d", None) for fr in frames]
    if any(r is None for r in results):
        # H1: drop result-less rows PER FRAME, never the whole save (a frame
        # lazy-reloaded after its row was publication-dropped has int_2d=None
        # forever; the old all-or-nothing skip silently truncated every
        # LATER frame's write too).  Remember the labels on the cursor so
        # they aren't re-selected (and re-lazy-loaded inside the open
        # writer) on every subsequent save.
        kept = [(fr, i, r) for fr, i, r in zip(frames, indices, results)
                if r is not None]
        missing = [int(i) for fr, i, r in zip(frames, indices, results)
                   if r is None]
        if not kept:
            # EVERY selected row lacks a result: structural (this output was
            # never computed for these frames), not the H1 mixed-drop
            # pathology -- skip silently and leave the cursor alone (a later
            # reintegrate may fill them in).
            return None
        if cursor is not None and replace_frame_indices is None:
            cursor.dropped.setdefault(group_path, set()).update(missing)
        logger.warning(
            "integrated_%s: skipping %d frame(s) with no result (%s%s); "
            "writing the remaining %d.", "2d", len(missing), missing[:8],
            "..." if len(missing) > 8 else "", len(kept),
        )
        frames = [fr for fr, _i, _r in kept]
        indices = [i for _fr, i, _r in kept]
        results = [r for _fr, _i, r in kept]
    return {
        "entry": entry,
        "group_path": group_path,
        "frames": frames,
        "indices": indices,
        "results": results,
        "cursor": cursor,
        "scan_index": scan.frames.index,
        "is_replace": replace_frame_indices is not None and group_path in h5f,
    }


def _commit_integrated_2d(f, prepared) -> None:
    if prepared is None or not prepared["results"]:
        return
    from xrd_tools.io.nexus import write_integrated_stack
    h5f = _h5(f)
    write_integrated_stack(
        h5f.require_group(prepared["entry"]),
        frame_indices=prepared["indices"],
        results_2d=prepared["results"],
        compression=INTEGRATED_STACK_COMPRESSION,
    )
    _refresh_group_cursor(
        prepared["cursor"], h5f, prepared["group_path"], prepared["scan_index"],
    )
    from xdart.modules.ewald.frame_series import clear_frame_position_cache
    clear_frame_position_cache(h5f.filename)


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

    # Core primitives own the record writes (6a: xrd_tools.io.nexus_record);
    # this function keeps only the GUI-side concerns: the append cursor
    # (skip already-written keys), LiveFrame thumbnail generation/skip
    # policy, and LiveFrame source-path resolution.
    from xrd_tools.io.nexus_record import (
        ensure_frames_container, stamp_source_base, write_frame_record,
    )

    h5f = _h5(f)
    h5_frames = ensure_frames_container(h5f.require_group(entry))
    output_dir = os.path.dirname(os.path.abspath(h5f.filename))
    existing_frame_keys = set(h5_frames.keys())

    # N1: the project root that per-frame source paths are stored relative
    # to (GUI Project Folder); absent -> absolute paths (back-compat).
    # stamp_source_base rejects an append under a DIFFERENT root (P2 #4).
    source_base = stamp_source_base(
        h5f[entry], getattr(scan, "source_base", None) or None
    )

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

        thumb = getattr(frame, "thumbnail", None)
        # PERF-5: for 1D-only (skip_2d) frames whose raw is reloadable from
        # source, don't generate a thumbnail at save time -- the Image Viewer
        # reloads the raw on demand via the per-frame source pointer written
        # just below (load_processed_raw_or_thumbnail tries the source master
        # before any stored thumbnail).  This avoids both the make_thumbnail
        # cost and, after a raw-free, a source reload purely to build a
        # preview.  Matches the batch precompute's frame.can_skip_thumbnail()
        # gate so a precompute-skipped frame is never re-thumbnailed here.  A
        # thumbnail already present (a 2D frame, or a prior reintegration) is
        # still persisted.
        _can_skip_thumb = (
            thumb is None
            and hasattr(frame, "can_skip_thumbnail")
            and frame.can_skip_thumbnail(getattr(scan, "skip_2d", False))
        )
        if thumb is None and not _can_skip_thumb and hasattr(frame, "make_thumbnail"):
            try:
                frame.make_thumbnail(global_mask=getattr(scan, "global_mask", None))
                thumb = getattr(frame, "thumbnail", None)
            except Exception:
                logger.debug(
                    "Failed to generate thumbnail for frame %s", idx,
                    exc_info=True,
                )
        write_frame_record(
            h5_frames, frame_key,
            thumbnail=thumb,
            source_path=_resolved_frame_source(frame, output_dir),
            source_frame_index=_frame_source_index(frame),
            timestamp=getattr(frame, "timestamp", None),
            source_base=source_base,
        )


def _resolved_frame_source(frame, output_dir: str | None) -> str:
    """LiveFrame-side source-path resolution (the record WRITE is the core's
    ``write_frame_source_ref``): absolute paths pass through; relative ones
    resolve via the frame's ``_resolved_source_path`` helper or against the
    output directory."""
    src_path = getattr(frame, "source_file", "") or ""
    if not src_path:
        return ""
    if not os.path.isabs(src_path):
        resolved = ""
        if hasattr(frame, "_resolved_source_path"):
            try:
                resolved = frame._resolved_source_path()
            except Exception:
                resolved = ""
        if resolved and os.path.exists(resolved):
            src_path = resolved
        elif output_dir:
            src_path = os.path.join(output_dir, src_path)
        else:
            src_path = os.path.abspath(src_path)
    return src_path


def _frame_source_index(frame) -> int:
    idx = getattr(frame, "source_frame_idx", None)
    if idx is None:
        idx = getattr(frame, "idx", 0)
    return int(idx)


def _metadata_tail_ids(scan, h5f, entry: str,
                       cursor: NexusWriteCursor) -> list[int] | None:
    """Return the ordered metadata tail, or ``None`` when reconciliation is needed."""
    ids = [int(idx) for idx in scan.frames.index]
    ds_path = f"{entry}/scan_data/frame_index"
    if ds_path not in h5f:
        return ids
    labels_ds = h5f[ds_path]
    n = int(labels_ds.shape[0])
    if n > len(ids):
        return None
    disk_last = int(labels_ds[n - 1]) if n else None
    memory_last = ids[n - 1] if n else None
    memory_sig = _index_structure_signature(scan.frames.index, n)
    if cursor.metadata == (n, disk_last, memory_sig) and memory_last == disk_last:
        return ids[n:]
    disk_ids = [int(x) for x in np.asarray(labels_ds[()]).ravel()]
    if disk_ids != ids[:n]:
        return None
    cursor.metadata = (n, disk_last, memory_sig)
    return ids[n:]


def _refresh_metadata_cursor(cursor: NexusWriteCursor, h5f, entry: str, scan_index=None) -> None:
    path = f"{entry}/scan_data/frame_index"
    if path not in h5f:
        cursor.metadata = None
        return
    labels = h5f[path]
    n = int(labels.shape[0])
    if scan_index is None:
        scan_index = [int(x) for x in np.asarray(labels[()]).ravel()]
    cursor.metadata = (
        n,
        int(labels[n - 1]) if n else None,
        _index_structure_signature(scan_index, n),
    )


def _write_incremental_metadata(f, scan, *, entry: str,
                                cursor: NexusWriteCursor,
                                finalize: bool) -> None:
    """Append metadata rows during acquisition and reconcile on uncertainty."""
    h5f = _h5(f)
    had_scan_data = f"{entry}/scan_data/frame_index" in h5f
    tail_ids = None if finalize else _metadata_tail_ids(scan, h5f, entry, cursor)
    if tail_ids is None:
        _write_scan_metadata(f, scan, entry=entry)
        _write_positioners(f, scan, entry=entry)
        _write_per_frame_geometry(f, scan, entry=entry)
        _refresh_metadata_cursor(cursor, h5f, entry, scan.frames.index)
        return
    if not tail_ids:
        return
    scan_data = getattr(scan, "scan_data", None)
    if scan_data is None:
        return
    rows = scan_data.reindex(tail_ids)
    geom = getattr(scan, "geometry", None)
    from xrd_tools.io.nexus import (
        upsert_per_frame_geometry,
        upsert_positioners,
        upsert_scan_metadata,
    )
    try:
        upsert_scan_metadata(h5f.require_group(entry), rows, tail_ids)
        if geom is not None:
            upsert_positioners(
                h5f.require_group(entry), rows, tail_ids, geom,
                allow_create=not had_scan_data,
            )
            upsert_per_frame_geometry(
                h5f.require_group(entry), rows, tail_ids, geom,
                allow_create=not had_scan_data,
            )
    except (KeyError, TypeError, ValueError):
        logger.debug("Incremental metadata append fell back to replacement",
                     exc_info=True)
        _write_scan_metadata(f, scan, entry=entry)
        _write_positioners(f, scan, entry=entry)
        _write_per_frame_geometry(f, scan, entry=entry)
    _refresh_metadata_cursor(cursor, h5f, entry, scan.frames.index)


def _write_positioners(f, scan, *, entry: str) -> None:
    """Write motor positioners under ``NXsample`` / ``NXdetector``.

    Delegates the layout to
    :func:`xrd_tools.io.nexus.write_positioners`, which reindexes
    ``scan_data`` to the integrated-frame set (so the per-frame
    dimension matches ``integrated_1d``/``2d``) and splits sample- vs
    detector-axis motors via the geometry.  No geometry → no-op.
    """
    from xrd_tools.io.nexus import write_positioners as _wp

    scan_data = getattr(scan, "scan_data", None)
    geom = getattr(scan, "geometry", None)
    frame_index = list(getattr(getattr(scan, "frames", None), "index", []) or [])
    _wp(_h5(f).require_group(entry), scan_data, frame_index, geom)


def _write_scan_metadata(f, scan, *, entry: str) -> None:
    """Persist the full per-frame scan metadata table (delegates to
    :func:`xrd_tools.io.nexus.write_scan_metadata`).

    Unlike positioners (geometry motors only), this stores every column the
    wrangler recorded in ``scan.scan_data`` so a reload restores the same
    metadata the live in-memory scan had — fixes the metadata panel showing
    only the incidence motor after a batch run reloads from disk.
    """
    from xrd_tools.io.nexus import write_scan_metadata as _wsm

    scan_data = getattr(scan, "scan_data", None)
    frame_index = list(getattr(getattr(scan, "frames", None), "index", []) or [])
    _wsm(_h5(f).require_group(entry), scan_data, frame_index)


def _write_per_frame_geometry(f, scan, *, entry: str) -> None:
    """Write derived per-frame pyFAI rotations + incidence angle.

    Delegates to
    :func:`xrd_tools.io.nexus.write_per_frame_geometry`, which
    reindexes ``scan_data`` to the frame set, derives rot1/2/3 +
    incident_angle via ``geometry.derive_per_frame``, and labels the
    rows with the actual frame ids (so a downstream join-by-frame_index
    lines up with ``integrated_1d``).  No geometry / no usable motor
    columns → no-op.
    """
    from xrd_tools.io.nexus import write_per_frame_geometry as _wg

    scan_data = getattr(scan, "scan_data", None)
    geom = getattr(scan, "geometry", None)
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
            from xrd_tools.integrate.calibration import PONI  # type: ignore
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


def _instrument_signature(scan) -> tuple:
    """Fingerprint persisted instrument fields so live changes rewrite them."""
    wavelength = None
    try:
        wavelength = scan.mg_args.get("wavelength")
    except AttributeError:
        pass
    poni = _representative_poni(scan)
    poni_sig = None
    if poni is not None:
        poni_sig = tuple(
            None if getattr(poni, key, None) is None
            else float(getattr(poni, key))
            for key in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")
        )
    return (
        None if wavelength is None else float(wavelength),
        poni_sig,
        _array_digest(getattr(scan, "global_mask", None)),
        _array_digest(getattr(scan, "detector_shape", None)),
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
    # Provenance fix: ``scan.mg_args`` defaults to {'wavelength': 1e-10}
    # (= 1.0 Å) and is never updated from the PONI, so it was being
    # persisted verbatim as wavelength_A=1.0.  Order (T1-4: same helpers +
    # semantics as the display-side display_data._get_wavelength):
    #   1. the integrator built from the PONI (authoritative; 1.0 Å allowed);
    #   2. a wavelength restored from a previously loaded v2 file
    #      (_persisted_wavelength_m — authoritative; covers save-as of a
    #      reloaded scan whose real wavelength is exactly 1.0 Å);
    #   3. mg_args, REJECTING the constructor's 1e-10 m sentinel.
    # When nothing real is available, skip the stamp (a bogus wavelength_A
    # silently corrupts any downstream Q↔2θ).  DEBUG, not WARNING: the
    # initial empty-file save at run start legitimately predates the
    # integrator, so a warning here is routine noise (the per-run saves
    # that follow stamp the real value).
    from xdart.modules.wavelength import normalize_wavelength_m
    wavelength = normalize_wavelength_m(
        getattr(getattr(scan, "_cached_integrator", None), "wavelength", None),
        allow_default_sentinel=True,
    )
    if wavelength is None:
        wavelength = normalize_wavelength_m(
            getattr(scan, "_persisted_wavelength_m", None),
            allow_default_sentinel=True,
        )
    if wavelength is None:
        mg_wl = scan.mg_args.get("wavelength") if scan.mg_args else None
        wavelength = normalize_wavelength_m(mg_wl)
        if wavelength is None and mg_wl is not None:
            logger.debug(
                "Skipping source/wavelength_A: the only candidate is the "
                "mg_args default sentinel / invalid value (%r) and no "
                "integrator or persisted wavelength is available.", mg_wl,
            )
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

    # Full-resolution detector (raw image) shape (H, W) — the shape the flat
    # mask indices index into.  Persisted so a reloaded thumbnail-only scan can
    # map the detector gap mask into thumbnail coordinates without a resident
    # full-res frame.  Additive (NXdetector extra field; external readers ignore
    # it); old files lacking it fall back to the live widget shape cache.
    dshape = getattr(scan, "detector_shape", None)
    if "detector_shape" in det:
        del det["detector_shape"]
    if dshape is not None:
        try:
            sds = det.create_dataset(
                "detector_shape", data=np.asarray(dshape, dtype=np.int64))
            sds.attrs["description"] = "full-resolution detector (raw) shape (H, W)"
        except (TypeError, ValueError):
            logger.debug("could not persist detector_shape %r", dshape)


def _write_stitched(f, scan, *, entry: str) -> None:
    """Write stitched 1D / 2D outputs via the shared primitive.

    Delegates to :func:`xrd_tools.io.nexus.write_stitched` (the
    symmetric counterpart to ``read_stitched``).  Only invoked when
    ``finalize=True`` (typically end-of-scan).

    Note the orientation owned by the primitive: ``stitched_2d`` is
    stored **as-is** ``(n_q, n_chi)`` with dims ``(q, chi)`` — unlike
    the per-frame ``integrated_2d`` stack ``(frame, chi, q)``.
    """
    from xrd_tools.io.nexus import write_stitched as _ws

    s1 = getattr(scan, "stitched_1d", None)
    s2 = getattr(scan, "stitched_2d", None)
    if s1 is None and s2 is None:
        return
    _ws(_h5(f).require_group(entry), stitched_1d=s1, stitched_2d=s2)


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def _quantize_thumbnail(arr, dtype: str = "uint8"):
    """Moved to the core (6a): xrd_tools.io.nexus_record.quantize_thumbnail."""
    from xrd_tools.io.nexus_record import quantize_thumbnail
    return quantize_thumbnail(arr, dtype=dtype)


__all__ = ["NexusWriteCursor", "save_scan_to_nexus"]
