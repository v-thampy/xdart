"""xdart v2 NeXus writer (xdart 0.37+ schema).

This module produces files conforming to the layout described in
``xdart/docs/nexus_stitch_refactor_plan.md`` §2.  The single public
entry point is :func:`save_sphere_to_nexus`, called from
:meth:`EwaldSphere._save_to_nexus`.

Key invariants of the v2 schema:

1. ``/entry/integrated_1d`` and ``/entry/integrated_2d`` are **stacked**
   datasets shape ``(N, nq)`` and ``(N, nchi, nq)`` respectively — never
   per-frame NXdata groups.  Slice-assignment per batch flush; no
   per-frame resize-append.
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

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Union

import h5py
import nexusformat.nexus as nx
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ssrl_xrd_tools.core.geometry import DiffractometerGeometry

    from xdart.modules.ewald.sphere import EwaldSphere


# ---------------------------------------------------------------------------
# File-opening helper — mirrors ssrl_xrd_tools.core.hdf5.catch_h5py_file
# semantics (NFS retry on transient OSError) but goes through
# ``nx.nxopen`` so the returned object is an ``NXroot`` view rather
# than a raw h5py.File.  Underlying h5py.File still reachable via
# ``root.nxfile.file`` for sections that haven't been ported yet.
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

    Used by the (still-unported) section helpers that operate directly
    on raw h5py groups during the incremental migration.  Will go away
    once every helper has been ported to nexusformat assignments.
    """
    return f.nxfile.file


def _assign_nxgroup(f, path: str, value) -> None:
    """Idempotent NXgroup assignment under an NXroot.

    nexusformat refuses to overwrite an existing :class:`NXgroup`
    via ``f[path] = group``; this helper deletes any existing entry
    first so callers can keep the same ``f[path] = NXdata(...)``
    pattern across both first-write and re-write paths.
    """
    if path in f:
        del f[path]
    f[path] = value


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def save_sphere_to_nexus(
    sphere: "EwaldSphere",
    path: Union[str, "Path"],
    *,
    mode: str = "a",
    entry: str = "entry",
    finalize: bool = False,
) -> None:
    """Write ``sphere``'s state into the file at ``path`` as a v2 NXroot.

    Parameters
    ----------
    sphere
        :class:`EwaldSphere` carrying the in-memory state.  Must expose
        ``arches`` (ordered), ``scan_data`` (pandas DataFrame),
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
    """
    import logging
    _logger = logging.getLogger(__name__)
    _verbose = _logger.isEnabledFor(logging.DEBUG)

    def _tick(label, t0):
        if _verbose:
            _logger.debug("save_sphere_to_nexus[%s]: %.3fs",
                          label, time.time() - t0)
        return time.time()

    _t_total = time.time()
    _t0 = time.time()
    with _open_with_retry(path, mode) as f:
        _t0 = _tick("open", _t0)
        _ensure_nxentry(f, entry)
        _t0 = _tick("entry", _t0)
        h5f = _h5(f)

        # 1. Provenance — only write on first save or on finalize.
        # The versions/config/inputs of an in-progress scan don't
        # change between saves; rewriting them every batch costs
        # ~10–30 ms each (importlib.metadata.version() lookups +
        # group-tree delete/recreate via h5py) for no benefit.
        if finalize or "reduction" not in h5f.get(entry, {}):
            _write_reduction(h5f, sphere, entry=entry)
        _t0 = _tick("reduction", _t0)

        # 2. Stacked integrated_1d and integrated_2d
        _write_integrated_1d(f, sphere, entry=entry)
        _t0 = _tick("integrated_1d", _t0)
        _write_integrated_2d(f, sphere, entry=entry)
        _t0 = _tick("integrated_2d", _t0)

        # 3. Per-frame metadata
        _write_per_frame_metadata(f, sphere, entry=entry)
        _t0 = _tick("per_frame_metadata", _t0)

        # 4. Raw motor positioners
        _write_positioners(f, sphere, entry=entry)
        _t0 = _tick("positioners", _t0)

        # 5. Derived per-frame geometry
        _write_per_frame_geometry(f, sphere, entry=entry)
        _t0 = _tick("per_frame_geometry", _t0)

        # 6. Instrument (PONI, wavelength)
        _write_instrument(f, sphere, entry=entry)
        _t0 = _tick("instrument", _t0)

        # 7. Stitched outputs (if present on the sphere)
        if finalize:
            _write_stitched(f, sphere, entry=entry)
            _t0 = _tick("stitched", _t0)

    if _verbose:
        _logger.debug("save_sphere_to_nexus[close+TOTAL]: %.3fs",
                      time.time() - _t_total)


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def _ensure_nxentry(f, entry: str) -> None:
    """Ensure ``/<entry>`` exists as an :class:`NXentry`.

    Uses nx assignment so the resulting group lives in nexusformat's
    in-memory tree and can be navigated (e.g. ``f[entry]``) by
    subsequent ported helpers in the same session.
    """
    if entry not in f:
        f[entry] = nx.NXentry()
    # Ensure NX_class is correctly set (idempotent on rewrites).
    f[entry].attrs["NX_class"] = "NXentry"
    if "default" not in f[entry].attrs:
        f[entry].attrs["default"] = "integrated_1d"


def _write_reduction(h5f, sphere, *, entry: str) -> None:
    """Write /entry/reduction/ via ssrl_xrd_tools provenance."""
    from ssrl_xrd_tools.core.provenance import write_provenance

    config: dict[str, Any] = {
        "bai_1d_args": dict(sphere.bai_1d_args),
        "bai_2d_args": dict(sphere.bai_2d_args),
    }
    if hasattr(sphere, "gi_config") and sphere.gi_config:
        config["gi_config"] = dict(sphere.gi_config)

    # Geometry: stored as a structured subgroup (handled specially in
    # write_provenance), so the convention is human-inspectable in HDF5.
    geom = getattr(sphere, "geometry", None)
    if geom is not None:
        config["geometry"] = {
            "convention": geom.convention,
            "mapping_json": geom.to_json(),
            "motor_sources": {
                m: m for m in geom.all_referenced_motors()
            },
        }

    inputs: dict[str, Any] = {}
    if hasattr(sphere, "raw_files") and sphere.raw_files:
        inputs["raw_files"] = list(sphere.raw_files)
    if hasattr(sphere, "meta_file") and sphere.meta_file:
        inputs["meta_file"] = str(sphere.meta_file)

    write_provenance(
        h5f,
        entry=entry,
        program="xdart",
        config=config,
        inputs=inputs or None,
    )


def _stack_arches(arches, attr: str) -> np.ndarray | None:
    """Stack a per-arch attribute (e.g. ``int_1d.intensity``) into a 2-D array.

    Returns ``None`` when no arch has the attribute populated.
    """
    rows: list[np.ndarray] = []
    for arch in arches:
        obj = arch
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is None:
            return None
        rows.append(np.asarray(obj, dtype=np.float32))
    if not rows:
        return None
    return np.stack(rows, axis=0)


def _ensure_nxdata_group(h5f, path: str, *, signal: str, axes: list[str]):
    """Create or reuse an NXdata h5py group with correct NX_class attrs.

    The integrated_1d / integrated_2d helpers need resizable h5py
    datasets (so per-save cost is O(new rows), not O(total rows)),
    which nexusformat doesn't expose ergonomically.  This helper
    encapsulates the "set the four NXdata attributes" pattern that
    keeps a hand-rolled h5py group NeXus-conformant.
    """
    if path in h5f:
        g = h5f[path]
    else:
        g = h5f.create_group(path)
    g.attrs["NX_class"] = "NXdata"
    g.attrs["signal"] = signal
    g.attrs["axes"] = axes
    return g


def _append_rows(g, name: str, data: np.ndarray,
                 *, maxshape: tuple, chunks: tuple,
                 attrs: Mapping[str, Any] | None = None) -> None:
    """Append/replace rows in a resizable dataset.

    Designed for the "stacked-per-frame" datasets in v2: each call to
    the writer adds K new rows.  We resize the on-disk dataset to
    ``len(data)`` and slice-assign only the rows beyond what's
    already there — so a save with 8 new frames at frame index 100
    only writes 8 rows instead of all 108.

    If the dataset doesn't exist yet, creates it with the supplied
    ``maxshape`` / ``chunks`` parameters so future appends work.
    If the in-memory ``data`` is *shorter* than what's on disk (the
    user re-loaded an older sphere, say), the dataset gets resized
    down and overwritten — that's the safer behavior than silently
    leaving stale rows.
    """
    if name not in g:
        ds = g.create_dataset(
            name, data=data,
            maxshape=maxshape, chunks=chunks,
        )
        if attrs:
            for k, v in attrs.items():
                ds.attrs[k] = v
        return
    ds = g[name]
    target_n = data.shape[0]
    current_n = ds.shape[0]
    if target_n == current_n:
        # Nothing new — most common path after the first save when
        # the in-memory frame count hasn't changed.
        if attrs:
            for k, v in attrs.items():
                ds.attrs[k] = v
        return
    if target_n < current_n:
        # In-memory shrank.  Truncate and rewrite the whole array;
        # safer than leaving stale rows in place.
        ds.resize(data.shape)
        ds[...] = data
    else:
        # Append-only path — the hot loop.  Resize and write only
        # the new tail.
        new_shape = list(ds.shape)
        new_shape[0] = target_n
        ds.resize(tuple(new_shape))
        ds[current_n:target_n, ...] = data[current_n:target_n, ...]
    if attrs:
        for k, v in attrs.items():
            ds.attrs[k] = v


def _write_static_axis(g, name: str, data: np.ndarray,
                       *, attrs: Mapping[str, Any] | None = None) -> None:
    """Write a fixed-shape axis (q, chi) that doesn't grow per frame.

    Idempotent: if the dataset already exists with the same shape it
    is left untouched; on shape change it's deleted and recreated.
    """
    if name in g:
        if g[name].shape == data.shape:
            if attrs:
                for k, v in attrs.items():
                    g[name].attrs[k] = v
            return
        del g[name]
    ds = g.create_dataset(name, data=data)
    if attrs:
        for k, v in attrs.items():
            ds.attrs[k] = v


def _write_integrated_1d(f, sphere, *, entry: str) -> None:
    """Write/extend ``/entry/integrated_1d`` as an NXdata group.

    Uses resizable h5py datasets directly (not nexusformat's NXdata
    constructor) so we can slice-assign only the NEW frames on each
    save.  Format on disk is identical to the all-nexusformat version
    — signal/axes/units attrs match the round-trip test.
    """
    arches = list(sphere.arches)
    if not arches:
        return

    intensity = _stack_arches(arches, "int_1d.intensity")
    if intensity is None:
        return
    radial = np.asarray(arches[0].int_1d.radial, dtype=np.float32)
    sigma = _stack_arches(arches, "int_1d.sigma")
    frame_index = np.array(
        [getattr(a, "idx", i) for i, a in enumerate(arches)], dtype=np.int32
    )
    N, nq = intensity.shape
    # Chunk along the frame axis; ~32 frames per chunk is a decent
    # trade-off between read locality (whole-pattern reads are one
    # chunk for up to 32 frames) and write granularity (we rewrite
    # at most one chunk's worth of frames per save).
    chunks_1d = (min(N, 32), nq)

    h5f = _h5(f)
    g = _ensure_nxdata_group(
        h5f, f"{entry}/integrated_1d",
        signal="intensity",
        axes=["frame_index", "q"],
    )
    _append_rows(g, "intensity", intensity,
                 maxshape=(None, nq), chunks=chunks_1d)
    _write_static_axis(g, "q", radial,
                       attrs={"units": _q_units(arches[0].int_1d)})
    _append_rows(g, "frame_index", frame_index,
                 maxshape=(None,), chunks=(min(N, 32),))
    if sigma is not None:
        _append_rows(g, "sigma", sigma,
                     maxshape=(None, nq), chunks=chunks_1d)


def _write_integrated_2d(f, sphere, *, entry: str) -> None:
    """Write/extend ``/entry/integrated_2d`` as an NXdata group.

    Per-frame intensity comes in as xdart-shape ``(nq, nchi)``; we
    transpose to ``(nchi, nq)`` so the stacked tensor is
    ``(N, nchi, nq)`` — that ordering matches the ``axes`` attribute
    ``["frame_index", "chi", "q"]``.  Resizable datasets keep
    per-save cost O(new rows).
    """
    arches = list(sphere.arches)
    if not arches:
        return

    intensity = _stack_arches(arches, "int_2d.intensity")
    if intensity is None:
        return
    intensity = np.transpose(intensity, (0, 2, 1)) if intensity.ndim == 3 else intensity
    radial = np.asarray(arches[0].int_2d.radial, dtype=np.float32)
    azimuthal = np.asarray(arches[0].int_2d.azimuthal, dtype=np.float32)
    frame_index = np.array(
        [getattr(a, "idx", i) for i, a in enumerate(arches)], dtype=np.int32
    )
    N, nchi, nq = intensity.shape
    # ~8 frames per chunk for 2D — at e.g. 500×500 f32 that's ~8 MB,
    # within h5py's recommended single-chunk write size.
    chunks_2d = (min(N, 8), nchi, nq)

    h5f = _h5(f)
    g = _ensure_nxdata_group(
        h5f, f"{entry}/integrated_2d",
        signal="intensity",
        axes=["frame_index", "chi", "q"],
    )
    _append_rows(g, "intensity", intensity,
                 maxshape=(None, nchi, nq), chunks=chunks_2d)
    _write_static_axis(g, "q", radial,
                       attrs={"units": _q_units(arches[0].int_2d)})
    _write_static_axis(
        g, "chi", azimuthal,
        attrs={"units": getattr(arches[0].int_2d,
                                "azimuthal_unit", "deg")},
    )
    _append_rows(g, "frame_index", frame_index,
                 maxshape=(None,), chunks=(min(N, 32),))


def _write_per_frame_metadata(f, sphere, *, entry: str) -> None:
    """Per-frame thumbnails + source refs as :class:`NXcollection` groups.

    Layout::

        /entry/frames/             NXcollection
            frame_NNNN/            NXcollection
                thumbnail          uint8 (with @vmin, @vmax, @dtype)
                map_raw            float32, optional
                timestamp          str, optional
                source_ref/        NXcollection, optional

    Performance note: per-frame groups are *append-only* during a
    scan — once a frame's thumbnail/metadata is on disk, it doesn't
    change.  We skip frames whose group already exists in the file,
    so subsequent saves are O(new frames) rather than O(total frames).
    The big win on long scans: a single-frame incremental save touches
    one tiny NXcollection instead of N of them.
    """
    arches = list(sphere.arches)
    if not arches:
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

    for arch in arches:
        idx = getattr(arch, "idx", arches.index(arch))
        frame_key = f"frame_{idx:04d}"

        # Skip frames that are already persisted — their per-frame
        # data is immutable post-integration.
        if frame_key in h5_frames:
            continue

        # Build a fresh per-frame NXcollection for new frames.
        fg = nx.NXcollection()

        thumb = getattr(arch, "thumbnail", None)
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
        # arch.map_raw verbatim, which silently dumped 18 MB per
        # Eiger frame into the .nxs (a ~150 ms HDF5 write each, the
        # dominant cost in [SAVE] timing).  Don't bring that back.

        src = getattr(arch, "source_ref", None)
        if isinstance(src, dict):
            sub = nx.NXcollection()
            for k, v in src.items():
                sub[k] = nx.NXfield(v)
            fg["source_ref"] = sub

        ts = getattr(arch, "timestamp", None)
        if ts is not None:
            fg["timestamp"] = nx.NXfield(str(ts))

        frames[frame_key] = fg


def _write_positioners(f, sphere, *, entry: str) -> None:
    """Write motor positioners under ``NXsample`` / ``NXdetector``.

    Layout::

        /entry/sample/                  NXsample
            positioners/                NXcollection
                <motor>/                NXpositioner
                    value               float32, with @units
        /entry/instrument/detector/     NXdetector  (parent class set by
                                        _write_instrument later, but we
                                        seed the detector group here)
            positioners/                NXcollection
                <motor>/                NXpositioner
                    value               float32, with @units

    The split between sample-axis and detector-axis motors comes from
    :class:`DiffractometerGeometry`; if no geometry is configured this
    is a no-op and downstream readers fall back to motor columns in
    ``scan_data``.
    """
    geom = getattr(sphere, "geometry", None)
    scan_data = getattr(sphere, "scan_data", None)
    if scan_data is None or len(scan_data) == 0:
        return

    sample_motors: tuple[str, ...] = (
        tuple(geom.sample_motors) if geom else ()
    )
    detector_motors: tuple[str, ...] = (
        tuple(geom.detector_motors) if geom else ()
    )

    def _build_positioners(motors: tuple[str, ...]) -> nx.NXcollection | None:
        """Build a positioners NXcollection from a motor name set."""
        present = [m for m in motors if m in scan_data.columns]
        if not present:
            return None
        coll = nx.NXcollection()
        for motor in present:
            pg = nx.NXpositioner()
            pg["value"] = nx.NXfield(
                np.asarray(scan_data[motor].values, dtype=np.float32),
                attrs={"units": "deg"},
            )
            coll[motor] = pg
        return coll

    # ── /entry/sample ─────────────────────────────────────────────
    sample_coll = _build_positioners(sample_motors)
    if sample_coll is not None:
        sample_path = f"{entry}/sample"
        if sample_path not in f:
            f[sample_path] = nx.NXsample()
        # ensure NX_class is right even if the group was created earlier
        f[sample_path].attrs["NX_class"] = "NXsample"
        # Replace positioners atomically (idempotent).
        if "positioners" in f[sample_path]:
            del f[sample_path]["positioners"]
        f[sample_path]["positioners"] = sample_coll

    # ── /entry/instrument/detector ────────────────────────────────
    # _write_instrument runs after us and also writes /instrument/detector
    # so we just make sure the path exists with the right NX_classes and
    # attach the positioners.  Detector is NXdetector, not NXinstrument
    # (the old code wrongly stamped this NX_class — fixed here as part
    # of the typed-constructor port).
    det_coll = _build_positioners(detector_motors)
    if det_coll is not None:
        instr_path = f"{entry}/instrument"
        if instr_path not in f:
            f[instr_path] = nx.NXinstrument()
        f[instr_path].attrs["NX_class"] = "NXinstrument"
        det_path = f"{instr_path}/detector"
        if "detector" not in f[instr_path]:
            f[instr_path]["detector"] = nx.NXdetector()
        f[det_path].attrs["NX_class"] = "NXdetector"
        if "positioners" in f[det_path]:
            del f[det_path]["positioners"]
        f[det_path]["positioners"] = det_coll


def _write_per_frame_geometry(f, sphere, *, entry: str) -> None:
    """Write derived per-frame pyFAI rotations + incidence angle.

    Layout::

        /entry/per_frame_geometry/      NXcollection
            frame_index                 int32 (N,)
            rot1                        float32 (N,), rad
            rot2                        float32 (N,), rad
            rot3                        float32 (N,), rad
            incident_angle              float32 (N,), deg  (optional)

    Computed from ``sphere.geometry.derive_per_frame(motors)`` —
    motors come from ``scan_data`` columns.  Stored as an
    NXcollection (not NXdata) because the multiple derived arrays
    don't share a single "signal"; viewers should pick whichever
    field they care about explicitly.
    """
    geom = getattr(sphere, "geometry", None)
    scan_data = getattr(sphere, "scan_data", None)
    if geom is None or scan_data is None or len(scan_data) == 0:
        return

    referenced = geom.all_referenced_motors()
    motors = {
        m: np.asarray(scan_data[m].values, dtype=float)
        for m in referenced
        if m in scan_data.columns
    }
    if not motors:
        return

    try:
        derived = geom.derive_per_frame(motors)
    except Exception:
        # If any active source motor is missing in scan_data we silently
        # skip — the geometry config blob is still persisted via
        # /reduction/config/geometry, so the user can re-derive later.
        return

    coll = nx.NXcollection()
    coll["frame_index"] = nx.NXfield(np.arange(len(scan_data), dtype=np.int32))
    for key, arr in derived.items():
        units = "deg" if key == "incident_angle" else "rad"
        coll[key] = nx.NXfield(arr.astype(np.float32), attrs={"units": units})
    _assign_nxgroup(f, f"{entry}/per_frame_geometry", coll)


def _write_instrument(f, sphere, *, entry: str) -> None:
    """Write :class:`NXinstrument` with :class:`NXsource` + :class:`NXdetector`.

    Coexists carefully with :func:`_write_positioners`, which may have
    already created ``/entry/instrument`` and ``/entry/instrument/
    detector``.  We don't replace those groups — we either create them
    if missing, or just stamp attrs and append child fields.  Replacing
    them would clobber the positioners we just wrote.

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
                                        _write_positioners; preserved
                                        across rewrites)
    """
    # ── /entry/instrument (NXinstrument) ──────────────────────────
    instr_path = f"{entry}/instrument"
    if instr_path not in f:
        f[instr_path] = nx.NXinstrument()
    f[instr_path].attrs["NX_class"] = "NXinstrument"
    instr = f[instr_path]

    # ── source (NXsource) ─────────────────────────────────────────
    wavelength = sphere.mg_args.get("wavelength")
    if wavelength is not None:
        src = nx.NXsource()
        src["wavelength_A"] = nx.NXfield(float(wavelength) * 1e10)
        if "source" in instr:
            del instr["source"]
        instr["source"] = src

    # ── detector (NXdetector) ─────────────────────────────────────
    det_path = f"{instr_path}/detector"
    if "detector" not in instr:
        instr["detector"] = nx.NXdetector()
    f[det_path].attrs["NX_class"] = "NXdetector"
    det = f[det_path]

    # PONI scalars (when arches exist with a poni attached)
    arches = list(sphere.arches)
    if arches:
        poni = getattr(arches[0], "poni", None)
        if poni is not None:
            for k in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3"):
                v = getattr(poni, k, None)
                if v is not None:
                    if k in det:
                        del det[k]
                    det[k] = nx.NXfield(float(v))

    # Global mask — flat indices of masked pixels (detector mask + the
    # user-supplied Mask File, combined via the wrangler).  Stored so
    # the viewer can overlay the mask without needing the original
    # mask file alongside the .nxs.
    gmask = getattr(sphere, "global_mask", None)
    if gmask is not None:
        try:
            arr = np.asarray(gmask, dtype=np.int64)
        except (TypeError, ValueError):
            arr = None
        if arr is not None and arr.size > 0:
            if "mask" in det:
                del det["mask"]
            det["mask"] = nx.NXfield(
                arr,
                attrs={"description": "flat pixel indices, shape (N,)"},
            )


def _write_stitched(f, sphere, *, entry: str) -> None:
    """Write stitched 1D / 2D outputs as :class:`NXdata` groups.

    Only invoked when ``finalize=True`` is passed to
    :func:`save_sphere_to_nexus` (typically end-of-scan).  Each
    stitched result is the combined pattern across all arches in the
    sphere, produced via ``ssrl_xrd_tools.integrate.multi.stitch_*``.

    Layout::

        /entry/stitched_1d/             NXdata
            intensity                   float32 (nq,)        @signal
            q                           float32 (nq,)        @units
            sigma                       float32 (nq,)        optional
        /entry/stitched_2d/             NXdata
            intensity                   float32 (nchi, nq)   @signal
            q                           float32 (nq,)        @units
            chi                         float32 (nchi,)      @units
    """
    s1 = getattr(sphere, "stitched_1d", None)
    if s1 is not None:
        nxdata = nx.NXdata(
            signal=nx.NXfield(np.asarray(s1.intensity, dtype=np.float32),
                              name="intensity"),
            axes=[nx.NXfield(np.asarray(s1.radial, dtype=np.float32),
                             name="q", units=_q_units(s1))],
        )
        if getattr(s1, "sigma", None) is not None:
            nxdata["sigma"] = nx.NXfield(
                np.asarray(s1.sigma, dtype=np.float32), name="sigma",
            )
        _assign_nxgroup(f, f"{entry}/stitched_1d", nxdata)

    s2 = getattr(sphere, "stitched_2d", None)
    if s2 is not None:
        nxdata = nx.NXdata(
            signal=nx.NXfield(np.asarray(s2.intensity, dtype=np.float32),
                              name="intensity"),
            axes=[
                nx.NXfield(np.asarray(s2.azimuthal, dtype=np.float32),
                           name="chi",
                           units=getattr(s2, "azimuthal_unit", "deg")),
                nx.NXfield(np.asarray(s2.radial, dtype=np.float32),
                           name="q", units=_q_units(s2)),
            ],
        )
        _assign_nxgroup(f, f"{entry}/stitched_2d", nxdata)


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def _q_units(result) -> str:
    """Pull a ``q`` unit string out of an IntegrationResult, with fallback."""
    unit = getattr(result, "unit", "") or ""
    if "A^-1" in unit:
        return "1/angstrom"
    if "nm^-1" in unit:
        return "1/nm"
    return unit or "1/angstrom"


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


__all__ = ["save_sphere_to_nexus"]
