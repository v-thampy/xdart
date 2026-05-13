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
    with _open_with_retry(path, mode) as f:
        # ``f`` is an NXroot view; ``_h5(f)`` reaches the underlying
        # h5py.File for section helpers that haven't been ported yet.
        # Mixing nx-assignment and h5py writes is safe (both target the
        # same on-disk file) but nx's in-memory cache won't reflect
        # changes made directly via h5py — so anything that needs to be
        # *visible to subsequent nx assignments* must go through nx.
        _ensure_nxentry(f, entry)
        h5f = _h5(f)

        # 1. Provenance (write once; idempotent re-writes are safe)
        _write_reduction(h5f, sphere, entry=entry)

        # 2. Stacked integrated_1d and integrated_2d — ported to nexusformat
        _write_integrated_1d(f, sphere, entry=entry)
        _write_integrated_2d(f, sphere, entry=entry)

        # 3. Per-frame metadata (thumbnails + source refs)
        _write_per_frame_metadata(h5f, sphere, entry=entry)

        # 4. Raw motor positioners
        _write_positioners(h5f, sphere, entry=entry)

        # 5. Derived per-frame geometry
        _write_per_frame_geometry(h5f, sphere, entry=entry)

        # 6. Instrument (PONI, wavelength) — write each call (cheap, scalars)
        _write_instrument(h5f, sphere, entry=entry)

        # 7. Stitched outputs (if present on the sphere)
        if finalize:
            _write_stitched(h5f, sphere, entry=entry)


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


def _write_integrated_1d(f, sphere, *, entry: str) -> None:
    """Write ``/entry/integrated_1d`` as an :class:`NXdata` group.

    Uses nexusformat constructors so ``signal``, ``axes``, ``units``
    and ``NX_class`` are set by the library, not by hand — the only
    way to instantiate :class:`NXdata` is to name a signal and its
    axes, which makes "forgot the signal attr" a compile-time impossi-
    bility instead of a silent on-disk bug.

    Assignment to ``f[path]`` replaces the existing group atomically,
    so the same idempotent semantics as ``_replace_ds`` apply.
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

    # NXdata's signature is (signal, axes, errors, weights, *args, **kwargs)
    # where signal/axes are NXfield instances.  The library writes the
    # appropriate @signal and @axes attrs based on the field names.
    # We attach sigma as a sibling field (named "sigma", not
    # "intensity_errors" — preserve the established on-disk contract).
    nxdata = nx.NXdata(
        signal=nx.NXfield(intensity, name="intensity"),
        axes=[
            nx.NXfield(frame_index, name="frame_index"),
            nx.NXfield(radial, name="q",
                       units=_q_units(arches[0].int_1d)),
        ],
    )
    if sigma is not None:
        nxdata["sigma"] = nx.NXfield(sigma, name="sigma")
    _assign_nxgroup(f, f"{entry}/integrated_1d", nxdata)


def _write_integrated_2d(f, sphere, *, entry: str) -> None:
    """Write ``/entry/integrated_2d`` as an :class:`NXdata` group.

    Per-frame intensity comes in as xdart-shape ``(nq, nchi)``; we
    transpose to ``(nchi, nq)`` so the stacked tensor is
    ``(N, nchi, nq)`` — that ordering matches the ``axes`` attribute
    ``["frame_index", "chi", "q"]`` exactly, which is what NeXus-aware
    viewers (DAWN, Mantid) expect for auto-plot.
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

    nxdata = nx.NXdata(
        signal=nx.NXfield(intensity, name="intensity"),
        axes=[
            nx.NXfield(frame_index, name="frame_index"),
            nx.NXfield(azimuthal, name="chi",
                       units=getattr(arches[0].int_2d,
                                     "azimuthal_unit", "deg")),
            nx.NXfield(radial, name="q",
                       units=_q_units(arches[0].int_2d)),
        ],
    )
    _assign_nxgroup(f, f"{entry}/integrated_2d", nxdata)


def _write_per_frame_metadata(h5f, sphere, *, entry: str) -> None:
    """Per-frame thumbnails + source refs.  No integrated arrays here."""
    arches = list(sphere.arches)
    if not arches:
        return

    g = h5f.require_group(f"{entry}/frames")
    g.attrs["NX_class"] = "NXcollection"
    for arch in arches:
        idx = getattr(arch, "idx", arches.index(arch))
        fg = g.require_group(f"frame_{idx:04d}")
        fg.attrs["NX_class"] = "NXcollection"

        # thumbnail: uncompressed uint8 (or uint16 if requested)
        thumb = getattr(arch, "thumbnail", None)
        if thumb is not None:
            arr, lut = _quantize_thumbnail(thumb)
            if "thumbnail" in fg:
                del fg["thumbnail"]
            ds = fg.create_dataset("thumbnail", data=arr)
            ds.attrs["vmin"] = lut[0]
            ds.attrs["vmax"] = lut[1]
            ds.attrs["dtype"] = lut[2]

        # map_raw heatmap — optional small reduced 2D
        map_raw = getattr(arch, "map_raw_thumb", None) or getattr(arch, "map_raw", None)
        if isinstance(map_raw, np.ndarray) and map_raw.ndim == 2:
            _replace_ds(fg, "map_raw", map_raw.astype(np.float32))

        # source_ref
        src = getattr(arch, "source_ref", None)
        if isinstance(src, dict):
            sub = fg.require_group("source_ref")
            for k, v in src.items():
                _replace_ds(sub, k, v)

        # timestamp
        ts = getattr(arch, "timestamp", None)
        if ts is not None:
            _replace_ds(fg, "timestamp", str(ts))


def _write_positioners(h5f, sphere, *, entry: str) -> None:
    """Write /entry/{sample,instrument/detector}/positioners/<motor>/."""
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

    def write_set(category_path: str, motors: tuple[str, ...]) -> None:
        if not motors:
            return
        cat = h5f.require_group(f"{entry}/{category_path}")
        cat.attrs["NX_class"] = (
            "NXsample" if category_path == "sample" else "NXinstrument"
        )
        pos = cat.require_group("positioners")
        pos.attrs["NX_class"] = "NXcollection"
        for motor in motors:
            if motor not in scan_data.columns:
                continue
            pg = pos.require_group(motor)
            pg.attrs["NX_class"] = "NXpositioner"
            _replace_ds(
                pg, "value",
                np.asarray(scan_data[motor].values, dtype=np.float32),
                attrs={"units": "deg"},
            )

    write_set("sample", sample_motors)
    write_set("instrument/detector", detector_motors)


def _write_per_frame_geometry(h5f, sphere, *, entry: str) -> None:
    """Derive rot1/rot2/rot3/incident_angle from positioners + geometry."""
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

    g = h5f.require_group(f"{entry}/per_frame_geometry")
    g.attrs["NX_class"] = "NXdata"
    for key, arr in derived.items():
        units = "deg" if key == "incident_angle" else "rad"
        _replace_ds(g, key, arr.astype(np.float32), attrs={"units": units})
    _replace_ds(
        g, "frame_index",
        np.arange(len(scan_data), dtype=np.int32),
    )


def _write_instrument(h5f, sphere, *, entry: str) -> None:
    """Write PONI / wavelength / energy + detector basics + mask."""
    instr = h5f.require_group(f"{entry}/instrument")
    instr.attrs["NX_class"] = "NXinstrument"

    wavelength = sphere.mg_args.get("wavelength")
    if wavelength is not None:
        src = instr.require_group("source")
        src.attrs["NX_class"] = "NXsource"
        _replace_ds(src, "wavelength_A", float(wavelength) * 1e10)

    det = instr.require_group("detector")
    det.attrs["NX_class"] = "NXdetector"

    # PONI scalars (when arches exist with a poni attached)
    arches = list(sphere.arches)
    if arches:
        poni = getattr(arches[0], "poni", None)
        if poni is not None:
            for k in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3"):
                v = getattr(poni, k, None)
                if v is not None:
                    _replace_ds(det, k, float(v))

    # Global mask — flat indices of masked pixels (detector mask + the
    # user-supplied Mask File, combined via the wrangler).  Stored so
    # the viewer can overlay the mask without needing the original
    # mask file alongside the .nxs.
    gmask = getattr(sphere, "global_mask", None)
    if gmask is not None:
        try:
            arr = np.asarray(gmask, dtype=np.int64)
            if arr.size > 0:
                _replace_ds(det, "mask", arr,
                            attrs={"description": "flat pixel indices, "
                                   "shape (N,)"})
        except Exception:
            pass


def _write_stitched(h5f, sphere, *, entry: str) -> None:
    """Write /entry/stitched_1d and /stitched_2d if present on sphere."""
    s1 = getattr(sphere, "stitched_1d", None)
    if s1 is not None:
        g = h5f.require_group(f"{entry}/stitched_1d")
        g.attrs["NX_class"] = "NXdata"
        _replace_ds(g, "intensity", np.asarray(s1.intensity, dtype=np.float32))
        _replace_ds(g, "q", np.asarray(s1.radial, dtype=np.float32),
                    attrs={"units": _q_units(s1)})
        if getattr(s1, "sigma", None) is not None:
            _replace_ds(g, "sigma", np.asarray(s1.sigma, dtype=np.float32))

    s2 = getattr(sphere, "stitched_2d", None)
    if s2 is not None:
        g = h5f.require_group(f"{entry}/stitched_2d")
        g.attrs["NX_class"] = "NXdata"
        _replace_ds(g, "intensity", np.asarray(s2.intensity, dtype=np.float32))
        _replace_ds(g, "q", np.asarray(s2.radial, dtype=np.float32),
                    attrs={"units": _q_units(s2)})
        _replace_ds(g, "chi", np.asarray(s2.azimuthal, dtype=np.float32),
                    attrs={"units": getattr(s2, "azimuthal_unit", "deg")})


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def _replace_ds(grp, name: str, data, attrs: Mapping[str, Any] | None = None) -> None:
    """Idempotent dataset write: delete-if-exists, then create."""
    if name in grp:
        del grp[name]
    if isinstance(data, np.ndarray):
        ds = grp.create_dataset(name, data=data)
    else:
        ds = grp.create_dataset(name, data=data)
    if attrs:
        for k, v in attrs.items():
            ds.attrs[k] = v


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
