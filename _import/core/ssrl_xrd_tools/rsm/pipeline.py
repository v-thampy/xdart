from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

import numpy as np

from ssrl_xrd_tools.core.geometry import PixelQMap
from ssrl_xrd_tools.io.image import (
    read_image,
    read_image_stack,
    read_images_parallel,
    find_image_files,
    get_detector_mask,
    apply_rotation,
)
from ssrl_xrd_tools.io.spec import (
    get_scan_path_info,
    get_energy_and_UB,
    get_angles,
)
from ssrl_xrd_tools.rsm.volume import RSMVolume
from ssrl_xrd_tools.rsm.gridding import (
    StreamingGridder,
    grid_img_data,
    grid_img_data_streaming,
)

logger = logging.getLogger(__name__)


def _as_path(path: Path | str) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _load_pickle(path: Path) -> Any | None:
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        logger.exception("Failed to load pickle: %s", path)
        return None


def _save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


@dataclass(slots=True)
class ScanInfo:
    spec_path: Path
    img_dir: Path
    h5_path: Path | None = None


from ssrl_xrd_tools.core.config import ExperimentConfig


def load_images(
    scan_name: str,
    scan_info: ScanInfo,
    rotation: int = 0,
    parallel: bool = True,
    detector: str = "Pilatus300k",
) -> np.ndarray | None:
    """Load image stack for a scan (HDF5 or individual image files)."""
    mask = get_detector_mask(detector)
    if scan_info.h5_path:
        return read_image_stack(
            scan_info.h5_path,
            mask=mask,
            rotation=rotation,
        )
    spec_name, scan_num = get_scan_path_info(scan_name)
    img_files = find_image_files(
        scan_info.img_dir,
        stem=f"_{spec_name}_scan{scan_num[:-2]}_",
    )
    if not img_files:
        logger.warning(
            "No image files found for scan %s in %s",
            scan_name, scan_info.img_dir,
        )
        return None
    if parallel and len(img_files) > 1:
        return read_images_parallel(
            img_files, rotation=rotation, mask=mask,
        )
    return np.stack([
        read_image(f, mask=mask, rotation=rotation)
        for f in img_files
    ])


def process_scan_data(
    scan_name: str,
    scan_info: ScanInfo,
    mapper: PixelQMap,
    diff_motors: list[str] | tuple[str, ...],
    bins: tuple[int, int, int],
    rotation: int = 0,
    roi: tuple[int, int, int, int] | None = None,
    parallel: bool = True,
    strict: bool = False,
    detector: str = "Pilatus300k",
    *,
    streaming: bool = True,
    chunk_size: int = 8,
    static_mask: np.ndarray | None = None,
) -> RSMVolume | None:
    """
    Pure processing path without cache I/O.

    Parameters
    ----------
    streaming : bool, default True
        When True, dispatch to :func:`grid_img_data_streaming` — the
        ``xu.Gridder3D`` accumulates chunks of ``chunk_size`` frames so
        only ``chunk_size × H × W × 32`` bytes of q + image memory are
        held at any time (regardless of how many frames the scan has).
        When False, fall back to the single-shot
        :func:`grid_img_data` which materialises the full ``(N, H, W)``
        stack plus the full ``(N, H, W)`` qx/qy/qz arrays — OOM-prone
        for large scans.
    chunk_size : int, default 8
        Frames per chunk; only consulted when ``streaming=True``.
    static_mask : ndarray of bool, optional
        2D detector / hot-pixel mask applied per-chunk before gridding.
        Only consulted when ``streaming=True``.

    Note
    ----
    Image loading is still eager here (``load_images`` returns the full
    stack into memory).  Streaming bounds the *gridding-side* memory
    cost (3 q-arrays × stack size), not the image-stack memory.  For
    fully lazy loading + streaming, use
    :func:`process_scan_from_sphere` on a v2 NeXus sphere where each
    arch's raw image is loaded one chunk at a time.
    """
    spec_file = scan_info.spec_path
    _, scan_num = get_scan_path_info(scan_name)

    try:
        energy, UB = get_energy_and_UB(spec_file, scan_num)
        angles = get_angles(spec_file, scan_num, diff_motors)
        img_arr = load_images(
            scan_name, scan_info,
            rotation=rotation,
            parallel=parallel,
            detector=detector,
        )

        if img_arr is None or img_arr.size == 0:
            logger.warning("No image data for scan %s", scan_name)
            return None

        if streaming:
            return grid_img_data_streaming(
                mapper,
                img_arr,
                angles,
                energy,
                UB=UB,
                bins=bins,
                chunk_size=chunk_size,
                roi=roi,
                static_mask=static_mask,
            )
        return grid_img_data(
            mapper,
            img_arr,
            angles,
            energy,
            UB=UB,
            bins=bins,
            roi=roi,
        )
    except Exception:
        logger.exception("Error processing scan %s", scan_name)
        if strict:
            raise
        return None


def process_scan(
    scan_name: str,
    scan_info: ScanInfo,
    bins: tuple[int, int, int],
    mapper: PixelQMap,
    diff_motors: list[str] | tuple[str, ...],
    pickle_dir: Path,
    rotation: int = 0,
    reprocess: bool = False,
    roi: tuple[int, int, int, int] | None = None,
    parallel: bool = True,
    strict: bool = False,
    detector: str = "Pilatus300k",
    *,
    streaming: bool = True,
    chunk_size: int = 8,
    static_mask: np.ndarray | None = None,
) -> RSMVolume | None:
    """
    Process one scan to an RSMVolume, using cache if available.

    ``streaming`` defaults to True (the safe / memory-bounded path);
    pass ``streaming=False`` to use the legacy single-shot gridder for
    back-compat / small scans.  See :func:`process_scan_data` for the
    full parameter contract.
    """
    sample_name, scan_num = get_scan_path_info(scan_name)
    pickle_file = pickle_dir / f"{sample_name}_{scan_num[:-2]}.pkl"

    if pickle_file.is_file() and not reprocess:
        cached = _load_pickle(pickle_file)
        if isinstance(cached, RSMVolume):
            return cached
        if isinstance(cached, tuple) and len(cached) == 4:
            return RSMVolume(
                h=np.asarray(cached[0]),
                k=np.asarray(cached[1]),
                l=np.asarray(cached[2]),
                intensity=np.asarray(cached[3]),
            )

    volume = process_scan_data(
        scan_name=scan_name,
        scan_info=scan_info,
        mapper=mapper,
        diff_motors=diff_motors,
        bins=bins,
        rotation=rotation,
        roi=roi,
        parallel=parallel,
        strict=strict,
        detector=detector,
        streaming=streaming,
        chunk_size=chunk_size,
        static_mask=static_mask,
    )

    if volume is not None:
        _save_pickle(pickle_file, volume)

    return volume


# ---------------------------------------------------------------------------
# v2 NeXus sphere as a data source
# ---------------------------------------------------------------------------
#
# Lets the RSM pipeline consume an xdart v2 :class:`EwaldSphere` directly,
# so a scan that's already been through xdart's 1D/2D integration can be
# re-used as the RSM input without re-parsing SPEC + raw image files.
#
# Per-sphere quantities pulled here:
#
# * Energy (eV)  — from ``sphere.mg_args['wavelength']`` (metres) unless
#   the caller passes ``energy=`` explicitly.
# * Per-frame motor positions — ``sphere.scan_data[motor].values``, indexed
#   by the frame IDs in ``sphere.arches.index``.
# * Raw images — one chunk at a time via ``arch._lazy_load_raw()``; arches
#   are released after each chunk so peak memory stays at chunk_size frames.
#
# UB is **not** in the v2 NeXus schema yet (xdart's integration pipeline
# doesn't need it); pass it in via ``UB=``.  When the schema grows a
# ``/entry/sample/UB`` field we'll auto-resolve it the same way as energy.
#
# To avoid a circular import (ssrl_xrd_tools is below xdart in the stack)
# the sphere is duck-typed against the protocol below.


class _ArchLike(Protocol):
    """Minimal arch interface needed by the v2-sphere RSM path."""
    idx: int
    map_raw: np.ndarray | None
    def _lazy_load_raw(self) -> bool: ...


class _ArchSeriesLike(Protocol):
    """Minimal ArchSeries interface — index + lazy __getitem__."""
    index: list[int]
    def __getitem__(self, idx: int) -> _ArchLike: ...


class _SphereLike(Protocol):
    """Minimal EwaldSphere interface for RSM v2-sphere processing."""
    scan_data: Any                       # pandas.DataFrame
    arches: _ArchSeriesLike
    mg_args: dict[str, Any]


def _energy_from_sphere(sphere: _SphereLike) -> float:
    """Resolve X-ray energy in eV from the sphere's stored wavelength.

    ``sphere.mg_args['wavelength']`` is in metres (pyFAI convention),
    so ``E = h c / λ`` simplifies to ``E_eV = 12398 / λ_Å``.
    """
    wavelength_m = sphere.mg_args.get("wavelength")
    if not wavelength_m or wavelength_m <= 0:
        raise ValueError(
            "sphere has no usable wavelength in mg_args; "
            "pass energy= explicitly."
        )
    return 12398.0 / (float(wavelength_m) * 1e10)


def _angles_for_indices(
    sphere: _SphereLike,
    diff_motors: list[str] | tuple[str, ...],
    indices: list[int] | None = None,
) -> list[np.ndarray]:
    """Pull per-frame motor arrays from ``sphere.scan_data``.

    Returns a list aligned with ``diff_motors``: one ndarray per motor,
    each of length ``len(indices)`` (or len(scan_data) if indices is None).
    Raises :class:`KeyError` if any motor is missing from the DataFrame.
    """
    cols = list(sphere.scan_data.columns)
    missing = [m for m in diff_motors if m not in cols]
    if missing:
        raise KeyError(
            f"motors {missing!r} not in sphere.scan_data (have {cols!r})"
        )
    if indices is None:
        return [
            np.asarray(sphere.scan_data[m].values, dtype=float)
            for m in diff_motors
        ]
    return [
        np.asarray(sphere.scan_data.loc[indices, m].values, dtype=float)
        for m in diff_motors
    ]


def _iter_sphere_chunks(
    sphere: _SphereLike,
    chunk_size: int,
) -> Iterator[tuple[np.ndarray, list[int]]]:
    """Yield ``(img_chunk, frame_indices)`` for the streaming gridder.

    Each ``img_chunk`` is ``(n_chunk, H, W)`` — a fresh ``np.stack``,
    independent of the source arches.  Memory promise:

    1. **Only ``chunk_size`` raw frames are resident at any time.**
       Frames materialised by this iterator via ``_lazy_load_raw`` are
       cleared (``arch.map_raw = None``) in a ``finally`` block after
       the consumer is done with each chunk.  Without this, the
       ``sphere.arches`` cache would hold every lazy-loaded frame in
       memory and the "memory-bounded" promise of streaming would
       degrade to "full-scan resident" for any user who actually has
       v2-reloaded arches.
    2. **Arches that arrived with ``map_raw`` already populated are
       left alone.**  We only free what we ourselves loaded — caller-
       owned data is not our responsibility to invalidate.
    3. **The stacked chunk is a copy** (``np.stack`` allocates fresh
       output), so it survives the per-arch clearing.
    """
    indices = list(sphere.arches.index)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0; got {chunk_size}")

    for start in range(0, len(indices), chunk_size):
        chunk_indices = indices[start : start + chunk_size]
        frames: list[np.ndarray] = []
        arches: list[_ArchLike] = []
        was_missing: list[bool] = []
        try:
            for idx in chunk_indices:
                arch = sphere.arches[idx]
                missing = arch.map_raw is None
                arches.append(arch)
                was_missing.append(missing)
                if missing:
                    ok = arch._lazy_load_raw()
                    if not ok or arch.map_raw is None:
                        raise RuntimeError(
                            f"could not lazy-load raw frame for arch {idx} "
                            f"(check source_file / source_frame_idx "
                            f"provenance)"
                        )
                frames.append(np.asarray(arch.map_raw))
            yield np.stack(frames, axis=0), chunk_indices
        finally:
            # Free only what we materialised ourselves; respect arches
            # that arrived with map_raw already populated.
            for arch, missing in zip(arches, was_missing):
                if missing:
                    arch.map_raw = None


def process_scan_from_sphere(
    sphere: _SphereLike,
    mapper: PixelQMap,
    diff_motors: list[str] | tuple[str, ...],
    bins: tuple[int, int, int],
    *,
    UB: np.ndarray | None = None,
    energy: float | None = None,
    chunk_size: int = 8,
    q_bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ] | None = None,
    roi: tuple[int, int, int, int] | None = None,
    static_mask: np.ndarray | None = None,
    scout_pad: float = 0.0,
) -> RSMVolume:
    """Stream-process a v2 :class:`EwaldSphere` into an :class:`RSMVolume`.

    Reads per-frame motor positions from ``sphere.scan_data``, energy
    from ``sphere.mg_args['wavelength']`` (unless ``energy=`` is given),
    and raw images one chunk at a time via ``arch._lazy_load_raw``.
    All frames flow through a single :class:`StreamingGridder` so peak
    memory is bounded by ``chunk_size`` regardless of total frame count.

    Parameters
    ----------
    sphere : EwaldSphere (duck-typed)
        Must expose ``scan_data`` (pandas DataFrame indexed by frame IDs),
        ``arches`` (indexable by frame ID, yielding objects with
        ``map_raw`` + ``_lazy_load_raw``), and ``mg_args`` (dict with
        ``wavelength`` in metres).
    mapper : PixelQMap
        Diffractometer convention + detector header.
    diff_motors : sequence of str
        Motor column names in ``sphere.scan_data`` to feed into
        ``xu.QConversion`` (sample axes first, then detector axes).
    bins : tuple of int
        ``xu.Gridder3D`` bin counts.
    UB : (3, 3) ndarray, optional
        Sample orientation matrix.  v2 NeXus doesn't store this yet —
        pass it explicitly until the schema is extended.
    energy : float, optional
        X-ray energy in eV.  ``None`` → resolved from the sphere.
    chunk_size : int
        Frames per chunk handed to :class:`StreamingGridder`.  Default 8.
    q_bounds : ((qx_lo, qx_hi), (qy_lo, qy_hi), (qz_lo, qz_hi)), optional
        Explicit grid bounds.  ``None`` → scout pass over the angles +
        detector corners (no raw image data is read for scouting).
    roi : (r0, r1, c0, c1), optional
        Detector ROI applied per-chunk and to the mapper header.
    static_mask : ndarray of bool, optional
        2D mask applied to every chunk before gridding — use this for
        a detector / hot-pixel mask that should be consistent across
        the whole scan.  See :meth:`StreamingGridder.add` for why this
        replaces the chunk-local std heuristic that earlier versions
        used.
    scout_pad : float
        Padding fraction applied to scouted bounds.  Ignored when
        ``q_bounds`` is supplied.

    Returns
    -------
    RSMVolume
        Gridded reciprocal-space volume.
    """
    if energy is None:
        energy = _energy_from_sphere(sphere)
    angles_full = _angles_for_indices(sphere, diff_motors)

    sg = StreamingGridder(mapper, bins)
    if q_bounds is None:
        # Scout uses the full angle arrays + the configured detector
        # size.  No raw image is loaded.
        sg.scout(
            [(angles_full, energy, UB,
              (mapper.header.Nch1, mapper.header.Nch2))],
            roi=roi,
            pad=scout_pad,
        )
    else:
        sg.set_bounds(*q_bounds)

    for img_chunk, chunk_indices in _iter_sphere_chunks(sphere, chunk_size):
        angles_chunk = _angles_for_indices(sphere, diff_motors, chunk_indices)
        sg.add(
            img_chunk,
            angles_chunk,
            energy,
            UB=UB,
            roi=roi,
            static_mask=static_mask,
        )
    return sg.to_volume()


@dataclass(frozen=True)
class SphereInput:
    """One scan's data for :func:`grid_spheres_streaming`.

    ``energy`` defaults to ``None`` so the per-sphere resolver kicks in;
    pass an explicit value to override.  ``UB`` and ``roi`` are
    per-sphere and may differ across the list.
    """
    sphere: Any  # _SphereLike — typed as Any to keep the dataclass simple
    energy: float | None = None
    UB: np.ndarray | None = None
    roi: tuple[int, int, int, int] | None = None


def grid_spheres_streaming(
    mapper: PixelQMap,
    sphere_inputs: Iterable[SphereInput],
    diff_motors: list[str] | tuple[str, ...],
    bins: tuple[int, int, int],
    *,
    chunk_size: int = 8,
    q_bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ] | None = None,
    static_mask: np.ndarray | None = None,
    scout_pad: float = 0.0,
) -> RSMVolume:
    """Stream multiple v2 :class:`EwaldSphere`s into one :class:`RSMVolume`.

    Per-sphere :class:`SphereInput` carries its own ``energy``, ``UB``,
    and ``roi``.  All spheres' frames feed a single
    :class:`StreamingGridder`, so total peak memory stays at
    ``chunk_size`` frames regardless of how many spheres or how many
    frames each contains.

    Strictly cheaper than ``[process_scan_from_sphere(s) for s in ...]``
    followed by :func:`combine_grids`: there's no intermediate per-scan
    volume, no ``RegularGridInterpolator`` rebin at the end, and no
    correlated-NaN bookkeeping.
    """
    sphere_inputs = list(sphere_inputs)
    if not sphere_inputs:
        raise ValueError("sphere_inputs must not be empty")

    # Resolve per-sphere energies up-front so the scout pass has them.
    resolved: list[tuple[SphereInput, float, list[np.ndarray]]] = []
    for si in sphere_inputs:
        e = si.energy if si.energy is not None else _energy_from_sphere(si.sphere)
        angles_full = _angles_for_indices(si.sphere, diff_motors)
        resolved.append((si, e, angles_full))

    sg = StreamingGridder(mapper, bins)
    if q_bounds is None:
        sg.scout(
            [
                (angles_full, energy, si.UB,
                 (mapper.header.Nch1, mapper.header.Nch2))
                for (si, energy, angles_full) in resolved
            ],
            roi=None,  # ROI is per-sphere; union across all without it
            pad=scout_pad,
        )
    else:
        sg.set_bounds(*q_bounds)

    for si, energy, _full_angles in resolved:
        for img_chunk, chunk_indices in _iter_sphere_chunks(si.sphere, chunk_size):
            angles_chunk = _angles_for_indices(
                si.sphere, diff_motors, chunk_indices,
            )
            sg.add(
                img_chunk,
                angles_chunk,
                energy,
                UB=si.UB,
                roi=si.roi,
                static_mask=static_mask,
            )
    return sg.to_volume()
