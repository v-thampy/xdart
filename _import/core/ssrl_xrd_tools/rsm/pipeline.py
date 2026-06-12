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
    :func:`process_scan_from_nexus` on a v2 NeXus scan where each
    frame's raw image is loaded one chunk at a time.
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
# v2 NeXus scan as a data source
# ---------------------------------------------------------------------------
#
# Lets the RSM pipeline consume an xdart v2 :class:`LiveScan` directly,
# so a scan that's already been through xdart's 1D/2D integration can be
# re-used as the RSM input without re-parsing SPEC + raw image files.
#
# Per-scan quantities pulled here:
#
# * Energy (eV)  — from ``scan.mg_args['wavelength']`` (metres) unless
#   the caller passes ``energy=`` explicitly.
# * Per-frame motor positions — ``scan.scan_data[motor].values``, indexed
#   by the frame IDs in ``scan.frames.index``.
# * Raw images — one chunk at a time via ``frame._lazy_load_raw()``; frames
#   are released after each chunk so peak memory stays at chunk_size frames.
#
# UB is **not** in the v2 NeXus schema yet (xdart's integration pipeline
# doesn't need it); pass it in via ``UB=``.  When the schema grows a
# ``/entry/sample/UB`` field we'll auto-resolve it the same way as energy.
#
# To avoid a circular import (ssrl_xrd_tools is below xdart in the stack)
# the scan is duck-typed against the protocol below.


class _FrameLike(Protocol):
    """Minimal frame interface needed by the v2-scan RSM path."""
    idx: int
    map_raw: np.ndarray | None
    def _lazy_load_raw(self) -> bool: ...


class _FrameSeriesLike(Protocol):
    """Minimal LiveFrameSeries interface — index + lazy __getitem__."""
    index: list[int]
    def __getitem__(self, idx: int) -> _FrameLike: ...


class _ScanLike(Protocol):
    """Frame-source scan with motor metadata for RSM processing."""
    @property
    def frame_indices(self) -> list[int]: ...
    def iter_chunks(self, chunk_size: int) -> Iterator[tuple[np.ndarray, list[int]]]: ...


def _energy_from_scan(scan: _ScanLike) -> float:
    """Resolve X-ray energy in eV from the scan's stored wavelength.

    ``scan.mg_args['wavelength']`` is in metres (pyFAI convention),
    so ``E = h c / λ`` simplifies to ``E_eV = 12398 / λ_Å``.
    """
    mg_args = getattr(scan, "mg_args", {}) or {}
    wavelength_m = mg_args.get("wavelength")
    if wavelength_m and wavelength_m > 0:
        return 12398.0 / (float(wavelength_m) * 1e10)
    explicit_energy_eV = getattr(scan, "energy_eV", None)
    if explicit_energy_eV is not None and np.isfinite(explicit_energy_eV):
        return float(explicit_energy_eV)
    explicit_energy_keV = getattr(scan, "energy_keV", None)
    if explicit_energy_keV is not None and np.isfinite(explicit_energy_keV):
        return float(explicit_energy_keV) * 1000.0
    metadata = getattr(scan, "metadata", {}) or {}
    if isinstance(metadata, dict):
        energy_eV = metadata.get("energy_eV")
        if energy_eV is not None and np.isfinite(energy_eV):
            return float(energy_eV)
        energy_keV = metadata.get("energy_keV")
        if energy_keV is not None and np.isfinite(energy_keV):
            return float(energy_keV) * 1000.0
    explicit_energy = getattr(scan, "energy", None)
    if explicit_energy is not None and np.isfinite(explicit_energy):
        return float(explicit_energy) * 1000.0  # legacy Scan.energy is keV
    raise ValueError(
        "scan has no usable wavelength or energy metadata; pass energy= explicitly."
    )


def _angles_for_indices(
    scan: _ScanLike,
    diff_motors: list[str] | tuple[str, ...],
    indices: list[int] | None = None,
) -> list[np.ndarray]:
    """Pull per-frame motor arrays from ``scan.scan_data``.

    Returns a list aligned with ``diff_motors``: one ndarray per motor,
    each of length ``len(indices)`` (or len(scan_data) if indices is None).
    Raises :class:`KeyError` if any motor is missing from the DataFrame.
    """
    scan_data = getattr(scan, "scan_data", None)
    if scan_data is None:
        metadata = getattr(scan, "metadata", {}) or {}
        if isinstance(metadata, dict):
            scan_data = metadata.get("scan_data")
    if scan_data is None:
        motors = getattr(scan, "motors", None)
        if motors:
            scan_data = motors
    if scan_data is None:
        raise ValueError("scan has no per-frame motor metadata")
    cols = list(scan_data.columns) if hasattr(scan_data, "columns") else list(scan_data)
    missing = [m for m in diff_motors if m not in cols]
    if missing:
        raise KeyError(
            f"motors {missing!r} not in scan.scan_data (have {cols!r})"
        )
    if indices is None:
        return [
            np.asarray(getattr(scan_data[m], "values", scan_data[m]), dtype=float)
            for m in diff_motors
        ]
    if hasattr(scan_data, "loc"):
        return [
            np.asarray(scan_data.loc[indices, m].values, dtype=float)
            for m in diff_motors
        ]
    labels = [int(idx) for idx in getattr(scan, "frame_indices")]
    row_of = {label: row for row, label in enumerate(labels)}
    rows = [row_of[int(idx)] for idx in indices]
    return [
        np.asarray(scan_data[m], dtype=float)[rows]
        for m in diff_motors
    ]


def _iter_scan_chunks(
    scan: _ScanLike,
    chunk_size: int,
) -> Iterator[tuple[np.ndarray, list[int]]]:
    """Yield ``(img_chunk, frame_indices)`` for the streaming gridder.

    Each ``img_chunk`` is ``(n_chunk, H, W)`` — a fresh ``np.stack``,
    independent of the source frames.  Memory promise:

    1. **Only ``chunk_size`` raw frames are resident at any time.**
       Frames materialised by this iterator via ``_lazy_load_raw`` are
       cleared (``frame.map_raw = None``) in a ``finally`` block after
       the consumer is done with each chunk.  Without this, the
       ``scan.frames`` cache would hold every lazy-loaded frame in
       memory and the "memory-bounded" promise of streaming would
       degrade to "full-scan resident" for any user who actually has
       v2-reloaded frames.
    2. **Arches that arrived with ``map_raw`` already populated are
       left alone.**  We only free what we ourselves loaded — caller-
       owned data is not our responsibility to invalidate.
    3. **The stacked chunk is a copy** (``np.stack`` allocates fresh
       output), so it survives the per-frame clearing.
    """
    source_iter = getattr(scan, "iter_chunks", None)
    if callable(source_iter):
        yield from source_iter(chunk_size)
        return

    indices = getattr(scan, "frame_indices", None)
    if indices is None:
        indices = scan.frames.index
    indices = list(indices)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0; got {chunk_size}")

    for start in range(0, len(indices), chunk_size):
        chunk_indices = indices[start : start + chunk_size]
        images: list[np.ndarray] = []
        frames: list[_FrameLike] = []
        was_missing: list[bool] = []
        try:
            for idx in chunk_indices:
                frame = scan.frames[idx]
                missing = frame.map_raw is None
                frames.append(frame)
                was_missing.append(missing)
                if missing:
                    ok = frame._lazy_load_raw()
                    if not ok or frame.map_raw is None:
                        raise RuntimeError(
                            f"could not lazy-load raw frame for frame {idx} "
                            f"(check source_file / source_frame_idx "
                            f"provenance)"
                        )
                images.append(np.asarray(frame.map_raw))
            yield np.stack(images, axis=0), chunk_indices
        finally:
            # Free only what we materialised ourselves; respect frames
            # that arrived with map_raw already populated.
            for frame, missing in zip(frames, was_missing):
                if missing:
                    frame.map_raw = None


def process_scan_from_nexus(
    scan: _ScanLike,
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
    """Stream-process a frame-source scan into an :class:`RSMVolume`.

    Reads per-frame motor positions from ``scan.scan_data``, energy
    from ``scan.mg_args['wavelength']`` (unless ``energy=`` is given),
    and raw images one chunk at a time via ``scan.iter_chunks``.
    All frames flow through a single :class:`StreamingGridder` so peak
    memory is bounded by ``chunk_size`` regardless of total frame count.

    Parameters
    ----------
    scan : FrameSource scan (duck-typed)
        Must expose chunk iteration plus per-frame motor metadata. xdart
        ``LiveScan`` and ``io.read.Scan`` both satisfy this boundary.
    mapper : PixelQMap
        Diffractometer convention + detector header.
    diff_motors : sequence of str
        Motor column names in ``scan.scan_data`` to feed into
        ``xu.QConversion`` (sample axes first, then detector axes).
    bins : tuple of int
        ``xu.Gridder3D`` bin counts.
    UB : (3, 3) ndarray, optional
        Sample orientation matrix.  v2 NeXus doesn't store this yet —
        pass it explicitly until the schema is extended.
    energy : float, optional
        X-ray energy in eV.  ``None`` → resolved from the scan.
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
        energy = _energy_from_scan(scan)
    angles_full = _angles_for_indices(scan, diff_motors)

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

    for img_chunk, chunk_indices in _iter_scan_chunks(scan, chunk_size):
        angles_chunk = _angles_for_indices(scan, diff_motors, chunk_indices)
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
class ScanInput:
    """One scan's data for :func:`grid_scans_streaming`.

    ``energy`` defaults to ``None`` so the per-scan resolver kicks in;
    pass an explicit value to override.  ``UB`` and ``roi`` are
    per-scan and may differ across the list.
    """
    scan: Any  # _ScanLike — typed as Any to keep the dataclass simple
    energy: float | None = None
    UB: np.ndarray | None = None
    roi: tuple[int, int, int, int] | None = None


def grid_scans_streaming(
    mapper: PixelQMap,
    scan_inputs: Iterable[ScanInput],
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
    """Stream multiple v2 :class:`LiveScan`s into one :class:`RSMVolume`.

    Per-scan :class:`ScanInput` carries its own ``energy``, ``UB``,
    and ``roi``.  All scans' frames feed a single
    :class:`StreamingGridder`, so total peak memory stays at
    ``chunk_size`` frames regardless of how many scans or how many
    frames each contains.

    Strictly cheaper than ``[process_scan_from_nexus(s) for s in ...]``
    followed by :func:`combine_grids`: there's no intermediate per-scan
    volume, no ``RegularGridInterpolator`` rebin at the end, and no
    correlated-NaN bookkeeping.
    """
    scan_inputs = list(scan_inputs)
    if not scan_inputs:
        raise ValueError("scan_inputs must not be empty")

    # Resolve per-scan energies up-front so the scout pass has them.
    resolved: list[tuple[ScanInput, float, list[np.ndarray]]] = []
    for si in scan_inputs:
        e = si.energy if si.energy is not None else _energy_from_scan(si.scan)
        angles_full = _angles_for_indices(si.scan, diff_motors)
        resolved.append((si, e, angles_full))

    sg = StreamingGridder(mapper, bins)
    if q_bounds is None:
        sg.scout(
            [
                (angles_full, energy, si.UB,
                 (mapper.header.Nch1, mapper.header.Nch2))
                for (si, energy, angles_full) in resolved
            ],
            roi=None,  # ROI is per-scan; union across all without it
            pad=scout_pad,
        )
    else:
        sg.set_bounds(*q_bounds)

    for si, energy, _full_angles in resolved:
        for img_chunk, chunk_indices in _iter_scan_chunks(si.scan, chunk_size):
            angles_chunk = _angles_for_indices(
                si.scan, diff_motors, chunk_indices,
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
