"""Batch stitching helper for the v2 ``LiveScan`` (xdart 0.37+).

Thin orchestration layer over
:mod:`ssrl_xrd_tools.integrate.multi` (which exports
``create_multigeometry_integrators``, ``stitch_1d``, ``stitch_2d``).

The wrangler calls :func:`run_stitch` once all frames are loaded.
Stitched outputs are stored on the scan as
``scan.stitched_1d`` / ``scan.stitched_2d`` and persisted by the
v2 NeXus writer's ``finalize=True`` pass.

Per plan §4.5: stitch is **batch-only**.  Per-image
``integrated_1d/2d`` is *also* written so the viewer can still show
per-image patterns; this module only writes the merged outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from xdart.modules.live import LiveScan


def run_stitch(
    scan: "LiveScan",
    *,
    mode: Literal["1d", "2d"] = "1d",
    norm_motor: str | None = None,
    mask: np.ndarray | None = None,
    method: str = "BBox",
    npt_1d: int = 2000,
    npt_rad_2d: int = 1500,
    npt_azim_2d: int = 720,
    radial_range: tuple[float, float] | None = None,
    azimuth_range: tuple[float, float] | None = None,
    unit: str = "q_A^-1",
) -> None:
    """Stitch the scan's per-frame images into a single merged pattern.

    Reads:

    * ``scan.frames`` — image stack (via ``frame.map_raw - frame.bg_raw``).
    * ``scan.geometry`` — :class:`DiffractometerGeometry` for the
      per-frame ``rot1``/``rot2`` derivation.
    * ``scan.scan_data`` — motor positions (DataFrame).
    * ``scan.frames[0].poni`` — base PONI geometry shared across
      all frames.

    Writes:

    * ``scan.stitched_1d`` (mode ``"1d"``) — an ``IntegrationResult1D``.
    * ``scan.stitched_2d`` (mode ``"2d"``) — an ``IntegrationResult2D``.

    Parameters
    ----------
    scan
        Source of frames, geometry, scan_data, and base PONI.
    mode
        ``"1d"`` for ``I(q)``, ``"2d"`` for ``I(q, χ)``.
    norm_motor
        Name of a column in ``scan.scan_data`` whose values are used
        to divide each image (per-image normalization, e.g. ``"i1"``).
        If ``None``, no normalization is applied.
    mask
        Optional global detector mask shared across all frames.  Per-image
        masks are not supported in v1 of stitch (see plan §4.5).
    method
        pyFAI integration method (default ``"BBox"`` — matches the
        reference notebook).
    npt_1d, npt_rad_2d, npt_azim_2d
        Output bin counts.  Defaults from plan §9 open question 4.
    radial_range, azimuth_range
        Optional explicit bin ranges; pyFAI auto-sizes if ``None``.
    unit
        Radial output unit string.

    Raises
    ------
    RuntimeError
        If ``scan.geometry`` is unset or no PONI is available.
    ValueError
        If ``mode`` isn't ``"1d"`` or ``"2d"``.
    """
    if mode not in ("1d", "2d"):
        raise ValueError(f"mode must be '1d' or '2d', got {mode!r}")
    if scan.geometry is None:
        raise RuntimeError(
            "LiveScan.geometry is unset — set it before stitching "
            "(use ssrl_xrd_tools.core.geometry.DiffractometerGeometry)."
        )
    frames = list(scan.frames)
    if not frames:
        raise RuntimeError("LiveScan has no frames — load frames first")

    base_poni = getattr(frames[0], "poni", None)
    if base_poni is None:
        raise RuntimeError("No PONI on frames[0] — cannot stitch")

    # Per-frame rotations (in degrees, since multi.py expects degrees)
    geom = scan.geometry
    motors = {
        m: np.asarray(scan.scan_data[m].values, dtype=float)
        for m in geom.all_referenced_motors()
        if m in scan.scan_data.columns
    }
    derived = geom.derive_per_frame(motors)
    # multi.py expects degrees, not radians — invert deg2rad on rot1/rot2
    rot1_deg = np.rad2deg(derived["rot1"])
    rot2_deg = np.rad2deg(derived["rot2"])

    # Image stack — bg-subtracted, optionally per-image normalized.
    #
    # P5: lazy-load ``map_raw`` for v2-reloaded frames.  On a scan
    # loaded from disk (vs. one freshly produced by the wrangler),
    # ``frame.map_raw`` is None until we ask :meth:`_lazy_load_raw``
    # to hydrate it from the source file (TIFF / NeXus master).
    # Without this call, ``frame.map_raw - frame.bg_raw`` would
    # TypeError on the None subtract and abort the whole stitch.
    # Frames whose source isn't on disk get logged + skipped rather
    # than crashing the whole run.
    #
    # Q2: when any frame is skipped, we must filter the geometry
    # arrays and normalisation lookup to match the surviving image
    # list.  Pre-Q2 the integrators were built from the full
    # ``rot1_deg/rot2_deg`` while ``images`` had fewer entries —
    # either pyFAI's MultiGeometry would mis-pair an image with the
    # wrong rotation, or it would length-mismatch and raise.
    # ``surviving_indices`` is the positional row in the original
    # ``frames`` (and therefore ``scan_data``, ``rot*_deg``) for
    # each frame that survived; we filter all three using it after
    # the loop.
    import logging as _logging
    _logger = _logging.getLogger(__name__)
    images = []
    skipped = []
    surviving_indices = []
    for i, frame in enumerate(frames):
        if frame.map_raw is None:
            try:
                frame._lazy_load_raw()
            except Exception as e:
                _logger.warning(
                    'stitch: lazy raw load failed for frame %s: %s',
                    frame.idx, e,
                )
        if frame.map_raw is None:
            skipped.append(frame.idx)
            continue
        img = np.asarray(frame.map_raw - frame.bg_raw, dtype=float)
        if norm_motor is not None and norm_motor in scan.scan_data.columns:
            denom = float(scan.scan_data[norm_motor].iloc[i])
            if denom != 0:
                img = img / denom
        images.append(img)
        surviving_indices.append(i)
    if skipped:
        _logger.warning(
            'stitch: skipped %d frames with no raw data: %s',
            len(skipped), skipped,
        )
    if not images:
        raise RuntimeError(
            'stitch: no frames with raw data available — '
            'all source files missing or unloadable'
        )
    img_stack = np.stack(images, axis=0)

    # Q2: align geometry arrays with the surviving image stack.
    # When no frames were skipped this is a no-op slice.
    surviving_idx_arr = np.asarray(surviving_indices, dtype=int)
    rot1_deg = rot1_deg[surviving_idx_arr]
    rot2_deg = rot2_deg[surviving_idx_arr]

    from ssrl_xrd_tools.integrate.multi import (
        create_multigeometry_integrators,
        stitch_1d,
        stitch_2d,
    )

    integrators = create_multigeometry_integrators(
        base_poni,
        rot1_angles=rot1_deg,
        rot2_angles=rot2_deg if np.any(rot2_deg) else None,
    )

    if mode == "1d":
        scan.stitched_1d = stitch_1d(
            img_stack,
            integrators,
            npt=npt_1d,
            unit=unit,
            method=method,
            mask=mask,
            radial_range=radial_range,
        )
    else:
        scan.stitched_2d = stitch_2d(
            img_stack,
            integrators,
            npt_rad=npt_rad_2d,
            npt_azim=npt_azim_2d,
            unit=unit,
            method=method,
            mask=mask,
            radial_range=radial_range,
            azimuth_range=azimuth_range,
        )


__all__ = ["run_stitch"]
