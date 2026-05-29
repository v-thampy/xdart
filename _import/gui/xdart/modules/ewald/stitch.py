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
    max_stack_bytes: float = 16e9,
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

    # Memory guard: pyFAI MultiGeometry needs every image resident at once
    # (the images list + np.stack below), so a long Eiger run can demand
    # hundreds of GB and swap the machine to death.  Estimate the float64
    # stack size up front and refuse loudly instead.  Raise ``max_stack_bytes``
    # if you have the RAM, or stitch a subset of frames.
    _shape = getattr(getattr(frames[0], "map_raw", None), "shape", None)
    if _shape is None and getattr(frames[0], "_lazy_load_raw", None) is not None:
        try:
            frames[0]._lazy_load_raw()
            _shape = getattr(frames[0].map_raw, "shape", None)
        except Exception:
            _shape = None
    if _shape is not None:
        # Peak holds the per-frame float ``images`` list AND the fresh
        # ``np.stack`` copy simultaneously (≈2× the float64 stack), on top
        # of any still-resident raw ``map_raw`` arrays.  Estimate the 2×
        # transient so the guard doesn't under-count and let an OOM through.
        one_stack = len(frames) * int(np.prod(_shape)) * 8  # float64 stack
        est_bytes = 2 * one_stack
        if est_bytes > max_stack_bytes:
            raise MemoryError(
                f"Stitch would need ~{est_bytes / 1e9:.1f} GB peak (list + "
                f"stacked copy) to hold {len(frames)} frames of {_shape} in "
                f"memory (limit {max_stack_bytes / 1e9:.1f} GB). pyFAI "
                "MultiGeometry requires all images at once — stitch fewer "
                "frames or raise max_stack_bytes if you have the RAM."
            )

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

    # Validate the normalization column up front.  Silently skipping a
    # requested ``norm_motor`` that isn't in scan_data would produce an
    # *un-normalized* stitch that looks normalized — fail loudly instead.
    if norm_motor is not None and norm_motor not in scan.scan_data.columns:
        raise ValueError(
            f"norm_motor {norm_motor!r} not found in scan_data columns "
            f"{list(scan.scan_data.columns)}; cannot normalize the stitch."
        )

    images = []
    skipped = []
    surviving_indices = []
    zero_norm = []  # frames with a zero normalization value (left un-normalized)
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
        if norm_motor is not None:
            denom = float(scan.scan_data[norm_motor].iloc[i])
            if denom != 0:
                img = img / denom
            else:
                # Dividing by zero would give inf/nan; leave this frame
                # un-normalized but flag it — silently mixing normalized
                # and un-normalized frames skews the stitch.
                zero_norm.append(frame.idx)
        images.append(img)
        surviving_indices.append(i)
    if zero_norm:
        _logger.warning(
            'stitch: %d frame(s) had %s == 0 and were left un-normalized '
            '(stitch may be biased): %s',
            len(zero_norm), norm_motor, zero_norm,
        )
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

    # The integrator-build + 1D/2D dispatch lives in ssrl_xrd_tools now
    # (keep-xdart-thin): this function's job is the LiveScan-specific
    # gathering above; the stitch orchestration is shared headless code.
    from ssrl_xrd_tools.integrate.multi import stitch_images

    result = stitch_images(
        img_stack,
        base_poni,
        rot1_deg,
        rot2_deg,
        mode=mode,
        npt_1d=npt_1d,
        npt_rad_2d=npt_rad_2d,
        npt_azim_2d=npt_azim_2d,
        unit=unit,
        method=method,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        mask=mask,
    )
    if mode == "1d":
        scan.stitched_1d = result
    else:
        scan.stitched_2d = result


__all__ = ["run_stitch"]
