"""Batch stitching helper for the v2 ``LiveScan`` (xdart 0.37+).

Thin orchestration layer over
:mod:`xrd_tools.integrate.multi` (which exports
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
    backend: str = "multigeometry",
    max_stack_bytes: float = 16e9,
) -> None:
    """Stitch the scan's per-frame images into a single merged pattern.

    Reads:

    * ``scan.frames`` — image stack (via ``frame.map_raw - frame.bg_raw``).
    * ``scan.geometry`` — :class:`Diffractometer` for the
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
    backend
        ``"multigeometry"`` (default, pyFAI MultiGeometry) or ``"pyfai_hist"``
        (the per-pixel q-histogram merge — q-Å⁻¹ only; the substrate the shared
        CorrectionStack + GI corrections layer on).

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
            "(use xrd_tools.core.geometry.Diffractometer)."
        )
    frames = list(scan.frames)
    if not frames:
        raise RuntimeError("LiveScan has no frames — load frames first")

    base_poni = getattr(frames[0], "poni", None)
    if base_poni is None:
        raise RuntimeError("No PONI on frames[0] — cannot stitch")

    # Frames whose raw we hydrate from disk during this stitch — cleared in
    # the ``finally`` below so a large reloaded scan doesn't keep every
    # full-resolution image resident afterward.  Declared up here because
    # the memory-guard probe below may be the first to load frames[0].
    lazy_loaded = []
    try:

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
            else:
                if frames[0].map_raw is not None:
                    lazy_loaded.append(frames[0])
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

        # Align scan_data to frame order BEFORE deriving motors/normalization.
        # ``frames`` is iterated in scan.frames.index order; scan_data is indexed
        # by frame id.  Using positional ``.values`` / ``.iloc[i]`` assumes the
        # two orders coincide — for gapped or out-of-order scans they don't, and
        # angles/monitors would attach to the wrong image.  Reindex to the actual
        # frame ids so row i corresponds to frames[i] (NaN for any frame missing
        # from scan_data).
        frame_ids = [f.idx for f in frames]
        scan_data = scan.scan_data
        if list(scan_data.index) != frame_ids:
            try:
                scan_data = scan_data.reindex(frame_ids)
            except (TypeError, ValueError) as e:
                # reindex raises on a non-unique index (duplicate frame ids in
                # the scan metadata).  Falling back to the original frame and
                # then reading it positionally (.iloc[i]) would silently attach
                # the wrong motor/monitor to each image — fail loudly instead.
                raise ValueError(
                    f"stitch: cannot align scan_data (index {list(scan_data.index)!r}) "
                    f"to frame ids {frame_ids!r}: {e}. The scan metadata likely has "
                    "duplicate/non-unique frame indices; fix the scan_data index "
                    "before stitching."
                ) from e

        # Per-frame rotations (in degrees, since multi.py expects degrees)
        geom = scan.geometry
        motors = {
            m: np.asarray(scan_data[m].values, dtype=float)
            for m in geom.all_referenced_motors()
            if m in scan_data.columns
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
        if norm_motor is not None and norm_motor not in scan_data.columns:
            raise ValueError(
                f"norm_motor {norm_motor!r} not found in scan_data columns "
                f"{list(scan_data.columns)}; cannot normalize the stitch."
            )

        images = []
        skipped = []
        surviving_indices = []
        zero_norm = []  # norm == 0 (left un-normalized, per project decision)
        nan_norm = []   # norm is NaN/inf (would silently corrupt — fatal)
        for i, frame in enumerate(frames):
            if frame.map_raw is None:
                try:
                    frame._lazy_load_raw()
                except Exception as e:
                    _logger.warning(
                        'stitch: lazy raw load failed for frame %s: %s',
                        frame.idx, e,
                    )
                else:
                    # Track frames we actually pulled in (vs. frames the wrangler
                    # already had resident) so we restore the prior state and
                    # don't free arrays another consumer still wants.  frames[0]
                    # may already be tracked by the memory-guard probe above.
                    if frame.map_raw is not None and not any(
                        frame is x for x in lazy_loaded
                    ):
                        lazy_loaded.append(frame)
            if frame.map_raw is None:
                skipped.append(frame.idx)
                continue
            img = np.asarray(frame.map_raw - frame.bg_raw, dtype=float)
            if norm_motor is not None:
                denom = float(scan_data[norm_motor].iloc[i])
                if not np.isfinite(denom):
                    # A NaN/inf monitor — typically a frame that fell out of
                    # scan_data during the frame-id reindex above — would
                    # divide to an all-NaN image and silently poison the whole
                    # stitch (NaN != 0, so the zero-guard below misses it).
                    # Collect the offenders and fail loudly after the loop.
                    nan_norm.append(frame.idx)
                elif denom != 0:
                    img = img / denom
                else:
                    # Dividing by zero would give inf/nan; leave this frame
                    # un-normalized but flag it (project decision: warn, don't
                    # raise, on a zero monitor).
                    zero_norm.append(frame.idx)
            images.append(img)
            surviving_indices.append(i)
        # Everything from here can raise (NaN guards, pyFAI); the ``finally``
        # frees the raw frame arrays we lazy-loaded so a large reloaded scan
        # doesn't keep every full-resolution image resident after the stitch.
        if nan_norm:
            raise ValueError(
                f'stitch: norm_motor {norm_motor!r} is NaN/inf for frame(s) '
                f'{nan_norm} (likely missing from scan_data after frame-id '
                'alignment); refusing to produce a corrupted stitch. Drop those '
                'frames or supply their monitor values.'
            )
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

        # Q2: align geometry arrays with the surviving images.
        # When no frames were skipped this is a no-op slice.
        surviving_idx_arr = np.asarray(surviving_indices, dtype=int)
        rot1_deg = rot1_deg[surviving_idx_arr]
        rot2_deg = rot2_deg[surviving_idx_arr]

        # A missing motor row (NaN after the frame-id reindex) yields a NaN
        # derived rotation, which pyFAI would turn into garbage geometry.
        # Fail clearly, naming the frames, instead of stitching nonsense.
        rot_bad = ~(np.isfinite(rot1_deg) & np.isfinite(rot2_deg))
        if rot_bad.any():
            bad_ids = [frames[surviving_indices[j]].idx
                       for j in np.flatnonzero(rot_bad)]
            raise ValueError(
                f'stitch: NaN detector rotation for frame(s) {bad_ids} — their '
                'angle motor(s) are missing from scan_data after frame-id '
                'alignment. Drop those frames or provide the motor positions.'
            )

        # The integrator-build + 1D/2D dispatch lives in xrd_tools now
        # (keep-xdart-thin): this function's job is the LiveScan-specific
        # gathering above; the stitch orchestration is shared headless code.
        # Pass the image *list* straight through (stitch_images accepts a
        # list) to avoid an extra full np.stack copy of every frame.
        from xrd_tools.integrate.multi import stitch_images

        result = stitch_images(
            images,
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
            backend=backend,
        )
        if mode == "1d":
            scan.stitched_1d = result
        else:
            scan.stitched_2d = result
    finally:
        # Release the raw arrays we pulled in for this stitch (only the
        # ones we hydrated — frames the wrangler had resident are left
        # alone).  They re-load lazily on next access.
        for fr in lazy_loaded:
            fr.map_raw = None


__all__ = ["run_stitch"]
