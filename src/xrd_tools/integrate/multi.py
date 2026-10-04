# xrd_tools/integrate/multi.py
"""
Multi-image stitching via pyFAI MultiGeometry.

The key pattern: when the detector is scanned to different angular positions
(for example ``del`` and ``nu`` on a stacked detector arm), every image
gets its own AzimuthalIntegrator with the detector angle encoded.
``create_multigeometry_integrators`` builds that list; ``stitch_1d`` /
``stitch_2d`` perform the stitched integration.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from xrd_tools.core.containers import (
    PONI,
    IntegrationResult1D,
    IntegrationResult2D,
)
from xrd_tools.corrections.detector import DEFAULT_MIN_FLAT_FRACTION
from xrd_tools.corrections.frame_preparation import correction_image
from xrd_tools.integrate import pyfai_numexpr
from xrd_tools.integrate.frame_workers import parallel_frames
from xrd_tools.integrate.detector_mask import mask_with_detector
from xrd_tools.integrate.calibration import poni_to_integrator
from xrd_tools.integrate.gid import _nan_empty_1d
from xrd_tools.integrate.stitch_gi import GI_CUTS, GI_UNITS

if TYPE_CHECKING:
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StitchDiagnostics:
    """Detached per-bin MultiGeometry coverage and normalization weight."""

    coverage: np.ndarray
    normalization: np.ndarray

    def __post_init__(self) -> None:
        coverage = np.array(self.coverage, dtype=np.float64, order="C", copy=True)
        normalization = np.array(
            self.normalization, dtype=np.float64, order="C", copy=True
        )
        if (
            coverage.shape != normalization.shape
            or coverage.ndim not in {1, 2}
            or not np.all(np.isfinite(coverage))
            or not np.all(np.isfinite(normalization))
            or np.any(coverage < 0)
            or np.any(normalization < 0)
        ):
            raise ValueError("Stitch diagnostics must be finite nonnegative peers")
        coverage.setflags(write=False)
        normalization.setflags(write=False)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "normalization", normalization)


def _normalization_factors(
    normalization: np.ndarray | Sequence[float] | None,
    count: int,
) -> np.ndarray | None:
    if normalization is None:
        return None
    values = np.asarray(normalization, dtype=float)
    if values.shape != (count,):
        raise ValueError(
            f"normalization length {values.shape} != number of images {count}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("normalization contains non-finite (nan/inf) values")
    if np.any(values <= 0):
        raise ValueError(
            "normalization contains zero or negative values "
            f"(monitor must be > 0): {values[values <= 0].tolist()}"
        )
    return values


def _scout_extents(integrator, radial_unit, radial: bool, azimuth: bool):
    """One frame's (radial (min, max), χ (min, max)) for MultiGeometry's range guesses, each None when not
    asked: what ``array_from_unit`` gives (pyFAI's ``center_array`` formula), its pixel positions (pyFAI's
    Cython) computed once for both and outside the numexpr lock, only the unit formulas under it. A geometry
    with a parallax correction, whose positions evaluate numexpr, or a fiber unit takes pyFAI's own call
    under the lock."""
    from pyFAI import units

    try:
        # MultiGeometry defaults to chi_disc=180 and mutates every member
        # before integrating.  Scout in that same convention; otherwise a
        # caller-supplied AI using chiDiscAtZero can produce a 0..360 range
        # which is then paired with -180..180 integration values, dropping
        # half the detector.
        integrator.setChiDiscAtPi()
        positions = None
        if getattr(integrator, "_parallax", True) is None and not isinstance(radial_unit, units.UnitFiber):
            positions = integrator.position_array(corners=False)

        def extent(unit):
            if positions is None:
                values = integrator.array_from_unit(unit=unit)
            else:  # center_array's own formula (scale=True) on those positions
                values = unit.equation(positions[..., 2], positions[..., 1], positions[..., 0],
                                       integrator.wavelength) * unit.scale
            values = np.asarray(values, dtype=float)
            return float(values.min()), float(values.max())

        with pyfai_numexpr.PYFAI_NUMEXPR_LOCK:  # the unit formulas are numexpr's
            return (extent(radial_unit) if radial else None,
                    extent(units.CHI_DEG) if azimuth else None)
    finally:
        integrator.reset(collect_garbage=False)


def _resolve_streaming_ranges(
    integrator_factory: Callable[[], Iterator[AzimuthalIntegrator]],
    image_count: int,
    *,
    unit: str,
    radial_range: tuple[float, float] | None,
    azimuth_range: tuple[float, float] | None,
    workers: int = 1,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Match MultiGeometry's global range guesses with O(one-AI) residency (one a worker: ``workers``
    frames scouted at once, ``_scout_extents``; their extents merge in frame order)."""

    if radial_range is not None and azimuth_range is not None:
        return radial_range, azimuth_range
    from pyFAI import units

    radial_unit = units.to_unit(unit)
    radial_min = math.inf
    radial_max = -math.inf
    azimuth_min = math.inf
    azimuth_max = -math.inf
    integrators = iter(integrator_factory())

    def read(index):
        try:
            return (next(integrators),)
        except StopIteration as error:
            raise ValueError(
                f"streaming Stitch received {index} integrators; "
                f"expected {image_count}"
            ) from error

    def scout(_index, integrator):
        return _scout_extents(integrator, radial_unit, radial_range is None, azimuth_range is None)

    def merge(extents):
        nonlocal radial_min, radial_max, azimuth_min, azimuth_max
        radial, azimuth = extents
        if radial is not None:
            radial_min, radial_max = min(radial_min, radial[0]), max(radial_max, radial[1])
        if azimuth is not None:
            azimuth_min, azimuth_max = min(azimuth_min, azimuth[0]), max(azimuth_max, azimuth[1])

    if workers == 1:
        for index in range(image_count):
            merge(scout(index, *read(index)))
    else:
        parallel_frames(image_count, read, scout, merge, workers)
    try:
        next(integrators)
    except StopIteration:
        pass
    else:
        raise ValueError("streaming Stitch received more integrators than declared")
    resolved_radial = (
        radial_range if radial_range is not None else (radial_min, radial_max)
    )
    resolved_azimuth = (
        azimuth_range if azimuth_range is not None else (azimuth_min, azimuth_max)
    )
    if any(
        not math.isfinite(value)
        for value in (*resolved_radial, *resolved_azimuth)
    ) or any(low >= high for low, high in (resolved_radial, resolved_azimuth)):
        raise ValueError("streaming Stitch could not resolve finite geometry ranges")
    return resolved_radial, resolved_azimuth


def _streaming_multigeometry(
    images: Iterable[np.ndarray],
    image_count: int,
    integrator_factory: Callable[[], Iterator[AzimuthalIntegrator]],
    *,
    mode: str,
    npt_rad: int,
    npt_azim: int,
    unit: str,
    method: str,
    radial_range: tuple[float, float] | None,
    azimuth_range: tuple[float, float] | None,
    mask: np.ndarray | None,
    normalization: np.ndarray | Sequence[float] | None,
    correct_solid_angle: bool,
    error_model: str | None,
    polarization_factor: float | None,
    frame_screen: Any = None,
    dark: np.ndarray | None = None,
    flat: np.ndarray | None = None,
    flat_min_fraction: float = DEFAULT_MIN_FLAT_FRACTION,
    gi=None,
    gi_angles: Sequence[tuple[float, float]] | None = None,
    profile_npt: int | None = None,
    gi_map_npt: tuple[int, int] | None = None,
    profile_unit: str | None = None,
    profile_radial_range: tuple[float, float] | None = None,
    profile_azimuth_range: tuple[float, float] | None = None,
    workers: int = 1,
    progress: Callable[[int], object] | None = None,
):
    """Returns (pyFAI's merged result, each GI cut's [signal, normalization, count, axis] in the plan's
    order, or () without GI, and with ``profile_npt`` beside a 2-D map pyFAI's merged I(q) over every χ,
    or over ``profile_azimuth_range`` (°), in that many bins, as a 1-D Stitch of the same frames makes it;
    else None). A 1-D Stitch's ``azimuth_range`` is its χ window, as pyFAI's (MultiGeometry's
    ``azimuth_range``); None: every χ.

    ``workers`` above 1 integrates that many frames at once on threads (``frame_workers``): this
    thread reads the frames and merges their sums in frame order, so the result is the same, bit for
    bit, as one worker's. It takes ``method="full"``, the method Stitch's operation runs.
    ``progress`` is told how many frames are merged, after each one.

    The profile is integrated in the same pass as the map, from the same frame and integrator, whose
    geometry pyFAI computes once for both (one pass instead of two: the frames are read and the
    integrators built once). It takes the map's unit and radial range, or with ``profile_unit`` its own
    unit and ``profile_radial_range`` (None: its frames' extent), as Int's 1-D and 2-D integrations do."""
    from pyFAI.multi_geometry import MultiGeometry

    pyfai_numexpr.guard_result_means()

    if type(image_count) is not int or image_count < 1:
        raise ValueError("streaming Stitch image count must be positive")
    if type(workers) is not int or workers < 1:
        raise ValueError("Stitch frame workers must be a positive int")
    if workers > 1 and method != "full":
        # the one method whose pyFAI geometry is proven to reach numexpr only where the workers hold the
        # lock (tests/core/test_stitch_frame_workers.py); Stitch's operation runs it
        raise ValueError(f"Stitch's frame workers integrate with pyFAI's 'full' splitting; {method!r} "
                         "takes one worker")
    if mode not in {"1d", "2d"}:
        raise ValueError(f"mode must be '1d' or '2d', got {mode!r}")
    factors = _normalization_factors(normalization, image_count)
    # pyFAI subtracts the dark in its preprocessing and applies the flat in the
    # normalization, like solid angle. Its Poisson model treats the dark as
    # noiseless (variance max(1, raw)); a measured dark carries its own counting
    # noise, so with a dark the variance max(1, raw) + dark is passed explicitly.
    # admitted by Int's owners (corrections.frame_preparation, shared with the xrayutilities path and RSM)
    dark, dark_invalid = correction_image(dark, "dark")
    flat, flat_invalid = correction_image(flat, "flat", min_fraction=flat_min_fraction)
    invalid = (dark_invalid if flat_invalid is None else flat_invalid if dark_invalid is None
               else dark_invalid | flat_invalid)
    fixed_radial, fixed_azimuth = _resolve_streaming_ranges(
        integrator_factory,
        image_count,
        unit=unit,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        workers=workers,
    )
    if gi is not None and (gi_angles is None or len(gi_angles) != image_count
                           or (mode == "2d" and (profile_npt is None or gi_map_npt is None))):
        raise ValueError("a GI cut needs one (incidence, tilt) per frame, and in 2-D I(q) and the GI map")
    if profile_npt is not None and (mode != "2d" or type(profile_npt) is not int or profile_npt < 1):
        raise ValueError("an I(q) profile goes beside a 2-D Stitch, with a positive point count")
    if profile_azimuth_range is not None and profile_npt is None:
        raise ValueError("a profile χ window goes with the I(q) profile beside a 2-D Stitch")
    if gi_map_npt is not None and (gi is None or mode != "2d" or type(gi_map_npt) is not tuple
                                   or len(gi_map_npt) != 2
                                   or any(type(points) is not int or points < 1 for points in gi_map_npt)):
        raise ValueError("a GI map goes beside a 2-D GI Stitch, with its q_ip and q_oop points")
    # a 1-D Stitch's ranges: its radial range (the map's unless it has its own), and its χ window or every
    # χ (scouted as a 1-D Stitch scouts it); scouted with the map's when both are the frames' extent
    profile_radial = profile_azimuth = None
    if profile_npt is not None:
        shared = profile_unit is None or (profile_unit, profile_radial_range) == (unit, radial_range)
        profile_unit = unit if profile_unit is None else profile_unit
        profile_radial = fixed_radial if shared else profile_radial_range
        profile_azimuth = (profile_azimuth_range if profile_azimuth_range is not None
                           else fixed_azimuth if azimuth_range is None else None)
        if profile_radial is None or profile_azimuth is None:
            profile_radial, profile_azimuth = _resolve_streaming_ranges(
                integrator_factory, image_count, unit=profile_unit, radial_range=profile_radial,
                azimuth_range=profile_azimuth, workers=workers)
    gi_bins = {} if gi is None else _gi_bins(integrator_factory, gi_angles, gi)
    gi_totals = None

    def add_gi(parts):
        nonlocal gi_totals
        if parts:
            gi_totals = [list(part) for part in parts] if gi_totals is None else [
                [total + value for total, value in zip(totals[:3], part[:3])] + [totals[3]]
                for totals, part in zip(gi_totals, parts)]

    image_iterator = iter(images)
    integrator_iterator = iter(integrator_factory())
    aggregate = profile_aggregate = None
    radial = azimuthal = None
    numexpr_lock = pyfai_numexpr.PYFAI_NUMEXPR_LOCK

    def read(index):
        """The next frame and its integrator, read in frame order."""
        try:
            from xrd_tools.sources.point_images import PointImageResult
            value = next(image_iterator)
            statistics = value if isinstance(value, PointImageResult) else None
            image = np.asarray(value.signal if statistics is not None else value, dtype=float)
        except StopIteration as error:
            raise ValueError(
                f"streaming Stitch received {index} images; expected {image_count}"
            ) from error
        try:
            integrator = next(integrator_iterator)
        except StopIteration as error:
            raise ValueError(
                f"streaming Stitch received {index} integrators; "
                f"expected {image_count}"
            ) from error
        return image, integrator, statistics

    def frame_part(index, image, integrator, statistics):
        """One frame's pyFAI sums, a frame worker's share: its integration result, I(q) beside a map,
        its GI sums and, screening, its total, with its radial and azimuthal axes. Its numexpr work (the
        polarization array, the GI sums) holds the pyFAI numexpr lock; the integrations run outside it."""
        try:
            if image.ndim != 2:
                raise ValueError(
                    f"streaming Stitch image must be 2-D, got shape {image.shape}"
                )
            for name, value in (("dark", dark), ("flat", flat)):
                if value is not None and value.shape != image.shape:
                    raise ValueError(
                        f"Stitch {name} shape {value.shape} does not match the frame {image.shape}"
                    )
            frame_dark = dark if statistics is None or statistics.exposure_multiplier else None
            frame_invalid = invalid if frame_dark is not None else flat_invalid
            if statistics is not None:
                missing = ~np.isfinite(image)
                frame_invalid = missing if frame_invalid is None else frame_invalid | missing
            frame_mask = (
                None
                if mask is None and frame_invalid is None
                else np.asarray(mask_with_detector(
                    integrator, frame_invalid if mask is None else
                    mask if frame_invalid is None else np.logical_or(mask, frame_invalid)), dtype=bool)
            )
            if frame_dark is not None:
                integrator.detector.darkcurrent = frame_dark
            monitor = 1.0 if factors is None else float(factors[index])
            variance = None if statistics is None else statistics.variance
            if error_model == "poisson" and frame_dark is not None:
                variance = np.maximum(image, 1.0) if variance is None else variance.copy()
                variance += np.clip(frame_dark, 0.0, None)
            if polarization_factor is not None:
                # the integrations' one numexpr use: pyFAI keeps the array on the integrator, and both
                # integrations below take this one
                with numexpr_lock:
                    integrator.polarization(image.shape, polarization_factor, with_checksum=True)
            multigeometry = MultiGeometry(
                [integrator],
                unit=unit,
                radial_range=fixed_radial,
                azimuth_range=fixed_azimuth,
                threadpoolsize=0,
            )
            if mode == "1d":
                result = multigeometry.integrate1d(
                    [image],
                    npt_rad,
                    correctSolidAngle=correct_solid_angle,
                    error_model=error_model if variance is None else None,
                    lst_variance=None if variance is None else [variance],
                    polarization_factor=polarization_factor,
                    normalization_factor=[monitor],
                    lst_mask=None if frame_mask is None else [frame_mask],
                    lst_flat=None if flat is None else [flat],
                    method=method,
                )
                current_shape = (npt_rad,)
            else:
                result = multigeometry.integrate2d(
                    [image],
                    npt_rad,
                    npt_azim,
                    correctSolidAngle=correct_solid_angle,
                    error_model=error_model if variance is None else None,
                    lst_variance=None if variance is None else [variance],
                    polarization_factor=polarization_factor,
                    normalization_factor=[monitor],
                    lst_mask=None if frame_mask is None else [frame_mask],
                    lst_flat=None if flat is None else [flat],
                    method=method,
                )
                current_shape = (npt_azim, npt_rad)
            result = _drop_unsplit_bins(result)
            profile = None
            if profile_npt is not None:
                profile = MultiGeometry(
                    [integrator], unit=profile_unit, radial_range=profile_radial, azimuth_range=profile_azimuth,
                    threadpoolsize=0,
                ).integrate1d(
                    [image],
                    profile_npt,
                    correctSolidAngle=correct_solid_angle,
                    error_model=error_model if variance is None else None,
                    lst_variance=None if variance is None else [variance],
                    polarization_factor=polarization_factor,
                    normalization_factor=[monitor],
                    lst_mask=None if frame_mask is None else [frame_mask],
                    lst_flat=None if flat is None else [flat],
                    method=method,
                )
                if any(np.asarray(values).shape != (profile_npt,)
                       for values in (profile.sum_signal, profile.sum_normalization, profile.count)):
                    raise ValueError("pyFAI returned an unexpected Stitch profile shape")
                profile = _drop_unsplit_bins(profile)
            current_signal = np.asarray(result.sum_signal)
            current_normalization = np.asarray(result.sum_normalization)
            current_count = np.asarray(result.count)
            if any(
                values.shape != current_shape
                for values in (
                    current_signal,
                    current_normalization,
                    current_count,
                )
            ):
                raise ValueError("pyFAI returned an unexpected Stitch accumulator shape")
            for value, label in (
                (result.sum_variance, "variance"),
                (result.sum_normalization2, "squared normalization"),
                (result.sigma, "sigma"),
            ):
                if value is not None and np.asarray(value).shape != current_shape:
                    raise ValueError(
                        f"pyFAI returned an unexpected Stitch {label} shape"
                    )
            # the cuts take I(q)'s points (a 2-D Stitch's I(q) beside its map); the GI map its own
            gi_part = ()
            if gi is not None:
                with numexpr_lock:  # its fiber units compile and evaluate numexpr formulas
                    gi_part = _gi_frame_sums(
                        integrator, image, gi_angles[index], gi, gi_bins,
                        npt_rad if mode == "1d" else profile_npt,
                        mask=frame_mask, correct_solid_angle=correct_solid_angle,
                        polarization_factor=polarization_factor, dark=frame_dark, flat=flat, monitor=monitor,
                        map_npt=gi_map_npt)
            total = None
            if frame_screen is not None:
                # Whole-frame monitor-normalised total over valid pixels: the
                # in-range signal legitimately shrinks at the output range edges.
                valid = np.isfinite(image) & (image >= 0)
                if frame_mask is not None:
                    valid &= ~frame_mask
                signal = float(image[valid].sum())
                if frame_dark is not None:
                    # A dropped frame on a detector with a pedestal still holds the dark.
                    signal -= float(frame_dark[valid].sum())
                total = (float('nan') if statistics is not None and not statistics.exposure_multiplier
                         else signal / monitor)
            return (result, profile, gi_part, total, np.asarray(result.radial, dtype=float),
                    np.asarray(result.azimuthal, dtype=float) if mode == "2d" else None)
        finally:
            integrator.reset(collect_garbage=False)

    def absorb(result, gi_part, profile):
        """Add one kept frame's sums to the merged ones."""
        nonlocal aggregate, profile_aggregate
        with numexpr_lock:  # pyFAI's union recomputes the means with numexpr
            aggregate = result if aggregate is None else aggregate.union(result)
            if profile is not None:
                profile_aggregate = profile if profile_aggregate is None else profile_aggregate.union(profile)
        add_gi(gi_part)

    merged = 0

    def merge(part):
        """Merge one frame's sums, in frame order."""
        nonlocal radial, azimuthal, merged
        result, profile, gi_part, total, current_radial, current_azimuthal = part
        if radial is None:
            radial = current_radial.copy()
        elif not np.array_equal(radial, current_radial):
            raise ValueError("pyFAI returned inconsistent Stitch radial axes")
        if mode == "2d":
            if azimuthal is None:
                azimuthal = current_azimuthal.copy()
            elif not np.array_equal(azimuthal, current_azimuthal):
                raise ValueError("pyFAI returned inconsistent Stitch azimuth axes")
        if frame_screen is None:
            absorb(copy.deepcopy(result) if aggregate is None else result, gi_part,
                   None if profile is None else
                   copy.deepcopy(profile) if profile_aggregate is None else profile)
        else:
            # Hold this frame until its row neighbours are known; merge the
            # released frames in their original order.
            kept_profile = None if profile is None else copy.deepcopy(profile)
            for (held, held_gi, held_profile), verdict in frame_screen.push(
                    total, (copy.deepcopy(result), gi_part, kept_profile)):
                if not verdict.excluded:
                    absorb(held, held_gi, held_profile)
        merged += 1
        if progress is not None:
            progress(merged)

    if workers == 1:
        for index in range(image_count):
            image, integrator, statistics = read(index)
            merge(frame_part(index, image, integrator, statistics))
            # Release the detector-sized frame before asking the source for its
            # successor.  The aggregate retains only pyFAI's O(nbin) result arrays.
            del image, integrator, statistics
    else:
        parallel_frames(image_count, read, frame_part, merge, workers)
    try:
        next(image_iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("streaming Stitch received more images than declared")
    try:
        next(integrator_iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("streaming Stitch received more integrators than declared")
    if frame_screen is not None:
        for (held, held_gi, held_profile), verdict in frame_screen.finish():
            if not verdict.excluded:
                absorb(held, held_gi, held_profile)
        if aggregate is None:
            raise ValueError("every Stitch frame was excluded as a dropped frame")
    if aggregate is None or radial is None:
        raise ValueError("streaming Stitch produced no result")
    return aggregate, () if gi_totals is None else tuple(gi_totals), profile_aggregate


def _drop_unsplit_bins(result):
    """``result`` (one frame's pyFAI result) without the bins pyFAI could not split into: its pixel
    splitting can leave a bin's coverage, weight and signal NaN together, a pixel meeting the top of the
    radial range with no area inside it (LaB6 scan 14 in 2-D: 12 of 287 frames, one bin each, all the last
    q bin). Such a bin holds nothing from this frame, so the other frames' data in it stand; merged as
    pyFAI merges, the NaN would erase them."""
    unsplit = np.isnan(result.count) | np.isnan(result.sum_normalization)
    if not unsplit.any():
        return result
    for values in (result.sum_signal, result.sum_normalization, result.count, result.sum_variance,
                   result.sum_normalization2):
        if values is not None:
            values[unsplit] = 0.0
    return result.__recalculate_means__()


def _gi_frame_sums(integrator, image, angles, gi, bins, npt, *, mask, correct_solid_angle,
                   polarization_factor, dark, flat, monitor, map_npt=None):
    """One frame's GI cuts as pyFAI's fiber sums, each [signal, normalization × pixel area/dist²,
    count, axis] in ``gi.cuts`` order, accumulated across frames as pyFAI's MultiGeometryFiber does.

    pyFAI's own 1-D fiber cut sums its 2-D q_ip × q_oop histogram over one axis. Here each cut is a
    thin 2-D pass, its own axis in ``npt`` bins and the other coordinate's range in ONE bin, so the
    scratch is O(npt), not npt² (Codex audit G-F1: an unbounded square). A pixel counts in a cut
    only inside the GI window (``_gi_window_mask``). Both passes share one fiber integrator, which
    keeps its per-pixel coordinates, and one pair of fiber units: pyFAI compiles a unit's formula
    whenever its angles change, so both passes and the window take these rather than each parsing
    its own. ``angles`` are the frame's (incident angle, tilt) in degrees
    (``StitchGI.pyfai_angles``); ``bins`` both coordinates' ranges.

    ``map_npt`` (q_ip points, q_oop points) also sums a GI Stitch 2D's q_ip-q_oop map, pyFAI's full
    2-D fiber histogram over the same ranges, window and weights, as the last sums [signal,
    normalization × pixel area/dist², count, (q_ip axis, q_oop axis)] in the (q_ip, q_oop) orientation.
    """
    from pyFAI.units import get_unit_fiber
    from xrd_tools.integrate.gid import _effective_gi_params, _fiber_from_integrator
    incidence, tilt = angles
    fiber = _fiber_from_integrator(integrator, incident_angle=incidence, tilt_angle=tilt,
                                   sample_orientation=gi.sample_orientation)
    angle, tilt, orientation = _effective_gi_params(fiber, None, None, None)
    units = {cut: get_unit_fiber(GI_UNITS[cut], incident_angle=angle, tilt_angle=tilt,
                                 sample_orientation=orientation) for cut in GI_CUTS}
    # The configuration integrate2d_fiber sets (its angles modulo a turn), so it keeps the geometry.
    fiber.reset_integrator(**units["q_ip"].get_config_shared())
    scale = fiber.pixel1 * fiber.pixel2 / fiber.dist ** 2 if correct_solid_angle else 1.0
    mask = _gi_window_mask(fiber, image.shape, gi, units, mask)
    sums = []
    for cut in gi.cuts:
        in_plane = cut == "q_ip"
        result = fiber.integrate2d_fiber(
            image, npt_ip=npt if in_plane else 1, unit_ip=units["q_ip"], ip_range=bins["q_ip"],
            npt_oop=1 if in_plane else npt, unit_oop=units["q_oop"], oop_range=bins["q_oop"],
            correctSolidAngle=correct_solid_angle, mask=mask, polarization_factor=polarization_factor,
            dark=dark, flat=flat, method=("no", "histogram", "cython"), normalization_factor=monitor,
            incident_angle=angle, tilt_angle=tilt, sample_orientation=orientation)
        axis = -2 if in_plane else -1  # the other coordinate's one bin, as integrate_fiber sums
        sums.append([np.asarray(result.sum_signal, dtype=np.float64).sum(axis=axis),
                     np.asarray(result._sum_normalization, dtype=np.float64).sum(axis=axis) * scale,
                     np.asarray(result.count, dtype=np.float64).sum(axis=axis),
                     np.asarray(result.inplane if in_plane else result.outofplane, dtype=np.float64)])
    if map_npt is not None:
        result = fiber.integrate2d_fiber(
            image, npt_ip=map_npt[0], unit_ip=units["q_ip"], ip_range=bins["q_ip"],
            npt_oop=map_npt[1], unit_oop=units["q_oop"], oop_range=bins["q_oop"],
            correctSolidAngle=correct_solid_angle, mask=mask, polarization_factor=polarization_factor,
            dark=dark, flat=flat, method=("no", "histogram", "cython"), normalization_factor=monitor,
            incident_angle=angle, tilt_angle=tilt, sample_orientation=orientation)
        # pyFAI's (q_oop, q_ip), the container's (q_ip, q_oop)
        sums.append([np.asarray(result.sum_signal, dtype=np.float64).T,
                     np.asarray(result._sum_normalization, dtype=np.float64).T * scale,
                     np.asarray(result.count, dtype=np.float64).T,
                     (np.asarray(result.inplane, dtype=np.float64), np.asarray(result.outofplane, dtype=np.float64))])
    return tuple(sums)


def _gi_window_mask(fiber, shape, gi, units, mask):
    """``mask`` (None: the detector's) with every pixel outside the GI window masked: a pixel counts
    only when each coordinate with an explicit range lies inside it, bounds included, as the xu_hist
    kernel windows its cuts.

    pyFAI's histogram truncates a fractional bin index in (-1, 0) to bin 0 and stretches the top edge
    to take the maximum, so a range admits pixels up to one bin below it (a thin pass's one-bin
    coordinate: its whole width) and a hair above it (Codex final review F1). An Auto range is the
    span of every pixel inside the other window (``_gi_bins``), so it needs no mask and an Auto cut is
    unchanged. ``units`` are the passes' fiber units: the coordinates are the ones they bin, and
    pyFAI's geometry cache keeps them for the passes rather than recomputing.
    """
    windows = [(cut, gi.bins(cut)) for cut in GI_CUTS if gi.bins(cut) is not None]
    if not windows:
        return mask
    outside = np.zeros(shape, dtype=bool)
    for cut, (low, high) in windows:
        values = fiber.array_from_unit(shape, "center", units[cut], scale=True)
        outside |= ~((values >= low) & (values <= high))
    return np.asarray(mask_with_detector(fiber, outside if mask is None else np.logical_or(mask, outside)),
                      dtype=bool)


def _gi_bins(integrator_factory, gi_angles, gi):
    """Both coordinates' ranges (the one 2-D pass needs both, whichever cuts are written): each its
    own range, else its full extent inside the other's range over every frame (one pass, no image
    reads)."""
    bins = {cut: gi.bins(cut) for cut in GI_CUTS}
    scouted = [cut for cut in GI_CUTS if bins[cut] is None]
    if not scouted:
        return bins
    from pyFAI.units import get_unit_fiber
    from xrd_tools.integrate.gid import _fiber_from_integrator
    low, high = dict.fromkeys(scouted, np.inf), dict.fromkeys(scouted, -np.inf)
    for integrator, (incidence, tilt) in zip(integrator_factory(), gi_angles):
        try:
            with pyfai_numexpr.PYFAI_NUMEXPR_LOCK:  # fiber units compile and evaluate numexpr formulas
                fiber = _fiber_from_integrator(integrator, incident_angle=float(incidence), tilt_angle=float(tilt),
                                               sample_orientation=gi.sample_orientation)
                unit = dict(incident_angle=float(incidence), tilt_angle=float(tilt),
                            sample_orientation=gi.sample_orientation, angle_unit="deg")
                shape = fiber.detector.shape
                q_ip = np.asarray(fiber.array_from_unit(
                    shape, "center", get_unit_fiber("qip_A^-1", **unit), scale=True))
                q_oop = np.asarray(fiber.array_from_unit(
                    shape, "center", get_unit_fiber("qoop_A^-1", **unit), scale=True))
                for cut in scouted:
                    values, other = (q_ip, q_oop) if cut == "q_ip" else (q_oop, q_ip)
                    inside = np.isfinite(values)
                    window = gi.window(cut)
                    if window is not None:
                        inside &= (other >= window[0]) & (other <= window[1])
                    if inside.any():
                        low[cut] = min(low[cut], float(values[inside].min()))
                        high[cut] = max(high[cut], float(values[inside].max()))
        finally:
            integrator.reset(collect_garbage=False)
    for cut in scouted:
        if not low[cut] < high[cut]:
            raise ValueError("no pixel lies inside the GI window")
        bins[cut] = (low[cut], high[cut])
    return bins


def gi_map_result(totals):
    """A GI Stitch 2D's q_ip-q_oop map, (q_ip, q_oop): Σ signal / Σ normalization where a bin has weight
    (NaN elsewhere, as the cuts), and its diagnostics."""
    signal, normalization, count, (q_ip, q_oop) = totals
    intensity = np.full(normalization.shape, np.nan, dtype=np.float64)
    occupied = normalization > 0
    intensity[occupied] = signal[occupied] / normalization[occupied]
    return (IntegrationResult2D(q_ip, q_oop, intensity, None, GI_UNITS["q_ip"], GI_UNITS["q_oop"]),
            StitchDiagnostics(count, normalization))


def gi_cut_result(totals, unit):
    """The GI cut's Σ signal / Σ normalization where a bin has weight (NaN elsewhere), and its
    diagnostics."""
    signal, normalization, count, axis = totals
    intensity = np.full(len(axis), np.nan, dtype=np.float64)
    occupied = normalization > 0
    intensity[occupied] = signal[occupied] / normalization[occupied]
    return IntegrationResult1D(axis, intensity, None, unit), StitchDiagnostics(count, normalization)


def stitch_1d_streaming(
    images: Iterable[np.ndarray],
    image_count: int,
    integrator_factory: Callable[[], Iterator[AzimuthalIntegrator]],
    npt: int = 1000,
    unit: str = "q_A^-1",
    method: str = "BBox",
    radial_range: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
    normalization: np.ndarray | Sequence[float] | None = None,
    correct_solid_angle: bool = True,
    error_model: str | None = None,
    polarization_factor: float | None = None,
    frame_screen: Any = None,
    dark: np.ndarray | None = None,
    flat: np.ndarray | None = None,
    flat_min_fraction: float = DEFAULT_MIN_FLAT_FRACTION,
    gi=None,
    gi_angles: Sequence[tuple[float, float]] | None = None,
    workers: int = 1,
    progress: Callable[[int], object] | None = None,
    azimuth_range: tuple[float, float] | None = None,
) -> tuple[IntegrationResult1D, StitchDiagnostics,
           tuple[tuple[IntegrationResult1D, StitchDiagnostics], ...]]:
    """Sequential MultiGeometry merge that retains only one detector frame per worker
    (``workers`` frames integrated at once, merged in frame order: the same result). ``azimuth_range``
    (°) is I(q)'s χ window, as MultiGeometry's (None: every χ); the GI cuts keep their own windows.

    Returns (I(q), diagnostics, the GI cuts): with ``gi`` (a ``StitchGI``) and each frame's pyFAI
    (incident angle, tilt) in ``gi_angles``, each of its cuts (``gi_cut_result``, in ``gi.cuts`` order) is binned with ``npt`` points
    from the same frames by pyFAI's fiber model; otherwise the cuts are ().
    """

    result, gi_totals, _profile = _streaming_multigeometry(
        images,
        image_count,
        integrator_factory,
        mode="1d",
        npt_rad=npt,
        npt_azim=1,
        unit=unit,
        method=method,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        mask=mask,
        normalization=normalization,
        correct_solid_angle=correct_solid_angle,
        error_model=error_model,
        polarization_factor=polarization_factor,
        frame_screen=frame_screen,
        dark=dark,
        flat=flat,
        flat_min_fraction=flat_min_fraction,
        gi=gi,
        gi_angles=gi_angles,
        workers=workers,
        progress=progress,
    )
    payload, diagnostics = _stitch_1d_payload(result, unit)
    return payload, diagnostics, tuple(gi_cut_result(totals, GI_UNITS[cut])
                                       for cut, totals in zip(() if gi is None else gi.cuts, gi_totals))


def _stitch_1d_payload(result, unit: str) -> tuple[IntegrationResult1D, StitchDiagnostics]:
    """Stitch I(q), uncertainty, coverage and weight; uncovered 1-D bins are NaN."""
    sigma = result.sigma if result.sigma is not None else None
    radial_unit = str(result.unit) if result.unit is not None else unit
    payload = IntegrationResult1D(
        radial=np.asarray(result.radial, dtype=float),
        intensity=_nan_empty_1d(result),
        sigma=_nan_empty_1d(result, sigma) if sigma is not None else None,
        unit=radial_unit,
    )
    return payload, StitchDiagnostics(result.count, result.sum_normalization)


#: pyFAI's 2-D pixel splitting can leave a bin a slightly negative coverage and weight, round-off of
#: about a millionth of a pixel (LaB6 scan 14, 287 frames merged: 2 of 360,000 bins, coverage down to
#: -7.6e-7 px and weight to -1.3e-13, where the largest bins hold 1984 px and 3.0e-4). A bin negative
#: by no more than this fraction of the map's largest holds round-off only; a larger negative is still
#: refused (``StitchDiagnostics``).
_SPLIT_ROUNDOFF = 1e-6


def _stitch_2d_payload(result) -> tuple[IntegrationResult2D, StitchDiagnostics]:
    """A merged 2-D result as Stitch returns it, in the container's (radial, azimuthal) orientation: the
    map and its per-bin coverage and weight, a bin holding only negative pixel-splitting round-off made
    empty as pyFAI leaves a bin nothing reaches (no coverage or weight, zero intensity)."""
    coverage = np.array(result.count, dtype=np.float64).T
    weight = np.array(result.sum_normalization, dtype=np.float64).T
    intensity = np.array(result.intensity, dtype=float).T
    sigma = None if result.sigma is None else np.array(result.sigma, dtype=float).T
    roundoff = (((coverage < 0) & (-coverage <= _SPLIT_ROUNDOFF * np.abs(coverage).max()))
                | ((weight < 0) & (-weight <= _SPLIT_ROUNDOFF * np.abs(weight).max())))
    coverage[roundoff] = weight[roundoff] = intensity[roundoff] = 0.0
    if sigma is not None:
        sigma[roundoff] = 0.0
    payload = IntegrationResult2D(
        radial=np.asarray(result.radial, dtype=float),
        azimuthal=np.asarray(result.azimuthal, dtype=float),
        intensity=intensity,
        sigma=sigma,
        unit=str(result.radial_unit),
        azimuthal_unit=str(result.azimuthal_unit),
    )
    return payload, StitchDiagnostics(coverage, weight)


def stitch_2d_streaming(
    images: Iterable[np.ndarray],
    image_count: int,
    integrator_factory: Callable[[], Iterator[AzimuthalIntegrator]],
    npt_rad: int = 1000,
    npt_azim: int = 1000,
    unit: str = "q_A^-1",
    method: str = "BBox",
    radial_range: tuple[float, float] | None = None,
    azimuth_range: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
    normalization: np.ndarray | Sequence[float] | None = None,
    correct_solid_angle: bool = True,
    error_model: str | None = None,
    polarization_factor: float | None = None,
    frame_screen: Any = None,
    dark: np.ndarray | None = None,
    flat: np.ndarray | None = None,
    flat_min_fraction: float = DEFAULT_MIN_FLAT_FRACTION,
    workers: int = 1,
    progress: Callable[[int], object] | None = None,
) -> tuple[IntegrationResult2D, StitchDiagnostics]:
    """Sequential 2-D MultiGeometry merge retaining one detector frame per worker (``workers`` as
    :func:`stitch_1d_streaming`)."""

    result, _gi, _profile = _streaming_multigeometry(
        images,
        image_count,
        integrator_factory,
        mode="2d",
        npt_rad=npt_rad,
        npt_azim=npt_azim,
        unit=unit,
        method=method,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        mask=mask,
        normalization=normalization,
        correct_solid_angle=correct_solid_angle,
        error_model=error_model,
        polarization_factor=polarization_factor,
        frame_screen=frame_screen,
        dark=dark,
        flat=flat,
        flat_min_fraction=flat_min_fraction,
        workers=workers,
        progress=progress,
    )
    return _stitch_2d_payload(result)


def stitch_2d_with_1d_streaming(
    images: Iterable[np.ndarray],
    image_count: int,
    integrator_factory: Callable[[], Iterator[AzimuthalIntegrator]],
    npt_rad: int = 1000,
    npt_azim: int = 1000,
    npt_1d: int = 1000,
    unit: str = "q_A^-1",
    method: str = "BBox",
    radial_range: tuple[float, float] | None = None,
    azimuth_range: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
    normalization: np.ndarray | Sequence[float] | None = None,
    correct_solid_angle: bool = True,
    error_model: str | None = None,
    polarization_factor: float | None = None,
    frame_screen: Any = None,
    dark: np.ndarray | None = None,
    flat: np.ndarray | None = None,
    flat_min_fraction: float = DEFAULT_MIN_FLAT_FRACTION,
    gi=None,
    gi_angles: Sequence[tuple[float, float]] | None = None,
    unit_1d: str | None = None,
    radial_range_1d: tuple[float, float] | None = None,
    workers: int = 1,
    progress: Callable[[int], object] | None = None,
    azimuth_range_1d: tuple[float, float] | None = None,
) -> tuple:
    """The 2-D map and I(q) beside it, as an Int 2D record holds them, from one pass over the frames:
    the map as :func:`stitch_2d_streaming` makes it, and I(q) over every χ in ``npt_1d`` bins as
    :func:`stitch_1d_streaming` makes it from the same frames, in the map's unit and radial range, or
    with ``unit_1d`` in its own and ``radial_range_1d`` (None: its frames' extent), as Int's two
    integrations; ``azimuth_range_1d`` (°) is I(q)'s χ window, as a 1-D Stitch's (None: every χ). With ``gi`` and every frame's (incidence, tilt) in ``gi_angles``, also its cuts as
    :func:`stitch_1d_streaming` makes them and the q_ip-q_oop map (``npt_rad`` × ``npt_azim`` over the
    cuts' ranges). ``workers`` as :func:`stitch_1d_streaming`.

    Returns (map, its diagnostics, I(q), its diagnostics, the GI cuts or (), the GI map or None)."""

    result, gi_totals, profile = _streaming_multigeometry(
        images,
        image_count,
        integrator_factory,
        mode="2d",
        npt_rad=npt_rad,
        npt_azim=npt_azim,
        unit=unit,
        method=method,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        mask=mask,
        normalization=normalization,
        correct_solid_angle=correct_solid_angle,
        error_model=error_model,
        polarization_factor=polarization_factor,
        frame_screen=frame_screen,
        dark=dark,
        flat=flat,
        flat_min_fraction=flat_min_fraction,
        profile_npt=npt_1d,
        gi=gi,
        gi_angles=gi_angles,
        gi_map_npt=None if gi is None else (npt_rad, npt_azim),
        profile_unit=unit_1d,
        profile_radial_range=radial_range_1d,
        profile_azimuth_range=azimuth_range_1d,
        workers=workers,
        progress=progress,
    )
    cuts = tuple(gi_cut_result(totals, GI_UNITS[cut]) for cut, totals in zip(() if gi is None else gi.cuts, gi_totals))
    gi_map = None if gi is None else gi_map_result(gi_totals[-1])
    return (*_stitch_2d_payload(result), *_stitch_1d_payload(profile, unit if unit_1d is None else unit_1d),
            cuts, gi_map)


@dataclass(frozen=True, slots=True)
class PONIIntegratorSeries:
    """Repeatable, lazy legacy PONI-plus-angle integrator series."""

    base_poni: PONI
    rotations: tuple[tuple[float, float], ...]

    def __len__(self) -> int:
        return len(self.rotations)

    def __iter__(self) -> Iterator[AzimuthalIntegrator]:
        for rot1, rot2 in self.rotations:
            integrator = poni_to_integrator(self.base_poni)
            integrator.rot1 = rot1
            integrator.rot2 = rot2
            yield integrator


def create_multigeometry_integrator_series(
    base_poni: PONI,
    rot1_angles: np.ndarray | Sequence[float],
    rot2_angles: np.ndarray | Sequence[float] | None = None,
) -> PONIIntegratorSeries:
    """
    Build a per-image list of AzimuthalIntegrators for a detector-angle scan.

    Each integrator starts from ``base_poni`` and has its ``rot1`` (and
    optionally ``rot2``) shifted by the corresponding scan angle.

    Parameters
    ----------
    base_poni : PONI
        Calibration geometry at the zero-angle detector position.
    rot1_angles : array-like of float
        Per-image in-plane detector rotation offsets **in degrees**
        (e.g. the ``del`` / ``tth`` motor values).
    rot2_angles : array-like of float or None, optional
        Per-image out-of-plane detector rotation offsets **in degrees**
        (e.g. the ``nu`` motor values).  If ``None``, only ``rot1`` varies.

    Iterating the result creates fresh integrators, so callers can make a
    bounded geometry-scout pass and then integrate one frame at a time.
    """
    rot1 = np.asarray(rot1_angles, dtype=float)
    rot2 = np.zeros_like(rot1) if rot2_angles is None else np.asarray(rot2_angles, dtype=float)
    if rot1.shape != rot2.shape:
        raise ValueError(
            f"rot1_angles length {rot1.shape} != rot2_angles length {rot2.shape}"
        )

    # Fail loud on a non-finite rotation rather than letting pyFAI silently
    # collapse the geometry (NaN rot → qmax≈0, that frame dropped with no error).
    # Mirrors the calibrated path (create_multigeometry_integrators_from_geometry):
    # a CompositeFrameSource NaN-pads a member lacking the rotation motor, so the
    # rot1_key/rot2_key column is present-but-NaN and slips past the missing-key
    # guard upstream.  This is the legacy (base_poni + rot*_key) twin of that fix.
    for _name, _vals, _supplied in (("rot1", rot1, True),
                                    ("rot2", rot2, rot2_angles is not None)):
        if _supplied and not np.all(np.isfinite(_vals)):
            bad = np.flatnonzero(~np.isfinite(_vals)).tolist()
            raise ValueError(
                f"{_name}_angles has non-finite value(s) at frame position(s) "
                f"{bad[:10]}{' …' if len(bad) > 10 else ''} — a grouped/composite "
                f"source is NaN-padding a member that lacks this detector-rotation "
                f"motor.  Every member of a multi-source stitch must provide it.")

    base_ai = poni_to_integrator(base_poni)
    base_rot1 = float(base_ai.rot1)
    base_rot2 = float(base_ai.rot2)

    base_ai.reset(collect_garbage=False)
    rotations = tuple(
        (
            base_rot1 + float(np.deg2rad(r1_deg)),
            base_rot2 + float(np.deg2rad(r2_deg)),
        )
        for r1_deg, r2_deg in zip(rot1, rot2)
    )

    logger.debug(
        "Prepared %d lazy per-angle integrators (rot2_varied=%s)",
        len(rotations),
        rot2_angles is not None,
    )
    return PONIIntegratorSeries(base_poni, rotations)


def create_multigeometry_integrators(
    base_poni: PONI,
    rot1_angles: np.ndarray | Sequence[float],
    rot2_angles: np.ndarray | Sequence[float] | None = None,
) -> list[AzimuthalIntegrator]:
    """Materialize the legacy PONI-plus-angle integrator series."""

    return list(
        create_multigeometry_integrator_series(
            base_poni,
            rot1_angles=rot1_angles,
            rot2_angles=rot2_angles,
        )
    )


@dataclass(frozen=True, slots=True)
class GeometryIntegratorSeries:
    """Repeatable, lazy per-frame pyFAI integrator series."""

    calibration: Any
    rotations: tuple[tuple[float, float, float], ...]
    #: Per-frame (dist, poni1, poni2) when the sample is off the rotation centre.
    translations: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.translations is not None and len(self.translations) != len(self.rotations):
            raise ValueError("per-frame translations and rotations differ in length")

    def __len__(self) -> int:
        return len(self.rotations)

    def __iter__(self) -> Iterator[AzimuthalIntegrator]:
        from xrd_tools.integrate.calibration import (  # noqa: PLC0415
            detector_calibration_to_integrator,
        )

        for index, (rot1, rot2, rot3) in enumerate(self.rotations):
            position = {} if self.translations is None else dict(
                zip(("dist", "poni1", "poni2"), self.translations[index]))
            yield detector_calibration_to_integrator(
                self.calibration,
                rot1=rot1,
                rot2=rot2,
                rot3=rot3,
                **position,
            )


def create_multigeometry_integrator_series_from_geometry(
    diffractometer: Any,
    motors: Any,
    *,
    base_calibration: Any = None,
) -> GeometryIntegratorSeries:
    """Build a repeatable lazy calibrated-integrator series.

    Closes stitching GAP A + GAP B vs :func:`create_multigeometry_integrators`:

    * **GAP A (fitted scales, not a hardwired ``deg2rad``):** per-frame rotations
      come from ``diffractometer.to_pyfai_per_frame(motors)``, so a *calibrated*
      goniometer's fitted per-axis scales/offsets are used (an uncalibrated preset
      whose ``AngleMapping.sign == 1`` reduces exactly to the old ``deg2rad`` path).
    * **GAP B (panel mount preserved):** the base geometry is a
      :class:`DetectorCalibration` carrying ``Detector_config`` (orientation), built
      via :func:`detector_calibration_to_integrator` — the orientation is no longer
      silently dropped.

    Per-frame ``rotN = base.poni.rotN + to_pyfai_per_frame()[rotN]`` (the same
    decomposition :meth:`Diffractometer.from_pyfai_goniometer` produces).

    Parameters
    ----------
    diffractometer : Diffractometer
        The instrument geometry (preset-built or gonio-fitted).
    motors : Mapping[str, array-like]
        Per-frame motor columns (degrees), keyed by motor name.
    base_calibration : DetectorCalibration, optional
        The base detector calibration; defaults to
        ``diffractometer.calibration``.

    Iterating the result creates fresh integrators.  This lets the streaming
    Stitch path make a bounded geometry-scout pass and then integrate one frame
    without retaining every pyFAI geometry cache.
    """
    cal = base_calibration if base_calibration is not None else getattr(
        diffractometer, "calibration", None)
    if cal is None:
        raise ValueError(
            "no DetectorCalibration: pass base_calibration= or use a "
            "Diffractometer carrying one (from_pyfai_goniometer / a fitted "
            "geometry). The base dist/poni/Detector_config is required to stitch.")

    # A coupled model owns all Euler rotations and still needs no GI incidence.
    if getattr(diffractometer, 'stacked_arm', None) is not None:
        if any(getattr(cal.poni, axis) != 0 for axis in ('rot1', 'rot2', 'rot3')):
            raise ValueError('stacked arm requires zero base PONI rotations')
        arm = diffractometer.stacked_arm
        values = arm.rotations(motors)
        rotations = tuple(zip(values['rot1'], values['rot2'], values['rot3']))
        return GeometryIntegratorSeries(cal, rotations, arm.translations(motors, cal.poni))

    # Compute ONLY the per-frame detector rotations (rot1/2/3) directly — a stitch
    # never needs the GI incidence, so a stitch scan that lacks the incidence motor
    # (e.g. psic's `eta`) must not crash; to_pyfai_per_frame stays strict for GI.
    def _frame_rot(mapping: Any) -> np.ndarray | None:
        if not mapping.is_active:
            return None
        if mapping.source_motor not in motors:
            raise KeyError(
                f"stitch geometry needs motor {mapping.source_motor!r} (it drives "
                f"a detector rotation) but the source provides {sorted(motors)}")
        col = np.asarray(motors[mapping.source_motor], dtype=float)
        # Fail loud on a non-finite rotation rather than letting pyFAI silently
        # collapse the geometry (NaN rot → qmax≈0, frames dropped with no error).
        # This is the multi-source-stitch footgun: a CompositeFrameSource NaN-pads
        # a member that lacks this detector-rotation motor (so the key is present
        # but the column carries NaN), which slips past the missing-key guard above.
        if not np.all(np.isfinite(col)):
            bad = np.flatnonzero(~np.isfinite(col)).tolist()
            raise ValueError(
                f"stitch geometry motor {mapping.source_motor!r} has non-finite "
                f"value(s) at frame position(s) {bad[:10]}"
                f"{' …' if len(bad) > 10 else ''} — a grouped/composite source is "
                f"NaN-padding a member that lacks this detector-rotation motor. "
                f"Every member of a multi-source stitch must provide it.")
        return np.deg2rad(mapping.apply(col))

    r1 = _frame_rot(diffractometer.rot1)
    r2 = _frame_rot(diffractometer.rot2)
    r3 = _frame_rot(diffractometer.rot3)
    nframes = next((len(r) for r in (r1, r2, r3) if r is not None), None)
    if nframes is None:  # no active detector rotation — use any motor column length
        col = next(iter(motors.values()), None)
        nframes = len(np.atleast_1d(np.asarray(col))) if col is not None else 1
    if any(len(values) != nframes for values in (r1, r2, r3) if values is not None):
        raise ValueError("stitch geometry motor columns have different lengths")
    base = cal.poni
    rotations: list[tuple[float, float, float]] = []
    for i in range(nframes):
        rotations.append(
            (
                float(base.rot1) + (float(r1[i]) if r1 is not None else 0.0),
                float(base.rot2) + (float(r2[i]) if r2 is not None else 0.0),
                float(base.rot3) + (float(r3[i]) if r3 is not None else 0.0),
            )
        )
    series = GeometryIntegratorSeries(cal, tuple(rotations))
    logger.debug("Prepared %d lazy per-frame integrators from a Diffractometer "
                         "(preset=%s)", len(series),
                 getattr(diffractometer, "preset", "?"))
    return series


def create_multigeometry_integrators_from_geometry(
    diffractometer: Any,
    motors: Any,
    *,
    base_calibration: Any = None,
) -> list[AzimuthalIntegrator]:
    """Materialize the calibrated integrator series for eager callers."""

    return list(
        create_multigeometry_integrator_series_from_geometry(
            diffractometer,
            motors,
            base_calibration=base_calibration,
        )
    )


def stitch_1d(
    images: list[np.ndarray] | np.ndarray,
    integrators: list[AzimuthalIntegrator],
    npt: int = 1000,
    unit: str = "q_A^-1",
    method: str = "BBox",
    radial_range: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
    normalization: np.ndarray | None = None,
    **kwargs: Any,
) -> IntegrationResult1D:
    """
    Stitch a list of images at different detector angles into a 1D pattern.

    Parameters
    ----------
    images : list of ndarray or 3-D ndarray
        Per-image detector frames, one per integrator.
    integrators : list of AzimuthalIntegrator
        Per-image integrators from :func:`create_multigeometry_integrators`.
    npt : int, optional
        Number of radial bins.
    unit : str, optional
        Radial unit, e.g. ``"q_A^-1"``, ``"2th_deg"``.
    method : str, optional
        Integration method.  Default is ``"BBox"``; MultiGeometry works best
        with histogram-based methods.
    radial_range : tuple of float or None, optional
        ``(min, max)`` radial range applied at MultiGeometry construction.
    mask : ndarray or None, optional
        Single detector mask applied to every image.
    normalization : array-like of float or None, optional
        Per-image monitor counts passed to pyFAI's ``normalization_factor``.
        MultiGeometry therefore accumulates raw signal over monitor-weighted
        normalization, matching its notebook/API convention.
    **kwargs
        Extra keyword arguments forwarded to ``mg.integrate1d``.

    Returns
    -------
    IntegrationResult1D
    """
    from pyFAI.multi_geometry import MultiGeometry

    pyfai_numexpr.guard_result_means()

    img_list = _prepare_images(images, None)
    factors = _normalization_factors(normalization, len(img_list))
    if "normalization_factor" in kwargs:
        raise ValueError(
            "pass monitor counts through normalization, not normalization_factor"
        )
    # Geometric gaps stay masked per-integrator even with an explicit mask
    # (with lst_mask=None pyFAI already falls back to each detector mask).
    lst_mask = ([mask_with_detector(ai, mask) for ai in integrators]
                if mask is not None else None)

    mg = MultiGeometry(integrators, unit=unit, radial_range=radial_range)
    result = mg.integrate1d(
        img_list,
        npt,
        lst_mask=lst_mask,
        method=method,
        normalization_factor=factors,
        **kwargs,
    )

    return _stitch_1d_payload(result, unit)[0]


def stitch_2d(
    images: list[np.ndarray] | np.ndarray,
    integrators: list[AzimuthalIntegrator],
    npt_rad: int = 1000,
    npt_azim: int = 1000,
    unit: str = "q_A^-1",
    method: str = "BBox",
    radial_range: tuple[float, float] | None = None,
    azimuth_range: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
    normalization: np.ndarray | None = None,
    **kwargs: Any,
) -> IntegrationResult2D:
    """
    Stitch a list of images at different detector angles into a 2D cake.

    Parameters
    ----------
    images : list of ndarray or 3-D ndarray
        Per-image detector frames, one per integrator.
    integrators : list of AzimuthalIntegrator
        Per-image integrators from :func:`create_multigeometry_integrators`.
    npt_rad : int, optional
        Number of radial bins.
    npt_azim : int, optional
        Number of azimuthal bins.
    unit : str, optional
        Radial unit.
    method : str, optional
        Integration method.
    radial_range : tuple of float or None, optional
        ``(min, max)`` radial range applied at MultiGeometry construction.
    azimuth_range : tuple of float or None, optional
        ``(min, max)`` azimuthal range (degrees) applied at MultiGeometry
        construction.
    mask : ndarray or None, optional
        Single detector mask applied to every image.
    normalization : array-like of float or None, optional
        Per-image monitor counts passed to pyFAI's ``normalization_factor``.
    **kwargs
        Extra keyword arguments forwarded to ``mg.integrate2d``.

    Returns
    -------
    IntegrationResult2D
        Intensity has shape ``(npt_rad, npt_azim)`` (transposed from pyFAI).
    """
    from pyFAI.multi_geometry import MultiGeometry

    pyfai_numexpr.guard_result_means()

    img_list = _prepare_images(images, None)
    factors = _normalization_factors(normalization, len(img_list))
    if "normalization_factor" in kwargs:
        raise ValueError(
            "pass monitor counts through normalization, not normalization_factor"
        )
    # Geometric gaps stay masked per-integrator even with an explicit mask
    # (with lst_mask=None pyFAI already falls back to each detector mask).
    lst_mask = ([mask_with_detector(ai, mask) for ai in integrators]
                if mask is not None else None)

    mg = MultiGeometry(
        integrators,
        unit=unit,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
    )
    result = mg.integrate2d(
        img_list,
        npt_rad,
        npt_azim,
        lst_mask=lst_mask,
        method=method,
        normalization_factor=factors,
        **kwargs,
    )

    # pyFAI returns intensity (npt_azim, npt_rad); transpose to (npt_rad, npt_azim)
    intensity = np.asarray(result.intensity, dtype=float).T
    sigma = (
        np.asarray(result.sigma, dtype=float).T
        if result.sigma is not None
        else None
    )
    unit_str = (
        str(result.unit[0]) if isinstance(result.unit, tuple) else str(result.unit)
    )
    return IntegrationResult2D(
        radial=np.asarray(result.radial, dtype=float),
        azimuthal=np.asarray(result.azimuthal, dtype=float),
        intensity=intensity,
        sigma=sigma,
        unit=unit_str,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_images(
    images: list[np.ndarray] | np.ndarray,
    normalization: np.ndarray | Sequence[float] | None,
) -> list[np.ndarray]:
    """Convert images to a list and apply optional per-image normalisation."""
    if isinstance(images, np.ndarray):
        if images.ndim == 3:
            img_list: list[np.ndarray] = [images[i] for i in range(images.shape[0])]
        elif images.ndim == 2:
            img_list = [images]
        else:
            raise ValueError(f"images ndarray must be 2D or 3D, got shape {images.shape}")
    else:
        img_list = [np.asarray(im, dtype=float) for im in images]

    if normalization is not None:
        norm = np.asarray(normalization, dtype=float)
        if norm.shape != (len(img_list),):
            raise ValueError(
                f"normalization length {norm.shape} != number of images {len(img_list)}"
            )
        if not np.all(np.isfinite(norm)):
            raise ValueError("normalization contains non-finite (nan/inf) values")
        if np.any(norm <= 0):
            # zero divides by zero; a NEGATIVE monitor flips the frame's sign and
            # silently cancels healthy frames in the stitch (finite, plausible, wrong).
            raise ValueError(
                "normalization contains zero or negative values "
                f"(monitor must be > 0): {norm[norm <= 0].tolist()}")
        img_list = [img / n for img, n in zip(img_list, norm)]

    return img_list


def stitch_images(
    images: list[np.ndarray] | np.ndarray,
    base_poni: PONI,
    rot1_angles: np.ndarray | Sequence[float],
    rot2_angles: np.ndarray | Sequence[float] | None = None,
    *,
    mode: str = "1d",
    npt_1d: int = 2000,
    npt_rad_2d: int = 1500,
    npt_azim_2d: int = 720,
    unit: str = "q_A^-1",
    method: str = "BBox",
    radial_range: tuple[float, float] | None = None,
    azimuth_range: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
    normalization: np.ndarray | Sequence[float] | None = None,
    backend: str = "multigeometry",
) -> IntegrationResult1D | IntegrationResult2D:
    """Stitch a detector-angle image stack into a 1D or 2D pattern.

    High-level entry point that builds the per-image MultiGeometry
    integrators from ``base_poni`` + per-image ``rot1``/``rot2`` offsets
    (degrees) and merges them with the chosen ``backend``.
    This is the orchestration the xdart GUI used to carry inline; keeping
    it here lets headless callers stitch without reimplementing the
    integrator-build + dispatch.

    Parameters mirror :func:`stitch_1d` / :func:`stitch_2d`; ``mode``
    selects which.  ``rot2_angles`` that are all-zero (or ``None``) are
    treated as a pure ``rot1`` scan.

    ``backend``
        ``"multigeometry"`` (default) — pyFAI MultiGeometry, which applies
        its OWN solid-angle/polarization corrections and supports any unit.
        ``"pyfai_hist"`` — the per-pixel q-map histogram merge (Σraw/Σnorm)
        from :mod:`xrd_tools.integrate.stitch_hist`; it emits ``|q|`` in Å⁻¹
        ONLY (``unit='q_A^-1'``) and does its own pixel splitting (``method``
        is ignored).  This is the substrate the shared CorrectionStack + GI
        corrections plug into; GI-corrected stitching (footprint/Fresnel/
        refraction) layers on it via ``stitch_hist.pyfai_gi_q_frames`` and is
        NOT wired here.
    """
    # Fail early on a count mismatch — feeding MultiGeometry an unequal
    # number of images and integrators silently mispairs images with the
    # wrong detector angle (or raises deep inside pyFAI).
    rot1 = np.asarray(rot1_angles, dtype=float)
    # Count images the same way _prepare_images interprets them: a 3-D
    # ndarray is a stack (count = shape[0]); a 2-D ndarray is a single
    # image (count = 1, NOT shape[0]); anything else is a sequence.
    if isinstance(images, np.ndarray):
        n_images = images.shape[0] if images.ndim == 3 else 1
    else:
        n_images = len(images)
    if n_images != rot1.shape[0]:
        raise ValueError(
            f"stitch_images: {n_images} images != {rot1.shape[0]} angles; "
            "one detector angle is required per image."
        )

    rot2 = (
        rot2_angles
        if rot2_angles is not None and np.any(np.asarray(rot2_angles))
        else None
    )
    integrators = create_multigeometry_integrators(
        base_poni, rot1_angles=rot1_angles, rot2_angles=rot2,
    )
    if backend == "pyfai_hist":
        return _stitch_pyfai_hist(
            images, integrators, mode=mode, npt_1d=npt_1d,
            npt_rad_2d=npt_rad_2d, npt_azim_2d=npt_azim_2d, unit=unit,
            method=method, radial_range=radial_range,
            azimuth_range=azimuth_range, mask=mask, normalization=normalization)
    if backend != "multigeometry":
        raise ValueError(
            "stitch_images: backend must be 'multigeometry' or 'pyfai_hist', "
            f"got {backend!r}")
    if mode == "1d":
        return stitch_1d(
            images, integrators, npt=npt_1d, unit=unit, method=method,
            radial_range=radial_range, mask=mask, normalization=normalization,
        )
    if mode == "2d":
        return stitch_2d(
            images, integrators, npt_rad=npt_rad_2d, npt_azim=npt_azim_2d,
            unit=unit, method=method, radial_range=radial_range,
            azimuth_range=azimuth_range, mask=mask, normalization=normalization,
        )
    raise ValueError(f"mode must be '1d' or '2d', got {mode!r}")


def _stitch_pyfai_hist(
    images, integrators, *, mode, npt_1d, npt_rad_2d, npt_azim_2d, unit,
    method, radial_range, azimuth_range, mask, normalization,
):
    """The ``pyfai_hist`` merge: build per-frame pyFAI q-maps and accumulate
    them on a shared q-grid (Σraw/Σnorm), instead of pyFAI MultiGeometry.

    q-only (``unit='q_A^-1'``); the histogram does its own pixel splitting so
    ``method`` is ignored.  Mirrors the ``pyfai_hist`` branch of
    :func:`xrd_tools.analysis.plans.run_stitch` for the non-GI case."""
    from xrd_tools.integrate.stitch_hist import pyfai_q_frames, stitch_q_grid
    if unit != "q_A^-1":
        raise ValueError(
            "stitch_images: backend='pyfai_hist' emits q in Å⁻¹ only "
            f"(unit='q_A^-1'); got unit={unit!r}. Use backend='multigeometry' "
            "for other units.")
    if method != "BBox":
        logger.warning(
            "stitch_images: method=%r is ignored by backend='pyfai_hist' "
            "(the histogram merge does its own pixel splitting).", method)
    npt_rad = npt_rad_2d if mode == "2d" else npt_1d

    def _frames():
        # Re-built per call: stitch_q_grid may make a scout pass + an
        # accumulation pass, so the factory must be replayable.
        return pyfai_q_frames(
            images, integrators, mask=mask, normalization=normalization)

    return stitch_q_grid(
        _frames, mode=mode, npt=npt_rad, npt_azim=npt_azim_2d, unit=unit,
        radial_range=radial_range, azimuth_range=azimuth_range)


__all__ = [
    "GeometryIntegratorSeries",
    "PONIIntegratorSeries",
    "StitchDiagnostics",
    "create_multigeometry_integrator_series",
    "create_multigeometry_integrators",
    "create_multigeometry_integrator_series_from_geometry",
    "create_multigeometry_integrators_from_geometry",
    "stitch_1d",
    "stitch_1d_streaming",
    "stitch_2d",
    "stitch_2d_streaming",
    "stitch_2d_with_1d_streaming",
    "stitch_images",
]
