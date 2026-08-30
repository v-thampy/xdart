# xrd_tools/integrate/multi.py
"""
Multi-image stitching via pyFAI MultiGeometry.

The key pattern: when the detector is scanned to different angular positions
(in-plane ``del`` / ``rot1`` and out-of-plane ``nu`` / ``rot2``), every image
gets its own AzimuthalIntegrator with the detector angle encoded.
``create_multigeometry_integrators`` builds that list; ``stitch_1d`` /
``stitch_2d`` perform the stitched integration.
"""

from __future__ import annotations

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
from xrd_tools.integrate.detector_mask import mask_with_detector
from xrd_tools.integrate.calibration import poni_to_integrator

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


def _resolve_streaming_ranges(
    integrator_factory: Callable[[], Iterator[AzimuthalIntegrator]],
    image_count: int,
    *,
    unit: str,
    radial_range: tuple[float, float] | None,
    azimuth_range: tuple[float, float] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Match MultiGeometry's global range guesses with O(one-AI) residency."""

    if radial_range is not None and azimuth_range is not None:
        return radial_range, azimuth_range
    from pyFAI import units

    radial_unit = units.to_unit(unit)
    radial_min = math.inf
    radial_max = -math.inf
    azimuth_min = math.inf
    azimuth_max = -math.inf
    observed = 0
    for integrator in integrator_factory():
        observed += 1
        try:
            # MultiGeometry defaults to chi_disc=180 and mutates every member
            # before integrating.  Scout in that same convention; otherwise a
            # caller-supplied AI using chiDiscAtZero can produce a 0..360 range
            # which is then paired with -180..180 integration values, dropping
            # half the detector.
            integrator.setChiDiscAtPi()
            if radial_range is None:
                values = np.asarray(
                    integrator.array_from_unit(unit=radial_unit), dtype=float
                )
                radial_min = min(radial_min, float(values.min()))
                radial_max = max(radial_max, float(values.max()))
            if azimuth_range is None:
                values = np.asarray(
                    integrator.array_from_unit(unit=units.CHI_DEG), dtype=float
                )
                azimuth_min = min(azimuth_min, float(values.min()))
                azimuth_max = max(azimuth_max, float(values.max()))
        finally:
            integrator.reset(collect_garbage=False)
    if observed != image_count:
        raise ValueError(
            f"streaming Stitch received {observed} integrators; "
            f"expected {image_count}"
        )
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
):
    from pyFAI.multi_geometry import MultiGeometry

    if type(image_count) is not int or image_count < 1:
        raise ValueError("streaming Stitch image count must be positive")
    if mode not in {"1d", "2d"}:
        raise ValueError(f"mode must be '1d' or '2d', got {mode!r}")
    factors = _normalization_factors(normalization, image_count)
    fixed_radial, fixed_azimuth = _resolve_streaming_ranges(
        integrator_factory,
        image_count,
        unit=unit,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
    )
    image_iterator = iter(images)
    integrator_iterator = iter(integrator_factory())
    signal = normalization_sum = count = variance = None
    radial = azimuthal = None
    radial_unit = unit
    azimuthal_unit = "chi_deg"
    for index in range(image_count):
        try:
            image = np.asarray(next(image_iterator), dtype=float)
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
        if image.ndim != 2:
            integrator.reset(collect_garbage=False)
            raise ValueError(
                f"streaming Stitch image must be 2-D, got shape {image.shape}"
            )
        try:
            frame_mask = (
                None
                if mask is None
                else np.asarray(mask_with_detector(integrator, mask), dtype=bool)
            )
        except BaseException:
            integrator.reset(collect_garbage=False)
            raise
        monitor = 1.0 if factors is None else float(factors[index])
        try:
            multigeometry = MultiGeometry(
                [integrator],
                unit=unit,
                radial_range=fixed_radial,
                azimuth_range=fixed_azimuth,
                threadpoolsize=0,
            )
        except BaseException:
            integrator.reset(collect_garbage=False)
            raise
        try:
            if mode == "1d":
                result = multigeometry.integrate1d(
                    [image],
                    npt_rad,
                    correctSolidAngle=correct_solid_angle,
                    error_model=error_model,
                    polarization_factor=polarization_factor,
                    normalization_factor=[monitor],
                    lst_mask=None if frame_mask is None else [frame_mask],
                    method=method,
                )
                current_shape = (npt_rad,)
            else:
                result = multigeometry.integrate2d(
                    [image],
                    npt_rad,
                    npt_azim,
                    correctSolidAngle=correct_solid_angle,
                    error_model=error_model,
                    polarization_factor=polarization_factor,
                    normalization_factor=[monitor],
                    lst_mask=None if frame_mask is None else [frame_mask],
                    method=method,
                )
                current_shape = (npt_azim, npt_rad)
            current_signal = np.asarray(result.sum_signal, dtype=np.float64)
            current_normalization = np.asarray(
                result.sum_normalization, dtype=np.float64
            )
            current_count = np.asarray(result.count, dtype=np.float64)
            if any(
                values.shape != current_shape
                for values in (
                    current_signal,
                    current_normalization,
                    current_count,
                )
            ):
                raise ValueError("pyFAI returned an unexpected Stitch accumulator shape")
            if signal is None:
                signal = np.zeros(current_shape, dtype=np.float64)
                normalization_sum = np.zeros_like(signal)
                count = np.zeros_like(signal)
            signal += current_signal
            normalization_sum += current_normalization
            count += current_count
            if result.sigma is not None:
                current_variance = np.asarray(result.sum_variance, dtype=np.float64)
                if current_variance.shape != current_shape:
                    raise ValueError("pyFAI returned an unexpected Stitch variance shape")
                if variance is None:
                    variance = current_variance.copy()
                else:
                    variance += current_variance
            elif variance is not None:
                raise ValueError("pyFAI returned inconsistent Stitch variance")
            current_radial = np.asarray(result.radial, dtype=float)
            if radial is None:
                radial = current_radial.copy()
            elif not np.array_equal(radial, current_radial):
                raise ValueError("pyFAI returned inconsistent Stitch radial axes")
            if mode == "2d":
                current_azimuthal = np.asarray(result.azimuthal, dtype=float)
                if azimuthal is None:
                    azimuthal = current_azimuthal.copy()
                elif not np.array_equal(azimuthal, current_azimuthal):
                    raise ValueError("pyFAI returned inconsistent Stitch azimuth axes")
                radial_unit = str(result.radial_unit)
                azimuthal_unit = str(result.azimuthal_unit)
            else:
                radial_unit = str(result.unit) if result.unit is not None else unit
        finally:
            integrator.reset(collect_garbage=False)
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
    if any(value is None for value in (signal, normalization_sum, count, radial)):
        raise ValueError("streaming Stitch produced no result")
    norm = np.maximum(normalization_sum, np.finfo("float32").tiny)
    invalid = count <= 0
    intensity = signal / norm
    intensity[invalid] = 0.0
    sigma = None
    if variance is not None:
        sigma = np.sqrt(variance) / norm
        sigma[invalid] = 0.0
    return (
        radial,
        azimuthal,
        intensity,
        sigma,
        count,
        normalization_sum,
        radial_unit,
        azimuthal_unit,
    )


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
) -> tuple[IntegrationResult1D, StitchDiagnostics]:
    """Sequential MultiGeometry merge that retains only one detector frame."""

    (
        radial,
        _azimuthal,
        intensity,
        sigma,
        coverage,
        normalization_sum,
        radial_unit,
        _azimuthal_unit,
    ) = _streaming_multigeometry(
        images,
        image_count,
        integrator_factory,
        mode="1d",
        npt_rad=npt,
        npt_azim=1,
        unit=unit,
        method=method,
        radial_range=radial_range,
        azimuth_range=None,
        mask=mask,
        normalization=normalization,
        correct_solid_angle=correct_solid_angle,
        error_model=error_model,
        polarization_factor=polarization_factor,
    )
    payload = IntegrationResult1D(
        radial=radial,
        intensity=intensity,
        sigma=sigma,
        unit=radial_unit,
    )
    diagnostics = StitchDiagnostics(coverage, normalization_sum)
    return payload, diagnostics


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
) -> tuple[IntegrationResult2D, StitchDiagnostics]:
    """Sequential 2-D MultiGeometry merge retaining one detector frame."""

    (
        radial,
        azimuthal,
        intensity,
        sigma,
        coverage,
        normalization_sum,
        radial_unit,
        azimuthal_unit,
    ) = _streaming_multigeometry(
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
    )
    payload = IntegrationResult2D(
        radial=radial,
        azimuthal=azimuthal,
        intensity=intensity.T,
        sigma=None if sigma is None else sigma.T,
        unit=radial_unit,
        azimuthal_unit=azimuthal_unit,
    )
    diagnostics = StitchDiagnostics(
        coverage.T,
        normalization_sum.T,
    )
    return payload, diagnostics


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

    def __len__(self) -> int:
        return len(self.rotations)

    def __iter__(self) -> Iterator[AzimuthalIntegrator]:
        from xrd_tools.integrate.calibration import (  # noqa: PLC0415
            detector_calibration_to_integrator,
        )

        for rot1, rot2, rot3 in self.rotations:
            yield detector_calibration_to_integrator(
                self.calibration,
                rot1=rot1,
                rot2=rot2,
                rot3=rot3,
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

    sigma = result.sigma if result.sigma is not None else None
    unit_str = str(result.unit) if result.unit is not None else unit
    return IntegrationResult1D(
        radial=np.asarray(result.radial, dtype=float),
        intensity=np.asarray(result.intensity, dtype=float),
        sigma=np.asarray(sigma, dtype=float) if sigma is not None else None,
        unit=unit_str,
    )


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
    "stitch_images",
]
