"""
Monitor normalization and intensity scaling helpers, and each frame's beam divisor
(``frame_divisors``: monitor, foil transmission, exposure), the one owner the Stitch
and RSM operations apply.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def normalize_monitor(
    image: np.ndarray,
    monitor: float,
    reference: float = 1.0,
) -> np.ndarray:
    """
    Normalize an image by a monitor count (e.g. ion-chamber *i1*).

    Scales the image by ``reference / monitor``, so that all images from a
    scan are comparable as if measured at the same beam flux.

    Parameters
    ----------
    image : ndarray
        Input detector image.
    monitor : float
        Measured monitor count for this exposure.
    reference : float, optional
        Target monitor value to normalize to.  All images in a series should
        use the same ``reference``.  Default is ``1.0``.

    Returns
    -------
    np.ndarray
        Normalized image in ``float64``.  If ``monitor <= 0``, a warning is
        emitted and the original image (cast to ``float64``) is returned
        unchanged.
    """
    out = np.asarray(image, dtype=float)
    if float(monitor) <= 0.0:
        logger.warning(
            "normalize_monitor: monitor value %g is <= 0; returning image unchanged.",
            monitor,
        )
        return out.copy()
    return out * (float(reference) / float(monitor))


def normalize_time(
    image: np.ndarray,
    exposure_time: float,
    reference_time: float = 1.0,
) -> np.ndarray:
    """
    Normalize an image by exposure time.

    Scales the image by ``reference_time / exposure_time`` so that images
    taken with different exposure times can be compared on a counts-per-second
    basis.

    Parameters
    ----------
    image : ndarray
        Input detector image.
    exposure_time : float
        Exposure time (in any consistent unit, e.g. seconds) for this image.
    reference_time : float, optional
        Target exposure time to normalize to.  Default is ``1.0``.

    Returns
    -------
    np.ndarray
        Time-normalized image in ``float64``.  If ``exposure_time <= 0``, a
        warning is emitted and the original image (cast to ``float64``) is
        returned unchanged.
    """
    out = np.asarray(image, dtype=float)
    if float(exposure_time) <= 0.0:
        logger.warning(
            "normalize_time: exposure_time %g is <= 0; returning image unchanged.",
            exposure_time,
        )
        return out.copy()
    return out * (float(reference_time) / float(exposure_time))


def normalize_stack(
    images: np.ndarray | list[np.ndarray],
    monitors: np.ndarray | Sequence[float],
    reference: float | None = None,
) -> list[np.ndarray]:
    """
    Normalize each image in a stack by its corresponding monitor count.

    This is the standard pre-processing step before multi-geometry stitching
    (:func:`~xrd_tools.integrate.multi.stitch_1d`): each image may have
    been measured with a different beam flux, so dividing by the per-image
    monitor count and multiplying by a common ``reference`` places all images
    on the same intensity scale.

    Parameters
    ----------
    images : ndarray (3-D) or list of ndarray
        Per-frame detector images.  A 3-D array ``(n_frames, ny, nx)`` is
        accepted as well as a list of 2-D arrays.
    monitors : array-like of float
        Per-frame monitor counts.  Must have the same length as ``images``.
    reference : float or None, optional
        Common reference monitor value.  If ``None``, the mean of all
        ``monitors`` values is used.

    Returns
    -------
    list of np.ndarray
        Normalized images in ``float64``, one per input frame.

    Raises
    ------
    ValueError
        If ``len(images) != len(monitors)``.
    """
    if isinstance(images, np.ndarray):
        if images.ndim == 3:
            img_list: list[np.ndarray] = [images[i] for i in range(images.shape[0])]
        elif images.ndim == 2:
            img_list = [images]
        else:
            raise ValueError(
                f"images ndarray must be 2D or 3D, got shape {images.shape}"
            )
    else:
        img_list = [np.asarray(im, dtype=float) for im in images]

    mon_arr = np.asarray(monitors, dtype=float)
    if mon_arr.shape != (len(img_list),):
        raise ValueError(
            f"monitors length {mon_arr.shape} != number of images {len(img_list)}"
        )

    ref = float(np.mean(mon_arr)) if reference is None else float(reference)

    return [normalize_monitor(img, mon, reference=ref) for img, mon in zip(img_list, mon_arr)]


def scale_to_range(
    intensity: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> np.ndarray:
    """
    Linearly scale an intensity array to a target ``[vmin, vmax]`` range.

    ``NaN`` values are ignored when determining the data extremes.  The
    output preserves ``NaN`` pixels.

    Parameters
    ----------
    intensity : ndarray
        Input intensity array (1-D or 2-D).
    vmin : float, optional
        Lower bound of the output range.  Default ``0.0``.
    vmax : float, optional
        Upper bound of the output range.  Default ``1.0``.

    Returns
    -------
    np.ndarray
        Scaled array in ``float64``.  Returns a copy of the input cast to
        ``float64`` and otherwise unchanged if the data range is zero.
    """
    out = np.asarray(intensity, dtype=float)
    data_min = float(np.nanmin(out))
    data_max = float(np.nanmax(out))
    data_range = data_max - data_min
    if data_range == 0.0:
        logger.warning("scale_to_range: data range is zero; returning image unchanged.")
        return out.copy()
    return (out - data_min) / data_range * (float(vmax) - float(vmin)) + float(vmin)


class BeamNormalizationRefused(ValueError):
    """A frame's monitor, foil status or exposure that cannot be admitted. ``code`` is
    the stable refusal code a calling operation reports as its own."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _foil_codes(foil_status) -> np.ndarray:
    foil = np.asarray(foil_status, dtype=np.float64)
    if not np.all(np.isfinite(foil)) or not np.all(foil == np.floor(foil)):
        raise BeamNormalizationRefused(
            "FOIL_STATUS_INVALID", "foil status must be finite integral values")
    if np.any((foil < 0) | (foil > 9999)):
        raise BeamNormalizationRefused(
            "FOIL_STATUS_INVALID", "foil status must be between 0 and 9999")
    return foil.astype(np.int64)


def _transmission(codes: np.ndarray, absorption_lengths) -> np.ndarray:
    # the code's four decimal digits, thousands first, each its foil's multiplicity
    digits = np.column_stack(
        (codes // 1000, (codes // 100) % 10, (codes // 10) % 10, codes % 10))
    return np.exp(-(digits @ np.asarray(absorption_lengths, dtype=np.float64)))


def _absorption_lengths(value) -> np.ndarray:
    lengths = np.asarray(value, dtype=np.float64)
    if lengths.shape != (4,) or not np.all(np.isfinite(lengths)) or np.any(lengths < 0):
        raise ValueError("foil absorption lengths must be four finite non-negative numbers")
    return lengths


def foil_transmission(foil_status, absorption_lengths) -> np.ndarray:
    """Each frame's transmission through its foils, as the RSM notebook's
    ``foil_to_factor``: the status code's four digits (thousands first) count the foils
    of each absorption length in the beam, and the transmission is
    ``exp(-sum(digit * length))``. Unlike the notebook, a non-integral code is refused
    rather than truncated, and so is a negative one or one above 9999
    (``BeamNormalizationRefused``)."""
    lengths = _absorption_lengths(absorption_lengths)
    transmission = _transmission(_foil_codes(foil_status), lengths)
    if not np.all(np.isfinite(transmission)) or np.any(transmission <= 0):
        raise BeamNormalizationRefused(
            "TRANSMISSION_INVALID", "foil transmission is non-finite or non-positive")
    return transmission


def frame_divisors(
    frame_count: int,
    *,
    monitor=None,
    foil=None,
    exposure=None,
    absorption_lengths=None,
) -> np.ndarray:
    """Each frame's finite positive beam divisor: the product of the factors given.

    - ``monitor``: a monitor's counts (one after the foils already includes their
      attenuation; one before them is given with ``foil``).
    - ``foil`` with ``absorption_lengths``: the foils' transmission
      (``foil_transmission``).
    - ``exposure``: the counting time in seconds. Never with a monitor, which
      already counts over it.

    No factor: ones. A frame's signal is divided by its divisor. The refusals
    (``BeamNormalizationRefused``) are checked in a fixed order: the arrays' shape,
    the foil status, the exposure, the monitor, then the transmission and the
    divisor. RSM's foil/exposure divisors are this function's, byte for byte.
    """
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("frame count must be a positive exact integer")
    if monitor is not None and exposure is not None:
        raise ValueError("an exposure is never combined with a monitor, which already counts over it")
    if (foil is None) is not (absorption_lengths is None):
        raise ValueError("foil status and absorption lengths are given together")
    lengths = None if absorption_lengths is None else _absorption_lengths(absorption_lengths)
    arrays = {name: np.asarray(value, dtype=np.float64)
              for name, value in (("monitor", monitor), ("foil", foil), ("exposure", exposure))
              if value is not None}
    if not arrays:
        result = np.ones(frame_count, dtype=np.float64)
        result.setflags(write=False)
        return result
    if any(value.shape != (frame_count,) for value in arrays.values()):
        raise BeamNormalizationRefused(
            "NORMALIZATION_METADATA_SHAPE_INVALID",
            "normalization metadata does not match frame membership")
    codes = None if foil is None else _foil_codes(arrays["foil"])
    if exposure is not None and (
            not np.all(np.isfinite(arrays["exposure"])) or np.any(arrays["exposure"] <= 0)):
        raise BeamNormalizationRefused(
            "EXPOSURE_INVALID", "exposure seconds must be finite and strictly positive")
    if monitor is not None and (
            not np.all(np.isfinite(arrays["monitor"])) or np.any(arrays["monitor"] <= 0)):
        raise BeamNormalizationRefused(
            "INVALID_MONITOR_VALUE", "monitor values must be finite and strictly positive")
    transmission = None if codes is None else _transmission(codes, lengths)
    factors = [value for value in (arrays.get("monitor"), transmission, arrays.get("exposure"))
               if value is not None]
    divisors = factors[0]
    for value in factors[1:]:
        divisors = divisors * value
    if transmission is not None and (
            not np.all(np.isfinite(transmission)) or np.any(transmission <= 0)):
        raise BeamNormalizationRefused(
            "TRANSMISSION_INVALID", "foil transmission is non-finite or non-positive")
    if not np.all(np.isfinite(divisors)) or np.any(divisors <= 0):
        names = "/".join(name for name in ("monitor", "foil", "exposure") if name in arrays)
        raise BeamNormalizationRefused(
            "NORMALIZATION_DIVISOR_INVALID", f"{names} divisor is non-finite or non-positive")
    result = np.array(divisors, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


__all__ = [
    "BeamNormalizationRefused",
    "foil_transmission",
    "frame_divisors",
    "normalize_monitor",
    "normalize_stack",
    "normalize_time",
    "scale_to_range",
]
