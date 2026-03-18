"""
Monitor normalization and intensity scaling helpers.
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
    (:func:`~ssrl_xrd_tools.integrate.multi.stitch_1d`): each image may have
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


__all__ = [
    "normalize_monitor",
    "normalize_stack",
    "normalize_time",
    "scale_to_range",
]
