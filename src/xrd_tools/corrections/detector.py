"""
Detector-level correction helpers applied before azimuthal integration.
"""

from __future__ import annotations

import numpy as np


def _as_float_image(image: np.ndarray) -> np.ndarray:
    """Return image as float64 ndarray."""
    return np.asarray(image, dtype=float)


def _require_same_shape(a: np.ndarray, b: np.ndarray, *, name_b: str) -> None:
    """Validate array shape compatibility for per-pixel operations."""
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: image shape {a.shape} != {name_b} shape {b.shape}"
        )


def subtract_dark(
    image: np.ndarray,
    dark: np.ndarray,
) -> np.ndarray:
    """
    Subtract a dark-current frame from an image.

    Parameters
    ----------
    image : ndarray
        Raw detector image.
    dark : ndarray
        Dark-current image with the same shape as ``image``.

    Returns
    -------
    np.ndarray
        Corrected image in ``float64``.
    """
    image_arr = _as_float_image(image)
    dark_arr = _as_float_image(dark)
    _require_same_shape(image_arr, dark_arr, name_b="dark")

    corrected = image_arr - dark_arr
    corrected[np.isnan(image_arr)] = np.nan
    return corrected


def apply_flatfield(
    image: np.ndarray,
    flat: np.ndarray,
    min_flat: float = 0.1,
) -> np.ndarray:
    """
    Apply flat-field correction by dividing by a flat image.

    Pixels with ``flat < min_flat`` are set to ``NaN`` to avoid unstable
    division by near-zero values.

    Parameters
    ----------
    image : ndarray
        Input detector image.
    flat : ndarray
        Flat-field image with the same shape as ``image``.
    min_flat : float, optional
        Minimum valid flat-field value.

    Returns
    -------
    np.ndarray
        Flat-field-corrected image in ``float64``.
    """
    image_arr = _as_float_image(image)
    flat_arr = _as_float_image(flat)
    _require_same_shape(image_arr, flat_arr, name_b="flat")

    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = image_arr / flat_arr
    corrected[flat_arr < float(min_flat)] = np.nan
    corrected[np.isnan(image_arr)] = np.nan
    return corrected


#: Flat pixels below this fraction of the flat's median are treated as dead.
DEFAULT_MIN_FLAT_FRACTION = 0.1


def prepare_flatfield(
    flat: np.ndarray,
    min_fraction: float = DEFAULT_MIN_FLAT_FRACTION,
) -> np.ndarray:
    """
    Return a flat field whose unusable pixels are NaN, which pyFAI masks.

    pyFAI applies a flat in the normalization denominator; a pixel is unusable
    when its flat is not finite, not positive, or below ``min_fraction`` of the
    median of the positive finite flat, because dividing by a (nearly) dead
    pixel amplifies its noise. ``min_fraction=0`` keeps every positive pixel.
    The flat is not rescaled: pyFAI divides by it as given.

    Parameters
    ----------
    flat : ndarray
        2-D flat-field image.
    min_fraction : float, optional
        Fraction of the median below which a pixel is masked.

    Returns
    -------
    np.ndarray
        Float64 copy with unusable pixels set to NaN.
    """
    array = np.array(flat, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("flat field must be a 2-D detector image")
    fraction = float(min_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError("minimum flat fraction must be from 0 to below 1")
    valid = np.isfinite(array) & (array > 0)
    if not valid.any():
        raise ValueError("flat field has no positive finite pixel")
    if fraction:
        valid &= array >= fraction * float(np.median(array[valid]))
    array[~valid] = np.nan
    return array


def apply_threshold(
    image: np.ndarray,
    threshold: float,
    low: float | None = None,
) -> np.ndarray:
    """
    Apply high/low threshold masking to an image.

    Parameters
    ----------
    image : ndarray
        Input detector image.
    threshold : float
        Upper threshold; values above this are set to ``NaN``.
    low : float or None, optional
        Lower threshold; when provided, values below this are set to ``NaN``.

    Returns
    -------
    np.ndarray
        Thresholded copy in ``float64``.
    """
    out = _as_float_image(image).copy()
    out[out > float(threshold)] = np.nan
    if low is not None:
        out[out < float(low)] = np.nan
    return out


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Apply a boolean pixel mask to an image.

    Parameters
    ----------
    image : ndarray
        Input detector image.
    mask : ndarray
        Boolean mask with the same shape as ``image``. ``True`` pixels are set
        to ``NaN``.

    Returns
    -------
    np.ndarray
        Masked copy in ``float64``.
    """
    out = _as_float_image(image).copy()
    mask_arr = np.asarray(mask, dtype=bool)
    _require_same_shape(out, mask_arr, name_b="mask")
    out[mask_arr] = np.nan
    return out


def combine_masks(
    *masks: np.ndarray | None,
) -> np.ndarray | None:
    """
    Combine multiple masks via logical OR.

    Parameters
    ----------
    *masks : ndarray or None
        Any number of boolean-like masks. ``None`` values are ignored.

    Returns
    -------
    np.ndarray or None
        Combined boolean mask, or ``None`` when all inputs are ``None``.

    Raises
    ------
    ValueError
        If provided masks do not all share the same shape.
    """
    valid_masks = [np.asarray(m, dtype=bool) for m in masks if m is not None]
    if not valid_masks:
        return None

    base_shape = valid_masks[0].shape
    for m in valid_masks[1:]:
        if m.shape != base_shape:
            raise ValueError(f"Mask shapes do not match: {base_shape} vs {m.shape}")

    combined = np.zeros(base_shape, dtype=bool)
    for m in valid_masks:
        combined |= m
    return combined


def correct_image(
    image: np.ndarray,
    dark: np.ndarray | None = None,
    flat: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    threshold: float | None = None,
    low_threshold: float | None = None,
) -> np.ndarray:
    """
    Apply the detector-correction pipeline in a fixed order.

    Order: dark subtraction → flat-field correction → mask → threshold.

    Parameters
    ----------
    image : ndarray
        Raw detector image.
    dark : ndarray or None, optional
        Dark-current frame.
    flat : ndarray or None, optional
        Flat-field frame.
    mask : ndarray or None, optional
        Boolean bad-pixel mask.
    threshold : float or None, optional
        Upper intensity threshold; values above are masked to ``NaN``.
    low_threshold : float or None, optional
        Lower intensity threshold; values below are masked to ``NaN``.

    Returns
    -------
    np.ndarray
        Corrected image in ``float64``.
    """
    corrected = _as_float_image(image).copy()

    if dark is not None:
        corrected = subtract_dark(corrected, dark)
    if flat is not None:
        corrected = apply_flatfield(corrected, flat)
    if mask is not None:
        corrected = apply_mask(corrected, mask)
    if threshold is not None:
        corrected = apply_threshold(corrected, threshold=threshold, low=low_threshold)

    return corrected


__all__ = [
    "DEFAULT_MIN_FLAT_FRACTION",
    "apply_flatfield",
    "apply_mask",
    "apply_threshold",
    "combine_masks",
    "correct_image",
    "prepare_flatfield",
    "subtract_dark",
]
