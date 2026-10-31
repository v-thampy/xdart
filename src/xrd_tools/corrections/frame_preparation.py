"""One frame's preparation for a histogram or a volume: the shared steps of Stitch's xrayutilities path and
RSM's gridding (AUD-9 stage 2b, design §4).

- ``correction_image``: a dark or flat admitted by Int's owners (``prepare_dark``, ``prepare_flatfield``), its
  unusable pixels given a neutral value and returned as an invalid mask that joins the frame mask.
- ``prepare_frame_signal``: the dark subtracted from a conditioned frame and the frame divided by its beam
  divisor (``normalization.frame_divisors``), in place.
- ``frame_valid``: the pixels a frame contributes.
- ``frame_weights``: the per-pixel normalization weight (solid angle × polarization × flat), as in pyFAI.
- ``polarization_axes`` and ``polarization_weight``: pyFAI's polarization factor from each pixel's elastic
  scattered direction.

Thresholds are Int's own owner, ``integrate.stitch_conditioning.condition_stitch_frame``. Every array a
function here creates is returned to its caller (no cache, no closure), so a caller's release fence sees it.
"""
from __future__ import annotations

import math

import numpy as np

from xrd_tools.corrections.detector import DEFAULT_MIN_FLAT_FRACTION, prepare_dark, prepare_flatfield

#: Unit axes by name, for a circle's rotation axis (``"x+"``, ``"z-"``: the letter).
AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


class FramePreparationRefused(ValueError):
    """A frame that cannot be prepared; ``code`` is stable, and the caller reports its own."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def correction_image(value, role: str, *, min_fraction: float = DEFAULT_MIN_FLAT_FRACTION):
    """A dark or flat image as (finite values, the pixels it invalidates), admitted by Int's owners.

    A dark pixel ``prepare_dark`` finds non-finite, or a flat pixel ``prepare_flatfield`` finds unusable
    (non-finite, not positive, or below ``min_fraction`` of the median), cannot correct its pixel: it joins
    the frame mask and carries a neutral value (0 for a dark, 1 for a flat). An image with no usable pixel
    is refused (``ValueError``) as Int refuses it. None: (None, None).
    """
    if role not in {"dark", "flat"}:
        raise ValueError("a correction image is a dark or a flat")
    if value is None:
        return None, None
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Stitch {role} must be a 2-D detector image")
    prepared = prepare_flatfield(array, min_fraction) if role == "flat" else prepare_dark(array)
    invalid = ~np.isfinite(prepared)
    del prepared
    if not invalid.any():
        return array, None
    return np.where(invalid, 1.0 if role == "flat" else 0.0, array), invalid


def prepare_frame_signal(image: np.ndarray, *, dark: np.ndarray | None, divisor: float) -> np.ndarray:
    """Subtract the dark from a conditioned float64 frame and divide it by its beam divisor, in place, as
    pyFAI does (the dark before the normalization). Overflow is refused (``NUMERIC_OVERFLOW``)."""
    if dark is not None:
        np.subtract(image, dark, out=image)
    try:
        with np.errstate(divide="raise", invalid="ignore", over="raise"):
            np.divide(image, divisor, out=image)
    except FloatingPointError as error:
        raise FramePreparationRefused("NUMERIC_OVERFLOW", "the frame's normalization overflowed") from error
    return image


def frame_valid(detector_mask: np.ndarray, *, mask: np.ndarray | None = None, finite=(), positive=()) -> np.ndarray:
    """The pixels a frame contributes: outside the detector's mask and ``mask``, finite in every ``finite``
    array, and finite and positive in every ``positive`` one (a weight)."""
    valid = np.logical_not(detector_mask)
    if mask is not None:
        np.logical_and(valid, np.logical_not(mask), out=valid)
    for value in finite:
        np.logical_and(valid, np.isfinite(value), out=valid)
    for value in positive:
        np.logical_and(valid, np.isfinite(value), out=valid)
        np.logical_and(valid, value > 0, out=valid)
    return valid


def frame_weights(solid: np.ndarray, *, polarization: np.ndarray | None = None,
                  flat: np.ndarray | None = None) -> np.ndarray:
    """Each pixel's normalization weight, as in pyFAI: the solid angle, times the polarization factor and
    the flat when given. ``solid`` itself is returned when neither is (no copy); otherwise a new array."""
    if polarization is None and flat is None:
        return solid
    weights = np.array(solid, dtype=np.float64, copy=True)
    if polarization is not None:
        np.multiply(weights, polarization, out=weights)
    if flat is not None:
        np.multiply(weights, flat, out=weights)
    return weights


def polarization_axes(beam, outer_detector_axis: str) -> tuple[np.ndarray, np.ndarray]:
    """The unit beam and the lab vertical: the outer detector circle (nu) scatters in the horizontal plane,
    so its rotation axis is the vertical, whatever the axes' names."""
    beam = np.asarray(beam, dtype=np.float64)
    axis = str(outer_detector_axis)
    if beam.shape != (3,) or not np.isfinite(beam).all() or axis[:1] not in AXES:
        raise ValueError("the beam or the detector axis is unsupported")
    beam = beam / np.linalg.norm(beam)
    vertical = np.asarray(AXES[axis[0]])
    if abs(float(beam @ vertical)) > 1e-12:
        raise ValueError("the nu circle axis is not perpendicular to the beam")
    return beam, vertical


def polarization_weight(qx, qy, qz, *, factor: float, wavelength_A: float, beam, vertical) -> np.ndarray:
    """pyFAI's polarization factor per pixel, from the elastic scattered direction.

    ``P = ½(1 + cos²2θ − f·cos2χ·sin²2θ)`` with χ measured from the horizontal axis (perpendicular to the
    beam and the vertical), so ``f = 1`` is fully horizontal polarization, as in pyFAI. ``q`` is in the frame
    of ``beam`` and ``vertical``; a q that is not elastic for ``wavelength_A`` is refused (``ValueError``).
    """
    k = 2 * math.pi / wavelength_A
    horizontal = np.cross(beam, vertical)
    # k_f = q + k·beam; beam, vertical and horizontal are orthonormal.
    along = qx * beam[0] + qy * beam[1] + qz * beam[2] + k
    up = qx * vertical[0] + qy * vertical[1] + qz * vertical[2]
    side = qx * horizontal[0] + qy * horizontal[1] + qz * horizontal[2]
    along *= along
    up *= up
    side *= side
    norm = along + up + side
    finite = np.isfinite(norm)
    if np.any(np.abs(norm[finite] / (k * k) - 1.0) > 1e-9):
        raise ValueError("q is not elastic for the mapping wavelength")
    side -= up
    side *= -factor
    side += along
    side /= norm
    side += 1.0
    side *= 0.5
    return side



def flat_detector_solid_angle(shape, *, distance: float, cch, pixel_size) -> np.ndarray:
    """Each pixel's solid angle relative to the direct-beam pixel's, for an untilted flat detector seen from a
    sample on the rotation axis (RSM: no sample offset, a rigid arm, so one array serves every frame): pyFAI's
    own (``CorrectionStack().normalization``, as Stitch's xrayutilities path takes it), with xrayutilities'
    header as a PONI at the direct-beam pixel's centre, ``poni = (cch + 0.5) × pixel``. ``distance`` and
    ``pixel_size`` in one unit. A tilted detector (the calibration's mount, C7) needs the tilts here too."""
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from xrd_tools.corrections.stack import CorrectionStack
    from xrd_tools.integrate.pyfai_numexpr import PYFAI_NUMEXPR_LOCK
    shape = tuple(int(value) for value in shape)
    unit = 1e-3  # pyFAI's metres from millimetres; the ratio is unit-free
    pixel1, pixel2 = (float(value) * unit for value in pixel_size)
    detector = Detector(pixel1=pixel1, pixel2=pixel2, max_shape=shape)
    integrator = AzimuthalIntegrator(dist=float(distance) * unit, poni1=(float(cch[0]) + 0.5) * pixel1,
                                     poni2=(float(cch[1]) + 0.5) * pixel2, detector=detector)
    integrator.auto_gc = False  # no pyFAI gc.collect() on each geometry change (CI-VIEWERS-HANG-128)
    with PYFAI_NUMEXPR_LOCK:
        values = np.ascontiguousarray(CorrectionStack().normalization(integrator, shape), dtype=np.float64)
    if values.shape != shape or not np.isfinite(values).all() or np.any(values <= 0):
        raise FramePreparationRefused("SOLID_ANGLE_INVALID", "the detector's solid angle is not finite and positive")
    return values

__all__ = [
    "AXES",
    "FramePreparationRefused",
    "correction_image",
    "frame_valid",
    "flat_detector_solid_angle",
    "frame_weights",
    "polarization_axes",
    "polarization_weight",
    "prepare_frame_signal",
]
