"""Stitch adapters for the existing Int detector-value and mask policies."""

from dataclasses import asdict, dataclass
import math

import numpy as np

from xrd_tools.core.geometry import ImageOrientation
from xrd_tools.core.invalid import detector_value_mask


@dataclass(frozen=True, slots=True)
class StitchConditioning:
    """Explicit Int policy; absence at a caller preserves its legacy reader."""

    mode: str = "off"
    threshold_min: float | None = None
    threshold_max: float | None = None

    def __post_init__(self):
        if self.mode not in {"off", "auto", "manual"}:
            raise ValueError("threshold mode must be off, auto or manual")
        for name in ("threshold_min", "threshold_max"):
            value = getattr(self, name)
            if value is not None and (type(value) not in {int, float} or not math.isfinite(value)):
                raise ValueError("threshold bounds must be finite or absent")
        if self.mode != "manual" and (self.threshold_min is not None or self.threshold_max is not None):
            raise ValueError("only manual conditioning carries threshold bounds")
        if self.mode == "manual":
            if self.threshold_min is None and self.threshold_max is None:
                raise ValueError("manual conditioning requires a threshold bound")
            if (self.threshold_min is not None and self.threshold_max is not None
                    and self.threshold_min > self.threshold_max):
                raise ValueError("threshold minimum cannot exceed maximum")

    def to_dict(self):
        return asdict(self)


def condition_stitch_frame(raw_image, conditioning: StitchConditioning | None):
    """Return an owning float64 frame, preserving native values for Int policy.

    No orientation or static masking occurs here. Those remain geometry inputs;
    disabling value conditioning never disables a detector or external mask.
    """
    if conditioning is not None and type(conditioning) is not StitchConditioning:
        raise TypeError("conditioning must be exact StitchConditioning or absent")
    raw = np.asarray(raw_image)
    if raw.ndim != 2:
        raise ValueError("Stitch requires a two-dimensional detector frame")
    image = np.array(raw, dtype=np.float64, order="C", copy=True)
    if conditioning is None or conditioning.mode == "off":
        return image
    if conditioning.mode == "auto":
        bad = detector_value_mask(None, raw, enabled=True)
    else:
        # Reuse the same inclusive comparison owner as native Int, including
        # comparisons against native integer values before float conversion.
        from xrd_tools.reduction.core import _threshold_mask

        bad = _threshold_mask(raw, conditioning)
    if bad is not None:
        image[bad] = np.nan
    return image


def load_stitch_frame(source, label, conditioning: StitchConditioning | None):
    """Read once; explicit policies ask capable sources for native detector data."""
    native = getattr(source, "load_native_frame", None)
    loader = native if conditioning is not None and callable(native) else source.load_frame
    return np.asarray(loader(label))


def prepare_stitch_mask(mask, image_orientation: ImageOrientation, expected_shape):
    """Coerce a raw mask and orient it once into the calibrated pixel frame.

    Runtime kernels accept this result directly and never reorient it. The
    input orientation is the same transformation used for detector images.
    """
    from xrd_tools.io.image import load_mask

    if type(image_orientation) is not ImageOrientation:
        raise TypeError("mask orientation must be exact ImageOrientation")
    value = image_orientation.apply(load_mask(mask))
    if value.ndim != 2 or value.shape != tuple(expected_shape):
        raise ValueError("oriented mask shape does not match calibrated detector shape")
    value = np.array(value, dtype=bool, order="C", copy=True)
    value.setflags(write=False)
    return value
