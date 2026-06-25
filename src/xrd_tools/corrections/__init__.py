"""
Corrections pipeline: detector, beam, and normalization helpers.
"""

from xrd_tools.corrections.beam import (
    absorption_correction,
    polarization_correction,
    solid_angle_correction,
)
from xrd_tools.corrections.detector import (
    apply_flatfield,
    apply_mask,
    apply_threshold,
    combine_masks,
    correct_image,
    subtract_dark,
)
from xrd_tools.corrections.normalization import (
    normalize_monitor,
    normalize_stack,
    normalize_time,
    scale_to_range,
)
from xrd_tools.corrections.stack import CorrectionStack
from xrd_tools.corrections.grazing import (
    GICorrectionStack,
    fresnel_transmission_sq,
    refracted_angle,
)

__all__ = [
    # the shared per-pixel pre-weight (stitch backends + RSM)
    "CorrectionStack",
    # grazing-incidence per-pixel corrections (GI mode)
    "GICorrectionStack",
    "fresnel_transmission_sq",
    "refracted_angle",
    # beam
    "absorption_correction",
    "polarization_correction",
    "solid_angle_correction",
    # detector
    "apply_flatfield",
    "apply_mask",
    "apply_threshold",
    "combine_masks",
    "correct_image",
    "subtract_dark",
    # normalization
    "normalize_monitor",
    "normalize_stack",
    "normalize_time",
    "scale_to_range",
]
