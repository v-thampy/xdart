"""
Corrections pipeline: detector, beam, and normalization helpers.
"""

from ssrl_xrd_tools.corrections.beam import (
    absorption_correction,
    polarization_correction,
    solid_angle_correction,
)
from ssrl_xrd_tools.corrections.detector import (
    apply_flatfield,
    apply_mask,
    apply_threshold,
    combine_masks,
    correct_image,
    subtract_dark,
)
from ssrl_xrd_tools.corrections.normalization import (
    normalize_monitor,
    normalize_stack,
    normalize_time,
    scale_to_range,
)

__all__ = [
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
