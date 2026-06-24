"""Geometry primitives for xrd_tools.

This package replaces the previous flat ``core/geometry.py`` module.
All historical public names continue to resolve through this package's
``__init__`` so ``from xrd_tools.core.geometry import X`` keeps
working unchanged:

- :class:`DiffractometerConfig` — xrayutilities-side config for building
  ``xu.HXRD``.
- :class:`DiffractometerGeometry` — per-frame motor → pyFAI rotation /
  GI incidence mapping for the v2 NeXus writer.
- :class:`AngleMapping`, :data:`Convention` — building blocks for the above.

Per-pixel q-space mapping for RSM lives alongside in
:mod:`xrd_tools.core.geometry.pixel_q` (added 2026-05).
"""
from __future__ import annotations

from xrd_tools.core.geometry.diffractometer import (
    AngleMapping,
    Convention,
    DetectorCalibration,
    Diffractometer,
    DiffractometerConfig,
    DiffractometerGeometry,
    ImageOrientation,
)
from xrd_tools.core.geometry.pixel_q import (
    DetectorHeader,
    PixelQMap,
)

__all__ = [
    # Per-frame, scalar (motor → pyFAI rotation / GI incidence)
    "AngleMapping",
    "Convention",
    "DiffractometerConfig",
    "DiffractometerGeometry",
    # Canonical one-object geometry (two derived adapter views)
    "Diffractometer",
    # Static detector calibration (PONI + Detector_config + image mount)
    "DetectorCalibration",
    "ImageOrientation",
    # Per-pixel q-space (RSM)
    "DetectorHeader",
    "PixelQMap",
]
