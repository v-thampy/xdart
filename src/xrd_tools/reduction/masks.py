# -*- coding: utf-8 -*-
"""Pure detector-mask coercion helpers for reduction plans."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from xrd_tools.core.scan import MaskSpec

logger = logging.getLogger(__name__)


def _mask_for_plan(mask: Any, shape: tuple[int, int] | None) -> np.ndarray | MaskSpec | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if shape is None and arr.ndim == 1:
        return MaskSpec(arr.copy())
    return _flat_mask_as_bool(mask, shape)


def _flat_mask_as_bool(mask: Any, shape: tuple[int, int] | None) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, MaskSpec):
        if shape is None:
            return None
        try:
            return mask.to_bool(shape)
        except ValueError as exc:
            # A flat-index mask that doesn't fit this image (wrong
            # detector/calibration, stale mask) makes MaskSpec.to_bool raise;
            # match the ndarray branch below and ignore it with a warning
            # rather than letting the ValueError kill the run thread (BUG-2).
            logger.warning("Ignoring mask: %s", exc)
            return None
    arr = np.asarray(mask)
    if shape is None:
        if arr.ndim == 2:
            return arr.astype(bool, copy=False)
        return None
    # A mask that doesn't fit this image (wrong detector/calibration, a
    # resized frame, a stale flat-index mask, ...) is ignored with a warning
    # rather than crashing the whole scan - reducing unmasked is far better
    # than aborting the run.  Structural problems degrade the same way.
    if arr.ndim == 2:
        if arr.shape != shape:
            logger.warning(
                "Ignoring mask: shape %s does not match image shape %s.",
                arr.shape, shape,
            )
            return None
        return arr.astype(bool, copy=False)
    if arr.ndim != 1:
        logger.warning("Ignoring mask: expected 1D flat mask, got shape %s.", arr.shape)
        return None
    if arr.dtype == bool:
        if arr.size != int(np.prod(shape)):
            logger.warning(
                "Ignoring boolean mask: length %d does not match image shape %s.",
                arr.size, shape,
            )
            return None
        return arr.reshape(shape)
    out = np.zeros(int(np.prod(shape)), dtype=bool)
    flat = np.asarray(arr, dtype=int).ravel()
    if flat.size and (flat.min() < 0 or flat.max() >= out.size):
        logger.warning(
            "Ignoring mask: flat indices out of bounds for image shape %s "
            "(index range [%d, %d], image has %d pixels).",
            shape, int(flat.min()), int(flat.max()), out.size,
        )
        return None
    out[flat] = True
    return out.reshape(shape)

