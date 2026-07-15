"""Headless source reachability probes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from xrd_tools.core.scan import SourceKind


class ProbeState(str, Enum):
    """Typed outcome of an explicit, format-adapter content probe (R1).

    Enumeration (:mod:`xrd_tools.sources.discover`,
    :mod:`xrd_tools.sources.directory_index`) never produces a
    :class:`ProbeState` itself — it is name-only and opens nothing.  A state
    is produced only when a caller explicitly probes one candidate via its
    adapter's ``probe`` callable.
    """

    READY = "ready"
    IN_PROGRESS = "in_progress"
    PROCESSED_OUTPUT = "processed_output"
    IMAGELESS = "imageless"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Result of probing one candidate: its typed state, a disclosed reason,
    and (when known) the specific :class:`SourceKind` to open it as."""

    state: ProbeState
    reason: str = ""
    kind: SourceKind | None = None


def probe_first_frame(source):
    """Return ``(reachable, first_image)`` — load the source's first frame as a
    strict 2-D raw image (the ROI-stats requirement).

    Mirrors the reintegrate raw path: a processed NeXus whose linked raw tree is
    missing raises (so ``reachable`` is False), and the strict ``load_frame``
    never substitutes a downsampled thumbnail.  A metadata-only source (``None``)
    or an empty scan is unreachable.  The decoded image is returned so the caller
    can reuse it (the ROI picker shows exactly this frame) instead of decoding a
    possibly multi-MB Eiger frame twice."""
    if source is None:
        return False, None
    try:
        idxs = list(source.frame_indices)
    except Exception:
        return False, None
    if not idxs:
        return False, None
    try:
        img = np.asarray(source.load_frame(idxs[0]))
    except Exception:
        return False, None
    if img.ndim == 2 and img.size > 0:
        return True, img
    return False, None


def raw_is_reachable(source):
    """True iff ``source`` can load its first frame as a 2-D raw image — the
    strict-raw probe the ROI stats require (see :func:`probe_first_frame`)."""
    return probe_first_frame(source)[0]


__all__ = ["ProbeResult", "ProbeState", "probe_first_frame", "raw_is_reachable"]
