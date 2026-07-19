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


#: error classes a probe/open/enumeration may treat as TRANSIENT (H18-R5/R10)
#: — concurrency/availability failures worth a debounced retry.  Everything
#: else is a DEFINITIVE observation.
TRANSIENT_PROBE_ERRORS = (
    PermissionError, BlockingIOError, InterruptedError, TimeoutError,
)


def observe_first_frame(source):
    """Return ``(reachable, first_image, transient)`` — the typed frame-0 probe.

    H18-R5: distinguishes a DEFINITIVE unreachable observation from a
    TRANSIENT probe error.  Definitive is the DEFAULT — a missing referenced
    master, an unresolvable record source, a metadata-only source, an empty
    scan, and a non-2-D payload all read as definitively unreachable (never
    reinterpret an ordinary ``False`` as transient).  Only the known
    concurrency/availability error classes — a sharing/permission denial
    (the Windows sharing-violation case), a blocking/interrupted read, or a
    timeout — report ``transient=True`` so a caller with a cache can
    debounce and retry instead of stranding the capability as unavailable."""
    if source is None:
        return False, None, False
    try:
        idxs = list(source.frame_indices)
    except Exception:
        return False, None, False
    if not idxs:
        return False, None, False
    try:
        img = np.asarray(source.load_frame(idxs[0]))
    except TRANSIENT_PROBE_ERRORS:
        return False, None, True
    except Exception:
        return False, None, False
    if img.ndim == 2 and img.size > 0:
        return True, img, False
    return False, None, False


def probe_first_frame(source):
    """Return ``(reachable, first_image)`` — load the source's first frame as a
    strict 2-D raw image (the ROI-stats requirement).

    Mirrors the reintegrate raw path: a processed NeXus whose linked raw tree is
    missing raises (so ``reachable`` is False), and the strict ``load_frame``
    never substitutes a downsampled thumbnail.  A metadata-only source (``None``)
    or an empty scan is unreachable.  The decoded image is returned so the caller
    can reuse it (the ROI picker shows exactly this frame) instead of decoding a
    possibly multi-MB Eiger frame twice.  Public legacy behavior: every failure
    mode reads as unreachable; :func:`observe_first_frame` is the typed seam."""
    reachable, img, _transient = observe_first_frame(source)
    return reachable, img


def raw_is_reachable(source):
    """True iff ``source`` can load its first frame as a 2-D raw image — the
    strict-raw probe the ROI stats require (see :func:`probe_first_frame`)."""
    return probe_first_frame(source)[0]


__all__ = ["ProbeResult", "ProbeState", "probe_first_frame", "raw_is_reachable"]
