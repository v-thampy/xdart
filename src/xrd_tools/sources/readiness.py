"""Headless source/read-result readiness composition.

``xrd_tools.core.scan.SourceCapabilities`` is what a frame source advertises
about itself.  ``xrd_tools.session.readiness.SourceCaps`` / ``ResultCaps`` are
the readiness-layer projections consumed by run gates and analysis launchers.
This module is the pure bridge between those two shapes.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xrd_tools.core.scan import (
    SourceCapabilities as CoreSourceCapabilities,
    SourceKind,
    SourceSpec,
)
from xrd_tools.session.readiness import ResultCaps, SourceCaps

__all__ = [
    "describe_source_readiness",
    "capabilities_for_processed",
    "nxwriter_finalization_policy",
    "observe_raw_reachability",
    "RawReachabilityObservation",
    "quiet_capability_observation",
]

_ENERGY_KEYS = frozenset({
    "energy",
    "energy_ev",
    "energy_eV",
    "energy_kev",
    "energy_keV",
    "wavelength",
    "wavelength_a",
    "wavelength_A",
})
_PSI_KEYS = frozenset({"psi", "sin2psi", "sin^2psi", "chi", "eta"})
_PHASE_CAPS = frozenset({"phase_result", "phase_fit", "phase_fractions"})

logger = logging.getLogger(__name__)

#: readers that emit generic expected-absence warnings while a capability
#: observation opens a container (H18-R7)
_READER_LOGGER_NAMES = ("xrd_tools.io.nexus", "xrd_tools.sources.nexus")
_EXPECTED_ABSENCE_MARKERS = ("Energy not found", "Wavelength not derivable")


_OBSERVATION_TLS = threading.local()


class _ThreadScopedAbsenceFilter(logging.Filter):
    """H18-R11: suppression is scoped to the OBSERVING THREAD's active
    observation stack — never process-global.  The single filter instance is
    installed once per reader logger and left in place; with no observation
    active on the CURRENT thread it is a pure pass-through, so a real warning
    emitted by an unrelated thread while another thread holds an observation
    open passes through unchanged (and is never attributed to the observing
    path).  Nested observations suppress into the innermost context."""

    def filter(self, record):  # noqa: A003 - logging API name
        stack = getattr(_OBSERVATION_TLS, "stack", None)
        if not stack:
            return True
        message = record.getMessage()
        if any(marker in message for marker in _EXPECTED_ABSENCE_MARKERS):
            stack[-1].append(message)
            return False
        return True


_SCOPED_FILTER = _ThreadScopedAbsenceFilter()
_FILTERS_INSTALLED = False
_FILTER_INSTALL_LOCK = threading.Lock()


def _ensure_scoped_filters_installed() -> None:
    global _FILTERS_INSTALLED
    if _FILTERS_INSTALLED:
        return
    with _FILTER_INSTALL_LOCK:
        if _FILTERS_INSTALLED:
            return
        for name in _READER_LOGGER_NAMES:
            reader_logger = logging.getLogger(name)
            if _SCOPED_FILTER not in reader_logger.filters:
                reader_logger.addFilter(_SCOPED_FILTER)
        _FILTERS_INSTALLED = True


@contextmanager
def quiet_capability_observation(subject_path, subject_kind):
    """H18-R7/R11: a capability observation is not an operator problem report.

    Missing energy/wavelength is EXPECTED for many raw detector masters (the
    run's energy authority is the selected PONI), so the generic readers'
    per-open WARNINGs are suppressed FOR THIS THREAD'S observation only and
    replaced by ONE structured DEBUG line that names the exact path and
    whether the subject is the configured acquisition source or the selected
    processed result.  Warnings that are not expected-absence, and warnings
    from unrelated threads, pass through unchanged; nesting is safe.
    """
    _ensure_scoped_filters_installed()
    suppressed: list[str] = []
    stack = getattr(_OBSERVATION_TLS, "stack", None)
    if stack is None:
        stack = _OBSERVATION_TLS.stack = []
    stack.append(suppressed)
    try:
        yield
    finally:
        stack.pop()
        if suppressed:
            logger.debug(
                "readiness observation for %s (%s): %s — expected absence; "
                "the run's energy authority is the selected PONI",
                subject_path, subject_kind,
                "; ".join(sorted(set(suppressed))))


def _observation_subject(value) -> tuple[str, str]:
    uri = _uri(value)
    kind = getattr(value, "kind", None)
    subject = ("selected processed result"
               if kind == SourceKind.PROCESSED_NEXUS
               else "configured acquisition source")
    return str(uri if uri is not None else value), subject



def describe_source_readiness(spec_or_source: Any, *, probe: bool = True) -> SourceCaps:
    """Project a source, URI, or :class:`SourceSpec` into readiness caps.

    True-live / unknown-length sources keep ``raw_reachable=True`` even when a
    frame-0 probe cannot yet load an image; a live acquisition may legitimately
    have no frame zero at gate time.

    Spec-level gates (H18, from the H5 parity findings — both adopt the
    controls-panel behavior so the core is correct for EVERY consumer):

    * An EMPTY location is never ready, even for ``SourceKind.LIVE`` — the
      escape hatch governs a *configured* live source; with no location there
      is nothing to run (the "label required even live" rule).
    * A non-live LOCAL path that does not exist has no frames.  ``open_source``
      happily builds e.g. an ``ImageFileSource`` around a typo'd path without
      stat-ing it (phantom ``frame_indices == [0]``), which used to report
      ``has_frames=has_raw=True`` for a nonexistent file (H5 finding 1) — a
      naive run gate over that answer would enable Run on nothing.  Remote
      ``scheme://`` URIs are not stat-able and pass through unchanged.
    """

    gated = _spec_gate(spec_or_source)
    if gated is not None:
        return gated

    subject_path, subject_kind = _observation_subject(spec_or_source)
    with quiet_capability_observation(subject_path, subject_kind):
        return _describe_open_source(spec_or_source, probe=probe)


def _describe_open_source(spec_or_source: Any, *, probe: bool = True) -> SourceCaps:
    source = _open_source(spec_or_source)
    if source is None:
        return _caps_from_classification(spec_or_source)

    source_caps = _core_caps(source)
    frame_indices, frame_count = _frame_indices(source)
    unknown_length = frame_count is None
    live_unknown = bool(
        source_caps.is_streaming
        or getattr(source, "kind", None) == SourceKind.LIVE
        or unknown_length
    )
    has_frames = bool(live_unknown or (frame_count is not None and frame_count > 0))

    if live_unknown:
        # True-live escape hatch: a live acquisition may legitimately have no
        # frame 0 yet, so classification and the frame-0 probe are SKIPPED
        # (same answers as before, without touching a file another process may
        # be writing — the readiness call must stay cheap on live refreshes).
        has_raw = True
        raw_reachable = True
    else:
        info = _classify(spec_or_source)
        has_raw = bool(
            getattr(source_caps, "has_raw_references", False)
            or getattr(info, "has_raw", False)
            or (has_frames and hasattr(source, "load_frame"))
        )
        if probe and has_raw:
            raw_reachable = bool(_probe_first_frame(source))
        else:
            raw_reachable = bool(has_raw)

    first_metadata = _first_metadata(source, frame_indices)
    motors = _motors(source)
    return SourceCaps(
        has_frames=has_frames,
        has_raw=has_raw,
        raw_reachable=raw_reachable,
        has_metadata=bool(
            source_caps.has_metadata
            or source_caps.has_scan_manifest
            or first_metadata
            or motors
        ),
        has_motors=bool(motors),
        has_energy=bool(_has_energy(first_metadata) or _attr_known(source, ("energy", "wavelength"))),
        has_geometry=bool(
            source_caps.has_geometry
            or _attr_known(source, ("geometry", "poni", "integrator"))
        ),
        has_psi_metadata=_has_any_key(first_metadata, _PSI_KEYS),
    )


def capabilities_for_processed(
    metadata: Mapping[str, Any],
    *,
    raw_reachable: bool | None = None,
) -> ResultCaps:
    """Project already-materialized processed metadata into result caps.

    This consumes ``metadata["capabilities"]`` as written by
    :func:`xrd_tools.io.read.get_metadata`; it intentionally does not reopen an
    HDF5 file or call ``io.schema.detect_capabilities``.

    ``raw_reachable`` lets the caller inject frame-0 PROBE truth (typically
    ``describe_source_readiness(path).raw_reachable``): the ``frames_record``
    capability only proves the record EXISTS, not that its raw master is still
    reachable, so the default mirror overstates reachability for an orphaned
    record (H5 finding 2).  Raw-dependent launcher gates should pass the probe
    answer; ``None`` (default) keeps the pure, no-reopen record mirror.
    """

    caps = {str(cap) for cap in metadata.get("capabilities", ()) or ()}
    has_1d = bool(
        metadata.get("has_1d")
        or caps & {"axis_kind_1d", "sigma_1d", "multi_result_1d"}
    )
    has_2d = bool(
        metadata.get("has_2d")
        or caps & {"two_d_kind", "sigma_2d", "multi_result_2d"}
    )
    has_raw = bool(caps & {"frames_record", "source_base"})
    has_scan_metadata = _metadata_has_scan_table(metadata) or bool(
        metadata.get("positioners")
        or metadata.get("n_frames")
        or _array_len(metadata.get("frames"))
    )
    reachable = (
        has_raw if raw_reachable is None else bool(has_raw and raw_reachable)
    )
    return ResultCaps(
        has_1d=has_1d,
        has_2d=has_2d,
        has_raw=has_raw,
        raw_reachable=reachable,
        has_scan_metadata=has_scan_metadata,
        has_rsm="rsm" in caps,
        has_phase_result=bool(caps & _PHASE_CAPS),
        has_psi_metadata=has_scan_metadata,
    )


_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+://")


def _spec_gate(value: Any) -> SourceCaps | None:
    """Spec-level readiness gate (H18) — see :func:`describe_source_readiness`.

    Applies only to ``str`` / ``Path`` / :class:`SourceSpec` inputs; an
    already-open source object is trusted as-is (its existence is its
    configuration — e.g. an in-memory live source has no URI to check).
    Returns the gated (all-False) caps, or ``None`` to proceed normally.
    """

    if not isinstance(value, (str, Path, SourceSpec)):
        return None
    uri = value.uri if isinstance(value, SourceSpec) else value
    text = str(uri or "").strip()
    if not text:
        # Nothing configured — never ready, live or not (label rule).
        return SourceCaps()
    kind = value.kind if isinstance(value, SourceSpec) else None
    if kind == SourceKind.LIVE:
        # A configured live location may not exist yet; the escape hatch owns it.
        return None
    if _URI_SCHEME_RE.match(text):
        # Remote resource (e.g. a future Tiled URI) — not stat-able here.
        return None
    if not Path(text).expanduser().exists():
        # Phantom-frame hazard: a typo'd local path has no frames (finding 1).
        return SourceCaps()
    return None


def _open_source(value: Any) -> Any | None:
    if hasattr(value, "frame_indices") and hasattr(value, "load_frame"):
        return value
    try:
        from xrd_tools.sources.registry import open_source

        return open_source(value)
    except Exception:
        return None


def _caps_from_classification(value: Any) -> SourceCaps:
    # True-live escape hatch on the FALLBACK path too: if open_source() failed but
    # the spec is a LIVE kind, a live acquisition may simply have no frame 0 yet —
    # keep it raw-reachable rather than collapsing to all-False (which would wrongly
    # gate the Run).  Mirrors the live_unknown branch on the open-succeeds path.
    if getattr(value, "kind", None) == SourceKind.LIVE:
        return SourceCaps(
            has_frames=True, has_raw=True, raw_reachable=True, has_metadata=False,
        )
    info = _classify(value)
    frame_count = getattr(info, "n_frames", 0) if info is not None else _count_frames(value)
    has_frames = bool(frame_count and frame_count > 0)
    has_raw = bool(getattr(info, "has_raw", False))
    return SourceCaps(
        has_frames=has_frames,
        has_raw=has_raw,
        raw_reachable=has_raw and has_frames,
        has_metadata=False,
    )


def _core_caps(source: Any) -> CoreSourceCapabilities:
    caps = getattr(source, "capabilities", None)
    if isinstance(caps, CoreSourceCapabilities):
        return caps
    return CoreSourceCapabilities()


def _frame_indices(source: Any) -> tuple[list[int], int | None]:
    try:
        indices = [int(idx) for idx in source.frame_indices]
    except Exception:
        return [], None
    return indices, len(indices)


def _probe_first_frame(source: Any) -> bool:
    try:
        from xrd_tools.sources.probe import probe_first_frame

        reachable, _image = probe_first_frame(source)
        return bool(reachable)
    except Exception:
        return False


def _classify(value: Any) -> Any | None:
    uri = _uri(value)
    if uri is None:
        return None
    try:
        from xrd_tools.io.image_source import classify_image_source

        return classify_image_source(uri)
    except Exception:
        return None


def _count_frames(value: Any) -> int:
    uri = _uri(value)
    if uri is None:
        return 0
    try:
        from xrd_tools.io.image import count_frames

        return int(count_frames(uri))
    except Exception:
        return 0


def _uri(value: Any) -> str | Path | None:
    if isinstance(value, SourceSpec):
        return value.uri
    if isinstance(value, (str, Path)):
        return value
    spec = getattr(value, "spec", None)
    if isinstance(spec, SourceSpec):
        return spec.uri
    return None


def _first_metadata(source: Any, frame_indices: list[int]) -> Mapping[str, Any]:
    if not frame_indices or not hasattr(source, "metadata_for"):
        return {}
    try:
        metadata = source.metadata_for(frame_indices[0])
    except Exception:
        return {}
    return dict(metadata or {})


def _motors(source: Any) -> Mapping[str, Any]:
    try:
        motors = getattr(source, "motors", None)
    except Exception:
        return {}
    return dict(motors or {})


def _attr_known(source: Any, names: tuple[str, ...]) -> bool:
    for name in names:
        try:
            value = getattr(source, name, None)
        except Exception:
            continue
        if value is not None:
            return True
    return False


def _has_energy(metadata: Mapping[str, Any]) -> bool:
    for key, value in metadata.items():
        if str(key) not in _ENERGY_KEYS:
            continue
        if _finite_value(value) or value is not None:
            return True
    return False


def _has_any_key(metadata: Mapping[str, Any], keys: frozenset[str]) -> bool:
    normalized = {str(key).lower() for key in metadata}
    return bool(normalized & {key.lower() for key in keys})


def _finite_value(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metadata_has_scan_table(metadata: Mapping[str, Any]) -> bool:
    scan_data = metadata.get("scan_data")
    if scan_data is None:
        return False
    empty = getattr(scan_data, "empty", None)
    if empty is not None:
        return not bool(empty)
    return True


def _array_len(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(len(value))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# R1 — finalized/in-progress readiness rule for the nexus-family format
# adapter's ``finalization_policy`` hook (see xrd_tools.sources.adapters).
# Distinct from describe_source_readiness()/SourceCaps above: this answers
# "is the CONTAINER done being written," not "what capabilities does an
# already-open source advertise."
# ---------------------------------------------------------------------------

def nxwriter_finalization_policy(path: str | Path) -> bool:
    """True iff *path* (a NeXus/HDF5 container) is finalized and safe to
    consume-and-retire; False means it should still be treated as
    in-progress/provisional.

    Delegates to :func:`xrd_tools.io.bluesky_nexus.is_unfinalized_nxwriter`:
    a non-Bluesky container has no ``end_time`` contract and is always
    finalized; a Bluesky/NXWriter run is finalized once ``end_time`` is
    stamped; an unreadable file (a half-written HDF5) is NOT finalized.  This
    is the same "defer, don't retire" rule the live directory watch already
    relies on — R1 exposes it as a format adapter's explicit policy hook
    rather than a one-off GUI check.
    """
    from xrd_tools.io.bluesky_nexus import is_unfinalized_nxwriter
    return not is_unfinalized_nxwriter(Path(path))


@dataclass(frozen=True)
class RawReachabilityObservation:
    """Typed frame-0 reachability observation (H18-R5).

    ``definitive=False`` marks a TRANSIENT probe error (sharing denial,
    momentary SMB/HDF5 hiccup): the observation is retry-worthy and must not
    be cached as a stable unreachable.  A missing referenced master, an
    empty/metadata-only source, and a successful load are all definitive.
    """

    reachable: bool
    definitive: bool
    detail: str = ""


def observe_raw_reachability(spec_or_source: Any) -> RawReachabilityObservation:
    """Additive typed raw reachability seam for cache-holding consumers.

    Mirrors the reachability half of :func:`describe_source_readiness`
    (whose public boolean default behavior is unchanged) but distinguishes a
    definitive unreachable result from a transient probe error via
    :func:`xrd_tools.sources.probe.observe_first_frame`.
    """

    gated = _spec_gate(spec_or_source)
    if gated is not None:
        return RawReachabilityObservation(
            bool(gated.raw_reachable), True, "spec gate")
    subject_path, subject_kind = _observation_subject(spec_or_source)
    with quiet_capability_observation(subject_path, subject_kind):
        return _observe_raw_reachability_open(spec_or_source)


def _observe_raw_reachability_open(spec_or_source: Any) -> RawReachabilityObservation:
    from xrd_tools.sources.probe import TRANSIENT_PROBE_ERRORS

    # H18-R10: EVERY stage of the observation — open, frame enumeration, and
    # the frame-0 load — flows through one typed path.  A transient failure
    # at any stage is non-definitive (the caller debounces and retries);
    # only an ACTUALLY streaming/live source may use the unknown-length live
    # escape hatch (an enumeration failure on a non-live source is a
    # definitive unreachable, never "live").
    try:
        source = _open_source_observed(spec_or_source)
    except TRANSIENT_PROBE_ERRORS as exc:
        return RawReachabilityObservation(
            False, False, f"transient open error: {exc}")
    if source is None:
        caps = _caps_from_classification(spec_or_source)
        return RawReachabilityObservation(
            bool(caps.raw_reachable), True, "classification fallback")
    source_caps = _core_caps(source)
    truly_live = bool(
        source_caps.is_streaming
        or getattr(source, "kind", None) == SourceKind.LIVE
    )
    if truly_live:
        return RawReachabilityObservation(True, True, "live escape hatch")
    try:
        indices = [int(idx) for idx in source.frame_indices]
    except TRANSIENT_PROBE_ERRORS as exc:
        return RawReachabilityObservation(
            False, False, f"transient enumeration error: {exc}")
    except Exception:
        return RawReachabilityObservation(
            False, True, "frame enumeration failed (non-live)")
    if not indices:
        return RawReachabilityObservation(False, True, "empty scan")
    try:
        from xrd_tools.sources.probe import observe_first_frame

        reachable, _image, transient = observe_first_frame(source)
    except Exception as exc:  # a raised probe error is transient by contract
        return RawReachabilityObservation(False, False, f"probe error: {exc}")
    return RawReachabilityObservation(
        bool(reachable), not transient,
        "frame-0 probe" if not transient else "transient probe error")


def _open_source_observed(value: Any) -> Any | None:
    """H18-R10 open stage: like ``_open_source`` but transient error classes
    PROPAGATE (typed) instead of collapsing into the definitive
    classification fallback.  The duck check itself is guarded — a property
    that raises must not escape untyped."""
    from xrd_tools.sources.probe import TRANSIENT_PROBE_ERRORS

    try:
        is_open_source = (hasattr(value, "frame_indices")
                          and hasattr(value, "load_frame"))
    except TRANSIENT_PROBE_ERRORS:
        raise
    except Exception:
        is_open_source = False
    if is_open_source:
        return value
    try:
        from xrd_tools.sources.registry import open_source

        return open_source(value)
    except TRANSIENT_PROBE_ERRORS:
        raise
    except Exception:
        return None
