"""One GUI adapter over the headless ``xrd_tools.session.project_frame``.

X1 GUI-adoption Slice 1.  This is the single place that turns a scan-qualified
selected-frame request ``(scan_key, frame_index, generation, purpose)`` into one
immutable :class:`~xrd_tools.session.FrameProjection`, so every downstream GUI
consumer (metadata, normalization, wavelength, capability, title, preview) shares
**one** store lookup per render generation instead of each re-deriving frame
facts from ``scan.scan_data`` or the publication store.

Design contracts (handoff non-negotiables 1-3):

* The authoritative store is the session ``FrameRecordStore``.  During a live run
  the wrangler thread owns one; a loaded/browse scan has none, so the adapter
  presents a thin **read-only value view** over the ``PublicationStore`` (whose
  every :class:`FramePublication` already wraps the round-trippable
  ``FrameRecord``).  Either way ``project_frame`` sees the same record surface.
* The value crossing back to the GUI thread is the frozen ``FrameProjection``.
  No live provider, HDF5 handle, ndarray owner block, or mutable record leaves
  this boundary.
* ``project_frame`` is called with ``hydrate=False`` (default): a resident,
  in-memory store lookup only.  Explicit hydration of an evicted old frame is a
  separate bounded-worker concern (Slice 4), never a GUI-thread disk read here.
* Request supersession is centralized: the latest generation wins, and a stale
  (older-generation) request can never overwrite the pinned projection.  One
  lookup is performed per ``(scan_key, frame_index, generation)`` and pinned; a
  repeat request for the same key returns the pinned value without a second
  lookup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from xrd_tools.session import FrameProjection, project_frame

logger = logging.getLogger(__name__)

DISPLAY_PURPOSE = "display"


@dataclass(frozen=True, slots=True)
class ProjectionRequest:
    """A scan-qualified request for one selected frame's projection."""

    scan_key: Any
    frame_index: int | str
    generation: int
    purpose: str = DISPLAY_PURPOSE


class _PublicationBackedStoreView:
    """Read-only ``project_frame`` store surface over a ``PublicationStore``.

    A browse/loaded scan has no session ``FrameRecordStore``; its frames live in
    the bounded ``PublicationStore`` as :class:`FramePublication` values that
    already carry the round-trippable ``FrameRecord`` (``publication.record``).
    This view exposes only the non-hydrating read surface ``project_frame`` needs
    — ``get`` and ``source_identity``.  ``project_frame`` duck-types
    ``hydratable_modes``/``persisted_modes`` and tolerates their absence
    (try/except -> ``None``), so a loaded record's resident results are reported
    ``AVAILABLE`` without this view fabricating persistence facts it cannot know.
    """

    __slots__ = ("_publication_store",)

    def __init__(self, publication_store) -> None:
        self._publication_store = publication_store

    def get(self, label):
        publication = self._publication_store.get(label)
        if publication is None:
            return None
        return getattr(publication, "record", None)

    # A browse view never triggers a GUI-thread disk hydration: get_or_hydrate
    # resolves to the resident record only.  Old-frame hydration is Slice 4's
    # bounded worker.
    get_or_hydrate = get

    def source_identity(self, label) -> str:
        publication = self._publication_store.get(label)
        if publication is None:
            return ""
        return str(getattr(publication, "source_identity", "") or "")


class FrameProjectionAdapter:
    """Centralized selected-frame projection with latest-request-wins pinning.

    Construct with two zero-arg callables that resolve the *current* ownership at
    request time (they change across runs/loads):

    * ``record_store_provider`` -> the session ``FrameRecordStore`` or ``None``;
    * ``publication_store_provider`` -> the bounded ``PublicationStore`` or ``None``.

    Optionally ``metadata_provider_provider`` -> a headless ``MetadataProvider``
    or ``None`` (owner-scoped scanned-motor/counter composition; used from Slice 2
    onward, ``None`` is a valid no-provider projection).
    """

    __slots__ = (
        "_record_store_provider",
        "_publication_store_provider",
        "_metadata_provider_provider",
        "_pinned_key",
        "_pinned",
        "_latest_generation",
        "_lookup_count",
    )

    def __init__(
        self,
        record_store_provider,
        publication_store_provider,
        metadata_provider_provider=None,
    ) -> None:
        self._record_store_provider = record_store_provider
        self._publication_store_provider = publication_store_provider
        self._metadata_provider_provider = metadata_provider_provider
        self._pinned_key: tuple | None = None
        self._pinned: FrameProjection | None = None
        self._latest_generation: int = -1
        self._lookup_count: int = 0

    # -- introspection (tests / diagnostics) -------------------------------- #

    @property
    def lookup_count(self) -> int:
        """Total number of ``project_frame`` lookups performed (memo misses)."""
        return self._lookup_count

    @property
    def pinned(self) -> FrameProjection | None:
        """The projection currently pinned for the latest generation."""
        return self._pinned

    # -- the single lookup boundary ----------------------------------------- #

    def project(
        self,
        request: ProjectionRequest,
        *,
        mode_1d: str | None = None,
        mode_2d: str | None = None,
    ) -> FrameProjection | None:
        """Return the immutable projection for ``request``.

        Supersession: a request older than the latest observed generation is
        dropped (returns ``None``) so a stale completion cannot change the
        display.  For the current generation, one lookup is performed and pinned;
        an identical repeat returns the pinned value with no second lookup.
        Returns ``None`` when no store or record is resolvable (nothing to show).
        """
        if request.generation < self._latest_generation:
            # Superseded: a newer selection/generation has already been pinned.
            return None

        key = (request.scan_key, request.frame_index, request.generation, request.purpose)
        if key == self._pinned_key and self._pinned is not None:
            return self._pinned

        store = self._resolve_store()
        if store is None:
            # No record surface (e.g. a viewer mode with no store): advance the
            # generation watermark but pin nothing.
            self._latest_generation = max(self._latest_generation, request.generation)
            self._pinned_key = None
            self._pinned = None
            return None

        provider = self._resolve_metadata_provider()
        try:
            projection = project_frame(
                store,
                request.frame_index,
                mode_1d=mode_1d,
                mode_2d=mode_2d,
                provider=provider,
            )
        except Exception:
            # project_frame is designed not to raise on conflicts (it returns a
            # typed ERROR projection); a raise here is an unexpected store/adapter
            # fault.  Fail locally and visibly for this frame rather than crash
            # the render, and do not pin a partial result.
            logger.warning(
                "frame projection failed for %r (generation %s)",
                request.frame_index, request.generation, exc_info=True,
            )
            self._latest_generation = max(self._latest_generation, request.generation)
            self._pinned_key = None
            self._pinned = None
            return None

        self._lookup_count += 1
        self._latest_generation = request.generation
        self._pinned_key = key
        self._pinned = projection
        return projection

    def invalidate(self) -> None:
        """Drop the pinned projection (e.g. on scan teardown)."""
        self._pinned_key = None
        self._pinned = None

    # -- ownership resolution ----------------------------------------------- #

    def _resolve_store(self):
        record_store = _call_provider(self._record_store_provider)
        if record_store is not None:
            return record_store
        publication_store = _call_provider(self._publication_store_provider)
        if publication_store is not None:
            return _PublicationBackedStoreView(publication_store)
        return None

    def _resolve_metadata_provider(self):
        return _call_provider(self._metadata_provider_provider)


def _call_provider(provider):
    """Resolve a zero-arg provider callable (or a bare value / ``None``)."""
    if provider is None:
        return None
    if callable(provider):
        try:
            return provider()
        except Exception:
            logger.debug("projection ownership provider failed", exc_info=True)
            return None
    return provider
