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
  lookup is performed per complete :class:`ProjectionRequest` identity and
  pinned; a repeat request returns the pinned value without a second lookup.
"""

from __future__ import annotations

import logging
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xrd_tools.session import FrameProjection, project_frame

logger = logging.getLogger(__name__)

DISPLAY_PURPOSE = "display"

#: Attribute the live run stamps on its ``FrameRecordStore`` to declare which
#: scan the store owns (see ``image_wrangler_thread`` store creation).  A store
#: that declares no identity (``None``) is unqualified and serves any request
#: (legacy/loaded-scan behavior); a store that declares one serves ONLY its scan.
STORE_SCAN_KEY_ATTR = "_xdart_scan_key"


def _store_serves_scan(store, requested_scan_key) -> bool:
    """Whether ``store`` may serve a request for ``requested_scan_key`` (X1-GUI-R2).

    A scan-qualified active-run store serves only its own scan; a store with no
    declared identity is unqualified and serves any request.  This is NOT a
    global preference reversal — an unmatched active store is skipped so the
    caller falls through to the browsed scan's publication-backed source.
    """
    owned = getattr(store, STORE_SCAN_KEY_ATTR, None)
    if owned is None:
        return True
    return requested_scan_key is not None and requested_scan_key == owned


def _scan_identity_candidates(value) -> frozenset[str]:
    """Return comparable spellings for one scan/source identity.

    Display scan keys are usually bare scan names, while publications carry a
    source path.  Compare those forms through the same canonical source-name
    parser used by the wrangler, without touching the filesystem.
    """
    if value in (None, ""):
        return frozenset()
    raw = str(value).strip()
    if not raw:
        return frozenset()
    # A FrameRecord identity may include ``#<frame>``; ownership is the source
    # container/series, not that concrete frame.
    source = raw.rsplit("#", 1)[0]
    portable = source.replace("\\", "/")
    leaf = portable.rsplit("/", 1)[-1]
    candidates = {source, portable, leaf}
    suffix = Path(leaf).suffix.lower()
    if suffix:
        candidates.add(Path(leaf).stem)
        try:
            from .wranglers.image_wrangler_thread import scan_name_from_source

            candidates.add(scan_name_from_source(portable))
        except Exception:
            logger.debug("could not canonicalize publication source identity",
                         exc_info=True)
    return frozenset(item.casefold() for item in candidates if item)


def _normalized_scan_identity(value) -> str:
    """Normalize one explicit scan identity without touching the filesystem."""
    if value in (None, ""):
        return ""
    source = str(value).strip().rsplit("#", 1)[0].replace("\\", "/")
    if not source:
        return ""
    return posixpath.normpath(source).casefold()


def _explicit_scan_owners_match(owner, requested_scan_key) -> bool:
    """Compare explicit owners without collapsing two qualified paths.

    A bare scan name may still be compared with a path-derived spelling for
    compatibility with loaded scans.  Once both sides identify a path, their
    normalized full paths are authoritative: a shared leaf or stem cannot
    prove that they are the same scan.
    """
    owner_identity = _normalized_scan_identity(owner)
    requested_identity = _normalized_scan_identity(requested_scan_key)
    if not owner_identity or not requested_identity:
        return False
    if owner_identity == requested_identity:
        return True
    if "/" in owner_identity and "/" in requested_identity:
        return False
    return bool(
        _scan_identity_candidates(owner)
        & _scan_identity_candidates(requested_scan_key)
    )


def _publication_serves_scan(publication, requested_scan_key) -> bool:
    """Whether a publication belongs to the requested display scan.

    A PublicationStore is shared across scan loads and can briefly retain the
    preceding scan while a new browse request is being established.  Label
    equality alone is therefore insufficient: frame zero is reused by nearly
    every scan.

    X1 Slice 3c (S3-OR1): the EXPLICIT immutable owner
    (``FramePublication.scan_key``, stamped by the production publish sites)
    is preferred and FINAL.  Qualified paths compare by normalized full path;
    the name/path compatibility bridge applies only when one identity is a bare
    scan name.  An explicit mismatch is never rescued by source-name inference.
    Only legacy UNSTAMPED publications fall back to the same positive source-
    identity proof: a per-frame source (e.g. ``frame_0001.tif``) that cannot
    prove its scan fails closed.
    """
    if publication is None:
        return False
    if requested_scan_key is None:
        return True
    requested = _scan_identity_candidates(requested_scan_key)
    if not requested:
        return False
    owner = getattr(publication, "scan_key", None)
    if owner is not None:
        return _explicit_scan_owners_match(owner, requested_scan_key)
    sources = {
        getattr(publication, "source_identity", None),
        getattr(getattr(publication, "view", None), "source_path", None),
        getattr(
            getattr(getattr(publication, "record", None), "view", None),
            "source_path",
            None,
        ),
    }
    return any(
        _explicit_scan_owners_match(source, requested_scan_key)
        for source in sources
        if source not in (None, "")
    )


@dataclass(frozen=True, slots=True)
class ProjectionRequest:
    """A scan-qualified request for one selected frame's projection.

    ``mode_1d``/``mode_2d`` are the canonical active display modes; they are part
    of the intrinsic identity (X1-GUI-R2/R3) because ``project_frame`` derives
    capabilities from them, so a mode change must yield a fresh projection even
    within one render generation.
    """

    scan_key: Any
    frame_index: int | str
    generation: int
    purpose: str = DISPLAY_PURPOSE
    mode_1d: str | None = None
    mode_2d: str | None = None


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

    __slots__ = ("_publication_store", "_scan_key")

    def __init__(self, publication_store, scan_key=None) -> None:
        self._publication_store = publication_store
        self._scan_key = scan_key

    def _publication(self, label):
        publication = self._publication_store.get(label)
        if not _publication_serves_scan(publication, self._scan_key):
            return None
        return publication

    def get(self, label):
        publication = self._publication(label)
        if publication is None:
            return None
        return getattr(publication, "record", None)

    # A browse view never triggers a GUI-thread disk hydration: get_or_hydrate
    # resolves to the resident record only.  Old-frame hydration is Slice 4's
    # bounded worker.
    get_or_hydrate = get

    def source_identity(self, label) -> str:
        publication = self._publication(label)
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

    def project(self, request: ProjectionRequest) -> FrameProjection | None:
        """Return the immutable projection for ``request``.

        Supersession: a request older than the latest observed generation is
        dropped (returns ``None``) so a stale completion cannot change the
        display.  For the current generation, one lookup is performed and pinned;
        an identical repeat (same scan/frame/generation/purpose AND canonical
        modes) returns the pinned value with no second lookup.  A mode change at
        the same generation is a distinct identity and performs a fresh lookup.
        Returns ``None`` when no store or record is resolvable (nothing to show).
        """
        if request.generation < self._latest_generation:
            # Superseded: a newer selection/generation has already been pinned.
            return None

        key = (
            request.scan_key, request.frame_index, request.generation,
            request.purpose, request.mode_1d, request.mode_2d,
        )
        if key == self._pinned_key and self._pinned is not None:
            return self._pinned

        store = self._resolve_store(request)
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
                mode_1d=request.mode_1d,
                mode_2d=request.mode_2d,
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

    def _resolve_store(self, request):
        # X1-GUI-R2: the active-run FrameRecordStore serves a request only when
        # its declared scan identity matches the requested scan.  During a paused
        # run the live store stays attached while the operator browses another
        # scan; using it unconditionally could project the active scan's frame
        # while the raw/cake/publication path shows the browsed scan's same
        # label.  An unmatched active store is skipped (NOT globally
        # de-preferred) so browsing falls through to the browsed scan's
        # publication-backed source; when neither owns the request the projection
        # fails closed (absent record).
        record_store = _call_provider(self._record_store_provider)
        if record_store is not None and _store_serves_scan(
                record_store, request.scan_key):
            return record_store
        publication_store = _call_provider(self._publication_store_provider)
        if publication_store is not None:
            return _PublicationBackedStoreView(
                publication_store, scan_key=request.scan_key)
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
