# -*- coding: utf-8 -*-
"""Ownership records for the acquisition/browse display split (X1 O-3).

The defect these types exist to remove is a mutable singleton: acquisition,
paused browse, display, integrator, writer and hydration all aliased ONE
``LiveScan``.  Loading a browsed scan mutated the object the paused run still
owned, and Resume then tried to recover the run by writing back into that same
object.  Three owners replace it:

``AcquisitionContext``
    The display-side identity and lifetime of the exact admitted
    ``FrozenRunConfiguration``: run identity, the acquisition ``LiveScan``, its
    display bindings and stores, and DETACHED calibration/mask/geometry stamps.
    It does not own or mutate the writer, PONI, mask or scientific
    configuration — the existing execution/writer owners stay authoritative.

``BrowseContext``
    One independently loaded processed scan: its own ``LiveScan``, its own
    display bindings and stores, its detached persisted provenance, the
    requested path and the exact browse load receipt.  It is not a second
    scientific authority.

``DisplaySelection``
    An immutable pointer choosing which context and which display generation
    the panels render.  Resume is a new selection, never a restoration.

Deliberately **Qt-free**: stdlib only — no GUI toolkit, no plotting library, no
HDF5, no pyFAI, no image reader, no array library — so the ownership contract
can be asserted headlessly and so importing it can never drag the GUI stack
into a display-logic test.  (The acceptance oracle checks this by scanning THIS
source for toolkit names, so none may appear here, not even in prose.)  The
records DO hold live object references — they are runtime owners, not value
snapshots — but every identity field is write-once and every mutable field
names its single writer.

Ownership rules enforced here rather than by convention:

* identity fields raise :class:`DisplayContextError` on reassignment, so an
  "alias snapshot" cannot quietly become a second mutable acquisition owner;
* :class:`DisplayBindings` is the COMPLETE display swap surface, built by the
  owning context — a partial swap is a construction error, not a silent
  mixed-context render;
* the acquisition finalization claim is one-shot, so a retried run-end seam can
  never finalize the same identity twice.
"""

from __future__ import annotations

import itertools
import os
import threading
from dataclasses import dataclass, field, fields as dataclass_fields
from enum import Enum

__all__ = [
    "CommitGate",
    "HydrationRequest",
    "FINALIZATION_FINALIZED",
    "FINALIZATION_IN_PROGRESS",
    "FINALIZATION_PENDING",
    "AcquisitionContext",
    "BrowseContext",
    "ContextKind",
    "DisplayBindings",
    "DisplayContextError",
    "DisplaySelection",
    "new_context_token",
]


#: The run-end finalization states (§9.1).  Explicit, because "an attempt was
#: made" and "the finalization succeeded" are different facts and only the
#: second may authorise releasing the run's identity.
FINALIZATION_PENDING = "pending"
FINALIZATION_IN_PROGRESS = "in_progress"
FINALIZATION_FINALIZED = "finalized"


class ContextKind(str, Enum):
    """Which owner a display selection points at."""

    ACQUISITION = "acquisition"
    BROWSE = "browse"


class DisplayContextError(RuntimeError):
    """The ONE typed error for a display-context ownership violation.

    Raised for an identity reassignment, a selection built for a context it
    does not name, and a browse admission whose receipt does not match its
    context.  A context mismatch is a typed error, never a silent fallback.
    """


#: Process-wide monotonic context counter.  ONE authority: a browse token, its
#: load receipt token and the selection that names it are all this counter's,
#: so no second token/generation source can drift from it.
_token_lock = threading.Lock()
_token_counter = itertools.count(1)

_UNSET = object()


def new_context_token(kind) -> str:
    """Mint one process-unique context token.

    The pid prefix keeps tokens comparable across a captured log from more than
    one process; the counter is the identity.
    """
    with _token_lock:
        serial = next(_token_counter)
    return f"{ContextKind(kind).value}-{os.getpid():x}-{serial:x}"


class _WriteOnceIdentity:
    """Mixin making the declared identity fields assignable exactly once.

    ``__slots__ = ()`` is load-bearing: the concrete records are ``slots=True``
    dataclasses, and a mixin with an implicit ``__dict__`` would silently give
    every instance one back.
    """

    __slots__ = ()

    #: Field names that may never be reassigned after construction.
    _IDENTITY_FIELDS: frozenset = frozenset()

    def __setattr__(self, name, value):
        if (name in self._IDENTITY_FIELDS
                and getattr(self, name, _UNSET) is not _UNSET):
            raise DisplayContextError(
                f"{type(self).__name__}.{name} is a write-once identity field; "
                "a display context is replaced, never re-pointed")
        object.__setattr__(self, name, value)


class CommitGate:
    """The linearization point between a background read and its insertion.

    A hydration request reads from disk OFF the GUI thread and UNLOCKED — that
    is the whole point of the background worker — and then inserts what it read
    into a store.  Between those two moments the display can move: Resume can
    invalidate a browse, a rescope can move the acquisition's sub-scan, a
    replacement can release a context.  Checking ownership only when the
    completion reaches the GUI is too late, because by then the payload is
    already in a store.

    So the gate is held around the INSERT and nothing else.  ``cancel()`` runs
    on the GUI thread and therefore waits, at worst, for one bounded store
    upsert — never for an ``.nxs`` open.  A request carries the epoch it was
    minted under; once the epoch moves or the gate is cancelled, the request may
    read to completion but may insert nowhere.

    This is not a fourth authority: exactly one gate belongs to each of the two
    context owners, created with them and cancelled by their own lifecycle.
    """

    __slots__ = ("_lock", "_epoch", "_cancelled")

    def __init__(self):
        self._lock = threading.Lock()
        self._epoch = 1
        self._cancelled = False

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def enter(self, epoch) -> bool:
        """Take the commit window for *epoch*, or refuse it.

        On ``True`` the caller HOLDS the gate and must call :meth:`leave`.
        """
        self._lock.acquire()
        if self._cancelled or epoch != self._epoch:
            self._lock.release()
            return False
        return True

    def leave(self) -> None:
        try:
            self._lock.release()
        except RuntimeError:
            pass

    def advance(self) -> int:
        """Move to a new epoch, invalidating every request minted before it."""
        with self._lock:
            self._epoch += 1
            return self._epoch

    def cancel(self) -> None:
        """Permanently withdraw commit authority (idempotent)."""
        with self._lock:
            self._cancelled = True
            self._epoch += 1


def _clear(obj) -> None:
    """``clear()`` an owned container/store.  Failures PROPAGATE.

    A swallowed release failure is a retained payload nobody can see: the
    run-end projection would record the seam as complete while the browse's
    store still held its frames.  The caller is a receipt substep, so raising
    is what gets the seam recorded and retried.
    """
    clear = getattr(obj, "clear", None)
    if callable(clear):
        clear()


@dataclass(frozen=True, slots=True)
class DisplayBindings:
    """The COMPLETE set of display-side bindings one context owns.

    Every field here is swapped together by the selection owner.  ``frame``,
    ``frame_ids``, ``frames`` and both viewer-row mappings are inside the swap
    surface deliberately: normal processed-scan browse and hydration read them
    and ``H5Viewer.data_reset()`` clears them, so leaving any of them shared
    would let a browse clear or repopulate the acquisition's rows while the
    scan and store pointers merely LOOKED split.
    """

    scan: object
    frame: object
    frame_ids: object
    frames: object
    viewer_rows_1d: object
    viewer_rows_2d: object
    record_store: object
    publication_store: object

    @classmethod
    def field_names(cls) -> tuple:
        return tuple(f.name for f in dataclass_fields(cls))


@dataclass(frozen=True, slots=True)
class HydrationRequest:
    """Everything a background hydration needs, decided when it was REQUESTED.

    The target stores and the commit authority travel WITH the request, so the
    worker never asks a live provider where a completed read belongs.  That
    lookup, done at execution time, is how a read made under a browse landed in
    the resumed run's store.
    """

    label: object
    purpose: str
    generation: int
    context_token: str
    context_scan_key: str
    context_source: str
    stores: tuple
    commit_gate: object
    epoch: int

    @property
    def owner(self) -> tuple:
        """The identity a completion must echo."""
        return (self.context_token, self.context_scan_key)


@dataclass(frozen=True, slots=True)
class DisplaySelection:
    """Which context, and at which display generation, the panels render.

    Frame identity is deliberately NOT here: the generation-stamped render pin
    remains the sole frame-selection owner, and a frame field would make this a
    second one.  Built only by :meth:`for_context`, which the GUI selection
    owner calls AFTER it has stamped the new display generation.
    """

    kind: ContextKind
    context_token: str
    scan_key: str
    source_path: str
    display_generation: int

    @classmethod
    def for_context(cls, context, display_generation: int) -> "DisplaySelection":
        """Stamp a selection naming *context* at an ALREADY-bumped generation."""
        return cls(
            kind=context.kind,
            context_token=context.context_token,
            scan_key=str(context.scan_key or ""),
            source_path=str(context.source_path or ""),
            display_generation=int(display_generation),
        )

    def names(self, context) -> bool:
        """Whether this selection is the one that named *context*."""
        return (context is not None
                and self.kind is context.kind
                and self.context_token == context.context_token)


@dataclass(slots=True)
class AcquisitionContext(_WriteOnceIdentity):
    """Runtime owner of the active run's display-side identity.

    Composed with — never extending — the frozen run configuration: the
    ``FrozenRunConfiguration`` stays a value object with its own admission
    owner, and this record adds the object-valued state plus the DETACHED
    identity stamps a later boundary can compare without touching the scan.

    Constructed ONCE by the run-admission owner.  Workers may populate the
    scan and stores it already owns; no GUI browse path may replace its fields.
    """

    context_token: str
    #: The EXACT admitted ``FrozenRunConfiguration`` (``is``-identical to the
    #: wrangler's admission ledger), or ``None`` for a run that has no wrangler
    #: admission at all (reintegrate/stitch).  An equal-valued, reconstructed,
    #: stale or fallback configuration is refused by the caller and lands here
    #: as ``None`` rather than as a tolerated substitute.
    run_configuration: object
    config_generation: int | None
    config_fingerprint: str
    #: Canonical scan identity stamped once at admission.
    run_scan_key: str
    source_path: str
    scan: object
    frame: object
    frame_ids: object
    frames: object
    viewer_rows_1d: object
    viewer_rows_2d: object
    publication_store: object
    #: Detached geometry/mask stamps — strings, so comparing them can never
    #: resurrect or mutate the array they describe.
    #: WHICH owner started this run (§8.1) — wrangler, reintegrate or stitch.
    #: Recorded so a later reader can tell "this run had no frozen
    #: configuration because it is a reintegrate" from "this run lost one".
    origin: str = ""
    poni_identity: str = ""
    mask_identity: str = ""
    geometry_identity: str = ""
    #: The live sub-scan key.  SINGLE WRITER: :meth:`rescope_to`, driven by the
    #: frame-driven scan-boundary owner.  Initialised to ``run_scan_key``.
    current_scan_key: str = ""
    #: This context's commit authority (§9.2.4).  Created with the context and
    #: cancelled by its own lifecycle — one per owner, not a new authority.
    commit_gate: CommitGate = field(default_factory=CommitGate)
    #: The run's ``FrameRecordStore`` once the streaming session creates one.
    #: SINGLE WRITER: :meth:`adopt_record_store`.
    record_store: object = None
    #: The run-end finalization state machine (§9.1).  A failed attempt
    #: returns to PENDING so the seam is genuinely retryable; only a
    #: successful attempt reaches FINALIZED, and only FINALIZED authorises
    #: release.  The parent consumed a one-shot claim BEFORE the fallible
    #: finalizer, so a failure could never be retried while the release seam —
    #: which asked only whether the claim had been taken — dropped the context
    #: anyway, and the scientific finalization was silently never done.
    #: SINGLE WRITER: :meth:`begin_finalization` / :meth:`fail_finalization` /
    #: :meth:`complete_finalization`.
    finalization_state: str = FINALIZATION_PENDING
    #: How many attempts have been made.  A DIAGNOSTIC fact only — never
    #: release authority.
    finalization_attempts: int = 0

    _IDENTITY_FIELDS = frozenset({
        "context_token", "run_configuration", "config_generation",
        "config_fingerprint", "run_scan_key", "source_path", "scan", "frame",
        "frame_ids", "frames", "viewer_rows_1d", "viewer_rows_2d",
        "publication_store", "origin", "poni_identity", "mask_identity",
        "geometry_identity",
    })

    def __post_init__(self):
        if not self.current_scan_key:
            self.current_scan_key = str(self.run_scan_key or "")

    @property
    def commit_epoch(self) -> int:
        return self.commit_gate.epoch

    @property
    def kind(self) -> ContextKind:
        return ContextKind.ACQUISITION

    @property
    def scan_key(self) -> str:
        """The key the display is currently scoped to within this run."""
        return self.current_scan_key or self.run_scan_key

    def rescope_to(self, scan_key) -> None:
        """Stamp the new sub-scan key at a genuine scan boundary.

        The commit epoch moves with it (§9.2.2): a request minted against the
        previous sub-scan is qualified against a key the display has left, so
        it must no longer be able to insert.
        """
        self.current_scan_key = str(scan_key or "")
        self.commit_gate.advance()

    def adopt_record_store(self, store) -> None:
        """Adopt the streaming session's per-run record store."""
        self.record_store = store

    def display_bindings(self) -> DisplayBindings:
        """The complete display surface the selection owner swaps IN."""
        return DisplayBindings(
            scan=self.scan,
            frame=self.frame,
            frame_ids=self.frame_ids,
            frames=self.frames,
            viewer_rows_1d=self.viewer_rows_1d,
            viewer_rows_2d=self.viewer_rows_2d,
            record_store=self.record_store,
            publication_store=self.publication_store,
        )

    @property
    def finalized(self) -> bool:
        """Whether the run scan's finalization has SUCCEEDED."""
        return self.finalization_state == FINALIZATION_FINALIZED

    @property
    def finalization_pending(self) -> bool:
        """Whether a finalization attempt may still be made."""
        return self.finalization_state == FINALIZATION_PENDING

    def begin_finalization(self):
        """Take the run scan for ONE attempt, or ``None``.

        ``None`` means either that an attempt is already in flight or that the
        scan has already been finalized — so a retry can neither race nor
        finalize the same identity twice.
        """
        if self.finalization_state != FINALIZATION_PENDING:
            return None
        self.finalization_state = FINALIZATION_IN_PROGRESS
        self.finalization_attempts += 1
        return self.scan

    def fail_finalization(self) -> None:
        """Return a failed attempt to the retryable state."""
        if self.finalization_state == FINALIZATION_IN_PROGRESS:
            self.finalization_state = FINALIZATION_PENDING

    def complete_finalization(self) -> None:
        """Record the ONE successful finalization."""
        self.finalization_state = FINALIZATION_FINALIZED


@dataclass(slots=True)
class BrowseContext(_WriteOnceIdentity):
    """Owner of ONE paused-run browse: its scan, its stores, its receipt.

    At most one exists at a time; replacing it releases the previous one.  The
    GUI request owner allocates the resources, the file task populates only
    those resources, and the GUI admission owner either installs the context
    once or releases it.  Nothing here is a second scientific authority: the
    calibration/mask/result stamps are detached provenance strings read back
    off the loaded scan for the trace, never applied to anything.
    """

    context_token: str
    load_generation: int
    #: The EXACT browse-load receipt this context's task carries, or ``None``
    #: when the diagnostic channel is off.  There is no second token slot: when
    #: a receipt exists its token IS ``context_token``.
    operation: object
    requested_path: str
    #: Canonical scan name (``scan_name_from_source``) — the ONE parser.
    scan_key: str
    scan: object
    frame: object
    frame_ids: object
    frames: object
    viewer_rows_1d: object
    viewer_rows_2d: object
    publication_store: object
    #: A browse serves from its own publication store; it never borrows the
    #: acquisition's scan-qualified record store.  Present so the swap surface
    #: is complete by construction rather than by omission.
    record_store: object = None
    #: This context's commit authority (§9.2.4).  Created with the context and
    #: cancelled by its own lifecycle — one per owner, not a new authority.
    commit_gate: CommitGate = field(default_factory=CommitGate)
    #: The EXACT immutable load request this context enqueued (the file task
    #: itself).  Admission compares the completion against THIS object, not
    #: against a token that a reconstructed task could also carry.  SINGLE
    #: WRITER: :meth:`adopt_load_request`, once, at enqueue.
    load_request: object = None
    #: SINGLE WRITER: the GUI admission owner.
    loaded: bool = False
    #: No further completion may be admitted.  Resume sets this WITHOUT
    #: discarding what the browse already holds: a read that was already in
    #: flight still lands in the browse's own store, where it belongs and
    #: where it can be seen to have landed — it simply cannot reach the
    #: display any more.
    invalidated: bool = False
    #: The owned payload has been dropped as well (replacement, run end).
    released: bool = False
    calibration_identity: str = ""
    mask_identity: str = ""
    result_identity: str = ""

    _IDENTITY_FIELDS = frozenset({
        "context_token", "load_generation", "operation", "requested_path",
        "scan_key", "scan", "frame", "frame_ids", "frames", "viewer_rows_1d",
        "viewer_rows_2d", "publication_store", "record_store",
    })

    def __post_init__(self):
        operation = self.operation
        if operation is not None:
            token = getattr(operation, "token", None)
            generation = getattr(operation, "load_generation", None)
            if token != self.context_token or generation != self.load_generation:
                raise DisplayContextError(
                    "a browse receipt must carry its own context token and "
                    f"load generation: receipt=({token!r}, {generation!r}) "
                    f"context=({self.context_token!r}, {self.load_generation!r})")

    @property
    def commit_epoch(self) -> int:
        return self.commit_gate.epoch

    @property
    def kind(self) -> ContextKind:
        return ContextKind.BROWSE

    @property
    def source_path(self) -> str:
        return self.requested_path

    def adopt_load_request(self, request) -> None:
        """Retain the exact enqueued request (write-once)."""
        if self.load_request is not None:
            raise DisplayContextError(
                "a browse context enqueues exactly one load request")
        self.load_request = request

    def admits(self, request) -> bool:
        """Whether *request* is EXACTLY the load this context is waiting for.

        Identity first — the completion must be the object this context
        enqueued — and then every value it carries is re-checked against the
        context.  A token and a generation are not an identity: a reconstructed
        request carrying the same pair but a foreign scan, path, name or receipt
        would otherwise cross a fail-closed admission boundary.
        """
        if self.released or self.invalidated or self.load_request is None:
            return False
        if request is not self.load_request:
            return False
        return (getattr(request, "context_token", None) == self.context_token
                and getattr(request, "load_generation", None)
                == self.load_generation
                and getattr(request, "scan", None) is self.scan
                and str(getattr(request, "fname", "")) == self.requested_path
                and str(getattr(request, "scan_name", "")) == self.scan_key
                and getattr(request, "operation", None) is self.operation)

    def matches(self, context_token, load_generation) -> bool:
        """Whether a completion names this context's token and load."""
        return (not self.released
                and not self.invalidated
                and context_token == self.context_token
                and load_generation == self.load_generation)

    def stamp_provenance(self, *, calibration="", mask="", result="") -> None:
        """Record the loaded scan's detached provenance (admission owner)."""
        self.calibration_identity = str(calibration or "")
        self.mask_identity = str(mask or "")
        self.result_identity = str(result or "")

    def mark_loaded(self) -> None:
        self.loaded = True

    def display_bindings(self) -> DisplayBindings:
        return DisplayBindings(
            scan=self.scan,
            frame=self.frame,
            frame_ids=self.frame_ids,
            frames=self.frames,
            viewer_rows_1d=self.viewer_rows_1d,
            viewer_rows_2d=self.viewer_rows_2d,
            record_store=self.record_store,
            publication_store=self.publication_store,
        )

    def invalidate(self) -> None:
        """Refuse every further completion, WITHOUT dropping the payload.

        This is what Resume does to a browse.  A read already in flight still
        completes into this context's own store — that is where it belongs, and
        destroying the store underneath it would turn a clean rejection into a
        half-written one — but nothing it produces can reach the display again.
        Idempotent.
        """
        self.invalidated = True
        self.loaded = False
        # §9.2.7 — commit authority goes FIRST.  A read already in flight may
        # finish reading; it may insert into nothing.
        self.commit_gate.cancel()

    def release(self) -> None:
        """Invalidate this browse AND drop everything it retained.

        Idempotent.  The identity fields stay bound so a late completion can
        still be REJECTED by token rather than crashing on a half-nulled
        record; what goes away is the retained payload, and the owner then
        drops its reference to the context itself.
        """
        if self.released:
            return
        # Invalidate FIRST: whatever happens to the payload, no completion may
        # be admitted from here on.  ``released`` is set only once every owned
        # container is actually empty, so a failed release is retried by the
        # run-end projection instead of being recorded as done.
        self.invalidate()
        for owned in (self.publication_store, self.frames, self.frame_ids,
                      self.viewer_rows_1d, self.viewer_rows_2d):
            _clear(owned)
        self.released = True
