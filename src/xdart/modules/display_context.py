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
from dataclasses import InitVar, dataclass, field, fields as dataclass_fields
from enum import Enum
from typing import Protocol, runtime_checkable

from xrd_tools.session.hydration import (
    HydrationPurpose,
    HydrationReadKey,
    HydrationScope,
    HydrationToken,
    normalize_hydration_purpose,
)

__all__ = [
    "CommitGate",
    "CurrentDisplayScope",
    "HydrationOwner",
    "HydrationRequest",
    "OneDViewerCommitPort",
    "Prepared1DBatchCommit",
    "Prepared2DCatalogCommit",
    "Prepared2DFrameCommit",
    "TwoDViewerCommitPort",
    "Viewer2DCatalogHydrationRequest",
    "Viewer2DCommitGate",
    "Viewer2DCleanupReceipt",
    "Viewer2DCleanupState",
    "Viewer2DContext",
    "Viewer2DDisposal",
    "Viewer2DDisposalToken",
    "Viewer2DFrameHydrationRequest",
    "Viewer2DReadActivation",
    "Viewer2DReadBudgetReceipt",
    "Viewer2DReceiptPhase",
    "Viewer2DRendererClearReceipt",
    "Viewer2DRendererClearRequest",
    "Viewer2DState",
    "Viewer1DBatchHydrationRequest",
    "Viewer1DCommitGate",
    "Viewer1DCommitReceipt",
    "Viewer1DContext",
    "Viewer1DCleanupPendingNotice",
    "Viewer1DReadBudgetReceipt",
    "Viewer1DReadBudgetState",
    "Viewer1DRendererClearReceipt",
    "Viewer1DRendererClearRequest",
    "Viewer1DState",
    "acknowledge_viewer_1d_cleanup_pending",
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
    VIEWER_2D = "viewer_2d"
    VIEWER_1D = "viewer_1d"


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

    __slots__ = ("_lock", "_epoch", "_cancelled", "_reserved_epoch")

    def __init__(self):
        self._lock = threading.Lock()
        self._epoch = 1
        self._cancelled = False
        self._reserved_epoch = 0

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
            if self._reserved_epoch:
                epoch = self._reserved_epoch
                self._reserved_epoch = 0
                return epoch
            self._epoch += 1
            return self._epoch

    def reserve_advance(self) -> int:
        """Invalidate the old epoch before an owner publishes replacement state.

        The owner then calls :meth:`advance` to consume this one reservation
        before publishing its immutable replacement reference.  The narrow
        interval between the calls can expose only the old coherent reference,
        whose epoch this gate already refuses.
        """
        with self._lock:
            if self._reserved_epoch:
                raise DisplayContextError("a commit-gate advance is in progress")
            self._epoch += 1
            self._reserved_epoch = self._epoch
            return self._epoch

    def cancel(self) -> None:
        """Permanently withdraw commit authority (idempotent)."""
        with self._lock:
            self._cancelled = True
            self._reserved_epoch = 0
            self._epoch += 1


class Viewer2DCommitGate(CommitGate):
    __slots__ = ()


class Viewer1DCommitGate(CommitGate):
    __slots__ = ()


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
class CurrentDisplayScope:
    """One coherent acquisition display scope, published by one reference."""

    scan_key: str
    source: str
    display_scan: object
    commit_epoch: int


def _owner_text(value) -> str:
    """One text rule for every owner field.  Never raises (§12.6 A.2)."""
    if value is None or type(value) in (bytes, bytearray):
        return ""
    if type(value) is str:
        return value
    # A ``str`` subclass can override truthiness or conversion.  It is not an
    # already-canonical value, and accepting it unchanged would let that code
    # execute later from ``qualified`` on the GUI completion path.
    if type(value) is not str and issubclass(type(value), str):
        return ""
    try:
        if not value:
            return ""
        text = str(value)
    except BaseException:
        return ""
    # Keep the stored value exact and inert even when a coercion protocol
    # returns a hostile ``str`` subclass.
    return text if type(text) is str else ""


def _owner_epoch(value) -> int:
    """A POSITIVE INTEGER epoch, or ``0``.  Never raises (§12.6 A.2).

    A bool is not an epoch, and neither is a string, a float or anything that
    merely coerces to one: those are malformed inputs, and inventing a number
    from them is how a forged owner compared equal to a real one.
    """
    # Exact ``int`` only.  An ``int`` subclass can override comparison and
    # escape this total decoder just as readily as a coercion-only object.
    if type(value) is not int:
        return 0
    return value if value > 0 else 0


@dataclass(frozen=True, slots=True)
class HydrationOwner:
    """WHO a hydration belongs to — values only (§10.3).

    ONE envelope, reused for the worker's queue identity, for the completion it
    echoes and for the GUI admission that follows.  Carrying a source string
    that no decision reads is not source qualification: two requests differing
    only in source collapsed into one dedupe token, so the second sub-scan's
    frame was never read.
    """

    context_token: str = ""
    scan_key: str = ""
    source: str = ""
    epoch: int = 0

    def __post_init__(self):
        """§12.6 A.1 — THE one construction contract.

        Normalization lives in the dataclass construction path itself, so a
        caller cannot choose which invariant the value object enforces by
        picking a constructor.  ``of()`` delegates here; it is not a second
        contract.  Any input yields either a complete canonical owner or the
        inert empty one, and nothing raises — the decode runs on a completion
        delivered through a Qt signal, where an exception would escape into the
        render path instead of dropping one stale completion.
        """
        object.__setattr__(self, "context_token", _owner_text(self.context_token))
        object.__setattr__(self, "scan_key", _owner_text(self.scan_key))
        object.__setattr__(self, "source", _owner_text(self.source))
        object.__setattr__(self, "epoch", _owner_epoch(self.epoch))

    @classmethod
    def of(cls, context_token="", scan_key="", source="", epoch=0):
        """Delegates to the ONE normalizing construction path."""
        return cls(context_token, scan_key, source, epoch)

    @property
    def qualified(self) -> bool:
        """Whether this owner is COMPLETE (§11.1.1, §12.6 A.3).

        All four fields, or none of the authority.  Reads only ALREADY
        NORMALIZED fields, so it is total: an epoch whose comparison or
        truthiness raises was turned into ``0`` at construction and can never
        reach this expression.
        """
        return bool(self.context_token and self.scan_key and self.source
                    and self.epoch > 0)

    def as_tuple(self) -> tuple:
        return (self.context_token, self.scan_key, self.source, self.epoch)


@dataclass(frozen=True, slots=True)
class HydrationRequest:
    """Everything a background hydration needs, decided when it was REQUESTED.

    The target stores and the commit authority travel WITH the request, so the
    worker never asks a live provider where a completed read belongs.  That
    lookup, done at execution time, is how a read made under a browse landed in
    the resumed run's store.
    """

    label: int | str
    purpose: HydrationPurpose
    generation: int
    #: §12.6 B.2 — the context's OWN projection, stored whole.  Four parallel
    #: scalars meant the owner was RECONSTRUCTED at every hop, and each
    #: reconstruction was another chance to pick a different source or a
    #: different normalization rule.
    owner: HydrationOwner
    stores: tuple
    commit_gate: object
    read_key: HydrationReadKey | None = None
    token: HydrationToken | None = None
    scope: HydrationScope = field(init=False)

    def __post_init__(self):
        if (
            type(self.label) not in (int, str)
            or self.label == ""
            or type(self.generation) is not int
            or self.generation < 0
        ):
            raise TypeError("hydration label/generation must be exact values")
        if type(self.owner) is not HydrationOwner or type(self.stores) is not tuple:
            raise TypeError("hydration owner and store references are malformed")
        purpose = normalize_hydration_purpose(self.purpose)
        scope = HydrationScope(*self.owner.as_tuple())
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "scope", scope)
        if (self.read_key is None) != (self.token is None):
            raise ValueError("read_key and token must be supplied together")
        if self.read_key is None:
            return
        if (
            type(self.read_key) is not HydrationReadKey
            or type(self.token) is not HydrationToken
        ):
            raise TypeError("typed hydration identity is malformed")
        if (
            self.token.read_key != self.read_key
            or self.read_key.scope != scope
            or self.read_key.purpose is not purpose
            or self.read_key.frame_identity != self.label
            or self.token.presentation_generation != self.generation
        ):
            raise ValueError("hydration request identity is inconsistent")

    @property
    def context_token(self) -> str:
        return self.owner.context_token

    @property
    def context_scan_key(self) -> str:
        return self.owner.scan_key

    @property
    def context_source(self) -> str:
        return self.owner.source

    @property
    def epoch(self) -> int:
        return self.owner.epoch

    @property
    def enqueueable(self) -> bool:
        """Whether this request may reach the worker at all (§12.6 D).

        An unresolvable owner produces a request that exists — so the caller
        can diagnose it — but can never be enqueued or committed.
        """
        return bool(self.owner.qualified and self.stores
                    and self.commit_gate is not None)


class Viewer2DState(str, Enum):
    EMPTY = "empty"
    CATALOG_LOADING = "catalog_loading"
    FRAME_LOADING = "frame_loading"
    READY = "ready"
    CLEANUP_PENDING = "cleanup_pending"
    CLOSED = "closed"


class Viewer1DState(str, Enum):
    EMPTY = "empty"
    LOADING = "loading"
    READY = "ready"
    CLEANUP_PENDING = "cleanup_pending"
    CLOSED = "closed"


class Viewer1DReadBudgetState(str, Enum):
    ACTIVE = "active"
    RETIRED = "retired"


@runtime_checkable
class OneDViewerCommitPort(Protocol):
    def commit(self, prepared): ...
    def cleanup_pending(self, notice): ...
    def complete(self, completion): ...


@dataclass(frozen=True, slots=True)
class Viewer1DContext:
    context_token: str
    generation: int
    paths: tuple[str, ...]
    commit_gate: Viewer1DCommitGate
    state: Viewer1DState = Viewer1DState.EMPTY

    def __post_init__(self):
        _viewer_malformed(type(self.context_token) is not str or not self.context_token
            or type(self.generation) is not int or self.generation < 0
            or type(self.paths) is not tuple or not 1 <= len(self.paths) <= 256
            or any(type(path) is not str or not path or len(os.fsencode(path)) > 4096
                   for path in self.paths)
            or type(self.commit_gate) is not Viewer1DCommitGate
            or type(self.state) is not Viewer1DState, "viewer 1-D context is malformed")

    @property
    def kind(self): return ContextKind.VIEWER_1D


@dataclass(frozen=True, slots=True)
class Viewer1DBatchHydrationRequest:
    paths: tuple[str, ...]
    policy: object
    generation: int
    commit_gate: Viewer1DCommitGate
    port: OneDViewerCommitPort
    read_key: HydrationReadKey
    token: HydrationToken
    gui_thread_id: int
    owner_identity: object
    owner_request_claim: object
    admitted_provider_identity: object

    def __post_init__(self):
        from xrd_tools.io.viewer_1d import Viewer1DFormatPolicy
        scope = self.read_key.scope if type(self.read_key) is HydrationReadKey else None
        _viewer_malformed(type(self.paths) is not tuple or not 1 <= len(self.paths) <= 256
            or any(type(path) is not str or not path or len(os.fsencode(path)) > 4096
                   for path in self.paths) or type(self.policy) is not Viewer1DFormatPolicy
            or type(self.generation) is not int or self.generation <= 0
            or type(self.commit_gate) is not Viewer1DCommitGate
            or not isinstance(self.port, OneDViewerCommitPort)
            or type(self.read_key) is not HydrationReadKey or type(self.token) is not HydrationToken
            or self.token.read_key != self.read_key
            or self.token.presentation_generation != self.generation
            or scope.scan_key != "viewer-1d" or scope.source != "viewer-1d"
            or scope.epoch != self.commit_gate.epoch
            or self.read_key.artifact_identity != "viewer-1d"
            or self.read_key.frame_identity != "batch"
            or self.read_key.purpose is not HydrationPurpose.ONE_D
            or type(self.gui_thread_id) is not int or self.gui_thread_id <= 0
            or self.owner_identity is None or self.owner_request_claim is None
            or self.admitted_provider_identity is None, "viewer 1-D request is malformed")

    @property
    def enqueueable(self): return not self.commit_gate.cancelled


_VIEWER_1D_FACTORY = object()


@dataclass(frozen=True, slots=True)
class Viewer1DReadBudgetReceipt:
    identity: object; request: Viewer1DBatchHydrationRequest; transport_token: HydrationToken
    capacity_bytes: int; reserved_bytes: int; gui_thread_id: int; worker_thread_id: int
    issuer: object; _claim: InitVar[object] = None
    _retirement: str | None = field(default=None, init=False, repr=False)
    def __post_init__(self, _claim):
        if (_claim is not _VIEWER_1D_FACTORY or self.identity is None
                or type(self.request) is not Viewer1DBatchHydrationRequest
                or self.transport_token is not self.request.token
                or type(self.capacity_bytes) is not int or type(self.reserved_bytes) is not int
                or not 0 < self.reserved_bytes <= self.capacity_bytes
                or self.gui_thread_id != self.request.gui_thread_id
                or type(self.worker_thread_id) is not int or self.worker_thread_id <= 0
                or self.worker_thread_id == self.gui_thread_id
                or self.issuer is not self.request.admitted_provider_identity):
            raise TypeError("viewer 1-D budget receipt is malformed")
    @property
    def state(self):
        return (Viewer1DReadBudgetState.ACTIVE if self._retirement is None
                else Viewer1DReadBudgetState.RETIRED)
    @property
    def retirement_reason(self): return self._retirement


def _mint_viewer_1d_budget_receipt(request, capacity, reserved, worker, issuer):
    return Viewer1DReadBudgetReceipt(object(), request, request.token, capacity,
        reserved, request.gui_thread_id, worker, issuer, _VIEWER_1D_FACTORY)


def _retire_viewer_1d_budget_receipt(receipt, reason):
    if (type(receipt) is not Viewer1DReadBudgetReceipt
            or receipt.state is not Viewer1DReadBudgetState.ACTIVE
            or reason not in {"transferred", "abandoned"}):
        raise RuntimeError("viewer 1-D receipt retirement is not current")
    object.__setattr__(receipt, "_retirement", reason)


@dataclass(frozen=True, slots=True)
class Prepared1DBatchCommit:
    request: Viewer1DBatchHydrationRequest; retired_receipt_identity: object
    identity: object; batch_identity: str; transfer: object
    _claim: InitVar[object] = None
    def __post_init__(self, _claim):
        if (_claim is not _VIEWER_1D_FACTORY
                or type(self.request) is not Viewer1DBatchHydrationRequest
                or self.retired_receipt_identity is None or self.identity is None
                or not _viewer_sha256_text(self.batch_identity) or self.transfer is None):
            raise TypeError("prepared viewer 1-D batch is malformed")


def _new_prepared_viewer_1d_commit(request, retired, identity, batch, transfer):
    current = transfer.inspect()
    if (getattr(current, "prepared_identity", None) is not identity
            or getattr(current, "retired_receipt_identity", None) is not retired
            or getattr(current, "batch_identity", None) != batch
            or getattr(current, "request_identity", None) is not request
            or getattr(current, "transport_token_identity", None) is not request.token
            or getattr(current, "issuer_transport_identity", None)
                is not request.admitted_provider_identity
            or getattr(current, "port_identity", None) is not request.port
            or transfer.request_identity is not request
            or transfer.active_receipt_identity is not retired
            or transfer.transport_token_identity is not request.token
            or transfer.issuer_transport_identity is not request.admitted_provider_identity
            or transfer.port_identity is not request.port):
        raise RuntimeError("prepared viewer 1-D custody is not current")
    return Prepared1DBatchCommit(request, retired, identity, batch, transfer,
                                 _VIEWER_1D_FACTORY)


@dataclass(frozen=True, slots=True)
class Viewer1DCommitReceipt:
    identity: object; transfer_identity: object; request_identity: object
    prepared_identity: object; transport_token_identity: object; port_identity: object
    owner_identity: object; batch_identity: str; decision: str
    _claim: InitVar[object] = None
    def __post_init__(self, _claim):
        if _claim is not _VIEWER_1D_FACTORY: raise TypeError("foreign viewer 1-D receipt")


def _new_viewer_1d_commit_receipt(transfer, request, prepared, owner):
    if (prepared.request is not request or prepared.transfer is not transfer
            or transfer.request_identity is not request
            or transfer.transport_token_identity is not request.token
            or transfer.port_identity is not request.port
            or transfer.active_receipt_identity is not prepared.retired_receipt_identity):
        raise RuntimeError("viewer 1-D commit receipt lineage is foreign")
    return Viewer1DCommitReceipt(object(), transfer.identity, request,
        prepared.identity, request.token, request.port, owner,
        prepared.batch_identity, "OWNER_OWNS", _VIEWER_1D_FACTORY)


@dataclass(frozen=True, slots=True)
class Viewer1DCleanupPendingNotice:
    identity: object; request: Viewer1DBatchHydrationRequest; transport_token: HydrationToken
    owner_identity: object; owner_request_claim: object; commit_gate_identity: object
    admission_generation: int; admitted_provider_identity: object; diagnostic: str | None
    issuer_transport_identity: object; issuer_claim: object; _claim: InitVar[object] = None
    _acknowledgement: object = field(default=None, init=False, repr=False)
    def __post_init__(self, _claim):
        if (_claim is not _VIEWER_1D_FACTORY or self.identity is None
                or type(self.request) is not Viewer1DBatchHydrationRequest
                or self.transport_token is not self.request.token
                or self.owner_identity is not self.request.owner_identity
                or self.owner_request_claim is not self.request.owner_request_claim
                or self.commit_gate_identity is not self.request.commit_gate
                or self.admission_generation != self.request.generation
                or self.admitted_provider_identity is not self.request.admitted_provider_identity
                or self.issuer_transport_identity is not self.admitted_provider_identity
                or self.diagnostic is not None and (type(self.diagnostic) is not str
                    or len(self.diagnostic.encode("utf8")) > 256)):
            raise TypeError("foreign cleanup notice")
    @property
    def acknowledgement(self):
        value = self._acknowledgement
        return value if (type(value) is _Viewer1DCleanupPendingAcknowledgement
            and value._claim is _VIEWER_1D_FACTORY and value.notice is self
            and value.owner_identity is self.owner_identity
            and value.owner_request_claim is self.owner_request_claim
            and value.port_identity is self.request.port) else None


def _new_viewer_1d_cleanup_notice(request, diagnostic, issuer, claim):
    if (issuer is not request.admitted_provider_identity or claim is None
            or claim is not getattr(issuer, "_viewer_1d_claim", None)):
        raise TypeError("foreign cleanup notice issuer")
    return Viewer1DCleanupPendingNotice(object(), request, request.token,
        request.owner_identity, request.owner_request_claim, request.commit_gate,
        request.generation, request.admitted_provider_identity, diagnostic,
        issuer, claim, _VIEWER_1D_FACTORY)


@dataclass(frozen=True, slots=True)
class _Viewer1DCleanupPendingAcknowledgement:
    identity: object
    notice: Viewer1DCleanupPendingNotice
    owner_identity: object
    owner_request_claim: object
    port_identity: object
    owner_generation: int
    _claim: object


def acknowledge_viewer_1d_cleanup_pending(notice, *, port, owner_identity,
        owner_request_claim, commit_gate, admitted_provider, owner_generation,
        owner_state):
    request = getattr(notice, "request", None)
    if (type(notice) is not Viewer1DCleanupPendingNotice or port is not request.port
            or owner_identity is not request.owner_identity
            or owner_request_claim is not request.owner_request_claim
            or commit_gate is not request.commit_gate
            or admitted_provider is not request.admitted_provider_identity
            or type(owner_generation) is not int or request.generation > owner_generation
            or owner_state is not Viewer1DState.CLEANUP_PENDING
            or notice.issuer_transport_identity is not request.admitted_provider_identity
            or notice.issuer_claim is not getattr(
                notice.issuer_transport_identity, "_viewer_1d_claim", None)):
        return notice
    if notice._acknowledgement is None:
        object.__setattr__(notice, "_acknowledgement", _Viewer1DCleanupPendingAcknowledgement(
            object(), notice, owner_identity, owner_request_claim, port,
            owner_generation, _VIEWER_1D_FACTORY))
    return notice


@dataclass(frozen=True, slots=True)
class Viewer1DRendererClearRequest:
    context_token: str; display_generation: int; batch_identity: str
    _claim: InitVar[object] = None
    acknowledgement_identity: object = field(default=None, init=False)
    def __post_init__(self, _claim):
        if (_claim is not _VIEWER_1D_FACTORY or type(self.context_token) is not str
                or not self.context_token or type(self.display_generation) is not int
                or self.display_generation <= 0 or not _viewer_sha256_text(self.batch_identity)):
            raise TypeError("bad 1-D clear request")


@dataclass(frozen=True, slots=True)
class Viewer1DRendererClearReceipt:
    identity: object; request: Viewer1DRendererClearRequest; cleared: bool
    mint_claim: object; _claim: InitVar[object] = None
    def __post_init__(self, _claim):
        if (_claim is not _VIEWER_1D_FACTORY or self.mint_claim is not _VIEWER_1D_FACTORY
                or type(self.request) is not Viewer1DRendererClearRequest
                or type(self.cleared) is not bool): raise TypeError("bad 1-D clear receipt")
        if self.cleared:
            if self.request.acknowledgement_identity is not None:
                raise RuntimeError("viewer 1-D clear is already acknowledged")
            object.__setattr__(self.request, "acknowledgement_identity", self.identity)


def _new_viewer_1d_renderer_clear_request(context, generation, batch):
    return Viewer1DRendererClearRequest(context, generation, batch, _VIEWER_1D_FACTORY)


def _new_viewer_1d_renderer_clear_receipt(request, cleared):
    return Viewer1DRendererClearReceipt(object(), request, cleared,
        _VIEWER_1D_FACTORY, _VIEWER_1D_FACTORY)


class Viewer2DReceiptPhase(str, Enum):
    CATALOG_R = "catalog_r"
    FRAME_A = "frame_a"
    FRAME_READY_A = "frame_ready_a"
    CLEANUP_PENDING = "cleanup_pending"
    RELEASED = "released"


class Viewer2DCleanupState(str, Enum):
    CLEANUP_PENDING = "cleanup_pending"
    CLEANED = "cleaned"
    STALE = "stale"


def _viewer_malformed(condition, message):
    if condition: raise TypeError(message)


def _viewer_sha256_text(value):
    return (type(value) is str and len(value) == 64
            and not set(value) - set("0123456789abcdef"))


@dataclass(frozen=True, slots=True)
class Viewer2DContext:
    context_token: str
    generation: int
    original_path: str
    commit_gate: Viewer2DCommitGate
    state: Viewer2DState = Viewer2DState.EMPTY

    def __post_init__(self):
        _viewer_malformed(type(self.context_token) is not str or not self.context_token
            or type(self.generation) is not int or self.generation < 0
            or type(self.original_path) is not str or not self.original_path
            or len(os.fsencode(self.original_path)) > 4096
            or type(self.commit_gate) is not Viewer2DCommitGate
            or type(self.state) is not Viewer2DState,
            "viewer context values are malformed")

    @property
    def kind(self):
        return ContextKind.VIEWER_2D


@dataclass(frozen=True, slots=True)
class Viewer2DReadBudgetReceipt:
    identity: object
    capacity: int
    reserved: int
    phase: Viewer2DReceiptPhase
    request_token: HydrationToken | None = None

    def __post_init__(self):
        from xrd_tools.io.viewer_2d import CATALOG_RESERVATION, viewer_2d_memory_ledger
        budget = viewer_2d_memory_ledger(1, 1).budget
        released = self.phase is Viewer2DReceiptPhase.RELEASED
        _viewer_malformed(self.identity is None or type(self.capacity) is not int
            or self.capacity != budget or type(self.reserved) is not int
            or type(self.phase) is not Viewer2DReceiptPhase
            or released and self.reserved != 0
            or not released and not 0 < self.reserved <= self.capacity
            or self.phase is Viewer2DReceiptPhase.CATALOG_R
            and self.reserved != CATALOG_RESERVATION
            or self.request_token is not None and type(self.request_token) is not HydrationToken,
            "viewer budget receipt is malformed")


@dataclass(frozen=True, slots=True)
class Viewer2DReadActivation:
    token: HydrationToken
    receipt: Viewer2DReadBudgetReceipt | None
    phase: Viewer2DReceiptPhase | None
    accepted: bool = True
    diagnostic: str = ""

    def __post_init__(self):
        _viewer_malformed(type(self.token) is not HydrationToken
            or type(self.accepted) is not bool or type(self.diagnostic) is not str,
            "viewer activation identity is malformed")
        if not self.accepted:
            if self.receipt is not None or self.phase is not None or not self.diagnostic:
                raise ValueError("refused viewer activation must be inert and diagnostic")
            return
        if type(self.receipt) is Viewer2DReadBudgetReceipt:
            self.receipt.__post_init__()
        if (type(self.receipt) is not Viewer2DReadBudgetReceipt
                or self.phase not in (Viewer2DReceiptPhase.CATALOG_R, Viewer2DReceiptPhase.FRAME_A)
                or self.receipt.phase is not self.phase
                or self.receipt.request_token is not self.token):
            raise ValueError("viewer activation phase is not readable")

    @property
    def receipt_identity(self):
        return None if self.receipt is None else self.receipt.identity


@runtime_checkable
class TwoDViewerCommitPort(Protocol):
    def activate(self, request): ...
    def dispose(self, disposal): ...
    def commit(self, prepared): ...
    def complete(self, completion): ...


def _viewer_request_identity(generation, gate, port, read_key, token, label):
    if (type(generation) is not int or generation < 0
            or type(gate) is not Viewer2DCommitGate
            or not isinstance(port, TwoDViewerCommitPort)
            or type(read_key) is not HydrationReadKey
            or type(token) is not HydrationToken
            or token.read_key != read_key
            or token.presentation_generation != generation
            or read_key.scope.scan_key != "viewer-2d"
            or read_key.scope.source != "viewer-2d"
            or read_key.artifact_identity != "viewer-2d"
            or read_key.frame_identity != label
            or read_key.purpose is not HydrationPurpose.PREVIEW
            or read_key.scope.epoch != gate.epoch):
        raise ValueError("viewer request identity is inconsistent")


class _ViewerRequest:
    @property
    def scope(self): return self.read_key.scope

    @property
    def enqueueable(self): return not self.commit_gate.cancelled


def _viewer_policy(value):
    from xrd_tools.io.viewer_2d import Viewer2DFormatPolicy
    if type(value) is not Viewer2DFormatPolicy:
        raise TypeError("viewer request requires an exact frozen format policy")
    return value


@dataclass(frozen=True, slots=True)
class Viewer2DCatalogHydrationRequest(_ViewerRequest):
    path: str
    policy: object
    generation: int
    commit_gate: Viewer2DCommitGate
    port: TwoDViewerCommitPort
    read_key: HydrationReadKey
    token: HydrationToken

    def __post_init__(self):
        if type(self.path) is not str or not self.path or len(os.fsencode(self.path)) > 4096:
            raise TypeError("viewer catalog path must be an exact bounded string")
        _viewer_policy(self.policy)
        _viewer_request_identity(self.generation, self.commit_gate, self.port,
                                 self.read_key, self.token, "catalog")


@dataclass(frozen=True, slots=True)
class Viewer2DFrameHydrationRequest(_ViewerRequest):
    label: int
    generation: int
    commit_gate: Viewer2DCommitGate
    port: TwoDViewerCommitPort
    catalog: object
    receipt_identity: object
    policy: object
    read_key: HydrationReadKey
    token: HydrationToken

    def __post_init__(self):
        from xrd_tools.io.viewer_2d import Viewer2DArtifactCatalog
        labels = getattr(self.catalog, "frame_labels", None)
        identity = getattr(self.catalog, "catalog_identity", None)
        if type(self.catalog) is Viewer2DArtifactCatalog:
            self.catalog.__post_init__()
        if (type(self.label) is not int or self.receipt_identity is None
                or type(self.catalog) is not Viewer2DArtifactCatalog
                or type(labels) is not tuple or self.label not in labels
                or type(identity) is not str or not identity
                or _viewer_policy(self.policy).identity != getattr(self.catalog, "policy_identity", None)):
            raise TypeError("viewer frame request values are malformed")
        _viewer_request_identity(self.generation, self.commit_gate, self.port,
                                 self.read_key, self.token, self.label)


@dataclass(frozen=True, slots=True)
class Prepared2DCatalogCommit:
    request: Viewer2DCatalogHydrationRequest
    activation: Viewer2DReadActivation
    catalog: object

    def __post_init__(self):
        from xrd_tools.io.viewer_2d import (
            CATALOG_RESERVATION, Viewer2DArtifactCatalog, viewer_2d_memory_ledger,
        )
        if type(self.request) is Viewer2DCatalogHydrationRequest:
            self.request.__post_init__()
        if type(self.activation) is Viewer2DReadActivation:
            self.activation.__post_init__()
        if type(self.catalog) is Viewer2DArtifactCatalog:
            self.catalog.__post_init__()
        receipt = getattr(self.activation, "receipt", None)
        _viewer_malformed(type(self.request) is not Viewer2DCatalogHydrationRequest
            or type(self.activation) is not Viewer2DReadActivation
            or not self.activation.accepted
            or self.activation.token is not self.request.token
            or self.activation.phase is not Viewer2DReceiptPhase.CATALOG_R
            or receipt.capacity != viewer_2d_memory_ledger(1, 1).budget
            or receipt.reserved != CATALOG_RESERVATION
            or type(self.catalog) is not Viewer2DArtifactCatalog
            or self.request.policy.identity != self.catalog.policy_identity
            or os.path.realpath(os.path.expanduser(self.request.path)) !=
                self.catalog.canonical_path,
            "prepared viewer catalog is inconsistent")


@dataclass(frozen=True, slots=True)
class Prepared2DFrameCommit:
    request: Viewer2DFrameHydrationRequest
    activation: Viewer2DReadActivation
    frame: object
    receipt_identity: object

    def __post_init__(self):
        from xrd_tools.io.viewer_2d import (
            Viewer2DFrame, _validate_frame_against_catalog,
            viewer_2d_selected_ledger,
        )
        if type(self.request) is Viewer2DFrameHydrationRequest:
            self.request.__post_init__()
        if type(self.activation) is Viewer2DReadActivation:
            self.activation.__post_init__()
        if type(self.frame) is Viewer2DFrame:
            self.frame.__post_init__()
        receipt = getattr(self.activation, "receipt", None)
        ledger = (viewer_2d_selected_ledger(self.request.catalog, self.request.label)
                  if type(self.request) is Viewer2DFrameHydrationRequest else None)
        _viewer_malformed(type(self.request) is not Viewer2DFrameHydrationRequest
            or type(self.activation) is not Viewer2DReadActivation
            or not self.activation.accepted
            or self.activation.token is not self.request.token
            or self.activation.phase is not Viewer2DReceiptPhase.FRAME_A
            or self.activation.receipt_identity is not self.request.receipt_identity
            or self.receipt_identity is not self.request.receipt_identity
            or receipt.capacity != ledger.budget or receipt.reserved != ledger.admission
            or type(self.frame) is not Viewer2DFrame
            or self.frame.catalog_identity !=
                getattr(self.request.catalog, "catalog_identity", None)
            or getattr(self.frame, "label", None) != self.request.label,
            "prepared viewer frame is inconsistent")
        _validate_frame_against_catalog(self.request.catalog, self.request.label, self.frame)


class Viewer2DDisposalToken:
    __slots__ = ()


def _viewer_disposal_expected(request):
    from xrd_tools.io.viewer_2d import (
        CATALOG_RESERVATION,
        viewer_2d_memory_ledger,
        viewer_2d_selected_ledger,
    )
    if type(request) is Viewer2DCatalogHydrationRequest:
        return (Viewer2DReceiptPhase.CATALOG_R,
                viewer_2d_memory_ledger(1, 1).budget,
                CATALOG_RESERVATION)
    ledger = viewer_2d_selected_ledger(request.catalog, request.label)
    return Viewer2DReceiptPhase.FRAME_A, ledger.budget, ledger.admission


def _viewer_quarantine_readable(request, activation):
    receipt = getattr(activation, "receipt", None)
    phase = getattr(activation, "phase", None)
    return (type(activation) is Viewer2DReadActivation
            and getattr(activation, "accepted", None) is True
            and type(getattr(activation, "diagnostic", None)) is str
            and getattr(activation, "token", None) is request.token
            and type(receipt) is Viewer2DReadBudgetReceipt
            and getattr(receipt, "request_token", None) is request.token
            and (phase is Viewer2DReceiptPhase.CATALOG_R
                 or phase is Viewer2DReceiptPhase.FRAME_A)
            and getattr(receipt, "phase", None) is phase
            and getattr(receipt, "identity", None) is not None
            and type(getattr(receipt, "capacity", None)) is int
            and receipt.capacity >= 0
            and type(getattr(receipt, "reserved", None)) is int
            and receipt.reserved >= 0
            and (type(request) is not Viewer2DFrameHydrationRequest
                 or receipt.identity is request.receipt_identity))


@dataclass(frozen=True, slots=True)
class Viewer2DDisposal:
    request: object
    activation: Viewer2DReadActivation
    prepared: object | None
    token: Viewer2DDisposalToken

    def __post_init__(self):
        request_type = type(self.request)
        _viewer_malformed(request_type not in (Viewer2DCatalogHydrationRequest,
                Viewer2DFrameHydrationRequest)
            or type(self.token) is not Viewer2DDisposalToken,
            "viewer disposal identity is inconsistent")
        self.request.__post_init__()
        readable = _viewer_quarantine_readable(self.request, self.activation)
        expected_phase, expected_capacity, expected_reserved = \
            _viewer_disposal_expected(self.request)
        normal = (readable and self.activation.phase is expected_phase
                  and self.activation.receipt.capacity == expected_capacity
                  and self.activation.receipt.reserved == expected_reserved)
        if normal:
            self.activation.__post_init__()
            if self.prepared is None:
                return
            prepared_type = (Prepared2DCatalogCommit
                if request_type is Viewer2DCatalogHydrationRequest
                else Prepared2DFrameCommit)
            _viewer_malformed(type(self.prepared) is not prepared_type
                or self.prepared.request is not self.request
                or self.prepared.activation is not self.activation,
                "viewer disposal identity is inconsistent")
            self.prepared.__post_init__()
            return
        _viewer_malformed(self.prepared is not None or not readable,
                          "viewer disposal identity is inconsistent")


@dataclass(frozen=True, slots=True)
class Viewer2DCleanupReceipt:
    token: Viewer2DDisposalToken
    state: Viewer2DCleanupState

    def __post_init__(self):
        _viewer_malformed(type(self.token) is not Viewer2DDisposalToken
            or type(self.state) is not Viewer2DCleanupState, "viewer cleanup receipt is malformed")


@dataclass(frozen=True, slots=True)
class Viewer2DRendererClearRequest:
    context_token: str
    generation: int
    catalog_identity: str
    label: int | None

    def __post_init__(self):
        _viewer_malformed(type(self.context_token) is not str or not self.context_token
            or type(self.generation) is not int or self.generation < 0
            or not _viewer_sha256_text(self.catalog_identity)
            or self.label is not None and (type(self.label) is not int or self.label < 0),
            "viewer renderer-clear request is malformed")


@dataclass(frozen=True, slots=True)
class Viewer2DRendererClearReceipt:
    request: Viewer2DRendererClearRequest
    cleared: bool

    def __post_init__(self):
        if type(self.request) is Viewer2DRendererClearRequest:
            self.request.__post_init__()
        _viewer_malformed(type(self.request) is not Viewer2DRendererClearRequest
            or type(self.cleared) is not bool, "viewer renderer-clear receipt is malformed")


@dataclass(frozen=True, slots=True)
class DisplaySelection:
    """Which context, and at which display generation, the panels render.

    Frame identity is deliberately NOT here: the generation-stamped render pin
    remains the sole frame-selection owner, and a frame field would make this a
    second one.  Built only by :meth:`for_context`, which the GUI selection
    owner calls AFTER it has stamped the new display generation.
    """

    kind: ContextKind
    #: §12.6 B.4 — the context's OWN projection, stored whole.  The parent
    #: copied token, key and source into separate fields, which let the
    #: selection independently choose the admitted source over the current one.
    owner: "HydrationOwner"
    display_generation: int

    @property
    def context_token(self) -> str:
        return self.owner.context_token

    @property
    def scan_key(self) -> str:
        return self.owner.scan_key

    @property
    def source_path(self) -> str:
        """The CURRENT source this selection names (compatibility accessor)."""
        return self.owner.source

    @classmethod
    def for_context(cls, context, display_generation: int) -> "DisplaySelection":
        """Stamp a selection naming *context* at an ALREADY-bumped generation."""
        return cls(
            kind=context.kind,
            owner=context.hydration_owner,
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
    #: The live key, source, exact display scan and commit epoch.  SINGLE
    #: WRITER: :meth:`rescope_to`, which publishes a complete immutable scope
    #: by replacing this one reference.
    _current_scope: object = field(init=False, default=_UNSET, repr=False)
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
        # §10.2: the SOLE commit authority a request captures.  Replaceable, it
        # would let a later assignment silently orphan the cancellation and
        # epoch that in-flight requests are already qualified against.
        "commit_gate",
    })

    def __post_init__(self):
        if self._current_scope is _UNSET:
            self._current_scope = CurrentDisplayScope(
                scan_key=str(self.run_scan_key or ""),
                source=str(self.source_path or ""),
                display_scan=self.scan,
                commit_epoch=self.commit_gate.epoch,
            )

    @property
    def current_scope(self) -> CurrentDisplayScope:
        """The one immutable scope reference currently published."""
        return self._current_scope

    @property
    def current_scan_key(self) -> str:
        return self._current_scope.scan_key

    @property
    def current_source(self) -> str:
        return self._current_scope.source

    @property
    def current_display_scan(self):
        return self._current_scope.display_scan

    @property
    def commit_epoch(self) -> int:
        return self._current_scope.commit_epoch

    @property
    def kind(self) -> ContextKind:
        return ContextKind.ACQUISITION

    @property
    def scan_key(self) -> str:
        """The key the display is currently scoped to within this run."""
        scope = self._current_scope
        return scope.scan_key or self.run_scan_key

    @property
    def source(self) -> str:
        """The source the display is currently scoped to within this run."""
        scope = self._current_scope
        return scope.source or self.source_path

    @property
    def admitted_source(self) -> str:
        """The immutable ACCEPTED source, kept for provenance (§12.6 B.5).

        Deliberately a different name from :attr:`source`: reading "the source"
        and getting the admitted root after a member transition is exactly the
        mixed identity §12.2 found, so the two are no longer interchangeable.
        """
        return self.source_path

    @property
    def hydration_owner(self) -> "HydrationOwner":
        """THE production mint (§12.6 B.1) — one owner, from CURRENT identity."""
        scope = self._current_scope
        return HydrationOwner(
            self.context_token,
            scope.scan_key or self.run_scan_key,
            scope.source or self.source_path,
            scope.commit_epoch,
        )

    def rescope_to(self, scan_key, source, display_scan=_UNSET) -> None:
        """Replace the sub-scan identity with one COMPLETE selection (§12.6 C).

        Key, source and display scan are validated BEFORE any is written, then
        advance under the same commit epoch.  Omitting ``display_scan`` keeps
        the existing scan for genuine within-scan callers; a caller that owns
        a new artifact passes its exact newly constructed scan.

        A genuine same-source rescope passes the current source explicitly —
        see :meth:`rescope_within_source`.
        """
        scan_key = str(scan_key or "")
        source = str(source or "")
        scope = self._current_scope
        display_scan = (
            scope.display_scan if display_scan is _UNSET else display_scan
        )
        if not scan_key or not source or display_scan is None:
            raise DisplayContextError(
                "a sub-scan boundary needs a complete "
                "(scan key, source, display scan) selection")
        epoch = self.commit_gate.reserve_advance()
        self._current_scope = CurrentDisplayScope(
            scan_key=scan_key,
            source=source,
            display_scan=display_scan,
            commit_epoch=epoch,
        )
        committed_epoch = self.commit_gate.advance()
        if committed_epoch != epoch:
            raise DisplayContextError("commit-gate epoch reservation changed")

    def rescope_within_source(self, scan_key) -> None:
        """A boundary that genuinely keeps the current source, said out loud."""
        scope = self._current_scope
        self.rescope_to(scan_key, scope.source, scope.display_scan)

    def adopt_record_store(self, store) -> None:
        """Adopt the streaming session's per-run record store."""
        self.record_store = store

    def display_bindings(self) -> DisplayBindings:
        """The complete display surface the selection owner swaps IN."""
        scope = self._current_scope
        return DisplayBindings(
            scan=scope.display_scan,
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

    def retire(self) -> None:
        """THE terminal action for an acquisition context (§10.2).

        Idempotent, and deliberately the mirror of ``BrowseContext.invalidate``:
        it withdraws commit authority and nothing else.  Dropping the owner
        reference without this left an in-flight request holding the gate and
        the exact store tuple it had captured, so its old epoch could still
        enter after run-end release and insert into the retired run's store.

        It records no second lifecycle authority — ``finalization_state``
        remains the one finalization fact — and it does not touch the run's
        last rendered values: what is cancelled is late COMMIT authority, not
        the accepted final display.
        """
        self.commit_gate.cancel()


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
    #: The whole-scan normalization aggregate folded by the ONE loader pass
    #: (E6-NORM-N1): an object-typed, write-once construction value.  This
    #: module never imports the value's own module or an array library; the
    #: consumer boundary validates the exact type.
    norm_aggregate: object = None
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
        # E6-NORM-N1: the aggregate travels with the context, write-once.
        "norm_aggregate",
        # §10.2: one gate, created with the context and mutated only through
        # its own methods — never replaceable by assignment.
        "commit_gate",
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

    @property
    def source(self) -> str:
        """One resolution rule with the acquisition owner: the live source."""
        return self.requested_path

    @property
    def admitted_source(self) -> str:
        return self.requested_path

    @property
    def hydration_owner(self) -> "HydrationOwner":
        """THE production mint for a browse (§12.6 B.1)."""
        return HydrationOwner(self.context_token, self.scan_key, self.source,
                              self.commit_epoch)

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
