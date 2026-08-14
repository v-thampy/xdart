"""Canonical headless contracts, custody, and runtime for standalone 1-D reads."""
from __future__ import annotations

import os
import threading
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from xrd_tools.io.viewer_1d import (
    Viewer1DBatchManifest, Viewer1DFormatPolicy, Viewer1DSourceInspection,
    close_viewer_1d_sources, decode_viewer_1d_sources,
    inspect_viewer_1d_sources, viewer_1d_inspection_is_open,
    viewer_1d_inspection_ledger,
)
from xrd_tools.session.hydration import (
    HydrationPurpose, HydrationReadKey, HydrationToken,
)
from xrd_tools.session.light_1d_retention import (
    Light1DBufferLayout, Light1DCleanupHooks, Light1DCleanupPending,
    Light1DFundingMode, Light1DLayout, Light1DModeData, Light1DModeLayout,
    Light1DHydrationToken, Light1DRecord, Light1DReleaseReceipt,
    Light1DRetentionLease, SessionResourceAuthority,
    acquire_light_1d_retention,
)

VIEWER_1D_R = 2 * 1024**2 + 256 * 4608 + 256
VIEWER_1D_FALLBACK_B = 884_736_000


def _malformed(condition, message):
    if condition: raise TypeError(message)


def _sha256_text(value):
    try: return type(value) is str and len(value) == 64 and len(bytes.fromhex(value)) == 32
    except ValueError: return False


def _diag(error):
    try: raw = str(error) or type(error).__name__
    except BaseException: raw = type(error).__name__
    while len(raw.encode("utf8")) > 256: raw = raw[:-1]
    return raw


def _is_control(error):
    return isinstance(error, MemoryError) or (isinstance(error, BaseException)
                                               and not isinstance(error, Exception))


class Viewer1DCommitGate:
    """Standalone linearization gate for one headless 1-D owner."""
    __slots__ = ("_lock", "_epoch", "_cancelled", "_reserved_epoch")
    def __init__(self):
        self._lock, self._epoch = threading.Lock(), 1
        self._cancelled, self._reserved_epoch = False, 0
    @property
    def epoch(self): return self._epoch
    @property
    def cancelled(self): return self._cancelled
    def enter(self, epoch):
        self._lock.acquire()
        if self._cancelled or epoch != self._epoch:
            self._lock.release(); return False
        return True
    def leave(self):
        try: self._lock.release()
        except RuntimeError: pass
    def advance(self):
        with self._lock:
            if self._reserved_epoch:
                epoch, self._reserved_epoch = self._reserved_epoch, 0
                return epoch
            self._epoch += 1; return self._epoch
    def reserve_advance(self):
        with self._lock:
            if self._reserved_epoch: raise RuntimeError("a commit-gate advance is in progress")
            self._epoch += 1; self._reserved_epoch = self._epoch
            return self._epoch
    def cancel(self):
        with self._lock:
            self._cancelled, self._reserved_epoch = True, 0
            self._epoch += 1


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
        scope = self.read_key.scope if type(self.read_key) is HydrationReadKey else None
        _malformed(type(self.paths) is not tuple or not 1 <= len(self.paths) <= 256
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


def viewer_1d_request_is_canonical(request, provider):
    if type(request) is not Viewer1DBatchHydrationRequest: return False
    try: request.__post_init__()
    except (AttributeError, TypeError, ValueError, RuntimeError): return False
    return request.admitted_provider_identity is provider


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
                or not _sha256_text(self.batch_identity) or self.transfer is None):
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
                or self.display_generation <= 0 or not _sha256_text(self.batch_identity)):
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


class Viewer1DReadFailureStage(str, Enum):
    DESCRIPTOR = "descriptor"
    PASS_ONE = "pass_one"
    AUTHORITY = "authority"
    LEASE = "lease"
    TOKEN = "token"
    OPERATION = "operation"
    PASS_TWO = "pass_two"


@dataclass(frozen=True, slots=True)
class Viewer1DReadFailure:
    stage: Viewer1DReadFailureStage
    diagnostic: str
    disposal: "Viewer1DDisposal"


@dataclass(frozen=True, slots=True)
class Viewer1DBatch:
    authority: SessionResourceAuthority
    lease: object
    row_identity: object
    manifest: Viewer1DBatchManifest
    batch_identity: str


@dataclass(frozen=True, slots=True)
class _BuildingCustody:
    graph: object
    claim: object


@dataclass(frozen=True, slots=True)
class _BatchCustody:
    batch: Viewer1DBatch
    batch_identity: str
    prepared_identity: object
    retired_receipt_identity: object
    request_identity: object
    transport_token_identity: object
    issuer_transport_identity: object
    port_identity: object
    claim: object


@dataclass(frozen=True, slots=True)
class _OwnerAdoption:
    holder: object
    receipt: object
    claim: object


@dataclass(frozen=True, slots=True)
class _DisposalCustody:
    graph: object
    disposal: object
    claim: object


@dataclass(frozen=True, slots=True)
class _Released:
    identity: object
    claim: object


class _BuildingGraph:
    __slots__ = ("inspection", "native_roots", "authority", "lease", "token",
                 "row_identity", "batch", "inner_token")
    def __init__(self):
        self.inspection, self.authority, self.lease, self.token = None, None, None, None
        self.native_roots, self.row_identity, self.batch, self.inner_token = [], None, None, None
    @property
    def manifest(self): return None if self.inspection is None else self.inspection.manifest
    def close_streams(self):
        if self.inspection is not None: close_viewer_1d_sources(self.inspection)


_FACTORY = object()


class Viewer1DAdoptionTransfer:
    """The one mutable cleanup-ownership cell; decision identity is authority."""
    __slots__ = ("identity", "request_identity", "transport_token_identity",
                 "active_receipt_identity", "issuer_transport_identity",
                 "port_identity", "_decision")
    def __init_subclass__(cls, **kwargs): raise TypeError("viewer 1-D transfer is not subclassable")
    def __init__(self, request, receipt, issuer, *, _claim=None):
        if _claim is not _FACTORY: raise TypeError("foreign viewer 1-D transfer")
        self.identity, self.request_identity = object(), request
        self.transport_token_identity, self.active_receipt_identity = request.token, receipt.identity
        self.issuer_transport_identity, self.port_identity = issuer, request.port
        self._decision = _BuildingCustody(_BuildingGraph(), _FACTORY)
    def inspect(self): return self._decision
    @property
    def owner_receipt(self):
        return self._decision.receipt if type(self._decision) is _OwnerAdoption else None
    def install_batch(self, batch, prepared_identity, retired_receipt_identity):
        current = self._decision
        if (type(current) is not _BuildingCustody or current.claim is not _FACTORY
                or current.graph.batch is not batch or prepared_identity is None
                or retired_receipt_identity is not self.active_receipt_identity):
            raise RuntimeError("viewer 1-D batch transfer is not current")
        decision = _BatchCustody(batch, batch.batch_identity, prepared_identity,
            retired_receipt_identity, self.request_identity, self.transport_token_identity,
            self.issuer_transport_identity, self.port_identity, _FACTORY)
        self._decision = decision; return decision
    def disposal(self, reason):
        current = self._decision
        if (type(current) is _DisposalCustody and current.claim is _FACTORY
                and current.disposal.transfer is self): return current.disposal
        if type(current) not in (_BuildingCustody, _BatchCustody) or current.claim is not _FACTORY:
            raise RuntimeError("viewer 1-D cleanup is not transport-owned")
        disposal = Viewer1DDisposal(self, str(reason)[:256], _claim=_FACTORY)
        graph = current.graph if type(current) is _BuildingCustody else current.batch
        self._decision = _DisposalCustody(graph, disposal, _FACTORY); return disposal
    def owner_holder(self, receipt):
        current = self._decision
        if (type(current) is not _OwnerAdoption or current.claim is not _FACTORY
                or current.receipt is not receipt):
            raise RuntimeError("viewer 1-D adoption receipt is stale")
        return current.holder


class Viewer1DDisposal:
    __slots__ = ("transfer", "reason", "_inner_token", "_hooks", "_no_lease_step")
    def __init__(self, transfer, reason, *, _claim=None):
        if _claim is not _FACTORY: raise TypeError("foreign viewer 1-D disposal")
        self.transfer, self.reason = transfer, reason
        self._inner_token = self._hooks = None; self._no_lease_step = 0
    def _cleanup(self, action):
        current = self.transfer.inspect()
        graph = (current.graph if type(current) is _DisposalCustody
            and current.claim is _FACTORY and current.disposal is self else None)
        if type(graph) is not _BuildingGraph: return
        if action == "cancel" and graph.token is not None:
            graph.lease.abandon_hydration(graph.token); graph.token = None
        elif action == "drain": graph.close_streams()
        elif action == "detach": graph.native_roots.clear()
        elif action == "verify" and (graph.native_roots or graph.token
                or graph.inspection is not None and viewer_1d_inspection_is_open(graph.inspection)):
            raise RuntimeError("viewer 1-D native roots remain")
    def _act(self, retry):
        current = self.transfer.inspect()
        if type(current) is _Released and current.claim is _FACTORY:
            self._hooks = self._inner_token = None; return True
        if (type(current) is not _DisposalCustody or current.claim is not _FACTORY
                or current.disposal is not self): return False
        graph, lease = current.graph, current.graph.lease
        building = graph if type(graph) is _BuildingGraph else None
        if lease is None:
            if building is None: return False
            steps = (building.close_streams, building.native_roots.clear,
                     lambda: _verify_unfunded_building(building),
                     lambda: setattr(building, "authority", None))
            while self._no_lease_step < len(steps):
                steps[self._no_lease_step](); self._no_lease_step += 1
            if self.transfer.inspect() is not current: return False
            self.transfer._decision = _Released(object(), _FACTORY)
            self._hooks = self._inner_token = None; return True
        if self._hooks is None:
            self._hooks = Light1DCleanupHooks(*(lambda action=action: self._cleanup(action)
                for action in ("cancel", "drain", "detach", "verify")))
        try:
            receipt = (lease.retry_cleanup(self._inner_token, hooks=self._hooks)
                       if retry and self._inner_token is not None else
                       lease.release(reason=self.reason, hooks=self._hooks))
        except Light1DCleanupPending as error:
            self._inner_token = error.token
            if _is_control(error.__cause__): raise _Viewer1DDisposalControl(error.__cause__) from None
            return False
        if (type(receipt) is not Light1DReleaseReceipt
                or receipt.released_bytes != lease.reserved_ndarray_bytes
                or lease.keys() or lease.pending_hydration_count or lease.active_borrow_count
                or lease.authority.snapshot().reservation_count): return False
        if self.transfer.inspect() is not current: return False
        self.transfer._decision = _Released(object(), _FACTORY)
        self._hooks = self._inner_token = None; return True
    def release(self): return self._act(False)
    def retry(self): return self._act(True)


class Viewer1DOwnerCleanupHolder:
    __slots__ = ("transfer", "batch", "borrow", "_inner_token")
    def __init__(self, transfer, batch, borrow, *, _claim=None):
        if _claim is not _FACTORY: raise TypeError("foreign viewer 1-D owner holder")
        self.transfer, self.batch, self.borrow, self._inner_token = transfer, batch, borrow, None
    def release(self, reason):
        current = self.transfer.inspect()
        if type(current) is _Released and current.claim is _FACTORY: return True
        if (type(current) is not _OwnerAdoption or current.claim is not _FACTORY
                or current.holder is not self): return False
        if self.borrow is not None: self.borrow.close(); self.borrow = None
        try:
            receipt = (self.batch.lease.retry_cleanup(self._inner_token)
                       if self._inner_token is not None else
                       self.batch.lease.release(reason=str(reason)[:256]))
        except Light1DCleanupPending as error:
            self._inner_token = error.token
            if _is_control(error.__cause__): raise error.__cause__
            return False
        if (type(receipt) is not Light1DReleaseReceipt
                or receipt.released_bytes != self.batch.lease.reserved_ndarray_bytes
                or self.batch.lease.keys() or self.batch.lease.pending_hydration_count
                or self.batch.lease.active_borrow_count
                or self.batch.authority.snapshot().reservation_count): return False
        if self.transfer.inspect() is not current: return False
        self.transfer._decision = _Released(object(), _FACTORY); return True


class _Viewer1DReaderControl(BaseException):
    def __init__(self, control, disposal): self.control, self.disposal = control, disposal


class _Viewer1DDisposalControl(BaseException):
    def __init__(self, control): self.control = control


@dataclass(frozen=True, slots=True)
class _Viewer1DPassTwoPermit:
    identity: object; operation: object; receipt_identity: object; issuer: object
    capacity_bytes: int; reserved_bytes: int; gui_thread_id: int
    worker_thread_id: int; claim: object


class Viewer1DReadOperation:
    __slots__ = ("request", "receipt_identity", "transfer", "inspection", "facts", "ledger",
                 "issuer", "gui_thread_id", "worker_thread_id", "_permit_identity", "_consumed")
    def __init__(self, request, receipt, transfer, inspection, *, _claim=None):
        if _claim is not _FACTORY: raise TypeError("foreign viewer 1-D operation")
        self.request, self.receipt_identity, self.transfer = request, receipt.identity, transfer
        self.inspection, self.facts = inspection, inspection.facts
        self.ledger = viewer_1d_inspection_ledger(inspection)
        self.issuer = receipt.issuer
        self.gui_thread_id, self.worker_thread_id = receipt.gui_thread_id, receipt.worker_thread_id
        self._permit_identity, self._consumed = None, False
    def complete(self, permit):
        current = self.transfer.inspect()
        if (type(permit) is not _Viewer1DPassTwoPermit or permit.claim is not _FACTORY
                or permit.identity is not self._permit_identity or permit.operation is not self
                or permit.receipt_identity is not self.receipt_identity or self._consumed
                or permit.issuer is not self.issuer or permit.capacity_bytes != self.ledger.B
                or permit.reserved_bytes != self.ledger.R
                or permit.gui_thread_id != self.gui_thread_id
                or permit.worker_thread_id != self.worker_thread_id
                or type(current) is not _BuildingCustody or current.claim is not _FACTORY
                or current.graph.inspection is not self.inspection
                or current.graph.manifest.ledger is not self.ledger):
            return _failure(self.transfer, Viewer1DReadFailureStage.PASS_TWO,
                            "viewer 1-D pass-two permit is stale")
        self._consumed = True; graph = current.graph
        try:
            modes = {}
            for index, values in enumerate(decode_viewer_1d_sources(self.inspection)):
                x, y, sigma = values
                graph.native_roots.extend(value for value in values if value is not None)
                modes[index] = Light1DModeData(x, y, sigma)
            graph.close_streams()
            record = Light1DRecord(graph.row_identity, self.request.generation, 0, modes,
                                   {"batch_identity": graph.manifest.identity})
            graph.lease.complete_hydration(graph.token, record)
            graph.token = None; graph.native_roots.clear()
            batch = Viewer1DBatch(graph.authority, graph.lease, graph.row_identity,
                                  graph.manifest, graph.manifest.identity)
            graph.batch = batch; return batch
        except BaseException as error:
            if _is_control(error):
                raise _Viewer1DReaderControl(error, self.transfer.disposal(_diag(error))) from None
            return _failure(self.transfer, Viewer1DReadFailureStage.PASS_TWO, _diag(error))


def _verify_unfunded_building(graph):
    if (graph.native_roots or graph.token is not None or graph.lease is not None
            or graph.inspection is not None and viewer_1d_inspection_is_open(graph.inspection)
            or graph.authority is not None
            and graph.authority.snapshot().reservation_count != 0):
        raise RuntimeError("viewer 1-D unfunded cleanup is incomplete")


def _funding_matches(graph, ledger, receipt):
    if (graph.manifest is None or graph.manifest.ledger is not ledger
            or not viewer_1d_inspection_is_open(graph.inspection)
            or type(graph.authority) is not SessionResourceAuthority
            or graph.authority.parent_allocation is not None
            or type(graph.lease) is not Light1DRetentionLease
            or graph.lease.authority is not graph.authority
            or type(graph.token) is not Light1DHydrationToken
            or graph.lease.requested_rows != 1 or graph.lease.row_cap != 1
            or graph.lease.reserved_ndarray_bytes != ledger.C
            or graph.lease.pending_hydration_count != 1
            or graph.lease.active_borrow_count != 0 or graph.lease.keys()
            or ledger.B != receipt.capacity_bytes or ledger.R != receipt.reserved_bytes):
        return False
    snapshot = graph.authority.snapshot()
    return (snapshot.capacity_bytes == ledger.A
        and dict(snapshot.committed_bytes) == {"viewer_1d_transient": ledger.T}
        and dict(snapshot.categories) == {"viewer_1d_transient": ledger.T, "light_1d": ledger.C}
        and snapshot.reserved_bytes == ledger.A and snapshot.available_bytes == 0
        and snapshot.reservation_count == 1)


def _failure(transfer, stage, diagnostic):
    return Viewer1DReadFailure(stage, _diag(diagnostic), transfer.disposal(diagnostic))


def viewer_1d_budget():
    try: ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError): ram = 0
    return VIEWER_1D_FALLBACK_B if ram <= 0 else min(1024**3, ram // 20)


def mint_viewer_1d_transfer(request, receipt, issuer):
    return Viewer1DAdoptionTransfer(request, receipt, issuer, _claim=_FACTORY)


def viewer_1d_disposal_is_current(disposal):
    if type(disposal) is not Viewer1DDisposal: return False
    current = disposal.transfer.inspect()
    return (type(current) is _DisposalCustody and current.claim is _FACTORY
            and current.disposal is disposal)


def viewer_1d_transfer_is_released(transfer):
    current = transfer.inspect() if type(transfer) is Viewer1DAdoptionTransfer else None
    return type(current) is _Released and current.claim is _FACTORY


def mint_viewer_1d_pass_two_permit(operation, receipt, issuer):
    current = operation.transfer.inspect() if type(operation) is Viewer1DReadOperation else None
    graph = current.graph if type(current) is _BuildingCustody else None
    if (type(operation) is not Viewer1DReadOperation
            or receipt.state is not Viewer1DReadBudgetState.RETIRED
            or receipt.retirement_reason != "transferred"
            or operation.receipt_identity is not receipt.identity
            or operation.request is not receipt.request or receipt.issuer is not issuer
            or operation.issuer is not issuer or operation.transfer.issuer_transport_identity is not issuer
            or operation.transfer.request_identity is not receipt.request
            or operation.transfer.transport_token_identity is not receipt.transport_token
            or operation.transfer.port_identity is not receipt.request.port
            or operation._permit_identity is not None or graph is None
            or current.claim is not _FACTORY or not _funding_matches(graph, operation.ledger, receipt)):
        raise TypeError("foreign viewer 1-D permit")
    permit = _Viewer1DPassTwoPermit(object(), operation, receipt.identity, issuer,
        receipt.capacity_bytes, receipt.reserved_bytes, receipt.gui_thread_id,
        receipt.worker_thread_id, _FACTORY)
    operation._permit_identity = permit.identity; return permit


def begin_viewer_1d_read(request, receipt, transfer):
    if (type(receipt) is not Viewer1DReadBudgetReceipt
            or receipt.state is not Viewer1DReadBudgetState.ACTIVE
            or receipt.request is not request or receipt.identity is not transfer.active_receipt_identity
            or receipt.capacity_bytes != viewer_1d_budget() or receipt.reserved_bytes != VIEWER_1D_R
            or receipt.worker_thread_id != threading.get_ident()):
        return _failure(transfer, Viewer1DReadFailureStage.DESCRIPTOR, "viewer 1-D receipt mismatch")
    current = transfer.inspect()
    if type(current) is not _BuildingCustody or current.claim is not _FACTORY:
        return _failure(transfer, Viewer1DReadFailureStage.DESCRIPTOR,
                        "viewer 1-D construction custody is stale")
    graph, stage = current.graph, Viewer1DReadFailureStage.PASS_ONE
    try:
        graph.inspection = inspect_viewer_1d_sources(
            request.paths, request.policy, receipt.capacity_bytes, receipt.reserved_bytes)
        ledger = viewer_1d_inspection_ledger(graph.inspection)
        layouts = []
        for index, (points, sigma) in enumerate(zip(ledger.counts, ledger.sigmas)):
            spec = lambda role: Light1DBufferLayout(points, 8, f"{index}:{role}", "<f8")
            layouts.append(Light1DModeLayout(index, spec("x"), spec("y"),
                                             spec("sigma") if sigma else None))
        graph.row_identity = ("viewer-1d", request.generation, graph.manifest.identity)
        stage = Viewer1DReadFailureStage.AUTHORITY
        graph.authority = SessionResourceAuthority(capacity_bytes=ledger.A,
            committed_bytes={"viewer_1d_transient": ledger.T})
        stage = Viewer1DReadFailureStage.LEASE
        graph.lease = acquire_light_1d_retention(graph.authority,
            owner=f"viewer-1d:{id(transfer)}", generation=request.generation,
            layout=Light1DLayout(tuple(layouts), 0), requested_rows=1,
            compatibility_byte_ceiling=ledger.C, gui_thread_id=request.gui_thread_id,
            funding_mode=Light1DFundingMode.HEADROOM)
        stage = Viewer1DReadFailureStage.TOKEN
        graph.token = graph.lease.issue_hydration_token(graph.row_identity)
        if not _funding_matches(graph, ledger, receipt):
            raise RuntimeError("viewer 1-D ledger mismatch")
        stage = Viewer1DReadFailureStage.OPERATION
        return Viewer1DReadOperation(request, receipt, transfer, graph.inspection, _claim=_FACTORY)
    except BaseException as error:
        if _is_control(error):
            raise _Viewer1DReaderControl(error, transfer.disposal(_diag(error))) from None
        return _failure(transfer, stage, _diag(error))


def adopt_prepared_viewer_1d(prepared, *, owner_identity, owner_request_claim,
                             port, owner_generation, commit_gate, owner_state):
    if (type(prepared) is not Prepared1DBatchCommit or prepared.request.port is not port
            or prepared.request.owner_identity is not owner_identity
            or prepared.request.owner_request_claim is not owner_request_claim
            or type(owner_generation) is not int or owner_generation != prepared.request.generation
            or commit_gate is not prepared.request.commit_gate
            or owner_state is not Viewer1DState.LOADING):
        raise TypeError("viewer 1-D owner lineage is foreign")
    transfer, current = prepared.transfer, prepared.transfer.inspect()
    if (type(current) is not _BatchCustody or current.claim is not _FACTORY
            or current.prepared_identity is not prepared.identity
            or current.batch_identity != prepared.batch_identity
            or current.retired_receipt_identity is not prepared.retired_receipt_identity
            or current.request_identity is not prepared.request
            or current.transport_token_identity is not prepared.request.token
            or current.issuer_transport_identity is not prepared.request.admitted_provider_identity
            or current.port_identity is not port or transfer.request_identity is not prepared.request
            or transfer.transport_token_identity is not prepared.request.token
            or transfer.active_receipt_identity is not prepared.retired_receipt_identity
            or transfer.issuer_transport_identity is not prepared.request.admitted_provider_identity
            or transfer.port_identity is not port
            or current.batch.batch_identity != current.batch.manifest.identity):
        raise RuntimeError("viewer 1-D batch custody is not adoptable")
    if not commit_gate.enter(prepared.request.read_key.scope.epoch):
        raise RuntimeError("viewer 1-D commit gate is stale")
    borrow = None
    try:
        borrow = current.batch.lease.borrow(current.batch.row_identity)
        if borrow is None: raise RuntimeError("viewer 1-D row is not resident")
        holder = Viewer1DOwnerCleanupHolder(transfer, current.batch, borrow, _claim=_FACTORY)
        receipt = _new_viewer_1d_commit_receipt(transfer, prepared.request, prepared, owner_identity)
        transfer._decision = _OwnerAdoption(holder, receipt, _FACTORY); return receipt
    except BaseException:
        if type(transfer.inspect()) is _BatchCustody and borrow is not None: borrow.close()
        raise
    finally: commit_gate.leave()


__all__ = [name for name in tuple(globals()) if name.startswith("Viewer1D") or name in {
    "OneDViewerCommitPort", "Prepared1DBatchCommit", "VIEWER_1D_R",
    "acknowledge_viewer_1d_cleanup_pending", "adopt_prepared_viewer_1d",
    "begin_viewer_1d_read", "mint_viewer_1d_pass_two_permit", "mint_viewer_1d_transfer",
    "viewer_1d_budget", "viewer_1d_disposal_is_current",
    "viewer_1d_request_is_canonical", "viewer_1d_transfer_is_released"}]
