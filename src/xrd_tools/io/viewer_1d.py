"""Bounded, same-open reader and single-cell custody for standalone 1-D data."""
from __future__ import annotations

import ast
import ctypes
import hashlib
import os
import stat
import struct
import sys
import threading
import zipfile
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

import numpy as np

from xrd_tools.session.light_1d_retention import (
    Light1DBufferLayout, Light1DCleanupHooks, Light1DCleanupPending,
    Light1DFundingMode, Light1DLayout, Light1DModeData, Light1DModeLayout,
    Light1DHydrationToken, Light1DRecord, Light1DReleaseReceipt,
    Light1DRetentionLease,
    SessionResourceAuthority,
    acquire_light_1d_retention,
)

VIEWER_1D_FD_RESERVE = 32
VIEWER_1D_R = 2 * 1024**2 + 256 * 4608 + 256
VIEWER_1D_FALLBACK_B = 884_736_000
_POLICY_ID = "viewer-1d-v1"
_FORMATS = {".xye": 1, ".csv": 2, ".npy": 3, ".npz": 4}
_DTYPE_CODES = {
    "|b1": 0x01, "|i1": 0x10, "<i2": 0x11, ">i2": 0x12,
    "<i4": 0x13, ">i4": 0x14, "<i8": 0x15, ">i8": 0x16,
    "|u1": 0x20, "<u2": 0x21, ">u2": 0x22, "<u4": 0x23,
    ">u4": 0x24, "<u8": 0x25, ">u8": 0x26, "<f2": 0x30,
    ">f2": 0x31, "<f4": 0x32, ">f4": 0x33, "<f8": 0x34, ">f8": 0x35,
}


@dataclass(frozen=True, slots=True)
class Viewer1DFormatPolicy:
    identity: str = _POLICY_ID
    def __post_init__(self):
        if self.identity != _POLICY_ID: raise ValueError("unsupported viewer 1-D policy")


@dataclass(frozen=True, slots=True)
class Viewer1DMemoryLedger:
    counts: tuple[int, ...]
    sigmas: tuple[bool, ...]
    encoded: tuple[int, ...]
    B: int
    P: int
    C: int
    R: int
    N: int
    T: int
    A: int


@dataclass(frozen=True, slots=True)
class Viewer1DSourceManifest:
    canonical_path: str
    format: str
    points: int
    sigma_present: bool
    x_label: str
    x_unit: str
    y_label: str
    source_sha256: bytes
    role_sha256: bytes
    scalar_record: bytes


@dataclass(frozen=True, slots=True)
class Viewer1DBatchManifest:
    identity: str
    policy_identity: str
    sources: tuple[Viewer1DSourceManifest, ...]
    ledger: Viewer1DMemoryLedger


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
    __slots__ = ("streams", "native_roots", "authority", "lease", "token",
                 "manifest", "row_identity", "batch", "inner_token")
    def __init__(self):
        self.streams, self.native_roots = [], []
        self.authority = self.lease = self.token = self.manifest = None
        self.row_identity = self.batch = self.inner_token = None

    def close_streams(self):
        while self.streams:
            stream = self.streams[-1]
            stream.close()
            self.streams.pop()


class Viewer1DAdoptionTransfer:
    """The one mutable cleanup-ownership cell; decision identity is authority."""
    __slots__ = ("identity", "request_identity", "transport_token_identity",
                 "active_receipt_identity", "issuer_transport_identity",
                 "port_identity", "_decision")
    def __init_subclass__(cls, **kwargs):
        raise TypeError("viewer 1-D transfer is not subclassable")
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
                or current.graph.batch is not batch
                or prepared_identity is None or retired_receipt_identity is not self.active_receipt_identity):
            raise RuntimeError("viewer 1-D batch transfer is not current")
        decision = _BatchCustody(batch, batch.batch_identity, prepared_identity,
            retired_receipt_identity, self.request_identity,
            self.transport_token_identity, self.issuer_transport_identity,
            self.port_identity, _FACTORY)
        self._decision = decision
        return decision

    def disposal(self, reason):
        current = self._decision
        if (type(current) is _DisposalCustody and current.claim is _FACTORY
                and current.disposal.transfer is self):
            return current.disposal
        if (type(current) not in (_BuildingCustody, _BatchCustody)
                or current.claim is not _FACTORY):
            raise RuntimeError("viewer 1-D cleanup is not transport-owned")
        disposal = Viewer1DDisposal(self, str(reason)[:256], _claim=_FACTORY)
        graph = current.graph if type(current) is _BuildingCustody else current.batch
        self._decision = _DisposalCustody(graph, disposal, _FACTORY)
        return disposal

    def owner_holder(self, receipt):
        current = self._decision
        if (type(current) is not _OwnerAdoption or current.claim is not _FACTORY
                or current.receipt is not receipt):
            raise RuntimeError("viewer 1-D adoption receipt is stale")
        return current.holder


_FACTORY = object()


class Viewer1DDisposal:
    __slots__ = ("transfer", "reason", "_inner_token", "_hooks", "_no_lease_step")
    def __init__(self, transfer, reason, *, _claim=None):
        if _claim is not _FACTORY: raise TypeError("foreign viewer 1-D disposal")
        self.transfer, self.reason = transfer, reason
        self._inner_token = self._hooks = None
        self._no_lease_step = 0

    def _act(self, retry):
        current = self.transfer.inspect()
        if type(current) is _Released and current.claim is _FACTORY: return True
        if (type(current) is not _DisposalCustody or current.claim is not _FACTORY
                or current.disposal is not self): return False
        graph = current.graph
        lease = graph.lease
        building = graph if type(graph) is _BuildingGraph else None
        if lease is None:
            if building is None: return False
            steps = (
                building.close_streams,
                building.native_roots.clear,
                lambda: _verify_unfunded_building(building),
                lambda: setattr(building, "authority", None),
            )
            while self._no_lease_step < len(steps):
                steps[self._no_lease_step]()
                self._no_lease_step += 1
            if self.transfer.inspect() is not current: return False
            self.transfer._decision = _Released(object(), _FACTORY)
            return True
        def cancel():
            if building is not None and building.token is not None:
                token = building.token
                lease.abandon_hydration(token)
                building.token = None
        def drain():
            if building is not None: building.close_streams()
        def detach():
            if building is not None: building.native_roots.clear()
        def verify():
            if building is not None and (building.streams or building.native_roots or building.token):
                raise RuntimeError("viewer 1-D native roots remain")
        if self._hooks is None:
            self._hooks = Light1DCleanupHooks(
                cancel=cancel, drain=drain, detach=detach, verify=verify)
        try:
            receipt = (lease.retry_cleanup(self._inner_token, hooks=self._hooks)
                       if retry and self._inner_token is not None else
                       lease.release(reason=self.reason, hooks=self._hooks))
        except Light1DCleanupPending as error:
            self._inner_token = error.token
            cause = error.__cause__
            if isinstance(cause, MemoryError) or not isinstance(cause, Exception):
                raise _Viewer1DDisposalControl(cause) from None
            return False
        if (type(receipt) is not Light1DReleaseReceipt
                or receipt.released_bytes != lease.reserved_ndarray_bytes
                or lease.keys() or lease.pending_hydration_count
                or lease.active_borrow_count
                or lease.authority.snapshot().reservation_count):
            return False
        if self.transfer.inspect() is not current: return False
        self.transfer._decision = _Released(object(), _FACTORY)
        return True

    def release(self): return self._act(False)
    def retry(self): return self._act(True)


class Viewer1DOwnerCleanupHolder:
    __slots__ = ("transfer", "batch", "borrow", "_inner_token")
    def __init__(self, transfer, batch, borrow, *, _claim=None):
        if _claim is not _FACTORY: raise TypeError("foreign viewer 1-D owner holder")
        self.transfer, self.batch, self.borrow = transfer, batch, borrow
        self._inner_token = None

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
        self.transfer._decision = _Released(object(), _FACTORY)
        return True


class _Viewer1DReaderControl(BaseException):
    def __init__(self, control, disposal): self.control, self.disposal = control, disposal


class _Viewer1DDisposalControl(BaseException):
    def __init__(self, control): self.control = control


@dataclass(frozen=True, slots=True)
class _Viewer1DPassTwoPermit:
    identity: object
    operation: object
    receipt_identity: object
    issuer: object
    capacity_bytes: int
    reserved_bytes: int
    gui_thread_id: int
    worker_thread_id: int
    claim: object


class Viewer1DReadOperation:
    __slots__ = ("request", "receipt_identity", "transfer", "facts", "ledger",
                 "issuer", "gui_thread_id", "worker_thread_id",
                 "_permit_identity", "_consumed")
    def __init__(self, request, receipt, transfer, facts, ledger, *, _claim=None):
        if _claim is not _FACTORY: raise TypeError("foreign viewer 1-D operation")
        self.request, self.receipt_identity, self.transfer = request, receipt.identity, transfer
        self.facts, self.ledger, self.issuer = facts, ledger, receipt.issuer
        self.gui_thread_id, self.worker_thread_id = receipt.gui_thread_id, receipt.worker_thread_id
        self._permit_identity, self._consumed = None, False

    def complete(self, permit):
        current = self.transfer.inspect()
        if (type(permit) is not _Viewer1DPassTwoPermit or permit.claim is not _FACTORY
                or permit.identity is not self._permit_identity or permit.operation is not self
                or permit.receipt_identity is not self.receipt_identity or self._consumed
                or permit.issuer is not self.issuer
                or permit.capacity_bytes != self.ledger.B
                or permit.reserved_bytes != self.ledger.R
                or permit.gui_thread_id != self.gui_thread_id
                or permit.worker_thread_id != self.worker_thread_id
                or type(current) is not _BuildingCustody or current.claim is not _FACTORY
                or current.graph.manifest.ledger is not self.ledger):
            return _failure(self.transfer, Viewer1DReadFailureStage.PASS_TWO,
                            "viewer 1-D pass-two permit is stale")
        self._consumed = True
        graph = current.graph
        try:
            modes = {}
            for index, fact in enumerate(self.facts):
                x, y, sigma = _read_fact(fact)
                graph.native_roots.extend(value for value in (x, y, sigma) if value is not None)
                modes[index] = Light1DModeData(x, y, sigma)
            graph.close_streams()
            record = Light1DRecord(graph.row_identity, self.request.generation, 0, modes,
                                   {"batch_identity": graph.manifest.identity})
            graph.lease.complete_hydration(graph.token, record)
            graph.token = None; graph.native_roots.clear()
            batch = Viewer1DBatch(graph.authority, graph.lease, graph.row_identity,
                                  graph.manifest, graph.manifest.identity)
            graph.batch = batch
            return batch
        except BaseException as error:
            if _is_control(error):
                disposal = self.transfer.disposal(_diag(error))
                raise _Viewer1DReaderControl(error, disposal) from None
            return _failure(self.transfer, Viewer1DReadFailureStage.PASS_TWO, _diag(error))


def viewer_1d_budget():
    try:
        ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        ram = 0
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
    from xdart.modules.display_context import Viewer1DReadBudgetState
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
            or operation._permit_identity is not None
            or graph is None or current.claim is not _FACTORY
            or not _funding_matches(graph, operation.ledger, receipt)):
        raise TypeError("foreign viewer 1-D permit")
    permit = _Viewer1DPassTwoPermit(object(), operation, receipt.identity, issuer,
        receipt.capacity_bytes, receipt.reserved_bytes, receipt.gui_thread_id,
        receipt.worker_thread_id, _FACTORY)
    operation._permit_identity = permit.identity
    return permit


def adopt_prepared_viewer_1d(prepared, *, owner_identity, owner_request_claim,
                             port, owner_generation, commit_gate, owner_state):
    from xdart.modules.display_context import (
        Prepared1DBatchCommit, Viewer1DState, _new_viewer_1d_commit_receipt,
    )
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
            or current.port_identity is not port
            or transfer.request_identity is not prepared.request
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
        receipt = _new_viewer_1d_commit_receipt(
            transfer, prepared.request, prepared, owner_identity)
        transfer._decision = _OwnerAdoption(holder, receipt, _FACTORY)
        return receipt
    except BaseException:
        if type(transfer.inspect()) is _BatchCustody and borrow is not None:
            borrow.close()
        raise
    finally: commit_gate.leave()


@dataclass(slots=True)
class _Fact:
    stream: object
    path: str
    state: tuple
    manifest: Viewer1DSourceManifest
    schema: int
    roles: object = None


def _diag(error):
    try: raw = str(error) or type(error).__name__
    except BaseException: raw = type(error).__name__
    while len(raw.encode("utf-8")) > 256: raw = raw[:-1]
    return raw


def _is_control(error):
    return isinstance(error, MemoryError) or (isinstance(error, BaseException)
                                               and not isinstance(error, Exception))


def _verify_unfunded_building(graph):
    if (graph.streams or graph.native_roots or graph.token is not None
            or graph.lease is not None
            or graph.authority is not None
            and graph.authority.snapshot().reservation_count != 0):
        raise RuntimeError("viewer 1-D unfunded cleanup is incomplete")


def _funding_matches(graph, ledger, receipt):
    if (graph.manifest is None or graph.manifest.ledger is not ledger
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
        and dict(snapshot.categories) == {
            "viewer_1d_transient": ledger.T, "light_1d": ledger.C}
        and snapshot.reserved_bytes == ledger.A
        and snapshot.available_bytes == 0
        and snapshot.reservation_count == 1)


def _failure(transfer, stage, diagnostic):
    disposal = transfer.disposal(diagnostic)
    return Viewer1DReadFailure(stage, _diag(diagnostic), disposal)


def _descriptor_limit_and_count():
    if os.name == "nt":
        crt = ctypes.CDLL("ucrtbase.dll", use_errno=True)
        function = crt._getmaxstdio; function.argtypes = []; function.restype = ctypes.c_int
        limit = function()
        if not 1 <= limit <= 8192: raise RuntimeError("descriptor headroom")
        count = 0
        for descriptor in range(limit):
            try: os.fstat(descriptor); count += 1
            except OSError as error:
                if getattr(error, "errno", None) != 9: raise
        return limit, count
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = hard if soft == resource.RLIM_INFINITY else soft
    directory = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
    if type(limit) is not int or limit <= 0 or not os.path.isdir(directory):
        raise RuntimeError("descriptor headroom")
    return limit, len(os.listdir(directory))


def _state(fd, path):
    opened, named = os.fstat(fd), os.stat(path)
    return (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns,
            opened.st_ctime_ns, getattr(opened, "st_gen", None), named.st_dev,
            named.st_ino, named.st_size, named.st_mtime_ns, named.st_ctime_ns,
            getattr(named, "st_gen", None))


def _digest(stream):
    stream.seek(0); digest = hashlib.sha256(); total = 0
    while True:
        block = stream.read(1024 * 1024)
        if not block: break
        digest.update(block); total += len(block)
    return digest.digest(), total


def _npy_header(stream, *, label=False):
    stream.seek(0); magic = stream.read(6)
    if magic != b"\x93NUMPY": raise ValueError("NPY magic")
    version = tuple(stream.read(2))
    if version not in {(1, 0), (2, 0), (3, 0)}: raise ValueError("NPY version")
    width = 2 if version == (1, 0) else 4
    raw_length = stream.read(width)
    if len(raw_length) != width: raise ValueError("NPY header")
    length = int.from_bytes(raw_length, "little")
    if length > 65536: raise ValueError("NPY header")
    raw = stream.read(length)
    if len(raw) != length: raise ValueError("NPY header")
    try: header = ast.literal_eval(raw.decode("latin1" if version != (3, 0) else "utf8").strip())
    except BaseException as error:
        if _is_control(error): raise
        raise ValueError("NPY header") from error
    if type(header) is not dict or set(header) != {"descr", "fortran_order", "shape"}:
        raise ValueError("NPY header schema")
    dtype, shape = np.dtype(header["descr"]), header["shape"]
    if (header["fortran_order"] is not False or dtype.hasobject or dtype.fields is not None
            or dtype.subdtype is not None or (dtype.kind != "U" if label else
                dtype.kind not in "biuf" or dtype.itemsize > 8)
            or type(shape) is not tuple): raise ValueError("NPY dtype/shape")
    size = dtype.itemsize
    for value in shape:
        if type(value) is not int or value < 0: raise ValueError("NPY shape")
        size *= value
    return dtype, shape, magic + bytes(version) + raw_length + raw, size


def _shape(shape):
    if len(shape) == 1 and 1 <= shape[0] <= 1_000_000: return shape[0], 1, True
    if len(shape) == 2 and 1 <= shape[0] <= 1_000_000 and shape[1] in (2, 3):
        return shape[0], shape[1], False
    raise ValueError("NPY shape")


def _dtype_code(dtype):
    value = dtype.str
    if value[0] == "=": value = ("<" if sys.byteorder == "little" else ">") + value[1:]
    try: return _DTYPE_CODES[value]
    except KeyError: raise ValueError("NPY dtype") from None


def _scan_text(stream, csv):
    stream.seek(0); count = columns = 0; labels = ("x", "", "intensity"); header = False
    first = True
    while True:
        raw = stream.readline(4098)
        if not raw: break
        if len(raw) > 4096 or b"\x00" in raw: raise ValueError("text line bound")
        if first and raw.startswith(b"\xef\xbb\xbf"): raw = raw[3:]
        first = False
        try: line = raw.decode("utf8").strip()
        except UnicodeDecodeError as error: raise ValueError("text encoding") from error
        if not line or line.lstrip().startswith("#"): continue
        if '"' in line or "'" in line or "\\" in line: raise ValueError("text quoting")
        cells = [value.strip() for value in line.split(",")] if csv else line.split()
        if len(cells) not in (2, 3) or any(not value for value in cells): raise ValueError("text columns")
        try: values = tuple(float(value) for value in cells)
        except ValueError:
            if not csv or count or header: raise ValueError("CSV header")
            if any(len(value.encode("utf8")) > 128 for value in cells): raise ValueError("CSV label")
            labels, header = (cells[0], "", cells[1]), True
            continue
        columns = columns or len(values)
        if len(values) != columns or not np.isfinite(values[0]) or any(np.isinf(v) for v in values[1:]):
            raise ValueError("text numeric values")
        if len(values) == 3 and np.isfinite(values[2]) and values[2] < 0:
            raise ValueError("negative sigma")
        count += 1
        if count > 1_000_000: raise ValueError("point count")
    if count < 1: raise ValueError("empty source")
    return count, columns, labels


def _record(index, format_code, schema, columns, dtypes, sigma, synthesized,
            labels, ordinals, selected, members, path_bytes, points, encoded,
            compressed=0, uncompressed=0):
    value = bytearray(64); value[0:8] = bytes((1, format_code, schema, columns,
        *(dtypes + (0, 0, 0))[:3], int(sigma) | (int(synthesized) << 1)))
    struct.pack_into(">H", value, 8, index); value[10:16] = bytes(ordinals)
    value[16:19] = bytes(len(label.encode()) for label in labels)
    value[19:22] = bytes((selected, members, sum(code != 0 for code in dtypes)))
    struct.pack_into(">HQQQQ", value, 22, path_bytes, points, encoded, compressed, uncompressed)
    return bytes(value)


def _inspect_npz(stream, index, path, digest, encoded):
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive:
        infos = archive.infolist(); names = [info.filename for info in infos]
        if (not 1 <= len(infos) <= 16 or len(set(names)) != len(names)
                or len(archive.comment) > 4096
                or sum(46 + len(info.filename.encode("utf8")) + len(info.extra)
                       + len(info.comment) for info in infos) > 128 * 1024):
            raise ValueError("NPZ member bound")
        for info in infos:
            parts = info.filename.replace("\\", "/").split("/")
            if (not info.filename.endswith(".npy") or info.is_dir() or info.flag_bits & 1
                    or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    or info.file_size > 1024**3 or len(info.extra) > 4096
                    or len(info.filename.encode("utf8")) > 256 or info.filename.startswith("/")
                    or ".." in parts or any(not part for part in parts)
                    or len(parts[0]) >= 2 and parts[0][1] == ":"):
                raise ValueError("NPZ member metadata")
        if sum(info.file_size for info in infos) > 1024**3: raise ValueError("NPZ size")
        stems = {name[:-4]: name for name in names}; labels = {"x_label", "x_unit", "y_label"}
        numeric = set(stems) - labels
        if {"x", "y"} <= numeric and numeric <= {"x", "y", "sigma"}: schema, roles = 1, ("x", "y") + (("sigma",) if "sigma" in numeric else ())
        elif numeric == {"data"}: schema, roles = 2, ("data",)
        elif len(numeric) == 1 and not numeric & {"x", "y", "sigma", "data"}: schema, roles = 3, tuple(numeric)
        else: raise ValueError("ambiguous NPZ schema")
        if set(stems) != set(roles) | (set(stems) & labels): raise ValueError("NPZ extra member")
        raw_records, position = [], archive.start_dir
        for info in infos:
            stream.seek(position); central = stream.read(46)
            if len(central) != 46 or central[:4] != b"PK\x01\x02": raise ValueError("NPZ central directory")
            lengths = struct.unpack_from("<HHH", central, 28)
            central += stream.read(sum(lengths)); position += len(central)
            stream.seek(info.header_offset); local = stream.read(30)
            if len(local) != 30 or local[:4] != b"PK\x03\x04": raise ValueError("NPZ local header")
            local_lengths = struct.unpack_from("<HH", local, 26)
            if local_lengths[0] > 256 or local_lengths[1] > 4096:
                raise ValueError("NPZ local metadata")
            local += stream.read(sum(local_lengths))
            raw_records.append((central, local))
        parsed, label_values = {}, {"x_label": "x", "x_unit": "",
            "y_label": os.path.splitext(os.path.basename(path))[0]}
        meta = hashlib.sha256(b"V1DMETA1")
        for role in (*roles, *(name for name in ("x_label", "x_unit", "y_label") if name in stems)):
            info = infos[names.index(stems[role])]
            with archive.open(info) as member:
                dtype, shape, prefix, payload = _npy_header(member, label=role in labels)
                if member.tell() + payload != info.file_size: raise ValueError("NPZ declared bytes")
                if role in labels:
                    if dtype.kind != "U" or shape != () or payload > 512: raise ValueError("NPZ label")
                    raw = member.read(payload)
                    if len(raw) != payload or member.read(1): raise ValueError("NPZ member EOF")
                    order = dtype.byteorder
                    if order in {"=", "|"}: order = "<" if sys.byteorder == "little" else ">"
                    text = raw.decode("utf-32-le" if order == "<" else "utf-32-be").rstrip("\0")
                    if type(text) is not str or not 1 <= len(text.encode("utf8")) <= 128 or "\0" in text:
                        raise ValueError("NPZ label")
                    label_values[role] = text
                else:
                    parsed[role] = (dtype, shape, info)
                    while member.read(1024 * 1024): pass
                ordinal = names.index(info.filename)
                tag = ((roles.index(role) if role in roles else
                        3 + ("x_label", "x_unit", "y_label").index(role)))
                encoded_name = info.filename.encode("utf8")
                central, local = raw_records[ordinal]
                meta.update(bytes((tag, ordinal)) + len(encoded_name).to_bytes(2, "big")
                    + encoded_name + len(central).to_bytes(4, "big") + central
                    + len(local).to_bytes(4, "big") + local
                    + len(prefix).to_bytes(4, "big") + prefix)
        if schema == 1:
            shapes = [parsed[role][1] for role in roles]
            if any(len(shape) != 1 for shape in shapes) or len({shape[0] for shape in shapes}) != 1:
                raise ValueError("NPZ named shape")
            points, columns, synthesized = shapes[0][0], len(roles), False
            if not 1 <= points <= 1_000_000: raise ValueError("point count")
            dtypes = tuple(_dtype_code(parsed[role][0]) for role in roles)
        else:
            points, columns, synthesized = _shape(parsed[roles[0]][1])
            dtypes = (_dtype_code(parsed[roles[0]][0]),)
        ordinals = tuple(names.index(stems.get(role, "")) if role in stems else 0xff
                         for role in (roles + (None,) * 3)[:3] + ("x_label", "x_unit", "y_label"))
        selected = len(roles) + sum(name in stems for name in labels)
        selected_infos = [infos[names.index(stems[role])] for role in
            (*roles, *(name for name in ("x_label", "x_unit", "y_label") if name in stems))]
        compressed = sum(info.compress_size for info in selected_infos)
        uncompressed = sum(info.file_size for info in selected_infos)
        label_tuple = (label_values["x_label"], label_values["x_unit"], label_values["y_label"])
        scalar = _record(index, 4, schema, columns, dtypes, columns == 3, synthesized,
                         label_tuple, ordinals, selected, len(infos), len(str(path).encode()),
                         points, encoded, compressed, uncompressed)
        manifest = Viewer1DSourceManifest(str(path), "npz", points, columns == 3,
            *label_tuple, digest, meta.digest(), scalar)
        return manifest, schema, MappingProxyType({role: stems[role] for role in roles})


def _inspect(stream, index, path, state, *, retain_roles=False):
    digest, encoded = _digest(stream)
    if encoded > 256 * 1024**2: raise ValueError("encoded size")
    suffix = os.path.splitext(path)[1].lower(); code = _FORMATS.get(suffix)
    if code is None: raise ValueError("viewer suffix")
    if suffix == ".npz":
        manifest, schema, roles = _inspect_npz(stream, index, path, digest, encoded)
        return _Fact(stream, str(path), state, manifest, schema,
                     roles if retain_roles else None)
    if suffix in {".xye", ".csv"}:
        points, columns, labels = _scan_text(stream, suffix == ".csv")
        dtypes, synthesized = (), False
    else:
        stream.seek(0); dtype, shape, _, payload = _npy_header(stream)
        points, columns, synthesized = _shape(shape)
        if stream.tell() + payload != encoded: raise ValueError("NPY EOF")
        dtypes, labels = (_dtype_code(dtype),), ("x", "", os.path.splitext(os.path.basename(path))[0])
    if any(len(label.encode("utf8")) > 128 for label in labels):
        raise ValueError("viewer label bound")
    scalar = _record(index, code, 0, columns, dtypes, columns == 3, synthesized,
                     labels, (0xff,) * 6, 0, 0, len(str(path).encode()), points, encoded)
    manifest = Viewer1DSourceManifest(str(path), suffix[1:], points, columns == 3,
        *labels, digest, bytes(32), scalar)
    return _Fact(stream, str(path), state, manifest, 0, None)


def _columns(array):
    value = np.asarray(array)
    if value.ndim == 1: return np.arange(len(value), dtype=float), np.array(value, dtype=float, copy=True), None
    return (np.array(value[:, 0], dtype=float, copy=True),
            np.array(value[:, 1], dtype=float, copy=True),
            None if value.shape[1] == 2 else np.array(value[:, 2], dtype=float, copy=True))


def _read_text(stream, csv, points):
    stream.seek(0); columns = None; result = None; row = 0
    first = True
    while True:
        raw = stream.readline(4098)
        if not raw: break
        if first and raw.startswith(b"\xef\xbb\xbf"): raw = raw[3:]
        first = False; line = raw.decode("utf8").strip()
        if not line or line.lstrip().startswith("#"): continue
        cells = [value.strip() for value in line.split(",")] if csv else line.split()
        try: values = tuple(float(value) for value in cells)
        except ValueError: continue
        if result is None:
            columns = len(values)
            result = tuple(np.empty(points, dtype=float) for _ in range(columns))
        for target, value in zip(result, values): target[row] = value
        row += 1
    if result is None or row != points: raise ValueError("text point count changed")
    return result[0], result[1], None if columns == 2 else result[2]


def _read_fact(fact):
    stream, manifest = fact.stream, fact.manifest
    recertified = _inspect(stream, int.from_bytes(manifest.scalar_record[8:10], "big"),
                           fact.path, fact.state, retain_roles=True)
    if recertified.manifest != manifest or _state(stream.fileno(), fact.path) != fact.state:
        raise ValueError("viewer source changed")
    if manifest.format in {"xye", "csv"}: values = _read_text(stream, manifest.format == "csv", manifest.points)
    elif manifest.format == "npy":
        stream.seek(0); values = _columns(np.load(stream, allow_pickle=False))
    else:
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            arrays = {role: np.load(archive.open(name), allow_pickle=False)
                      for role, name in recertified.roles.items()}
        if recertified.schema == 1:
            values = (np.array(arrays["x"], dtype=float, copy=True),
                      np.array(arrays["y"], dtype=float, copy=True),
                      None if "sigma" not in arrays else np.array(arrays["sigma"], dtype=float, copy=True))
        else: values = _columns(next(iter(arrays.values())))
    x, y, sigma = values
    if (len(x) != manifest.points or not np.all(np.isfinite(x))
            or np.any(np.isinf(y)) or sigma is not None and
            (np.any(np.isinf(sigma)) or np.any(sigma[np.isfinite(sigma)] < 0))):
        raise ValueError("viewer numeric values changed")
    if _state(stream.fileno(), fact.path) != fact.state: raise ValueError("viewer source changed")
    return values


def begin_viewer_1d_read(request, receipt, transfer):
    from xdart.modules.display_context import Viewer1DReadBudgetReceipt, Viewer1DReadBudgetState
    if (type(receipt) is not Viewer1DReadBudgetReceipt or receipt.state is not Viewer1DReadBudgetState.ACTIVE
            or receipt.request is not request or receipt.identity is not transfer.active_receipt_identity
            or receipt.capacity_bytes != viewer_1d_budget() or receipt.reserved_bytes != VIEWER_1D_R
            or receipt.worker_thread_id != threading.get_ident()):
        return _failure(transfer, Viewer1DReadFailureStage.DESCRIPTOR, "viewer 1-D receipt mismatch")
    current = transfer.inspect()
    if type(current) is not _BuildingCustody or current.claim is not _FACTORY:
        return _failure(transfer, Viewer1DReadFailureStage.DESCRIPTOR,
                        "viewer 1-D construction custody is stale")
    graph = current.graph
    stage = Viewer1DReadFailureStage.DESCRIPTOR
    try:
        paths = tuple(os.path.realpath(os.path.expanduser(path)) for path in request.paths)
        if any(len(os.fsencode(path)) > 4096 for path in paths): raise ValueError("path bound")
        limit, count = _descriptor_limit_and_count()
        if limit - count < len(paths) + VIEWER_1D_FD_RESERVE + 1:
            raise RuntimeError("descriptor headroom")
        descriptors = []
        try:
            for _ in paths: descriptors.append(os.open(os.devnull, os.O_RDONLY))
            for descriptor, path in zip(descriptors, paths):
                source = os.open(path, os.O_RDONLY)
                try:
                    source_state = os.fstat(source)
                    if (not stat.S_ISREG(source_state.st_mode)
                            or source_state.st_size > 256 * 1024**2):
                        raise ValueError("regular source/encoded size")
                    os.lseek(source, 0, os.SEEK_CUR); os.dup2(source, descriptor)
                finally: os.close(source)
            for index, descriptor in enumerate(descriptors):
                graph.streams.append(os.fdopen(descriptor, "rb", buffering=0))
                descriptors[index] = -1
        finally:
            for descriptor in descriptors:
                if descriptor >= 0: os.close(descriptor)
        stage = Viewer1DReadFailureStage.PASS_ONE
        facts = tuple(_inspect(stream, index, path, _state(stream.fileno(), path))
                      for index, (stream, path) in enumerate(zip(graph.streams, paths)))
        if any(_state(fact.stream.fileno(), fact.path) != fact.state for fact in facts):
            raise ValueError("viewer source changed")
        counts = tuple(fact.manifest.points for fact in facts)
        sigmas = tuple(fact.manifest.sigma_present for fact in facts)
        encoded = tuple(os.fstat(fact.stream.fileno()).st_size for fact in facts)
        if sum(encoded) > 1024**3: raise ValueError("encoded total")
        C = 8 * sum(n * (2 + int(sigma)) for n, sigma in zip(counts, sigmas))
        P, B = len(paths) * max(counts), receipt.capacity_bytes
        N, T = 9 * 8 * P, VIEWER_1D_R + max(3 * C, 9 * 8 * P)
        ledger = Viewer1DMemoryLedger(counts, sigmas, encoded, B, P, C, VIEWER_1D_R, N, T, C + T)
        if ledger.A > B: raise ValueError(f"viewer 1-D budget requires {ledger.A}/{B}")
        identity = hashlib.sha256(b"V1DBATCH1" + b"".join(
            fact.manifest.scalar_record + fact.manifest.source_sha256 +
            fact.manifest.role_sha256 + fact.manifest.canonical_path.encode()
            for fact in facts)).hexdigest()
        graph.manifest = Viewer1DBatchManifest(identity, request.policy.identity,
                                               tuple(fact.manifest for fact in facts), ledger)
        layouts = []
        for index, (points, sigma) in enumerate(zip(counts, sigmas)):
            spec = lambda role: Light1DBufferLayout(points, 8, f"{index}:{role}", "<f8")
            layouts.append(Light1DModeLayout(index, spec("x"), spec("y"),
                                             spec("sigma") if sigma else None))
        graph.row_identity = ("viewer-1d", request.generation, identity)
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
        return Viewer1DReadOperation(request, receipt, transfer, facts, ledger,
                                     _claim=_FACTORY)
    except BaseException as error:
        if _is_control(error):
            disposal = transfer.disposal(_diag(error))
            raise _Viewer1DReaderControl(error, disposal) from None
        return _failure(transfer, stage, _diag(error))


__all__ = [name for name in tuple(globals()) if name.startswith("Viewer1D") or
           name in {"VIEWER_1D_FD_RESERVE", "VIEWER_1D_R", "begin_viewer_1d_read",
                    "viewer_1d_budget", "mint_viewer_1d_transfer",
                    "mint_viewer_1d_pass_two_permit", "adopt_prepared_viewer_1d",
                    "viewer_1d_disposal_is_current", "viewer_1d_transfer_is_released"}]
