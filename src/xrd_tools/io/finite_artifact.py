"""Immutable finite-artifact naming, candidate ownership, and publication.

Finite operations never open an existing public artifact for writing.  This
module owns one smaller protocol: build and validate a same-directory private
file, atomically link it to an absent public ``.nexus`` name, and never remove
that public name after publication is observable.  Append/Live continuity is
deliberately outside this module.

Document adapters are trusted in-process repository code, not a Python
sandbox boundary.  They must use only the supplied public file-object methods
and must never inspect, duplicate, retain, or export the private backing file
descriptor.  Protecting against a deliberately hostile same-process adapter
would require process isolation or a second post-adapter copy.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading
from typing import Callable, ContextManager

from xrd_tools.io.output_path import (
    NEW_OUTPUT_SUFFIX,
    artifact_family_from_source,
    resolve_finite_output_target,
)
from xrd_tools.io.output_transaction import (
    StreamTerminal,
    get_output_transaction_coordinator,
    replace_into_place,
    revalidate_stream_terminal,
)


_FACTORY = object()
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_OPERATION_KIND = re.compile(r"[a-z0-9][a-z0-9-]{0,39}\Z")
# ONE family vocabulary, imported rather than restated.  This module used to
# keep its own ASCII-only copy, so a family `output_path` accepted (`Sample A
# 001`) was rejected here -- two spellings of the same rule is how the widening
# would have silently failed one layer down.
from xrd_tools.io.output_path import _ARTIFACT_FAMILY
_SCHEMA_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_COPY_BLOCK_BYTES = 1024 * 1024
_CANDIDATE_SPACE_MARGIN_BYTES = 64 * 1024 * 1024
_MAX_PATH_BYTES = 4096
_MAX_ENTRY_BYTES = 4096
_MAX_CANONICAL_BYTES = 64 * 1024
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1

# Frozen cross-adapter spellings.  The lineage child lives under the selected
# NXentry; adapters persist the schema/policy plus request identities there.
FINITE_LINEAGE_NODE_NAME = "finite_artifact"
FINITE_LINEAGE_SCHEMA = "xdart.finite-artifact-lineage.v1"
FINITE_LINEAGE_MAX_BYTES = _MAX_CANONICAL_BYTES
FINITE_PUBLICATION_POLICY = "IMMUTABLE_SUCCESSOR_V1"
FINITE_PARENT_DIRECTORY_FSYNC_WARNING = (
    "FINITE_PARENT_DIRECTORY_FSYNC_UNCONFIRMED"
)
FINITE_CANDIDATE_CLEANUP_WARNING = "FINITE_CANDIDATE_CLEANUP_INCOMPLETE"
FINITE_SLOT_LEASE_WARNING = "finite-slot-lease-release-failed"
FINITE_DESCRIPTOR_CLOSE_WARNING = "FINITE_DESCRIPTOR_CLOSE_INCOMPLETE"

# Narrow deterministic fault seams.  Production uses the standard library.
_fsync = os.fsync


def _fsync_parent(descriptor: int) -> None:
    _fsync(descriptor)


class FiniteArtifactError(RuntimeError):
    """Base class for immutable finite-publication refusals."""


class FiniteArtifactCollision(FiniteArtifactError):
    """A foreign or non-identical object occupied an owned namespace."""


class FiniteArtifactIntegrityError(FiniteArtifactError):
    """An admitted source, candidate, or public terminal lost exact identity."""


class FiniteArtifactCapacityError(FiniteArtifactError):
    """The exact candidate directory lacks the frozen finite-space budget."""

    def __init__(
        self,
        request: "FiniteArtifactRequest",
        *,
        required_bytes: int,
        available_bytes: int,
    ) -> None:
        if (
            type(request) is not FiniteArtifactRequest
            or type(required_bytes) is not int
            or required_bytes < 0
            or type(available_bytes) is not int
            or available_bytes < 0
        ):
            raise TypeError("finite capacity refusal requires exact byte counts")
        self.request = request
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes
        self.cleanup_directory = str(Path(request.output_artifact).parent)
        super().__init__(
            "finite candidate needs "
            f"{required_bytes} bytes but only {available_bytes} bytes are "
            "available in the exact manual-cleanup directory: "
            f"{self.cleanup_directory}"
        )


class FiniteArtifactPublicationHeld(FiniteArtifactIntegrityError):
    """The publisher linked its candidate but final inspection is unresolved."""

    def __init__(
        self,
        request: "FiniteArtifactRequest",
        cause: BaseException,
        *,
        hidden_orphan: str | None = None,
        diagnostics: tuple[str, ...] = (),
    ) -> None:
        if (
            type(request) is not FiniteArtifactRequest
            or not isinstance(cause, BaseException)
            or hidden_orphan is not None
            and (type(hidden_orphan) is not str or not hidden_orphan)
            or type(diagnostics) is not tuple
            or any(
                type(item) is not str or len(item) > 1024
                for item in diagnostics
            )
        ):
            raise TypeError("held finite publication requires an exact request")
        self.request = request
        self.cause = cause
        self.hidden_orphan = hidden_orphan
        self.diagnostics = diagnostics
        super().__init__(
            "finite publication is visible but exact terminal inspection is held"
        )


class FiniteArtifactDisposition(str, Enum):
    COMMITTED = "COMMITTED"
    ALREADY_COMMITTED = "ALREADY_COMMITTED"
    ABORTED = "ABORTED"


class FiniteCandidateWriteDisposition(str, Enum):
    """One exact adapter-to-publisher private-candidate outcome."""

    ABORTED = "ABORTED"


@dataclass(frozen=True, slots=True)
class FiniteFileSnapshot:
    path: str
    size: int
    digest: str
    device: int
    inode: int
    mode: int
    mtime_ns: int
    ctime_ns: int

    def __post_init__(self) -> None:
        if (
            type(self.path) is not str
            or not self.path
            or type(self.size) is not int
            or self.size < 0
            or type(self.digest) is not str
            or not _LOWER_HEX_64.fullmatch(self.digest)
            or any(
                type(value) is not int or value < 0
                for value in (
                    self.device,
                    self.inode,
                    self.mode,
                    self.mtime_ns,
                    self.ctime_ns,
                )
            )
            or any(
                value > _I64_MAX
                for value in (
                    self.size,
                    self.device,
                    self.inode,
                    self.mode,
                    self.mtime_ns,
                    self.ctime_ns,
                )
            )
        ):
            raise TypeError("finite file snapshot is invalid")


@dataclass(frozen=True, slots=True)
class FiniteSourceAdmission:
    snapshot: FiniteFileSnapshot
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if _claim is not _FACTORY or type(self.snapshot) is not FiniteFileSnapshot:
            raise TypeError("finite source admission is not capture-owned")

    @property
    def path(self) -> str:
        return self.snapshot.path


@dataclass(frozen=True, slots=True)
class FiniteSourceSeedReceipt:
    source_snapshot: FiniteFileSnapshot
    candidate_snapshot: FiniteFileSnapshot
    source_digest: str
    candidate_digest: str
    byte_count: int
    copy_strategy: str
    receipt_digest: str
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        payload = {
            "byte_count": self.byte_count,
            "candidate_digest": self.candidate_digest,
            "candidate_inode": self.candidate_snapshot.inode,
            "candidate_path": self.candidate_snapshot.path,
            "copy_strategy": self.copy_strategy,
            "domain": "xdart.finite-source-seed-receipt.v1",
            "source_digest": self.source_digest,
            "source_inode": self.source_snapshot.inode,
            "source_path": self.source_snapshot.path,
        }
        try:
            _canonical_receipt, expected_digest = _identity(payload)
        except (TypeError, ValueError) as error:
            raise TypeError("finite source seed receipt is invalid") from error
        if (
            _claim is not _FACTORY
            or type(self.source_snapshot) is not FiniteFileSnapshot
            or type(self.candidate_snapshot) is not FiniteFileSnapshot
            or self.source_snapshot.path == self.candidate_snapshot.path
            or self.source_snapshot.digest != self.candidate_snapshot.digest
            or self.source_digest != self.source_snapshot.digest
            or self.candidate_digest != self.candidate_snapshot.digest
            or type(self.byte_count) is not int
            or self.byte_count != self.source_snapshot.size
            or self.byte_count != self.candidate_snapshot.size
            or self.copy_strategy != "bounded-copy-v1"
            or self.receipt_digest != expected_digest
        ):
            raise TypeError("finite source seed receipt is not factory-owned")


@dataclass(frozen=True, slots=True)
class FiniteOperationContext:
    source_snapshot: FiniteFileSnapshot
    request_generation_identity: str
    resource_allocation_identity: str
    route_identity: str
    custody_identity: str
    canonical_json: str
    context_identity: str

    def __post_init__(self) -> None:
        if type(self.source_snapshot) is not FiniteFileSnapshot:
            raise TypeError("finite operation context requires an exact source snapshot")
        for role, value in (
            ("request generation", self.request_generation_identity),
            ("resource allocation", self.resource_allocation_identity),
            ("route", self.route_identity),
            ("custody", self.custody_identity),
            ("context", self.context_identity),
        ):
            if type(value) is not str or not _LOWER_HEX_64.fullmatch(value):
                raise ValueError(f"finite {role} identity is invalid")
        canonical, identity = _identity(_operation_context_payload(self))
        if self.canonical_json != canonical or self.context_identity != identity:
            raise ValueError("finite operation context is not canonical")


@dataclass(frozen=True, slots=True)
class FinitePredecessorReceipt:
    source_snapshot: FiniteFileSnapshot
    terminal: StreamTerminal | None
    artifact_family_v1: str | None
    version_identity: str | None
    publication_identity: str | None
    lineage_identity: str | None

    def __post_init__(self) -> None:
        if type(self.source_snapshot) is not FiniteFileSnapshot:
            raise TypeError("finite predecessor requires an exact source snapshot")
        finite_values = (
            self.artifact_family_v1,
            self.version_identity,
            self.publication_identity,
            self.lineage_identity,
        )
        finite_parent = all(value is not None for value in finite_values)
        if not finite_parent and not all(value is None for value in finite_values):
            raise ValueError("finite predecessor lineage is partially populated")
        if finite_parent:
            if (
                type(self.artifact_family_v1) is not str
                or not _ARTIFACT_FAMILY.fullmatch(self.artifact_family_v1)
            ):
                raise ValueError("finite predecessor family is invalid")
            for value in (
                self.version_identity,
                self.publication_identity,
                self.lineage_identity,
            ):
                if type(value) is not str or not _LOWER_HEX_64.fullmatch(value):
                    raise ValueError("finite predecessor identity is invalid")
        if self.terminal is not None:
            if type(self.terminal) is not StreamTerminal:
                raise TypeError("finite predecessor terminal is invalid")
            snapshot = self.source_snapshot
            terminal = self.terminal
            if (
                terminal.target != snapshot.path
                or terminal.size != snapshot.size
                or terminal.digest != snapshot.digest
                or (
                    terminal.device,
                    terminal.inode,
                    terminal.mtime_ns,
                    terminal.ctime_ns,
                )
                != (
                    snapshot.device,
                    snapshot.inode,
                    snapshot.mtime_ns,
                    snapshot.ctime_ns,
                )
            ):
                raise ValueError("finite predecessor terminal does not match source")


@dataclass(frozen=True, slots=True)
class FiniteArtifactLineage:
    canonical_json: str
    lineage_identity: str
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _FACTORY
            or type(self.canonical_json) is not str
            or len(self.canonical_json.encode("utf-8", errors="strict"))
            > FINITE_LINEAGE_MAX_BYTES
            or type(self.lineage_identity) is not str
            or not _LOWER_HEX_64.fullmatch(self.lineage_identity)
            or hashlib.sha256(
                self.canonical_json.encode("utf-8", errors="strict")
            ).hexdigest()
            != self.lineage_identity
        ):
            raise TypeError("finite artifact lineage is not factory-owned")


@dataclass(frozen=True, slots=True)
class FiniteCandidateValidation:
    lineage: FiniteArtifactLineage

    def __post_init__(self) -> None:
        if type(self.lineage) is not FiniteArtifactLineage:
            raise TypeError("finite candidate validation is invalid")


@dataclass(frozen=True, slots=True)
class _ValidatedCandidateReceipt:
    """Attempt-local proof joining exact bytes to semantic validation."""

    snapshot: FiniteFileSnapshot
    validation: FiniteCandidateValidation
    operation_identity: str
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _FACTORY
            or type(self.snapshot) is not FiniteFileSnapshot
            or type(self.validation) is not FiniteCandidateValidation
            or type(self.operation_identity) is not str
            or not _LOWER_HEX_64.fullmatch(self.operation_identity)
        ):
            raise TypeError("validated candidate receipt is not factory-owned")


@dataclass(frozen=True, slots=True)
class FiniteCommittedInspection:
    terminal: StreamTerminal
    lineage: FiniteArtifactLineage

    def __post_init__(self) -> None:
        if (
            type(self.terminal) is not StreamTerminal
            or type(self.lineage) is not FiniteArtifactLineage
        ):
            raise TypeError("finite committed inspection is invalid")


@dataclass(frozen=True, slots=True)
class FiniteArtifactRequest:
    source_artifact: str
    output_artifact: str
    artifact_family: str
    operation_kind: str
    source_graph_identity: str
    entry: str
    scientific_identity: str
    output_schema: str
    algorithm_identity: str
    preservation_identity: str
    operation_context: FiniteOperationContext
    predecessor: FinitePredecessorReceipt
    lineage: FiniteArtifactLineage
    canonical_version_json: str
    version_identity: str
    publication_identity: str
    operation_identity: str

    @property
    def operation_context_identity(self) -> str:
        return self.operation_context.context_identity

    def __post_init__(self) -> None:
        for role, value in (
            ("source artifact", self.source_artifact),
            ("output artifact", self.output_artifact),
        ):
            if (
                type(value) is not str
                or not value
                or not os.path.isabs(value)
                or "\x00" in value
                or len(value.encode("utf-8")) > _MAX_PATH_BYTES
            ):
                raise TypeError(f"finite {role} path is invalid")
        if self.source_artifact == self.output_artifact:
            raise ValueError("finite source and output must be distinct")
        if Path(self.output_artifact).suffix != NEW_OUTPUT_SUFFIX:
            raise ValueError("finite output must end in exact .nexus")
        if not _ARTIFACT_FAMILY.fullmatch(self.artifact_family):
            raise ValueError("finite artifact family is invalid")
        if not _OPERATION_KIND.fullmatch(self.operation_kind):
            raise ValueError("finite operation kind is invalid")
        if (
            type(self.entry) is not str
            or not self.entry
            or "\x00" in self.entry
            or "/" in self.entry
            or self.entry in {".", ".."}
            or len(self.entry.encode("utf-8", errors="strict")) > _MAX_ENTRY_BYTES
        ):
            raise ValueError("finite entry is invalid")
        if not _SCHEMA_TOKEN.fullmatch(self.output_schema):
            raise ValueError("finite output schema is invalid")
        for role, value in (
            ("source graph", self.source_graph_identity),
            ("science", self.scientific_identity),
            ("algorithm", self.algorithm_identity),
            ("preservation", self.preservation_identity),
            ("version", self.version_identity),
            ("publication", self.publication_identity),
            ("operation", self.operation_identity),
        ):
            if type(value) is not str or not _LOWER_HEX_64.fullmatch(value):
                raise ValueError(f"finite {role} identity is invalid")
        if (
            type(self.operation_context) is not FiniteOperationContext
            or type(self.predecessor) is not FinitePredecessorReceipt
            or type(self.lineage) is not FiniteArtifactLineage
            or self.operation_context.source_snapshot.path != self.source_artifact
            or self.predecessor.source_snapshot
            != self.operation_context.source_snapshot
        ):
            raise ValueError("finite request source custody is inconsistent")
        if (
            type(self.canonical_version_json) is not str
            or len(self.canonical_version_json.encode("utf-8"))
            > _MAX_CANONICAL_BYTES
        ):
            raise ValueError("finite canonical version input is invalid")
        expected_version = {
            "algorithm_identity": self.algorithm_identity,
            "domain": "xdart.finite-artifact-version.v1",
            "entry": self.entry,
            "operation_kind": self.operation_kind,
            "output_schema": self.output_schema,
            "preservation_identity": self.preservation_identity,
            "scientific_identity": self.scientific_identity,
            "source_graph_identity": self.source_graph_identity,
        }
        canonical_version, version = _identity(expected_version)
        _publication_json, publication = _identity({
            "domain": "xdart.finite-artifact-publication.v1",
            "output_artifact": self.output_artifact,
            "version_identity": version,
        })
        _operation_json, operation = _identity({
            "domain": "xdart.finite-artifact-operation.v1",
            "operation_context_identity": self.operation_context_identity,
            "output_artifact": self.output_artifact,
            "publication_identity": publication,
            "source_artifact": self.source_artifact,
            "version_identity": version,
        })
        expected_lineage = _lineage_for_request_fields(
            artifact_family=self.artifact_family,
            operation_kind=self.operation_kind,
            source_graph_identity=self.source_graph_identity,
            entry=self.entry,
            scientific_identity=self.scientific_identity,
            output_schema=self.output_schema,
            algorithm_identity=self.algorithm_identity,
            preservation_identity=self.preservation_identity,
            publication_identity=publication,
            version_identity=version,
            predecessor=self.predecessor,
        )
        if (
            self.source_artifact != _resolved_parent_path(self.source_artifact)
            or self.output_artifact != _resolved_parent_path(self.output_artifact)
            or self.canonical_version_json != canonical_version
            or self.version_identity != version
            or self.publication_identity != publication
            or self.operation_identity != operation
            or self.lineage != expected_lineage
            or self.predecessor.version_identity == version
            or self.predecessor.publication_identity == publication
            or self.predecessor.lineage_identity == expected_lineage.lineage_identity
        ):
            raise ValueError("finite request identities are not canonical")


class _BindingOwner:
    # An underscore is an explicit trusted-code boundary, not a Python
    # security mechanism.  FiniteDocumentAdapter implementations must never
    # inspect this owner or duplicate/export its backing descriptor.
    __slots__ = ("active", "descriptor", "lock", "writable")

    def __init__(self, descriptor: int, *, writable: bool) -> None:
        self.active = True
        self.descriptor = descriptor
        self.lock = threading.RLock()
        self.writable = writable

    def revoke(self) -> BaseException | None:
        with self.lock:
            if not self.active:
                return None
            self.active = False
            descriptor = self.descriptor
            self.descriptor = -1
            return _close_descriptor_once(descriptor)


@dataclass(eq=False, frozen=True, slots=True)
class FiniteCandidateBinding:
    request: FiniteArtifactRequest
    candidate_identity: str
    _owner: _BindingOwner
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _FACTORY
            or type(self.request) is not FiniteArtifactRequest
            or type(self.candidate_identity) is not str
            or not _LOWER_HEX_64.fullmatch(self.candidate_identity)
            or type(self._owner) is not _BindingOwner
        ):
            raise TypeError("finite candidate binding is not factory-owned")

    @property
    def closed(self) -> bool:
        with self._owner.lock:
            return not self._owner.active

    def _require_descriptor(self, *, write: bool = False) -> int:
        if not self._owner.active:
            raise ValueError("finite candidate capability is revoked")
        if write and not self._owner.writable:
            raise OSError("finite candidate capability is read-only")
        return self._owner.descriptor

    def readable(self) -> bool:
        return not self.closed

    def writable(self) -> bool:
        with self._owner.lock:
            return self._owner.active and self._owner.writable

    def seekable(self) -> bool:
        return not self.closed

    def read(self, size: int) -> bytes:
        if type(size) is not int or size < 0:
            raise ValueError("finite candidate reads require a bounded size")
        with self._owner.lock:
            return os.read(self._require_descriptor(), size)

    def readinto(self, buffer) -> int:
        view = memoryview(buffer).cast("B")
        with self._owner.lock:
            block = os.read(self._require_descriptor(), len(view))
            view[:len(block)] = block
            return len(block)

    def write(self, data) -> int:
        view = memoryview(data).cast("B")
        with self._owner.lock:
            descriptor = self._require_descriptor(write=True)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("finite candidate write made no progress")
                written += count
            return written

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if type(offset) is not int or type(whence) is not int:
            raise TypeError("finite candidate seek arguments must be exact integers")
        with self._owner.lock:
            return os.lseek(self._require_descriptor(), offset, whence)

    def tell(self) -> int:
        return self.seek(0, os.SEEK_CUR)

    def truncate(self, size: int | None = None) -> int:
        with self._owner.lock:
            descriptor = self._require_descriptor(write=True)
            selected = os.lseek(descriptor, 0, os.SEEK_CUR) if size is None else size
            if type(selected) is not int or selected < 0:
                raise ValueError("finite candidate truncate size is invalid")
            os.ftruncate(descriptor, selected)
            return selected

    def flush(self) -> None:
        # HDF5's file-object driver calls this after it has emitted its own
        # buffers.  The publisher performs the one durability fsync after the
        # document context has closed and this capability has been revoked.
        with self._owner.lock:
            self._require_descriptor()

    def __copy__(self):
        raise TypeError("finite candidate binding is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("finite candidate binding is not copyable")

    def __reduce__(self):
        raise TypeError("finite candidate binding is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("finite candidate binding is not serializable")


@dataclass(eq=False, frozen=True, slots=True)
class FiniteSeedBinding:
    """Publisher-owned proof joining one byte-exact seed to its candidate."""

    request: FiniteArtifactRequest
    receipt: FiniteSourceSeedReceipt
    candidate_identity: str
    _owner: _BindingOwner
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _FACTORY
            or type(self.request) is not FiniteArtifactRequest
            or type(self.receipt) is not FiniteSourceSeedReceipt
            or self.receipt.source_snapshot
            != self.request.operation_context.source_snapshot
            or type(self.candidate_identity) is not str
            or not _LOWER_HEX_64.fullmatch(self.candidate_identity)
            or type(self._owner) is not _BindingOwner
            or not self._owner.active
        ):
            raise TypeError("finite seed binding is not publisher-owned")

    @property
    def active(self) -> bool:
        with self._owner.lock:
            return self._owner.active

    def authorizes(self, candidate: FiniteCandidateBinding) -> bool:
        """Prove one candidate capability shares this exact publisher owner."""

        return (
            type(candidate) is FiniteCandidateBinding
            and candidate.request is self.request
            and candidate.candidate_identity == self.candidate_identity
            and candidate._owner is self._owner
            and self.active
            and not candidate.closed
        )

    def __copy__(self):
        raise TypeError("finite seed binding is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("finite seed binding is not copyable")

    def __reduce__(self):
        raise TypeError("finite seed binding is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("finite seed binding is not serializable")


@dataclass(frozen=True, slots=True)
class FiniteDocumentAdapter:
    """Trusted publisher-owned sessions over one revocable private file.

    Adapters must use only the public ``FiniteCandidateBinding`` file-object
    methods.  They must not inspect underscored binding state, duplicate or
    export the backing descriptor, retain native handles past their context,
    reopen a candidate by path, or spawn path-based writers.  The publisher
    revokes ordinary bindings and closes conforming document contexts; it does
    not claim process isolation from deliberately hostile Python code.
    """

    open_writer: Callable[[FiniteCandidateBinding], ContextManager[object]]
    write: Callable[[object], object]
    open_reader: Callable[[FiniteCandidateBinding], ContextManager[object]]
    validate: Callable[[object], FiniteCandidateValidation]

    def __post_init__(self) -> None:
        if any(
            not callable(value)
            for value in (
                self.open_writer,
                self.write,
                self.open_reader,
                self.validate,
            )
        ):
            raise TypeError("finite document adapter fields must be callable")


@dataclass(frozen=True, slots=True)
class FiniteSeededDocumentAdapter:
    """Seed-aware adapter whose writer receives publisher-minted seed proof."""

    open_writer: Callable[[FiniteCandidateBinding], ContextManager[object]]
    write: Callable[
        [object, FiniteSeedBinding, FiniteCandidateBinding], object
    ]
    open_reader: Callable[[FiniteCandidateBinding], ContextManager[object]]
    validate: Callable[[object], FiniteCandidateValidation]

    def __post_init__(self) -> None:
        if any(
            not callable(value)
            for value in (
                self.open_writer,
                self.write,
                self.open_reader,
                self.validate,
            )
        ):
            raise TypeError("finite seeded document adapter fields must be callable")


@dataclass(frozen=True, slots=True)
class FiniteArtifactResult:
    disposition: FiniteArtifactDisposition
    request: FiniteArtifactRequest
    terminal: StreamTerminal | None
    seed_receipt: FiniteSourceSeedReceipt | None
    commit_identity: str | None
    hidden_orphan: str | None
    diagnostics: tuple[str, ...]
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        committed = self.disposition in {
            FiniteArtifactDisposition.COMMITTED,
            FiniteArtifactDisposition.ALREADY_COMMITTED,
        }
        if (
            _claim is not _FACTORY
            or type(self.request) is not FiniteArtifactRequest
            or committed != (type(self.terminal) is StreamTerminal)
            or committed != (
                type(self.commit_identity) is str
                and bool(_LOWER_HEX_64.fullmatch(self.commit_identity or ""))
            )
            or committed
            and self.commit_identity != _commit_identity(
                self.request, self.terminal,
            )
            or self.seed_receipt is not None
            and type(self.seed_receipt) is not FiniteSourceSeedReceipt
            or type(self.diagnostics) is not tuple
            or any(type(item) is not str or len(item) > 1024 for item in self.diagnostics)
            or self.hidden_orphan is not None
            and (type(self.hidden_orphan) is not str or not self.hidden_orphan)
        ):
            raise TypeError("finite artifact result is invalid")


def _canonical(payload: dict[str, object]) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        raw = encoded.encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("finite canonical identity input is invalid") from error
    if len(raw) > _MAX_CANONICAL_BYTES:
        raise ValueError("finite canonical identity input is oversized")
    return encoded


def _identity(payload: dict[str, object]) -> tuple[str, str]:
    canonical = _canonical(payload)
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _operation_context_payload(
    context: FiniteOperationContext,
) -> dict[str, object]:
    snapshot = context.source_snapshot
    return {
        "custody_identity": context.custody_identity,
        "domain": "xdart.finite-artifact-operation-context.v1",
        "request_generation_identity": context.request_generation_identity,
        "resource_allocation_identity": context.resource_allocation_identity,
        "route_identity": context.route_identity,
        "source_snapshot": {
            "ctime_ns": snapshot.ctime_ns,
            "device": snapshot.device,
            "digest": snapshot.digest,
            "inode": snapshot.inode,
            "mode": snapshot.mode,
            "mtime_ns": snapshot.mtime_ns,
            "size": snapshot.size,
        },
    }


def finite_operation_context(
    source: FiniteSourceAdmission,
    *,
    request_generation_identity: str,
    resource_allocation_identity: str,
    route_identity: str,
    custody_identity: str,
) -> FiniteOperationContext:
    if type(source) is not FiniteSourceAdmission:
        raise TypeError("finite operation context requires source admission")
    values = tuple(
        _hex_identity(value, role)
        for role, value in (
            ("request generation identity", request_generation_identity),
            ("resource allocation identity", resource_allocation_identity),
            ("route identity", route_identity),
            ("custody identity", custody_identity),
        )
    )
    return _new_operation_context(source.snapshot, *values)


def _new_operation_context(
    snapshot: FiniteFileSnapshot,
    request_generation_identity: str,
    resource_allocation_identity: str,
    route_identity: str,
    custody_identity: str,
) -> FiniteOperationContext:
    payload = {
        "custody_identity": custody_identity,
        "domain": "xdart.finite-artifact-operation-context.v1",
        "request_generation_identity": request_generation_identity,
        "resource_allocation_identity": resource_allocation_identity,
        "route_identity": route_identity,
        "source_snapshot": {
            "ctime_ns": snapshot.ctime_ns,
            "device": snapshot.device,
            "digest": snapshot.digest,
            "inode": snapshot.inode,
            "mode": snapshot.mode,
            "mtime_ns": snapshot.mtime_ns,
            "size": snapshot.size,
        },
    }
    canonical, identity = _identity(payload)
    return FiniteOperationContext(
        snapshot,
        request_generation_identity,
        resource_allocation_identity,
        route_identity,
        custody_identity,
        canonical,
        identity,
    )


def capture_finite_predecessor(
    source: FiniteSourceAdmission,
    *,
    terminal: StreamTerminal | None = None,
    artifact_family_v1: str | None = None,
    version_identity: str | None = None,
    publication_identity: str | None = None,
    lineage_identity: str | None = None,
) -> FinitePredecessorReceipt:
    if type(source) is not FiniteSourceAdmission:
        raise TypeError("finite predecessor requires source admission")
    return FinitePredecessorReceipt(
        source.snapshot,
        terminal,
        artifact_family_v1,
        version_identity,
        publication_identity,
        lineage_identity,
    )


def _terminal_lineage_payload(
    terminal: StreamTerminal | None,
) -> dict[str, object] | None:
    if terminal is None:
        return None
    return {
        "ctime_ns": terminal.ctime_ns,
        "device": terminal.device,
        "digest": terminal.digest,
        "inode": terminal.inode,
        "mtime_ns": terminal.mtime_ns,
        "size": terminal.size,
        "target": terminal.target,
    }


def _lineage_payload(
    *,
    artifact_family: str,
    operation_kind: str,
    source_graph_identity: str,
    entry: str,
    scientific_identity: str,
    output_schema: str,
    algorithm_identity: str,
    preservation_identity: str,
    publication_identity: str,
    version_identity: str,
    predecessor: FinitePredecessorReceipt,
) -> dict[str, object]:
    source = predecessor.source_snapshot
    return {
        "algorithm_identity": algorithm_identity,
        "artifact_family_v1": artifact_family,
        "entry": entry,
        "operation_kind": operation_kind,
        "output_schema": output_schema,
        "predecessor": {
            "artifact_family_v1": predecessor.artifact_family_v1,
            "lineage_identity": predecessor.lineage_identity,
            "publication_identity": predecessor.publication_identity,
            "source_artifact": source.path,
            "source_digest": source.digest,
            "source_size": source.size,
            "terminal": _terminal_lineage_payload(predecessor.terminal),
            "version_identity": predecessor.version_identity,
        },
        "preservation_identity": preservation_identity,
        "publication_identity": publication_identity,
        "publication_policy": FINITE_PUBLICATION_POLICY,
        "schema": FINITE_LINEAGE_SCHEMA,
        "scientific_identity": scientific_identity,
        "source_graph_identity": source_graph_identity,
        "version_identity": version_identity,
    }


def _lineage_for_request_fields(**values) -> FiniteArtifactLineage:
    canonical, identity = _identity(_lineage_payload(**values))
    return FiniteArtifactLineage(canonical, identity, _FACTORY)


def admit_finite_artifact_lineage(value: str | bytes) -> FiniteArtifactLineage:
    if isinstance(value, bytes):
        raw = value
        try:
            canonical = raw.decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ValueError("finite lineage is not strict UTF-8") from error
    elif type(value) is str:
        canonical = value
        try:
            raw = canonical.encode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ValueError("finite lineage is not strict UTF-8") from error
    else:
        raise TypeError("finite lineage must be text or bytes")
    if len(raw) > FINITE_LINEAGE_MAX_BYTES:
        raise ValueError("finite lineage exceeds its encoded ceiling")
    try:
        payload = json.loads(canonical)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("finite lineage JSON is invalid") from error
    own_keys = {
        "algorithm_identity",
        "artifact_family_v1",
        "entry",
        "operation_kind",
        "output_schema",
        "predecessor",
        "preservation_identity",
        "publication_identity",
        "publication_policy",
        "schema",
        "scientific_identity",
        "source_graph_identity",
        "version_identity",
    }
    predecessor_keys = {
        "artifact_family_v1",
        "lineage_identity",
        "publication_identity",
        "source_artifact",
        "source_digest",
        "source_size",
        "terminal",
        "version_identity",
    }
    terminal_keys = {
        "ctime_ns",
        "device",
        "digest",
        "inode",
        "mtime_ns",
        "size",
        "target",
    }
    predecessor = payload.get("predecessor") if type(payload) is dict else None
    terminal = predecessor.get("terminal") if type(predecessor) is dict else None
    own_hashes = (
        payload.get("algorithm_identity") if type(payload) is dict else None,
        payload.get("preservation_identity") if type(payload) is dict else None,
        payload.get("publication_identity") if type(payload) is dict else None,
        payload.get("scientific_identity") if type(payload) is dict else None,
        payload.get("source_graph_identity") if type(payload) is dict else None,
        payload.get("version_identity") if type(payload) is dict else None,
    )
    predecessor_finite = (
        predecessor.get("artifact_family_v1") if type(predecessor) is dict else None,
        predecessor.get("lineage_identity") if type(predecessor) is dict else None,
        predecessor.get("publication_identity") if type(predecessor) is dict else None,
        predecessor.get("version_identity") if type(predecessor) is dict else None,
    )
    source_artifact = (
        predecessor.get("source_artifact") if type(predecessor) is dict else None
    )
    source_size = predecessor.get("source_size") if type(predecessor) is dict else None
    if (
        type(payload) is not dict
        or set(payload) != own_keys
        or type(predecessor) is not dict
        or set(predecessor) != predecessor_keys
        or payload.get("schema") != FINITE_LINEAGE_SCHEMA
        or payload.get("publication_policy") != FINITE_PUBLICATION_POLICY
        or type(payload.get("artifact_family_v1")) is not str
        or not _ARTIFACT_FAMILY.fullmatch(payload["artifact_family_v1"])
        or type(payload.get("operation_kind")) is not str
        or not _OPERATION_KIND.fullmatch(payload["operation_kind"])
        or type(payload.get("output_schema")) is not str
        or not _SCHEMA_TOKEN.fullmatch(payload["output_schema"])
        or any(type(item) is not str or not _LOWER_HEX_64.fullmatch(item)
               for item in own_hashes)
        or type(payload.get("entry")) is not str
        or not payload["entry"]
        or "/" in payload["entry"]
        or payload["entry"] in {".", ".."}
        or len(payload["entry"].encode("utf-8", errors="strict"))
        > _MAX_ENTRY_BYTES
        or type(source_artifact) is not str
        or not os.path.isabs(source_artifact)
        or "\x00" in source_artifact
        or len(source_artifact.encode("utf-8", errors="strict")) > _MAX_PATH_BYTES
        or type(source_size) is not int
        or not 0 <= source_size <= _I64_MAX
        or type(predecessor.get("source_digest")) is not str
        or not _LOWER_HEX_64.fullmatch(predecessor["source_digest"])
        or not (
            all(item is None for item in predecessor_finite)
            or (
                type(predecessor_finite[0]) is str
                and _ARTIFACT_FAMILY.fullmatch(predecessor_finite[0])
                and all(
                    type(item) is str and _LOWER_HEX_64.fullmatch(item)
                    for item in predecessor_finite[1:]
                )
            )
        )
        or terminal is not None
        and (
            type(terminal) is not dict
            or set(terminal) != terminal_keys
            or any(
                type(terminal.get(key)) is not int
                or not 0 <= terminal[key] <= _I64_MAX
                for key in ("ctime_ns", "device", "inode", "mtime_ns", "size")
            )
            or type(terminal.get("digest")) is not str
            or not _LOWER_HEX_64.fullmatch(terminal["digest"])
            or terminal.get("target") != source_artifact
            or terminal.get("digest") != predecessor.get("source_digest")
            or terminal.get("size") != source_size
        )
        or _canonical(payload) != canonical
    ):
        raise ValueError("finite lineage schema or canonical form is invalid")
    identity = hashlib.sha256(raw).hexdigest()
    return FiniteArtifactLineage(canonical, identity, _FACTORY)


def finite_lineage_hdf_path(entry: str) -> str:
    if (
        type(entry) is not str
        or not entry
        or "\x00" in entry
        or "/" in entry
        or entry in {".", ".."}
        or len(entry.encode("utf-8", errors="strict")) > _MAX_ENTRY_BYTES
    ):
        raise ValueError("finite lineage entry is invalid")
    return f"/{entry}/reduction/config/{FINITE_LINEAGE_NODE_NAME}"


def _lineage_group(parent, name: str, *, create: bool):
    import h5py

    link = parent.get(name, getlink=True)
    if link is None:
        if not create:
            raise FiniteArtifactIntegrityError(
                f"finite lineage group is missing: {name}"
            )
        return parent.create_group(name)
    if type(link) is not h5py.HardLink:
        raise FiniteArtifactIntegrityError(
            f"finite lineage group link is not local hard: {name}"
        )
    value = parent.get(name)
    if not isinstance(value, h5py.Group):
        raise FiniteArtifactIntegrityError(
            f"finite lineage group has the wrong object type: {name}"
        )
    return value


def write_finite_artifact_lineage(document, request: FiniteArtifactRequest) -> None:
    """Persist the exact deterministic lineage in a private HDF5 candidate."""

    import h5py

    if type(request) is not FiniteArtifactRequest:
        raise TypeError("finite lineage writer requires an exact request")
    entry = _lineage_group(document, request.entry, create=True)
    reduction = _lineage_group(entry, "reduction", create=True)
    config = _lineage_group(reduction, "config", create=True)
    existing_link = config.get(FINITE_LINEAGE_NODE_NAME, getlink=True)
    if existing_link is not None:
        existing = config.get(FINITE_LINEAGE_NODE_NAME)
        if type(existing_link) is not h5py.HardLink or not isinstance(
            existing,
            h5py.Dataset,
        ):
            raise FiniteArtifactIntegrityError(
                "finite lineage writer refuses a nonlocal or non-dataset node"
            )
        del config[FINITE_LINEAGE_NODE_NAME]
    raw = request.lineage.canonical_json.encode("utf-8", errors="strict")
    node = config.create_dataset(
        FINITE_LINEAGE_NODE_NAME,
        shape=(len(raw),),
        dtype="u1",
    )
    node[...] = memoryview(raw)


def require_finite_artifact_lineage(
    document,
    request: FiniteArtifactRequest | None = None,
    *,
    entry: str | None = None,
) -> FiniteArtifactLineage:
    """Read one exact local scalar lineage node and optionally match a request."""

    import h5py

    if (
        request is not None
        and type(request) is not FiniteArtifactRequest
        or request is not None
        and entry is not None
    ):
        raise TypeError("finite lineage reader request is invalid")
    if entry is not None:
        finite_lineage_hdf_path(entry)
    entry_name = request.entry if request is not None else entry
    if entry_name is None:
        candidates = []
        for name in document:
            link = document.get(name, getlink=True)
            value = document.get(name)
            if type(link) is h5py.HardLink and isinstance(value, h5py.Group):
                if value.attrs.get("NX_class") in {"NXentry", b"NXentry"}:
                    candidates.append(name)
        if len(candidates) != 1:
            raise FiniteArtifactIntegrityError(
                "finite lineage reader requires one exact NXentry"
            )
        entry_name = candidates[0]
    entry = _lineage_group(document, entry_name, create=False)
    reduction = _lineage_group(entry, "reduction", create=False)
    config = _lineage_group(reduction, "config", create=False)
    link = config.get(FINITE_LINEAGE_NODE_NAME, getlink=True)
    node = config.get(FINITE_LINEAGE_NODE_NAME)
    if (
        type(link) is not h5py.HardLink
        or not isinstance(node, h5py.Dataset)
        or len(node.shape) != 1
        or not 0 < node.shape[0] <= FINITE_LINEAGE_MAX_BYTES
        or node.maxshape != node.shape
        or node.chunks is not None
        or node.compression is not None
        or node.shuffle
        or node.fletcher32
        or node.scaleoffset is not None
        or node.is_virtual
        or len(node.attrs) != 0
        or node.dtype.kind != "u"
        or node.dtype.itemsize != 1
        or node.id.get_create_plist().get_external_count() != 0
    ):
        raise FiniteArtifactIntegrityError(
            "finite lineage node layout is not the exact local scalar schema"
        )
    value = bytes(node[...])
    lineage = admit_finite_artifact_lineage(value)
    if request is not None and lineage != request.lineage:
        raise FiniteArtifactIntegrityError(
            "finite lineage node does not match the requested successor"
        )
    return lineage


def _hex_identity(value: object, role: str) -> str:
    if type(value) is not str or not _LOWER_HEX_64.fullmatch(value):
        raise ValueError(f"{role} must be a lowercase SHA-256 identity")
    return value


def _resolved_parent_path(path: Path | str) -> str:
    try:
        shown = os.fspath(path)
    except TypeError as error:
        raise TypeError("finite path must be path-like") from error
    if type(shown) is not str or not shown or "\x00" in shown:
        raise ValueError("finite path is invalid")
    absolute = os.path.abspath(os.path.normpath(shown))
    resolved = os.path.join(os.path.realpath(os.path.dirname(absolute)), os.path.basename(absolute))
    if len(resolved.encode("utf-8")) > _MAX_PATH_BYTES:
        raise ValueError("finite path exceeds the UTF-8 bound")
    return os.path.normcase(resolved)


def finite_artifact_request(
    *,
    source_admission: FiniteSourceAdmission,
    operation_context: FiniteOperationContext,
    predecessor: FinitePredecessorReceipt,
    destination_directory: Path | str,
    operation_kind: str,
    source_graph_identity: str,
    entry: str,
    scientific_identity: str,
    output_schema: str,
    algorithm_identity: str,
    preservation_identity: str,
    artifact_family: str | None = None,
    explicit_target: Path | str | None = None,
) -> FiniteArtifactRequest:
    """Freeze deterministic scientific, publication, and execution identities."""

    if type(source_admission) is not FiniteSourceAdmission:
        raise TypeError("finite request requires exact source admission")
    if type(operation_context) is not FiniteOperationContext:
        raise TypeError("finite request requires exact operation context")
    if type(predecessor) is not FinitePredecessorReceipt:
        raise TypeError("finite request requires exact predecessor receipt")
    if (
        operation_context.source_snapshot != source_admission.snapshot
        or predecessor.source_snapshot != source_admission.snapshot
    ):
        raise ValueError("finite request custody receipts disagree")
    source = source_admission.path
    family = artifact_family_from_source(source, artifact_family)
    if type(operation_kind) is not str or not _OPERATION_KIND.fullmatch(operation_kind):
        raise ValueError("finite operation kind is invalid")
    if (
        type(entry) is not str
        or not entry
        or "\x00" in entry
        or "/" in entry
        or entry in {".", ".."}
        or len(entry.encode("utf-8", errors="strict")) > _MAX_ENTRY_BYTES
    ):
        raise ValueError("finite entry is invalid")
    if type(output_schema) is not str or not _SCHEMA_TOKEN.fullmatch(output_schema):
        raise ValueError("finite output schema is invalid")
    source_graph = _hex_identity(source_graph_identity, "source graph identity")
    science = _hex_identity(scientific_identity, "scientific identity")
    algorithm = _hex_identity(algorithm_identity, "algorithm identity")
    preservation = _hex_identity(preservation_identity, "preservation identity")
    version_json, version = _identity({
        "algorithm_identity": algorithm,
        "domain": "xdart.finite-artifact-version.v1",
        "entry": entry,
        "operation_kind": operation_kind,
        "output_schema": output_schema,
        "preservation_identity": preservation,
        "scientific_identity": science,
        "source_graph_identity": source_graph,
    })
    try:
        root = Path(os.fspath(destination_directory))
    except TypeError as error:
        raise TypeError("finite output directory must be path-like") from error
    resolved_root = Path(os.path.realpath(os.path.abspath(os.path.normpath(root))))
    if not resolved_root.is_dir():
        raise ValueError("finite output directory must already exist")
    # An explicit target no longer NAMES the artifact.  Under ADR-0010 the
    # public name is the stable slot, so honouring a caller's filename would
    # reopen the very bypass the closed vocabulary exists to shut.  Its
    # DIRECTORY constraint still binds: silently dropping the check would let a
    # caller aim at another directory and be quietly redirected there instead
    # of refused, which is worse than either honouring or rejecting it.
    if explicit_target is not None and os.fspath(explicit_target) != "":
        requested = Path(_resolved_parent_path(explicit_target))
        if requested.parent != Path(os.path.normcase(os.fspath(resolved_root))):
            raise ValueError(
                "finite explicit target must share the destination directory"
            )
    target = resolve_finite_output_target(
        resolved_root,
        family,
        operation_token=operation_kind,
    )
    output = _resolved_parent_path(target)
    if source == output:
        raise ValueError("finite source and output must be distinct")
    _publication_json, publication = _identity({
        "domain": "xdart.finite-artifact-publication.v1",
        "output_artifact": output,
        "version_identity": version,
    })
    _operation_json, operation = _identity({
        "domain": "xdart.finite-artifact-operation.v1",
        "operation_context_identity": operation_context.context_identity,
        "output_artifact": output,
        "publication_identity": publication,
        "source_artifact": source,
        "version_identity": version,
    })
    lineage = _lineage_for_request_fields(
        artifact_family=family,
        operation_kind=operation_kind,
        source_graph_identity=source_graph,
        entry=entry,
        scientific_identity=science,
        output_schema=output_schema,
        algorithm_identity=algorithm,
        preservation_identity=preservation,
        publication_identity=publication,
        version_identity=version,
        predecessor=predecessor,
    )
    if (
        predecessor.version_identity == version
        or predecessor.publication_identity == publication
        or predecessor.lineage_identity == lineage.lineage_identity
    ):
        raise ValueError("finite successor lineage cannot self-reference")
    return FiniteArtifactRequest(
        source,
        output,
        family,
        operation_kind,
        source_graph,
        entry,
        science,
        output_schema,
        algorithm,
        preservation,
        operation_context,
        predecessor,
        lineage,
        version_json,
        version,
        publication,
        operation,
    )


def _replay_finite_artifact_request(
    *,
    source_admission: FiniteSourceAdmission,
    operation_context: FiniteOperationContext,
    predecessor: FinitePredecessorReceipt,
    output_artifact: Path | str,
    artifact_family: str,
    operation_kind: str,
    source_graph_identity: str,
    entry: str,
    scientific_identity: str,
    output_schema: str,
    algorithm_identity: str,
    preservation_identity: str,
    expected_lineage: FiniteArtifactLineage,
    expected_version_identity: str,
    expected_publication_identity: str,
    expected_operation_identity: str,
) -> FiniteArtifactRequest:
    """Reconstruct one frozen request without re-running occupied-path policy."""

    if (
        type(source_admission) is not FiniteSourceAdmission
        or type(operation_context) is not FiniteOperationContext
        or type(predecessor) is not FinitePredecessorReceipt
        or type(expected_lineage) is not FiniteArtifactLineage
        or operation_context.source_snapshot != source_admission.snapshot
        or predecessor.source_snapshot != source_admission.snapshot
    ):
        raise TypeError("finite request replay requires exact custody receipts")
    source = source_admission.path
    output = _resolved_parent_path(output_artifact)
    if not Path(output).parent.is_dir():
        raise ValueError("finite replay output directory must already exist")
    family = artifact_family_from_source(source, artifact_family)
    source_graph = _hex_identity(source_graph_identity, "source graph identity")
    science = _hex_identity(scientific_identity, "scientific identity")
    algorithm = _hex_identity(algorithm_identity, "algorithm identity")
    preservation = _hex_identity(preservation_identity, "preservation identity")
    version_json, version = _identity({
        "algorithm_identity": algorithm,
        "domain": "xdart.finite-artifact-version.v1",
        "entry": entry,
        "operation_kind": operation_kind,
        "output_schema": output_schema,
        "preservation_identity": preservation,
        "scientific_identity": science,
        "source_graph_identity": source_graph,
    })
    _publication_json, publication = _identity({
        "domain": "xdart.finite-artifact-publication.v1",
        "output_artifact": output,
        "version_identity": version,
    })
    _operation_json, operation = _identity({
        "domain": "xdart.finite-artifact-operation.v1",
        "operation_context_identity": operation_context.context_identity,
        "output_artifact": output,
        "publication_identity": publication,
        "source_artifact": source,
        "version_identity": version,
    })
    lineage = _lineage_for_request_fields(
        artifact_family=family,
        operation_kind=operation_kind,
        source_graph_identity=source_graph,
        entry=entry,
        scientific_identity=science,
        output_schema=output_schema,
        algorithm_identity=algorithm,
        preservation_identity=preservation,
        publication_identity=publication,
        version_identity=version,
        predecessor=predecessor,
    )
    if (
        version != expected_version_identity
        or publication != expected_publication_identity
        or operation != expected_operation_identity
        or lineage != expected_lineage
    ):
        raise ValueError("finite request replay identities changed")
    return FiniteArtifactRequest(
        source,
        output,
        family,
        operation_kind,
        source_graph,
        entry,
        science,
        output_schema,
        algorithm,
        preservation,
        operation_context,
        predecessor,
        lineage,
        version_json,
        version,
        publication,
        operation,
    )


def _state(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _entry_stat(
    normalized: str,
    parent_descriptor: int | None,
    name: str | None,
) -> os.stat_result:
    if parent_descriptor is None:
        return os.lstat(normalized)
    if type(name) is not str or not name or os.path.basename(name) != name:
        raise ValueError("finite directory entry name is invalid")
    return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)


def _entry_open(
    normalized: str,
    parent_descriptor: int | None,
    name: str | None,
) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    if parent_descriptor is None:
        return os.open(normalized, flags)
    return os.open(name, flags, dir_fd=parent_descriptor)


def _capture_regular(
    shown_path: Path | str,
    *,
    parent_descriptor: int | None = None,
    name: str | None = None,
    hash_content: bool,
) -> tuple[tuple[int, int, int, int, int, int], str | None]:
    normalized = _resolved_parent_path(shown_path)
    try:
        lexical_before = _entry_stat(normalized, parent_descriptor, name)
        if stat.S_ISLNK(lexical_before.st_mode):
            raise FiniteArtifactIntegrityError(
                "finite file symbolic links are refused"
            )
        descriptor = _entry_open(normalized, parent_descriptor, name)
    except FiniteArtifactIntegrityError:
        raise
    except OSError as error:
        raise FiniteArtifactIntegrityError(
            f"finite file is unavailable: {normalized}"
        ) from error
    digest = hashlib.sha256() if hash_content else None
    primary: BaseException | None = None
    opened: os.stat_result | None = None
    finished: os.stat_result | None = None
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise FiniteArtifactIntegrityError("finite file must be regular")
        if digest is not None:
            while True:
                block = os.read(descriptor, _COPY_BLOCK_BYTES)
                if not block:
                    break
                digest.update(block)
        finished = os.fstat(descriptor)
    except BaseException as error:
        primary = error
    close_error = _close_descriptor_once(descriptor)
    if primary is not None:
        if close_error is not None:
            _attach_secondary_close(
                primary, close_error, "finite regular-file observation",
            )
        raise primary.with_traceback(primary.__traceback__)
    if close_error is not None:
        raise FiniteArtifactIntegrityError(
            f"finite file close is incomplete: {normalized}"
        ) from close_error
    if opened is None or finished is None:
        raise RuntimeError("finite file observation lost its descriptor state")
    try:
        lexical_after = _entry_stat(normalized, parent_descriptor, name)
    except OSError as error:
        raise FiniteArtifactIntegrityError(
            f"finite file disappeared: {normalized}"
        ) from error
    observations = {
        _state(lexical_before),
        _state(opened),
        _state(finished),
        _state(lexical_after),
    }
    if len(observations) != 1 or stat.S_ISLNK(lexical_after.st_mode):
        raise FiniteArtifactIntegrityError(
            f"finite file changed during observation: {normalized}"
        )
    return observations.pop(), None if digest is None else digest.hexdigest()


def _snapshot_from_capture(
    shown_path: Path | str,
    state: tuple[int, int, int, int, int, int],
    digest: str | None,
) -> FiniteFileSnapshot:
    if digest is None:
        raise FiniteArtifactIntegrityError("finite snapshot lacks a digest")
    return FiniteFileSnapshot(
        _resolved_parent_path(shown_path),
        state[3],
        digest,
        state[0],
        state[1],
        state[2],
        state[4],
        state[5],
    )


def _snapshot_path(path: Path | str) -> FiniteFileSnapshot:
    return _snapshot_from_capture(
        path,
        *_capture_regular(path, hash_content=True),
    )


def _snapshot_at(
    parent_descriptor: int,
    name: str,
    shown_path: Path | str,
) -> FiniteFileSnapshot:
    return _snapshot_from_capture(
        shown_path,
        *_capture_regular(
            shown_path,
            parent_descriptor=parent_descriptor,
            name=name,
            hash_content=True,
        ),
    )


def capture_finite_source(path: Path | str) -> FiniteSourceAdmission:
    return FiniteSourceAdmission(_snapshot_path(path), _FACTORY)


def _same_object(left: FiniteFileSnapshot, right: FiniteFileSnapshot) -> bool:
    return (left.device, left.inode) == (right.device, right.inode)


def _snapshot_state(snapshot: FiniteFileSnapshot) -> tuple[int, int, int, int, int, int]:
    return (
        snapshot.device,
        snapshot.inode,
        snapshot.mode,
        snapshot.size,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
    )


def _observe_regular(path: Path | str) -> tuple[int, int, int, int, int, int]:
    state, _digest_value = _capture_regular(path, hash_content=False)
    return state


def _observe_regular_at(
    parent_descriptor: int,
    name: str,
    shown_path: Path | str,
) -> tuple[int, int, int, int, int, int]:
    state, _digest_value = _capture_regular(
        shown_path,
        parent_descriptor=parent_descriptor,
        name=name,
        hash_content=False,
    )
    return state


def _try_observe(path: Path | str) -> tuple[int, int, int, int, int, int] | None:
    normalized = _resolved_parent_path(path)
    try:
        os.lstat(normalized)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise FiniteArtifactIntegrityError(
            f"finite namespace cannot be inspected: {normalized}"
        ) from error
    return _observe_regular(normalized)


def _try_observe_at(
    parent_descriptor: int,
    name: str,
    shown_path: Path | str,
) -> tuple[int, int, int, int, int, int] | None:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise FiniteArtifactIntegrityError(
            f"finite namespace cannot be inspected: {shown_path}"
        ) from error
    return _observe_regular_at(parent_descriptor, name, shown_path)


def _require_source(admission: FiniteSourceAdmission) -> FiniteFileSnapshot:
    if type(admission) is not FiniteSourceAdmission:
        raise TypeError("finite seed requires an exact source admission")
    if _observe_regular(admission.path) != _snapshot_state(admission.snapshot):
        raise FiniteArtifactIntegrityError("finite source changed after admission")
    return admission.snapshot


def _unlink_candidate(
    parent_descriptor: int,
    path: Path,
    expected: FiniteFileSnapshot,
) -> None:
    """Remove one observed owned alias in a cooperative private namespace.

    POSIX has no compare-inode-and-unlink syscall.  The destination directory
    is therefore an authority boundary: cooperating publishers and trusted
    adapters may use it concurrently, but an actor with the same filesystem
    credentials must not replace this randomized private name between the
    ownership observation and ``unlinkat``.  Public final names are never
    removed by this helper.
    """

    try:
        current = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (int(current.st_dev), int(current.st_ino))
        != (expected.device, expected.inode)
    ):
        raise FiniteArtifactIntegrityError(
            f"candidate cleanup refused a foreign object: {path}"
        )
    try:
        os.unlink(path.name, dir_fd=parent_descriptor)
    except FileNotFoundError:
        return


def _bounded_diagnostic(prefix: str, error: BaseException) -> str:
    name = type(error).__name__[:128]
    try:
        detail = str(error).encode("utf-8")[:768].decode("utf-8", "ignore")
    except BaseException:
        detail = name
    return f"{prefix}:{name}:{detail}"[:1024]


def _close_descriptor_once(descriptor: int) -> BaseException | None:
    """Close exactly once; POSIX close errors have ambiguous descriptor state."""

    try:
        os.close(descriptor)
    except BaseException as error:
        return error
    return None


def _attach_secondary_close(
    primary: BaseException,
    close_error: BaseException,
    role: str,
) -> None:
    try:
        primary.add_note(
            _bounded_diagnostic(FINITE_DESCRIPTOR_CLOSE_WARNING, close_error)
            + f":{role}"
        )
    except BaseException:
        pass


def _commit_identity(
    request: FiniteArtifactRequest,
    terminal: StreamTerminal,
) -> str:
    _canonical_commit, identity = _identity({
        "digest": terminal.digest,
        "domain": "xdart.finite-artifact-commit.v1",
        "inode": terminal.inode,
        "publication_identity": request.publication_identity,
        "size": terminal.size,
    })
    return identity


class FiniteArtifactPublisher:
    """One-shot local publisher for an immutable finite artifact."""

    def __init__(
        self,
        request: FiniteArtifactRequest,
        *,
        cancel_token: threading.Event | None = None,
    ) -> None:
        if type(request) is not FiniteArtifactRequest:
            raise TypeError("finite publisher requires an exact request")
        if cancel_token is not None and type(cancel_token) is not threading.Event:
            raise TypeError("finite cancellation token must be an exact Event")
        self.request = request
        self._cancel_token = cancel_token
        self._used = False
        self._lock = threading.RLock()
        self._slot_hold = None

    def __copy__(self):
        raise TypeError("finite publisher is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("finite publisher is not copyable")

    def _cancelled(self) -> bool:
        return self._cancel_token is not None and self._cancel_token.is_set()

    def _require_request_source(self) -> FiniteFileSnapshot:
        expected = self.request.operation_context.source_snapshot
        if _observe_regular(expected.path) != _snapshot_state(expected):
            raise FiniteArtifactIntegrityError(
                "finite request source changed after operation-context capture"
            )
        return expected

    def _acquire_slot(self) -> None:
        """Hold the H23 lease on the PUBLIC SLOT for this whole publication.

        Until 2026-09-04 the finite path held NO lease at all: `grep -rn
        "coordinator.admit\\|\\.acquire_lease(" src/` returned nothing in this
        module, in `reintegrate_successor.py` or in `scan_session.py`.  The
        no-clobber `os.link` at the end WAS the entire concurrency guard for
        Reintegrate, Stitch and RSM -- two concurrent operations on one slot
        were resolved by whichever linked first, and the loser was told the slot
        was already committed.

        That is why this lands BEFORE atomic replacement rather than with it.
        Replacement removes the link, and removing the link without this lease
        would turn a typed refusal into a silent last-writer-wins.  A second
        operation on the same slot now gets `LeaseUnavailable`, the same typed
        failure ordinary Run has always produced for a contended target.

        Held from before the first candidate byte (the parent is open but
        nothing is written yet) until `close_parent`, which every exit path
        runs.
        """
        self._slot_hold = get_output_transaction_coordinator().hold_target(
            self.request.output_artifact, label="finite-artifact",
        )

    def _release_slot(self) -> BaseException | None:
        """Give the slot back, once, on every exit path.

        Returns the first failure rather than raising: releasing a hold must
        never mask the outcome -- success or failure -- the caller came for.
        """
        hold = self._slot_hold
        if hold is None:
            return None
        self._slot_hold = None
        try:
            get_output_transaction_coordinator().release_target(hold)
        except BaseException as error:
            return error
        return None

    def _open_parent(self) -> tuple[int, tuple[int, int, int, int, int, int]]:
        parent = Path(self.request.output_artifact).parent
        try:
            named_before = os.lstat(parent)
        except OSError as error:
            raise FiniteArtifactIntegrityError(
                "finite output parent cannot be admitted"
            ) from error
        if stat.S_ISLNK(named_before.st_mode) or not stat.S_ISDIR(
            named_before.st_mode
        ):
            raise FiniteArtifactIntegrityError(
                "finite output parent must be a named non-symlink directory"
            )
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(parent, flags)
        except OSError as error:
            raise FiniteArtifactIntegrityError(
                "finite output parent cannot be opened without symlinks"
            ) from error
        try:
            opened = os.fstat(descriptor)
            named_after = os.lstat(parent)
        except BaseException as primary:
            close_error = _close_descriptor_once(descriptor)
            if close_error is not None:
                _attach_secondary_close(
                    primary, close_error, "output parent admission",
                )
            raise primary.with_traceback(primary.__traceback__)
        if (
            stat.S_ISLNK(named_after.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(named_after.st_mode)
            or _state(named_before)[:3] != _state(opened)[:3]
            or _state(opened)[:3] != _state(named_after)[:3]
        ):
            primary = FiniteArtifactIntegrityError(
                "finite output parent changed during admission"
            )
            close_error = _close_descriptor_once(descriptor)
            if close_error is not None:
                _attach_secondary_close(
                    primary, close_error, "output parent admission",
                )
            raise primary
        return descriptor, _state(opened)

    def _require_parent(
        self,
        descriptor: int,
        admitted: tuple[int, int, int, int, int, int],
    ) -> None:
        opened = os.fstat(descriptor)
        named = os.lstat(Path(self.request.output_artifact).parent)
        # Candidate creation and publication legitimately change directory
        # size/timestamps.  Custody is the still-named directory object and its
        # directory type, not an immutable directory-content snapshot.
        if (
            _state(opened)[:3] != admitted[:3]
            or _state(named)[:3] != admitted[:3]
            or stat.S_ISLNK(named.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(named.st_mode)
        ):
            raise FiniteArtifactIntegrityError("finite output parent changed")

    def _require_candidate_capacity(
        self,
        parent_descriptor: int,
        source: FiniteFileSnapshot,
    ) -> None:
        # A finite replacement starts as one complete source-sized private
        # copy.  Reserve ten percent for replacement metadata/results, with a
        # fixed 64 MiB floor so small inputs do not pass on a nearly full
        # filesystem.  Existing exact public versions bypass this budget.
        margin = max(
            _CANDIDATE_SPACE_MARGIN_BYTES,
            (source.size + 9) // 10,
        )
        required = source.size + margin
        observed = os.fstatvfs(parent_descriptor)
        fragment = int(observed.f_frsize)
        available = int(observed.f_bavail) * fragment
        if fragment <= 0 or available < 0:
            raise FiniteArtifactIntegrityError(
                "finite candidate filesystem capacity is unavailable"
            )
        if available < required:
            raise FiniteArtifactCapacityError(
                self.request,
                required_bytes=required,
                available_bytes=available,
            )

    def _reserve_candidate(self, parent_descriptor: int) -> FiniteFileSnapshot:
        name = (
            f".xdart-finite-{self.request.version_identity}-"
            f"{secrets.token_hex(16)}.candidate"
        )
        flags = (
            os.O_CREAT
            | os.O_EXCL
            | os.O_RDWR
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
        except FileExistsError as error:
            raise FiniteArtifactCollision(
                f"finite private candidate already exists: {name}"
            ) from error
        path = Path(self.request.output_artifact).parent / name
        provisional: FiniteFileSnapshot | None = None
        primary: BaseException | None = None
        close_cause: BaseException | None = None
        try:
            state = os.fstat(descriptor)
            provisional = FiniteFileSnapshot(
                str(path),
                int(state.st_size),
                hashlib.sha256(b"").hexdigest(),
                int(state.st_dev),
                int(state.st_ino),
                int(state.st_mode),
                int(state.st_mtime_ns),
                int(state.st_ctime_ns),
            )
            if not stat.S_ISREG(state.st_mode):
                raise FiniteArtifactIntegrityError(
                    "finite candidate reservation is not regular"
                )
        except BaseException as error:
            primary = error
        close_error = _close_descriptor_once(descriptor)
        if close_error is not None:
            if primary is None:
                primary = FiniteArtifactIntegrityError(
                    "finite candidate reservation close is incomplete"
                )
                close_cause = close_error
            else:
                _attach_secondary_close(
                    primary, close_error, "candidate reservation",
                )
        if primary is not None:
            if provisional is not None:
                try:
                    _unlink_candidate(parent_descriptor, path, provisional)
                except BaseException as cleanup_error:
                    try:
                        primary.add_note(_bounded_diagnostic(
                            FINITE_CANDIDATE_CLEANUP_WARNING, cleanup_error
                        ) + f":{path}")
                    except BaseException:
                        pass
            else:
                try:
                    primary.add_note(
                        f"{FINITE_CANDIDATE_CLEANUP_WARNING}:{path}"
                    )
                except BaseException:
                    pass
            if close_cause is not None:
                raise primary from close_cause
            raise primary.with_traceback(primary.__traceback__)
        if provisional is None:
            raise RuntimeError("finite reservation lost its provisional identity")
        try:
            captured = _snapshot_at(parent_descriptor, name, path)
            if not _same_object(captured, provisional):
                raise FiniteArtifactIntegrityError(
                    "finite candidate changed during reservation"
                )
            if stat.S_IMODE(captured.mode) != 0o600:
                raise FiniteArtifactIntegrityError("finite candidate mode is not 0600")
            return captured
        except BaseException as primary:
            try:
                _unlink_candidate(parent_descriptor, path, provisional)
            except BaseException as cleanup_error:
                try:
                    primary.add_note(_bounded_diagnostic(
                        FINITE_CANDIDATE_CLEANUP_WARNING, cleanup_error
                    ) + f":{path}")
                except BaseException:
                    pass
            raise

    def _open_binding(
        self,
        parent_descriptor: int,
        candidate: Path,
        expected: FiniteFileSnapshot,
        *,
        writable: bool,
    ) -> tuple[FiniteCandidateBinding, _BindingOwner]:
        flags = (
            os.O_RDWR if writable else os.O_RDONLY
        ) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(
                candidate.name,
                flags,
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise FiniteArtifactIntegrityError(
                "finite candidate capability cannot be opened"
            ) from error
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (int(opened.st_dev), int(opened.st_ino))
                != (expected.device, expected.inode)
            ):
                raise FiniteArtifactIntegrityError(
                    "finite candidate changed before capability open"
                )
        except BaseException as primary:
            close_error = _close_descriptor_once(descriptor)
            if close_error is not None:
                _attach_secondary_close(
                    primary, close_error, "candidate capability admission",
                )
            raise primary.with_traceback(primary.__traceback__)
        owner = _BindingOwner(descriptor, writable=writable)
        _candidate_json, candidate_identity = _identity({
            "device": expected.device,
            "domain": "xdart.finite-candidate-capability.v1",
            "inode": expected.inode,
            "operation_identity": self.request.operation_identity,
        })
        return (
            FiniteCandidateBinding(
                self.request,
                candidate_identity,
                owner,
                _FACTORY,
            ),
            owner,
        )

    def _run_document_session(
        self,
        parent_descriptor: int,
        candidate: Path,
        expected: FiniteFileSnapshot,
        *,
        opener: Callable[[FiniteCandidateBinding], ContextManager[object]],
        action: Callable[[object], object],
        writable: bool,
    ) -> tuple[FiniteCandidateBinding, object]:
        binding, owner = self._open_binding(
            parent_descriptor,
            candidate,
            expected,
            writable=writable,
        )
        primary: BaseException | None = None
        result: object | None = None
        try:
            with opener(binding) as document:
                result = action(document)
        except BaseException as error:
            primary = error
        close_error = owner.revoke()
        if primary is not None:
            if close_error is not None:
                _attach_secondary_close(
                    primary, close_error, "candidate capability",
                )
            raise primary.with_traceback(primary.__traceback__)
        if close_error is not None:
            raise FiniteArtifactIntegrityError(
                "finite candidate capability close is incomplete"
            ) from close_error
        return binding, result

    def _run_seeded_document_session(
        self,
        parent_descriptor: int,
        candidate: Path,
        expected: FiniteFileSnapshot,
        *,
        opener: Callable[[FiniteCandidateBinding], ContextManager[object]],
        action: Callable[
            [object, FiniteSeedBinding, FiniteCandidateBinding], object
        ],
        receipt: FiniteSourceSeedReceipt,
    ) -> tuple[FiniteCandidateBinding, object]:
        binding, owner = self._open_binding(
            parent_descriptor,
            candidate,
            expected,
            writable=True,
        )
        seed_binding = FiniteSeedBinding(
            self.request,
            receipt,
            binding.candidate_identity,
            owner,
            _FACTORY,
        )
        primary: BaseException | None = None
        result: object | None = None
        try:
            with opener(binding) as document:
                result = action(document, seed_binding, binding)
        except BaseException as error:
            primary = error
        close_error = owner.revoke()
        if primary is not None:
            if close_error is not None:
                _attach_secondary_close(
                    primary, close_error, "seeded candidate capability",
                )
            raise primary.with_traceback(primary.__traceback__)
        if close_error is not None:
            raise FiniteArtifactIntegrityError(
                "finite seeded candidate capability close is incomplete"
            ) from close_error
        return binding, result

    def _seed_candidate(
        self,
        parent_descriptor: int,
        candidate_path: Path,
        reservation: FiniteFileSnapshot,
        admission: FiniteSourceAdmission,
    ) -> FiniteSourceSeedReceipt:
        source = _require_source(admission)
        if source != self.request.operation_context.source_snapshot:
            raise FiniteArtifactIntegrityError(
                "finite seed admission does not match the request source snapshot"
            )
        source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        target_flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        source_descriptor = os.open(source.path, source_flags)
        target_descriptor: int | None = None
        primary: BaseException | None = None
        close_cause: BaseException | None = None
        try:
            target_descriptor = os.open(
                candidate_path.name,
                target_flags,
                dir_fd=parent_descriptor,
            )
            opened_source = os.fstat(source_descriptor)
            opened_target = os.fstat(target_descriptor)
            if (
                (int(opened_source.st_dev), int(opened_source.st_ino))
                != (source.device, source.inode)
                or (int(opened_target.st_dev), int(opened_target.st_ino))
                != (reservation.device, reservation.inode)
            ):
                raise FiniteArtifactIntegrityError(
                    "finite seed descriptor identity changed"
                )
            os.ftruncate(target_descriptor, 0)
            digest = hashlib.sha256()
            copied = 0
            while True:
                block = os.read(source_descriptor, _COPY_BLOCK_BYTES)
                if not block:
                    break
                digest.update(block)
                view = memoryview(block)
                offset = 0
                while offset < len(view):
                    written = os.write(target_descriptor, view[offset:])
                    if written <= 0:
                        raise OSError("finite seed copy made no progress")
                    offset += written
                copied += len(block)
            _fsync(target_descriptor)
        except BaseException as error:
            primary = error
        for descriptor, role in (
            (target_descriptor, "seed target"),
            (source_descriptor, "seed source"),
        ):
            if descriptor is None:
                continue
            close_error = _close_descriptor_once(descriptor)
            if close_error is None:
                continue
            if primary is None:
                primary = FiniteArtifactIntegrityError(
                    f"finite {role} close is incomplete"
                )
                close_cause = close_error
            else:
                _attach_secondary_close(primary, close_error, role)
        if primary is not None:
            if close_cause is not None:
                raise primary from close_cause
            raise primary.with_traceback(primary.__traceback__)
        _require_source(admission)
        candidate = _snapshot_at(
            parent_descriptor,
            candidate_path.name,
            candidate_path,
        )
        if (
            copied != source.size
            or digest.hexdigest() != source.digest
            or candidate.size != source.size
            or candidate.digest != source.digest
        ):
            raise FiniteArtifactIntegrityError("finite source seed is not byte exact")
        receipt_payload = {
            "byte_count": copied,
            "candidate_digest": candidate.digest,
            "candidate_inode": candidate.inode,
            "candidate_path": candidate.path,
            "copy_strategy": "bounded-copy-v1",
            "domain": "xdart.finite-source-seed-receipt.v1",
            "source_digest": source.digest,
            "source_inode": source.inode,
            "source_path": source.path,
        }
        _receipt_json, receipt_digest = _identity(receipt_payload)
        return FiniteSourceSeedReceipt(
            source,
            candidate,
            source.digest,
            candidate.digest,
            copied,
            "bounded-copy-v1",
            receipt_digest,
            _FACTORY,
        )

    def _fsync_candidate(
        self,
        parent_descriptor: int,
        path: Path,
        expected: FiniteFileSnapshot,
    ) -> None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        primary: BaseException | None = None
        try:
            state = os.fstat(descriptor)
            if (int(state.st_dev), int(state.st_ino)) != (
                expected.device,
                expected.inode,
            ):
                raise FiniteArtifactIntegrityError(
                    "finite candidate changed before file fsync"
                )
            _fsync(descriptor)
        except BaseException as error:
            primary = error
        close_error = _close_descriptor_once(descriptor)
        if primary is not None:
            if close_error is not None:
                _attach_secondary_close(
                    primary, close_error, "candidate fsync descriptor",
                )
            raise primary.with_traceback(primary.__traceback__)
        if close_error is not None:
            raise FiniteArtifactIntegrityError(
                "finite candidate fsync descriptor close is incomplete"
            ) from close_error

    def _inspect_exact(
        self,
        path: Path,
        inspect_committed: Callable[
            [Path, FiniteArtifactRequest],
            FiniteCommittedInspection,
        ],
        *,
        parent_descriptor: int,
        parent_state: tuple[int, int, int, int, int, int],
        collision: bool,
        expected: FiniteFileSnapshot | None = None,
    ) -> StreamTerminal:
        error_type = FiniteArtifactCollision if collision else FiniteArtifactIntegrityError
        try:
            self._require_parent(parent_descriptor, parent_state)
            before = _observe_regular_at(parent_descriptor, path.name, path)
            inspection = inspect_committed(path, self.request)
            if type(inspection) is not FiniteCommittedInspection:
                raise TypeError(
                    "committed inspector did not return FiniteCommittedInspection"
                )
            terminal = inspection.terminal
            if inspection.lineage != self.request.lineage:
                raise FiniteArtifactIntegrityError(
                    "finite public lineage is not the exact requested lineage"
                )
            self._require_parent(parent_descriptor, parent_state)
            revalidate_stream_terminal(path, terminal)
            self._require_parent(parent_descriptor, parent_state)
            after = _observe_regular_at(parent_descriptor, path.name, path)
            if (
                before != after
                or terminal.target != str(path)
                or terminal.size != after[3]
                or (terminal.device, terminal.inode, terminal.mtime_ns, terminal.ctime_ns)
                != (after[0], after[1], after[4], after[5])
                or expected is not None
                and (
                    terminal.digest != expected.digest
                    or after[:5] != _snapshot_state(expected)[:5]
                )
            ):
                raise FiniteArtifactIntegrityError(
                    "finite public terminal does not match exact storage facts"
                )
            return terminal
        except BaseException as error:
            if isinstance(error, error_type):
                raise
            raise error_type(
                "finite public occupant is not the exact requested version"
            ) from error

    def _accept_validated_own_link(
        self,
        path: Path,
        receipt: _ValidatedCandidateReceipt,
        accept_validated_commit: Callable[[FiniteCandidateValidation], object],
        *,
        parent_descriptor: int,
        parent_state: tuple[int, int, int, int, int, int],
    ) -> StreamTerminal:
        """Seal one newly linked validated inode with one public-name hash."""

        try:
            if (
                type(receipt) is not _ValidatedCandidateReceipt
                or receipt.operation_identity != self.request.operation_identity
                or receipt.validation.lineage != self.request.lineage
            ):
                raise FiniteArtifactIntegrityError(
                    "finite validated candidate receipt changed"
                )
            self._require_parent(parent_descriptor, parent_state)
            public = _snapshot_at(parent_descriptor, path.name, path)
            candidate = receipt.snapshot
            # Linking and removing the private alias changes ctime.  Every
            # content-bearing fact, including mtime and the whole-file digest,
            # must still be identical to the post-fsync validated candidate.
            if (
                (
                    public.device,
                    public.inode,
                    public.mode,
                    public.size,
                    public.mtime_ns,
                    public.digest,
                )
                != (
                    candidate.device,
                    candidate.inode,
                    candidate.mode,
                    candidate.size,
                    candidate.mtime_ns,
                    candidate.digest,
                )
            ):
                raise FiniteArtifactIntegrityError(
                    "finite public link does not match its validated candidate"
                )
            terminal = StreamTerminal(
                str(path),
                public.size,
                public.digest,
                1,
                public.device,
                public.inode,
                public.mtime_ns,
                public.ctime_ns,
            )
            self._require_parent(parent_descriptor, parent_state)
            if _observe_regular_at(
                parent_descriptor, path.name, path,
            ) != _snapshot_state(public):
                raise FiniteArtifactIntegrityError(
                    "finite public link changed after final hashing"
                )
            if accept_validated_commit(receipt.validation) is not None:
                raise TypeError("validated commit callback must return None")
            self._require_parent(parent_descriptor, parent_state)
            revalidate_stream_terminal(path, terminal)
            self._require_parent(parent_descriptor, parent_state)
            if _observe_regular_at(
                parent_descriptor, path.name, path,
            ) != _snapshot_state(public):
                raise FiniteArtifactIntegrityError(
                    "finite public link changed during commit acceptance"
                )
            return terminal
        except BaseException as error:
            if isinstance(error, FiniteArtifactIntegrityError):
                raise
            raise FiniteArtifactIntegrityError(
                "finite validated public link cannot be accepted"
            ) from error

    def _result(
        self,
        disposition: FiniteArtifactDisposition,
        *,
        terminal: StreamTerminal | None,
        seed_receipt: FiniteSourceSeedReceipt | None,
        hidden_orphan: str | None = None,
        diagnostics: tuple[str, ...] = (),
    ) -> FiniteArtifactResult:
        commit = None if terminal is None else _commit_identity(self.request, terminal)
        return FiniteArtifactResult(
            disposition,
            self.request,
            terminal,
            seed_receipt,
            commit,
            hidden_orphan,
            diagnostics,
            _FACTORY,
        )

    def _cleanup(
        self,
        parent_descriptor: int,
        candidate: Path,
        authority: FiniteFileSnapshot,
    ) -> tuple[str | None, BaseException | None]:
        try:
            _unlink_candidate(parent_descriptor, candidate, authority)
            return None, None
        except BaseException as error:
            try:
                current = _try_observe_at(
                    parent_descriptor,
                    candidate.name,
                    candidate,
                )
            except BaseException:
                current = None
            hidden = (
                str(candidate)
                if current is not None
                and current[:2] == (authority.device, authority.inode)
                else None
            )
            return hidden, error

    def publish(
        self,
        adapter: FiniteDocumentAdapter | FiniteSeededDocumentAdapter,
        *,
        inspect_committed: Callable[
            [Path, FiniteArtifactRequest],
            FiniteCommittedInspection,
        ],
        seed: FiniteSourceAdmission | None = None,
        prepublish: Callable[[], object] | None = None,
        accept_validated_commit: Callable[
            [FiniteCandidateValidation], object
        ] | None = None,
    ) -> FiniteArtifactResult:
        """Publish through one trusted adapter without mutating the source.

        ``adapter`` is trusted repository code and must obey the capability
        boundary documented by ``FiniteDocumentAdapter``.  In particular it
        must not inspect or duplicate the private backing descriptor.  This
        protocol proves closure for conforming file-object adapters; it is not
        a sandbox for arbitrary hostile same-process Python.

        A seeded caller may accept its exact attempt-local validation after the
        newly linked public name receives one matching whole-file hash.  That
        shortcut never applies to an existing occupant, replay, or collision.
        """
        if type(adapter) not in {
            FiniteDocumentAdapter,
            FiniteSeededDocumentAdapter,
        }:
            raise TypeError("finite publisher requires an exact document adapter")
        if not callable(inspect_committed):
            raise TypeError("finite committed inspector must be callable")
        if prepublish is not None and not callable(prepublish):
            raise TypeError("finite prepublication check must be callable")
        if accept_validated_commit is not None and not callable(
            accept_validated_commit
        ):
            raise TypeError("finite validated commit callback must be callable")
        if seed is not None and type(seed) is not FiniteSourceAdmission:
            raise TypeError("finite seed must be an exact source admission")
        if type(adapter) is FiniteSeededDocumentAdapter and seed is None:
            raise TypeError("finite seeded adapter requires one source admission")
        if (
            accept_validated_commit is not None
            and type(adapter) is not FiniteSeededDocumentAdapter
        ):
            raise TypeError(
                "finite validated commit callback requires a seeded adapter"
            )
        with self._lock:
            if self._used:
                raise RuntimeError("finite publisher is one-shot")
            self._used = True
            target = Path(self.request.output_artifact)
            parent_descriptor, parent_state = self._open_parent()
            # The slot lease comes AFTER the parent opens (so a bad parent needs
            # no release) and BEFORE any candidate byte exists.  A failure here
            # must not leak the descriptor the line above just took.
            try:
                self._acquire_slot()
            except BaseException:
                _close_descriptor_once(parent_descriptor)
                raise
            candidate: Path | None = None
            reservation: FiniteFileSnapshot | None = None
            seed_receipt: FiniteSourceSeedReceipt | None = None
            diagnostics: list[str] = []
            hidden_orphan: str | None = None
            own_link_observed = False
            candidate_consumed = False
            parent_closed = False

            def close_parent(primary: BaseException | None = None) -> None:
                nonlocal parent_closed
                if parent_closed:
                    return
                parent_closed = True
                # Give the slot back on EVERY exit path -- this helper is the
                # one point all four of them pass through.  A release failure is
                # reported, never raised: it must not mask the outcome the
                # caller came for.
                slot_error = self._release_slot()
                if slot_error is not None:
                    diagnostics.append(_bounded_diagnostic(
                        FINITE_SLOT_LEASE_WARNING, slot_error,
                    ))
                    if primary is not None:
                        # Codex F2 on `da228738`.  The `diagnostics` list only
                        # reaches the caller through a RESULT object, and a
                        # pre-publication failure re-raises the primary error
                        # instead of returning one -- so the warning was
                        # appended to a list nobody would ever read.  Attach it
                        # to the exception, exactly as a descriptor-close
                        # failure already does.
                        try:
                            primary.add_note(_bounded_diagnostic(
                                FINITE_SLOT_LEASE_WARNING, slot_error,
                            ))
                        except BaseException:
                            pass
                close_error = _close_descriptor_once(parent_descriptor)
                if close_error is None:
                    return
                diagnostics.append(_bounded_diagnostic(
                    FINITE_DESCRIPTOR_CLOSE_WARNING, close_error,
                ))
                if primary is not None:
                    _attach_secondary_close(
                        primary, close_error, "output parent",
                    )

            def cleanup_result(
                disposition: FiniteArtifactDisposition,
            ) -> FiniteArtifactResult:
                hidden: str | None = None
                cleanup_error: BaseException | None = None
                if candidate is not None and reservation is not None:
                    hidden, cleanup_error = self._cleanup(
                        parent_descriptor,
                        candidate,
                        reservation,
                    )
                if cleanup_error is not None:
                    diagnostics.append(_bounded_diagnostic(
                        FINITE_CANDIDATE_CLEANUP_WARNING,
                        cleanup_error,
                    ))
                close_parent()
                return self._result(
                    disposition,
                    terminal=None,
                    seed_receipt=seed_receipt,
                    hidden_orphan=hidden,
                    diagnostics=tuple(diagnostics),
                )

            try:
                self._require_parent(parent_descriptor, parent_state)
                request_source = self._require_request_source()
                if seed is not None and seed.snapshot != request_source:
                    raise FiniteArtifactIntegrityError(
                        "finite seed admission does not match the request source snapshot"
                    )
                if self._cancelled():
                    return cleanup_result(FiniteArtifactDisposition.ABORTED)
                # NO OCCUPANT EARLY EXIT.  This used to observe the slot
                # first and, if anything was there, inspect it and return
                # ALREADY_COMMITTED without building a candidate at all -- so a
                # repeat DISCARDED its own newly computed science and kept the
                # older file.  That is what made "change npt and Reintegrate
                # again" impossible in place, and it is what the maintainer's
                # ruling of 2026-09-04 overturns.  The slot is now always built
                # and always replaced; the H23 hold taken above is what keeps a
                # CONCURRENT operation out, which the no-clobber link used to do.
                if self._cancelled():
                    return cleanup_result(FiniteArtifactDisposition.ABORTED)

                self._require_candidate_capacity(
                    parent_descriptor,
                    request_source,
                )
                reservation = self._reserve_candidate(parent_descriptor)
                candidate = Path(reservation.path)
                if seed is not None:
                    seed_receipt = self._seed_candidate(
                        parent_descriptor,
                        candidate,
                        reservation,
                        seed,
                    )
                if self._cancelled():
                    return cleanup_result(FiniteArtifactDisposition.ABORTED)

                if type(adapter) is FiniteSeededDocumentAdapter:
                    if seed_receipt is None:
                        raise FiniteArtifactIntegrityError(
                            "finite seeded adapter lost its copy receipt"
                        )
                    _binding, write_result = self._run_seeded_document_session(
                        parent_descriptor,
                        candidate,
                        reservation,
                        opener=adapter.open_writer,
                        action=adapter.write,
                        receipt=seed_receipt,
                    )
                else:
                    _binding, write_result = self._run_document_session(
                        parent_descriptor,
                        candidate,
                        reservation,
                        opener=adapter.open_writer,
                        action=adapter.write,
                        writable=True,
                    )
                if write_result is FiniteCandidateWriteDisposition.ABORTED:
                    return cleanup_result(FiniteArtifactDisposition.ABORTED)
                written = _snapshot_at(
                    parent_descriptor,
                    candidate.name,
                    candidate,
                )
                if not _same_object(written, reservation):
                    raise FiniteArtifactIntegrityError(
                        "finite writer replaced its candidate"
                    )
                if self._cancelled():
                    return cleanup_result(FiniteArtifactDisposition.ABORTED)
                self._fsync_candidate(parent_descriptor, candidate, written)
                durable_state = _observe_regular_at(
                    parent_descriptor,
                    candidate.name,
                    candidate,
                )
                if durable_state != _snapshot_state(written):
                    raise FiniteArtifactIntegrityError(
                        "finite candidate changed across file fsync"
                    )
                _validation_binding, validation = self._run_document_session(
                    parent_descriptor,
                    candidate,
                    written,
                    opener=adapter.open_reader,
                    action=adapter.validate,
                    writable=False,
                )
                if (
                    type(validation) is not FiniteCandidateValidation
                    or validation.lineage != self.request.lineage
                ):
                    raise FiniteArtifactIntegrityError(
                        "finite candidate validation did not prove exact lineage"
                    )
                if _observe_regular_at(
                    parent_descriptor,
                    candidate.name,
                    candidate,
                ) != durable_state:
                    raise FiniteArtifactIntegrityError(
                        "finite candidate changed during semantic validation"
                    )
                validated = _ValidatedCandidateReceipt(
                    written,
                    validation,
                    self.request.operation_identity,
                    _FACTORY,
                )
                if prepublish is not None:
                    prepublish()
                if self._cancelled():
                    return cleanup_result(FiniteArtifactDisposition.ABORTED)
                self._require_request_source()
                if seed is not None:
                    if _require_source(seed) != request_source:
                        raise FiniteArtifactIntegrityError(
                            "finite seed admission does not match the request source snapshot"
                        )
                if _observe_regular_at(
                    parent_descriptor,
                    candidate.name,
                    candidate,
                ) != durable_state:
                    raise FiniteArtifactIntegrityError(
                        "finite candidate changed before publication"
                    )
                self._require_parent(parent_descriptor, parent_state)
                if self._cancelled():
                    return cleanup_result(FiniteArtifactDisposition.ABORTED)
                link_error: BaseException | None = None
                try:
                    # ATOMIC REPLACEMENT (ADR-0010; maintainer ruling
                    # 2026-09-04).  Was a NO-CLOBBER link, so an occupied slot
                    # kept the OLD file and discarded newly validated science.
                    # `os.replace` is atomic on POSIX and Windows, so the prior
                    # slot stays whole and visible right up to this instant.
                    #
                    # UNCONDITIONAL, as ruled: the slot name is fully determined
                    # by family plus operation, so an occupant is by definition
                    # this operation's own output.  A hand-placed file at that
                    # name IS destroyed -- an accepted consequence.  The
                    # CONCURRENT case is guarded by the H23 hold, not by this
                    # syscall.
                    replace_into_place(
                        candidate.name,
                        target.name,
                        verb="publish",
                        src_dir_fd=parent_descriptor,
                        dst_dir_fd=parent_descriptor,
                    )
                except BaseException as error:
                    link_error = error
                else:
                    # The rename is the public visibility point AND it consumed
                    # the candidate name.  Later namespace inspection may fail,
                    # but accounting must retain HELD state rather than treating
                    # a visible effect as a pre-publication abort.
                    own_link_observed = True
                    candidate_consumed = True
                try:
                    final_state = _try_observe_at(
                        parent_descriptor,
                        target.name,
                        target,
                    )
                except FiniteArtifactIntegrityError as observation_error:
                    try:
                        os.stat(
                            target.name,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        raise observation_error
                    if link_error is not None:
                        # The rename may have landed before reporting failure.
                        # An occupied final name plus an unavailable exact
                        # observation is therefore an unresolved public effect,
                        # never a proven pre-publication abort.  The old
                        # `FileExistsError` exemption is gone with the
                        # no-clobber link: `os.replace` cannot raise it.
                        own_link_observed = True
                    raise FiniteArtifactCollision(
                        "finite publication found a foreign public occupant"
                    ) from observation_error
                if (
                    final_state is not None
                    and final_state[:5] == (
                        validated.snapshot.device,
                        validated.snapshot.inode,
                        validated.snapshot.mode,
                        validated.snapshot.size,
                        validated.snapshot.mtime_ns,
                    )
                ):
                    disposition = FiniteArtifactDisposition.COMMITTED
                    own_link_observed = True
                elif link_error is not None:
                    # The rename FAILED.  Whatever occupies the slot is the
                    # prior file, exactly as ADR-0010 promises, so this is that
                    # failure and not a foreign writer.  Ordered before the
                    # branch below: reversed, a failed rename over an existing
                    # slot reported "changed under an exclusive hold" and buried
                    # the real cause.
                    raise link_error.with_traceback(link_error.__traceback__)
                elif final_state is not None:
                    # The rename reported success and yet the slot is not our
                    # inode: something outside this process wrote it in between.
                    # Under the H23 hold no xdart operation can, so this is an
                    # integrity failure -- NOT the old "already committed, reuse
                    # it" reading, which only made sense when a no-clobber link
                    # could legitimately lose a race.
                    raise FiniteArtifactIntegrityError(
                        "finite slot changed under an exclusive hold"
                    )
                else:
                    raise FiniteArtifactIntegrityError(
                        "finite publication effect cannot be established"
                    )
                if candidate_consumed:
                    # Nothing left to unlink: the rename moved the candidate ONTO
                    # the slot.  Asking anyway raises FileNotFoundError and hangs
                    # a spurious cleanup warning off a clean publication.
                    hidden_orphan, cleanup_error = None, None
                else:
                    hidden_orphan, cleanup_error = self._cleanup(
                        parent_descriptor,
                        candidate,
                        validated.snapshot,
                    )
                if cleanup_error is not None:
                    diagnostics.append(_bounded_diagnostic(
                        FINITE_CANDIDATE_CLEANUP_WARNING, cleanup_error
                    ))
                if disposition is FiniteArtifactDisposition.COMMITTED:
                    try:
                        _fsync_parent(parent_descriptor)
                    except BaseException as error:
                        diagnostics.append(_bounded_diagnostic(
                            FINITE_PARENT_DIRECTORY_FSYNC_WARNING, error
                        ))
                if (
                    disposition is FiniteArtifactDisposition.COMMITTED
                    and accept_validated_commit is not None
                ):
                    terminal = self._accept_validated_own_link(
                        target,
                        validated,
                        accept_validated_commit,
                        parent_descriptor=parent_descriptor,
                        parent_state=parent_state,
                    )
                else:
                    terminal = self._inspect_exact(
                        target,
                        inspect_committed,
                        parent_descriptor=parent_descriptor,
                        parent_state=parent_state,
                        collision=(
                            disposition
                            is FiniteArtifactDisposition.ALREADY_COMMITTED
                        ),
                        expected=(
                            validated.snapshot
                            if disposition is FiniteArtifactDisposition.COMMITTED
                            else None
                        ),
                    )
                close_parent()
                return self._result(
                    disposition,
                    terminal=terminal,
                    seed_receipt=seed_receipt,
                    hidden_orphan=hidden_orphan,
                    diagnostics=tuple(diagnostics),
                )
            except BaseException as primary:
                if (
                    candidate is not None
                    and reservation is not None
                    and not candidate_consumed
                ):
                    retry_hidden, cleanup_error = self._cleanup(
                        parent_descriptor,
                        candidate,
                        reservation,
                    )
                    hidden_orphan = retry_hidden
                    if cleanup_error is not None:
                        diagnostic = _bounded_diagnostic(
                            FINITE_CANDIDATE_CLEANUP_WARNING, cleanup_error
                        )
                        diagnostics.append(diagnostic)
                        try:
                            primary.add_note(
                                diagnostic + f":{hidden_orphan}"
                            )
                        except BaseException:
                            pass
                close_parent(primary)
                # Once a public link is observable it is intentionally retained.
                # An exact terminal failure is an integrity error, never abort.
                if own_link_observed and not isinstance(
                    primary, FiniteArtifactPublicationHeld,
                ):
                    raise FiniteArtifactPublicationHeld(
                        self.request,
                        primary,
                        hidden_orphan=hidden_orphan,
                        diagnostics=tuple(diagnostics),
                    ) from primary
                raise primary.with_traceback(primary.__traceback__)


__all__ = [
    "FINITE_CANDIDATE_CLEANUP_WARNING",
    "FINITE_DESCRIPTOR_CLOSE_WARNING",
    "FINITE_LINEAGE_NODE_NAME",
    "FINITE_LINEAGE_MAX_BYTES",
    "FINITE_LINEAGE_SCHEMA",
    "FINITE_PARENT_DIRECTORY_FSYNC_WARNING",
    "FINITE_SLOT_LEASE_WARNING",
    "FINITE_PUBLICATION_POLICY",
    "FiniteArtifactCollision",
    "FiniteArtifactCapacityError",
    "FiniteArtifactDisposition",
    "FiniteArtifactError",
    "FiniteArtifactIntegrityError",
    "FiniteArtifactPublicationHeld",
    "FiniteArtifactPublisher",
    "FiniteArtifactRequest",
    "FiniteArtifactResult",
    "FiniteArtifactLineage",
    "FiniteCandidateBinding",
    "FiniteCandidateValidation",
    "FiniteCommittedInspection",
    "FiniteDocumentAdapter",
    "FiniteSeededDocumentAdapter",
    "FiniteFileSnapshot",
    "FiniteSourceAdmission",
    "FiniteSourceSeedReceipt",
    "FiniteSeedBinding",
    "FiniteOperationContext",
    "FinitePredecessorReceipt",
    "admit_finite_artifact_lineage",
    "capture_finite_predecessor",
    "capture_finite_source",
    "finite_artifact_request",
    "finite_lineage_hdf_path",
    "finite_operation_context",
    "require_finite_artifact_lineage",
    "write_finite_artifact_lineage",
]
