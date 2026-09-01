"""Shared immutable contract for standalone Stitch and RSM operations.

The module-specific branches keep their scientific plans and runners.  This
module binds only exact source custody, a frozen plan fingerprint, one explicit
standalone-artifact intent, progress, and the terminal commit identity.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import re
import threading
from collections.abc import Callable, Mapping

from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition,
    AnalysisSourceReceipt,
    MetadataTablePlan,
    MetadataTableResult,
    MetadataTableRequalificationPlan,
    analysis_canonical_fingerprint,
    run_metadata_table,
    run_metadata_table_requalification,
)
from xrd_tools.io.analysis_artifact import (
    ANALYSIS_SCHEMA_VERSION,
    ANALYSIS_SCHEMA_VERSION_V2,
    AnalysisArtifactCleanupPending,
    AnalysisArtifactKind,
    AnalysisArtifactOutput,
    AnalysisArtifactOutputSnapshot,
    AnalysisArtifactOverwrite,
    AnalysisArtifactReceipt,
    AnalysisArtifactRequest,
    analysis_execution_attestation_digest,
    admit_analysis_artifact as _admit_analysis_artifact,
    canonical_analysis_provenance,
    inspect_analysis_artifact,
)
from xrd_tools.io import (
    revalidate_stream_terminal,
    StreamTerminal,
    TransactionPhase,
    stream_terminal_object_revision,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_MODULE_SOURCE_FACTORY = object()
_MODULE_SOURCE_GROUP_FACTORY = object()
_MODULE_RECEIPT_FACTORY = object()
_XU_STITCH_REQUEST_FACTORY = object()
_RSM_V2_REQUEST_FACTORY = object()
_MAX_RSM_GROUP_MEMBERS = 16
_MAX_RSM_GROUP_FRAMES = 4096
_XU_INTENT_TOP_KEYS = {
    "schema_version",
    "kind",
    "backend",
    "source",
    "asset",
    "effective_geometry",
    "detector",
    "corrections",
    "plan",
    "observations",
    "runtime_requirements",
    "output",
    "holds",
}


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TypeError(f"{name} must be lowercase SHA-256 hex")
    return value


def _exact_tuple(value: object, name: str) -> tuple:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    return value


class ModuleKind(str, Enum):
    STITCH = "stitch"
    RSM = "rsm"


class ModuleDisposition(str, Enum):
    COMMITTED = "committed"
    REFUSED = "refused"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ModuleArtifactRefused(RuntimeError):
    def __init__(self, code: str):
        if type(code) is not str or not code:
            raise TypeError("module artifact refusal code must be a nonempty string")
        self.code = code
        super().__init__(code)


class _ModuleCommitCancelled(RuntimeError):
    pass


class _ModuleCommitRefused(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class _ModuleWriterFailed(RuntimeError):
    def __init__(self, error: BaseException):
        self.error = error
        super().__init__("module writer failed")


class _ModulePrepublishFailed(RuntimeError):
    def __init__(self, error: BaseException):
        self.error = error
        super().__init__("module prepublication check failed")


def _failure_diagnostic(error: BaseException) -> str:
    try:
        message = str(error)
    except BaseException:
        message = "exception message unavailable"
    kind = type(error)
    return f"{kind.__module__}.{kind.__qualname__}: {message}"[:4096]


@dataclass(frozen=True, slots=True)
class MetadataColumnSelector:
    """One exact physical metadata column; occurrence is zero-based."""

    name: str
    occurrence: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.name.strip() != self.name
            or type(self.occurrence) is not int
            or self.occurrence < 0
        ):
            raise TypeError("metadata selector is invalid")


@dataclass(eq=False, frozen=True, slots=True)
class ModuleSourceReceipt:
    """Exact retained source receipt plus explicit module frame membership."""

    analysis: AnalysisSourceReceipt
    kind: ModuleKind
    table_fingerprint: str
    selected_labels: tuple[int, ...]
    resolved_selectors: tuple[MetadataColumnSelector, ...] = ()
    fingerprint: str = field(init=False)
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        labels = _exact_tuple(self.selected_labels, "selected labels")
        selectors = _exact_tuple(self.resolved_selectors, "resolved selectors")
        if (
            _claim is not _MODULE_SOURCE_FACTORY
            or type(self.analysis) is not AnalysisSourceReceipt
            or self.analysis.schema_version != "analysis-source-v2"
            or type(self.kind) is not ModuleKind
            or not labels
            or any(type(label) is not int for label in labels)
            or len(set(labels)) != len(labels)
            or any(type(selector) is not MetadataColumnSelector for selector in selectors)
            or len({(selector.name, selector.occurrence) for selector in selectors})
            != len(selectors)
        ):
            raise TypeError("module source receipt is invalid")
        _require_sha256(self.analysis.source_fingerprint, "source fingerprint")
        _require_sha256(self.table_fingerprint, "table fingerprint")
        source_positions = {label: index for index, label in enumerate(self.analysis.labels)}
        try:
            positions = tuple(source_positions[label] for label in labels)
        except KeyError as error:
            raise ValueError("selected label is outside the admitted source") from error
        if positions != tuple(sorted(positions)):
            raise ValueError("selected labels must retain admitted source order")
        fingerprint = analysis_canonical_fingerprint(
            "module-source-v1",
            (
                self.kind,
                self.analysis.source_fingerprint,
                self.table_fingerprint,
                labels,
                tuple((selector.name, selector.occurrence) for selector in selectors),
            ),
        )
        object.__setattr__(self, "fingerprint", fingerprint)

    def __copy__(self):
        raise TypeError("module source receipt is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("module source receipt is not copyable")

    def __reduce__(self):
        raise TypeError("module source receipt is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("module source receipt is not serializable")

    @classmethod
    def from_metadata_table(
        cls,
        table: MetadataTableResult,
        *,
        kind: ModuleKind,
        selected_labels: tuple[int, ...] | None = None,
        resolved_selectors: tuple[MetadataColumnSelector, ...] = (),
    ) -> "ModuleSourceReceipt":
        if (
            type(table) is not MetadataTableResult
            or table.disposition is not AnalysisDisposition.COMPLETED
            or type(table.receipt) is not AnalysisSourceReceipt
        ):
            raise ValueError("a completed exact metadata table is required")
        if type(kind) is not ModuleKind:
            raise TypeError("module kind must be exact")
        if selected_labels is not None and type(selected_labels) is not tuple:
            raise TypeError("selected labels must be an exact tuple")
        if type(resolved_selectors) is not tuple or any(
            type(selector) is not MetadataColumnSelector
            for selector in resolved_selectors
        ):
            raise TypeError("resolved selectors must be an exact selector tuple")
        current = run_metadata_table(MetadataTablePlan(table.receipt.source_spec))
        if (
            current.disposition is not AnalysisDisposition.COMPLETED
            or type(current.receipt) is not AnalysisSourceReceipt
            or current.receipt != table.receipt
            or current.labels != table.labels
            or current.table_fingerprint != table.table_fingerprint
        ):
            raise ValueError("metadata table is no longer an exact source fact")
        names = tuple(column.name for column in current.columns)
        if any(
            names.count(selector.name) <= selector.occurrence
            for selector in resolved_selectors
        ):
            raise ValueError("resolved metadata selector is outside the exact table")
        labels = table.labels if selected_labels is None else selected_labels
        return cls(
            current.receipt,
            kind,
            current.table_fingerprint,
            labels,
            resolved_selectors,
            _MODULE_SOURCE_FACTORY,
        )


@dataclass(eq=False, frozen=True, slots=True)
class ModuleSourceGroupReceipt:
    """Ordered exact RSM source receipts for one grouped transaction."""

    kind: ModuleKind
    members: tuple[ModuleSourceReceipt, ...]
    fingerprint: str = field(init=False)
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        members = _exact_tuple(self.members, "module source group members")
        if (
            _claim is not _MODULE_SOURCE_GROUP_FACTORY
            or self.kind is not ModuleKind.RSM
            or not 1 <= len(members) <= _MAX_RSM_GROUP_MEMBERS
            or any(type(member) is not ModuleSourceReceipt for member in members)
            or any(member.kind is not ModuleKind.RSM for member in members)
            or len({member.fingerprint for member in members}) != len(members)
        ):
            raise TypeError("module source group receipt is invalid")
        if sum(len(member.selected_labels) for member in members) > _MAX_RSM_GROUP_FRAMES:
            raise ValueError("module source group exceeds 4096 selected frames")
        object.__setattr__(
            self,
            "fingerprint",
            analysis_canonical_fingerprint(
                "module-source-group-v1",
                tuple(member.fingerprint for member in members),
            ),
        )

    @property
    def selected_frame_count(self) -> int:
        return sum(len(member.selected_labels) for member in self.members)

    @classmethod
    def from_members(
        cls,
        members: tuple[ModuleSourceReceipt, ...],
    ) -> "ModuleSourceGroupReceipt":
        return cls(ModuleKind.RSM, members, _MODULE_SOURCE_GROUP_FACTORY)

    def __copy__(self):
        raise TypeError("module source group receipt is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("module source group receipt is not copyable")

    def __replace__(self, /, **_changes):
        raise TypeError("module source group receipt is not replaceable")

    def __reduce__(self):
        raise TypeError("module source group receipt is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("module source group receipt is not serializable")


@dataclass(eq=False, frozen=True, slots=True)
class ModuleOutputRequest:
    """Intent for one fixed-entry standalone analysis artifact."""

    target: str | Path
    kind: AnalysisArtifactKind
    overwrite: AnalysisArtifactOverwrite = AnalysisArtifactOverwrite.CREATE_NEW
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.kind) is not AnalysisArtifactKind
            or type(self.overwrite) is not AnalysisArtifactOverwrite
        ):
            raise TypeError("module output request is invalid")
        try:
            target = os.path.normcase(
                os.path.abspath(os.path.expanduser(os.fspath(self.target)))
            )
        except TypeError as error:
            raise TypeError("module output target must be path-like") from error
        if Path(target).suffix.casefold() != ".nexus":
            raise ValueError("module output target must end in .nexus")
        object.__setattr__(self, "target", target)
        object.__setattr__(
            self,
            "fingerprint",
            analysis_canonical_fingerprint(
                "module-output-v1",
                (target, self.kind, self.overwrite),
            ),
        )


def module_plan_fingerprint(kind: ModuleKind, frozen_plan: object) -> str:
    if type(kind) is not ModuleKind:
        raise TypeError("module plan kind must be ModuleKind")
    return analysis_canonical_fingerprint(
        "module-plan-v1", (kind, frozen_plan)
    )


def module_provenance_digest(
    kind: ModuleKind,
    frozen_provenance: Mapping[str, object],
) -> str:
    if type(kind) is not ModuleKind:
        raise TypeError("module provenance kind must be ModuleKind")
    if type(frozen_provenance) is not dict:
        raise TypeError("module provenance must be an exact dictionary")
    canonical = canonical_analysis_provenance(frozen_provenance)
    return analysis_canonical_fingerprint(
        "module-provenance-v1", (kind, json.loads(canonical))
    )


@dataclass(eq=False, frozen=True, slots=True)
class ModuleOperationRequest:
    source: ModuleSourceReceipt | ModuleSourceGroupReceipt
    output: ModuleOutputRequest
    plan_fingerprint: str
    provenance_digest: str
    fingerprint: str = field(init=False)
    _xu_stitch_claim: InitVar[object] = None
    _xu_stitch_v2_bound: bool = field(init=False, default=False, repr=False)
    _rsm_v2_bound: bool = field(init=False, default=False, repr=False)

    def __post_init__(self, _xu_stitch_claim: object) -> None:
        single_source = type(self.source) is ModuleSourceReceipt
        group_source = type(self.source) is ModuleSourceGroupReceipt
        xu_bound = _xu_stitch_claim is _XU_STITCH_REQUEST_FACTORY
        rsm_v2_bound = _xu_stitch_claim is _RSM_V2_REQUEST_FACTORY
        if type(self.output) is not ModuleOutputRequest:
            raise TypeError("module request requires exact source and output values")
        if xu_bound and rsm_v2_bound:
            raise TypeError("module request cannot bind two private branches")
        if rsm_v2_bound:
            if not group_source:
                raise TypeError("RSM v2 request requires an exact source group")
        elif not single_source:
            raise TypeError("module request requires one exact source receipt")
        _require_sha256(self.plan_fingerprint, "plan fingerprint")
        _require_sha256(self.provenance_digest, "provenance digest")
        compatible = (
            self.source.kind is ModuleKind.STITCH
            and self.output.kind in {
                AnalysisArtifactKind.STITCH_1D,
                AnalysisArtifactKind.STITCH_2D,
            }
        ) or (
            self.source.kind is ModuleKind.RSM
            and self.output.kind is AnalysisArtifactKind.RSM
        )
        if not compatible:
            raise ValueError("module source and artifact kinds do not match")
        if xu_bound and (
            self.source.kind is not ModuleKind.STITCH
            or self.output.kind is not AnalysisArtifactKind.STITCH_1D
        ):
            raise TypeError("XU Stitch request requires exact Stitch 1-D intent")
        if rsm_v2_bound and (
            self.source.kind is not ModuleKind.RSM
            or self.output.kind is not AnalysisArtifactKind.RSM
        ):
            raise TypeError("RSM v2 request requires exact RSM artifact intent")
        object.__setattr__(self, "_xu_stitch_v2_bound", xu_bound)
        object.__setattr__(self, "_rsm_v2_bound", rsm_v2_bound)
        object.__setattr__(
            self,
            "fingerprint",
            analysis_canonical_fingerprint(
                "module-request-v1",
                (
                    "module-request-v2-rsm-bound",
                    self.source.fingerprint,
                    self.output.fingerprint,
                    self.plan_fingerprint,
                    self.provenance_digest,
                )
                if rsm_v2_bound
                else (
                    "xu-stitch-v2-bound",
                    self.source.fingerprint,
                    self.output.fingerprint,
                    self.plan_fingerprint,
                    self.provenance_digest,
                )
                if xu_bound
                else (
                    self.source.fingerprint,
                    self.output.fingerprint,
                    self.plan_fingerprint,
                    self.provenance_digest,
                ),
            ),
        )

    @property
    def kind(self) -> ModuleKind:
        return self.source.kind

    @property
    def artifact_source_fingerprint(self) -> str:
        if type(self.source) is ModuleSourceGroupReceipt:
            return self.source.fingerprint
        return self.source.analysis.source_fingerprint

    def __copy__(self):
        raise TypeError("module operation request is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("module operation request is not copyable")

    def __replace__(self, **_changes):
        raise TypeError("module operation request is not replaceable")

    def __reduce__(self):
        raise TypeError("module operation request is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("module operation request is not serializable")


def xu_stitch_module_request(
    source: ModuleSourceReceipt,
    output: ModuleOutputRequest,
    plan_fingerprint: str,
    provenance_digest: str,
) -> ModuleOperationRequest:
    """Construct the one standard request authorized for XU artifact v2."""

    return ModuleOperationRequest(
        source,
        output,
        plan_fingerprint,
        provenance_digest,
        _XU_STITCH_REQUEST_FACTORY,
    )


def _rsm_v2_module_request(
    source: ModuleSourceGroupReceipt,
    output: ModuleOutputRequest,
    plan_fingerprint: str,
    provenance_digest: str,
) -> ModuleOperationRequest:
    """Construct the one private request branch authorized for RSM v2."""

    return ModuleOperationRequest(
        source,
        output,
        plan_fingerprint,
        provenance_digest,
        _RSM_V2_REQUEST_FACTORY,
    )


def _validate_xu_stitch_v2_intent(
    request: ModuleOperationRequest,
    provenance: Mapping[str, object],
) -> None:
    source = provenance.get("source")
    asset = provenance.get("asset")
    effective = provenance.get("effective_geometry")
    detector = provenance.get("detector")
    corrections = provenance.get("corrections")
    plan = provenance.get("plan")
    observations = provenance.get("observations")
    runtime = provenance.get("runtime_requirements")
    output = provenance.get("output")
    holds = provenance.get("holds")
    if (
        request._xu_stitch_v2_bound is not True
        or set(provenance) != _XU_INTENT_TOP_KEYS
        or provenance.get("schema_version")
        != "stitch-operation-v2-xu-intent"
        or provenance.get("kind") != "stitch"
        or provenance.get("backend") != "xu_hist"
        or type(source) is not dict
        or set(source)
        != {
            "source_fingerprint",
            "module_source_fingerprint",
            "metadata_table_fingerprint",
            "selected_labels",
            "input_manifest",
        }
        or source.get("source_fingerprint")
        != request.source.analysis.source_fingerprint
        or source.get("module_source_fingerprint")
        != request.source.fingerprint
        or source.get("metadata_table_fingerprint")
        != request.source.table_fingerprint
        or source.get("selected_labels")
        != list(request.source.selected_labels)
        or type(source.get("input_manifest")) is not dict
        or type(asset) is not dict
        or set(asset)
        != {
            "lexical_relative_path",
            "resolved_relative_path",
            "byte_count",
            "raw_sha256",
            "semantic_fingerprint",
            "receipt_fingerprint",
        }
        or any(
            _SHA256.fullmatch(asset.get(name, "")) is None
            for name in (
                "raw_sha256",
                "semantic_fingerprint",
                "receipt_fingerprint",
            )
        )
        or type(effective) is not dict
        or _SHA256.fullmatch(effective.get("fingerprint", "")) is None
        or type(detector) is not dict
        or not detector
        or type(corrections) is not dict
        or not corrections
        or type(plan) is not dict
        or set(plan)
        != {
            "backend",
            "mode",
            "unit",
            "method",
            "radial_range",
            "npt_1d",
            "monitor_selector",
            "use_detector_mask",
            "max_frame_bytes",
            "asset_receipt_fingerprint",
            "effective_geometry_fingerprint",
            "plan_fingerprint",
        }
        or plan.get("backend") != "xu_hist"
        or plan.get("mode") != "1d"
        or plan.get("unit") != "q_A^-1"
        or plan.get("method") != "numpy_histogram_center_v1"
        or plan.get("use_detector_mask") is not True
        or plan.get("plan_fingerprint") != request.plan_fingerprint
        or plan.get("asset_receipt_fingerprint")
        != asset.get("receipt_fingerprint")
        or plan.get("effective_geometry_fingerprint")
        != effective.get("fingerprint")
        or type(observations) is not dict
        or not observations
        or type(runtime) is not dict
        or not runtime
        or type(output) is not dict
        or output
        != {
            "target": request.output.target,
            "kind": request.output.kind.value,
            "overwrite": request.output.overwrite.value,
            "output_fingerprint": request.output.fingerprint,
        }
        or type(holds) is not list
        or not holds
        or any(type(item) is not str or not item for item in holds)
    ):
        raise ModuleArtifactRefused("XU_INTENT_PROVENANCE_MISMATCH")


@dataclass(frozen=True, slots=True)
class ModuleProgress:
    request: ModuleOperationRequest
    revision: int
    stage: str
    completed: int
    total: int

    def __post_init__(self) -> None:
        if (
            type(self.request) is not ModuleOperationRequest
            or type(self.revision) is not int
            or self.revision < 1
            or type(self.stage) is not str
            or not self.stage
            or type(self.completed) is not int
            or type(self.total) is not int
            or self.total < 1
            or self.completed < 0
            or self.completed > self.total
        ):
            raise TypeError("module progress is invalid")


@dataclass(eq=False, frozen=True, slots=True)
class ModuleCommitReceipt:
    request: ModuleOperationRequest
    output: ModuleOutputRequest
    terminal: StreamTerminal
    result_fingerprint: str
    execution_attestation_digest: str | None = field(default=None, kw_only=True)
    fingerprint: str = field(init=False)
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _MODULE_RECEIPT_FACTORY
            or type(self.request) is not ModuleOperationRequest
            or self.output is not self.request.output
            or type(self.terminal) is not StreamTerminal
            or self.terminal.target != self.output.target
            or stream_terminal_object_revision(self.terminal) is None
        ):
            raise TypeError("module commit receipt lacks exact modern output authority")
        _require_sha256(self.result_fingerprint, "result fingerprint")
        if self.execution_attestation_digest is not None:
            _require_sha256(
                self.execution_attestation_digest,
                "execution attestation digest",
            )
        domain = (
            "module-commit-v1"
            if self.execution_attestation_digest is None
            else "module-commit-v2"
        )
        identity = (
            self.request.fingerprint,
            self.terminal.target,
            self.terminal.size,
            self.terminal.digest,
            self.terminal.ordinal,
            stream_terminal_object_revision(self.terminal),
            self.result_fingerprint,
        )
        if self.execution_attestation_digest is not None:
            identity += (self.execution_attestation_digest,)
        object.__setattr__(
            self,
            "fingerprint",
            analysis_canonical_fingerprint(
                domain,
                identity,
            ),
        )

    def __copy__(self):
        raise TypeError("module commit receipt is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("module commit receipt is not copyable")

    def __reduce__(self):
        raise TypeError("module commit receipt is not serializable")

    def __reduce_ex__(self, _protocol):
        raise TypeError("module commit receipt is not serializable")

    @classmethod
    def from_artifact(
        cls,
        request: ModuleOperationRequest,
        artifact: AnalysisArtifactReceipt,
    ) -> "ModuleCommitReceipt":
        if (
            type(request) is not ModuleOperationRequest
            or type(artifact) is not AnalysisArtifactReceipt
            or artifact.request._module_owner is not request
        ):
            raise TypeError("artifact receipt does not match the exact module request")
        try:
            expected = module_artifact_request(
                request,
                json.loads(artifact.request.provenance_json),
                execution_attestation=(
                    None
                    if artifact.request.execution_attestation_json is None
                    else json.loads(
                        artifact.request.execution_attestation_json
                    )
                ),
                execution_attestation_digest=(
                    artifact.request.execution_attestation_digest
                ),
            )
        except (ModuleArtifactRefused, TypeError, ValueError) as error:
            raise TypeError(
                "artifact receipt does not match the exact module request"
            ) from error
        fields = (
            "target",
            "kind",
            "overwrite",
            "request_fingerprint",
            "source_fingerprint",
            "plan_fingerprint",
            "provenance_digest",
            "provenance_json",
            "schema_version",
            "execution_attestation_digest",
            "execution_attestation_json",
        )
        if any(
            getattr(artifact.request, name) != getattr(expected, name)
            for name in fields
        ):
            raise TypeError("artifact receipt does not match the exact module request")
        revalidate_stream_terminal(artifact.request.target, artifact.terminal)
        inspection = inspect_analysis_artifact(
            artifact.request.target,
            expected_request=artifact.request,
        )
        revalidate_stream_terminal(artifact.request.target, artifact.terminal)
        if inspection != artifact.inspection:
            raise TypeError("artifact receipt no longer matches exact inspected result")
        if (
            inspection.execution_attestation_digest
            != artifact.request.execution_attestation_digest
            or inspection.execution_attestation_json
            != artifact.request.execution_attestation_json
        ):
            raise TypeError(
                "artifact execution attestation no longer matches its request"
            )
        return cls(
            request,
            request.output,
            artifact.terminal,
            artifact.inspection.result_fingerprint,
            _claim=_MODULE_RECEIPT_FACTORY,
            execution_attestation_digest=(
                artifact.inspection.execution_attestation_digest
            ),
        )


@dataclass(eq=False, frozen=True, slots=True)
class ModuleTerminalResult:
    request: ModuleOperationRequest
    disposition: ModuleDisposition
    code: str
    commit: ModuleCommitReceipt | None = None
    diagnostic: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.request) is not ModuleOperationRequest
            or type(self.disposition) is not ModuleDisposition
            or type(self.code) is not str
            or not self.code
            or type(self.diagnostic) is not str
            or (
                self.disposition is ModuleDisposition.COMMITTED
                and (
                    type(self.commit) is not ModuleCommitReceipt
                    or self.commit.request is not self.request
                )
            )
            or (
                self.disposition is not ModuleDisposition.COMMITTED
                and self.commit is not None
            )
        ):
            raise TypeError("module terminal result is invalid")


@dataclass(frozen=True, slots=True)
class ModuleArtifactOutputSnapshot:
    """Artifact state plus the outer module handoff still owed to its caller."""

    artifact: AnalysisArtifactOutputSnapshot
    module_pending: bool
    retryable: bool

    @property
    def target(self) -> str:
        return self.artifact.target

    @property
    def phase(self) -> TransactionPhase:
        return self.artifact.phase

    @property
    def writer_started(self) -> bool:
        return self.artifact.writer_started

    @property
    def writer_finished(self) -> bool:
        return self.artifact.writer_finished

    @property
    def remaining_lease_owners(self):
        return self.artifact.remaining_lease_owners

    @property
    def receipt(self) -> AnalysisArtifactReceipt | None:
        return self.artifact.receipt


def _source_requalification(
    source: ModuleSourceReceipt | ModuleSourceGroupReceipt,
    *,
    cancel_token: threading.Event | None = None,
):
    if type(source) is ModuleSourceGroupReceipt:
        result = None
        for member in source.members:
            result = _source_requalification(member, cancel_token=cancel_token)
            if result.disposition is not AnalysisDisposition.COMPLETED:
                return result
        return result
    if type(source) is not ModuleSourceReceipt:
        raise TypeError("source requalification requires an exact source receipt")
    result = run_metadata_table_requalification(
        MetadataTableRequalificationPlan(
            source.analysis,
            source.table_fingerprint,
        ),
        cancel_token=cancel_token,
    )
    if (
        result.disposition is AnalysisDisposition.COMPLETED
        and (
            result.receipt is not source.analysis
            or result.table_fingerprint != source.table_fingerprint
        )
    ):
        raise ModuleArtifactRefused("SOURCE_IDENTITY_MISMATCH")
    return result


def module_artifact_request(
    request: ModuleOperationRequest,
    provenance: Mapping[str, object],
    *,
    execution_attestation: Mapping[str, object] | None = None,
    execution_attestation_digest: str | None = None,
) -> AnalysisArtifactRequest:
    if type(request) is not ModuleOperationRequest:
        raise TypeError("artifact binding requires exact ModuleOperationRequest")
    if request._rsm_v2_bound:
        raise ModuleArtifactRefused("RSM_V2_ARTIFACT_NOT_IMPLEMENTED")
    if type(provenance) is not dict:
        raise TypeError("module provenance must be an exact dictionary")
    carries_attestation = (
        execution_attestation is not None
        or execution_attestation_digest is not None
    )
    if carries_attestation and (
        execution_attestation is None
        or type(execution_attestation_digest) is not str
    ):
        raise TypeError(
            "module execution attestation and digest must be supplied together"
        )
    if carries_attestation:
        _validate_xu_stitch_v2_intent(request, provenance)
        observed = analysis_execution_attestation_digest(
            request.output.kind,
            execution_attestation,
            request_fingerprint=request.fingerprint,
        )
        if observed != execution_attestation_digest:
            raise ModuleArtifactRefused("EXECUTION_ATTESTATION_MISMATCH")
        selected_count = execution_attestation.get("selected_frame_count")
        release_count = execution_attestation.get("release_check_frame_count")
        expected_count = len(request.source.selected_labels)
        if selected_count != expected_count or release_count != expected_count:
            raise ModuleArtifactRefused("EXECUTION_ATTESTATION_COUNT_MISMATCH")
    artifact = AnalysisArtifactRequest(
        request.output.target,
        request.output.kind,
        request.output.overwrite,
        request.fingerprint,
        request.source.analysis.source_fingerprint,
        request.plan_fingerprint,
        request.provenance_digest,
        provenance,
        schema_version=(
            ANALYSIS_SCHEMA_VERSION_V2
            if carries_attestation
            else ANALYSIS_SCHEMA_VERSION
        ),
        execution_attestation_digest=execution_attestation_digest,
        execution_attestation=execution_attestation,
        module_owner=request,
    )
    frozen_provenance = json.loads(artifact.provenance_json)
    if (
        module_provenance_digest(request.kind, frozen_provenance)
        != request.provenance_digest
    ):
        raise ModuleArtifactRefused("PROVENANCE_MISMATCH")
    return artifact


class ModuleArtifactOutput:
    """Exact request binding around one admitted standalone output owner."""

    def __init__(
        self,
        request: ModuleOperationRequest,
        output: AnalysisArtifactOutput,
    ) -> None:
        if (
            type(request) is not ModuleOperationRequest
            or type(output) is not AnalysisArtifactOutput
            or output.request._module_owner is not request
        ):
            raise TypeError("module artifact owner requires exact values")
        foreign = output.request
        expected = module_artifact_request(
            request,
            json.loads(foreign.provenance_json),
            execution_attestation=(
                None
                if foreign.execution_attestation_json is None
                else json.loads(foreign.execution_attestation_json)
            ),
            execution_attestation_digest=foreign.execution_attestation_digest,
        )
        fields = (
            "target",
            "kind",
            "overwrite",
            "request_fingerprint",
            "source_fingerprint",
            "plan_fingerprint",
            "provenance_digest",
            "provenance_json",
            "schema_version",
            "execution_attestation_digest",
            "execution_attestation_json",
        )
        if any(getattr(foreign, name) != getattr(expected, name) for name in fields):
            raise TypeError("artifact output is foreign to the module request")
        self.request = request
        self._output = output
        self._publish_started = False
        self._pending: tuple[ModuleDisposition, str, str] | None = None
        self._terminal: ModuleTerminalResult | None = None

    def __copy__(self):
        raise TypeError("module artifact output owner is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("module artifact output owner is not copyable")

    @property
    def snapshot(self) -> ModuleArtifactOutputSnapshot:
        artifact = self._output.snapshot
        module_pending = self._publish_started and (
            self._terminal is None or bool(artifact.remaining_lease_owners)
        )
        return ModuleArtifactOutputSnapshot(
            artifact,
            module_pending,
            artifact.retryable
            or (
                module_pending
                and artifact.phase is TransactionPhase.COMMITTED
                and artifact.receipt is not None
            ),
        )

    def _committed(
        self,
        artifact: AnalysisArtifactReceipt,
    ) -> ModuleTerminalResult:
        if self._terminal is not None:
            return self._terminal
        commit = ModuleCommitReceipt.from_artifact(
            self.request,
            artifact,
        )
        self._terminal = ModuleTerminalResult(
            self.request,
            ModuleDisposition.COMMITTED,
            "OK",
            commit,
        )
        return self._terminal

    def _settle_pending(self) -> ModuleTerminalResult:
        if self._terminal is not None:
            return self._terminal
        if self._pending is None:
            raise RuntimeError("module cleanup finished without a retained outcome")
        disposition, code, diagnostic = self._pending
        self._terminal = ModuleTerminalResult(
            self.request,
            disposition,
            code,
            diagnostic=diagnostic,
        )
        return self._terminal

    def publish(
        self,
        write_result: Callable[[object], object],
        *,
        cancel_token: threading.Event | None = None,
        prepublish_check: Callable[[], object] | None = None,
    ) -> ModuleTerminalResult:
        if cancel_token is not None and type(cancel_token) is not threading.Event:
            raise TypeError("module cancellation token must be threading.Event")
        if not callable(write_result):
            raise TypeError("module artifact writer must be callable")
        if prepublish_check is not None and not callable(prepublish_check):
            raise TypeError("module prepublication check must be callable")
        if self._publish_started:
            raise RuntimeError("module artifact publication is one-shot")
        self._publish_started = True

        def prepublish() -> None:
            if cancel_token is not None and cancel_token.is_set():
                raise _ModuleCommitCancelled("CANCELLED")
            try:
                result = _source_requalification(
                    self.request.source,
                    cancel_token=cancel_token,
                )
            except ModuleArtifactRefused as error:
                raise _ModuleCommitRefused(error.code) from error
            if result.disposition is AnalysisDisposition.CANCELLED:
                raise _ModuleCommitCancelled("CANCELLED")
            if result.disposition is not AnalysisDisposition.COMPLETED:
                raise _ModuleCommitRefused(result.code)
            if prepublish_check is not None:
                try:
                    prepublish_check()
                except ModuleArtifactRefused as error:
                    if type(error) is not ModuleArtifactRefused:
                        raise _ModulePrepublishFailed(error) from error
                    try:
                        code = error.code
                    except BaseException as code_error:
                        raise _ModulePrepublishFailed(code_error) from code_error
                    if type(code) is not str or not code:
                        raise _ModulePrepublishFailed(error) from error
                    raise _ModuleCommitRefused(code) from error
                except BaseException as error:
                    raise _ModulePrepublishFailed(error) from error

        def guarded_writer(entry: object) -> object:
            try:
                return write_result(entry)
            except BaseException as error:
                raise _ModuleWriterFailed(error) from error

        if cancel_token is not None and cancel_token.is_set():
            self._pending = (
                ModuleDisposition.CANCELLED,
                "CANCELLED",
                "",
            )
            try:
                self._output.abort()
            except BaseException as error:
                raise AnalysisArtifactCleanupPending(self.snapshot) from error
            return self._settle_pending()
        try:
            artifact = self._output._publish_retained(
                guarded_writer,
                prepublish=prepublish,
            )
        except (_ModuleWriterFailed, _ModulePrepublishFailed) as error:
            self._pending = (
                ModuleDisposition.FAILED,
                "OUTPUT_FAILED",
                _failure_diagnostic(error.error),
            )
            if self._output.snapshot.remaining_lease_owners:
                raise AnalysisArtifactCleanupPending(self.snapshot) from error.error
            return self._settle_pending()
        except _ModuleCommitCancelled:
            self._pending = (
                ModuleDisposition.CANCELLED,
                "CANCELLED",
                "",
            )
            if self._output.snapshot.remaining_lease_owners:
                raise AnalysisArtifactCleanupPending(self.snapshot)
            return self._settle_pending()
        except _ModuleCommitRefused as error:
            self._pending = (
                ModuleDisposition.REFUSED,
                error.code,
                "",
            )
            if self._output.snapshot.remaining_lease_owners:
                raise AnalysisArtifactCleanupPending(self.snapshot) from error
            return self._settle_pending()
        except AnalysisArtifactCleanupPending as error:
            snapshot = self._output.snapshot
            if self._pending is None:
                primary = error.__cause__
                if isinstance(
                    primary, (_ModuleWriterFailed, _ModulePrepublishFailed)
                ):
                    self._pending = (
                        ModuleDisposition.FAILED,
                        "OUTPUT_FAILED",
                        _failure_diagnostic(primary.error),
                    )
                elif isinstance(primary, _ModuleCommitCancelled):
                    self._pending = (
                        ModuleDisposition.CANCELLED,
                        "CANCELLED",
                        "",
                    )
                elif isinstance(primary, _ModuleCommitRefused):
                    self._pending = (
                        ModuleDisposition.REFUSED,
                        primary.code,
                        "",
                    )
                else:
                    self._pending = (
                        ModuleDisposition.FAILED,
                        "OUTPUT_FAILED",
                        _failure_diagnostic(primary if primary is not None else error),
                    )
            if (
                snapshot.phase is TransactionPhase.ABORTED
                and not snapshot.remaining_lease_owners
            ):
                return self._settle_pending()
            raise AnalysisArtifactCleanupPending(self.snapshot) from (
                error.__cause__ if error.__cause__ is not None else error
            )
        except BaseException as error:
            diagnostic = _failure_diagnostic(error)
            self._pending = (
                ModuleDisposition.FAILED,
                "OUTPUT_FAILED",
                diagnostic,
            )
            if self._output.snapshot.remaining_lease_owners:
                raise AnalysisArtifactCleanupPending(self.snapshot) from error
            return self._settle_pending()
        try:
            terminal = self._committed(artifact)
            self._output._complete_retained()
            return terminal
        except BaseException as error:
            raise AnalysisArtifactCleanupPending(self.snapshot) from error

    def retry_cleanup(self) -> ModuleTerminalResult:
        if self._terminal is not None:
            if not self._output.snapshot.remaining_lease_owners:
                return self._terminal
            try:
                self._output._complete_retained()
            except BaseException as error:
                raise AnalysisArtifactCleanupPending(self.snapshot) from error
            return self._terminal
        try:
            snapshot = self._output._retry_retained()
        except AnalysisArtifactCleanupPending as error:
            raise AnalysisArtifactCleanupPending(self.snapshot) from (
                error.__cause__ if error.__cause__ is not None else error
            )
        if snapshot.phase is TransactionPhase.COMMITTED:
            if snapshot.receipt is None:
                raise AnalysisArtifactCleanupPending(self.snapshot)
            try:
                terminal = self._committed(snapshot.receipt)
                self._output._complete_retained()
                return terminal
            except BaseException as error:
                raise AnalysisArtifactCleanupPending(self.snapshot) from error
        if snapshot.phase is TransactionPhase.ABORTED:
            return self._settle_pending()
        raise AnalysisArtifactCleanupPending(self.snapshot)


def admit_module_artifact(
    request: ModuleOperationRequest,
    provenance: Mapping[str, object],
    *,
    execution_attestation: Mapping[str, object] | None = None,
    execution_attestation_digest: str | None = None,
    cancel_token: threading.Event | None = None,
    coordinator: object | None = None,
) -> ModuleArtifactOutput:
    if type(request) is not ModuleOperationRequest:
        raise TypeError("module artifact admission requires exact request")
    if cancel_token is not None and type(cancel_token) is not threading.Event:
        raise TypeError("module cancellation token must be threading.Event")
    if cancel_token is not None and cancel_token.is_set():
        raise ModuleArtifactRefused("CANCELLED")
    result = _source_requalification(
        request.source,
        cancel_token=cancel_token,
    )
    if result.disposition is not AnalysisDisposition.COMPLETED:
        raise ModuleArtifactRefused(result.code)
    artifact_request = module_artifact_request(
        request,
        provenance,
        execution_attestation=execution_attestation,
        execution_attestation_digest=execution_attestation_digest,
    )
    output = _admit_analysis_artifact(
        artifact_request,
        coordinator=coordinator,
    )
    return ModuleArtifactOutput(request, output)


__all__ = [
    "MetadataColumnSelector",
    "ModuleArtifactOutput",
    "ModuleArtifactOutputSnapshot",
    "ModuleArtifactRefused",
    "ModuleCommitReceipt",
    "ModuleDisposition",
    "ModuleKind",
    "ModuleOperationRequest",
    "ModuleOutputRequest",
    "ModuleProgress",
    "ModuleSourceGroupReceipt",
    "ModuleSourceReceipt",
    "ModuleTerminalResult",
    "admit_module_artifact",
    "module_artifact_request",
    "module_plan_fingerprint",
    "module_provenance_digest",
    "xu_stitch_module_request",
]
