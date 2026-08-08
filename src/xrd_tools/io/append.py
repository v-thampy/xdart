"""Strict, Qt-free Append qualification and committed lineage values."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Callable, Iterable, Mapping

import h5py
import numpy as np
from .schema import (
    ACCEPTED_SCHEMA_NAMES,
    DEFAULT_MODE_KEY,
    MODE_SUBGROUP_NAMES,
    PRIMARY_MODE_ATTR,
    PROCESSED_SCHEMA_VERSION,
    SCHEMA_NAME_ATTR,
    SCHEMA_VERSION_ATTR,
    SOURCE_BASE_ATTR,
)
LINEAGE_DATASET = "append_lineage"
LINEAGE_VERSION = 1
class AppendDisposition(str, Enum):
    WRITE = "write"
    SKIP = "skip"
    REFUSE = "refuse"

class AppendPreflightState(str, Enum):
    RESERVED = "reserved"
    BOUND = "bound"
    NOOP = "noop"
    COMMITTED = "committed"
    ABORTED = "aborted"
    RETRYABLE = "retryable"
    INTEGRITY_HOLD = "integrity_hold"
class AppendRefused(RuntimeError):
    def __init__(self, decision: "AppendDecision") -> None:
        super().__init__(decision.reason or "Append target refused")
        self.decision = decision


class AppendSourceGraphRefused(AppendRefused):
    """A readable detector graph is not totally qualifiable for Append."""

    def __init__(self, reason: str, *, source_generation: int = 0) -> None:
        super().__init__(_decision(
            AppendDisposition.REFUSE,
            reason=str(reason),
            source_generation=int(source_generation),
        ))


class AppendPreflightCleanupError(RuntimeError):
    def __init__(self, primary: BaseException, owner: "AppendPreflight") -> None:
        super().__init__(
            f"Append preflight cleanup retained for {owner.target}: {primary}")
        self.primary = primary
        self.owner = owner
@dataclass(frozen=True)
class AppendExternalMember:
    path: str
    dataset_path: str
    size: int
    mtime_ns: int
    source_start: int
    source_stop: int
    ordinal: int
    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError("external member path is required")
        if not str(self.dataset_path).strip():
            raise ValueError("external member dataset path is required")
        object.__setattr__(self, "path", _normalize(self.path))
        object.__setattr__(self, "dataset_path", str(self.dataset_path))
        if int(self.size) < 0 or int(self.mtime_ns) < 0:
            raise ValueError("external member size/mtime must be non-negative")
        if int(self.source_start) < 0 or int(self.source_stop) <= int(self.source_start):
            raise ValueError("external member source range is invalid")
        if int(self.ordinal) < 0:
            raise ValueError("external member ordinal must be non-negative")
@dataclass(frozen=True)
class AppendImageMember:
    path: str
    size: int
    mtime_ns: int
    source_start: int
    source_stop: int
    ordinal: int

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError("image-series member path is required")
        object.__setattr__(self, "path", _normalize(self.path))
        if int(self.size) < 0 or int(self.mtime_ns) < 0:
            raise ValueError("image-series member size/mtime must be non-negative")
        if int(self.source_start) < 0 or int(self.source_stop) <= int(self.source_start):
            raise ValueError("image-series member source range is invalid")
        if int(self.ordinal) < 0:
            raise ValueError("image-series member ordinal must be non-negative")
@dataclass(frozen=True)
class AppendSource:
    path: str
    adapter_id: str
    size: int
    mtime_ns: int
    extent: int
    digest: str | None = None
    dataset_paths: tuple[str, ...] = ()
    image_members: tuple[AppendImageMember, ...] = ()
    external_members: tuple[AppendExternalMember, ...] = ()
    generation: int = 0
    def __post_init__(self) -> None:
        if not str(self.path).strip() or not str(self.adapter_id).strip():
            raise ValueError("source path and adapter identity are required")
        object.__setattr__(self, "path", _normalize(self.path))
        object.__setattr__(self, "adapter_id", str(self.adapter_id))
        object.__setattr__(self, "dataset_paths", tuple(
            str(path) for path in self.dataset_paths
        ))
        object.__setattr__(self, "image_members", tuple(self.image_members))
        object.__setattr__(self, "external_members", tuple(self.external_members))
        if min(int(self.size), int(self.mtime_ns), int(self.extent),
               int(self.generation)) < 0:
            raise ValueError("source size/mtime/extent/generation must be non-negative")
        if self.digest is not None and not str(self.digest).strip():
            raise ValueError("a supplied source digest may not be empty")
        if any(not path.strip() for path in self.dataset_paths):
            raise ValueError("source dataset paths may not be empty")
        if self.image_members and self.external_members:
            raise ValueError("image-series and external members are distinct owners")
        members = self.external_members or self.image_members
        _require_member_partition(members, int(self.extent))

@dataclass(frozen=True)
class AppendIntent:
    entry: str
    source_base: str
    source_identity: str
    science_fingerprint: str
    modes: tuple[str, ...]
    source: AppendSource
    labels: tuple[int, ...]
    def __post_init__(self) -> None:
        object.__setattr__(self, "entry", str(self.entry))
        object.__setattr__(self, "source_base", (
            _normalize(self.source_base) if str(self.source_base).strip() else ""
        ))
        object.__setattr__(self, "source_identity", str(self.source_identity))
        object.__setattr__(self, "science_fingerprint", str(self.science_fingerprint))
        object.__setattr__(self, "modes", tuple(str(mode) for mode in self.modes))
        object.__setattr__(self, "labels", tuple(int(label) for label in self.labels))
        if not self.entry or not self.source_identity or not self.science_fingerprint:
            raise ValueError("Append entry/source/science identities are required")
        if not self.modes or len(self.modes) != len(set(self.modes)):
            raise ValueError("Append requires a non-empty unique mode tuple")
        _require_contiguous(self.labels, "proposed Append labels")

@dataclass(frozen=True, slots=True)
class AppendCommittedPrefix:
    target: str
    intent: AppendIntent
    lineage_json: str
    def __post_init__(self) -> None:
        if not str(self.target).strip():
            raise ValueError("committed Append prefix target is required")
        if type(self.intent) is not AppendIntent:
            raise TypeError("committed Append prefix requires an exact AppendIntent")
        lineage = json.loads(str(self.lineage_json))
        if not isinstance(lineage, dict):
            raise ValueError("committed Append prefix lineage is not an object")
        if (type(lineage.get("version")) is not int or lineage.get("version") != LINEAGE_VERSION
                or lineage.get("state") != "committed"):
            raise ValueError("committed Append prefix lineage is not committed")
        labels = _lineage_labels(lineage)
        epochs = lineage.get("epochs") or ()
        if labels != self.intent.labels or not epochs:
            raise ValueError("committed Append prefix labels are inconsistent")
        lineage_identity = (lineage.get("entry"), lineage.get("source_base"),
                            lineage.get("source_identity"), lineage.get("science_fingerprint"),
                            tuple(lineage.get("modes") or ()))
        if lineage_identity != _intent_identity(self.intent):
            raise ValueError("committed Append prefix identity is inconsistent")
        if _source_dict(_source_from_dict(epochs[-1]["source"])) != _source_dict(self.intent.source):
            raise ValueError("committed Append prefix source is inconsistent")
        object.__setattr__(self, "target", _normalize(self.target))
        object.__setattr__(self, "lineage_json", _json(lineage))
    @property
    def committed_labels(self) -> tuple[int, ...]:
        return self.intent.labels
@dataclass(frozen=True)
class AppendDecision:
    disposition: AppendDisposition
    labels: tuple[int, ...]
    committed_labels: tuple[int, ...]
    reason: str
    lineage_json: str | None
    source_generation: int = 0
    @property
    def skip_labels(self) -> tuple[int, ...]:
        return self.committed_labels
    @property
    def write_labels(self) -> tuple[int, ...]:
        return self.labels
    @property
    def lineage(self) -> Mapping[str, Any] | None:
        if self.lineage_json is None:
            return None
        return json.loads(self.lineage_json)

@dataclass(frozen=True, slots=True)
class AppendPreflightSnapshot:
    disposition: AppendDisposition
    skip_labels: tuple[int, ...]
    write_labels: tuple[int, ...]
    source_generation: int
    state: AppendPreflightState
    reason: str = ""
@dataclass(frozen=True, slots=True)
class _AppendPreflightBinding:
    transaction: Any
    lease: Any
    transaction_owner: Any
    target_owner: Any
    owners: dict[Any, Any]
    decision: AppendDecision
class AppendPreflight:
    __slots__ = (
        "_target", "_intent", "_transaction", "_lease", "_transaction_owner",
        "_target_owner", "_owners", "_decision", "_committed_prefix",
        "_file_lock", "_state", "_consumer",
    )
    def __init__(self, *args, **kwargs) -> None:
        if kwargs.pop("_factory", None) is not _PREFLIGHT_FACTORY:
            raise TypeError("AppendPreflight values are created by prepare_append_preflight")
        (
            self._target, self._intent, self._transaction, self._lease,
            self._transaction_owner, self._target_owner, self._owners,
            self._decision, self._committed_prefix, self._file_lock,
        ) = args
        self._state = AppendPreflightState.RESERVED
        self._consumer: Callable[[AppendDecision], None] | None = None
    def __copy__(self):
        raise TypeError("AppendPreflight authority cannot be copied")
    def __deepcopy__(self, _memo):
        raise TypeError("AppendPreflight authority cannot be copied")

    @property
    def target(self) -> Path:
        return Path(self._target)

    @property
    def snapshot(self) -> AppendPreflightSnapshot:
        decision = self._decision
        return AppendPreflightSnapshot(
            decision.disposition,
            decision.skip_labels,
            decision.write_labels,
            decision.source_generation,
            self._state,
            decision.reason,
        )

    def _release(self) -> None:
        for role in tuple(self._owners):
            self._transaction.release_lease_owner(
                self._lease, role, self._owners[role]
            )
            del self._owners[role]

    def _consume(
        self,
        target: str | Path,
        *,
        consumer: Callable[[AppendDecision], None] | None = None,
    ) -> _AppendPreflightBinding:
        if self._state is not AppendPreflightState.RESERVED:
            raise RuntimeError("Append preflight was already consumed or settled")
        if _normalize(target) != self._target:
            raise ValueError("Append preflight names a different output target")
        if self._decision.disposition is not AppendDisposition.WRITE:
            raise AppendRefused(self._decision)
        self._state = AppendPreflightState.BOUND
        self._consumer = consumer
        return _AppendPreflightBinding(
            self._transaction,
            self._lease,
            self._transaction_owner,
            self._target_owner,
            self._owners,
            self._decision,
        )

    def _set_consumer(self, consumer: Callable[[AppendDecision], None]) -> None:
        if self._state is not AppendPreflightState.BOUND:
            raise RuntimeError("Append consumer requires the exact bound preflight")
        if self._consumer is not None and self._consumer is not consumer:
            raise RuntimeError("Append preflight already has a different consumer")
        self._consumer = consumer

    def _commit_epoch(self, decision: AppendDecision) -> None:
        if self._state is not AppendPreflightState.BOUND:
            raise RuntimeError("Append epoch commit requires the bound preflight")
        expected = self._decision.skip_labels + self._decision.write_labels
        if decision.write_labels or decision.skip_labels != expected:
            raise RuntimeError("Append epoch anchor does not match the pending epoch")
        self._decision = decision

    def extend(self, intent: AppendIntent) -> AppendPreflightSnapshot:
        if self._state not in {
            AppendPreflightState.RESERVED, AppendPreflightState.BOUND,
        }:
            raise RuntimeError("terminal Append preflight cannot extend")
        if intent == self._intent:
            return self.snapshot
        if int(intent.source.generation) <= int(self._intent.source.generation):
            raise AppendRefused(_decision(
                AppendDisposition.REFUSE,
                reason="same-run source generation did not advance",
                source_generation=intent.source.generation,
            ))
        decision = _extend_pending_decision(self._decision, self._intent, intent)
        if decision.disposition is AppendDisposition.REFUSE:
            raise AppendRefused(decision)
        if self._state is AppendPreflightState.RESERVED:
            with (nullcontext() if self._file_lock is None else self._file_lock):
                if self._committed_prefix is None:
                    decision = qualify_append(self._target, intent)
                else:
                    decision = qualify_append(
                        self._target, intent,
                        committed_prefix=self._committed_prefix,
                    )
        if decision.disposition is AppendDisposition.REFUSE:
            raise AppendRefused(decision)
        self._intent = intent
        self._decision = decision
        if self._consumer is not None:
            self._consumer(decision)
        return self.snapshot

    def truncate(self, written_labels: Iterable[int]) -> AppendPreflightSnapshot:
        if self._state is not AppendPreflightState.BOUND:
            raise RuntimeError("only a bound Append preflight can truncate")
        self._decision, self._intent = truncate_append_epoch(
            self._decision, self._intent, written_labels)
        if self._consumer is not None:
            self._consumer(self._decision)
        return self.snapshot

    def complete_noop(self) -> AppendPreflightSnapshot:
        if (self._state is not AppendPreflightState.RESERVED
                or self._decision.disposition is not AppendDisposition.SKIP):
            raise RuntimeError("only an unbound exact no-op preflight may complete")
        try:
            with (nullcontext() if self._file_lock is None else self._file_lock):
                self._transaction.abandon(self._lease)
            self._release()
        except BaseException:
            self._state = self._cleanup_failure_state()
            raise
        self._state = AppendPreflightState.NOOP
        return self.snapshot

    def abort(self) -> AppendPreflightSnapshot:
        if self._state is AppendPreflightState.RESERVED:
            try:
                with (nullcontext() if self._file_lock is None else self._file_lock):
                    self._transaction.abandon(self._lease)
                self._release()
            except BaseException:
                self._state = self._cleanup_failure_state()
                raise
            self._state = AppendPreflightState.ABORTED
            return self.snapshot
        if self._state is AppendPreflightState.BOUND:
            raise RuntimeError("bound Append preflight must abort through its sink")
        return self.snapshot

    def retry_cleanup(self) -> AppendPreflightSnapshot:
        if self._state is not AppendPreflightState.RETRYABLE:
            return self.snapshot
        try:
            with (nullcontext() if self._file_lock is None else self._file_lock):
                snapshot = self._transaction.snapshot()
                if snapshot.pending_actions:
                    if snapshot.cleanup_token is None:
                        raise RuntimeError("Append cleanup has no exact retry owner")
                    self._transaction.retry_cleanup(snapshot.cleanup_token)
                if self._transaction.snapshot().phase.value not in {
                    "aborted", "committed",
                }:
                    self._transaction.abandon(self._lease)
            self._release()
        except BaseException:
            self._state = self._cleanup_failure_state()
            raise
        phase = self._transaction.snapshot().phase.value
        self._state = (AppendPreflightState.COMMITTED if phase == "committed"
                       else AppendPreflightState.NOOP
                       if self._decision.disposition is AppendDisposition.SKIP
                       else AppendPreflightState.ABORTED)
        return self.snapshot

    def _terminal(self, state: AppendPreflightState) -> None:
        self._state = state

    def _cleanup_failure_state(self) -> AppendPreflightState:
        if self._transaction.snapshot().phase.value == "integrity-hold":
            return AppendPreflightState.INTEGRITY_HOLD
        return AppendPreflightState.RETRYABLE

_PREFLIGHT_FACTORY = object()

def truncate_append_source(source: AppendSource, extent: int) -> AppendSource:
    if extent < 0 or extent > int(source.extent):
        raise ValueError("truncated source extent is outside the observed epoch")

    def truncate_members(members):
        out = []
        for member in members:
            if int(member.source_start) >= extent:
                break
            stop = min(int(member.source_stop), extent)
            values = asdict(member)
            values["source_stop"] = stop
            out.append(type(member)(**values))
            if stop == extent:
                break
        return tuple(out)

    image_members = truncate_members(source.image_members)
    external_members = truncate_members(source.external_members)
    dataset_paths = source.dataset_paths
    if external_members and len(dataset_paths) > len(external_members):
        dataset_paths = dataset_paths[:len(external_members)]
    return AppendSource(
        path=source.path,
        adapter_id=source.adapter_id,
        size=source.size,
        mtime_ns=source.mtime_ns,
        extent=extent,
        digest=source.digest,
        dataset_paths=dataset_paths,
        image_members=image_members,
        external_members=external_members,
        generation=source.generation,
    )

def _normalize(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _normalize_base(path: str | Path) -> str:
    return "" if not str(path).strip() else _normalize(path)

def _require_member_partition(members, extent: int) -> None:
    ordinals = tuple(int(member.ordinal) for member in members)
    if ordinals != tuple(range(len(ordinals))):
        raise ValueError("source members require exact ordered ordinals")
    cursor = 0
    for member in members:
        if int(member.source_start) != cursor:
            raise ValueError("source member ranges must be contiguous from zero")
        cursor = int(member.source_stop)
    if members and cursor != int(extent):
        raise ValueError("source member ranges must end at the declared extent")

def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))

def _source_dict(source: AppendSource) -> dict[str, Any]:
    return {
        "path": source.path,
        "adapter_id": source.adapter_id,
        "size": int(source.size),
        "mtime_ns": int(source.mtime_ns),
        "extent": int(source.extent),
        "digest": source.digest,
        "dataset_paths": list(source.dataset_paths),
        "image_members": [asdict(member) for member in source.image_members],
        "external_members": [asdict(member) for member in source.external_members],
        "generation": int(source.generation),
    }

def _intent_identity(intent: AppendIntent) -> tuple[Any, ...]:
    return (intent.entry, intent.source_base, intent.source_identity, intent.science_fingerprint, intent.modes)
def _source_from_dict(raw_source: Mapping[str, Any]) -> AppendSource:
    numbers = [raw_source.get(key) for key in ("size", "mtime_ns", "extent", "generation")]
    numbers.extend(member.get(key) for kind in ("image_members", "external_members")
                   for member in raw_source.get(kind, ())
                   for key in ("size", "mtime_ns", "source_start", "source_stop", "ordinal"))
    if any(type(value) is not int for value in numbers):
        raise ValueError("Append source/member integers require exact JSON integers")
    values = dict(raw_source)
    values["image_members"] = tuple(
        AppendImageMember(**item) for item in raw_source["image_members"])
    values["external_members"] = tuple(
        AppendExternalMember(**item) for item in raw_source["external_members"])
    return AppendSource(**values)
def _base_lineage(intent: AppendIntent) -> dict[str, Any]:
    return {
        "version": LINEAGE_VERSION,
        "state": "pending",
        "entry": intent.entry,
        "source_base": intent.source_base,
        "source_identity": intent.source_identity,
        "science_fingerprint": intent.science_fingerprint,
        "modes": list(intent.modes),
        "epochs": [],
    }

def _decision(
    disposition: AppendDisposition,
    labels: Iterable[int] = (),
    committed_labels: Iterable[int] = (),
    reason: str = "",
    lineage: Mapping[str, Any] | None = None,
    source_generation: int = 0,
) -> AppendDecision:
    return AppendDecision(
        disposition,
        tuple(int(label) for label in labels),
        tuple(int(label) for label in committed_labels),
        str(reason),
        None if lineage is None else _json(lineage),
        int(source_generation),
    )

def _require_contiguous(labels: tuple[int, ...], role: str) -> None:
    if len(labels) != len(set(labels)):
        raise ValueError(f"{role} contain duplicates")
    if labels and labels != tuple(range(labels[0], labels[-1] + 1)):
        raise ValueError(f"{role} are gapped or unordered")

def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return value.item() if hasattr(value, "item") else value

def _mode_group(entry: h5py.Group, mode: str) -> h5py.Group:
    try:
        dimension, mode_key = mode.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"malformed Append mode {mode!r}") from exc
    if dimension not in {"1d", "2d"}:
        raise ValueError(f"unknown Append dimension {dimension!r}")
    top_name = f"integrated_{dimension}"
    top = entry.get(top_name)
    if not isinstance(top, h5py.Group):
        raise ValueError(f"missing {top_name} group")
    primary = str(_decode(top.attrs.get(PRIMARY_MODE_ATTR, DEFAULT_MODE_KEY)))
    if mode_key == primary:
        return top
    try:
        subgroup = MODE_SUBGROUP_NAMES[mode_key]
    except KeyError as exc:
        raise ValueError(f"unknown Append mode key {mode_key!r}") from exc
    group = top.get(subgroup)
    if not isinstance(group, h5py.Group):
        raise ValueError(f"missing persisted mode group {top_name}/{subgroup}")
    return group

def _disk_labels(entry: h5py.Group, modes: tuple[str, ...]) -> tuple[int, ...]:
    selected: tuple[int, ...] | None = None
    for mode in modes:
        group = _mode_group(entry, mode)
        dataset = group.get("frame_index")
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1:
            raise ValueError(f"{group.name}/frame_index is not a rank-1 dataset")
        if dataset.dtype.kind not in "iu":
            raise ValueError(f"{group.name}/frame_index is not integral")
        labels = tuple(int(value) for value in np.asarray(dataset[()]).ravel())
        _require_contiguous(labels, f"{group.name}/frame_index")
        if selected is None:
            selected = labels
        elif labels != selected:
            raise ValueError("configured mode frame histories differ")
    return selected or ()

def _read_lineage(entry: h5py.Group) -> dict[str, Any]:
    dataset = entry.get(f"reduction/config/{LINEAGE_DATASET}")
    if not isinstance(dataset, h5py.Dataset) or dataset.shape != ():
        raise ValueError("missing or malformed committed Append lineage")
    raw = _decode(dataset[()])
    value = json.loads(str(raw))
    if not isinstance(value, dict):
        raise ValueError("Append lineage is not an object")
    return value

def _member_extends(
    prior: list[dict[str, Any]],
    current: list[dict[str, Any]],
    *,
    grow_last: bool,
) -> bool:
    if len(current) < len(prior):
        return False
    if not prior:
        return True
    for index, old in enumerate(prior):
        new = current[index]
        if index < len(prior) - 1:
            if new != old:
                return False
            continue
        if not grow_last:
            if new != old:
                return False
            continue
        fixed = ("path", "dataset_path", "source_start", "ordinal")
        if any(new.get(key) != old.get(key) for key in fixed):
            return False
        if any(int(new.get(key, -1)) < int(old.get(key, -1))
               for key in ("size", "mtime_ns", "source_stop")):
            return False
    return True

def _source_extends(prior: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    for key in ("path", "adapter_id"):
        if prior.get(key) != current.get(key):
            raise ValueError(f"source does not extend exact {key}")
    prior_paths = list(prior.get("dataset_paths") or ())
    current_paths = list(current.get("dataset_paths") or ())
    if current_paths[:len(prior_paths)] != prior_paths:
        raise ValueError("source does not extend exact dataset paths")
    for key in ("size", "mtime_ns", "extent"):
        if int(current.get(key, -1)) < int(prior.get(key, -1)):
            raise ValueError(f"source {key} regressed")
    if prior.get("digest") is not None and current.get("digest") is None:
        raise ValueError("source digest evidence was downgraded")
    prior_images = list(prior.get("image_members") or ())
    current_images = list(current.get("image_members") or ())
    if not _member_extends(prior_images, current_images, grow_last=False):
        raise ValueError("source does not extend exact image-member history")
    prior_external = list(prior.get("external_members") or ())
    current_external = list(current.get("external_members") or ())
    if not _member_extends(prior_external, current_external, grow_last=True):
        raise ValueError("source does not extend exact external-member history")
    if int(current.get("extent", -1)) == int(prior.get("extent", -1)):
        if ({key: value for key, value in current.items() if key != "generation"}
                != {key: value for key, value in prior.items()
                    if key != "generation"}):
            raise ValueError("same-extent source identity changed")

def _extend_pending_decision(
    prior_decision: AppendDecision,
    prior_intent: AppendIntent,
    current_intent: AppendIntent,
) -> AppendDecision:
    immutable = (
        "entry", "source_base", "source_identity", "science_fingerprint", "modes",
    )
    if any(getattr(prior_intent, key) != getattr(current_intent, key)
           for key in immutable):
        return _decision(
            AppendDisposition.REFUSE,
            reason="live source extension changed Append identity",
            source_generation=current_intent.source.generation,
        )
    try:
        old_labels = prior_intent.labels
        new_labels = current_intent.labels
        if new_labels[:len(old_labels)] != old_labels or len(new_labels) < len(old_labels):
            raise ValueError("live source extension remapped requested labels")
        _source_extends(
            _source_dict(prior_intent.source),
            _source_dict(current_intent.source),
        )
        committed = prior_decision.skip_labels
        if new_labels[:len(committed)] != committed:
            raise ValueError("live source extension remapped committed prefix")
        write = new_labels[len(committed):]
        lineage = dict(prior_decision.lineage or {})
        epochs = list(lineage.get("epochs") or ())
        pending = {
            "source": _source_dict(current_intent.source),
            "labels": list(write),
        }
        if epochs and lineage.get("state") == "pending":
            epochs[-1] = pending
        else:
            epochs.append(pending)
        lineage["state"] = "pending"
        lineage["epochs"] = epochs
        return _decision(
            AppendDisposition.WRITE,
            write,
            committed,
            lineage=lineage,
            source_generation=current_intent.source.generation,
        )
    except (ValueError, TypeError, KeyError) as exc:
        return _decision(
            AppendDisposition.REFUSE,
            reason=str(exc),
            source_generation=current_intent.source.generation,
        )

def begin_same_run_lineage(intent: AppendIntent) -> AppendDecision:
    if len(intent.labels) > int(intent.source.extent):
        return _decision(AppendDisposition.REFUSE, reason="labels exceed source extent")
    if not intent.labels:
        return _decision(
            AppendDisposition.SKIP, reason="absent source has no labels",
            source_generation=intent.source.generation,
        )
    lineage = _base_lineage(intent)
    lineage["epochs"].append(
        {"source": _source_dict(intent.source), "labels": list(intent.labels)})
    return _decision(
        AppendDisposition.WRITE, intent.labels, lineage=lineage,
        source_generation=intent.source.generation,
    )

def extend_same_run_lineage(
    decision: AppendDecision,
    prior_intent: AppendIntent,
    intent: AppendIntent,
) -> AppendDecision:
    if intent == prior_intent:
        return decision
    if int(intent.source.generation) <= int(prior_intent.source.generation):
        return _decision(
            AppendDisposition.REFUSE,
            reason="same-run source generation did not advance",
            source_generation=intent.source.generation,
        )
    return _extend_pending_decision(decision, prior_intent, intent)

def truncate_append_epoch(
    decision: AppendDecision,
    intent: AppendIntent,
    written_labels: Iterable[int],
) -> tuple[AppendDecision, AppendIntent]:
    labels = tuple(int(label) for label in written_labels)
    _require_contiguous(labels, "graceful Stop labels")
    if decision.write_labels[:len(labels)] != labels:
        raise ValueError("graceful Stop labels are not the pending prefix")
    committed = decision.skip_labels
    written = committed + labels
    extent = intent.source.extent if written == intent.labels else len(written)
    source = truncate_append_source(intent.source, extent)
    lineage = dict(decision.lineage or {})
    epochs = list(lineage.get("epochs") or ())
    if not epochs:
        raise ValueError("pending Append lineage has no epoch to truncate")
    epochs[-1] = {"source": _source_dict(source), "labels": list(labels)}
    lineage.update(state="pending", epochs=epochs)
    truncated = _decision(
        AppendDisposition.WRITE, labels, committed, lineage=lineage,
        source_generation=source.generation,
    )
    return truncated, AppendIntent(
        intent.entry, intent.source_base, intent.source_identity,
        intent.science_fingerprint, intent.modes, source, committed + labels,
    )
def seal_append_epoch(decision: AppendDecision) -> AppendDecision:
    if decision.disposition is not AppendDisposition.WRITE or decision.lineage is None:
        raise AppendRefused(decision)
    committed = decision.skip_labels + decision.write_labels
    lineage = dict(decision.lineage)
    if _lineage_labels(lineage) != committed:
        raise ValueError("Append epoch lineage does not match its exact labels")
    lineage["state"] = "committed"
    return _decision(
        AppendDisposition.WRITE,
        (),
        committed,
        lineage=lineage,
        source_generation=decision.source_generation,
    )
def _lineage_labels(lineage: Mapping[str, Any]) -> tuple[int, ...]:
    epochs = lineage.get("epochs")
    if not isinstance(epochs, list):
        raise ValueError("Append lineage epochs are malformed")
    flattened: list[int] = []
    prior_source = None
    for epoch in epochs:
        if not isinstance(epoch, dict) or not isinstance(epoch.get("source"), dict):
            raise ValueError("Append epoch is malformed")
        raw_source = epoch["source"]
        source = _source_from_dict(raw_source)
        if _json(_source_dict(source)) != _json(raw_source):
            raise ValueError("Append epoch source is not canonical")
        if prior_source is not None:
            _source_extends(_source_dict(prior_source), raw_source)
        prior_source = source
        raw_labels = epoch.get("labels")
        if (not isinstance(raw_labels, list)
                or any(type(label) is not int for label in raw_labels)):
            raise ValueError("Append epoch labels require exact JSON integers")
        labels = tuple(raw_labels)
        _require_contiguous(labels, "Append epoch labels")
        if flattened and labels and labels[0] != flattened[-1] + 1:
            raise ValueError("Append epochs are gapped or overlapping")
        flattened.extend(labels)
    values = tuple(flattened)
    _require_contiguous(values, "Append lineage labels")
    return values

def decode_committed_append_prefix(
    handle: h5py.File,
    *,
    entry: str = "entry",
) -> AppendCommittedPrefix:
    if not isinstance(handle, h5py.File) or not handle.id.valid:
        raise TypeError("committed Append prefix requires an open h5py.File")
    group = handle.get(entry)
    if not isinstance(group, h5py.Group):
        raise ValueError(f"foreign or missing entry {entry!r}")
    if str(_decode(group.attrs.get(SCHEMA_NAME_ATTR, ""))) not in ACCEPTED_SCHEMA_NAMES:
        raise ValueError("foreign processed schema identity")
    if int(group.attrs.get(SCHEMA_VERSION_ATTR, -1)) != PROCESSED_SCHEMA_VERSION:
        raise ValueError("foreign processed schema version")
    stored_base = _normalize_base(str(_decode(group.attrs.get(SOURCE_BASE_ATTR, ""))))
    lineage = _read_lineage(group)
    if lineage.get("entry") != entry:
        raise ValueError("foreign Append entry")
    if lineage.get("source_base") != stored_base:
        raise ValueError("wrong source base")
    modes = lineage.get("modes")
    if (not isinstance(modes, list) or not modes
            or any(not isinstance(mode, str) for mode in modes)):
        raise ValueError("malformed Append modes")
    epochs = lineage.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise ValueError("existing target has no committed Append epoch")
    labels = _lineage_labels(lineage)
    if _disk_labels(group, tuple(modes)) != labels:
        raise ValueError("disk rows do not match exact committed lineage")
    source = _source_from_dict(epochs[-1]["source"])
    intent = AppendIntent(
        entry, stored_base, lineage.get("source_identity", ""),
        lineage.get("science_fingerprint", ""), tuple(modes), source, labels,
    )
    return AppendCommittedPrefix(_normalize(handle.filename), intent, _json(lineage))

def qualify_append(
    target: str | Path,
    intent: AppendIntent,
    *,
    committed_prefix: AppendCommittedPrefix | None = None,
) -> AppendDecision:
    target = Path(target)
    try:
        normalized = _normalize(target)
        if committed_prefix is not None:
            if type(committed_prefix) is not AppendCommittedPrefix:
                raise TypeError("committed prefix must be an exact AppendCommittedPrefix")
            if committed_prefix.target != normalized:
                raise ValueError("committed Append prefix names a different target")
            observed = committed_prefix.intent
            if _intent_identity(observed) != _intent_identity(intent):
                raise ValueError("committed Append prefix identity changed")
            if intent.labels[:len(observed.labels)] != observed.labels:
                raise ValueError("requested labels remap the committed Append prefix")
            _source_extends(_source_dict(observed.source), _source_dict(intent.source))
        if not target.exists():
            if committed_prefix is not None:
                raise ValueError("committed Append prefix target disappeared")
            return begin_same_run_lineage(intent)

        with h5py.File(target, "r") as handle:
            current = decode_committed_append_prefix(handle, entry=intent.entry)
        if committed_prefix is not None:
            observed_lineage = json.loads(committed_prefix.lineage_json)
            current_lineage = json.loads(current.lineage_json)
            observed_epochs = observed_lineage["epochs"]
            if current.committed_labels[:len(committed_prefix.committed_labels)] \
                    != committed_prefix.committed_labels:
                raise ValueError("committed Append labels no longer retain observed prefix")
            if _json({"epochs": current_lineage["epochs"][:len(observed_epochs)]}) \
                    != _json({"epochs": observed_epochs}):
                raise ValueError("committed Append epochs diverged from observed prefix")
        if _intent_identity(current.intent) != _intent_identity(intent):
            raise ValueError("foreign Append identity")
        labels = current.committed_labels
        lineage = json.loads(current.lineage_json)
        prior_source = _source_dict(current.intent.source)
        current_source = _source_dict(intent.source)
        _source_extends(prior_source, current_source)
        prior_extent = int(prior_source.get("extent", -1))
        if intent.labels[:len(labels)] != labels:
            raise ValueError("requested labels do not preserve the committed prefix")
        suffix = intent.labels[len(labels):]
        if int(intent.source.extent) == prior_extent:
            if suffix:
                raise ValueError("same-extent source proposed new labels")
            return _decision(
                AppendDisposition.SKIP,
                committed_labels=labels,
                reason="exact source already committed",
                source_generation=intent.source.generation,
            )
        if not suffix:
            raise ValueError("growing source supplied no new labels")
        expected_start = labels[-1] + 1 if labels else suffix[0]
        if suffix[0] != expected_start:
            raise ValueError("new labels overlap or gap committed rows")
        if int(intent.source.extent) - prior_extent != len(suffix):
            raise ValueError("source extent growth does not match selected labels")
        pending = dict(lineage)
        pending["state"] = "pending"
        pending["epochs"] = list(lineage["epochs"]) + [
            {"source": current_source, "labels": list(suffix)}
        ]
        return _decision(
            AppendDisposition.WRITE,
            suffix,
            labels,
            lineage=pending,
            source_generation=intent.source.generation,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return _decision(
            AppendDisposition.REFUSE,
            reason=str(exc),
            source_generation=intent.source.generation,
        )

def prepare_append_preflight(
    target: str | Path,
    intent: AppendIntent,
    *,
    committed_prefix: AppendCommittedPrefix | None = None,
    file_lock=None,
    refusal_mapper: Callable[[str, AppendDecision], BaseException] | None = None,
) -> AppendPreflight:
    from .output_transaction import (
        LeaseOwner,
        OwnerToken,
        get_output_transaction_coordinator,
    )

    normalized = _normalize(target)
    coordinator = get_output_transaction_coordinator()
    transaction_owner = OwnerToken("append-preflight-transaction")
    target_owner = OwnerToken("append-preflight-target")
    owners = {
        role: OwnerToken(f"append-preflight-{role.value}") for role in LeaseOwner
    }
    transaction = None
    lease = None
    decision = None
    try:
        with (nullcontext() if file_lock is None else file_lock):
            transaction = coordinator.admit(
                normalized,
                transaction_owner=transaction_owner,
                target_owner=target_owner,
            )
            lease = transaction.acquire_lease(
                admission=transaction.admission,
                transaction_owner=transaction_owner,
                target_owner=target_owner,
                owners=owners,
            )
            if committed_prefix is None:
                decision = qualify_append(normalized, intent)
            else:
                decision = qualify_append(
                    normalized, intent, committed_prefix=committed_prefix,
                )
            if decision.disposition is AppendDisposition.REFUSE:
                error = AppendRefused(decision)
                if refusal_mapper is not None:
                    error = refusal_mapper(normalized, decision)
                    if not isinstance(error, BaseException):
                        raise TypeError("Append refusal mapper must return an exception")
                raise error
    except BaseException as primary:
        if transaction is None or lease is None:
            raise
        try:
            with (nullcontext() if file_lock is None else file_lock):
                transaction.abandon(lease)
            for role in tuple(owners):
                transaction.release_lease_owner(lease, role, owners[role])
                del owners[role]
        except BaseException as cleanup:
            retained = decision
            if retained is None and isinstance(primary, AppendRefused):
                retained = primary.decision
            if retained is None:
                retained = _decision(
                    AppendDisposition.REFUSE,
                    reason=str(primary),
                    source_generation=intent.source.generation,
                )
            owner = AppendPreflight(
                normalized, intent, transaction, lease, transaction_owner,
                target_owner, owners, retained, committed_prefix, file_lock,
                _factory=_PREFLIGHT_FACTORY,
            )
            owner._state = owner._cleanup_failure_state()
            raise AppendPreflightCleanupError(primary, owner) from cleanup
        raise
    return AppendPreflight(
        normalized,
        intent,
        transaction,
        lease,
        transaction_owner,
        target_owner,
        owners,
        decision,
        committed_prefix,
        file_lock,
        _factory=_PREFLIGHT_FACTORY,
    )

def _write_lineage(entry: h5py.Group, lineage: Mapping[str, Any]) -> None:
    reduction = entry.require_group("reduction")
    config = reduction.require_group("config")
    if LINEAGE_DATASET in config:
        del config[LINEAGE_DATASET]
    config.create_dataset(LINEAGE_DATASET, data=_json(lineage))

def stage_append_lineage(entry: h5py.Group, decision: AppendDecision) -> None:
    if decision.disposition is not AppendDisposition.WRITE or decision.lineage is None:
        raise AppendRefused(decision)
    lineage = dict(decision.lineage)
    lineage["state"] = "pending"
    _write_lineage(entry, lineage)

def commit_append_lineage(
    entry: h5py.Group,
    decision: AppendDecision,
    *,
    written_labels: Iterable[int],
) -> None:
    if decision.disposition is not AppendDisposition.WRITE or decision.lineage is None:
        raise AppendRefused(decision)
    labels = tuple(int(label) for label in written_labels)
    if labels != decision.labels:
        raise ValueError(
            f"Append wrote labels {labels}, expected exact epoch {decision.labels}"
        )
    lineage = dict(decision.lineage)
    lineage["state"] = "committed"
    _write_lineage(entry, lineage)

def science_fingerprint(value: Any) -> str:
    def normalize(item: Any):
        if is_dataclass(item):
            return {field.name: normalize(getattr(item, field.name))
                    for field in fields(item)}
        if type(item) is object:
            return {"__sentinel__": "unset"}
        if isinstance(item, Enum):
            return normalize(item.value)
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Mapping):
            return {str(key): normalize(val) for key, val in sorted(
                item.items(), key=lambda pair: str(pair[0]))}
        if isinstance(item, (list, tuple)):
            return [normalize(part) for part in item]
        if isinstance(item, np.ndarray):
            return normalize(item.tolist())
        if isinstance(item, np.generic):
            return item.item()
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
        raise TypeError(
            f"unsupported science fingerprint value: {type(item).__qualname__}")

    payload = json.dumps(normalize(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

__all__ = [
    "AppendCommittedPrefix",
    "AppendDecision",
    "AppendDisposition",
    "AppendExternalMember",
    "AppendImageMember",
    "AppendIntent",
    "AppendPreflight",
    "AppendPreflightCleanupError",
    "AppendPreflightSnapshot",
    "AppendPreflightState",
    "AppendRefused",
    "AppendSource",
    "AppendSourceGraphRefused",
    "LINEAGE_DATASET",
    "LINEAGE_VERSION",
    "begin_same_run_lineage",
    "commit_append_lineage",
    "decode_committed_append_prefix",
    "extend_same_run_lineage",
    "prepare_append_preflight",
    "qualify_append",
    "science_fingerprint",
    "seal_append_epoch",
    "stage_append_lineage",
    "truncate_append_epoch",
    "truncate_append_source",
]
