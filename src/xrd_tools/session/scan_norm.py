# -*- coding: utf-8 -*-
"""Frozen whole-scan normalization aggregate (Slice-5 canonical value).

``ScanNormAggregate`` is the headless shared value behind normalization
channel choices: an exact display-context/scan identity, a monotonic
producer revision, ``row_count``, and per-channel
``{canonical_key: (finite_sum, finite_count)}`` statistics folded under the one
ABSENT rule (:func:`xrd_tools.core.metadata.resolve_monitor_norm` — a row
contributes to a channel only when the kernel yields a finite positive
value; every folded row contributes to ``row_count``).

Ownership boundaries: this module owns the VALUE, its incremental fold and
the consumer acceptance gate only.  There is no store, no reader and no
GUI wiring here — producers fold on the existing whole-scan passes and
consumers derive choices from an accepted aggregate revision (adoption is
E6-parented work).  A consumer accepts only a monotonically newer revision
of the exact current identity; an equal-revision replay is an idempotent
no-op; an aggregate for any other identity is rejected outright and never
means "reread scan_data".

Identity is an opaque tuple of primitives supplied by the producer.  The
two documented shapes are the acquisition identity
``(generation, fingerprint, artifact, source_scan)`` and the Browse
identity ``(context_token, scan_key, requested_path)``; this module never
imports the GUI types behind them.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from xrd_tools.core.metadata import numeric_metadata, resolve_monitor_norm

__all__ = [
    "ScanNormAggregate",
    "accepts_norm_aggregate",
    "channel_is_partial",
    "empty_norm_aggregate",
    "fold_norm_metadata",
    "next_norm_revision",
]

_IDENTITY_PRIMITIVES = (str, int, float, bool, type(None))


def _canonical_channel_key(key: str) -> str:
    """The exact case-equivalence spelling used by ``resolve_monitor_norm``."""

    return key.lower()


def _validated_identity(identity: Any) -> tuple[Any, ...]:
    if isinstance(identity, (str, bytes)) or not isinstance(identity, Sequence):
        raise TypeError(
            "identity must be a non-string sequence of primitives, got "
            f"{type(identity).__name__}"
        )
    values = tuple(identity)
    for element in values:
        if not isinstance(element, _IDENTITY_PRIMITIVES):
            raise TypeError(
                "identity elements must be str/int/float/bool/None, got "
                f"{type(element).__name__}"
            )
    return values


@dataclass(frozen=True)
class ScanNormAggregate:
    """Immutable per-identity normalization statistics.

    ``channels`` maps the kernel's deterministic lowercase metadata key to
    ``(finite_sum, finite_count)`` where ``1 <= finite_count <= row_count``
    and ``finite_sum`` is a finite positive float; a channel key exists
    only once at least one row passed the ABSENT-rule kernel.  Metadata keys
    that differ only by the kernel's case-insensitive equivalence belong to
    one channel and contribute at most once per row.
    """

    identity: tuple[Any, ...]
    revision: int
    row_count: int
    channels: Mapping[str, tuple[float, int]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _validated_identity(self.identity))
        for name in ("revision", "row_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if not isinstance(self.channels, Mapping):
            raise TypeError(
                f"channels must be a mapping, got {type(self.channels).__name__}"
            )
        frozen: dict[str, tuple[float, int]] = {}
        for key, stat in self.channels.items():
            if not isinstance(key, str) or not key.strip():
                raise TypeError(f"channel keys must be non-empty str, got {key!r}")
            canonical_key = _canonical_channel_key(key)
            if canonical_key in frozen:
                raise ValueError(
                    "channel keys must be unique under case-insensitive "
                    f"normalization, got duplicate {key!r}"
                )
            try:
                total, count = stat
            except (TypeError, ValueError):
                raise ValueError(
                    f"channel {key!r} must map to (finite_sum, finite_count)"
                ) from None
            total = float(total)
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"channel {key!r} finite_count must be an int")
            if not isfinite(total) or total <= 0.0:
                raise ValueError(
                    f"channel {key!r} finite_sum must be finite and > 0, got {total!r}"
                )
            if not 1 <= count <= self.row_count:
                raise ValueError(
                    f"channel {key!r} finite_count must be in 1..row_count "
                    f"({self.row_count}), got {count}"
                )
            frozen[canonical_key] = (total, count)
        object.__setattr__(
            self,
            "channels",
            MappingProxyType(dict(sorted(frozen.items()))),
        )


def empty_norm_aggregate(identity: Sequence[Any]) -> ScanNormAggregate:
    """The revision-0 aggregate for one exact identity."""

    return ScanNormAggregate(_validated_identity(identity), 0, 0, {})


def fold_norm_metadata(
    aggregate: ScanNormAggregate, metadata: Mapping[str, Any] | None
) -> ScanNormAggregate:
    """Fold ONE row's metadata; returns a new value at the SAME revision.

    Every row increments ``row_count``.  A channel accumulates only when
    :func:`resolve_monitor_norm` yields a value for that canonical numeric key
    (the one ABSENT rule: absent, nonnumeric, non-finite, zero and
    negative all contribute nothing).
    """

    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError(
            f"metadata must be a mapping or None, got {type(metadata).__name__}"
        )
    folded = dict(aggregate.channels)
    canonical_keys = {
        _canonical_channel_key(key) for key in numeric_metadata(metadata)
    }
    for key in sorted(canonical_keys):
        value = resolve_monitor_norm(metadata, key)
        if value is None:
            continue
        total, count = folded.get(key, (0.0, 0))
        folded[key] = (total + value, count + 1)
    return ScanNormAggregate(
        aggregate.identity, aggregate.revision, aggregate.row_count + 1, folded
    )


def next_norm_revision(aggregate: ScanNormAggregate) -> ScanNormAggregate:
    """Publish: bump the monotonic revision, content unchanged."""

    return ScanNormAggregate(
        aggregate.identity,
        aggregate.revision + 1,
        aggregate.row_count,
        dict(aggregate.channels),
    )


def accepts_norm_aggregate(
    current: ScanNormAggregate | None,
    candidate: ScanNormAggregate,
    identity: Sequence[Any],
) -> bool:
    """Consumer gate: exact identity, monotonically newer revision.

    ``identity`` is the consumer's CURRENT exact context identity.  A
    candidate for any other identity is rejected.  With no held aggregate
    for this identity (``current`` is ``None`` or holds another identity —
    a context switch), any candidate of the exact identity is fresh.  An
    equal-revision replay is idempotent: not accepted, not an error.
    """

    if not isinstance(candidate, ScanNormAggregate):
        raise TypeError(
            f"candidate must be a ScanNormAggregate, got {type(candidate).__name__}"
        )
    expected = _validated_identity(identity)
    if candidate.identity != expected:
        return False
    if current is None or current.identity != expected:
        return True
    return candidate.revision > current.revision


def channel_is_partial(aggregate: ScanNormAggregate, key: str) -> bool:
    """True when the channel misses at least one folded row.

    Loud (``KeyError``) for a channel the aggregate does not carry — an
    ABSENT channel is a different state than a partial one.
    """

    return aggregate.channels[key][1] < aggregate.row_count
