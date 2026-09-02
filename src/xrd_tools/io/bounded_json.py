"""Exact, allocation-bounded admission for caller-owned JSON-shaped values."""

from __future__ import annotations

import json
import math
from typing import Literal


class BoundedJsonError(ValueError):
    """Typed refusal raised before an untrusted tree is canonicalized."""

    def __init__(
        self,
        reason: Literal["schema", "bytes", "cardinality", "mutation"],
        message: str,
    ) -> None:
        self.reason = reason
        super().__init__(message)


def bounded_utf8_size(
    value: object,
    *,
    role: str,
    max_bytes: int,
) -> int:
    """Count bounded UTF-8 bytes without first allocating an encoding."""

    if type(value) is not str:
        raise BoundedJsonError("schema", f"{role} must be an exact string")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise TypeError("bounded UTF-8 limit must be a positive exact integer")
    total = 0
    for character in value:
        code = ord(character)
        if 0xD800 <= code <= 0xDFFF:
            raise BoundedJsonError(
                "schema", f"{role} contains a lone surrogate",
            )
        total += (
            1 if code <= 0x7F else
            2 if code <= 0x7FF else
            3 if code <= 0xFFFF else 4
        )
        if total > max_bytes:
            raise BoundedJsonError(
                "bytes", f"{role} exceeds its UTF-8 byte ceiling",
            )
    return total


def _string_charge(
    value: str,
    *,
    role: str,
    max_string_bytes: int,
    max_encoded_bytes: int,
) -> int:
    bounded_utf8_size(value, role=role, max_bytes=max_string_bytes)
    canonical_bytes = 2
    for character in value:
        code = ord(character)
        width = (
            1 if code <= 0x7F else
            2 if code <= 0x7FF else
            3 if code <= 0xFFFF else 4
        )
        if code <= 0x1F:
            canonical_bytes += 6
        elif character in {'"', "\\"}:
            canonical_bytes += 2
        else:
            canonical_bytes += width
        if canonical_bytes > max_encoded_bytes:
            return max_encoded_bytes + 1
    return canonical_bytes


def bounded_json_snapshot(
    value: object,
    *,
    role: str,
    max_encoded_bytes: int,
    max_key_bytes: int,
    max_string_bytes: int,
    max_depth: int,
    max_nodes: int,
    max_children: int,
) -> tuple[object, int]:
    """Detach and exactly bound a JSON tree before schema work.

    Only exact built-in JSON container and scalar types are admitted.  The
    traversal charges a conservative encoded size before the single final
    canonical encoding, so an oversized caller value is never materialized as
    another oversized byte string first.
    """

    limits = (
        max_encoded_bytes,
        max_key_bytes,
        max_string_bytes,
        max_depth,
        max_nodes,
        max_children,
    )
    if any(type(limit) is not int or limit <= 0 for limit in limits):
        raise TypeError("bounded JSON limits must be positive exact integers")
    if type(role) is not str or not role:
        raise TypeError("bounded JSON role must be a nonempty exact string")

    active: set[int] = set()
    root: list[object] = [None]
    stack: list[tuple[str, object, int, object, object]] = [
        ("visit", value, 1, root, 0),
    ]
    nodes = 0
    charged = 0

    def add(amount: int) -> None:
        nonlocal charged
        charged = min(max_encoded_bytes + 1, charged + amount)
        if charged > max_encoded_bytes:
            raise BoundedJsonError(
                "bytes", f"{role} exceeds its canonical byte ceiling",
            )

    def assign(destination: object, slot: object, selected: object) -> None:
        if type(destination) is list:
            list.__setitem__(destination, slot, selected)
        else:
            dict.__setitem__(destination, slot, selected)

    while stack:
        kind, current, depth, destination, slot = stack.pop()
        if kind == "leave":
            active.remove(id(current))
            continue
        if depth > max_depth:
            raise BoundedJsonError(
                "cardinality", f"{role} exceeds its depth ceiling",
            )
        nodes += 1
        if nodes > max_nodes:
            raise BoundedJsonError(
                "cardinality", f"{role} exceeds its node ceiling",
            )
        current_type = type(current)
        if current_type in {dict, list}:
            identity = id(current)
            if identity in active:
                raise BoundedJsonError(
                    "schema", f"{role} contains an active-container cycle",
                )
            count = (
                dict.__len__(current)
                if current_type is dict else list.__len__(current)
            )
            if count > max_children:
                raise BoundedJsonError(
                    "cardinality",
                    f"{role} container exceeds its child ceiling",
                )
            if current_type is list:
                children = list.__getitem__(current, slice(0, count + 1))
                if len(children) != count or list.__len__(current) != count:
                    raise BoundedJsonError(
                        "mutation", f"{role} mutated during snapshot",
                    )
                detached: object = [None] * count
                items: tuple[tuple[object, object], ...] = tuple(
                    enumerate(children)
                )
            else:
                captured: list[tuple[object, object]] = []
                try:
                    iterator = iter(dict.items(current))
                    for _index in range(count + 1):
                        try:
                            captured.append(next(iterator))
                        except StopIteration:
                            break
                except RuntimeError as error:
                    raise BoundedJsonError(
                        "mutation", f"{role} mutated during snapshot",
                    ) from error
                if len(captured) != count or dict.__len__(current) != count:
                    raise BoundedJsonError(
                        "mutation", f"{role} mutated during snapshot",
                    )
                detached = {}
                items = tuple(captured)
                for key, _child in items:
                    if type(key) is not str:
                        raise BoundedJsonError(
                            "schema",
                            f"{role} mapping keys must be exact strings",
                        )
                    if depth + 1 > max_depth:
                        raise BoundedJsonError(
                            "cardinality",
                            f"{role} exceeds its depth ceiling",
                        )
                    nodes += 1
                    if nodes > max_nodes:
                        raise BoundedJsonError(
                            "cardinality",
                            f"{role} exceeds its node ceiling",
                        )
                    add(_string_charge(
                        key,
                        role=f"{role} key",
                        max_string_bytes=max_key_bytes,
                        max_encoded_bytes=max_encoded_bytes,
                    ))
            assign(destination, slot, detached)
            active.add(identity)
            stack.append(("leave", current, depth, destination, slot))
            add(2 + max(0, count - 1) + (count if current_type is dict else 0))
            for child_slot, child in reversed(items):
                stack.append(("visit", child, depth + 1, detached, child_slot))
            continue
        if current_type is str:
            add(_string_charge(
                current,
                role=f"{role} string",
                max_string_bytes=max_string_bytes,
                max_encoded_bytes=max_encoded_bytes,
            ))
            detached_scalar = current
        elif current is None:
            add(4)
            detached_scalar = None
        elif current_type is bool:
            add(4 if current else 5)
            detached_scalar = current
        elif current_type is int:
            if not -(2**63) <= current <= 2**63 - 1:
                raise BoundedJsonError(
                    "schema", f"{role} integer is outside signed 64-bit range",
                )
            add(20)
            detached_scalar = current
        elif current_type is float:
            if not math.isfinite(current):
                raise BoundedJsonError(
                    "schema", f"{role} float must be finite",
                )
            token = json.dumps(current, allow_nan=False)
            if len(token) > 32:
                raise BoundedJsonError(
                    "schema", f"{role} float token is unsupported",
                )
            add(len(token))
            detached_scalar = current
        else:
            raise BoundedJsonError(
                "schema",
                f"{role} contains non-JSON value {current_type.__name__}",
            )
        assign(destination, slot, detached_scalar)

    snapshot = root[0]
    raw = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8", errors="strict")
    if len(raw) > max_encoded_bytes:
        raise BoundedJsonError(
            "bytes", f"{role} exceeds its exact encoded-size ceiling",
        )
    return snapshot, len(raw)


__all__ = [
    "BoundedJsonError",
    "bounded_json_snapshot",
    "bounded_utf8_size",
]
