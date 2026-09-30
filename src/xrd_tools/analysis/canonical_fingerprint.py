"""Engine-light canonical identity framing for analysis values.

This module deliberately avoids importing NumPy.  NumPy-backed values are
supported when NumPy is already present because constructing an ``ndarray``
necessarily imports it; scalar and container fingerprints stay stdlib-only.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import sys
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any


_OUTPUT_QNAN_BITS = 0x7FF8000000000000
_CANONICAL_PREFIX = b"xrd_tools.analysis.canonical.v1\x00"


class InvalidCanonicalValue(ValueError):
    pass


def _numpy_array_type():
    numpy = sys.modules.get("numpy")
    return None if numpy is None else getattr(numpy, "ndarray", None)


def _is_numpy_array(value: Any) -> bool:
    array_type = _numpy_array_type()
    return array_type is not None and type(value) is array_type


def _canonical_array(value: Any, *, allow_missing: bool):
    numpy = sys.modules.get("numpy")
    array_type = None if numpy is None else getattr(numpy, "ndarray", None)
    if (
        array_type is None
        or type(value) is not array_type
        or value.dtype.kind not in "biufc"
        or value.dtype.fields is not None
    ):
        raise InvalidCanonicalValue("unsupported canonical ndarray")
    array = numpy.ascontiguousarray(value)
    if array.dtype.kind in "fc" and not numpy.isfinite(array).all():
        valid_missing = False
        if allow_missing and array.dtype.str == "<f8":
            invalid = ~numpy.isfinite(array)
            valid_missing = bool(
                numpy.all(array.view("<u8")[invalid] == _OUTPUT_QNAN_BITS)
            )
        if not valid_missing:
            raise InvalidCanonicalValue("nonfinite canonical ndarray")
    return array


_pack_size = struct.Struct(">Q").pack
_pack_float = struct.Struct(">d").pack
_NONE_FRAME = b"\x00" + _pack_size(0)
_FALSE_FRAME = b"\x01" + _pack_size(1) + b"\x00"
_TRUE_FRAME = b"\x01" + _pack_size(1) + b"\x01"
_FLOAT_HEADER = b"\x03" + _pack_size(8)


def _int_payload(value: int) -> bytes:
    magnitude = abs(value)
    raw = b"" if magnitude == 0 else magnitude.to_bytes(
        (magnitude.bit_length() + 7) // 8, "big"
    )
    return (b"\x01" if value < 0 else b"\x00") + raw


class _Encoding:
    """One canonical encoding of a value, in two passes.

    Frames are length-prefixed, so a container's size is needed before its
    items. ``measure`` validates the value and records each container's size
    (with a mapping's sorted items, an enum's identity or an array's
    normalized copy) once; ``emit`` then streams the frames from those
    records. Every value is visited twice whatever its depth: measuring
    again inside every emitting container re-measured each subtree once per
    level of nesting, which made a 287-frame scan's input check take seconds.
    """

    __slots__ = ("allow_missing", "array_type", "records")

    def __init__(self, allow_missing: bool) -> None:
        self.allow_missing = allow_missing
        self.array_type = _numpy_array_type()
        self.records: dict[int, Any] = {}

    def measure(self, value: Any, active: set[int]) -> int:
        kind = type(value)
        if kind is float:
            if not math.isfinite(value):
                raise InvalidCanonicalValue("nonfinite canonical float")
            return 17
        if kind is str:
            return 9 + (len(value) if value.isascii() else len(value.encode("utf-8")))
        if kind is int:
            return 10 + (abs(value).bit_length() + 7) // 8
        if value is None:
            return 9
        if kind is bool:
            return 10
        if kind is tuple or kind is list:
            marker = self._enter(value, active)
            if marker is None:  # measured before: a shared, not a cyclic, reference
                return 9 + self.records[id(value)]
            try:
                size = 8
                for item in value:
                    size += self.measure(item, active)
            finally:
                active.discard(marker)
            self.records[marker] = size
            return 9 + size
        if kind is bytes:
            return 9 + len(value)
        if isinstance(value, Enum):
            marker = self._enter(value, active)
            if marker is None:
                return 9 + self.records[id(value)][0]
            try:
                identity = f"{type(value).__module__}.{type(value).__qualname__}"
                size = self.measure(identity, active) + self.measure(value.value, active)
            finally:
                active.discard(marker)
            self.records[marker] = (size, identity)
            return 9 + size
        if isinstance(value, Path):
            return 9 + len(os.fspath(value).encode("utf-8"))
        if isinstance(value, Mapping):
            marker = self._enter(value, active)
            if marker is None:
                return 9 + self.records[id(value)][0]
            try:
                if any(type(key) is not str for key in value):
                    raise InvalidCanonicalValue(
                        "canonical mapping keys must be exact str"
                    )
                items = sorted(value.items(), key=lambda item: item[0].encode("utf-8"))
                size = 8
                for key, item in items:
                    size += self.measure(key, active) + self.measure(item, active)
            finally:
                active.discard(marker)
            self.records[marker] = (size, items)
            return 9 + size
        if self.array_type is not None and kind is self.array_type:
            marker = self._enter(value, active)
            if marker is None:
                return 9 + self.records[id(value)][0]
            try:
                array = _canonical_array(value, allow_missing=self.allow_missing)
                size = (self.measure(array.dtype.str, active) + 8 + 8 * array.ndim
                        + array.nbytes)
            finally:
                active.discard(marker)
            self.records[marker] = (size, array)
            return 9 + size
        raise InvalidCanonicalValue(f"unsupported canonical value {type(value)!r}")

    def _enter(self, value: Any, active: set[int]) -> int | None:
        """Track a container on the current path: its marker, or None when it was measured before."""
        marker = id(value)
        if marker in active:
            raise InvalidCanonicalValue("cyclic canonical value")
        if marker in self.records:
            return None
        active.add(marker)
        return marker

    def emit(self, value: Any, write) -> None:
        """Stream the frames of a value ``measure`` accepted."""
        kind = type(value)
        if kind is float:
            write(_FLOAT_HEADER + _pack_float(value))
        elif kind is str:
            raw = value.encode("utf-8")
            write(b"\x04" + _pack_size(len(raw)) + raw)
        elif kind is int:
            payload = _int_payload(value)
            write(b"\x02" + _pack_size(len(payload)) + payload)
        elif value is None:
            write(_NONE_FRAME)
        elif kind is bool:
            write(_TRUE_FRAME if value else _FALSE_FRAME)
        elif kind is tuple or kind is list:
            write(b"\x08" + _pack_size(self.records[id(value)]) + _pack_size(len(value)))
            for item in value:
                self.emit(item, write)
        elif kind is bytes:
            write(b"\x05" + _pack_size(len(value)))
            write(value)
        elif isinstance(value, Enum):
            size, identity = self.records[id(value)]
            write(b"\x07" + _pack_size(size))
            self.emit(identity, write)
            self.emit(value.value, write)
        elif isinstance(value, Path):
            raw = os.fspath(value).encode("utf-8")
            write(b"\x06" + _pack_size(len(raw)) + raw)
        elif isinstance(value, Mapping):
            size, items = self.records[id(value)]
            write(b"\x09" + _pack_size(size) + _pack_size(len(items)))
            for key, item in items:
                self.emit(key, write)
                self.emit(item, write)
        else:  # an ndarray
            size, array = self.records[id(value)]
            write(b"\x0a" + _pack_size(size))
            self.emit(array.dtype.str, write)
            write(_pack_size(array.ndim) + b"".join(_pack_size(dimension) for dimension in array.shape))
            write(memoryview(array).cast("B"))


def _canonical_frame(
    value: Any,
    active: set[int],
    emit=None,
    *,
    allow_missing: bool = False,
) -> int:
    """The size of ``value``'s canonical frame, validating it; with ``emit``, also stream its bytes."""
    encoding = _Encoding(allow_missing)
    size = encoding.measure(value, active)
    if emit is not None:
        encoding.emit(value, emit)
    return size


def _canonical_stream(value: Any, emit, *, allow_missing: bool = False) -> int:
    emit(_CANONICAL_PREFIX)
    return len(_CANONICAL_PREFIX) + _canonical_frame(
        value, set(), emit, allow_missing=allow_missing
    )


def _canonical_bytes(value: Any, *, allow_missing: bool = False) -> bytes:
    output = bytearray()
    _canonical_stream(value, output.extend, allow_missing=allow_missing)
    return bytes(output)


def _canonical_frame_charge(value: Any, *, allow_missing: bool = False) -> int:
    return _canonical_frame(value, set(), allow_missing=allow_missing)


def _canonical_charge(value: Any, *, allow_missing: bool = False) -> int:
    return len(_CANONICAL_PREFIX) + _canonical_frame_charge(
        value, allow_missing=allow_missing
    )


def _digest(value: Any, *, allow_missing: bool = False) -> str:
    digest = hashlib.sha256()
    _canonical_stream(value, digest.update, allow_missing=allow_missing)
    return digest.hexdigest()


class _PublicFingerprintContainer(Enum):
    LIST = "list"
    TUPLE = "tuple"
    MAPPING = "mapping"


# The fully-qualified Enum type name is part of every public fingerprint.
# Preserve the accepted identity even though the implementation is extracted.
_PublicFingerprintContainer.__module__ = "xrd_tools.analysis.scan_operations"


def _public_fingerprint_projection(value: Any, active: set[int]) -> Any:
    recursive = type(value) in {list, tuple} or isinstance(value, Mapping)
    marker = id(value)
    if recursive:
        if marker in active:
            raise InvalidCanonicalValue("cyclic canonical value")
        active.add(marker)
    try:
        if type(value) is list:
            return (
                _PublicFingerprintContainer.LIST,
                tuple(
                    _public_fingerprint_projection(item, active)
                    for item in value
                ),
            )
        if type(value) is tuple:
            return (
                _PublicFingerprintContainer.TUPLE,
                tuple(
                    _public_fingerprint_projection(item, active)
                    for item in value
                ),
            )
        if isinstance(value, Mapping):
            return (
                _PublicFingerprintContainer.MAPPING,
                {
                    key: _public_fingerprint_projection(item, active)
                    for key, item in value.items()
                },
            )
        return value
    finally:
        if recursive:
            active.remove(marker)


def analysis_canonical_fingerprint(
    domain: str,
    value: Any,
    *,
    allow_missing: bool = False,
) -> str:
    """Hash one type-framed value in an explicit public identity domain."""

    if type(domain) is not str or not domain or domain.strip() != domain:
        raise InvalidCanonicalValue("fingerprint domain must be an exact token")
    if type(allow_missing) is not bool:
        raise InvalidCanonicalValue("allow_missing must be an exact bool")
    return _digest(
        (
            "analysis-public-fingerprint-v1",
            domain,
            _public_fingerprint_projection(value, set()),
        ),
        allow_missing=allow_missing,
    )


# Preserve legacy introspection and pickle identities for the extracted API.
_LEGACY_MODULE = "xrd_tools.analysis.scan_operations"
for _legacy_value in (
    InvalidCanonicalValue,
    _canonical_array,
    _canonical_frame,
    _canonical_stream,
    _canonical_bytes,
    _canonical_frame_charge,
    _canonical_charge,
    _digest,
    _PublicFingerprintContainer,
    _public_fingerprint_projection,
    analysis_canonical_fingerprint,
):
    _legacy_value.__module__ = _LEGACY_MODULE
del _legacy_value


__all__ = ["InvalidCanonicalValue", "analysis_canonical_fingerprint"]
