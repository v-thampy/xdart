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


def _canonical_frame(
    value: Any,
    active: set[int],
    emit=None,
    *,
    allow_missing: bool = False,
) -> int:
    recursive = (
        isinstance(value, (tuple, list, Mapping, Enum))
        or _is_numpy_array(value)
    )
    marker = id(value)
    if recursive:
        if marker in active:
            raise InvalidCanonicalValue("cyclic canonical value")
        active.add(marker)
    try:
        if value is None:
            tag, payload = 0x00, b""
        elif type(value) is bool:
            tag, payload = 0x01, b"\x01" if value else b"\x00"
        elif isinstance(value, Enum):
            identity = f"{type(value).__module__}.{type(value).__qualname__}"
            size = _canonical_frame(identity, active, allow_missing=allow_missing)
            size += _canonical_frame(value.value, active, allow_missing=allow_missing)
            if emit is not None:
                emit(b"\x07" + struct.pack(">Q", size))
                _canonical_frame(identity, active, emit, allow_missing=allow_missing)
                _canonical_frame(value.value, active, emit, allow_missing=allow_missing)
            return 9 + size
        elif type(value) is int:
            magnitude = abs(value)
            raw = b"" if magnitude == 0 else magnitude.to_bytes(
                (magnitude.bit_length() + 7) // 8, "big"
            )
            tag, payload = 0x02, (b"\x01" if value < 0 else b"\x00") + raw
        elif type(value) is float:
            if not math.isfinite(value):
                raise InvalidCanonicalValue("nonfinite canonical float")
            tag, payload = 0x03, struct.pack(">d", value)
        elif type(value) is str:
            tag, payload = 0x04, value.encode("utf-8")
        elif type(value) is bytes:
            tag, payload = 0x05, value
        elif isinstance(value, Path):
            tag, payload = 0x06, os.fspath(value).encode("utf-8")
        elif type(value) in {list, tuple}:
            items = tuple(value) if type(value) is list else value
            size = 8 + sum(
                _canonical_frame(item, active, allow_missing=allow_missing)
                for item in items
            )
            if emit is not None:
                emit(b"\x08" + struct.pack(">Q", size))
                emit(struct.pack(">Q", len(items)))
                for item in items:
                    _canonical_frame(item, active, emit, allow_missing=allow_missing)
            return 9 + size
        elif isinstance(value, Mapping):
            if any(type(key) is not str for key in value):
                raise InvalidCanonicalValue(
                    "canonical mapping keys must be exact str"
                )
            items = sorted(value.items(), key=lambda item: item[0].encode("utf-8"))
            size = 8 + sum(
                _canonical_frame(key, active, allow_missing=allow_missing)
                + _canonical_frame(item, active, allow_missing=allow_missing)
                for key, item in items
            )
            if emit is not None:
                emit(b"\x09" + struct.pack(">Q", size))
                emit(struct.pack(">Q", len(items)))
                for key, item in items:
                    _canonical_frame(key, active, emit, allow_missing=allow_missing)
                    _canonical_frame(item, active, emit, allow_missing=allow_missing)
            return 9 + size
        elif _is_numpy_array(value):
            array = _canonical_array(value, allow_missing=allow_missing)
            dtype_size = _canonical_frame(
                array.dtype.str, active, allow_missing=allow_missing
            )
            size = dtype_size + 8 + 8 * array.ndim + array.nbytes
            if emit is not None:
                emit(b"\x0a" + struct.pack(">Q", size))
                _canonical_frame(
                    array.dtype.str, active, emit, allow_missing=allow_missing
                )
                emit(struct.pack(">Q", array.ndim))
                for dimension in array.shape:
                    emit(struct.pack(">Q", dimension))
                emit(memoryview(array).cast("B"))
            return 9 + size
        else:
            raise InvalidCanonicalValue(
                f"unsupported canonical value {type(value)!r}"
            )
        if emit is not None:
            emit(bytes((tag,)) + struct.pack(">Q", len(payload)))
            emit(payload)
        return 9 + len(payload)
    finally:
        if recursive:
            active.discard(marker)


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
