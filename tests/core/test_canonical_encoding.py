"""The canonical encoder's bytes, sizes, digests and refusals are exactly those of the encoder it
replaced (frozen below verbatim from ``canonical_fingerprint.py`` at ``540b2bef``), which re-measured
every subtree once per level of nesting: stored fingerprints must not change, only the time to make them
(maintainer, 2026-09-30: "the startup, esp with the 287 frame scan is rather slow")."""

import hashlib
import math
import os
import struct
from collections import namedtuple
from collections.abc import Mapping
from enum import Enum, IntEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from xrd_tools.analysis import canonical_fingerprint as cf
from xrd_tools.analysis.canonical_fingerprint import (
    InvalidCanonicalValue, _canonical_array, _is_numpy_array,
)


def _frozen_frame(
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
            size = _frozen_frame(identity, active, allow_missing=allow_missing)
            size += _frozen_frame(value.value, active, allow_missing=allow_missing)
            if emit is not None:
                emit(b"\x07" + struct.pack(">Q", size))
                _frozen_frame(identity, active, emit, allow_missing=allow_missing)
                _frozen_frame(value.value, active, emit, allow_missing=allow_missing)
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
                _frozen_frame(item, active, allow_missing=allow_missing)
                for item in items
            )
            if emit is not None:
                emit(b"\x08" + struct.pack(">Q", size))
                emit(struct.pack(">Q", len(items)))
                for item in items:
                    _frozen_frame(item, active, emit, allow_missing=allow_missing)
            return 9 + size
        elif isinstance(value, Mapping):
            if any(type(key) is not str for key in value):
                raise InvalidCanonicalValue(
                    "canonical mapping keys must be exact str"
                )
            items = sorted(value.items(), key=lambda item: item[0].encode("utf-8"))
            size = 8 + sum(
                _frozen_frame(key, active, allow_missing=allow_missing)
                + _frozen_frame(item, active, allow_missing=allow_missing)
                for key, item in items
            )
            if emit is not None:
                emit(b"\x09" + struct.pack(">Q", size))
                emit(struct.pack(">Q", len(items)))
                for key, item in items:
                    _frozen_frame(key, active, emit, allow_missing=allow_missing)
                    _frozen_frame(item, active, emit, allow_missing=allow_missing)
            return 9 + size
        elif _is_numpy_array(value):
            array = _canonical_array(value, allow_missing=allow_missing)
            dtype_size = _frozen_frame(
                array.dtype.str, active, allow_missing=allow_missing
            )
            size = dtype_size + 8 + 8 * array.ndim + array.nbytes
            if emit is not None:
                emit(b"\x0a" + struct.pack(">Q", size))
                _frozen_frame(
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


def _frozen_bytes(value, allow_missing=False):
    output = bytearray()
    output.extend(cf._CANONICAL_PREFIX)
    _frozen_frame(value, set(), output.extend, allow_missing=allow_missing)
    return bytes(output)


class _Mode(Enum):
    ONE = "one"
    PAIR = ("a", 2)


class _Level(IntEnum):
    LOW = 1
    HIGH = 2


def _corpus(seed):
    rng = np.random.default_rng(seed)
    scalars = [None, True, False, 0, 1, -1, 255, 256, -256, 2 ** 70, -(2 ** 70), 0.0, -0.0, 1.5, 5e-324,
               1.7976931348623157e308, "", "x", "ünïcödé", "𝛼β", b"", b"\x00\xff", Path("a/b"),
               _Mode.ONE, _Mode.PAIR, _Level.HIGH]
    arrays = [np.array([[1, 2]], dtype="<i2"), np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(2, 3).T,
              np.arange(10, dtype=">i4")[::3], np.array(3.5), np.array([True, False]),
              np.array([1 + 2j]), np.arange(4, dtype="<u8"), np.asfortranarray(np.ones((2, 2), dtype="<f4"))]

    def build(depth):
        choice = rng.integers(0, 6 if depth < 5 else 2)
        if choice == 0:
            return scalars[rng.integers(len(scalars))]
        if choice == 1:
            return arrays[rng.integers(len(arrays))]
        items = [build(depth + 1) for _ in range(rng.integers(0, 5))]
        if choice == 2:
            return items
        if choice == 3:
            return tuple(items)
        mapping = {"".join(rng.choice(list("abcÅé𝛼z"), size=rng.integers(0, 4))): item for item in items}
        return MappingProxyType(mapping) if choice == 5 else mapping

    shared = [1, (2.5, "s"), {"k": np.arange(3)}]
    return [*scalars, *arrays, *(build(0) for _ in range(300)),
            {"a": shared, "b": shared, "c": [shared, (shared,)]}, [[[[[[[[1]]]]]]]]]


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_the_encoder_writes_the_bytes_sizes_and_digests_it_replaced(seed):
    for value in _corpus(seed):
        expected = _frozen_bytes(value)
        assert cf._canonical_bytes(value) == expected, value
        assert cf._canonical_charge(value) == len(expected)
        assert cf._digest(value) == hashlib.sha256(expected).hexdigest()
        assert cf._canonical_frame(value, set()) == len(expected) - len(cf._CANONICAL_PREFIX)
    missing = np.array([0x7FF8000000000000, 0x3FF0000000000000], dtype="<u8").view("<f8")
    assert cf._canonical_bytes((missing, "x"), allow_missing=True) == _frozen_bytes((missing, "x"), True)
    value = {"name": ("run", 3), "public": [1.0, None, {"z": b"x"}]}
    assert cf.analysis_canonical_fingerprint("domain", value) == hashlib.sha256(_frozen_bytes((
        "analysis-public-fingerprint-v1", "domain",
        cf._public_fingerprint_projection(value, set())))).hexdigest()


def _outcome(encode, value, **options):
    try:
        return "bytes", encode(value, **options)
    except Exception as error:  # noqa: BLE001 - the error itself is compared
        return type(error), str(error)


def test_the_encoder_refuses_what_it_refused_with_the_same_errors():
    cyclic = []
    cyclic.append(cyclic)
    looped = {}
    looped["self"] = looped
    Pair = namedtuple("Pair", "a b")
    for value in (object(), {"bad": object()}, {1: "bad"}, {1, 2}, iter((1, 2)), cyclic, [1, looped],
                  np.array([object()], dtype=object), np.array(["x"]), float("nan"), float("-inf"),
                  np.array([np.nan]), np.array([np.inf]), np.array([1.0 + complex(0.0, np.inf)]),
                  np.float64(1.0), np.int64(3), Pair(1, 2), [1, 2.0, float("inf")],
                  np.array([0x7FF8000000000001], dtype="<u8").view("<f8")):
        expected = _outcome(_frozen_bytes, value)
        assert expected[0] is InvalidCanonicalValue, value
        for allow_missing in (False, True):
            expected = _outcome(_frozen_bytes, value, allow_missing=allow_missing)
            assert _outcome(cf._canonical_bytes, value, allow_missing=allow_missing) == expected, value
            charge = _outcome(cf._canonical_charge, value, allow_missing=allow_missing)
            assert charge == (expected if expected[0] != "bytes" else ("bytes", len(expected[1])))
    empty = np.zeros((0, 3))  # neither encodes an empty array (memoryview refuses the cast); both size it
    assert _outcome(cf._canonical_bytes, empty) == _outcome(_frozen_bytes, empty)
    assert cf._canonical_charge(empty) == len(cf._CANONICAL_PREFIX) + _frozen_frame(empty, set())


def test_each_value_is_measured_once_and_emitted_once_whatever_its_depth(monkeypatch):
    calls = {"measure": 0, "emit": 0}
    for name in calls:
        original = getattr(cf._Encoding, name)

        def counted(self, *args, _original=original, _name=name):
            calls[_name] += 1
            return _original(self, *args)
        monkeypatch.setattr(cf._Encoding, name, counted)
    value = 1
    for _ in range(40):
        value = (value,)
    assert cf._digest(value) == hashlib.sha256(_frozen_bytes(value)).hexdigest()
    assert calls == {"measure": 41, "emit": 41}  # the old encoder: 820 measures, 41 emits
