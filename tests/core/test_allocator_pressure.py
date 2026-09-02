from __future__ import annotations

import ctypes
import inspect

import pytest

import xrd_tools.core.allocator_pressure as allocator_pressure
from xrd_tools.core.allocator_pressure import (
    AllocatorPressureCallFailed,
    AllocatorPressureUnavailable,
    bind_darwin_allocator_pressure_relief,
)


class _Function:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.argtypes = None
        self.restype = None
        self.calls: list[tuple[object, object]] = []
        self.failure = failure

    def __call__(self, zone: object, goal: object) -> int:
        self.calls.append((zone, goal))
        if self.failure is not None:
            raise self.failure
        return 0


class _Library:
    def __init__(self, function: _Function) -> None:
        self.malloc_zone_pressure_relief = function


def test_non_darwin_platform_refuses_without_loading_library(monkeypatch) -> None:
    monkeypatch.setattr(allocator_pressure.sys, "platform", "linux")
    monkeypatch.setattr(
        allocator_pressure.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-Darwin binding must not load a library")
        ),
    )

    with pytest.raises(AllocatorPressureUnavailable):
        bind_darwin_allocator_pressure_relief()


def test_missing_darwin_symbol_is_a_bounded_unavailable_refusal(monkeypatch) -> None:
    monkeypatch.setattr(allocator_pressure.sys, "platform", "darwin")
    monkeypatch.setattr(
        allocator_pressure.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(AllocatorPressureUnavailable):
        bind_darwin_allocator_pressure_relief()


def test_bound_darwin_call_has_no_product_or_array_input(monkeypatch) -> None:
    function = _Function()
    library = _Library(function)
    monkeypatch.setattr(allocator_pressure.sys, "platform", "darwin")
    monkeypatch.setattr(
        allocator_pressure.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: library,
    )

    bound = bind_darwin_allocator_pressure_relief()
    assert not hasattr(bound, "__dict__")
    assert tuple(inspect.signature(bound.relieve).parameters) == ()
    assert function.argtypes == (ctypes.c_void_p, ctypes.c_size_t)
    assert function.restype is ctypes.c_size_t

    bound.relieve()

    assert function.calls == [(None, 0)]


def test_darwin_call_failure_is_normalized_and_return_zero_is_not_failure(
    monkeypatch,
) -> None:
    successful = _Function()
    monkeypatch.setattr(allocator_pressure.sys, "platform", "darwin")
    monkeypatch.setattr(
        allocator_pressure.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: _Library(successful),
    )
    bind_darwin_allocator_pressure_relief().relieve()
    assert successful.calls == [(None, 0)]

    failed = _Function(failure=OSError("synthetic allocator call failure"))
    monkeypatch.setattr(
        allocator_pressure.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: _Library(failed),
    )
    with pytest.raises(AllocatorPressureCallFailed):
        bind_darwin_allocator_pressure_relief().relieve()
    assert failed.calls == [(None, 0)]
