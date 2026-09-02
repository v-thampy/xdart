"""Platform-gated allocator pressure relief with no scientific ownership.

The RSM operation uses this small boundary only after its ordinary weak-root
release proofs have passed.  The bound capability accepts no array or owner;
it merely asks Darwin's public allocator to return currently reusable pages.
Importing this module is safe on every platform.  Binding is explicit so an
unsupported platform or unavailable symbol is a bounded refusal rather than a
silent no-op.
"""

from __future__ import annotations

import ctypes
import sys


class AllocatorPressureUnavailable(RuntimeError):
    """The required platform allocator-pressure capability is unavailable."""


class AllocatorPressureCallFailed(RuntimeError):
    """The bound allocator-pressure capability could not be invoked."""


_CAPABILITY_FACTORY = object()


class _DarwinAllocatorPressureRelief:
    """One process-local binding of ``malloc_zone_pressure_relief``.

    The library is retained solely to keep the foreign function binding valid.
    No product value, ndarray, grid, source, or result is accepted or retained.
    """

    __slots__ = ("_library", "_function")

    def __init__(
        self,
        library: object,
        function: object,
        claim: object,
    ) -> None:
        if claim is not _CAPABILITY_FACTORY:
            raise TypeError("allocator-pressure capability is factory-owned")
        self._library = library
        self._function = function

    def relieve(self) -> None:
        """Request maximal relief from all Darwin malloc zones.

        Darwin documents a null zone as all zones and a zero goal as maximal
        pressure relief.  The return value is deliberately ignored: the public
        API returned zero in the authenticated diagnostic even while RSS fell,
        so it is neither a success nor an acceptance oracle.
        """

        try:
            self._function(None, 0)  # type: ignore[operator]
        except Exception as error:
            raise AllocatorPressureCallFailed(
                "Darwin allocator pressure relief call failed"
            ) from error


def bind_darwin_allocator_pressure_relief() -> _DarwinAllocatorPressureRelief:
    """Bind Darwin's public allocator-pressure API or refuse explicitly."""

    if sys.platform != "darwin":
        raise AllocatorPressureUnavailable(
            "Darwin allocator pressure relief is unavailable on this platform"
        )
    try:
        library = ctypes.CDLL(None)
        function = library.malloc_zone_pressure_relief
        function.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        function.restype = ctypes.c_size_t
    except Exception as error:
        raise AllocatorPressureUnavailable(
            "Darwin malloc_zone_pressure_relief is unavailable"
        ) from error
    return _DarwinAllocatorPressureRelief(
        library,
        function,
        _CAPABILITY_FACTORY,
    )
