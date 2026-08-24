# -*- coding: utf-8 -*-
"""Lazy per-container metadata providers (R2).

A :class:`MetadataProvider` abstracts *where per-frame metadata comes from*
behind one small surface — ``metadata_for(frame_index)``, ``motors()``,
``scan_table()``, ``constants()``, ``wavelength()`` — so ``FrameSource``, the
benchmark, and the wrangler adapter all consume the SAME provider instead of
each re-deriving Bluesky columns.  It fills the R1 format adapter's reserved
``metadata_provider`` seam rather than adding a second metadata registry.

The Bluesky provider is built on the OPEN entry group of a
:class:`~xrd_tools.sources.cursor.ContainerCursor` and reads lazily: nothing is
materialized for descriptor creation, frame count, or the first detector read.
``complete_metadata_for`` can project one owner-scoped row without building the
whole table.  The legacy table-oriented surfaces still materialize the
per-frame table + scanned motors + fixed constants ONCE into plain numpy arrays;
after that the live h5py entry reference is dropped, so repeated reads touch
only memory and never reopen the source master.  Providers return ordinary
Python/numpy values, never live ``h5py`` dataset objects.

Plain NeXus / Eiger raw stacks (no Bluesky marker) get an
:class:`EmptyMetadataProvider`: their per-frame metadata is sidecar-sourced and
stays owned by the existing sidecar path, unchanged by R2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from xrd_tools.core.energy import WavelengthUnit

__all__ = [
    "MetadataProvider",
    "BlueskyMetadataProvider",
    "EmptyMetadataProvider",
    "MetadataSourceClosedError",
    "metadata_provider_for_open_entry",
]

#: Sentinel marking a provider whose owning cursor closed before its table was
#: ever materialized — the live handle is gone, so reads must fail clearly.
_SOURCE_CLOSED: Any = object()


class MetadataSourceClosedError(RuntimeError):
    """A lazy provider was read after its owning cursor closed before the
    per-frame table was materialized (handoff §4.2.5: reads after close fail
    clearly, they do not silently return empty)."""


class MetadataProvider:
    """Small provider surface consumed by sources / benchmark / wrangler."""

    def metadata_for(self, frame_index: int) -> Mapping[str, Any]:
        return {}

    def complete_metadata_for(self, frame_index: int) -> Mapping[str, Any]:
        """One complete row, including scanned motors.

        Generic providers compose their existing surfaces.  Source-specific
        providers may override this to avoid materializing whole-scan arrays.
        """
        pos = int(frame_index)
        out = dict(self.metadata_for(pos))
        for name, values in self.motors().items():
            if 0 <= pos < len(values):
                value = np.asarray(values[pos])
                out[str(name)] = value.item() if value.shape == () else values[pos]
        return out

    def frame_count(self) -> int | None:
        """The authoritative number of frames this provider serves, or ``None``
        when the provider does not know (e.g. a sidecar-sourced stack).

        Exposed so a consumer composing a per-frame row can bound the frame
        identity with the SAME authority ``metadata_for`` uses internally,
        rather than guessing the count from per-frame array lengths (which can
        legitimately differ from the frame count for baseline/aborted points)."""
        return None

    def motors(self) -> dict[str, np.ndarray]:
        return {}

    def scan_table(self) -> dict[str, np.ndarray]:
        return {}

    def constants(self) -> dict[str, float]:
        return {}

    def wavelength(self) -> float | None:
        return None

    def wavelength_unit(self) -> WavelengthUnit | None:
        """The DECLARED unit of :meth:`wavelength`, or ``None`` when unknown.

        X1 R3-P1: a wavelength value with no declared unit contributes no
        evidence — consumers must never infer a unit from value magnitude.
        Generic/direct construction stays unknown-unit unless the caller
        declares one (the cursor composition point declares ANGSTROM because
        the descriptor's ``_read_wavelength`` explicitly returns angstroms)."""
        return None

    def mark_source_closed(self) -> None:
        """Notify the provider that its owning cursor's handle has closed.

        A materialized or handle-free provider (all numpy) ignores this; a lazy
        provider that never materialized seals itself so a later read fails
        clearly instead of silently returning empty (handoff §4.2.5)."""


class EmptyMetadataProvider(MetadataProvider):
    """No per-frame acquisition metadata (plain NeXus/Eiger; sidecar-sourced).

    Carries a cheap wavelength when the descriptor already resolved one, so a
    non-Bluesky stack still reports its wavelength without any table build.
    """

    def __init__(self, *, wavelength: float | None = None,
                 wavelength_unit: WavelengthUnit | None = None) -> None:
        self._wavelength = wavelength
        self._wavelength_unit = wavelength_unit

    def wavelength(self) -> float | None:
        return self._wavelength

    def wavelength_unit(self) -> WavelengthUnit | None:
        return self._wavelength_unit


class BlueskyMetadataProvider(MetadataProvider):
    """Lazy per-frame metadata for a Bluesky/apstools NXWriter container.

    Preserves the exact semantics of
    :meth:`xrd_tools.sources.nexus.NexusStackSource.metadata_for`: scanned
    motors surface as whole-array columns (``motors``); per-frame
    counters/EPOCH/counting-time surface via ``metadata_for``; fixed motors +
    eiger counting time broadcast as constant per-frame columns (a per-frame
    column always wins over a broadcast constant).
    """

    def __init__(self, entry_grp: Any, *, frame_count: int,
                 wavelength: float | None = None,
                 wavelength_unit: WavelengthUnit | None = None,
                 scanned_motor_names: tuple[str, ...] | None = None,
                 all_motor_names: tuple[str, ...] | None = None) -> None:
        _validate_prequalified_motor_pair(scanned_motor_names, all_motor_names)
        self._entry = entry_grp          # live h5py group; dropped after build
        self._frame_count = int(frame_count)
        self._wavelength = wavelength
        self._wavelength_unit = wavelength_unit
        self._table: dict[str, np.ndarray] | None = None
        self._motors: dict[str, np.ndarray] | None = None
        self._constants: dict[str, float] | None = None
        self._scanned_motor_names = scanned_motor_names
        self._all_motor_names = all_motor_names
        self._counter_names = None
        if scanned_motor_names is not None:
            from xrd_tools.io.bluesky_nexus import _average_default_counter_names
            self._counter_names = _average_default_counter_names(entry_grp)

    def frame_count(self) -> int | None:
        return self._frame_count

    def complete_metadata_for(self, frame_index: int) -> Mapping[str, Any]:
        """Read one complete Bluesky row without building whole-scan arrays."""
        pos = int(frame_index)
        if not 0 <= pos < self._frame_count:
            return {}
        if self._table is not None:
            return super().complete_metadata_for(pos)

        entry = self._entry
        if entry is _SOURCE_CLOSED:
            raise MetadataSourceClosedError(
                "metadata provider read after its cursor closed before the "
                "per-frame table was materialized; open a fresh cursor"
            )
        if entry is None:
            return {}

        from xrd_tools.io.bluesky_nexus import (
            _BLUESKY_COUNT_TIME_COL,
            _DEFAULT_BLUESKY_COUNTERS,
            bluesky_constant_metadata,
            bluesky_motor_names,
        )

        data = entry.get("data")
        scanned = (tuple(bluesky_motor_names(entry))
                   if self._scanned_motor_names is None
                   else self._scanned_motor_names)
        counters = (_DEFAULT_BLUESKY_COUNTERS if self._counter_names is None
                    else self._counter_names)
        names = tuple(dict.fromkeys(
            (*scanned, *counters,
             _BLUESKY_COUNT_TIME_COL, "EPOCH")
        ))
        per_frame_names: set[str] = set()
        out: dict[str, Any] = {}
        if data is not None:
            for name in names:
                dataset = data.get(name)
                if (
                    dataset is None
                    or getattr(dataset, "ndim", None) != 1
                    or getattr(getattr(dataset, "dtype", None), "kind", "O") not in "fiub"
                ):
                    continue
                per_frame_names.add(str(name))
                if pos >= int(dataset.shape[0]):
                    continue
                try:
                    out[str(name)] = float(dataset[pos])
                except (TypeError, ValueError, OSError):
                    continue
        constants = bluesky_constant_metadata(
            entry, exclude=per_frame_names,
            motor_names=self._all_motor_names,
            bounded=self._all_motor_names is not None,
        )
        for name, value in constants.items():
            out.setdefault(str(name), float(value))
        return out

    def mark_source_closed(self) -> None:
        # Seal only when never materialized; a materialized provider is pure
        # numpy and stays fully usable after its cursor closes.
        if self._table is None:
            self._entry = _SOURCE_CLOSED

    def _ensure_table(self) -> None:
        if self._table is not None:
            return
        from xrd_tools.io.bluesky_nexus import (
            bluesky_angles,
            bluesky_constant_metadata,
            bluesky_per_frame_table,
        )

        entry = self._entry
        if entry is _SOURCE_CLOSED:
            raise MetadataSourceClosedError(
                "metadata provider read after its cursor closed before the "
                "per-frame table was materialized; open a fresh cursor")
        if entry is None:
            # Materialized already but table missing (defensive): empty.
            self._table, self._motors, self._constants = {}, {}, {}
            return
        table = {k: np.asarray(v) for k, v in bluesky_per_frame_table(entry).items()}
        motors = {k: np.asarray(v) for k, v in bluesky_angles(entry).items()}
        constants = {
            str(k): float(v)
            for k, v in bluesky_constant_metadata(entry, exclude=table.keys()).items()
        }
        self._table, self._motors, self._constants = table, motors, constants
        # Drop the live handle reference: all further reads are pure-memory.
        self._entry = None

    def motors(self) -> dict[str, np.ndarray]:
        self._ensure_table()
        return dict(self._motors or {})

    def scan_table(self) -> dict[str, np.ndarray]:
        self._ensure_table()
        return dict(self._table or {})

    def constants(self) -> dict[str, float]:
        self._ensure_table()
        return dict(self._constants or {})

    def wavelength(self) -> float | None:
        return self._wavelength

    def wavelength_unit(self) -> WavelengthUnit | None:
        return self._wavelength_unit

    def metadata_for(self, frame_index: int) -> Mapping[str, Any]:
        self._ensure_table()
        table = self._table or {}
        motors = self._motors or {}
        constants = self._constants or {}
        pos = int(frame_index)
        # Bounds-checked, never negative-wrapped (mirrors NexusStackSource F7a).
        if not 0 <= pos < self._frame_count:
            return {}
        out: dict[str, Any] = {}
        for name, arr in table.items():
            if name in motors:
                continue  # scanned motors surface as whole-array columns
            if 0 <= pos < len(arr):
                try:
                    out[name] = float(arr[pos])
                except (TypeError, ValueError):
                    pass
        for name, val in constants.items():
            out.setdefault(name, float(val))
        return out


def _validate_prequalified_motor_pair(scanned, all_names) -> None:
    pair = (scanned, all_names)
    if any(value is not None for value in pair) and any(
        type(value) is not tuple or any(type(name) is not str for name in value)
        for value in pair
    ):
        raise TypeError("prevalidated motor names must be paired exact tuples")


def metadata_provider_for_open_entry(
    entry_grp: Any,
    *,
    frame_count: int,
    wavelength: float | None = None,
    wavelength_unit: WavelengthUnit | None = None,
    is_bluesky: bool | None = None,
    scanned_motor_names: tuple[str, ...] | None = None,
    all_motor_names: tuple[str, ...] | None = None,
) -> MetadataProvider:
    """Build the right provider for an OPEN entry group.

    A Bluesky/NXWriter entry gets a lazy :class:`BlueskyMetadataProvider`; every
    other raw stack gets an :class:`EmptyMetadataProvider` (sidecar behavior
    preserved).  ``is_bluesky`` may be supplied (e.g. from the descriptor) to
    avoid re-detecting; otherwise it is probed once from the open group.
    ``wavelength_unit`` is the caller's explicit unit declaration for
    ``wavelength`` (R3-P1) — retained as value-only provider state.
    """
    _validate_prequalified_motor_pair(scanned_motor_names, all_motor_names)
    if entry_grp is None:
        return EmptyMetadataProvider(
            wavelength=wavelength, wavelength_unit=wavelength_unit)
    if is_bluesky is None:
        try:
            from xrd_tools.io.bluesky_nexus import is_bluesky_nxwriter
            is_bluesky = bool(is_bluesky_nxwriter(entry_grp))
        except Exception:
            is_bluesky = False
    if is_bluesky:
        return BlueskyMetadataProvider(
            entry_grp, frame_count=frame_count, wavelength=wavelength,
            wavelength_unit=wavelength_unit,
            scanned_motor_names=scanned_motor_names,
            all_motor_names=all_motor_names)
    return EmptyMetadataProvider(
        wavelength=wavelength, wavelength_unit=wavelength_unit)
