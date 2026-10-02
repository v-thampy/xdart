"""A fit's detached input: what was admitted, which part of it, its linear values, and how it is fitted.

FIT-TOOLS-2026-10 (Codex's design review F4: "``AnalysisInput`` is a mutable numerical input, not a
current-result/staleness receipt"). A fit host (Stitch, RSM, a notebook) captures a ``FitInputSnapshot`` when a fit
is asked for and keeps its ``key``; the result is adopted only while the host's current ``FitInputKey`` is equal.

The key is made of facts the host already holds, so nothing is hashed again: the admitted result's identity, path
and admission fingerprint (a result replaced at the same path is re-admitted with another), the host's generation
(a reload or a reopen bumps it), the part selected (panel, cut, slice or box parameters), whether the presentation
background (Set BG) is subtracted before the fit, and the fit settings. Presentation is not in it: the values are the
linear scientific ones whatever Log or the colour map shows. The snapshot holds its own read-only copies of the
values, their validity mask and their coordinates (one ``FitAxis`` per dimension, in the values' order, with its
unit), and the result's recorded wavelength (None: not recorded; a fit that needs one refuses by name).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Whether the presentation background is subtracted before the fit (Set BG applies: "presentation") or not.
BACKGROUND_POLICIES = ("none", "presentation")
_SCHEMA = "fit-input-snapshot-v1"


def _frozen(value: object) -> bool:
    """A plain, hashable, comparable setting: a str, a number, a bool, None, or a tuple of them."""
    if value is None or type(value) in {str, int, float, bool}:
        return type(value) is not float or not math.isnan(value)
    return type(value) is tuple and all(_frozen(item) for item in value)


def _pairs(value: object, what: str) -> tuple[tuple[str, object], ...]:
    if (type(value) is not tuple or any(type(item) is not tuple or len(item) != 2 or type(item[0]) is not str
                                        or not _frozen(item[1]) for item in value)
            or len({name for name, _setting in value}) != len(value)):
        raise TypeError(f"a fit input's {what} are (name, value) pairs of plain values, each name once")
    return value


@dataclass(frozen=True, slots=True)
class FitInputKey:
    """What a fit's input is, without its values (the module docstring). Equal keys: the same input; a fit whose key
    is not the host's current one is stale."""

    result_identity: str
    path: str
    fingerprint: str
    generation: int
    selection: tuple[tuple[str, object], ...] = ()
    background: str = "none"
    settings: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if any(type(value) is not str for value in (self.result_identity, self.path, self.fingerprint)):
            raise TypeError("a fit input's result identity, path and fingerprint are text")
        if type(self.generation) is not int:
            raise TypeError("a fit input's generation is an int")
        if self.background not in BACKGROUND_POLICIES:
            raise TypeError(f"a fit input's background policy is one of {BACKGROUND_POLICIES}")
        _pairs(self.selection, "selection parameters")
        _pairs(self.settings, "settings")


@dataclass(frozen=True, slots=True)
class FitAxis:
    """One coordinate of the fitted values: its name, its unit and its values (bin centres), owned and read-only."""

    name: str
    unit: str
    values: np.ndarray

    def __post_init__(self) -> None:
        if type(self.name) is not str or type(self.unit) is not str:
            raise TypeError("a fit axis's name and unit are text")
        values = np.array(self.values, dtype=float, copy=True, order="C")
        if values.ndim != 1 or values.size == 0:
            raise ValueError("a fit axis's values are one non-empty dimension")
        values.flags.writeable = False
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class FitInputSnapshot:
    """A fit's detached input (the module docstring): its ``key``, its coordinates, its linear values and their
    validity ``mask`` (True: a sample to fit; None: every finite sample), all owned and read-only, and the result's
    recorded wavelength."""

    key: FitInputKey
    axes: tuple[FitAxis, ...]
    values: np.ndarray
    mask: np.ndarray | None = None
    wavelength_angstrom: float | None = None

    def __post_init__(self) -> None:
        if type(self.key) is not FitInputKey:
            raise TypeError("a fit input snapshot's key is a FitInputKey")
        if type(self.axes) is not tuple or not self.axes or any(type(axis) is not FitAxis for axis in self.axes):
            raise TypeError("a fit input snapshot's axes are FitAxis, one per dimension")
        values = np.array(self.values, dtype=float, copy=True, order="C")
        if values.shape != tuple(axis.values.size for axis in self.axes):
            raise ValueError("a fit input's values have one dimension per axis, each its axis's length")
        values.flags.writeable = False
        object.__setattr__(self, "values", values)
        if self.mask is not None:
            mask = np.array(self.mask, copy=True, order="C")
            if mask.dtype != bool or mask.shape != values.shape:
                raise ValueError("a fit input's mask is a boolean array of its values' shape")
            mask.flags.writeable = False
            object.__setattr__(self, "mask", mask)
        wavelength = self.wavelength_angstrom
        if wavelength is not None and (type(wavelength) not in {int, float} or not math.isfinite(wavelength)
                                       or wavelength <= 0):
            raise ValueError("a fit input's wavelength is a finite positive number of angstrom, or None")

    def is_current(self, key: FitInputKey) -> bool:
        """Whether this input is still the host's current one (its ``key`` is ``key``)."""
        return self.key == key

    def fitted_values(self) -> np.ndarray:
        """The values to fit, a sample outside the mask NaN (a fitter takes only the finite ones): read-only."""
        if self.mask is None:
            return self.values
        values = np.where(self.mask, self.values, np.nan)
        values.flags.writeable = False
        return values

    def trace_input(self):
        """A one-dimensional input as the displayed-fit operations' trace (``DisplayedTraceInput``), so a host's
        pattern is fitted by the same policy as Int's displayed one: its receipt is this snapshot's key."""
        from xrd_tools.analysis.display_fit_operations import DisplayedTraceInput, DisplayedTraceReceipt
        if len(self.axes) != 1:
            raise ValueError("a displayed-fit trace is one-dimensional")
        axis, key = self.axes[0], self.key
        receipt = DisplayedTraceReceipt(_SCHEMA, key.generation, key.fingerprint, key.result_identity,
                                        Path(key.path), ";".join(f"{name}={value!r}" for name, value in key.selection))
        return DisplayedTraceInput(axis.values, self.fitted_values(), axis.name, axis.unit, key.result_identity, 0,
                                   receipt)


__all__ = ["BACKGROUND_POLICIES", "FitAxis", "FitInputKey", "FitInputSnapshot"]
