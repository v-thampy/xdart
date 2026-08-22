"""Bounded, Qt-free display-background aggregation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

_DOMAINS = frozenset({"raw", "integrated_1d", "integrated_2d"})
_BLOCK = 65_536
def _shape(value: object, *, ndim: int) -> tuple[int, ...]:
    if (type(value) is not tuple or len(value) != ndim
            or any(type(part) is not int or part <= 0 for part in value)):
        raise ValueError("display-background shape is invalid")
    return value
def _bytes_root(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(contiguous.shape)
    result.setflags(write=False)
    return result
def _is_bytes_backed(array: np.ndarray) -> bool:
    root: object = array
    seen: set[int] = set()
    while isinstance(root, np.ndarray) and id(root) not in seen:
        seen.add(id(root))
        root = root.base
    return isinstance(root, bytes) and not array.flags.writeable
@dataclass(frozen=True, slots=True)
class DisplayBackgroundPlan:
    domain: str
    contributor_ids: tuple[str, ...]
    value_shapes: tuple[tuple[int, ...], ...]
    axis_shapes: tuple[tuple[tuple[int, ...], ...], ...]
    axis_units: tuple[tuple[str, ...], ...]
    differing_grid_1d: bool = False
    def __post_init__(self) -> None:
        count = len(self.contributor_ids)
        ndim = 1 if self.domain == "integrated_1d" else 2
        axis_count = {"raw": 0, "integrated_1d": 1,
                      "integrated_2d": 2}.get(self.domain, -1)
        if (
            self.domain not in _DOMAINS
            or type(self.contributor_ids) is not tuple
            or not self.contributor_ids
            or any(type(item) is not str or not item for item in self.contributor_ids)
            or len(set(self.contributor_ids)) != count
            or type(self.value_shapes) is not tuple or len(self.value_shapes) != count
            or type(self.axis_shapes) is not tuple or len(self.axis_shapes) != count
            or type(self.axis_units) is not tuple or len(self.axis_units) != count
            or type(self.differing_grid_1d) is not bool
            or (self.differing_grid_1d and self.domain != "integrated_1d")
        ):
            raise ValueError("display-background plan is invalid")
        for value_shape, axis_shapes, axis_units in zip(
            self.value_shapes, self.axis_shapes, self.axis_units, strict=True
        ):
            _shape(value_shape, ndim=ndim)
            if (type(axis_shapes) is not tuple or len(axis_shapes) != axis_count
                or type(axis_units) is not tuple or len(axis_units) != axis_count
                or any(type(unit) is not str for unit in axis_units)
            ):
                raise ValueError("display-background axis identity is invalid")
            for axis_shape in axis_shapes:
                _shape(axis_shape, ndim=1)
        if self.domain != "integrated_1d" and any(
                shape != self.value_shapes[0] for shape in self.value_shapes):
            raise ValueError("display-background value shapes differ")
        if any(units != self.axis_units[0] for units in self.axis_units):
            raise ValueError("display-background axis units differ")
@dataclass(frozen=True, slots=True)
class DisplayBackgroundResult:
    domain: str
    contributor_ids: tuple[str, ...]
    values: np.ndarray
    finite_counts: np.ndarray
    axes: tuple[np.ndarray, ...]
    axis_units: tuple[str, ...]
    result_identity: tuple[object, ...]
    diagnostics: tuple[str, ...] = ()
    def __post_init__(self) -> None:
        if (
            self.domain not in _DOMAINS
            or type(self.contributor_ids) is not tuple
            or type(self.values) is not np.ndarray or self.values.dtype != np.dtype(np.float64)
            or type(self.finite_counts) is not np.ndarray or self.finite_counts.dtype != np.dtype(np.uint64)
            or self.values.shape != self.finite_counts.shape
            or type(self.axes) is not tuple or type(self.axis_units) is not tuple
            or len(self.axes) != len(self.axis_units)
            or any(type(axis) is not np.ndarray or axis.ndim != 1 for axis in self.axes)
            or not _is_bytes_backed(self.values)
            or not _is_bytes_backed(self.finite_counts)
            or any(not _is_bytes_backed(axis) for axis in self.axes)
            or type(self.result_identity) is not tuple
            or type(self.diagnostics) is not tuple
            or any(type(item) is not str for item in self.diagnostics)
        ):
            raise ValueError("display-background result is invalid")
def _strict_axis(axis: np.ndarray) -> None:
    if (type(axis) is not np.ndarray or axis.ndim != 1
            or axis.dtype != np.dtype(np.float64) or not axis.flags.c_contiguous):
        raise ValueError("display-background axis is not numeric contiguous 1-D")
    prior: float | None = None
    for item in axis:
        value = float(item)
        if not np.isfinite(value) or prior is not None and value <= prior:
            raise ValueError("display-background axis must be finite increasing")
        prior = value
def _same_axis(first: np.ndarray, second: np.ndarray) -> bool:
    if first.shape != second.shape:
        return False
    for start in range(0, first.size, _BLOCK):
        stop = min(first.size, start + _BLOCK)
        if not np.array_equal(first[start:stop], second[start:stop]):
            return False
    return True
def run_display_background(
    plan: DisplayBackgroundPlan,
    contributors: tuple[tuple[np.ndarray, ...], ...],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> DisplayBackgroundResult:
    """Return the promoted finite mean for one frozen ordered plan."""
    if type(plan) is not DisplayBackgroundPlan:
        raise TypeError("display-background plan must be exact")
    plan.__post_init__()
    if (
        type(contributors) is not tuple
        or len(contributors) != len(plan.contributor_ids)
        or cancelled is not None and not callable(cancelled)
    ):
        raise ValueError("display-background contributors are invalid")
    axis_count = {"raw": 0, "integrated_1d": 1, "integrated_2d": 2}[plan.domain]
    for index, item in enumerate(contributors):
        if type(item) is not tuple or len(item) != axis_count + 1:
            raise ValueError("display-background contributor tuple is invalid")
        value, *axes = item
        if (
            type(value) is not np.ndarray
            or value.dtype.kind not in "fiu"
            or not value.flags.c_contiguous
            or value.shape != plan.value_shapes[index]
            or tuple(axis.shape for axis in axes) != plan.axis_shapes[index]
        ):
            raise ValueError("display-background contributor identity differs")
        for axis in axes:
            _strict_axis(axis)
        if (plan.domain == "integrated_2d"
                and (axes[0].shape != (value.shape[0],)
                     or axes[1].shape != (value.shape[1],))):
            raise ValueError("display-background 2-D axes differ")
        if plan.domain == "integrated_1d" and axes[0].shape != value.shape:
            raise ValueError("display-background 1-D axis differs")
        if (plan.differing_grid_1d and (value.dtype != np.dtype(np.float64)
                                       or axes[0].dtype != np.dtype(np.float64))):
            raise ValueError("differing-grid 1-D requires C float64 inputs")
    reference_axes = contributors[0][1:]
    if plan.domain == "integrated_2d" and any(
            not _same_axis(reference, axis) for item in contributors[1:]
            for reference, axis in zip(reference_axes, item[1:], strict=True)):
        raise ValueError("display-background 2-D grids differ")
    if plan.domain == "integrated_1d":
        differs = any(not _same_axis(reference_axes[0], item[1])
                      for item in contributors[1:])
        if differs != plan.differing_grid_1d:
            raise ValueError("display-background interpolation policy differs")
    shape = plan.value_shapes[0]
    size = int(np.prod(shape))
    sums = np.zeros(size, dtype=np.float64)
    counts = np.zeros(size, dtype=np.uint64)
    work = np.empty(min(size, _BLOCK), dtype=np.float64)
    predicate = np.empty(work.shape, dtype=np.bool_)
    reference = reference_axes[0] if reference_axes else None
    for item in contributors:
        if cancelled is not None and cancelled():
            raise InterruptedError("display-background operation cancelled")
        source = item[0].reshape(-1)
        interpolated = None
        if reference is not None and not _same_axis(reference, item[1]):
            interpolated = np.interp(
                reference, item[1], item[0], left=np.nan, right=np.nan
            )
            source = interpolated
        contributor_finite = 0
        for start in range(0, size, _BLOCK):
            if cancelled is not None and cancelled():
                raise InterruptedError("display-background operation cancelled")
            stop = min(size, start + _BLOCK)
            width = stop - start
            np.copyto(work[:width], source[start:stop], casting="unsafe")
            np.isfinite(work[:width], out=predicate[:width])
            contributor_finite += int(predicate[:width].sum())
            np.add(counts[start:stop], predicate[:width], out=counts[start:stop])
            np.logical_not(predicate[:width], out=predicate[:width])
            np.copyto(work[:width], 0.0, where=predicate[:width])
            np.add(sums[start:stop], work[:width], out=sums[start:stop])
        if contributor_finite == 0:
            raise ValueError("display-background contributor has no finite values")
        del interpolated
    for start in range(0, size, _BLOCK):
        stop = min(size, start + _BLOCK)
        width = stop - start
        np.greater(counts[start:stop], 0, out=predicate[:width])
        np.divide(sums[start:stop], counts[start:stop], out=sums[start:stop],
                  where=predicate[:width])
        np.logical_not(predicate[:width], out=predicate[:width])
        np.copyto(sums[start:stop], np.nan, where=predicate[:width])
    values = _bytes_root(sums.reshape(shape), np.dtype(np.float64))
    finite_counts = _bytes_root(counts.reshape(shape), np.dtype(np.uint64))
    axes = tuple(_bytes_root(axis, np.dtype(np.float64)) for axis in reference_axes)
    identity = ("display-background", plan.domain, plan.contributor_ids,
                plan.value_shapes, plan.axis_shapes, plan.axis_units)
    return DisplayBackgroundResult(
        plan.domain, plan.contributor_ids, values, finite_counts, axes,
        plan.axis_units[0], identity,
        (f"contributors={len(contributors)}", f"finite={int(counts.sum())}"))

__all__ = ["DisplayBackgroundPlan", "DisplayBackgroundResult",
           "run_display_background"]
