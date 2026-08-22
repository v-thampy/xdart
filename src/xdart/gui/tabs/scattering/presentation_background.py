"""Workspace-local bounded custody for presentation-only backgrounds."""
from __future__ import annotations
from dataclasses import dataclass
from threading import Lock
import numpy as np

from xrd_tools.core.metadata import resolve_monitor_norm
from xrd_tools.reduction.background import (
    DisplayBackgroundPlan, DisplayBackgroundResult, run_display_background,
)
from .operation_values import OperationContextStamp

_CAPACITY = 536_870_912
_BLOCK = 65_536


def _immutable_copy(array: np.ndarray) -> np.ndarray:
    copied = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    copied.setflags(write=False)
    return copied


def prepare_background_plan(payloads, domain: str, current, *,
                            contributor_count=None, norm_channel: str = ""):
    """Freeze contributors separately from every exact renderer target."""
    if (domain not in {"raw", "integrated_1d", "integrated_2d"} or not payloads
            or type(norm_channel) is not str):
        return None
    count = len(payloads) if contributor_count is None else contributor_count
    if type(count) is not int or not 1 <= count <= len(payloads):
        return None
    items, units, identities = [], [], []
    for payload in payloads:
        view, frame = payload.view, payload.frame_key
        if domain == "raw":
            item, row_units = (view.raw,), ()
        elif domain == "integrated_1d":
            axis = view.axis_1d
            item = (view.intensity_1d, None if axis is None else axis.values)
            row_units = (() if axis is None else (f"{axis.label}\0{axis.unit}",))
        else:
            y_axis, x_axis = view.axis_2d_y, view.axis_2d_x
            item = (view.intensity_2d,
                    None if y_axis is None else y_axis.values,
                    None if x_axis is None else x_axis.values)
            row_units = (() if y_axis is None or x_axis is None else
                         (f"{y_axis.label}\0{y_axis.unit}",
                          f"{x_axis.label}\0{x_axis.unit}"))
        if any(type(array) is not np.ndarray for array in item):
            items.append(None); units.append(row_units); identities.append(""); continue
        items.append(item); units.append(row_units)
        identities.append(f"{frame.source_scan}\0{frame.artifact}\0{frame.local_frame_label}\0{id(frame)}")
    contributors = items[:count]
    if any(item is None for item in contributors):
        return None
    try:
        current_index = next(index for index, payload in enumerate(payloads)
                             if payload.frame_key is current)
    except StopIteration:
        return None
    differing = (domain == "integrated_1d" and any(
        not np.array_equal(contributors[0][1], item[1])
        for item in contributors[1:]))
    try:
        plan = DisplayBackgroundPlan(
            domain, tuple(identities[:count]), tuple(item[0].shape for item in contributors),
            tuple(tuple(axis.shape for axis in item[1:]) for item in contributors),
            tuple(units[:count]), differing)
    except (TypeError, ValueError):
        return None
    indices = tuple(range(len(items))) if domain == "integrated_1d" else (current_index,)
    keys, targets, facts = [], [], []
    for index in indices:
        item = items[index]
        if item is None:
            if index < count: return None
            continue
        divisor = 1.0
        if domain == "integrated_1d" and norm_channel:
            resolved = resolve_monitor_norm(payloads[index].view.metadata_numeric, norm_channel)
            if resolved is None:
                if index < count: return None
                continue
            divisor = float(resolved)
        if domain == "integrated_1d" and units[index] != plan.axis_units[0]:
            return None
        frame = payloads[index].frame_key
        keys.append(id(frame)); targets.append(
            (*item, divisor) if domain == "integrated_1d" else item)
        facts.append((id(frame), item[0].shape,
                      tuple(axis.shape for axis in item[1:]), units[index], divisor))
    if not targets:
        return None
    return (plan, tuple(contributors), tuple(keys), tuple(targets),
            (norm_channel, tuple(facts)))


@dataclass(frozen=True, slots=True)
class DisplayBackgroundTransferReceipt:
    reservation: int
    domain: str
    active_key: tuple[object, ...]
    result_identity: tuple[object, ...]
    retained_bytes: int

    def __post_init__(self) -> None:
        if (type(self.reservation) is not int or self.reservation < 1
                or self.domain not in {"raw", "integrated_1d", "integrated_2d"}
                or type(self.active_key) is not tuple
                or type(self.result_identity) is not tuple
                or type(self.retained_bytes) is not int or self.retained_bytes < 0):
            raise ValueError("display-background transfer receipt is invalid")


@dataclass(frozen=True, slots=True)
class DisplayBackgroundRendererReleaseReceipt:
    active_key: tuple[object, ...]
    released: bool


class PresentationBackgroundOwner:
    """Own one RESERVED/STAGED/ACTIVE bundle under one fixed byte grant."""

    def __init__(self, *, capacity_bytes: int) -> None:
        if type(capacity_bytes) is not int or capacity_bytes != _CAPACITY:
            raise ValueError("presentation background requires its standalone 512 MiB grant")
        self._lock = Lock()
        self._capacity = capacity_bytes
        self._phase = "EMPTY"
        self._next_reservation = 1
        self._reservation: int | None = None
        self._stamp: OperationContextStamp | None = None
        self._active_key: tuple[object, ...] | None = None
        self._plan: DisplayBackgroundPlan | None = None
        self._contributors: tuple[tuple[np.ndarray, ...], ...] = ()
        self._projection_keys: tuple[int, ...] = ()
        self._projection_indices: tuple[int, ...] = ()
        self._projection_shapes: tuple[tuple[int, ...], ...] = ()
        self._result: DisplayBackgroundResult | None = None
        self._display_values: tuple[np.ndarray, ...] = ()
        self._reserved_bytes = 0
        self._active_bytes = 0
        self._last_finalize: tuple[int, str] | None = None

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return self._reserved_bytes

    @property
    def active_bytes(self) -> int:
        with self._lock:
            return self._active_bytes

    @property
    def active_key(self) -> tuple[object, ...] | None:
        with self._lock:
            return self._active_key if self._phase == "ACTIVE" else None

    def reserve(
        self, plan: DisplayBackgroundPlan,
        contributors: tuple[tuple[np.ndarray, ...], ...],
        *, stamp: OperationContextStamp, active_key: tuple[object, ...],
        projection_keys: tuple[int, ...], projection_indices: tuple[int, ...] = (),
        projection_shapes: tuple[tuple[int, ...], ...] = (),
    ) -> int | None:
        if (type(plan) is not DisplayBackgroundPlan
                or type(stamp) is not OperationContextStamp
                or type(active_key) is not tuple
                or type(projection_keys) is not tuple
                or type(projection_indices) is not tuple
                or type(projection_shapes) is not tuple):
            return None
        if projection_shapes:
            if (projection_indices or len(projection_keys) != len(projection_shapes)
                    or any(type(shape) is not tuple or not shape
                           or any(type(part) is not int or part <= 0 for part in shape)
                           for shape in projection_shapes)):
                return None
        elif (len(projection_keys) != len(projection_indices) or not projection_indices
              or any(type(index) is not int or index < 0
                     or index >= len(contributors) for index in projection_indices)):
            return None
        try:
            plan.__post_init__(); stamp.__post_init__()
            if type(contributors) is not tuple or len(contributors) != len(plan.contributor_ids):
                return None
            arrays = tuple(array for item in contributors for array in item)
            if any(type(array) is not np.ndarray for array in arrays):
                return None
            if plan.differing_grid_1d and any(
                    array.dtype != np.dtype(np.float64) or not array.flags.c_contiguous
                    for array in arrays):
                return None
            c_bytes = sum(array.nbytes for array in arrays)
            shape = plan.value_shapes[0]
            v = int(np.prod(shape))
            x = (0 if plan.domain == "raw" else
                 8 * shape[0] if plan.domain == "integrated_1d" else
                 8 * (shape[0] + shape[1]))
            if not projection_shapes:
                projection_shapes = tuple(plan.value_shapes[index]
                                          for index in projection_indices)
            d_bytes = 8 * sum(int(np.prod(item)) for item in projection_shapes)
            q = c_bytes + (32 + 8 * int(plan.differing_grid_1d)) * v + x + d_bytes + 589_824
        except (TypeError, ValueError, OverflowError):
            return None
        with self._lock:
            if self._phase not in {"EMPTY", "RELEASED"} or q > self._capacity:
                return None
            reservation = self._next_reservation
            self._next_reservation += 1
            self._phase = "RESERVED"
            self._reservation, self._stamp = reservation, stamp
            self._active_key, self._plan = active_key, plan
            self._projection_keys = projection_keys
            self._projection_indices = projection_indices
            self._projection_shapes = projection_shapes
            self._reserved_bytes = q
        try:
            copied = tuple(tuple(_immutable_copy(array) for array in item)
                           for item in contributors)
        except BaseException:
            self.abort(reservation, "COPY_FAILED")
            return None
        with self._lock:
            if self._reservation != reservation or self._phase != "RESERVED":
                return None
            self._contributors = copied
        return reservation

    def run_and_stage(self, reservation: int, cancelled) -> DisplayBackgroundTransferReceipt:
        with self._lock:
            if self._reservation != reservation or self._phase != "RESERVED":
                raise RuntimeError("display-background reservation is not runnable")
            plan, contributors = self._plan, self._contributors
        if plan is None:
            raise RuntimeError("display-background plan was lost")
        result = run_display_background(plan, contributors, cancelled=cancelled)
        retained = result.values.nbytes + result.finite_counts.nbytes + sum(
            axis.nbytes for axis in result.axes)
        with self._lock:
            if self._reservation != reservation or self._phase != "RESERVED":
                raise RuntimeError("display-background reservation became stale")
            self._result = result
            self._contributors = ()
            self._phase = "STAGED"
            self._active_bytes = retained
            receipt = DisplayBackgroundTransferReceipt(
                reservation, result.domain, self._active_key or (),
                result.result_identity, retained)
        return receipt

    def promote(
        self, receipt: DisplayBackgroundTransferReceipt,
        targets: tuple[tuple[np.ndarray, ...], ...],
    ) -> bool:
        with self._lock:
            if (type(receipt) is not DisplayBackgroundTransferReceipt
                    or self._reservation != receipt.reservation
                    or self._phase != "STAGED" or self._result is None
                    or receipt.active_key != self._active_key
                    or receipt.result_identity != self._result.result_identity
                    or len(targets) != len(self._projection_shapes)):
                return False
            result, shapes = self._result, self._projection_shapes
        displayed: list[np.ndarray] = []
        try:
            for target, shape in zip(targets, shapes, strict=True):
                source = target[0]
                divisor = target[2] if result.domain == "integrated_1d" and len(target) == 3 else None
                if (type(source) is not np.ndarray or source.shape != shape
                        or result.domain == "integrated_1d"
                        and (type(divisor) is not float or not np.isfinite(divisor)
                             or divisor <= 0.0)):
                    raise ValueError("display target changed shape")
                owner = np.empty(shape, dtype=np.float64)
                flat = owner.reshape(-1); source_flat = source.reshape(-1)
                if result.domain == "integrated_1d" and not np.array_equal(
                        target[1], result.axes[0]):
                    for start in range(0, flat.size, _BLOCK):
                        stop = min(flat.size, start + _BLOCK)
                        background = np.interp(target[1][start:stop], result.axes[0],
                                               result.values, left=np.nan, right=np.nan)
                        np.subtract(source_flat[start:stop], background, out=flat[start:stop])
                else:
                    np.subtract(source, result.values, out=owner, casting="unsafe")
                if result.domain == "integrated_1d":
                    np.divide(owner, divisor, out=owner)
                owner.setflags(write=False); displayed.append(owner)
        except BaseException:
            return False
        with self._lock:
            if self._reservation != receipt.reservation or self._phase != "STAGED":
                return False
            self._display_values = tuple(displayed)
            self._phase = "ACTIVE"
            self._active_bytes = receipt.retained_bytes + sum(x.nbytes for x in displayed)
        return True

    def projection(self) -> tuple[str, tuple[tuple[int, np.ndarray], ...], tuple[object, ...]] | None:
        with self._lock:
            if self._phase != "ACTIVE" or self._plan is None or self._active_key is None:
                return None
            return (self._plan.domain,
                    tuple(zip(self._projection_keys, self._display_values, strict=True)),
                    self._active_key)

    def release(self, receipt: DisplayBackgroundRendererReleaseReceipt | None = None) -> bool:
        with self._lock:
            if self._phase == "ACTIVE" and (
                    type(receipt) is not DisplayBackgroundRendererReleaseReceipt
                    or not receipt.released or receipt.active_key != self._active_key):
                return False
            if self._phase in {"EMPTY", "RELEASED"}:
                return True
            if self._phase in {"RESERVED", "STAGED", "CLEANUP_PENDING"}:
                self._phase = "CLEANUP_PENDING"
                return False
            self._phase = "CLEANUP_PENDING"
            self._drop_locked()
            return True

    def abort(self, reservation: int, outcome: str) -> bool:
        with self._lock:
            if self._reservation != reservation:
                return self._last_finalize == (reservation, outcome)
            self._phase = "CLEANUP_PENDING"
            self._last_finalize = (reservation, outcome)
            self._drop_locked()
            return True

    def finalize(self, reservation: int, outcome: str) -> None:
        with self._lock:
            if self._reservation != reservation:
                return
            self._last_finalize = (reservation, outcome)
            keep = outcome == "TRANSFERRED" and self._phase == "STAGED"
        if not keep:
            self.abort(reservation, outcome)

    def _drop_locked(self) -> None:
        self._contributors = (); self._result = None; self._display_values = ()
        self._plan = None; self._stamp = None; self._active_key = None
        self._projection_keys = (); self._projection_indices = (); self._projection_shapes = ()
        self._reservation = None; self._reserved_bytes = 0; self._active_bytes = 0
        self._phase = "RELEASED"


__all__ = ["DisplayBackgroundRendererReleaseReceipt",
           "DisplayBackgroundTransferReceipt", "PresentationBackgroundOwner"]
