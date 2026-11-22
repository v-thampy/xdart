"""Workspace-local bounded custody for presentation-only backgrounds."""
from __future__ import annotations
from collections import OrderedDict, deque
from dataclasses import dataclass
from threading import Lock
import weakref
import numpy as np

from xrd_tools.core.metadata import resolve_monitor_norm
from xrd_tools.reduction.background import (
    DisplayBackgroundPlan, DisplayBackgroundResult, run_display_background,
)
from .operation_values import OperationContextStamp
from .scientific_axes import heavy_projection, present_gi_map, requested_image_axis, same_sampled_bins

_CAPACITY = 536_870_912
_BLOCK = 65_536
#: the backgrounds that follow the frame drawn (``frame_values``): the 2D Viewer's raw one and Int's 2-D one
_FOLLOWING = frozenset({"raw", "integrated_2d"})


def _same_grid(shown, reference) -> bool:
    """Whether two presented map grids (y and x axes: values, label, unit, edges) are the same."""
    if type(shown) is not tuple or type(reference) is not tuple or len(shown) != 2 or len(reference) != 2:
        return False
    for left, right in zip(shown, reference, strict=True):
        values, other = getattr(left, "values", None), getattr(right, "values", None)
        if (type(values) is not np.ndarray or type(other) is not np.ndarray or values.shape != other.shape
                or not np.array_equal(values, other)
                or (getattr(left, "label", None), getattr(left, "unit", None), getattr(left, "edges", None))
                != (getattr(right, "label", None), getattr(right, "unit", None), getattr(right, "edges", None))):
            return False
    return True


def _immutable_copy(array: np.ndarray) -> np.ndarray:
    copied = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    copied.setflags(write=False)
    return copied


def prepare_background_plan(payloads, domain: str, current, *,
                            contributor_count=None, norm_channel: str = "",
                            image_axis: str | None = None):
    """Freeze contributors separately from every exact renderer target."""
    if (domain not in {"raw", "integrated_1d", "integrated_2d"} or not payloads
            or type(norm_channel) is not str):
        return None
    count = len(payloads) if contributor_count is None else contributor_count
    if type(count) is not int or not 1 <= count <= len(payloads):
        return None
    items, units, identities = [], [], []
    grids = {}
    for index, payload in enumerate(payloads):
        view, frame = payload.view, payload.frame_key
        if domain == "raw":
            item, row_units = (view.raw,), ()
        elif domain == "integrated_1d":
            axis = view.axis_1d
            item = (view.intensity_1d, None if axis is None else axis.values)
            row_units = (() if axis is None else (f"{axis.label}\0{axis.unit}",))
        elif image_axis is not None:
            shown = present_gi_map(payload, image_axis)
            heavy = heavy_projection(shown, requested_axis=requested_image_axis(shown, image_axis))
            if heavy is None or heavy.cake is None:
                items.append(None); units.append(()); identities.append(""); continue
            y_axis, x_axis = heavy.cake_y, heavy.cake_x
            grids[index] = (y_axis, x_axis)
            item = (heavy.cake, y_axis.values, x_axis.values)
            row_units = (f"{y_axis.label}\0{y_axis.unit}", f"{x_axis.label}\0{x_axis.unit}")
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
        keys.append(id(frame))
        targets.append((*item, divisor) if domain == "integrated_1d" else
                       (*item, grids[index]) if index in grids else item)
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
        #: a keyed layout's background (``activate_keyed``): its page's own immutable source, re-sliced at
        #: every draw, so it outlives slice steps and a change of the result shown
        self._keyed_source: object | None = None
        #: a following background's drawn frame (``frame_values``): a weak reference to the image its displayed
        #: values were taken from, so the same image gives the same array and the owner never keeps a frame alive
        self._shown_source: weakref.ref | None = None
        #: the presented map grid captured for a following 2-D background
        self._shown_grid: tuple | None = None
        #: a 1-D background's values for each pattern drawn (``trace_values``), least recently drawn first: keyed by
        #: the pattern's id, each holding weak references to its intensity and axis (an id can come back as
        #: another pattern's), the divisor and the values; charged to this owner's grant, the oldest dropped first
        self._followed: OrderedDict[int, tuple] = OrderedDict()
        self._followed_bytes = 0
        #: the patterns gone since (their weak references' callbacks, which only append, never lock): their values
        #: are dropped at the next ``trace_values``, so a long Run's patterns do not stay held up to the grant
        self._followed_gone: deque = deque()
        #: the normalization channel a 1-D background was set under ("" for none): its mean is of the contributors
        #: each divided by its own monitor, so a curve normalized otherwise does not fit it (``follow_trace``)
        self._norm_channel = ""

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
        projection_shapes: tuple[tuple[int, ...], ...] = (), norm_channel: str = "",
        contributor_divisors: tuple[float, ...] = (),
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
            if type(contributor_divisors) is not tuple or contributor_divisors and (
                    plan.domain != "integrated_1d" or len(contributor_divisors) != len(contributors)
                    or any(type(divisor) is not float or not np.isfinite(divisor) or divisor <= 0.0
                           for divisor in contributor_divisors)):
                return None
            arrays = tuple(array for item in contributors for array in item)
            if any(type(array) is not np.ndarray for array in arrays):
                return None
            normalized_bytes = tuple(8 * item[0].size for item in contributors) if contributor_divisors else ()
            # Contributors on different grids are interpolated in float64 (run_display_background): a display
            # copy's float32 rows (a Browse selection's detached ones, his pick "In 1.2.18") are copied as float64
            # here, charged at 8 bytes a value, instead of refusing the Set
            widen = plan.differing_grid_1d
            def copied_bytes(array):
                return 8 * array.size if widen else array.nbytes
            c_bytes = (sum(normalized_bytes) + sum(copied_bytes(array) for item in contributors for array in item[1:])
                       if normalized_bytes else sum(copied_bytes(array) for array in arrays))
            widen_bytes = max((8 * array.size for array in arrays), default=0) if widen else 0
            shape = plan.value_shapes[0]
            v = int(np.prod(shape))
            x = (0 if plan.domain == "raw" else
                 8 * shape[0] if plan.domain == "integrated_1d" else
                 8 * (shape[0] + shape[1]))
            if not projection_shapes:
                projection_shapes = tuple(plan.value_shapes[index]
                                          for index in projection_indices)
            d_bytes = 8 * sum(int(np.prod(item)) for item in projection_shapes)
            stage_bytes = (32 + 8 * int(plan.differing_grid_1d)) * v + x + d_bytes + 589_824
            q = c_bytes + max(stage_bytes, max(normalized_bytes, default=0), widen_bytes)
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
            self._norm_channel = norm_channel if type(norm_channel) is str else ""
        try:
            if contributor_divisors:
                # Normalize one contributor at a time, inside the copy grant. Use its own monitor before averaging.
                scaled_rows = zip(contributors, contributor_divisors, strict=True)
                copied = tuple(
                    (_immutable_copy(np.divide(item[0], divisor, dtype=np.float64)),
                     *(_immutable_copy(np.ascontiguousarray(axis, dtype=np.float64) if widen else axis)
                       for axis in item[1:]))
                    for item, divisor in scaled_rows
                )
            elif widen:
                copied = tuple(tuple(_immutable_copy(np.ascontiguousarray(array, dtype=np.float64)) for array in item)
                               for item in contributors)
            else:
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

    def activate_keyed(self, source: object, *, stamp: OperationContextStamp,
                       active_key: tuple[object, ...], nbytes: int) -> bool:
        """Take a keyed layout's background (RSM's six panels; AUD-9 stage 2b G8d): ``source`` is its page's
        own immutable background (RSM: the result's viewer values, sharing the payload's read-only buffer, and
        its viewer model), charged ``nbytes`` against this owner's one grant. Unlike a frame background there
        is nothing to average (one result), so it goes from EMPTY or RELEASED straight to ACTIVE; the page
        builds the background's panels at the indices drawn from it (``keyed_source``), and
        ``apply_keyed_presentation_background`` subtracts them. Released as every background is: ``release``
        with the view's receipt. ``active_key[2]`` is "keyed", the view's release domain."""
        if (source is None or type(stamp) is not OperationContextStamp or type(active_key) is not tuple
                or len(active_key) < 3 or active_key[2] != "keyed"
                or type(nbytes) is not int or not 0 <= nbytes <= self._capacity):
            return False
        try:
            stamp.__post_init__()
        except (TypeError, ValueError):
            return False
        with self._lock:
            if self._phase not in {"EMPTY", "RELEASED"}:
                return False
            self._reservation = self._next_reservation
            self._next_reservation += 1
            self._stamp, self._active_key, self._keyed_source = stamp, active_key, source
            self._active_bytes = nbytes
            self._phase = "ACTIVE"
        return True

    def keyed_source(self) -> object | None:
        """The ACTIVE keyed background's source, or None (none, or a frame background)."""
        with self._lock:
            return self._keyed_source if self._phase == "ACTIVE" else None

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
            result, shapes, plan = self._result, self._projection_shapes, self._plan
        displayed: list[np.ndarray] = []
        grid = None
        try:
            for target, shape in zip(targets, shapes, strict=True):
                source = target[0]
                divisor = target[2] if result.domain == "integrated_1d" and len(target) == 3 else None
                if (type(source) is not np.ndarray or source.shape != shape
                        or result.domain == "integrated_1d"
                        and (type(divisor) is not float or not np.isfinite(divisor)
                             or divisor <= 0.0)):
                    raise ValueError("display target changed shape")
                if result.domain == "integrated_2d" and len(target) == 4:
                    grid = target[3]
                    if (not _same_grid(grid, grid) or plan is None
                            or tuple(f"{axis.label}\0{axis.unit}" for axis in grid) != plan.axis_units[0]
                            or any(not np.array_equal(axis.values, reference)
                                   for axis, reference in zip(grid, result.axes, strict=True))):
                        return False
                owner = np.empty(shape, dtype=np.float64)
                flat = owner.reshape(-1)
                if result.domain == "integrated_1d":
                    # I/m - <I_bg/m_bg> (his pick "Each by its own monitor"): the contributors were divided by their
                    # own monitors before the mean
                    np.divide(source, divisor, out=owner, casting="unsafe")
                    if not same_sampled_bins(target[1], result.axes[0]):
                        for start in range(0, flat.size, _BLOCK):
                            stop = min(flat.size, start + _BLOCK)
                            flat[start:stop] -= np.interp(target[1][start:stop], result.axes[0],
                                                          result.values, left=np.nan, right=np.nan)
                    else:
                        np.subtract(owner, result.values, out=owner)
                else:
                    np.subtract(source, result.values, out=owner, casting="unsafe")
                owner.setflags(write=False); displayed.append(owner)
        except BaseException:
            return False
        with self._lock:
            if self._reservation != receipt.reservation or self._phase != "STAGED":
                return False
            self._display_values = tuple(displayed)
            if result.domain in _FOLLOWING and len(targets) == 1:
                self._shown_source = weakref.ref(targets[0][0])
                self._shown_grid = grid
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

    def frame_values(self, frame_key: int, source, grid=()):
        """A raw background (the 2D Viewer's) or an Int 2-D one subtracted from the frame drawn now
        (LIVE-1218-VIEWER2D-BG): the displayed values for ``source``, the same array while the same frame or image
        is drawn; a note (a str) when the background does not fit ``source``, which is drawn as it is (his word: "if
        the shapes don't match the BG should be ignored. No need to throw an error"); None when no such background
        is active. A 2-D background fits a map of its shape whose presented axes (``grid``: the shown map's y and x
        axes) equal the grid captured at Set. The values replace the frame displayed
        before, in the one displayed frame this background's reservation charged, so its memory stays as
        reserved."""
        with self._lock:
            if (self._phase != "ACTIVE" or self._result is None or self._plan is None
                    or self._plan.domain not in _FOLLOWING or len(self._display_values) != 1):
                return None
            result, reservation, domain = self._result, self._reservation, self._plan.domain
            same_image = self._shown_source is not None and self._shown_source() is source
            if domain == "integrated_2d":
                if self._shown_grid is None and same_image and _same_grid(grid, grid):
                    self._shown_grid = grid
                if self._shown_grid is None:
                    return "The background's original map grid is unavailable, so this frame is drawn as it is."
                if type(source) is np.ndarray and source.shape == result.values.shape:
                    if not _same_grid(grid, self._shown_grid):
                        return "The background's map axes differ from this frame's, which is drawn as it is."
            if same_image:
                return self._display_values[0]
        shape = tuple(result.values.shape)
        if type(source) is not np.ndarray or source.shape != shape:
            frame = " × ".join(map(str, getattr(source, "shape", ())))
            return (f"The background ({' × '.join(map(str, shape))}) does not fit this frame ({frame}), "
                    "which is drawn as it is.")
        try:
            displayed = np.empty(shape, dtype=np.float64)
            np.subtract(source, result.values, out=displayed, casting="unsafe")
            displayed.setflags(write=False)
            followed = weakref.ref(source)
        except (TypeError, ValueError):
            return None
        with self._lock:
            if (self._reservation != reservation or self._phase != "ACTIVE"
                    or len(self._display_values) != 1):
                return None
            self._display_values, self._projection_keys = (displayed,), (frame_key,)
            self._shown_source = followed
        return displayed

    def follow_trace(self, trace):
        """The view's follow (``ScientificView.expect_display_background``): an Int 1-D background (Int 1D's, the 1D
        Viewer's, Stitch 1D's) subtracted from a curve as it is drawn, any curve, also an Overlay's kept curve whose
        frame is gone (his words: "the whole idea is to apply it across frames"; Codex's event-2 HOLD). The curve
        carries what it takes (``TraceProjection.background_basis``): its native axis, that axis's quantity and
        unit, and the channel it was normalized by. The values are its drawn intensity minus the background at its
        axis (``trace_values``). A note (a str) when it does not fit (no basis, another channel, another quantity or
        unit), so it is drawn as it is; None when no 1-D background is active."""
        with self._lock:
            if (self._phase != "ACTIVE" or self._result is None or self._plan is None
                    or self._plan.domain != "integrated_1d"):
                return None
            channel = self._norm_channel
        basis = getattr(trace, "background_basis", None)
        if basis is None:
            return "This curve has no native axis to match the background's, so it is drawn as it is."
        axis, unit_key, drawn_channel = basis
        if drawn_channel != channel:
            def named(value):
                return f"normalized by {value}" if value else "not normalized"
            return (f"The background is {named(channel)}; this curve is {named(drawn_channel)}, so it is drawn as "
                    "it is.")
        return self.trace_values(trace.intensity, axis, unit_key)

    def trace_values(self, intensity, axis, unit_key: str):
        """An Int 1-D background subtracted from a curve drawn now: ``intensity`` is the curve's drawn values (already
        divided by its own monitor when normalized), on ``axis``, its native bins, whose quantity and unit are
        ``unit_key`` ("label\\0unit"). The values are intensity - background, the background taken at the curve's
        axis when its grid differs (NaN outside the background's range): with the background the mean of its
        contributors each divided by its own monitor, I/m - <I_bg/m_bg> (his pick "Each by its own monitor"). A note
        (a str) when the axis is on another quantity or unit, or the values would exceed the grant, so it is drawn as
        it is; None when no 1-D background is active or the curve is not one. The same curve drawn again gets the
        same array."""
        with self._lock:
            self._drop_gone_patterns_locked()
            if (self._phase != "ACTIVE" or self._result is None or self._plan is None
                    or self._plan.domain != "integrated_1d"):
                return None
            if (type(intensity) is not np.ndarray or type(axis) is not np.ndarray or intensity.ndim != 1
                    or axis.shape != intensity.shape or type(unit_key) is not str):
                return None
            result, reservation = self._result, self._reservation
            units = result.axis_units
            if (unit_key,) != units:
                def named(key):
                    label, _, unit = str(key).partition("\0")
                    return f"{label} ({unit})" if unit else label
                return (f"The background is on {named(units[0])}; this curve is on "
                        f"{named(unit_key)}, so it is drawn as it is.")
            entry = self._followed.get(id(intensity))
            if (entry is not None and entry[0]() is intensity and entry[1]() is axis
                    and entry[2] == unit_key):
                self._followed.move_to_end(id(intensity))
                return entry[3]
            base = self._active_bytes - self._followed_bytes
            if base + intensity.size * np.dtype(np.float64).itemsize > self._capacity:
                return "This curve exceeds the display background allowance, so it is drawn as it is."
        try:
            displayed = np.empty(intensity.shape, dtype=np.float64)
            if same_sampled_bins(axis, result.axes[0]):
                np.subtract(intensity, result.values, out=displayed, casting="unsafe")
            else:
                for start in range(0, displayed.size, _BLOCK):
                    stop = min(displayed.size, start + _BLOCK)
                    background = np.interp(axis[start:stop], result.axes[0], result.values,
                                           left=np.nan, right=np.nan)
                    np.subtract(intensity[start:stop], background, out=displayed[start:stop])
            displayed.setflags(write=False)
            gone = self._followed_gone
            held = (weakref.ref(intensity, lambda ref, key=id(intensity): gone.append((key, ref))),
                    weakref.ref(axis), unit_key, displayed)
        except (TypeError, ValueError):
            return None
        with self._lock:
            if self._reservation != reservation or self._phase != "ACTIVE":
                return None
            base = self._active_bytes - self._followed_bytes  # what the background holds besides these values
            prior = self._followed.pop(id(intensity), None)
            if prior is not None:
                self._followed_bytes -= prior[3].nbytes
            while self._followed and base + self._followed_bytes + displayed.nbytes > self._capacity:
                _key, oldest = self._followed.popitem(last=False)
                self._followed_bytes -= oldest[3].nbytes
            if base + self._followed_bytes + displayed.nbytes <= self._capacity:
                self._followed[id(intensity)] = held
                self._followed_bytes += displayed.nbytes
            self._active_bytes = base + self._followed_bytes
        return displayed

    def _drop_gone_patterns_locked(self) -> None:
        while self._followed_gone:
            key, ref = self._followed_gone.popleft()
            entry = self._followed.get(key)
            if entry is not None and entry[0] is ref:
                del self._followed[key]
                self._followed_bytes -= entry[3].nbytes
                self._active_bytes -= entry[3].nbytes

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
        self._keyed_source = None
        self._shown_source = self._shown_grid = None
        self._followed.clear(); self._followed_bytes = 0; self._followed_gone.clear(); self._norm_channel = ""
        self._contributors = (); self._result = None; self._display_values = ()
        self._plan = None; self._stamp = None; self._active_key = None
        self._projection_keys = (); self._projection_indices = (); self._projection_shapes = ()
        self._reservation = None; self._reserved_bytes = 0; self._active_bytes = 0
        self._phase = "RELEASED"


__all__ = ["DisplayBackgroundRendererReleaseReceipt",
           "DisplayBackgroundTransferReceipt", "PresentationBackgroundOwner"]
