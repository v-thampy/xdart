"""Qt-free, bounded six-product viewer model for committed RSM volumes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import InitVar, dataclass, field

import numpy as np

from xrd_tools.analysis.canonical_fingerprint import (
    analysis_canonical_fingerprint,
)
from xrd_tools.session.display_logic import PanelKey, PanelRole


RSM_HK = PanelKey(PanelRole.SLICE_2D, "HK")
RSM_HL = PanelKey(PanelRole.SLICE_2D, "HL")
RSM_KL = PanelKey(PanelRole.SLICE_2D, "KL")
RSM_H = PanelKey(PanelRole.PROJ_1D, "H")
RSM_K = PanelKey(PanelRole.PROJ_1D, "K")
RSM_L = PanelKey(PanelRole.PROJ_1D, "L")

RSM_VIEWER_LAYOUT = (
    (RSM_HK, RSM_HL, RSM_KL),
    (RSM_H, RSM_K, RSM_L),
)
RSM_VIEWER_PANEL_ORDER = tuple(
    panel for row in RSM_VIEWER_LAYOUT for panel in row
)

_MAX_RSM_VIEWER_SNAPSHOTS = 8
_MAX_RSM_VIEWER_COMPONENTS = 32
_MAX_RSM_VIEWER_CACHE_BYTES = 64 * 1024 * 1024
_MAX_RSM_VIEWER_WORK_BYTES = 128 * 1024 * 1024
_RSM_VIEWER_FACTORY = object()
_QNAN_F4_BITS = np.uint32(0x7FC00000)
_F4 = np.dtype("<f4")
_F8 = np.dtype("<f8")
_I8 = np.dtype("<i8")

_ComponentKey = tuple[str, str, str] | tuple[str, str, str, int]
_SnapshotKey = tuple[str, str]


class RSMViewerRefused(ValueError):
    """A viewer state, product, or bounded cache transition was refused."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _viewer_uncopyable(kind: str):
    def copy_value(self):
        raise TypeError(f"{kind} is not copyable")

    def deepcopy_value(self, _memo):
        raise TypeError(f"{kind} is not copyable")

    def reduce_value(self):
        raise TypeError(f"{kind} is not serializable")

    def reduce_ex_value(self, _protocol):
        raise TypeError(f"{kind} is not serializable")

    def replace_value(self, /, **_changes):
        raise TypeError(f"{kind} is not replaceable")

    return (
        copy_value,
        deepcopy_value,
        reduce_value,
        reduce_ex_value,
        replace_value,
    )


def _require_digest(value: object) -> str:
    if type(value) is not str or len(value) != 64:
        raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
    try:
        raw = bytes.fromhex(value)
    except ValueError:
        raise RSMViewerRefused("RSM_VIEW_STATE_INVALID") from None
    if len(raw) != 32 or value != value.lower():
        raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
    return value


def _immutable_bytes(values: np.ndarray) -> bytes:
    root: object = values
    seen: set[int] = set()
    while type(root) is np.ndarray:
        marker = id(root)
        if marker in seen:
            break
        seen.add(marker)
        base = root.base
        if base is None:
            break
        root = base
    while type(root) is memoryview:
        if not root.readonly:
            break
        root = root.obj
    if type(root) is bytes and len(root) == values.nbytes:
        return root
    try:
        return values.tobytes(order="C")
    except MemoryError as error:
        raise RSMViewerRefused("RSM_VIEW_PRODUCT_TOO_LARGE") from error


def _freeze_float32(values: object) -> np.ndarray:
    try:
        contiguous = np.array(values, dtype=_F4, order="C", copy=True)
    except MemoryError as error:
        raise RSMViewerRefused("RSM_VIEW_PRODUCT_TOO_LARGE") from error
    except (TypeError, ValueError, OverflowError) as error:
        raise RSMViewerRefused("RSM_VIEW_STATE_INVALID") from error
    if contiguous.ndim < 1 or contiguous.size < 1 or np.isinf(contiguous).any():
        raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
    contiguous.view(np.uint32)[np.isnan(contiguous)] = _QNAN_F4_BITS
    try:
        immutable = contiguous.tobytes(order="C")
    except MemoryError as error:
        raise RSMViewerRefused("RSM_VIEW_PRODUCT_TOO_LARGE") from error
    return np.frombuffer(immutable, dtype=_F4).reshape(contiguous.shape)


def _validate_canonical_float32(
    values: object,
    *,
    ndim: int,
    finite: bool,
) -> np.ndarray:
    if (
        type(values) is not np.ndarray
        or values.dtype != _F4
        or values.ndim != ndim
        or values.size < 1
        or not values.flags.c_contiguous
        or values.flags.writeable
    ):
        raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
    flat = values.reshape(-1)
    bits = flat.view(np.uint32)
    for start in range(0, flat.size, 65_536):
        stop = min(start + 65_536, flat.size)
        block = flat[start:stop]
        if np.isinf(block).any():
            raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
        nan = np.isnan(block)
        if finite and nan.any():
            raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
        if nan.any() and np.any(bits[start:stop][nan] != _QNAN_F4_BITS):
            raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
    return values


def _validate_source(
    result_fingerprint: object,
    axes: object,
    intensity: object,
) -> tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    fingerprint = _require_digest(result_fingerprint)
    if (
        type(axes) is not tuple
        or len(axes) != 3
        or tuple(
            item[0]
            for item in axes
            if type(item) is tuple and len(item) == 2
        )
        != ("h", "k", "l")
    ):
        raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
    axis_values = tuple(
        _validate_canonical_float32(item[1], ndim=1, finite=True)
        for item in axes
    )
    if any(
        values.size > 1
        and not np.all(np.diff(values.astype(np.float64)) > 0)
        for values in axis_values
    ):
        raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
    volume = _validate_canonical_float32(
        intensity,
        ndim=3,
        finite=False,
    )
    if volume.shape != tuple(values.size for values in axis_values):
        raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
    return fingerprint, axis_values, volume


@dataclass(eq=False, frozen=True, slots=True)
class RSMViewerValues:
    """Factory-issued authority over one strict resident RSM result."""

    result_fingerprint: str
    _axis_values: InitVar[tuple[tuple[str, np.ndarray], ...]]
    _intensity_values: InitVar[np.ndarray]
    _claim: InitVar[object] = None
    _axis_buffers: tuple[tuple[str, bytes], ...] = field(init=False, repr=False)
    _intensity_buffer: bytes = field(init=False, repr=False)
    _shape: tuple[int, int, int] = field(init=False, repr=False)

    def __post_init__(
        self,
        _axis_values: tuple[tuple[str, np.ndarray], ...],
        _intensity_values: np.ndarray,
        _claim: object,
    ) -> None:
        if _claim is not _RSM_VIEWER_FACTORY:
            raise TypeError("RSM viewer values are factory-issued")
        result, axes, intensity = _validate_source(
            self.result_fingerprint,
            _axis_values,
            _intensity_values,
        )
        object.__setattr__(
            self,
            "_axis_buffers",
            tuple(
                (name, _immutable_bytes(values))
                for (name, _), values in zip(
                    _axis_values,
                    axes,
                    strict=True,
                )
            ),
        )
        object.__setattr__(
            self,
            "_intensity_buffer",
            _immutable_bytes(intensity),
        )
        object.__setattr__(self, "result_fingerprint", result)
        object.__setattr__(self, "_shape", intensity.shape)

    @staticmethod
    def _view(buffer: bytes, shape: tuple[int, ...]) -> np.ndarray:
        return np.frombuffer(buffer, dtype=_F4).reshape(shape)

    @property
    def axes(self) -> tuple[tuple[str, np.ndarray], ...]:
        return tuple(
            (name, self._view(buffer, (self._shape[index],)))
            for index, (name, buffer) in enumerate(self._axis_buffers)
        )

    @property
    def intensity(self) -> np.ndarray:
        return self._view(self._intensity_buffer, self._shape)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    (
        __copy__,
        __deepcopy__,
        __reduce__,
        __reduce_ex__,
        __replace__,
    ) = _viewer_uncopyable("RSM viewer values")


def make_rsm_viewer_values(
    source: object,
) -> RSMViewerValues:
    try:
        from xrd_tools.io.analysis_artifact import (
            ANALYSIS_SCHEMA_VERSION_V2,
            AnalysisArtifactKind,
            AnalysisArtifactPayload,
            AnalysisArtifactResultProjection,
        )

        if type(source) is AnalysisArtifactResultProjection:
            if (
                source.kind is not AnalysisArtifactKind.RSM
                or source.axis_units
                != (("h", None), ("k", None), ("l", None))
                or source.sigma is not None
                or source.coverage is not None
                or source.normalization is not None
            ):
                raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
            result_fingerprint = source.result_fingerprint
            axes = source.axes
            intensity = source.intensity
        elif type(source) is AnalysisArtifactPayload:
            inspection = source.inspection
            if (
                inspection.kind is not AnalysisArtifactKind.RSM
                or inspection.schema_version != ANALYSIS_SCHEMA_VERSION_V2
                or inspection.axis_units
                != (("h", None), ("k", None), ("l", None))
                or inspection.has_sigma
                or inspection.has_stitch_diagnostics
            ):
                raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
            result_fingerprint = source.result_fingerprint
            axes = source.axes
            intensity = source.intensity
        else:
            raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
        return RSMViewerValues(
            result_fingerprint,
            axes,
            intensity,
            _RSM_VIEWER_FACTORY,
        )
    except RSMViewerRefused:
        raise
    except MemoryError as error:
        raise RSMViewerRefused("RSM_VIEW_PRODUCT_TOO_LARGE") from error
    except (OverflowError, TypeError, ValueError) as error:
        raise RSMViewerRefused("RSM_VIEW_STATE_INVALID") from error


@dataclass(eq=False, frozen=True, slots=True)
class RSMViewerState:
    result_fingerprint: str
    h_index: int
    k_index: int
    l_index: int
    fingerprint: str = field(init=False)
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _RSM_VIEWER_FACTORY
            or type(self.h_index) is not int
            or type(self.k_index) is not int
            or type(self.l_index) is not int
            or min(self.h_index, self.k_index, self.l_index) < 0
        ):
            raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
        _require_digest(self.result_fingerprint)
        object.__setattr__(
            self,
            "fingerprint",
            analysis_canonical_fingerprint(
                "rsm-viewer-state-v1",
                (
                    self.result_fingerprint,
                    self.h_index,
                    self.k_index,
                    self.l_index,
                ),
            ),
        )

    (
        __copy__,
        __deepcopy__,
        __reduce__,
        __reduce_ex__,
        __replace__,
    ) = _viewer_uncopyable("RSM viewer state")


def make_rsm_viewer_state(
    result_fingerprint: str,
    shape: tuple[int, int, int],
    *,
    h_index: int | None = None,
    k_index: int | None = None,
    l_index: int | None = None,
) -> RSMViewerState:
    fingerprint = _require_digest(result_fingerprint)
    indices = _resolve_rsm_viewer_indices(
        shape,
        h_index=h_index,
        k_index=k_index,
        l_index=l_index,
    )
    return RSMViewerState(
        fingerprint,
        *indices,
        _RSM_VIEWER_FACTORY,
    )


def _resolve_rsm_viewer_indices(
    shape: tuple[int, int, int],
    *,
    h_index: int | None = None,
    k_index: int | None = None,
    l_index: int | None = None,
) -> tuple[int, int, int]:
    if (
        type(shape) is not tuple
        or len(shape) != 3
        or any(type(value) is not int or value < 1 for value in shape)
    ):
        raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
    requested = (h_index, k_index, l_index)
    indices = tuple(
        size // 2 if value is None else value
        for size, value in zip(shape, requested, strict=True)
    )
    if any(
        type(value) is not int or not 0 <= value < size
        for value, size in zip(indices, shape, strict=True)
    ):
        raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
    return indices


@dataclass(eq=False, frozen=True, slots=True)
class RSMViewerProduct:
    panel_key: PanelKey
    x_axis: np.ndarray
    y_axis_or_none: np.ndarray | None
    values: np.ndarray
    source_indices: tuple[int | None, int | None, int | None]
    _finite_count: int = field(init=False, repr=False)
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _RSM_VIEWER_FACTORY
            or type(self.panel_key) is not PanelKey
            or not any(self.panel_key is key for key in RSM_VIEWER_PANEL_ORDER)
            or type(self.x_axis) is not np.ndarray
            or self.x_axis.dtype != _F4
            or self.x_axis.ndim != 1
            or not self.x_axis.flags.c_contiguous
            or self.x_axis.flags.writeable
            or type(self.values) is not np.ndarray
            or self.values.dtype != _F4
            or not self.values.flags.c_contiguous
            or self.values.flags.writeable
            or type(self.source_indices) is not tuple
            or len(self.source_indices) != 3
            or any(
                value is not None and (type(value) is not int or value < 0)
                for value in self.source_indices
            )
        ):
            raise TypeError("RSM viewer product is invalid")
        if self.panel_key.role is PanelRole.SLICE_2D:
            if (
                type(self.y_axis_or_none) is not np.ndarray
                or self.y_axis_or_none.dtype != _F4
                or self.y_axis_or_none.ndim != 1
                or not self.y_axis_or_none.flags.c_contiguous
                or self.y_axis_or_none.flags.writeable
                or self.values.ndim != 2
                or self.values.shape
                != (self.y_axis_or_none.size, self.x_axis.size)
                or sum(value is not None for value in self.source_indices) != 1
            ):
                raise TypeError("RSM viewer slice product is invalid")
        elif (
            self.panel_key.role is not PanelRole.PROJ_1D
            or self.y_axis_or_none is not None
            or self.values.ndim != 1
            or self.values.shape != self.x_axis.shape
            or self.source_indices != (None, None, None)
        ):
            raise TypeError("RSM viewer projection product is invalid")
        object.__setattr__(self, "_finite_count", _finite_count(self.values))

    (
        __copy__,
        __deepcopy__,
        __reduce__,
        __reduce_ex__,
        __replace__,
    ) = _viewer_uncopyable("RSM viewer product")


def _snapshot_unique_arrays(
    products: tuple[RSMViewerProduct, ...],
) -> dict[int, np.ndarray]:
    unique: dict[int, np.ndarray] = {}
    for product in products:
        for values in (
            product.x_axis,
            product.y_axis_or_none,
            product.values,
        ):
            if values is not None:
                unique.setdefault(id(values), values)
    return unique


def _snapshot_bytes(products: tuple[RSMViewerProduct, ...]) -> int:
    return sum(values.nbytes for values in _snapshot_unique_arrays(products).values())


def _array_projection(values: np.ndarray) -> tuple[object, ...]:
    return (values.dtype.str, values.shape, values.tobytes(order="C"))


def _product_projection(product: RSMViewerProduct) -> tuple[object, ...]:
    return (
        product.panel_key.role.value,
        product.panel_key.instance,
        product.source_indices,
        _array_projection(product.x_axis),
        (
            None
            if product.y_axis_or_none is None
            else _array_projection(product.y_axis_or_none)
        ),
        _array_projection(product.values),
    )


def _snapshot_fingerprint(
    state: RSMViewerState,
    products: tuple[RSMViewerProduct, ...],
    finite_counts: tuple[int, ...],
    cache_bytes: int,
) -> str:
    return analysis_canonical_fingerprint(
        "rsm-viewer-snapshot-v1",
        (
            (
                state.result_fingerprint,
                state.h_index,
                state.k_index,
                state.l_index,
                state.fingerprint,
            ),
            tuple(_product_projection(product) for product in products),
            finite_counts,
            cache_bytes,
        ),
    )


@dataclass(eq=False, frozen=True, slots=True)
class RSMViewerSnapshot:
    state: RSMViewerState
    products: tuple[RSMViewerProduct, ...]
    finite_counts: tuple[int, ...]
    cache_bytes: int
    fingerprint: str = field(init=False)
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _RSM_VIEWER_FACTORY
            or type(self.state) is not RSMViewerState
            or type(self.products) is not tuple
            or len(self.products) != len(RSM_VIEWER_PANEL_ORDER)
            or any(
                type(product) is not RSMViewerProduct
                or product.panel_key is not key
                for product, key in zip(
                    self.products,
                    RSM_VIEWER_PANEL_ORDER,
                    strict=True,
                )
            )
            or type(self.finite_counts) is not tuple
            or len(self.finite_counts) != len(self.products)
            or any(type(value) is not int or value < 0 for value in self.finite_counts)
            or type(self.cache_bytes) is not int
            or self.cache_bytes != _snapshot_bytes(self.products)
        ):
            raise TypeError("RSM viewer snapshot is invalid")
        object.__setattr__(
            self,
            "fingerprint",
            _snapshot_fingerprint(
                self.state,
                self.products,
                self.finite_counts,
                self.cache_bytes,
            ),
        )

    def product(self, key: PanelKey) -> RSMViewerProduct:
        if type(key) is not PanelKey:
            raise KeyError(key)
        for product in self.products:
            if product.panel_key is key:
                return product
        raise KeyError(key)

    (
        __copy__,
        __deepcopy__,
        __reduce__,
        __reduce_ex__,
        __replace__,
    ) = _viewer_uncopyable("RSM viewer snapshot")


def _finite_count(values: np.ndarray) -> int:
    return int(np.count_nonzero(np.isfinite(values)))


def _projection_work_bytes(
    intensity: np.ndarray,
    output_size: int,
) -> tuple[int, int, int, int, int, int]:
    input_bytes = intensity.nbytes
    mask_bytes = intensity.size * np.dtype(bool).itemsize
    zero_bytes = intensity.size * _F4.itemsize
    sum_bytes = output_size * _F8.itemsize
    count_bytes = output_size * _I8.itemsize
    output_bytes = output_size * _F4.itemsize
    return (
        input_bytes,
        mask_bytes,
        zero_bytes,
        sum_bytes,
        count_bytes,
        output_bytes,
    )


def _snapshot_shape_bytes(shape: tuple[int, int, int]) -> int:
    h_size, k_size, l_size = shape
    return _F4.itemsize * (
        h_size * k_size
        + h_size * l_size
        + k_size * l_size
        + 2 * (h_size + k_size + l_size)
    )


def _require_work_bytes(total: int) -> None:
    if total > _MAX_RSM_VIEWER_WORK_BYTES:
        raise RSMViewerRefused("RSM_VIEW_PRODUCT_TOO_LARGE")


def _viewer_finite_mask(intensity: np.ndarray) -> np.ndarray:
    return np.isfinite(intensity)


def _projection_values(
    intensity: np.ndarray,
    *,
    retained_axis: int,
) -> np.ndarray:
    output_size = intensity.shape[retained_axis]
    charges = _projection_work_bytes(intensity, output_size)
    _require_work_bytes(sum(charges))
    finite = _viewer_finite_mask(intensity)
    zeroed = np.where(finite, intensity, np.float32(0.0))
    reduction_axes = tuple(axis for axis in range(3) if axis != retained_axis)
    sums = np.sum(zeroed, axis=reduction_axes, dtype=np.float64)
    counts = np.sum(finite, axis=reduction_axes, dtype=np.int64)
    projected = np.empty((output_size,), dtype=_F4)
    for index in range(output_size):
        if counts[index] == 0:
            projected.view(np.uint32)[index] = _QNAN_F4_BITS
        else:
            projected[index] = sums[index] / counts[index]
    if np.isinf(projected).any():
        raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
    del finite, zeroed, sums, counts
    return _freeze_float32(projected)


def _build_rsm_viewer_product(
    panel_key: PanelKey,
    state: RSMViewerState,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    intensity: np.ndarray,
) -> RSMViewerProduct:
    def product(
        key: PanelKey,
        x_axis: np.ndarray,
        y_axis_or_none: np.ndarray | None,
        product_values: np.ndarray,
        source_indices: tuple[int | None, int | None, int | None],
    ) -> RSMViewerProduct:
        return RSMViewerProduct(
            key,
            x_axis,
            y_axis_or_none,
            product_values,
            source_indices,
            _RSM_VIEWER_FACTORY,
        )

    h_axis, k_axis, l_axis = axes
    if panel_key is RSM_HK:
        return product(
            panel_key,
            h_axis,
            k_axis,
            _freeze_float32(intensity[:, :, state.l_index].T),
            (None, None, state.l_index),
        )
    if panel_key is RSM_HL:
        return product(
            panel_key,
            h_axis,
            l_axis,
            _freeze_float32(intensity[:, state.k_index, :].T),
            (None, state.k_index, None),
        )
    if panel_key is RSM_KL:
        return product(
            panel_key,
            k_axis,
            l_axis,
            _freeze_float32(intensity[state.h_index, :, :].T),
            (state.h_index, None, None),
        )
    if panel_key is RSM_H:
        values = _projection_values(intensity, retained_axis=0)
        return product(
            panel_key,
            h_axis,
            None,
            values,
            (None, None, None),
        )
    if panel_key is RSM_K:
        values = _projection_values(intensity, retained_axis=1)
        return product(
            panel_key,
            k_axis,
            None,
            values,
            (None, None, None),
        )
    if panel_key is RSM_L:
        values = _projection_values(intensity, retained_axis=2)
        return product(
            panel_key,
            l_axis,
            None,
            values,
            (None, None, None),
        )
    raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")


def _component_keys(state: RSMViewerState) -> tuple[_ComponentKey, ...]:
    result = state.result_fingerprint
    return (
        (result, "slice", "l", state.l_index),
        (result, "slice", "k", state.k_index),
        (result, "slice", "h", state.h_index),
        (result, "projection", "h"),
        (result, "projection", "k"),
        (result, "projection", "l"),
    )


def _resident_arrays(
    components: OrderedDict[_ComponentKey, RSMViewerProduct],
    snapshots: OrderedDict[_SnapshotKey, RSMViewerSnapshot],
) -> dict[int, np.ndarray]:
    products = list(components.values())
    for snapshot in snapshots.values():
        products.extend(snapshot.products)
    return _snapshot_unique_arrays(tuple(products))


def _resident_bytes(
    components: OrderedDict[_ComponentKey, RSMViewerProduct],
    snapshots: OrderedDict[_SnapshotKey, RSMViewerSnapshot],
) -> int:
    return sum(
        values.nbytes
        for values in _resident_arrays(components, snapshots).values()
    )


def _snapshot_references(
    snapshot: RSMViewerSnapshot,
    product: RSMViewerProduct,
) -> bool:
    return any(value is product for value in snapshot.products)


class RSMViewerModel:
    """Bounded coupled component/snapshot LRU for one committed RSM result."""

    __slots__ = (
        "_components",
        "_snapshots",
        "_current",
        "_result_fingerprint",
        "_values",
    )

    def __init__(self) -> None:
        self._components: OrderedDict[
            _ComponentKey, RSMViewerProduct
        ] = OrderedDict()
        self._snapshots: OrderedDict[
            _SnapshotKey, RSMViewerSnapshot
        ] = OrderedDict()
        self._current: RSMViewerSnapshot | None = None
        self._result_fingerprint: str | None = None
        self._values: RSMViewerValues | None = None

    def __copy__(self):
        raise TypeError("RSM viewer model is not copyable")

    def __deepcopy__(self, _memo):
        raise TypeError("RSM viewer model is not copyable")

    @property
    def current_snapshot(self) -> RSMViewerSnapshot | None:
        return self._current

    @property
    def component_count(self) -> int:
        return len(self._components)

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    @property
    def resident_bytes(self) -> int:
        return _resident_bytes(self._components, self._snapshots)

    @property
    def component_keys(self) -> tuple[_ComponentKey, ...]:
        return tuple(self._components)

    @property
    def snapshot_keys(self) -> tuple[_SnapshotKey, ...]:
        return tuple(self._snapshots)

    def _axis_values(
        self,
        result_fingerprint: str,
        source_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values: list[np.ndarray] = []
        for name, source in zip(("h", "k", "l"), source_axes, strict=True):
            key: _ComponentKey = (
                result_fingerprint,
                "projection",
                name,
            )
            cached = self._components.get(key)
            values.append(
                cached.x_axis if cached is not None else _freeze_float32(source)
            )
        return values[0], values[1], values[2]

    def _plan_insert(
        self,
        candidate: RSMViewerSnapshot,
        component_keys: tuple[_ComponentKey, ...],
    ) -> tuple[
        OrderedDict[_ComponentKey, RSMViewerProduct],
        OrderedDict[_SnapshotKey, RSMViewerSnapshot],
    ]:
        current = self._current
        if candidate.cache_bytes > _MAX_RSM_VIEWER_CACHE_BYTES:
            raise RSMViewerRefused("RSM_VIEW_PRODUCT_TOO_LARGE")
        if current is not None:
            side_by_side = _snapshot_unique_arrays(
                (*current.products, *candidate.products)
            )
            if (
                sum(values.nbytes for values in side_by_side.values())
                > _MAX_RSM_VIEWER_CACHE_BYTES
            ):
                raise RSMViewerRefused("RSM_VIEW_PRODUCT_TOO_LARGE")

        same_result = self._result_fingerprint == candidate.state.result_fingerprint
        components = (
            OrderedDict(self._components) if same_result else OrderedDict()
        )
        snapshots = (
            OrderedDict(self._snapshots) if same_result else OrderedDict()
        )
        for key, product in zip(
            component_keys,
            candidate.products,
            strict=True,
        ):
            components[key] = product
            components.move_to_end(key)
        snapshot_key = (
            candidate.state.result_fingerprint,
            candidate.state.fingerprint,
        )
        snapshots[snapshot_key] = candidate
        snapshots.move_to_end(snapshot_key)

        while len(snapshots) > _MAX_RSM_VIEWER_SNAPSHOTS:
            removed = False
            for key, snapshot in tuple(snapshots.items()):
                if snapshot is not candidate:
                    del snapshots[key]
                    removed = True
                    break
            if not removed:
                raise RSMViewerRefused("RSM_VIEW_PRODUCT_TOO_LARGE")

        while (
            len(components) > _MAX_RSM_VIEWER_COMPONENTS
            or _resident_bytes(components, snapshots)
            > _MAX_RSM_VIEWER_CACHE_BYTES
        ):
            removed = False
            candidate_products = {id(product) for product in candidate.products}
            for component_key, product in tuple(components.items()):
                if id(product) in candidate_products:
                    continue
                for snapshot_key, snapshot in tuple(snapshots.items()):
                    if _snapshot_references(snapshot, product):
                        del snapshots[snapshot_key]
                del components[component_key]
                removed = True
                break
            if not removed:
                raise RSMViewerRefused("RSM_VIEW_PRODUCT_TOO_LARGE")

        component_ids = {id(product) for product in components.values()}
        if any(
            id(product) not in component_ids
            for snapshot in snapshots.values()
            for product in snapshot.products
        ):
            raise RuntimeError("RSM viewer cache plan orphaned a snapshot product")
        return components, snapshots

    def snapshot(
        self,
        values: RSMViewerValues | None = None,
        *,
        h_index: int | None = None,
        k_index: int | None = None,
        l_index: int | None = None,
    ) -> RSMViewerSnapshot:
        if values is None:
            values = self._values
        if type(values) is not RSMViewerValues:
            raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
        result = values.result_fingerprint
        if (
            self._result_fingerprint == result
            and self._values is not values
        ):
            raise RSMViewerRefused("RSM_VIEW_STATE_INVALID")
        try:
            result = _require_digest(result)
            indices = _resolve_rsm_viewer_indices(
                values.shape,
                h_index=h_index,
                k_index=k_index,
                l_index=l_index,
            )
            cached_key: _SnapshotKey | None = None
            cached_snapshot: RSMViewerSnapshot | None = None
            if self._result_fingerprint == result:
                for key, snapshot in self._snapshots.items():
                    cached_state = snapshot.state
                    if (
                        cached_state.result_fingerprint == result
                        and cached_state.h_index == indices[0]
                        and cached_state.k_index == indices[1]
                        and cached_state.l_index == indices[2]
                    ):
                        cached_key = key
                        cached_snapshot = snapshot
                        break
            if cached_key is not None and cached_snapshot is not None:
                component_pairs = tuple(zip(
                    _component_keys(cached_snapshot.state),
                    cached_snapshot.products,
                    strict=True,
                ))
                for component_key, product in component_pairs:
                    if self._components.get(component_key) is not product:
                        raise RuntimeError("RSM viewer cache snapshot is orphaned")
                self._snapshots.move_to_end(cached_key)
                for component_key, _product in component_pairs:
                    self._components.move_to_end(component_key)
                self._current = cached_snapshot
                self._result_fingerprint = result
                self._values = values
                return cached_snapshot

            state = RSMViewerState(
                result,
                *indices,
                _RSM_VIEWER_FACTORY,
            )
            required_bytes = _snapshot_shape_bytes(values.shape)
            if required_bytes > _MAX_RSM_VIEWER_CACHE_BYTES:
                raise RSMViewerRefused("RSM_VIEW_PRODUCT_TOO_LARGE")
            if (
                self._current is not None
                and self._result_fingerprint != result
                and self._current.cache_bytes + required_bytes
                > _MAX_RSM_VIEWER_CACHE_BYTES
            ):
                raise RSMViewerRefused("RSM_VIEW_PRODUCT_TOO_LARGE")
            source_axes = tuple(item[1] for item in values.axes)
            source_intensity = values.intensity
            for output_size in values.shape:
                _require_work_bytes(
                    sum(_projection_work_bytes(source_intensity, output_size))
                )
            component_keys = _component_keys(state)
            axis_values = self._axis_values(result, source_axes)
            staged_products: list[RSMViewerProduct] = []
            for component_key, panel_key in zip(
                component_keys,
                RSM_VIEWER_PANEL_ORDER,
                strict=True,
            ):
                cached_product = self._components.get(component_key)
                staged_products.append(
                    cached_product
                    if cached_product is not None
                    else _build_rsm_viewer_product(
                        panel_key,
                        state,
                        axis_values,
                        source_intensity,
                    )
                )
            products = tuple(staged_products)
            finite_counts = tuple(product._finite_count for product in products)
            cache_bytes = _snapshot_bytes(products)
            candidate = RSMViewerSnapshot(
                state,
                products,
                finite_counts,
                cache_bytes,
                _RSM_VIEWER_FACTORY,
            )
            planned_components, planned_snapshots = self._plan_insert(
                candidate,
                component_keys,
            )
        except RSMViewerRefused:
            raise
        except MemoryError as error:
            raise RSMViewerRefused("RSM_VIEW_PRODUCT_TOO_LARGE") from error
        except (OverflowError, TypeError, ValueError) as error:
            raise RSMViewerRefused("RSM_VIEW_STATE_INVALID") from error

        self._components = planned_components
        self._snapshots = planned_snapshots
        self._current = candidate
        self._result_fingerprint = result
        self._values = values
        return candidate


__all__ = [
    "RSM_H",
    "RSM_HK",
    "RSM_HL",
    "RSM_K",
    "RSM_KL",
    "RSM_L",
    "RSM_VIEWER_LAYOUT",
    "RSM_VIEWER_PANEL_ORDER",
    "RSMViewerModel",
    "RSMViewerProduct",
    "RSMViewerRefused",
    "RSMViewerSnapshot",
    "RSMViewerState",
    "RSMViewerValues",
    "make_rsm_viewer_state",
    "make_rsm_viewer_values",
]
