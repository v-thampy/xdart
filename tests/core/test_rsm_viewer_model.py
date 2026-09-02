from __future__ import annotations

import builtins
import copy
import itertools
import os
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.analysis.canonical_fingerprint import (
    analysis_canonical_fingerprint,
)
from xrd_tools.io.analysis_artifact import (
    ANALYSIS_SCHEMA_VERSION_V2,
    AnalysisArtifactKind,
    AnalysisArtifactInspection,
    AnalysisArtifactPayload,
    project_analysis_artifact_result,
)
import xrd_tools.io.analysis_artifact as artifact_module
import xrd_tools.session.rsm_viewer_model as module
from xrd_tools.session.display_logic import PanelKey, PanelRole
from xrd_tools.session.rsm_viewer_model import (
    RSM_H,
    RSM_HK,
    RSM_HL,
    RSM_K,
    RSM_KL,
    RSM_L,
    RSM_VIEWER_LAYOUT,
    RSM_VIEWER_PANEL_ORDER,
    RSMViewerModel,
    RSMViewerRefused,
    RSMViewerValues,
    make_rsm_viewer_state,
    make_rsm_viewer_values,
)


def _projection(
    shape: tuple[int, int, int] = (3, 4, 5),
    *,
    intensity: np.ndarray | None = None,
    offset: float = 0.0,
):
    axes = tuple(
        (
            name,
            np.linspace(
                offset + index,
                offset + index + 1.0,
                size,
                dtype=np.float64,
            ),
        )
        for index, (name, size) in enumerate(
            zip(("h", "k", "l"), shape, strict=True)
        )
    )
    if intensity is None:
        intensity = (
            np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
            + offset
        )
    return project_analysis_artifact_result(
        kind=AnalysisArtifactKind.RSM,
        axes=axes,
        axis_units=(("h", None), ("k", None), ("l", None)),
        intensity=intensity,
        sigma=None,
        coverage=None,
        normalization=None,
    )


def _values(
    shape: tuple[int, int, int] = (3, 4, 5),
    *,
    intensity: np.ndarray | None = None,
    offset: float = 0.0,
) -> RSMViewerValues:
    return make_rsm_viewer_values(
        _projection(shape, intensity=intensity, offset=offset)
    )


def _cache_facts(model: RSMViewerModel) -> tuple[object, ...]:
    return (
        model.current_snapshot,
        model.component_keys,
        model.snapshot_keys,
        model.component_count,
        model.snapshot_count,
        model.resident_bytes,
        tuple(id(value) for value in model._components.values()),
        tuple(id(value) for value in model._snapshots.values()),
        model._values,
    )


def _independent_resident_bytes(model: RSMViewerModel) -> int:
    arrays: dict[int, np.ndarray] = {}
    products = list(model._components.values())
    for snapshot in model._snapshots.values():
        products.extend(snapshot.products)
    for product in products:
        for values in (
            product.x_axis,
            product.y_axis_or_none,
            product.values,
        ):
            if values is not None:
                arrays.setdefault(id(values), values)
    return sum(values.nbytes for values in arrays.values())


def _array_projection(values: np.ndarray) -> tuple[object, ...]:
    return (values.dtype.str, values.shape, values.tobytes(order="C"))


def _snapshot_oracle(snapshot) -> str:
    state = snapshot.state
    products = tuple(
        (
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
        for product in snapshot.products
    )
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
            products,
            snapshot.finite_counts,
            snapshot.cache_bytes,
        ),
    )


def _finite_projection(values: np.ndarray, retained_axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    reduction_axes = tuple(axis for axis in range(3) if axis != retained_axis)
    sums = np.sum(
        np.where(finite, values, np.float32(0.0)),
        axis=reduction_axes,
        dtype=np.float64,
    )
    counts = np.sum(finite, axis=reduction_axes, dtype=np.int64)
    result = np.empty((values.shape[retained_axis],), dtype="<f4")
    for index in range(result.size):
        if counts[index] == 0:
            result.view("<u4")[index] = np.uint32(0x7FC00000)
        else:
            result[index] = sums[index] / counts[index]
    return result


def test_exact_layout_has_six_repeated_role_keys_in_frozen_order():
    assert module._MAX_RSM_VIEWER_SNAPSHOTS == 8
    assert module._MAX_RSM_VIEWER_COMPONENTS == 32
    assert module._MAX_RSM_VIEWER_CACHE_BYTES == 64 << 20
    assert module._MAX_RSM_VIEWER_WORK_BYTES == 128 << 20
    assert RSM_VIEWER_LAYOUT == (
        (RSM_HK, RSM_HL, RSM_KL),
        (RSM_H, RSM_K, RSM_L),
    )
    assert RSM_VIEWER_PANEL_ORDER == (
        RSM_HK,
        RSM_HL,
        RSM_KL,
        RSM_H,
        RSM_K,
        RSM_L,
    )
    assert tuple((key.role, key.instance) for key in RSM_VIEWER_PANEL_ORDER) == (
        (PanelRole.SLICE_2D, "HK"),
        (PanelRole.SLICE_2D, "HL"),
        (PanelRole.SLICE_2D, "KL"),
        (PanelRole.PROJ_1D, "H"),
        (PanelRole.PROJ_1D, "K"),
        (PanelRole.PROJ_1D, "L"),
    )
    assert len(set(RSM_VIEWER_PANEL_ORDER)) == 6


def test_values_require_factory_owned_rsm_projection_and_reissue_is_distinct():
    projection = _projection()
    first = make_rsm_viewer_values(projection)
    second = make_rsm_viewer_values(projection)
    assert first is not second
    assert first.result_fingerprint == projection.result_fingerprint
    with pytest.raises(TypeError, match="factory-issued"):
        RSMViewerValues(
            projection.result_fingerprint,
            projection.axes,
            projection.intensity,
        )
    with pytest.raises(RSMViewerRefused, match="RSM_VIEW_STATE_INVALID"):
        make_rsm_viewer_values(object())

    stitch = project_analysis_artifact_result(
        kind=AnalysisArtifactKind.STITCH_1D,
        axes=(("q", np.arange(3, dtype=np.float64)),),
        axis_units=(("q", "q_A^-1"),),
        intensity=np.ones(3),
        sigma=None,
        coverage=None,
        normalization=None,
    )
    with pytest.raises(RSMViewerRefused, match="RSM_VIEW_STATE_INVALID"):
        make_rsm_viewer_values(stitch)

    model = RSMViewerModel()
    snapshot = model.snapshot(first)
    facts = _cache_facts(model)
    with pytest.raises(RSMViewerRefused, match="RSM_VIEW_STATE_INVALID"):
        model.snapshot(second)
    assert _cache_facts(model) == facts
    assert model.current_snapshot is snapshot


def test_values_own_immutable_bytes_and_return_fresh_metadata_views():
    values = _values()
    first_axes = values.axes
    second_axes = values.axes
    first_intensity = values.intensity
    second_intensity = values.intensity
    assert first_axes[0][1] is not second_axes[0][1]
    assert first_intensity is not second_intensity
    assert first_intensity.dtype == np.dtype("<f4")
    assert first_intensity.flags.c_contiguous
    assert not first_intensity.flags.writeable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        first_intensity.shape = (first_intensity.size,)
        first_axes[0][1].shape = (first_axes[0][1].size, 1)
    assert values.intensity.shape == values.shape
    assert values.axes[0][1].shape == (values.shape[0],)
    with pytest.raises(ValueError, match="WRITEABLE|writable"):
        values.intensity.flags.writeable = True
    for operation in (
        lambda: copy.copy(values),
        lambda: copy.deepcopy(values),
        lambda: pickle.dumps(values),
    ):
        with pytest.raises(TypeError):
            operation()


def test_factory_accepts_exact_resident_rsm_v2_payload_projection():
    projection = _projection((2, 3, 4))
    inspection = AnalysisArtifactInspection(
        path="/resident/not-read",
        storage_revision=(1, 2, 3, 4, 5),
        kind=AnalysisArtifactKind.RSM,
        group="rsm",
        shape=projection.intensity.shape,
        axes=tuple((name, values.size) for name, values in projection.axes),
        axis_units=(("h", None), ("k", None), ("l", None)),
        has_sigma=False,
        has_stitch_diagnostics=False,
        request_fingerprint="1" * 64,
        source_fingerprint="2" * 64,
        plan_fingerprint="3" * 64,
        provenance_digest="4" * 64,
        provenance_json="{}",
        provenance_sha256="5" * 64,
        result_fingerprint=projection.result_fingerprint,
        schema_version=ANALYSIS_SCHEMA_VERSION_V2,
        execution_attestation_digest="6" * 64,
        execution_attestation_json="{}",
    )
    payload = AnalysisArtifactPayload(
        inspection,
        projection.axes,
        projection.intensity,
        None,
        None,
        None,
        artifact_module._ANALYSIS_PAYLOAD_FACTORY,
    )
    values = make_rsm_viewer_values(payload)
    assert values.result_fingerprint == payload.result_fingerprint
    assert values.shape == projection.intensity.shape
    np.testing.assert_array_equal(values.intensity, projection.intensity)


def test_default_state_and_each_exact_bound_refuse_without_cache_touch():
    values = _values((3, 4, 5))
    model = RSMViewerModel()
    snapshot = model.snapshot(values)
    assert (
        snapshot.state.h_index,
        snapshot.state.k_index,
        snapshot.state.l_index,
    ) == (1, 2, 2)
    baseline = _cache_facts(model)
    invalid = (
        {"h_index": -1},
        {"h_index": 3},
        {"k_index": 4},
        {"l_index": 5},
        {"h_index": True},
        {"k_index": np.int64(1)},
        {"l_index": 1.0},
    )
    for keywords in invalid:
        with pytest.raises(RSMViewerRefused, match="RSM_VIEW_STATE_INVALID"):
            model.snapshot(**keywords)
        assert _cache_facts(model) == baseline


def test_state_fingerprint_domain_and_every_field_participates():
    result = "a" * 64
    baseline = make_rsm_viewer_state(result, (3, 4, 5))
    assert baseline.fingerprint == analysis_canonical_fingerprint(
        "rsm-viewer-state-v1",
        (result, 1, 2, 2),
    )
    variants = {
        make_rsm_viewer_state("b" * 64, (3, 4, 5)).fingerprint,
        make_rsm_viewer_state(result, (3, 4, 5), h_index=0).fingerprint,
        make_rsm_viewer_state(result, (3, 4, 5), k_index=0).fingerprint,
        make_rsm_viewer_state(result, (3, 4, 5), l_index=0).fingerprint,
    }
    assert baseline.fingerprint not in variants
    assert len(variants) == 4
    assert make_rsm_viewer_state(result, (3, 4, 5)).fingerprint == baseline.fingerprint


def test_three_slice_orientation_projection_and_canonical_empty_nan_contracts(
    monkeypatch,
):
    shape = (3, 4, 5)
    intensity = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    intensity[0, :, :] = np.nan
    intensity[1, 2, 3] = np.nan
    values = _values(shape, intensity=intensity)
    source = values.intensity
    monkeypatch.setattr(
        np,
        "nanmean",
        lambda *_args, **_kwargs: pytest.fail("nanmean must not be called"),
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        snapshot = RSMViewerModel().snapshot(values, h_index=1, k_index=2, l_index=3)
    assert not captured
    hk, hl, kl, h, k, l = snapshot.products
    np.testing.assert_array_equal(hk.values, source[:, :, 3].T)
    np.testing.assert_array_equal(hl.values, source[:, 2, :].T)
    np.testing.assert_array_equal(kl.values, source[1, :, :].T)
    np.testing.assert_array_equal(h.values, _finite_projection(source, 0))
    np.testing.assert_array_equal(k.values, _finite_projection(source, 1))
    np.testing.assert_array_equal(l.values, _finite_projection(source, 2))
    assert tuple(product.values.shape for product in snapshot.products) == (
        (4, 3),
        (5, 3),
        (5, 4),
        (3,),
        (4,),
        (5,),
    )
    assert tuple(product.source_indices for product in snapshot.products) == (
        (None, None, 3),
        (None, 2, None),
        (1, None, None),
        (None, None, None),
        (None, None, None),
        (None, None, None),
    )
    assert h.values.view("<u4")[0] == np.uint32(0x7FC00000)
    assert snapshot.finite_counts == tuple(
        int(np.count_nonzero(np.isfinite(product.values)))
        for product in snapshot.products
    )


def test_products_are_detached_readonly_c_order_f4_and_share_interned_axes():
    values = _values((3, 4, 5))
    source_axes = tuple(item[1] for item in values.axes)
    source_intensity = values.intensity
    snapshot = RSMViewerModel().snapshot(values)
    hk, hl, kl, h, k, l = snapshot.products
    assert hk.x_axis is hl.x_axis is h.x_axis
    assert hk.y_axis_or_none is kl.x_axis is k.x_axis
    assert hl.y_axis_or_none is kl.y_axis_or_none is l.x_axis
    np.testing.assert_array_equal(hk.x_axis, source_axes[0])
    np.testing.assert_array_equal(hl.x_axis, source_axes[0])
    np.testing.assert_array_equal(h.x_axis, source_axes[0])
    np.testing.assert_array_equal(hk.y_axis_or_none, source_axes[1])
    np.testing.assert_array_equal(kl.x_axis, source_axes[1])
    np.testing.assert_array_equal(k.x_axis, source_axes[1])
    np.testing.assert_array_equal(hl.y_axis_or_none, source_axes[2])
    np.testing.assert_array_equal(kl.y_axis_or_none, source_axes[2])
    np.testing.assert_array_equal(l.x_axis, source_axes[2])
    for product in snapshot.products:
        for array in (product.x_axis, product.y_axis_or_none, product.values):
            if array is None:
                continue
            assert type(array) is np.ndarray
            assert array.dtype == np.dtype("<f4")
            assert array.flags.c_contiguous
            assert not array.flags.writeable
            with pytest.raises(ValueError, match="WRITEABLE|writable"):
                array.flags.writeable = True
        assert not np.shares_memory(product.values, source_intensity)
    for source_axis in source_axes:
        assert all(
            not np.shares_memory(source_axis, product_axis)
            for product in snapshot.products
            for product_axis in (product.x_axis, product.y_axis_or_none)
            if product_axis is not None
        )
    h_size, k_size, l_size = values.shape
    assert snapshot.cache_bytes == 4 * (
        h_size * k_size
        + h_size * l_size
        + k_size * l_size
        + 2 * (h_size + k_size + l_size)
    )


def test_snapshot_fingerprint_binds_exact_science_counts_state_and_cache_bytes():
    snapshot = RSMViewerModel().snapshot(_values())
    assert snapshot.fingerprint == _snapshot_oracle(snapshot)
    assert len(snapshot.products) == len(snapshot.finite_counts) == 6
    assert snapshot.product(RSM_HK) is snapshot.products[0]
    with pytest.raises(KeyError):
        snapshot.product(PanelKey(PanelRole.SLICE_2D, "HK"))
    for operation in (
        lambda: copy.copy(snapshot),
        lambda: copy.deepcopy(snapshot),
        lambda: pickle.dumps(snapshot),
    ):
        with pytest.raises(TypeError):
            operation()


def test_snapshot_hit_reads_no_source_and_performs_no_calculation(monkeypatch):
    values = _values()
    model = RSMViewerModel()
    snapshot = model.snapshot(values)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cache hit performed forbidden work")

    monkeypatch.setattr(module, "_build_rsm_viewer_product", forbidden)
    monkeypatch.setattr(module, "_freeze_float32", forbidden)
    monkeypatch.setattr(module, "_snapshot_fingerprint", forbidden)
    monkeypatch.setattr(module, "_finite_count", forbidden)
    monkeypatch.setattr(RSMViewerValues, "axes", property(forbidden))
    monkeypatch.setattr(RSMViewerValues, "intensity", property(forbidden))
    assert model.snapshot() is snapshot


@pytest.mark.parametrize(
    "keyword,panel_index,panel_key",
    (
        ("l_index", 0, RSM_HK),
        ("k_index", 1, RSM_HL),
        ("h_index", 2, RSM_KL),
    ),
)
def test_each_single_axis_change_builds_and_counts_only_its_normal_slice(
    monkeypatch,
    keyword,
    panel_index,
    panel_key,
):
    values = _values((4, 4, 4))
    model = RSMViewerModel()
    baseline = model.snapshot(values)
    built: list[PanelKey] = []
    counted: list[int] = []
    real_build = module._build_rsm_viewer_product
    real_count = module._finite_count

    def build(key, *args, **kwargs):
        built.append(key)
        return real_build(key, *args, **kwargs)

    def count(array):
        counted.append(array.size)
        return real_count(array)

    monkeypatch.setattr(module, "_build_rsm_viewer_product", build)
    monkeypatch.setattr(module, "_finite_count", count)
    changed = model.snapshot(**{keyword: 0})
    assert built == [panel_key]
    assert len(counted) == 1
    assert changed.products[panel_index] is not baseline.products[panel_index]
    assert all(
        changed.products[index] is baseline.products[index]
        for index in range(6)
        if index != panel_index
    )


def test_unique_byte_ledger_counts_same_object_once_and_equal_copy_twice():
    shape = (3, 3, 3)
    values = _values(shape, intensity=np.ones(shape))
    model = RSMViewerModel()
    first = model.snapshot(values, l_index=0)
    second = model.snapshot(l_index=1)
    assert np.array_equal(first.products[0].values, second.products[0].values)
    assert first.products[0].values is not second.products[0].values
    assert model.resident_bytes == _independent_resident_bytes(model)
    assert model.resident_bytes == (
        first.cache_bytes + second.products[0].values.nbytes
    )


def test_byte_pressure_removes_dependent_snapshots_before_exact_components(
    monkeypatch,
):
    values = _values((4, 4, 4))
    model = RSMViewerModel()
    snapshots = tuple(
        model.snapshot(
            values if l_index == 0 else None,
            l_index=l_index,
        )
        for l_index in range(3)
    )
    hk_bytes = snapshots[-1].products[0].values.nbytes
    cap = snapshots[-1].cache_bytes + hk_bytes
    monkeypatch.setattr(module, "_MAX_RSM_VIEWER_CACHE_BYTES", cap)
    fourth = model.snapshot(l_index=3)
    result = values.result_fingerprint
    assert (result, "slice", "l", 0) not in model.component_keys
    assert (result, "slice", "l", 1) not in model.component_keys
    assert (result, "slice", "l", 2) in model.component_keys
    assert (result, "slice", "l", 3) in model.component_keys
    assert (result, snapshots[0].state.fingerprint) not in model.snapshot_keys
    assert (result, snapshots[1].state.fingerprint) not in model.snapshot_keys
    assert (result, snapshots[2].state.fingerprint) in model.snapshot_keys
    assert (result, fourth.state.fingerprint) in model.snapshot_keys
    assert model.resident_bytes == _independent_resident_bytes(model) == cap


def test_snapshot_hit_touches_its_snapshot_and_six_components_in_frozen_order():
    values = _values((3, 3, 4))
    model = RSMViewerModel()
    snapshots = tuple(
        model.snapshot(
            values if l_index == 0 else None,
            l_index=l_index,
        )
        for l_index in range(3)
    )
    first = snapshots[0]
    first_key = (values.result_fingerprint, first.state.fingerprint)
    assert model.snapshot(l_index=0) is first
    assert model.snapshot_keys[-1] == first_key
    assert model.component_keys[-6:] == module._component_keys(first.state)


def test_ninth_snapshot_and_thirty_third_component_follow_exact_lru_cascade():
    values = _values((11, 11, 11))
    model = RSMViewerModel()
    snapshots = []
    states = list(itertools.product(range(3), repeat=3))[:10]
    for h_index, k_index, l_index in states[:9]:
        snapshots.append(
            model.snapshot(
                values if not snapshots else None,
                h_index=h_index,
                k_index=k_index,
                l_index=l_index,
            )
        )
    first_key = (
        values.result_fingerprint,
        snapshots[0].state.fingerprint,
    )
    assert model.snapshot_count == 8
    assert first_key not in model.snapshot_keys
    assert model.resident_bytes == _independent_resident_bytes(model)

    component_model = RSMViewerModel()
    component_snapshots = []
    for index in range(10):
        component_snapshots.append(
            component_model.snapshot(
                values if index == 0 else None,
                h_index=index,
                k_index=index,
                l_index=index,
            )
        )
    assert component_model.component_count == 32
    assert (
        values.result_fingerprint,
        "slice",
        "l",
        0,
    ) not in component_model.component_keys
    assert component_model.resident_bytes == _independent_resident_bytes(
        component_model
    )


def test_snapshot_and_work_caps_refuse_before_first_product_scratch(monkeypatch):
    values = _values((3, 3, 3))
    required = module._snapshot_shape_bytes(values.shape)
    monkeypatch.setattr(module, "_MAX_RSM_VIEWER_CACHE_BYTES", required)
    exact = RSMViewerModel()
    assert exact.snapshot(values).cache_bytes == required

    under = RSMViewerModel()
    monkeypatch.setattr(module, "_MAX_RSM_VIEWER_CACHE_BYTES", required - 1)
    monkeypatch.setattr(
        module,
        "_freeze_float32",
        lambda *_args, **_kwargs: pytest.fail("oversize allocated a product"),
    )
    with pytest.raises(RSMViewerRefused, match="RSM_VIEW_PRODUCT_TOO_LARGE"):
        under.snapshot(values)
    assert _cache_facts(under)[0:6] == (None, (), (), 0, 0, 0)

    monkeypatch.undo()
    total = sum(module._projection_work_bytes(values.intensity, 3))
    monkeypatch.setattr(module, "_MAX_RSM_VIEWER_WORK_BYTES", total - 1)
    monkeypatch.setattr(
        module,
        "_viewer_finite_mask",
        lambda *_args, **_kwargs: pytest.fail("over-cap allocated finite mask"),
    )
    work_model = RSMViewerModel()
    with pytest.raises(RSMViewerRefused, match="RSM_VIEW_PRODUCT_TOO_LARGE"):
        work_model.snapshot(values)
    assert work_model.current_snapshot is None


def test_projection_work_charge_is_exact_and_exact_boundary_proceeds(monkeypatch):
    values = _values((3, 3, 3))
    intensity = values.intensity
    charges = module._projection_work_bytes(intensity, 3)
    assert charges == (
        4 * 27,
        27,
        4 * 27,
        8 * 3,
        8 * 3,
        4 * 3,
    )
    assert sum(charges) == 9 * 27 + 20 * 3
    monkeypatch.setattr(module, "_MAX_RSM_VIEWER_WORK_BYTES", sum(charges))
    assert RSMViewerModel().snapshot(values).products[3].values.shape == (3,)


def test_projection_uses_float64_sum_before_float32_projection():
    intensity = np.array([[[1.0e8, 1.0, -1.0e8]]], dtype=np.float64)
    snapshot = RSMViewerModel().snapshot(
        _values((1, 1, 3), intensity=intensity)
    )
    expected = np.float32(1.0 / 3.0)
    assert snapshot.product(RSM_H).values[0] == expected


def test_failure_and_new_result_side_by_side_refusal_preserve_exact_cache_graph(
    monkeypatch,
):
    first_values = _values((4, 4, 4), offset=0.0)
    second_values = _values((4, 4, 4), offset=100.0)
    model = RSMViewerModel()
    baseline = model.snapshot(first_values)
    facts = _cache_facts(model)
    real_build = module._build_rsm_viewer_product

    def fail_hk(key, *args, **kwargs):
        if key is RSM_HK:
            raise MemoryError("injected")
        return real_build(key, *args, **kwargs)

    monkeypatch.setattr(module, "_build_rsm_viewer_product", fail_hk)
    with pytest.raises(RSMViewerRefused, match="RSM_VIEW_PRODUCT_TOO_LARGE"):
        model.snapshot(l_index=0)
    assert _cache_facts(model) == facts
    assert model.current_snapshot is baseline

    monkeypatch.setattr(module, "_build_rsm_viewer_product", real_build)
    monkeypatch.setattr(
        module,
        "_MAX_RSM_VIEWER_CACHE_BYTES",
        baseline.cache_bytes,
    )
    monkeypatch.setattr(
        module,
        "_freeze_float32",
        lambda *_args, **_kwargs: pytest.fail("side-by-side refusal allocated"),
    )
    with pytest.raises(RSMViewerRefused, match="RSM_VIEW_PRODUCT_TOO_LARGE"):
        model.snapshot(second_values)
    assert _cache_facts(model) == facts


@pytest.mark.parametrize(
    "target",
    ("fingerprint", "planning"),
)
def test_fingerprint_and_eviction_plan_failures_preserve_exact_cache_graph(
    monkeypatch,
    target,
):
    values = _values((4, 4, 4))
    model = RSMViewerModel()
    baseline = model.snapshot(values)
    facts = _cache_facts(model)
    attribute = (
        "_snapshot_fingerprint" if target == "fingerprint" else "_resident_bytes"
    )
    original = getattr(module, attribute)

    def injected(*_args, **_kwargs):
        raise RuntimeError(f"injected {target} failure")

    monkeypatch.setattr(module, attribute, injected)
    with pytest.raises(RuntimeError, match=f"injected {target} failure"):
        model.snapshot(l_index=0)
    monkeypatch.setattr(module, attribute, original)
    assert _cache_facts(model) == facts
    assert model.current_snapshot is baseline


def test_successful_result_change_fully_invalidates_and_a_b_a_recomputes():
    first_values = _values((3, 3, 3), offset=0.0)
    second_values = _values((3, 3, 3), offset=100.0)
    model = RSMViewerModel()
    first = model.snapshot(first_values)
    first_product_ids = tuple(id(product) for product in first.products)
    second = model.snapshot(second_values)
    assert all(
        key[0] == second_values.result_fingerprint
        for key in (*model.component_keys, *model.snapshot_keys)
    )
    assert not any(
        product is old
        for product in second.products
        for old in first.products
    )
    rebuilt = model.snapshot(first_values)
    assert rebuilt is not first
    assert not any(id(product) in first_product_ids for product in rebuilt.products)
    assert all(
        key[0] == first_values.result_fingerprint
        for key in (*model.component_keys, *model.snapshot_keys)
    )


def test_all_index_cache_and_invalidation_paths_call_no_disk_or_science(
    monkeypatch,
):
    first_values = _values((3, 3, 3), offset=0.0)
    second_values = _values((3, 3, 3), offset=10.0)

    def guarded(real):
        def call(*args, **kwargs):
            frame = sys._getframe(1)
            while frame is not None:
                if frame.f_globals.get("__name__") == module.__name__:
                    raise AssertionError("viewer called disk or science")
                frame = frame.f_back
            return real(*args, **kwargs)

        return call

    monkeypatch.setattr(builtins, "open", guarded(builtins.open))
    monkeypatch.setattr(os, "stat", guarded(os.stat))
    h5py = sys.modules.get("h5py")
    if h5py is not None:
        monkeypatch.setattr(h5py, "File", guarded(h5py.File))
    artifact_module = sys.modules["xrd_tools.io.analysis_artifact"]
    for name in ("inspect_analysis_artifact", "read_analysis_artifact"):
        monkeypatch.setattr(
            artifact_module,
            name,
            lambda *_args, **_kwargs: pytest.fail(
                "viewer replayed analysis artifact I/O"
            ),
        )
    science_module = sys.modules.get("xrd_tools.analysis.rsm_operation")
    if science_module is not None:
        for name in (
            "prepare_rsm_operation",
            "prepare_rsm_operation_v2",
            "run_rsm_operation",
            "run_rsm_operation_v2",
        ):
            monkeypatch.setattr(
                science_module,
                name,
                lambda *_args, **_kwargs: pytest.fail(
                    "viewer replayed RSM science"
                ),
            )
    model = RSMViewerModel()
    first = model.snapshot(first_values)
    assert model.snapshot() is first
    model.snapshot(l_index=0)
    model.snapshot(h_index=0)
    model.snapshot(k_index=0)
    model.snapshot(second_values)


def test_singleton_axes_preserve_exact_shapes_copies_and_bounds():
    values = _values((1, 1, 1), intensity=np.array([[[7.0]]]))
    snapshot = RSMViewerModel().snapshot(values)
    assert tuple(product.values.shape for product in snapshot.products) == (
        (1, 1),
        (1, 1),
        (1, 1),
        (1,),
        (1,),
        (1,),
    )
    assert snapshot.finite_counts == (1, 1, 1, 1, 1, 1)
    assert all(np.all(product.values == np.float32(7.0)) for product in snapshot.products)
    with pytest.raises(RSMViewerRefused, match="RSM_VIEW_STATE_INVALID"):
        RSMViewerModel().snapshot(values, h_index=1)


def test_fresh_import_is_qt_hdf5_xdart_and_rsm_science_free():
    script = r'''
import sys
import xrd_tools.session.rsm_viewer_model
forbidden = (
    "PyQt5", "PyQt6", "PySide2", "PySide6", "pyqtgraph", "xdart", "h5py",
    "xrayutilities", "xrd_tools.analysis.rsm_operation",
    "xrd_tools.io.analysis_artifact",
)
loaded = sorted(
    name for name in sys.modules
    if any(name == token or name.startswith(token + ".") for token in forbidden)
)
assert not loaded, loaded
'''
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"),
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
