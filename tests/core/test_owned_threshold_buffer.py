"""Focused contracts for the reduction worker's owned threshold buffer."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from xrd_tools.reduction import (
    Frame,
    GI1DMode,
    GI2DMode,
    GIMode,
    Integration1DPlan,
    Integration2DPlan,
    ReductionPlan,
    Scan,
    run_reduction,
)
import xrd_tools.reduction.core as reduction_core


def _r1d(value: float) -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([value, value + 1.0]),
        sigma=None,
        unit="q_A^-1",
    )


def _r2d(value: float) -> IntegrationResult2D:
    return IntegrationResult2D(
        radial=np.array([0.0, 1.0]),
        azimuthal=np.array([-1.0, 1.0]),
        intensity=np.full((2, 2), value),
        sigma=None,
        unit="q_A^-1",
    )


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    (
        (
            None,
            None,
            ((-np.inf, -2.0, -1.0, -0.0, 0.0),
             (1.0, 2.0, np.inf, np.nan, 0.5)),
        ),
        (
            -1.0,
            None,
            ((np.nan, np.nan, -1.0, -0.0, 0.0),
             (1.0, 2.0, np.inf, np.nan, 0.5)),
        ),
        (
            None,
            1.0,
            ((-np.inf, -2.0, -1.0, -0.0, 0.0),
             (1.0, np.nan, np.nan, np.nan, 0.5)),
        ),
        (
            -1.0,
            1.0,
            ((np.nan, np.nan, -1.0, -0.0, 0.0),
             (1.0, np.nan, np.nan, np.nan, 0.5)),
        ),
        (
            1.0,
            -1.0,
            ((np.nan, np.nan, np.nan, np.nan, np.nan),
             (np.nan, np.nan, np.nan, np.nan, np.nan)),
        ),
        (
            0.0,
            0.0,
            ((np.nan, np.nan, np.nan, -0.0, 0.0),
             (np.nan, np.nan, np.nan, np.nan, np.nan)),
        ),
    ),
)
def test_threshold_helpers_match_literal_numeric_edge_oracle(
    lower: float | None,
    upper: float | None,
    expected: tuple[tuple[float, ...], ...],
) -> None:
    source = np.array(
        [
            [-np.inf, -2.0, -1.0, -0.0, 0.0],
            [1.0, 2.0, np.inf, np.nan, 0.5],
        ],
        dtype=np.float64,
    )
    source_before = source.copy()
    plan = ReductionPlan(threshold_min=lower, threshold_max=upper)

    expected_array = np.array(expected, dtype=np.float64)
    copied = reduction_core._apply_thresholds(source, plan)
    owned = np.array(source, dtype=float, copy=True)
    actual = reduction_core._apply_thresholds_owned(owned, plan)

    assert actual is owned
    assert np.array_equal(actual, expected_array, equal_nan=True)
    assert np.array_equal(copied, expected_array, equal_nan=True)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected_array))
    np.testing.assert_array_equal(np.signbit(copied), np.signbit(expected_array))
    assert np.array_equal(source, source_before, equal_nan=True)
    if lower is None and upper is None:
        assert copied is source
    else:
        assert copied is not source


def test_owned_thresholds_reject_borrowed_readonly_or_non_float64() -> None:
    plan = ReductionPlan(threshold_min=0.0)
    owner = np.arange(9.0)
    borrowed = owner.reshape(3, 3)
    readonly = np.arange(9.0).reshape(3, 3).copy()
    readonly.setflags(write=False)
    float32 = np.arange(9, dtype=np.float32).reshape(3, 3).copy()

    for invalid in (borrowed, readonly, float32):
        with pytest.raises(ValueError, match="writable owning float64"):
            reduction_core._apply_thresholds_owned(invalid, plan)


@pytest.mark.parametrize(
    ("raw_dtype", "raw_readonly"),
    (
        (np.uint16, False),
        (np.float64, False),
        (np.float64, True),
    ),
)
def test_reduce_frame_routes_fresh_owned_buffer_without_mutating_raw(
    monkeypatch: pytest.MonkeyPatch,
    raw_dtype: type[np.generic],
    raw_readonly: bool,
) -> None:
    # The float64 case is the important alias sentinel: ``astype(float)`` must
    # still allocate a fresh buffer when the source already has worker dtype.
    raw = np.array([[0, 5], [10, 20]], dtype=raw_dtype)
    raw_before = raw.copy()
    if raw_readonly:
        raw.setflags(write=False)
    seen: dict[str, object] = {}
    original_owned = reduction_core._apply_thresholds_owned

    def observe_owned(image, plan):
        seen.update(
            dtype=image.dtype,
            owning=image.flags.owndata,
            writeable=image.flags.writeable,
            shares_raw=np.shares_memory(image, raw),
        )
        return original_owned(image, plan)

    def reject_copy_helper(*_args, **_kwargs):
        raise AssertionError("worker threshold path restored the redundant copy")

    monkeypatch.setattr(reduction_core, "_apply_thresholds_owned", observe_owned)
    monkeypatch.setattr(reduction_core, "_apply_thresholds", reject_copy_helper)
    monkeypatch.setattr(
        reduction_core,
        "integrate_1d",
        lambda image, ai, **kwargs: _r1d(float(np.nansum(image))),
    )

    result = run_reduction(
        ReductionPlan(threshold_min=1.0, threshold_max=10.0),
        Scan("owned", [Frame(0, image=raw)], integrator=object()),
    )

    assert result.n_processed == 1
    assert seen == {
        "dtype": np.dtype(float),
        "owning": True,
        "writeable": True,
        "shares_raw": False,
    }
    np.testing.assert_array_equal(raw, raw_before)


@pytest.mark.parametrize("gi", (False, True), ids=("standard", "gi"))
def test_owned_thresholds_preserve_1d_2d_science_input_hashes(
    monkeypatch: pytest.MonkeyPatch,
    gi: bool,
) -> None:
    raw = np.array(
        [[0, 1, 2, 3], [4, 5, 100, 7], [8, 9, 10, 11]],
        dtype=np.uint16,
    )
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(npt=2),
        integration_2d=Integration2DPlan(npt_rad=2, npt_azim=2),
        gi=(
            GIMode(
                incident_angle=0.1,
                mode_1d=GI1DMode.Q_TOTAL,
                mode_2d=GI2DMode.Q_CHI,
            )
            if gi else None
        ),
        threshold_min=1.0,
        threshold_max=10.0,
    )
    expected = np.array(
        [
            [np.nan, 1.0, 2.0, 3.0],
            [4.0, 5.0, np.nan, 7.0],
            [8.0, 9.0, 10.0, np.nan],
        ],
        dtype=np.float64,
    )
    expected_hash = hashlib.sha256(
        np.ascontiguousarray(expected).tobytes()
    ).hexdigest()
    observed: list[str] = []

    def capture_1d(image, *_args, **_kwargs):
        observed.append(hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())
        return _r1d(float(np.nansum(image)))

    def capture_2d(image, *_args, **_kwargs):
        observed.append(hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())
        return _r2d(float(np.nanmean(image)))

    if gi:
        monkeypatch.setattr(
            reduction_core,
            "poni_to_fiber_integrator",
            lambda *args, **kwargs: object(),
        )
        monkeypatch.setattr(reduction_core, "integrate_gi_polar_1d", capture_1d)
        monkeypatch.setattr(reduction_core, "integrate_gi_polar", capture_2d)
        scan = Scan("gi", [Frame(0, image=raw)], poni=object())
    else:
        monkeypatch.setattr(reduction_core, "integrate_1d", capture_1d)
        monkeypatch.setattr(reduction_core, "integrate_2d", capture_2d)
        scan = Scan("standard", [Frame(0, image=raw)], integrator=object())

    result = run_reduction(plan, scan)

    assert result.n_processed == 1
    assert observed == [expected_hash, expected_hash]
    assert result.frames[0].result_1d is not None
    assert result.frames[0].result_2d is not None


def _csr_float32_plan(**kwargs) -> ReductionPlan:
    one = Integration1DPlan(
        npt=2, method="csr", radial_range=(0.1, 1.0),
        azimuth_range=(-30.0, 30.0), monitor_key="i0",
        error_model="poisson", polarization_factor=0.9,
        extra={"correctSolidAngle": False, "dummy": -1.0,
               "delta_dummy": 0.0, "safe": True},
    )
    two = Integration2DPlan(
        npt_rad=2, npt_azim=2, method="csr", error_model="azimuthal",
        extra={"correctSolidAngle": False, "safe": True},
    )
    return ReductionPlan(integration_1d=one, integration_2d=two, **kwargs)


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.uint32))
def test_unsigned_csr_reuses_one_owned_float32_buffer(monkeypatch, ai_fixture, dtype):
    raw = np.arange(10_000, dtype=np.uint32).reshape(100, 100).astype(dtype)
    raw[0, 0] = np.iinfo(dtype).max
    before, seen = raw.copy(), []

    def capture_1d(image, _ai, **_kwargs):
        seen.append(image)
        return _r1d(1)

    def capture_2d(image, _ai, **_kwargs):
        seen.append(image)
        return _r2d(1)

    monkeypatch.setattr(reduction_core, "integrate_1d", capture_1d)
    monkeypatch.setattr(reduction_core, "integrate_2d", capture_2d)
    result = run_reduction(
        _csr_float32_plan(mask_saturation=True),
        Scan("f32", [Frame(0, image=raw, metadata={"i0": 2.0})],
             integrator=ai_fixture),
    )

    assert result.n_processed == 1
    assert len(seen) == 2 and seen[0] is seen[1]
    image = seen[0]
    assert image.dtype == np.dtype(np.float32)
    assert image.flags.owndata and image.flags.c_contiguous and image.flags.writeable
    assert not np.shares_memory(image, raw)
    np.testing.assert_array_equal(raw, before)


def test_float32_csr_eligibility_fails_closed():
    raw, base = np.arange(20, dtype=np.uint16).reshape(4, 5), _csr_float32_plan()
    cases = (
        (raw.astype(np.float32), None, base),
        (raw.astype(np.int16), None, base),
        (raw.astype(">u2"), None, base),
        (np.asfortranarray(raw), None, base),
        (raw, 0.0, base),
        (raw, None, _csr_float32_plan(threshold_min=0.0)),
        (raw, None, _csr_float32_plan(gi=GIMode(incident_angle=0.1))),
        (raw, None, ReductionPlan(
            integration_1d=Integration1DPlan(unit="chi_deg"), integration_2d=None)),
        (raw, None, ReductionPlan(
            integration_1d=Integration1DPlan(method="cython"), integration_2d=None)),
        (raw, None, ReductionPlan(
            integration_1d=Integration1DPlan(error_model="variance"),
            integration_2d=None)),
    )
    for image, background, plan in cases:
        assert not reduction_core._can_use_owned_float32_csr(
            image, background, plan, integrator_input_safe=True)


@pytest.mark.parametrize("extra", (
    {"variance": np.ones((2, 2))}, {"dark": np.ones((2, 2))},
    {"flat": np.ones((2, 2))}, {"absorption": np.ones((2, 2))},
    {"unknown": 1}, {"safe": 1}, {"dummy": np.inf}, {"dummy": 0.1},
))
def test_float32_csr_refuses_unproven_integration_extras(extra):
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(extra=extra), integration_2d=None)
    assert not reduction_core._can_use_owned_float32_csr(
        np.ones((2, 2), dtype=np.uint16), None, plan,
        integrator_input_safe=True)


def test_custom_integrator_keeps_prior_float64_input(monkeypatch):
    seen = []
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, _ai, **_kwargs: seen.append(image) or _r1d(1))
    scan = Scan("custom-ai", [Frame(0, image=np.ones((2, 2), np.uint16))],
                integrator=object())
    run_reduction(ReductionPlan(), scan)
    assert len(seen) == 1 and seen[0].dtype == np.dtype(float)


def test_float32_guard_requires_stock_ai_and_detector_dummy(monkeypatch, ai_fixture):
    assert reduction_core._stock_pyfai_float32_input_semantics(ai_fixture)
    assert not reduction_core._stock_pyfai_float32_input_semantics(
        type("CustomAI", (type(ai_fixture),), {})())
    monkeypatch.setattr(type(ai_fixture.detector), "get_dummies",
                        lambda *_args: (-1.0, 0.0))
    assert not reduction_core._stock_pyfai_float32_input_semantics(ai_fixture)


def _digest(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@pytest.mark.slow
def test_real_pyfai_float32_csr_matches_prior_float64_science(
    ai_fixture, synthetic_image,
):
    raw = np.clip(synthetic_image, 0, np.iinfo(np.uint32).max).astype(np.uint32)
    raw[0, :2] = (np.iinfo(np.uint32).max - 1, np.iinfo(np.uint32).max)
    before = raw.copy()
    mask = np.zeros(raw.shape, dtype=bool)
    mask[10, 10] = True
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(npt=128, method="csr"),
        integration_2d=Integration2DPlan(
            npt_rad=64, npt_azim=72, method="csr"), mask=mask)
    prior = raw.astype(float)
    expected_1d = reduction_core.integrate_1d(
        prior, ai_fixture, npt=128, method="csr", mask=mask)
    expected_2d = reduction_core.integrate_2d(
        prior, ai_fixture, npt_rad=64, npt_azim=72, method="csr", mask=mask)
    actual = run_reduction(
        plan, Scan("real-f32", [Frame(0, image=raw)], integrator=ai_fixture),
    ).frames[0]

    for observed, expected in ((actual.result_1d, expected_1d),
                               (actual.result_2d, expected_2d)):
        assert observed is not None
        for name in ("radial", "intensity", "sigma"):
            left, right = getattr(observed, name), getattr(expected, name)
            assert (left is None) == (right is None)
            if left is not None:
                assert _digest(left) == _digest(right)
    assert _digest(actual.result_2d.azimuthal) == _digest(expected_2d.azimuthal)
    np.testing.assert_array_equal(raw, before)
