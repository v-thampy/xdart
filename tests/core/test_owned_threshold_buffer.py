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


def test_owned_thresholds_preserve_background_and_saturation_interactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.full((100, 100), 10, dtype=np.uint16)
    raw[:2, :] = np.iinfo(np.uint16).max
    # This value must be rejected before background subtraction: subtracting
    # first would turn 101 into the otherwise-admitted value 99.
    raw[2, 0] = 101
    raw[3, 0] = 0
    raw_before = raw.copy()
    captured: dict[str, np.ndarray | None] = {}

    def capture(image, ai, **kwargs):
        captured["image"] = np.array(image, copy=True)
        mask = kwargs["mask"]
        captured["mask"] = None if mask is None else np.array(mask, copy=True)
        return _r1d(float(np.nansum(image)))

    monkeypatch.setattr(reduction_core, "integrate_1d", capture)
    result = run_reduction(
        ReductionPlan(
            threshold_min=5.0,
            threshold_max=100.0,
            mask_saturation=True,
        ),
        Scan(
            "interactions",
            [Frame(0, image=raw, background=2.0)],
            integrator=object(),
        ),
    )

    assert result.n_processed == 1
    image = captured["image"]
    mask = captured["mask"]
    assert image is not None and mask is not None
    assert image[4, 4] == 8.0
    assert np.isnan(image[:2, :]).all()
    assert np.isnan(image[2, 0])
    assert np.isnan(image[3, 0])
    assert mask[:2, :].all()
    assert int(mask.sum()) == 200
    assert not mask[2, 0] and not mask[3, 0]
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
