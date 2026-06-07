from __future__ import annotations

import numpy as np
import pytest

from ssrl_xrd_tools.core.containers import IntegrationResult1D
from ssrl_xrd_tools.reduction import (
    Integration1DPlan,
    MemorySink,
    ReductionPlan,
    XYESink,
    run_reduction,
)
from ssrl_xrd_tools.sources import MemoryFrameSource
import ssrl_xrd_tools.reduction.core as reduction_core


def _r1d(value: float) -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([value, value + 1.0]),
        sigma=np.array([0.1, 0.2]),
        unit="q_A^-1",
    )


def test_run_reduction_accepts_frame_source(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        reduction_core,
        "integrate_1d",
        lambda image, ai, **kwargs: _r1d(float(np.sum(image))),
    )
    source = MemoryFrameSource([np.ones((2, 2)), np.full((2, 2), 2.0)])
    source.integrator = object()

    result = run_reduction(
        ReductionPlan(integration_1d=Integration1DPlan(npt=2)),
        source,
    )

    assert result.n_processed == 2
    assert np.allclose(result.frames[0].result_1d.intensity, [4.0, 5.0])
    assert np.allclose(result.frames[1].result_1d.intensity, [8.0, 9.0])


def test_run_reduction_fans_out_to_memory_and_xye(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        reduction_core,
        "integrate_1d",
        lambda image, ai, **kwargs: _r1d(float(np.sum(image))),
    )
    source = MemoryFrameSource([np.ones((2, 2))], name="fanout")
    memory = MemorySink()
    xye = XYESink(tmp_path)

    result = run_reduction(
        ReductionPlan(integration_1d=Integration1DPlan(npt=2)),
        source.to_scan(integrator=object()),
        sink=[memory, xye],
    )

    assert result.n_processed == 1
    assert 0 in memory.frames
    out = tmp_path / "fanout_0000.xye"
    assert out.exists()
    saved = np.loadtxt(out)
    assert np.allclose(saved[:, 1], [4.0, 5.0])


def test_run_reduction_executor_and_freeze_policy_are_explicit_todos(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        reduction_core,
        "integrate_1d",
        lambda image, ai, **kwargs: _r1d(float(np.sum(image))),
    )
    source = MemoryFrameSource([np.ones((2, 2))])

    with pytest.warns(RuntimeWarning, match="RESTRUCTURE-TODO\\(WS-C\\)"):
        run_reduction(
            ReductionPlan(integration_1d=Integration1DPlan(npt=2)),
            source.to_scan(integrator=object()),
            executor=object(),
            gi_freeze_mode="scout_union",
        )
