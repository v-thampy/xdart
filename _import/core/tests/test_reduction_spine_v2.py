from __future__ import annotations

import h5py
import numpy as np
import pytest

from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.core.scan import FrameGeometry
from ssrl_xrd_tools.reduction import (
    Frame,
    GI1DMode,
    GI2DMode,
    GIMode,
    Integration1DPlan,
    Integration2DPlan,
    MemorySink,
    NexusSink,
    ReductionPlan,
    Scan,
    XYESink,
    run_reduction,
)
from ssrl_xrd_tools.io.frame_view import read_frame_view
from ssrl_xrd_tools.io.nexus import (
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
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


def _r2d(value: float) -> IntegrationResult2D:
    return IntegrationResult2D(
        radial=np.array([0.0, 1.0]),
        azimuthal=np.array([-1.0, 1.0]),
        intensity=np.full((2, 2), value, dtype=float),
        sigma=None,
        unit="qip_A^-1",
        azimuthal_unit="qoop_A^-1",
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


def test_nexus_sink_preserves_non_numeric_scan_metadata(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        reduction_core,
        "integrate_1d",
        lambda image, ai, **kwargs: _r1d(float(np.sum(image))),
    )
    path = tmp_path / "mixed_metadata.nxs"
    scan = Scan(
        "mixed",
        [
            Frame(
                5,
                image=np.ones((2, 2)),
                metadata={"phase": "alpha", "operator": "sam", "temperature": 295.5},
            ),
            Frame(
                8,
                image=np.full((2, 2), 2.0),
                metadata={"phase": "beta", "operator": "lee", "temperature": 296.0},
            ),
        ],
        integrator=object(),
    )

    run_reduction(
        ReductionPlan(integration_1d=Integration1DPlan(npt=2)),
        scan,
        NexusSink(path, overwrite=True),
    )

    with h5py.File(path, "r") as h5:
        entry = h5["entry"]
        assert entry.attrs["ssrl_schema"] == PROCESSED_SCHEMA_NAME
        assert entry.attrs["ssrl_schema_version"] == PROCESSED_SCHEMA_VERSION
        phase = entry["scan_data/phase"]
        assert h5py.check_string_dtype(phase.dtype) is not None
        assert phase.attrs["ssrl_dtype"] == "string"
        assert list(phase.asstr()[()]) == ["alpha", "beta"]
        assert entry["scan_data/temperature"].attrs["ssrl_dtype"] == "float32"

    view = read_frame_view(path, frame=8)
    assert view.metadata_raw["phase"] == "beta"
    assert view.metadata_raw["operator"] == "lee"
    assert view.metadata_numeric["temperature"] == pytest.approx(296.0)


def test_run_reduction_gi_resolves_incidence_and_dispatches_modes(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, dict]] = []
    fake_fi = object()

    monkeypatch.setattr(
        reduction_core,
        "poni_to_fiber_integrator",
        lambda poni, **kwargs: calls.append(("fiber", kwargs)) or fake_fi,
    )

    def fake_gi_1d(image, fi, **kwargs):
        calls.append(("gi1d", kwargs))
        assert fi is fake_fi
        return _r1d(float(np.sum(image)))

    def fake_gi_2d(image, fi, **kwargs):
        calls.append(("gi2d", kwargs))
        assert fi is fake_fi
        return _r2d(float(np.sum(image)))

    monkeypatch.setattr(reduction_core, "integrate_gi_1d", fake_gi_1d)
    monkeypatch.setattr(reduction_core, "integrate_gi_2d", fake_gi_2d)

    scan = Scan(
        "gi",
        [
            Frame(0, image=np.ones((2, 2)), metadata={"Th": 0.15, "I0": 10.0}),
            Frame(1, image=np.full((2, 2), 2.0), metadata={"th": 0.20, "I0": 20.0}),
        ],
        poni=object(),
    )
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(npt=3, monitor_key="i0"),
        integration_2d=Integration2DPlan(
            npt_rad=4,
            npt_azim=5,
            unit="qip_A^-1",
            monitor_key="I0",
            extra={"x_range": (-1.0, 1.0), "y_range": (0.0, 2.0)},
        ),
        gi=GIMode(
            incidence_motor="TH",
            mode_1d="q_oop",
            mode_2d="qip_qoop",
            npt_oop=7,
        ),
    )

    result = run_reduction(plan, scan)

    assert result.n_processed == 2
    assert calls[0] == (
        "fiber",
        {"incident_angle": 0.15, "tilt_angle": 0.0, "sample_orientation": 1},
    )
    first_1d = calls[1][1]
    assert first_1d["unit"] == "qoop_A^-1"
    assert first_1d["vertical_integration"] is True
    assert first_1d["npt_oop"] == 7
    assert first_1d["incident_angle"] == 0.15
    assert first_1d["normalization_factor"] == 10.0
    first_2d = calls[2][1]
    assert first_2d["unit"] == "qip_A^-1"
    assert first_2d["radial_range"] == (-1.0, 1.0)
    assert first_2d["azimuth_range"] == (0.0, 2.0)
    assert first_2d["incident_angle"] == 0.15
    assert first_2d["normalization_factor"] == 10.0
    assert calls[3][1]["incident_angle"] == 0.20


def test_run_reduction_gi_dispatches_polar_and_exit_angle_modes(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    monkeypatch.setattr(reduction_core, "poni_to_fiber_integrator", lambda *a, **k: object())
    monkeypatch.setattr(
        reduction_core,
        "integrate_gi_polar_1d",
        lambda image, fi, **kwargs: calls.append("polar_1d") or _r1d(1.0),
    )
    monkeypatch.setattr(
        reduction_core,
        "integrate_gi_exitangles_1d",
        lambda image, fi, **kwargs: calls.append("exit_1d") or _r1d(2.0),
    )
    monkeypatch.setattr(
        reduction_core,
        "integrate_gi_polar",
        lambda image, fi, **kwargs: calls.append("polar_2d") or _r2d(3.0),
    )
    monkeypatch.setattr(
        reduction_core,
        "integrate_gi_exitangles",
        lambda image, fi, **kwargs: calls.append("exit_2d") or _r2d(4.0),
    )

    scan = Scan(
        "gi",
        [Frame(0, image=np.ones((2, 2)), geometry=FrameGeometry(incident_angle=0.3))],
        poni=object(),
    )
    run_reduction(
        ReductionPlan(
            integration_1d=Integration1DPlan(),
            integration_2d=Integration2DPlan(),
            gi=GIMode(mode_1d=GI1DMode.Q_TOTAL, mode_2d=GI2DMode.Q_CHI),
        ),
        scan,
    )
    run_reduction(
        ReductionPlan(
            integration_1d=Integration1DPlan(),
            integration_2d=Integration2DPlan(),
            gi=GIMode(mode_1d=GI1DMode.EXIT_ANGLE, mode_2d=GI2DMode.EXIT_ANGLES),
        ),
        scan,
    )

    assert calls == ["polar_1d", "polar_2d", "exit_1d", "exit_2d"]


def test_gi_incident_angle_must_be_resolvable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(reduction_core, "poni_to_fiber_integrator", lambda *a, **k: object())
    scan = Scan("gi", [Frame(0, image=np.ones((2, 2)))], poni=object())
    with pytest.raises(ValueError, match="incident_angle"):
        run_reduction(ReductionPlan(gi=GIMode()), scan)


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
