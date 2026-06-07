"""Tests for the scan/frame headless reduction API."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
from ssrl_xrd_tools.reduction import (
    CancelToken,
    Frame,
    GIMode,
    Integration1DPlan,
    Integration2DPlan,
    MaskSpec,
    MemorySink,
    NexusSink,
    ReductionPlan,
    Scan,
    run_reduction,
)
import ssrl_xrd_tools.reduction.core as reduction_core


def _r1d(value: float, *, unit: str = "q_A^-1") -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([value, value + 1.0]),
        sigma=None,
        unit=unit,
    )


def _r2d(value: float, *, unit: str = "q_A^-1") -> IntegrationResult2D:
    return IntegrationResult2D(
        radial=np.array([0.0, 1.0]),
        azimuthal=np.array([-1.0, 1.0]),
        intensity=np.full((2, 2), value),
        sigma=None,
        unit=unit,
    )


def test_scan_sorts_frames_and_synthesizes_metadata() -> None:
    scan = Scan(
        name="sample_scan1",
        frames=[
            Frame(index=2, image=np.zeros((2, 2)), metadata={"I0": 20}),
            Frame(index=1, image=np.zeros((2, 2)), metadata={"I0": 10}),
        ],
        energy=12.398,
        wavelength=1.0,
        motors={"tth": [0.0, 1.0]},
        sample_name="sample",
    )

    assert [f.index for f in scan.frames] == [1, 2]
    meta = scan.to_metadata()
    assert meta is not None
    assert meta.scan_id == "sample_scan1"
    np.testing.assert_allclose(meta.counters["i0"], [10, 20])
    np.testing.assert_allclose(meta.angles["tth"], [0.0, 1.0])


def test_scan_frame_source_iterates_bounded_chunks() -> None:
    from ssrl_xrd_tools.reduction import FrameSource

    scan = Scan(
        "source",
        [Frame(2, image=np.full((2, 2), 2.0)),
         Frame(0, image=np.zeros((2, 2))),
         Frame(1, image=np.ones((2, 2)))],
    )
    assert isinstance(scan, FrameSource)
    chunks = list(scan.iter_chunks(2))
    assert [indices for _, indices in chunks] == [[0, 1], [2]]
    np.testing.assert_array_equal(chunks[0][0][:, 0, 0], [0.0, 1.0])


def test_scan_frame_source_clears_images_loaded_by_chunks() -> None:
    calls: list[int] = []

    def _loader(frame: Frame) -> np.ndarray:
        calls.append(frame.index)
        return np.full((2, 2), frame.index)

    preloaded = Frame(0, image=np.zeros((2, 2)))
    lazy = [Frame(1, loader=_loader), Frame(2, loader=_loader)]
    scan = Scan("bounded", [preloaded, *lazy])

    chunks = list(scan.iter_chunks(1))
    assert [indices for _, indices in chunks] == [[0], [1], [2]]
    assert calls == [1, 2]
    assert preloaded.image is not None
    assert lazy[0].image is None
    assert lazy[1].image is None


def test_frame_loads_with_custom_loader_once() -> None:
    calls: list[int] = []

    def _loader(frame: Frame) -> np.ndarray:
        calls.append(frame.index)
        return np.full((2, 2), frame.index)

    frame = Frame(index=3, loader=_loader)

    np.testing.assert_array_equal(frame.load_image(), np.full((2, 2), 3))
    np.testing.assert_array_equal(frame.load_image(), np.full((2, 2), 3))
    assert calls == [3]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"integration_1d": None, "integration_2d": None},
            "integration_1d",
        ),
    ],
)
def test_reduction_plan_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ReductionPlan(**kwargs)


def test_run_reduction_chunk_size_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    # chunk_size is a run_reduction kwarg now, not a plan field.
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kwargs: _r1d(1.0))
    scan = Scan("scan", [Frame(0, image=np.ones((2, 2)))], integrator=object())
    with pytest.raises(ValueError, match="chunk_size"):
        run_reduction(ReductionPlan(), scan, chunk_size=0)


def test_integration_plan_validation() -> None:
    with pytest.raises(ValueError, match="npt"):
        Integration1DPlan(npt=0)
    with pytest.raises(ValueError, match="npt_rad"):
        Integration2DPlan(npt_rad=0)


def test_gi_mode_is_a_sum_type() -> None:
    # GIMode bundles the GI-only parameters into one optional field on
    # ReductionPlan; the type system rules out "gi=False, incident_angle=2.5".
    gi = GIMode(incident_angle=2.5)
    assert gi.incident_angle == 2.5
    assert gi.tilt_angle == 0.0
    plan = ReductionPlan(gi=gi)
    assert plan.gi is gi
    # GIMode is frozen — can't mutate after construction
    with pytest.raises(Exception):
        gi.incident_angle = 5.0  # type: ignore[misc]


def test_run_reduction_1d_2d_mask_threshold_norm_and_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_1d(image, ai, **kwargs):
        calls.append({"kind": "1d", "image": image.copy(), **kwargs})
        return _r1d(float(np.nansum(image)), unit=kwargs["unit"])

    def fake_2d(image, ai, **kwargs):
        calls.append({"kind": "2d", "image": image.copy(), **kwargs})
        return _r2d(float(np.nanmean(image)), unit=kwargs["unit"])

    monkeypatch.setattr(reduction_core, "integrate_1d", fake_1d)
    monkeypatch.setattr(reduction_core, "integrate_2d", fake_2d)

    events = []
    scan = Scan(
        name="scan",
        frames=[
            Frame(
                index=0,
                image=np.array([[1.0, 2.0], [100.0, -5.0]]),
                background=np.ones((2, 2)),
                mask=np.array([[False, True], [False, False]]),
            )
        ],
        integrator=object(),
    )
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(
            npt=5,
            unit="2th_deg",
            method="BBox",
        ),
        integration_2d=Integration2DPlan(
            npt_rad=6,
            npt_azim=7,
            unit="q_A^-1",
            method="csr",
        ),
        mask=np.array([[False, False], [True, False]]),
        threshold_min=0.0,
        threshold_max=50.0,
    )
    scan.frames[0].normalization_factor = 9.0

    result = run_reduction(plan, scan, chunk_size=1, progress_cb=events.append)

    assert result.n_processed == 1
    assert result.frames[0].result_1d is not None
    assert result.frames[0].result_2d is not None
    assert [e.stage for e in events] == [
        "start", "chunk", "load", "integrate", "write", "finish"
    ]
    assert calls[0]["npt"] == 5
    assert calls[0]["method"] == "BBox"
    assert calls[0]["normalization_factor"] == 9.0
    np.testing.assert_array_equal(
        calls[0]["mask"],
        np.array([[False, True], [True, False]]),
    )
    assert not np.isnan(calls[0]["image"][0, 1])  # mask passed separately
    assert calls[0]["image"][0, 0] == 0.0  # background subtracted
    assert np.isnan(calls[0]["image"][1, 0])  # threshold_max
    assert np.isnan(calls[0]["image"][1, 1])  # threshold_min
    assert calls[1]["npt_rad"] == 6
    assert calls[1]["npt_azim"] == 7


def test_run_reduction_resolves_flat_mask_spec_after_image_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_1d(image, ai, **kwargs):
        calls.append(kwargs)
        return _r1d(1.0)

    monkeypatch.setattr(reduction_core, "integrate_1d", fake_1d)

    result = run_reduction(
        ReductionPlan(mask=MaskSpec(np.array([0, 3]))),
        Scan("scan", [Frame(0, image=np.ones((2, 2)))], integrator=object()),
    )

    assert result.n_processed == 1
    np.testing.assert_array_equal(
        calls[0]["mask"],
        np.array([[True, False], [False, True]]),
    )


def test_run_reduction_uses_independent_1d_2d_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_1d(image, ai, **kwargs):
        calls.append({"kind": "1d", **kwargs})
        return _r1d(1.0)

    def fake_2d(image, ai, **kwargs):
        calls.append({"kind": "2d", **kwargs})
        return IntegrationResult2D(
            radial=np.array([0.0, 1.0]),
            azimuthal=np.array([-5.0, 5.0]),
            intensity=np.ones((2, 2)),
            unit=kwargs["unit"],
        )

    monkeypatch.setattr(reduction_core, "integrate_1d", fake_1d)
    monkeypatch.setattr(reduction_core, "integrate_2d", fake_2d)

    result = run_reduction(
        ReductionPlan(
            integration_1d=Integration1DPlan(
                npt=11,
                unit="q_A^-1",
                method="csr",
                radial_range=(1.0, 2.0),
                azimuth_range=(-30.0, 30.0),
                monitor_key="i0",
            ),
            integration_2d=Integration2DPlan(
                npt_rad=12,
                npt_azim=13,
                unit="2th_deg",
                method="BBox",
                radial_range=(10.0, 20.0),
                azimuth_range=(80.0, 100.0),
                azimuth_offset=90.0,
                monitor_key="i1",
            ),
        ),
        Scan(
            "scan",
            [Frame(0, image=np.ones((2, 2)), metadata={"I0": 25.0, "i1": 50.0})],
            integrator=object(),
        ),
    )

    assert calls[0]["normalization_factor"] == 25.0
    assert calls[0]["npt"] == 11
    assert calls[0]["unit"] == "q_A^-1"
    assert calls[0]["radial_range"] == (1.0, 2.0)
    assert calls[0]["azimuth_range"] == (-30.0, 30.0)
    assert calls[1]["normalization_factor"] == 50.0
    assert calls[1]["npt_rad"] == 12
    assert calls[1]["npt_azim"] == 13
    assert calls[1]["unit"] == "2th_deg"
    assert calls[1]["radial_range"] == (10.0, 20.0)
    assert calls[1]["azimuth_range"] == (-10.0, 10.0)
    np.testing.assert_allclose(result.frames[0].result_2d.azimuthal, [85.0, 95.0])


def test_run_reduction_cancellation_and_clear_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = CancelToken()
    seen: list[int] = []

    def fake_1d(image, ai, **kwargs):
        seen.append(int(image[0, 0]))
        token.cancel()
        return _r1d(float(image[0, 0]))

    monkeypatch.setattr(reduction_core, "integrate_1d", fake_1d)
    scan = Scan(
        name="scan",
        frames=[
            Frame(index=0, image=np.zeros((1, 1))),
            Frame(index=1, image=np.ones((1, 1))),
        ],
        integrator=object(),
    )

    result = run_reduction(
        ReductionPlan(),
        scan,
        chunk_size=2,
        clear_frame_images=True,
        cancel_token=token,
    )

    assert result.cancelled
    assert result.n_processed == 1
    assert seen == [0]
    assert scan.frames[0].image is None
    assert scan.frames[1].image is not None


def test_memory_sink_can_be_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kwargs: _r1d(1.0))
    sink = MemorySink()
    result = run_reduction(
        ReductionPlan(),
        Scan("scan", [Frame(0, image=np.ones((2, 2)))], integrator=object()),
        sink,
    )

    assert result.frames[0] is sink.frames[0]


def test_nexus_sink_writes_frame_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kwargs: _r1d(3.0))
    out = tmp_path / "scan.nxs"
    scan = Scan(
        "scan",
        [Frame(0, image=np.ones((2, 2)))],
        integrator=object(),
        energy=12.398,
        wavelength=1.0,
    )

    result = run_reduction(ReductionPlan(), scan, NexusSink(out, overwrite=True))

    assert result.output_path == out
    with h5py.File(out, "r") as h5:
        # Stacked read_scan-compatible layout: (n_frames, n_q).
        assert "entry/integrated_1d/intensity" in h5
        assert list(h5["entry/integrated_1d/frame_index"][()]) == [0]
        np.testing.assert_allclose(
            h5["entry/integrated_1d/intensity"][0],
            [3.0, 4.0],
        )


def test_reduction_validation_for_shapes_and_duplicate_frames() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Scan(
            "scan",
            [Frame(0, image=np.ones((2, 2))), Frame(0, image=np.ones((2, 2)))],
        )

    scan = Scan("scan", [Frame(0, image=np.ones((2, 2)))], integrator=object())
    with pytest.raises(ValueError, match="mask shape"):
        run_reduction(ReductionPlan(mask=np.ones((3, 3), dtype=bool)), scan)

    scan = Scan(
        "scan",
        [Frame(0, image=np.ones((2, 2)), background=np.ones((3, 3)))],
        integrator=object(),
    )
    with pytest.raises(ValueError, match="background shape"):
        run_reduction(ReductionPlan(), scan)

    scan = Scan("scan", [Frame(0)], integrator=object())
    with pytest.raises(ValueError, match="no image"):
        run_reduction(ReductionPlan(), scan)


def test_nexus_sink_flush_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kwargs: _r1d(float(image[0, 0])))
    calls = []

    class FakeH5:
        def flush(self):
            calls.append("flush")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(reduction_core, "open_nexus_writer", lambda *a, **k: FakeH5())
    monkeypatch.setattr(reduction_core, "write_nexus_frame", lambda *a, **k: None)

    result = run_reduction(
        ReductionPlan(),
        Scan(
            "scan",
            [
                Frame(0, image=np.zeros((2, 2))),
                Frame(1, image=np.ones((2, 2))),
                Frame(2, image=np.full((2, 2), 2.0)),
            ],
            integrator=object(),
            energy=12.398,
            wavelength=1.0,
        ),
        NexusSink("scan.nxs", overwrite=True, flush_every=2),
    )

    assert result.n_processed == 3
    assert calls == ["flush", "flush", "close"]  # frame 2, finish, close

    with pytest.raises(ValueError, match="flush_every"):
        NexusSink("scan.nxs", flush_every=0).begin(
            Scan("scan", [Frame(0, image=np.ones((2, 2)))], integrator=object()),
            ReductionPlan(),
        )


def test_existing_notebook_import_surface_still_imports() -> None:
    from ssrl_xrd_tools.analysis.fitting import FitConfig, PhaseFitter, fit_peaks
    from ssrl_xrd_tools.analysis.strain import sin2psi_analysis
    from ssrl_xrd_tools.integrate import (
        create_multigeometry_integrators,
        integrate_1d,
        integrate_2d,
        stitch_1d,
        stitch_2d,
    )

    assert integrate_1d
    assert integrate_2d
    assert create_multigeometry_integrators
    assert stitch_1d
    assert stitch_2d
    assert PhaseFitter
    assert fit_peaks
    assert FitConfig
    assert sin2psi_analysis


# ---------------------------------------------------------------------------
# scan_data persistence (#nexussink_scan_data — core provenance)
# ---------------------------------------------------------------------------

def test_scan_to_scan_data_assembles_per_frame_table() -> None:
    import math

    import pandas as pd

    scan = Scan(
        name="conditions",
        frames=[
            # frame 0 has tag (string) but no stress; frame 1 has stress but no tag
            Frame(index=0, image=np.zeros((2, 2)),
                  metadata={"temperature": 300.0, "tag": "a"}),
            Frame(index=1, image=np.zeros((2, 2)),
                  metadata={"temperature": 310.0, "stress": 5.0}),
        ],
        motors={"th": np.array([0.1, 0.2])},
    )
    df = scan.to_scan_data()

    assert list(df.index) == [0, 1]
    assert {"temperature", "tag", "stress", "th"} <= set(df.columns)
    # union of frame metadata, aligned per frame
    assert df.loc[0, "temperature"] == 300.0
    assert df.loc[1, "temperature"] == 310.0
    # per-frame motor array folded in
    assert df.loc[0, "th"] == 0.1 and df.loc[1, "th"] == 0.2
    # keys missing on a frame → NaN/None
    assert pd.isna(df.loc[0, "stress"])
    assert pd.isna(df.loc[1, "tag"])
    # column order is metadata-first-seen then motors
    assert list(df.columns)[:3] == ["temperature", "tag", "stress"]


def test_scan_to_scan_data_empty_scan() -> None:
    df = Scan(name="empty", frames=[]).to_scan_data()
    assert len(df) == 0
    assert list(df.columns) == []


def test_nexussink_persists_scan_data_roundtrip(tmp_path: Path) -> None:
    from ssrl_xrd_tools.io.read import get_metadata

    path = tmp_path / "scan_conditions.nxs"
    scan = Scan(
        name="ramp",
        frames=[
            Frame(index=1, image=np.zeros((2, 2)),
                  metadata={"temperature": 300.0, "stress": 2.0, "th": 0.15}),
            Frame(index=2, image=np.zeros((2, 2)),
                  metadata={"temperature": 305.0, "stress": 4.0, "th": 0.25}),
            Frame(index=3, image=np.zeros((2, 2)),
                  metadata={"temperature": 310.0, "stress": 6.0, "th": 0.35}),
        ],
    )
    sink = NexusSink(path)
    sink.begin(scan, ReductionPlan())
    for frame in scan.frames:
        sink.write(
            frame,
            reduction_core.FrameReduction(
                frame_index=frame.index, result_1d=_r1d(float(frame.index)),
            ),
        )
    sink.finish(reduction_core.ReductionResult("ramp", {}, len(scan)))

    meta = get_metadata(path)
    sd = meta["scan_data"]
    assert {"temperature", "stress", "th"} <= set(sd)
    # per-frame condition columns round-trip, aligned to frame indices 1,2,3
    np.testing.assert_allclose(sd["temperature"], [300.0, 305.0, 310.0])
    np.testing.assert_allclose(sd["stress"], [2.0, 4.0, 6.0])
    np.testing.assert_allclose(sd["th"], [0.15, 0.25, 0.35])
