from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from ssrl_xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D, PONI
from ssrl_xrd_tools.reduction import FrameReduction, ReductionPlan

from xdart.modules.live import LiveFrame, LiveScan
from xdart.modules.reduction import plan_from_live_scan
from xdart.modules.sources import (
    LiveScanFrameSource,
    PublicationSink,
    build_reduction_job,
)


def _poni() -> PONI:
    return PONI(
        dist=0.1,
        poni1=0.01,
        poni2=0.02,
        wavelength=1e-10,
        detector="Detector",
    )


def _r1d() -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([1.0, 2.0]),
        sigma=None,
        unit="q_A^-1",
    )


def _r2d() -> IntegrationResult2D:
    return IntegrationResult2D(
        radial=np.array([0.0, 1.0]),
        azimuthal=np.array([-1.0, 1.0]),
        intensity=np.ones((2, 2)),
        sigma=None,
        unit="q_A^-1",
    )


def test_live_scan_frame_source_exposes_headless_contract(tmp_path):
    scan = LiveScan(
        "scan",
        data_file=str(tmp_path / "scan.nxs"),
        scan_data=pd.DataFrame(
            {"i0": [10.0], "sample": ["A"]},
            index=[3],
        ),
        mg_args={"wavelength": 1e-10},
    )
    frame = LiveFrame(
        idx=3,
        map_raw=np.arange(4).reshape(2, 2),
        poni=_poni(),
        scan_info={"th": 0.2},
    )
    scan.frames[3] = frame

    source = LiveScanFrameSource(scan)

    assert source.frame_indices == [3]
    assert source.wavelength == 1.0
    assert np.asarray(source.load_frame(3)).sum() == 6
    assert source.metadata_for(3)["th"] == 0.2
    assert source.metadata_for(3)["i0"] == 10.0
    assert source.metadata_for(3)["sample"] == "A"
    canonical = source.to_scan()
    assert canonical.name == "scan"
    assert canonical.frame_indices == [3]
    assert canonical.scan_data.loc[3, "sample"] == "A"


def test_plan_from_live_scan_preserves_gi_submodes(tmp_path):
    scan = LiveScan(
        "gi",
        data_file=str(tmp_path / "gi.nxs"),
        gi=True,
        incidence_motor="th",
        bai_1d_args={"gi_mode_1d": "q_oop", "npt_oop": 123, "numpoints": 7},
        bai_2d_args={"gi_mode_2d": "q_chi", "npt_rad": 8, "npt_azim": 9},
    )

    plan = plan_from_live_scan(scan, gi_incident_angle=0.25)

    assert plan.gi is not None
    assert plan.gi.incident_angle == 0.25
    assert plan.gi.incidence_motor == "th"
    assert plan.gi.mode_1d.value == "q_oop"
    assert plan.gi.mode_2d.value == "q_chi"
    assert plan.gi.npt_oop == 123
    assert plan.integration_1d.npt == 7
    assert plan.integration_2d.npt_rad == 8


def test_build_reduction_job_uses_source_plan_and_default_memory_sink(tmp_path):
    scan = LiveScan("scan", data_file=str(tmp_path / "scan.nxs"))
    scan.frames[1] = LiveFrame(idx=1, map_raw=np.ones((2, 2)), poni=_poni())

    job = build_reduction_job(scan, frame_indices=[1], integrate_2d=False)

    assert isinstance(job.source, LiveScanFrameSource)
    assert isinstance(job.plan, ReductionPlan)
    assert job.source.frame_indices == [1]
    assert job.plan.integration_1d is not None
    assert job.plan.integration_2d is None


def test_publication_sink_publishes_frame_publication():
    publications = []
    sink = PublicationSink(publications.append, generation=4)
    scan = SimpleNamespace(name="scan")
    frame = SimpleNamespace(index=2, source_path=None, source_frame_index=None, image=None)
    sink.begin(scan, ReductionPlan())

    sink.write(
        frame,
        FrameReduction(
            frame_index=2,
            result_1d=_r1d(),
            result_2d=_r2d(),
            metadata={"phase": "alpha", "i0": 10.0},
        ),
    )

    assert len(publications) == 1
    pub = publications[0]
    assert pub.label == 2
    assert pub.generation == 4
    assert pub.metadata_raw["phase"] == "alpha"
    assert pub.metadata_numeric["i0"] == 10.0
