from __future__ import annotations

import numpy as np
import pandas as pd

from ssrl_xrd_tools.core.containers import PONI

from xdart.modules.live import LiveFrame, LiveScan
from xdart.modules.reduction import plan_from_live_scan
from xdart.modules.sources import LiveScanFrameSource


def _poni() -> PONI:
    return PONI(
        dist=0.1,
        poni1=0.01,
        poni2=0.02,
        wavelength=1e-10,
        detector="Detector",
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
