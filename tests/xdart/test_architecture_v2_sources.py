"""LiveScan -> core adapter contract (architecture v2).

``xdart.modules.reduction.scan_from_live_scan`` is THE LiveScan->core
adapter (the duplicate ``LiveScanFrameSource`` was deleted in the 6d
monorepo step).  These tests pin the contract surface the old FrameSource
tests covered: headless Scan exposure, scan_data metadata merge (numeric
AND string columns), and the wavelength extraction rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xrd_tools.core.containers import PONI

from xdart.modules.live import LiveFrame, LiveScan
from xdart.modules.reduction import plan_from_live_scan, scan_from_live_scan


def _poni() -> PONI:
    return PONI(
        dist=0.1,
        poni1=0.01,
        poni2=0.02,
        wavelength=1e-10,
        detector="Detector",
    )


def test_scan_from_live_scan_exposes_headless_contract(tmp_path):
    scan = LiveScan(
        "scan",
        data_file=str(tmp_path / "scan.nxs"),
        scan_data=pd.DataFrame(
            {"i0": [10.0], "sample": ["A"]},
            index=[3],
        ),
        mg_args={"wavelength": 0.7293e-10},
    )
    frame = LiveFrame(
        idx=3,
        map_raw=np.arange(4).reshape(2, 2),
        poni=_poni(),
        scan_info={"th": 0.2},
    )
    scan.frames[3] = frame

    canonical = scan_from_live_scan(scan)

    assert canonical.name == "scan"
    assert canonical.frame_indices == [3]
    assert canonical.wavelength == pytest.approx(0.7293)
    headless = canonical.frames[0]
    assert np.asarray(headless.image).sum() == 6
    assert headless.metadata["th"] == 0.2
    assert headless.metadata["i0"] == 10.0
    assert headless.metadata["sample"] == "A"   # string column survives
    # the core Scan is directly chunk-iterable (the RSM/stitch boundary)
    chunks = list(canonical.iter_chunks(8))
    assert len(chunks) == 1 and chunks[0][1] == [3]


def test_scan_from_live_scan_rejects_default_wavelength_sentinel(tmp_path):
    scan = LiveScan(
        "scan",
        data_file=str(tmp_path / "scan.nxs"),
        mg_args={"wavelength": 1e-10},   # the 1.0 Å default sentinel
    )
    scan.frames[1] = LiveFrame(idx=1, map_raw=np.ones((2, 2)), poni=_poni())

    assert scan_from_live_scan(scan).wavelength is None


def test_scan_from_live_scan_uses_authoritative_reloaded_wavelength(tmp_path):
    scan = LiveScan(
        "scan",
        data_file=str(tmp_path / "scan.nxs"),
        mg_args={"wavelength": 1e-10},
    )
    scan._persisted_wavelength_m = 1e-10
    scan.frames[1] = LiveFrame(idx=1, map_raw=np.ones((2, 2)), poni=_poni())

    assert scan_from_live_scan(scan).wavelength == pytest.approx(1.0)


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
