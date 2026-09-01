from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.analysis.xu_stitch_calibration import (
    XuStitchCalibrationInput,
    canonical_surface_resource_bytes,
    capture_xu_stitch_calibration,
)
from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
from xrd_tools.integrate.xu_stitch import (
    XuPowderQProvider,
    XuStitchScienceRefused,
    resolve_xu_stitch_effective_geometry,
    run_xu_hist_stitch_1d,
)


def _receipt(tmp_path):
    project = tmp_path / "project"
    target = project / "calibration" / "xu" / "surface.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(canonical_surface_resource_bytes())
    return capture_xu_stitch_calibration(
        XuStitchCalibrationInput("calibration/xu/surface.json"),
        project_root=project,
    )


class _OneFrameSource:
    def __init__(self, *, del_value=14.0, nu_value=-10.0, monitor=2.0):
        self.image = np.ones((195, 1475), dtype=np.float32)
        self.metadata = {
            "del": del_value,
            "nu": nu_value,
            "monitor": monitor,
        }
        self.loads = 0

    def metadata_for(self, label):
        assert label == 7
        return self.metadata

    def load_frame(self, label):
        assert label == 7
        self.loads += 1
        return self.image


def test_xu_effective_geometry_and_q_provider_use_one_shared_root(tmp_path):
    receipt = _receipt(tmp_path)
    owner = xu_runtime_session()
    with owner as session:
        geometry = resolve_xu_stitch_effective_geometry(receipt, session)
        assert geometry.projection.shape == (195, 1475)
        assert geometry.projection.mask_count == 2730
        assert geometry.projection.energy_eV == 17000.018
        provider = XuPowderQProvider(
            geometry,
            receipt.projection.value["acquisition"],
        )
        lease = provider.frame(nu=-10.0, del_=14.0)
        q = lease.q_magnitude()
        assert q.shape == (195, 1475)
        assert q.dtype == np.dtype(np.float64)
        assert np.isfinite(q).any()
        q = None
        lease.release()
    assert owner.execution_record is not None
    assert owner.execution_record.restore_passed is True


def test_xu_hist_science_releases_each_frame_and_restores_runtime(tmp_path):
    receipt = _receipt(tmp_path)
    source = _OneFrameSource()
    result = run_xu_hist_stitch_1d(
        receipt,
        source,
        frame_indices=(7,),
        q_min_A_inverse=1.0,
        q_max_A_inverse=5.2,
        npt=64,
        monitor_key="monitor",
        max_frame_bytes=4 * 1024 * 1024,
    )
    assert source.loads == 1
    assert result.selected_frame_count == 1
    assert result.release_check_frame_count == 1
    assert result.runtime.restore_passed is True
    assert result.payload.unit == "q_A^-1"
    assert result.payload.radial.shape == (64,)
    assert result.diagnostics.coverage.shape == (64,)
    assert np.all(result.diagnostics.coverage >= 0)
    assert np.all(result.diagnostics.coverage <= 2**24)
    assert np.array_equal(
        result.diagnostics.coverage,
        np.floor(result.diagnostics.coverage),
    )
    occupied = result.diagnostics.normalization > 0
    assert occupied.any()
    assert np.isfinite(result.payload.intensity[occupied]).all()
    assert np.isnan(result.payload.intensity[~occupied]).all()


def test_xu_hist_refuses_domain_before_loading_frame(tmp_path):
    receipt = _receipt(tmp_path)
    source = _OneFrameSource(del_value=46.0)
    with pytest.raises(XuStitchScienceRefused) as raised:
        run_xu_hist_stitch_1d(
            receipt,
            source,
            frame_indices=(7,),
            q_min_A_inverse=1.0,
            q_max_A_inverse=5.2,
            npt=64,
            monitor_key="monitor",
            max_frame_bytes=4 * 1024 * 1024,
        )
    assert raised.value.code == "XU_CALIBRATION_DOMAIN_EXCEEDED"
    assert source.loads == 0


def test_xu_hist_refuses_invalid_monitor_before_loading_frame(tmp_path):
    receipt = _receipt(tmp_path)
    source = _OneFrameSource(monitor=0.0)
    with pytest.raises(XuStitchScienceRefused) as raised:
        run_xu_hist_stitch_1d(
            receipt,
            source,
            frame_indices=(7,),
            q_min_A_inverse=1.0,
            q_max_A_inverse=5.2,
            npt=64,
            monitor_key="monitor",
            max_frame_bytes=4 * 1024 * 1024,
        )
    assert raised.value.code == "INVALID_MONITOR_VALUE"
    assert source.loads == 0
