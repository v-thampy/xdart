from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json

import numpy as np
import pytest

from xrd_tools.analysis.xu_stitch_calibration import (
    XuStitchCalibrationInput,
    canonical_surface_resource_bytes,
    capture_xu_stitch_calibration,
)
from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
from xrd_tools.integrate import xu_stitch
from xrd_tools.integrate.xu_stitch import (
    XuPowderQProvider,
    XuStitchScienceRefused,
    resolve_xu_stitch_effective_geometry,
    run_xu_hist_stitch_1d,
)

_SOLID_ANGLE_PIN = json.loads(canonical_surface_resource_bytes())["corrections"][
    "solid_angle_sha256"
]


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
        self.frame_indices = (7,)

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
        with pytest.raises(TypeError, match="invalid"):
            replace(geometry, hxrd=object())
        provider = XuPowderQProvider(
            geometry,
            receipt.projection.value["acquisition"],
        )
        lease = provider.frame(nu=-10.0, del_=14.0)
        assert lease._root_ref() is lease._root
        q = lease.q_magnitude()
        assert q.shape == (195, 1475)
        assert q.dtype == np.dtype(np.float64)
        assert np.isfinite(q).any()
        q = None
        lease.release()
    assert owner.execution_record is not None
    assert owner.execution_record.restore_passed is True


def _resolve(receipt):
    with xu_runtime_session() as session:
        return resolve_xu_stitch_effective_geometry(receipt, session)


def _host_solid_angle_stack(values):
    class _Stack:
        def normalization(self, ai, shape):
            assert tuple(shape) == values.shape
            return values

    return _Stack


def test_xu_solid_angle_reference_authenticates_and_is_what_integrates(tmp_path):
    reference = xu_stitch._solid_angle_reference(_SOLID_ANGLE_PIN, (195, 1475))
    assert reference.dtype == np.dtype("<f8")
    assert reference.flags.c_contiguous
    assert hashlib.sha256(reference.tobytes(order="C")).hexdigest() == _SOLID_ANGLE_PIN

    geometry = _resolve(_receipt(tmp_path))
    projection = geometry.projection
    # Whatever this host's libm/SIMD produced, integration uses the pinned bytes.
    assert (
        hashlib.sha256(geometry.solid_angle.tobytes(order="C")).hexdigest()
        == _SOLID_ANGLE_PIN
    )
    assert projection.solid_angle_sha256 == _SOLID_ANGLE_PIN
    assert projection.solid_angle_match in {"exact", "tolerance"}
    assert (projection.solid_angle_match == "exact") == (
        projection.solid_angle_sha256_host == _SOLID_ANGLE_PIN
    )
    assert 0.0 <= projection.solid_angle_max_rel_dev <= 1e-12
    provenance = projection.to_provenance()
    assert provenance["solid_angle_sha256_host"] == projection.solid_angle_sha256_host
    assert provenance["solid_angle_match"] == projection.solid_angle_match
    assert provenance["solid_angle_max_rel_dev"] == projection.solid_angle_max_rel_dev
    with pytest.raises(TypeError, match="invalid"):
        replace(
            projection,
            solid_angle_match="exact",
            solid_angle_max_rel_dev=1e-13,
        )
    with pytest.raises(TypeError, match="invalid"):
        replace(
            projection,
            solid_angle_sha256_host="0" * 64,
            solid_angle_match="exact",
            solid_angle_max_rel_dev=0.0,
        )
    with pytest.raises(TypeError, match="invalid"):
        replace(
            projection,
            solid_angle_sha256_host="0" * 64,
            solid_angle_match="tolerance",
            solid_angle_max_rel_dev=2e-12,
        )


def test_xu_solid_angle_exact_and_tolerance_paths(tmp_path, monkeypatch):
    receipt = _receipt(tmp_path)
    reference = xu_stitch._solid_angle_reference(_SOLID_ANGLE_PIN, (195, 1475))

    monkeypatch.setattr(
        xu_stitch, "CorrectionStack", _host_solid_angle_stack(reference.copy())
    )
    exact = _resolve(receipt)
    assert exact.projection.solid_angle_match == "exact"
    assert exact.projection.solid_angle_max_rel_dev == 0.0
    assert exact.projection.solid_angle_sha256_host == _SOLID_ANGLE_PIN

    one_ulp_up = np.nextafter(reference, np.inf)
    monkeypatch.setattr(
        xu_stitch, "CorrectionStack", _host_solid_angle_stack(one_ulp_up)
    )
    tolerated = _resolve(receipt)
    assert tolerated.projection.solid_angle_match == "tolerance"
    assert tolerated.projection.solid_angle_sha256_host == hashlib.sha256(
        one_ulp_up.tobytes(order="C")
    ).hexdigest()
    assert tolerated.projection.solid_angle_sha256_host != _SOLID_ANGLE_PIN
    assert 0.0 < tolerated.projection.solid_angle_max_rel_dev < 1e-15
    assert tolerated.projection.fingerprint == exact.projection.fingerprint
    assert np.array_equal(tolerated.solid_angle, reference)
    assert not np.array_equal(tolerated.solid_angle, one_ulp_up)


def test_xu_solid_angle_refuses_beyond_tolerance(tmp_path, monkeypatch):
    receipt = _receipt(tmp_path)
    reference = xu_stitch._solid_angle_reference(_SOLID_ANGLE_PIN, (195, 1475))
    monkeypatch.setattr(
        xu_stitch,
        "CorrectionStack",
        _host_solid_angle_stack(reference * (1.0 + 1e-9)),
    )
    with pytest.raises(XuStitchScienceRefused) as raised:
        _resolve(receipt)
    assert raised.value.code == "XU_SOLID_ANGLE_MISMATCH"
    message = str(raised.value)
    assert "max_rel_dev=1.0" in message
    assert "e-09" in message
    assert "rtol=1e-12" in message
    assert f"reference sha256={_SOLID_ANGLE_PIN}" in message

    monkeypatch.setattr(
        xu_stitch,
        "CorrectionStack",
        _host_solid_angle_stack(np.where(reference > 0.8, 0.0, reference)),
    )
    with pytest.raises(XuStitchScienceRefused) as raised:
        _resolve(receipt)
    assert raised.value.code == "XU_SOLID_ANGLE_MISMATCH"
    assert "finite positive" in str(raised.value)


def test_xu_solid_angle_refuses_unavailable_or_forged_reference(
    tmp_path, monkeypatch
):
    receipt = _receipt(tmp_path)
    reference = xu_stitch._solid_angle_reference(_SOLID_ANGLE_PIN, (195, 1475))

    def _npy(values):
        buffer = io.BytesIO()
        np.save(buffer, values, allow_pickle=False)
        return buffer.getvalue()

    def _missing():
        raise FileNotFoundError("sidecar not packaged")

    forged = [
        _missing,
        lambda: b"",
        lambda: b"\x93NUMPY" + b"\0" * xu_stitch._MAX_SOLID_ANGLE_RESOURCE_BYTES,
        lambda: _npy(reference)[:-8],
        lambda: _npy(np.nextafter(reference, np.inf)),
        lambda: _npy(reference.astype(np.float32)),
        lambda: _npy(reference[:, :1474]),
    ]
    for loader in forged:
        monkeypatch.setattr(xu_stitch, "_solid_angle_reference_bytes", loader)
        with pytest.raises(XuStitchScienceRefused) as raised:
            _resolve(receipt)
        assert raised.value.code == "XU_SOLID_ANGLE_REFERENCE_UNAVAILABLE"

    # Same values in Fortran order hash identically once made C-contiguous;
    # that is the same array, not a forgery.
    for loader in (lambda: _npy(reference), lambda: _npy(np.asfortranarray(reference))):
        monkeypatch.setattr(xu_stitch, "_solid_angle_reference_bytes", loader)
        geometry = _resolve(receipt)
        assert geometry.projection.solid_angle_sha256 == _SOLID_ANGLE_PIN
        assert geometry.solid_angle.flags.c_contiguous
        assert np.array_equal(geometry.solid_angle, reference)


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
    assert result.observations.source_del_range_deg == (14.0, 14.0)
    assert result.observations.source_nu_range_deg == (-10.0, -10.0)
    assert result.observations.source_energy_range_eV is None
    assert result.observations.control_domain_extrapolation_frame_count == 0
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


def test_xu_hist_checks_optional_energy_and_control_domain_before_frame_load(tmp_path):
    receipt = _receipt(tmp_path)
    source = _OneFrameSource(del_value=45.0, nu_value=29.0)
    source.metadata["energy"] = 17000.018
    result = run_xu_hist_stitch_1d(
        receipt,
        source,
        frame_indices=(7,),
        q_min_A_inverse=1.0,
        q_max_A_inverse=5.2,
        npt=64,
        monitor_key="monitor",
        source_energy_key="energy",
        max_frame_bytes=4 * 1024 * 1024,
    )
    assert result.observations.source_energy_range_eV == (17000.018, 17000.018)
    assert result.observations.control_domain_extrapolation_frame_count == 1
    assert result.observations.warnings == (
        "XU_CALIBRATION_EXTRAPOLATION_WITHIN_VALIDATED_SCAN",
    )

    conflict = _OneFrameSource()
    conflict.metadata["energy"] = 18000.0
    with pytest.raises(XuStitchScienceRefused) as raised:
        run_xu_hist_stitch_1d(
            receipt,
            conflict,
            frame_indices=(7,),
            q_min_A_inverse=1.0,
            q_max_A_inverse=5.2,
            npt=64,
            monitor_key="monitor",
            source_energy_key="energy",
            max_frame_bytes=4 * 1024 * 1024,
        )
    assert raised.value.code == "XU_SOURCE_ENERGY_CONFLICT"
    assert conflict.loads == 0


def test_xu_hist_refuses_duplicate_reordered_or_foreign_source_selection(tmp_path):
    receipt = _receipt(tmp_path)
    source = _OneFrameSource()
    source.frame_indices = (7, 8)
    for labels in ((7, 7), (8, 7), (9,)):
        with pytest.raises(XuStitchScienceRefused) as raised:
            run_xu_hist_stitch_1d(
                receipt,
                source,
                frame_indices=labels,
                q_min_A_inverse=1.0,
                q_max_A_inverse=5.2,
                npt=64,
                monitor_key="monitor",
                max_frame_bytes=4 * 1024 * 1024,
            )
        assert raised.value.code == "SOURCE_SELECTION_IDENTITY_MISMATCH"
    assert source.loads == 0
