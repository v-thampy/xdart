from __future__ import annotations

import os
from pathlib import Path

import h5py
import pytest

from xrd_tools.core.provenance import write_provenance
from xrd_tools.session.experiment_reload import LegacyRecordAdapter, ReloadStatus
from xrd_tools.session.experiment_state import (
    CalibrationState,
    EnergySource,
    EnergyState,
    ExperimentProvenance,
    ExperimentState,
    FactStatus,
    GeometryState,
    MaskState,
    OrientationState,
    PoniValues,
    SampleState,
)


ROOT = Path(__file__).resolve().parents[2]


def _data_root() -> Path:
    configured = os.environ.get("XDART_TEST_DATA")
    candidates = [
        Path(configured) if configured else None,
        ROOT / "test_data",
        ROOT.parents[1] / "test_data",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate
    pytest.skip("XDART_TEST_DATA/real test_data is unavailable")


def _exact_state() -> ExperimentState:
    return ExperimentState(
        experiment_id="exact-record",
        revision=7,
        calibration=CalibrationState(
            values=PoniValues(0.2, 0.01, 0.02, 0.0, 0.0, 0.0, 1.0e-10),
            detector_id="Eiger4M",
            value_fingerprint="v" * 64,
            source_sha256="s" * 64,
            source_uri="exact.poni",
            mask=MaskState.absent(),
            status=FactStatus.PRESENT,
        ),
        geometry=GeometryState(),
        energy=EnergyState(
            wavelength_m=1.0e-10,
            source=EnergySource.CALIBRATION,
            evidence={"calibration": 1.0e-10},
            status=FactStatus.PRESENT,
        ),
        sample=SampleState(
            sample_id="sample-exact",
            display_name="Exact sample",
            orientation=OrientationState(
                4, "pyfai-fiber-sample-orientation-v1"
            ),
        ),
        provenance=ExperimentProvenance(source_kind="test"),
    )


def test_exact_record_round_trips_typed_state_and_fingerprint(tmp_path: Path) -> None:
    state = _exact_state()
    record = tmp_path / "exact.nxs"
    with h5py.File(record, "w") as handle:
        write_provenance(handle, config={"experiment": state.to_provenance()})

    result = LegacyRecordAdapter().read(record)
    assert result.status is ReloadStatus.EXACT
    assert result.experiment == state
    assert result.content_fingerprint == state.content_fingerprint
    assert result.energy is state.energy or result.energy == state.energy


def test_partial_real_record_recovers_run_configuration_without_defaults() -> None:
    record = (
        _data_root()
        / "xdart_processed_data"
        / "Combi4_Angledependence_samz_4p9_03271002.nxs"
    )
    if not record.exists():
        pytest.skip(f"real partial fixture unavailable: {record}")

    result = LegacyRecordAdapter().read(record)
    assert result.status is ReloadStatus.PARTIAL
    assert result.experiment is None
    assert result.energy is not None
    assert result.energy.status is FactStatus.PRESENT
    assert result.energy.wavelength_m == pytest.approx(0.9762535309700808e-10)
    assert result.geometry is not None
    assert result.geometry.incidence.motor_name == "th"
    assert result.geometry.incidence.kind.value == "motor"
    assert result.sample is not None
    assert result.sample.orientation.code == 4


def test_legacy_real_record_reads_declared_source_wavelength_without_manual_default() -> None:
    record = (
        _data_root()
        / "test_relative_path"
        / "xdart_processed_data"
        / "LaB6_detx54_detyn1p5_eta3_scan001.nxs"
    )
    if not record.exists():
        pytest.skip(f"real legacy fixture unavailable: {record}")

    result = LegacyRecordAdapter().read(record)
    assert result.status is ReloadStatus.LEGACY
    assert result.energy is not None
    assert result.energy.wavelength_m > 0.0
    assert set(result.energy.evidence) == {"legacy_source"}
    if result.geometry is not None:
        assert result.geometry.incidence.motor_name != "Manual"


def test_real_record_without_reduction_is_absent_not_invented() -> None:
    record = _data_root() / "nexus" / "LaB6_0710_1025pm_00005.nxs"
    if not record.exists():
        pytest.skip(f"real absent fixture unavailable: {record}")

    result = LegacyRecordAdapter().read(record)
    assert result.status is ReloadStatus.ABSENT
    assert result.experiment is None
    assert result.calibration is None
    assert result.geometry is None
    assert result.energy is None
    assert result.sample is None


def test_open_failure_is_distinct_from_absent(tmp_path: Path) -> None:
    broken = tmp_path / "truncated.nxs"
    broken.write_bytes(b"not an HDF5 record")

    result = LegacyRecordAdapter().read(broken)
    assert result.status is ReloadStatus.OPEN_FAILURE
    assert result.status is not ReloadStatus.ABSENT
    assert result.reason


def test_disagreeing_persisted_wavelengths_fail_closed_as_conflict(
    tmp_path: Path,
) -> None:
    record = tmp_path / "conflict.nxs"
    with h5py.File(record, "w") as handle:
        write_provenance(
            handle,
            config={"run_configuration": {"schema_version": 1, "gi": {}}},
        )
        mono = handle.require_group("entry/instrument/monochromator")
        mono["wavelength"] = 1.0
        source = handle.require_group("entry/instrument/source")
        source["wavelength_A"] = 1.2

    result = LegacyRecordAdapter().read(record)
    assert result.status is ReloadStatus.PARTIAL
    assert result.energy is not None
    assert result.energy.status is FactStatus.CONFLICT
    assert result.energy.wavelength_m is None
    assert set(result.energy.evidence) == {"headless", "legacy_source"}


def test_malformed_geometry_envelope_is_absent_not_defaulted(
    tmp_path: Path,
) -> None:
    record = tmp_path / "malformed-geometry.nxs"
    with h5py.File(record, "w") as handle:
        write_provenance(
            handle,
            config={"run_configuration": {"gi": {"enabled": "false"}}},
        )

    result = LegacyRecordAdapter().read(record)
    assert result.status is ReloadStatus.ABSENT
    assert result.geometry is None
    assert result.reason


@pytest.mark.parametrize("marker", ["not-a-mapping", None, []])
def test_malformed_exact_marker_cannot_downgrade_to_partial(
    tmp_path: Path,
    marker: object,
) -> None:
    record = tmp_path / "malformed-exact-marker.nxs"
    with h5py.File(record, "w") as handle:
        write_provenance(
            handle,
            config={
                "experiment": marker,
                "run_configuration": {"gi": {"enabled": False}},
            },
        )

    result = LegacyRecordAdapter().read(record)
    assert result.status is ReloadStatus.ABSENT
    assert result.experiment is None
    assert result.geometry is None
    assert "experiment" in result.reason.lower()


@pytest.mark.parametrize("marker", ["not-a-mapping", None, []])
def test_malformed_run_marker_cannot_downgrade_to_legacy(
    tmp_path: Path,
    marker: object,
) -> None:
    record = tmp_path / "malformed-run-marker.nxs"
    with h5py.File(record, "w") as handle:
        write_provenance(
            handle,
            config={
                "run_configuration": marker,
                "gi_config": {"enabled": False},
            },
        )

    result = LegacyRecordAdapter().read(record)
    assert result.status is ReloadStatus.ABSENT
    assert result.experiment is None
    assert result.geometry is None
    assert "run configuration" in result.reason.lower()


@pytest.mark.parametrize(
    "config",
    [
        {"run_configuration": {}},
        {"gi_config": {}},
        {"gi_config": {"th_val": None}},
        {"gi_config": {"tilt_angle": None}},
        {"gi_config": {"incidence_motor": ""}},
    ],
)
def test_empty_legacy_envelopes_do_not_invent_experiment_facts(
    tmp_path: Path,
    config: dict[str, object],
) -> None:
    record = tmp_path / "empty-envelope.nxs"
    with h5py.File(record, "w") as handle:
        write_provenance(handle, config=config)

    result = LegacyRecordAdapter().read(record)
    assert result.status is ReloadStatus.ABSENT
    assert result.experiment is None
    assert result.calibration is None
    assert result.geometry is None
    assert result.energy is None
    assert result.sample is None
    assert result.reason


@pytest.mark.parametrize(
    ("config", "status"),
    [
        ({"run_configuration": {}}, ReloadStatus.PARTIAL),
        ({"gi_config": {}}, ReloadStatus.LEGACY),
    ],
)
def test_empty_envelope_keeps_genuine_persisted_energy_fact(
    tmp_path: Path,
    config: dict[str, object],
    status: ReloadStatus,
) -> None:
    record = tmp_path / "empty-envelope-with-energy.nxs"
    with h5py.File(record, "w") as handle:
        write_provenance(handle, config=config)
        mono = handle.require_group("entry/instrument/monochromator")
        mono["wavelength"] = 1.0

    result = LegacyRecordAdapter().read(record)
    assert result.status is status
    assert result.geometry is None
    assert result.energy is not None
    assert result.energy.wavelength_m == pytest.approx(1.0e-10)


def test_partial_reload_does_not_relabel_run_fingerprint_as_experiment_content(
    tmp_path: Path,
) -> None:
    record = tmp_path / "partial-run-fingerprint.nxs"
    with h5py.File(record, "w") as handle:
        write_provenance(
            handle,
            config={"run_configuration": {"fingerprint": "run-fingerprint"}},
        )
        mono = handle.require_group("entry/instrument/monochromator")
        mono["wavelength"] = 1.0

    result = LegacyRecordAdapter().read(record)
    assert result.status is ReloadStatus.PARTIAL
    assert result.energy is not None
    assert result.content_fingerprint is None


def test_partial_source_only_mask_is_conflicting_evidence_not_present(
    tmp_path: Path,
) -> None:
    record = tmp_path / "partial-source-only-mask.nxs"
    poni_values = {
        "dist": 0.2,
        "poni1": 0.01,
        "poni2": 0.02,
        "rot1": 0.0,
        "rot2": 0.0,
        "rot3": 0.0,
        "wavelength_m": 1.0e-10,
    }
    with h5py.File(record, "w") as handle:
        write_provenance(
            handle,
            config={
                "run_configuration": {
                    "mask_file": "mask.edf",
                    "scientific_signature": {
                        "accepted_scientific_assets": {
                            "poni_values": poni_values,
                        },
                    },
                },
            },
        )

    result = LegacyRecordAdapter().read(record)
    assert result.status is ReloadStatus.PARTIAL
    assert result.calibration is not None
    assert result.calibration.mask.status is FactStatus.CONFLICT
    assert result.calibration.mask.source_uri == "mask.edf"
    assert result.calibration.mask.sha256 == ""


def test_malformed_exact_projection_fails_closed(tmp_path: Path) -> None:
    state = _exact_state()
    payload = state.to_provenance()
    payload["content_fingerprint"] = "0" * 64
    record = tmp_path / "bad-exact.nxs"
    with h5py.File(record, "w") as handle:
        write_provenance(handle, config={"experiment": payload})

    result = LegacyRecordAdapter().read(record)
    assert result.status is ReloadStatus.ABSENT
    assert result.experiment is None
    assert "fingerprint" in result.reason.lower()
