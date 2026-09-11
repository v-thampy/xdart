from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.core.containers import IntegrationResult1D
from xrd_tools.core.provenance import write_provenance
from xrd_tools.io.nexus import write_nexus
from xrd_tools.session.experiment_reload import ExperimentRecordReader, ReloadStatus
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
            orientation=OrientationState(4, "pyfai-fiber-sample-orientation-v1"),
        ),
        provenance=ExperimentProvenance(source_kind="test"),
    )


def _write_current_record(path: Path, config: dict[str, object]) -> None:
    write_nexus(
        path,
        results_1d={
            0: IntegrationResult1D(
                radial=np.array([1.0], dtype=np.float32),
                intensity=np.ones(1, dtype=np.float32),
                unit="q_A^-1",
            ),
        },
    )
    with h5py.File(path, "r+") as handle:
        write_provenance(handle, config=config)


def test_exact_record_round_trips_typed_state_and_fingerprint(tmp_path: Path) -> None:
    state = _exact_state()
    record = tmp_path / "exact.nexus"
    _write_current_record(record, {"experiment": state.to_provenance()})

    result = ExperimentRecordReader().read(record)
    assert result.status is ReloadStatus.EXACT
    assert result.experiment == state
    assert result.content_fingerprint == state.content_fingerprint


@pytest.mark.parametrize(
    "config",
    [
        {"run_configuration": {"schema_version": 1, "gi": {}}},
        {"gi_config": {"enabled": False}},
        {"run_configuration": {}, "gi_config": {}},
    ],
)
def test_historical_fallback_envelopes_are_absent(
    tmp_path: Path, config: dict[str, object]
) -> None:
    record = tmp_path / "historical.nexus"
    _write_current_record(record, config)

    result = ExperimentRecordReader().read(record)
    assert result.status is ReloadStatus.ABSENT
    assert result.experiment is None
    assert result.reason == "exact experiment provenance is absent"


@pytest.mark.parametrize("marker", ["not-a-mapping", None, []])
def test_malformed_exact_marker_fails_closed(
    tmp_path: Path, marker: object
) -> None:
    record = tmp_path / "malformed.nexus"
    _write_current_record(
        record,
        {
            "experiment": marker,
            "run_configuration": {"gi": {"enabled": False}},
        },
    )

    result = ExperimentRecordReader().read(record)
    assert result.status is ReloadStatus.ABSENT
    assert result.experiment is None
    assert result.reason == "exact experiment provenance is absent"


def test_open_failure_is_distinct_from_absent(tmp_path: Path) -> None:
    broken = tmp_path / "truncated.nexus"
    broken.write_bytes(b"not an HDF5 record")

    result = ExperimentRecordReader().read(broken)
    assert result.status is ReloadStatus.OPEN_FAILURE
    assert result.reason
