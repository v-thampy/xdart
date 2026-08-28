from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_tools.core.provenance import write_provenance
from xrd_tools.io.schema import (
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    SCHEMA_NAME_ATTR,
    SCHEMA_VERSION_ATTR,
)
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


def _stamp_current_record(handle: h5py.File) -> None:
    entry = handle.require_group("entry")
    entry.attrs[SCHEMA_NAME_ATTR] = PROCESSED_SCHEMA_NAME
    entry.attrs[SCHEMA_VERSION_ATTR] = PROCESSED_SCHEMA_VERSION
    result = entry.create_group("integrated_1d")
    result.attrs["NX_class"] = "NXdata"
    result.attrs["signal"] = "intensity"
    result.attrs["axes"] = ("frame_index", "q")
    result.create_dataset("frame_index", data=np.array([0], dtype=np.int64))
    result.create_dataset("q", data=np.array([1.0], dtype=np.float32))
    result.create_dataset(
        "intensity", data=np.ones((1, 1), dtype=np.float32),
    )


def test_exact_record_round_trips_typed_state_and_fingerprint(tmp_path: Path) -> None:
    state = _exact_state()
    record = tmp_path / "exact.nexus"
    with h5py.File(record, "w") as handle:
        _stamp_current_record(handle)
        write_provenance(handle, config={"experiment": state.to_provenance()})

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
    with h5py.File(record, "w") as handle:
        _stamp_current_record(handle)
        write_provenance(handle, config=config)

    result = ExperimentRecordReader().read(record)
    assert result.status is ReloadStatus.ABSENT
    assert result.experiment is None
    assert "exact experiment" in result.reason


@pytest.mark.parametrize("marker", ["not-a-mapping", None, []])
def test_malformed_exact_marker_fails_closed(
    tmp_path: Path, marker: object
) -> None:
    record = tmp_path / "malformed.nexus"
    with h5py.File(record, "w") as handle:
        _stamp_current_record(handle)
        write_provenance(
            handle,
            config={
                "experiment": marker,
                "run_configuration": {"gi": {"enabled": False}},
            },
        )

    result = ExperimentRecordReader().read(record)
    assert result.status is ReloadStatus.ABSENT
    assert result.experiment is None
    assert "exact experiment" in result.reason


def test_open_failure_is_distinct_from_absent(tmp_path: Path) -> None:
    broken = tmp_path / "truncated.nexus"
    broken.write_bytes(b"not an HDF5 record")

    result = ExperimentRecordReader().read(broken)
    assert result.status is ReloadStatus.OPEN_FAILURE
    assert result.reason
