"""Shared production-shaped fixtures for the frozen E2-SD oracle."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from xdart.gui.tabs.scattering.contracts import (
    AdmissionReceipt,
    SourceCapture,
    SourceObservation,
    SourceObservationRequest,
    SourceObservationStatus,
    StartCapture,
)
from xdart.gui.tabs.scattering.events import RequestId
from xdart.gui.tabs.scattering.output_preflight import prepare_output
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec, image_series_spec


class Sources:
    def __init__(self) -> None:
        self.epoch = 0
        self.knowledge = None

    def capture(self, source, request_id):
        self.epoch += 1
        return SourceCapture(request_id, self.epoch, source)

    def cancel(self, _request_id) -> None:
        return

    def observe(
        self, request: SourceObservationRequest
    ) -> SourceObservation:
        return SourceObservation(
            request.observation_id,
            request.intent_revision,
            request.source,
            SourceObservationStatus.AVAILABLE,
            "raw_0001.tif",
            True,
            False,
        )

    def cancel_observation(self, _observation_id: int) -> None:
        return

    def publish_motor_knowledge(self, observation) -> None:
        self.knowledge = observation

    def project_motor_knowledge(self, source, candidate_fingerprint=None):
        knowledge = self.knowledge
        if knowledge is None or knowledge.source != source:
            return None
        if (
            candidate_fingerprint is not None
            and knowledge.candidate_fingerprint != candidate_fingerprint
        ):
            return None
        return knowledge


def store(
    tmp_path: Path, *, output_mode: str = "Overwrite"
) -> RunIntentStore:
    raw = tmp_path / "raw_0001.tif"
    raw.write_bytes(b"raw")
    return RunIntentStore(
        RunIntent(
            source_spec=image_series_spec(raw),
            poni_file=str(tmp_path / "cal.poni"),
            save_path=str(tmp_path / "processed.nxs"),
            output_mode=output_mode,
        )
    )


def write_poni(path: Path) -> None:
    path.write_text(
        "poni_version: 2\nDetector: Pilatus100k\nDetector_config: {}\n"
        "Distance: 0.1234\nPoni1: 0.05\nPoni2: 0.06\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n"
    )


def write_motor_container(path: Path, motor: str = "halpha") -> None:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        detector = entry.create_group("instrument").create_group("detector")
        detector.create_dataset(
            "data", data=np.ones((1, 195, 487), dtype=np.uint16)
        )
        positioners = entry["instrument"].create_group("positioners")
        positioners.create_group(motor).create_dataset("value", data=0.2)


def directory_start(
    tmp_path: Path,
    *,
    motors: tuple[str, ...] = ("halpha",),
    output_mode: str = "Overwrite",
) -> tuple[RunIntentStore, StartCapture]:
    raw = tmp_path / "raw"
    raw.mkdir()
    for index, motor in enumerate(motors):
        write_motor_container(raw / f"scan_{index}.nxs", motor)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    intent = RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode=output_mode,
        gi=GIIntent(enabled=True, incidence_motor="Manual"),
    )
    run_store = RunIntentStore(intent)
    snapshot = run_store.snapshot()
    request = RequestId(1)
    capture = SourceCapture(request, 1, source)
    return run_store, StartCapture(request, 1, snapshot, capture)


def admit_with_session(
    start: StartCapture,
) -> tuple[AdmissionReceipt, object]:
    sessions: list[object] = []
    receipt = prepare_output(
        start,
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    assert len(sessions) == 1
    return receipt, sessions[0]


def external_eiger_capture(
    tmp_path: Path,
) -> tuple[RunIntentStore, SourceCapture, StartCapture, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_poni(tmp_path / "cal.poni")
    sidecar = raw / "scan_data_000001.h5"
    master = raw / "scan_master.h5"
    with h5py.File(sidecar, "w") as handle:
        handle.create_group("entry").create_group("data").create_dataset(
            "data",
            data=np.ones((2, 195, 487), dtype=np.uint16),
            maxshape=(None, 195, 487),
        )
    with h5py.File(master, "w") as handle:
        data = handle.create_group("entry").create_group("data")
        data["data_000001"] = h5py.ExternalLink(
            sidecar.name, "/entry/data/data"
        )
    source = DirectorySourceSpec(raw, suffixes=(".h5",))
    run_store = RunIntentStore(
        RunIntent(
            source_spec=source,
            poni_file=str(tmp_path / "cal.poni"),
            save_path=str(tmp_path / "processed"),
            output_mode="Overwrite",
        )
    )
    snapshot = run_store.snapshot()
    request = RequestId(1)
    capture = SourceCapture(request, 1, source)
    return (
        run_store,
        capture,
        StartCapture(request, 1, snapshot, capture),
        sidecar,
    )
