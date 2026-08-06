"""Retained green guards for behavior accepted at the rejected E2-S tip."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fabio
import numpy as np

from tests.xdart.scattering._e2sd_support import (
    admit_with_session,
    directory_start,
    write_poni,
)
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import (
    StandardRunExecutor,
    _StandardRun,
)
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.adapters.target_reservation import (
    RunResources,
    TargetLease,
)
from xdart.gui.tabs.scattering.contracts import (
    AdmissionFailure,
    AdmissionReceipt,
    SourceCapture,
    SourceObservationRequest,
    StartCapture,
)
from xdart.gui.tabs.scattering.events import CleanupStatus, RequestId, RunIdentity
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec


def _identity() -> RunIdentity:
    return RunIdentity(1, "a" * 64)


def test_target_lease_remains_last_while_other_owners_are_live(
    tmp_path: Path,
) -> None:
    class Session:
        def finish(self, **_kwargs):
            raise RuntimeError("session still owns the writer")

    class Sink:
        def abort(self, _result) -> None:
            raise RuntimeError("writer close failed")

    target = tmp_path / "same-target.nxs"
    lease = TargetLease.acquire((target,))
    run = _StandardRun(
        None,
        _identity(),
        None,
        None,
        Session(),
        None,
        target,
        sink=Sink(),
        resources=RunResources(None, None, lease),
    )
    executor = StandardRunExecutor()
    executor._active = run

    receipt = executor._cleanup(run)

    assert receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert run.resources is not None
    assert run.resources.target_lease is lease
    assert target.resolve(strict=False) in TargetLease._reserved


def test_two_independent_tiff_series_keep_distinct_group_identity(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for prefix in ("alpha", "beta"):
        for index in (1, 2):
            fabio.tifimage.TifImage(
                data=np.full((4, 4), index, dtype=np.uint16)
            ).write(str(raw / f"{prefix}_{index:04d}.tif"))
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".tif",))
    run_store = RunIntentStore(
        RunIntent(
            source_spec=source,
            poni_file=str(poni),
            save_path=str(tmp_path / "processed"),
            output_mode="Overwrite",
        )
    )
    snapshot = run_store.snapshot()
    request = RequestId(2)
    start = StartCapture(
        request, 1, snapshot, SourceCapture(request, 1, source)
    )

    receipt, session = admit_with_session(start)
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        assert len(deferred.entries) == 2
        assert {
            tuple(candidate.path.name for candidate in entry.candidates)
            for entry in deferred.entries
        } == {
            ("alpha_0001.tif", "alpha_0002.tif"),
            ("beta_0001.tif", "beta_0002.tif"),
        }
        assert {entry.target.name for entry in deferred.entries} == {
            "alpha.nexus",
            "beta.nexus",
        }
    finally:
        session.close()


def test_authoritative_motor_choices_are_candidate_intersection(
    tmp_path: Path,
) -> None:
    _store, start = directory_start(tmp_path, motors=("halpha", "eta"))
    receipt, session = admit_with_session(start)
    try:
        assert receipt.gi_motor_choices == ()
    finally:
        session.close()


def test_recursive_passive_observation_is_direct_only(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    nested = raw / "nested"
    nested.mkdir(parents=True)
    (raw / "direct.nxs").write_bytes(b"not opened")
    (nested / "hidden.nxs").write_bytes(b"not walked")
    source = DirectorySourceSpec(
        raw, recursive=True, suffixes=(".nxs",)
    )
    request = SourceObservationRequest(1, 0, source)
    adapter = FilesystemSourceAdapter()

    with patch("h5py.File", side_effect=AssertionError("passive HDF open")):
        observation = adapter.observe(request)

    assert observation.direct_child_count == 1
    assert observation.subdirectories_deferred is True


def test_append_refusal_acquires_zero_resources(tmp_path: Path) -> None:
    _store, start = directory_start(tmp_path, output_mode="Append")
    executor = StandardRunExecutor()
    with patch.object(
        executor_module,
        "build_admission_receipt",
        side_effect=AssertionError("Append opened source resources"),
    ):
        token = executor.begin_admission(start)
        deadline = __import__("time").monotonic() + 3.0
        result = None
        while result is None and __import__("time").monotonic() < deadline:
            result = executor.poll_admission(token)
            __import__("time").sleep(0.005)

    assert type(result) is AdmissionFailure
    assert "Append is deferred to H23" in result.reason
    operation = executor._admission
    assert operation is not None
    assert operation.directory_session is None
    assert operation.target_lease is None
