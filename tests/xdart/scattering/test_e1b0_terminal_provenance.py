from __future__ import annotations

import ast
import inspect
from pathlib import Path

import h5py
import numpy as np
import pytest

import xrd_tools.reduction.core as reduction_core
from xrd_tools.core.containers import IntegrationResult1D
from xrd_tools.core.provenance import read_provenance
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.reduction import (
    Frame,
    Integration1DPlan,
    NexusSink,
    ReductionPlan,
    Scan,
    run_reduction,
)
from xrd_tools.session.run_configuration import RunIntent
from xdart.gui.tabs.scattering import events
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import (
    ExecutorAccepted,
    LifecycleError,
    LifecycleStatus,
    OwnersClosed,
    PreflightAccepted,
    RunIdentity,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase


def _configuration():
    return RunIntent(
        source_spec=SourceSpec(Path("/data/frame_0001.tif"), SourceKind.IMAGE_FILE),
    ).freeze()


def _result_1d(value: float = 1.0) -> IntegrationResult1D:
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([value, value + 1.0]),
        sigma=np.array([0.1, 0.2]),
        unit="q_A^-1",
    )


def _write_provenance(tmp_path, monkeypatch, sink: NexusSink) -> Path:
    monkeypatch.setattr(
        reduction_core,
        "integrate_1d",
        lambda image, ai, **kwargs: _result_1d(float(np.sum(image))),
    )
    raw = tmp_path / "raw_0000.tif"
    raw.write_bytes(b"raw pointer target")
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(
            npt=2,
            unit="q_A^-1",
            method="csr",
            radial_range=(0.0, 1.0),
        ),
        integration_2d=None,
    )
    scan = Scan(
        "provenance",
        [Frame(0, image=np.ones((2, 2)), source_path=raw)],
        integrator=object(),
    )
    result = run_reduction(plan, scan, sink)
    assert result.n_processed == 1
    return sink.path


def _running() -> tuple[ScatteringCoordinator, RunIdentity]:
    coordinator = ScatteringCoordinator()
    request = coordinator.begin_start().request_id
    assert request is not None
    accepted = coordinator.preflight_accepted(
        PreflightAccepted(request, _configuration()),
    )
    identity = accepted.run_identity
    assert identity is not None
    assert coordinator.executor_accepted(ExecutorAccepted(identity)).status is LifecycleStatus.APPLIED
    return coordinator, identity


def test_nexus_sink_round_trips_full_frozen_configuration_provenance(tmp_path, monkeypatch):
    configuration = _configuration()
    path = _write_provenance(
        tmp_path,
        monkeypatch,
        NexusSink(
            tmp_path / "configuration.nxs",
            overwrite=True,
            run_configuration_provenance=configuration.as_provenance(),
        ),
    )

    assert read_provenance(path)["config"]["run_configuration"] == configuration.as_provenance()


def test_nexus_sink_keeps_generation_and_fingerprint_exact_for_equal_content_configs(
    tmp_path,
    monkeypatch,
):
    first = _configuration()
    second = RunIntent.from_frozen(first).freeze()
    assert first.fingerprint == second.fingerprint
    assert first.generation != second.generation

    first_path = _write_provenance(
        tmp_path,
        monkeypatch,
        NexusSink(tmp_path / "first.nxs", overwrite=True, run_configuration_provenance=first.as_provenance()),
    )
    second_path = _write_provenance(
        tmp_path,
        monkeypatch,
        NexusSink(tmp_path / "second.nxs", overwrite=True, run_configuration_provenance=second.as_provenance()),
    )

    first_saved = read_provenance(first_path)["config"]["run_configuration"]
    second_saved = read_provenance(second_path)["config"]["run_configuration"]
    assert first_saved["fingerprint"] == second_saved["fingerprint"] == first.fingerprint
    assert (first_saved["generation"], second_saved["generation"]) == (
        first.generation,
        second.generation,
    )


def test_nexus_sink_detaches_caller_provenance_before_output(tmp_path, monkeypatch):
    configuration = _configuration()
    supplied = configuration.as_provenance()
    sink = NexusSink(
        tmp_path / "detached.nxs",
        overwrite=True,
        run_configuration_provenance=supplied,
    )
    supplied["generation"] = 999
    supplied["gi"]["mode_1d"] = "changed-after-construction"

    path = _write_provenance(tmp_path, monkeypatch, sink)

    assert read_provenance(path)["config"]["run_configuration"] == configuration.as_provenance()


@pytest.mark.parametrize(
    "provenance",
    [
        {"schema_version": 1, "generation": 0, "fingerprint": "valid"},
        {"schema_version": 1, "generation": True, "fingerprint": "valid"},
        {"schema_version": 1, "generation": 1, "fingerprint": ""},
        {"generation": 1, "fingerprint": "valid"},
    ],
)
def test_nexus_sink_rejects_malformed_identity_before_output_creation(
    tmp_path,
    monkeypatch,
    provenance,
):
    output = tmp_path / "refused.nxs"
    opened: list[object] = []

    def unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("malformed provenance opened an output")

    monkeypatch.setattr(reduction_core, "open_nexus_writer", unexpected_open)

    def construct_or_begin() -> None:
        sink = NexusSink(output, run_configuration_provenance=provenance)
        sink.begin(
            Scan("refused", [], integrator=object()),
            ReductionPlan(
                integration_1d=Integration1DPlan(
                    npt=2,
                    unit="q_A^-1",
                    method="csr",
                    radial_range=(0.0, 1.0),
                ),
                integration_2d=None,
            ),
        )

    with pytest.raises(ValueError):
        construct_or_begin()

    assert opened == []
    assert not output.exists()


def test_nexus_sink_omits_run_configuration_without_changing_reduction_schema(tmp_path, monkeypatch):
    path = _write_provenance(tmp_path, monkeypatch, NexusSink(tmp_path / "legacy.nxs", overwrite=True))

    provenance = read_provenance(path)
    assert "run_configuration" not in provenance["config"]
    with h5py.File(path, "r") as h5:
        assert "entry/reduction/config/bai_1d_args" in h5


def test_terminal_normal_completion_is_exact_and_reusable():
    coordinator, identity = _running()
    before = coordinator.event_sequence

    ended = coordinator.execution_ended(events.ExecutionEnded(identity))
    finalized = coordinator.durable_final(events.DurableFinal(identity))

    assert ended.status is LifecycleStatus.APPLIED
    assert ended.phase is RunPhase.FINALIZING
    assert finalized.status is LifecycleStatus.APPLIED
    assert coordinator.phase is RunPhase.IDLE
    assert coordinator.active_run_identity is None
    assert coordinator.event_sequence == before + 2


def test_stop_completion_needs_execution_end_not_owners_closed():
    coordinator, identity = _running()

    stopped = coordinator.stop_requested(events.StopRequested(identity))
    before_owners_closed = coordinator.event_sequence
    premature_owners_closed = coordinator.owners_closed(OwnersClosed(identity))
    ended = coordinator.execution_ended(events.ExecutionEnded(identity))
    finalized = coordinator.durable_final(events.DurableFinal(identity))

    assert stopped.status is LifecycleStatus.APPLIED
    assert stopped.phase is RunPhase.STOPPING
    assert premature_owners_closed.status is LifecycleStatus.SUPERSEDED
    assert coordinator.event_sequence == before_owners_closed + 2
    assert ended.phase is RunPhase.FINALIZING
    assert finalized.phase is RunPhase.IDLE


def test_fatal_requires_matching_owners_closed_before_reset():
    coordinator, identity = _running()

    failed = coordinator.fatal(events.FatalExecution(identity))
    premature_reset = coordinator.reset()
    closed = coordinator.owners_closed(OwnersClosed(identity))
    reset = coordinator.reset()

    assert failed.phase is RunPhase.FAILED
    assert premature_reset.status is LifecycleStatus.REJECTED
    assert closed.status is LifecycleStatus.APPLIED
    assert reset.phase is RunPhase.IDLE


def test_active_close_retains_only_private_cleanup_identity_until_exact_acknowledgement():
    coordinator, identity = _running()

    closed = coordinator.close()
    reconstructed = RunIdentity(identity.generation, identity.fingerprint)
    wrong = coordinator.owners_closed(OwnersClosed(reconstructed))
    acknowledged = coordinator.owners_closed(OwnersClosed(identity))

    assert closed.status is LifecycleStatus.APPLIED
    assert closed.phase is RunPhase.STOPPING
    assert closed.run_identity is identity
    assert coordinator.request_id is None
    assert coordinator.attempt_run_identity is None
    assert coordinator.active_run_identity is None
    assert wrong.status is LifecycleStatus.SUPERSEDED
    assert acknowledged.phase is RunPhase.CLOSED


def test_failed_cleanup_already_acknowledged_can_close_immediately():
    coordinator, identity = _running()
    assert coordinator.fatal(events.FatalExecution(identity)).phase is RunPhase.FAILED
    assert coordinator.owners_closed(OwnersClosed(identity)).phase is RunPhase.FAILED

    closed = coordinator.close()

    assert closed.status is LifecycleStatus.APPLIED
    assert closed.phase is RunPhase.CLOSED
    assert closed.run_identity is None


def test_duplicate_active_close_is_inert_without_replacing_cleanup_identity():
    coordinator, identity = _running()
    first = coordinator.close()
    before = coordinator.event_sequence

    duplicate = coordinator.close()
    acknowledged = coordinator.owners_closed(OwnersClosed(identity))

    assert first.run_identity is identity
    assert duplicate.status is LifecycleStatus.SUPERSEDED
    assert coordinator.event_sequence == before + 1
    assert acknowledged.phase is RunPhase.CLOSED


def test_idle_and_pre_executor_close_are_immediate():
    idle = ScatteringCoordinator()
    assert idle.close().phase is RunPhase.CLOSED

    coordinator = ScatteringCoordinator()
    request = coordinator.begin_start().request_id
    assert request is not None
    assert coordinator.preflight_accepted(PreflightAccepted(request, _configuration())).phase is RunPhase.STARTING

    closed = coordinator.close()

    assert closed.status is LifecycleStatus.APPLIED
    assert closed.phase is RunPhase.CLOSED
    assert closed.run_identity is None


def test_terminal_malformed_stale_duplicate_and_post_close_events_are_inert():
    coordinator, identity = _running()
    reconstructed = RunIdentity(identity.generation, identity.fingerprint)
    stale = RunIdentity(identity.generation + 1, identity.fingerprint)

    assert coordinator.durable_final(events.DurableFinal(identity)).status is LifecycleStatus.REJECTED
    before = coordinator.event_sequence
    assert coordinator.execution_ended(object()).status is LifecycleStatus.SUPERSEDED
    assert coordinator.execution_ended(events.ExecutionEnded(reconstructed)).status is LifecycleStatus.SUPERSEDED
    assert coordinator.execution_ended(events.ExecutionEnded(stale)).status is LifecycleStatus.SUPERSEDED
    assert coordinator.event_sequence == before

    assert coordinator.execution_ended(events.ExecutionEnded(identity)).status is LifecycleStatus.APPLIED
    duplicate_before = coordinator.event_sequence
    assert coordinator.execution_ended(events.ExecutionEnded(identity)).status is LifecycleStatus.SUPERSEDED
    assert coordinator.event_sequence == duplicate_before
    assert coordinator.durable_final(events.DurableFinal(identity)).status is LifecycleStatus.APPLIED
    assert coordinator.durable_final(events.DurableFinal(identity)).status is LifecycleStatus.SUPERSEDED

    post_close, close_identity = _running()
    post_close.close()
    before_post_close = post_close.event_sequence
    assert post_close.durable_final(events.DurableFinal(close_identity)).status is LifecycleStatus.SUPERSEDED
    assert post_close.event_sequence == before_post_close


def test_terminal_surface_stays_headless_and_does_not_retain_frozen_configuration():
    coordinator_source = Path(inspect.getsourcefile(ScatteringCoordinator) or "")
    assert coordinator_source.name == "coordinator.py"
    source = Path(reduction_core.__file__).read_text()
    assert "FrozenRunConfiguration" not in source
    imports = [
        alias.name
        for node in ast.walk(ast.parse(coordinator_source.read_text()))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert not any(name.startswith(("PyQt", "qtpy", "xdart.gui.tabs.static_scan")) for name in imports)
    assert source.count("class NexusSink") == 1
