"""Focused thin-GUI oracle for the headless Average Scan operation."""
from __future__ import annotations
import copy
from dataclasses import fields, replace
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
import tifffile
from xdart.gui.tabs.scattering.adapters import external_operation as adapter
from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
from xdart.gui.tabs.scattering.browse_values import BrowseCleanupReceipt
from xdart.gui.tabs.scattering.controls_editing import EditRefusal, reduce_control_edit
from xdart.gui.tabs.scattering.controls_inventory import AVERAGE_SCAN
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.operation_values import (
    OperationContextStamp,
    OperationIdentity,
    OperationPending,
    OperationTerminal,
    OperationTerminalStatus,
    OperationUpdate,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_operations import (
    AverageOperationState,
    WorkspaceOperationOwner,
    WorkspaceRefreshEffect,
)
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.reduction import (
    AverageCommand, AverageFiniteCountsEvidence, AveragePendingPhase,
    AverageScanPending, AverageScanRecipe, AverageScanResult,
    AverageScanRunner, GIMode, Integration1DPlan, Integration2DPlan,
    ReductionPlan,
)
from xrd_tools.session.intent_store import IntentFreezeAccepted, RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent, ThresholdIntent


def _set_average_state(
    page,
    identity: OperationIdentity,
    revision: int,
    target: str,
    *,
    entry: str = "entry",
    pending: OperationPending | None = None,
    source_root: str | None = None,
) -> WorkspaceOperationOwner:
    operations = page._workspace_operations
    operations._average = AverageOperationState(
        identity,
        revision,
        target,
        entry,
        pending,
        source_root,
    )
    return operations


def _result(disposition: str, target: str = "/detached/average.nxs") -> AverageScanResult:
    from xrd_tools.io.output_transaction import StreamTerminal
    from xrd_tools.reduction.average import (
        _average_output_artifact, _average_target,
    )
    # A real Average result NEVER names the ANCHOR.  The successor route writes
    # <family>.average-<version>.nexus, and the page verifies a terminal by
    # RECOMPUTING exactly that from anchor + version_identity
    # (workspace_operations.py:719).  Returning the anchor here fabricated an
    # impossible result that every reload path then correctly refused, so the
    # tests were asserting refusals they had manufactured themselves.
    version = "e" * 64
    artifact = _average_output_artifact(_average_target(target), version)
    committed = disposition == "COMMITTED"
    evidence = (AverageFiniteCountsEvidence(
        "average_scan_v1", 2, (1, 1), "<u4", "c" * 64, 1, 2, 0,
        (1, 1), "gzip", 1, True, False) if committed else None)
    diagnostics = {
        "COMMITTED": ("", ""),
        "REFUSED": ("AVERAGE_TEST_REFUSED", "detached refusal"),
        "CANCELLED": ("AVERAGE_TEST_CANCELLED", "detached cancellation"),
        "ABORTED": ("AVERAGE_TEST_ABORT", "detached diagnostic"),
    }
    code, diagnostic = diagnostics[disposition]
    return AverageScanResult(
        disposition=disposition, target=artifact, entry="entry",
        version_identity=version,
        operation_identity="a" * 64, science_identity="b" * 64,
        contributor_extent=2, logical_labels=(1,),
        committed_labels=(1,) if committed else (),
        metadata_denominators=(("I0", 2),) if committed else (), finite_counts=evidence,
        diagnostic_code=code, diagnostic=diagnostic,
        h23_phase="committed" if committed else None,
        commit_identity=(
            StreamTerminal(artifact, 1, "d" * 64, 1, 1, 1, 1, 1)
            if committed else None
        ),
    )


def _source(tmp_path: Path, arrays=None) -> SourceSpec:
    tmp_path.mkdir(exist_ok=True)
    arrays = arrays or (np.arange(4, dtype="u2").reshape(2, 2),
                        np.arange(4, dtype="u2").reshape(2, 2) + 2)
    files = []
    for index, array in enumerate(arrays, 1):
        path = tmp_path / f"scan_{index:04d}.tif"
        tifffile.imwrite(path, array); files.append(str(path))
    selected = Path(files[0])
    return SourceSpec(tmp_path, SourceKind.TIFF_SERIES, options={
        "selected_file": str(selected), "files": tuple(files),
        "pattern": "scan_*.tif", "scan_name": "scan", "metadata_format": None,
    })


def _configuration(
    source,
    reduction=None,
    *,
    output_mode="Overwrite",
    live_mode=False,
    batch_mode=False,
    poni_file="",
    mask_file="",
    background=None,
    project_root="",
    max_cores=1,
):
    from xrd_tools.reduction.background import FrameBackgroundPlan

    reduction = ReductionPlan() if reduction is None else reduction
    one = reduction.integration_1d or Integration1DPlan()
    two = reduction.integration_2d
    one_args = {
        **dict(one.extra),
        "npt": one.npt,
        "unit": one.unit,
        "method": one.method,
        "radial_range": one.radial_range,
        "azimuth_range": one.azimuth_range,
        "monitor": one.monitor_key,
        "error_model": one.error_model,
        "polarization_factor": one.polarization_factor,
        "chi_npt_rad": one.npt_rad,
        "chi_offset": one.azimuth_offset,
    }
    two_args = {} if two is None else {
        **dict(two.extra),
        "npt_rad": two.npt_rad,
        "npt_azim": two.npt_azim,
        "unit": two.unit,
        "method": two.method,
        "radial_range": two.radial_range,
        "azimuth_range": two.azimuth_range,
        "chi_offset": two.azimuth_offset,
        "monitor": two.monitor_key,
        "error_model": two.error_model,
        "polarization_factor": two.polarization_factor,
    }
    gi = GIIntent()
    if reduction.gi is not None:
        value = reduction.gi
        gi = GIIntent(
            enabled=True,
            incidence_motor=value.incidence_motor or "Manual",
            th_val=0.1 if value.incident_angle is None else value.incident_angle,
            sample_orientation=value.sample_orientation,
            tilt_angle=value.tilt_angle,
            mode_1d=value.mode_1d.value,
            mode_2d=value.mode_2d.value,
            gi_exit_angle_convention=value.gi_exit_angle_convention,
        )
    intent = RunIntent(
        source_spec=source,
        processing_mode="Int 2D" if two is not None else "Int 1D",
        output_mode=output_mode,
        live_mode=live_mode,
        batch_mode=batch_mode,
        max_cores=max_cores,
        bai_1d_args=one_args,
        bai_2d_args=two_args,
        gi=gi,
        threshold=ThresholdIntent(
            apply_threshold=(
                reduction.threshold_min is not None
                or reduction.threshold_max is not None
            ),
            threshold_min=reduction.threshold_min,
            threshold_max=reduction.threshold_max,
            mask_saturation=reduction.mask_saturation,
        ),
        poni_file=poni_file,
        mask_file=mask_file,
        background=(FrameBackgroundPlan() if background is None else background),
        project_root=str(project_root),
        run_options={"series_average": True},
    )
    frozen = RunIntentStore(intent).freeze(expected_revision=0)
    assert type(frozen) is IntentFreezeAccepted
    return frozen.configuration


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        (
            "standard",
            ReductionPlan(
                integration_1d=Integration1DPlan(npt=13),
                integration_2d=Integration2DPlan(npt_rad=7, npt_azim=5),
            ),
        ),
        (
            "int1d",
            ReductionPlan(integration_1d=Integration1DPlan(npt=17)),
        ),
        (
            "gi-manual",
            ReductionPlan(
                integration_1d=Integration1DPlan(npt=19, unit="qip_A^-1"),
                integration_2d=Integration2DPlan(
                    npt_rad=11, npt_azim=9, unit="qip_A^-1",
                ),
                gi=GIMode(
                    incident_angle=0.27, sample_orientation=3,
                    tilt_angle=0.04, mode_1d="q_ip",
                    mode_2d="qip_qoop",
                ),
            ),
        ),
        (
            "gi-motor",
            ReductionPlan(
                integration_1d=Integration1DPlan(npt=23, unit="qoop_A^-1"),
                integration_2d=Integration2DPlan(
                    npt_rad=12, npt_azim=10, unit="q_A^-1",
                ),
                gi=GIMode(
                    incidence_motor="theta", sample_orientation=2,
                    tilt_angle=-0.03, mode_1d="q_oop", mode_2d="q_chi",
                ),
            ),
        ),
        (
            "threshold",
            ReductionPlan(
                integration_1d=Integration1DPlan(npt=29),
                integration_2d=Integration2DPlan(npt_rad=8, npt_azim=6),
                threshold_min=2.5, threshold_max=4094.5,
            ),
        ),
        (
            "saturation",
            ReductionPlan(
                integration_1d=Integration1DPlan(npt=31),
                integration_2d=Integration2DPlan(npt_rad=9, npt_azim=7),
                mask_saturation=True,
            ),
        ),
    ),
)
def test_frozen_configuration_has_one_independent_mask_free_reduction_policy(
    tmp_path, case, expected,
) -> None:
    from xdart.gui.tabs.scattering.output_preflight import (
        native_int_reduction_plan,
    )

    configuration = _configuration(_source(tmp_path / case), expected)

    actual = native_int_reduction_plan(configuration)

    assert actual == expected
    assert actual.mask is None


def test_average_recipe_identity_carries_current_marker_and_decodes_legacy(
    tmp_path,
) -> None:
    from xrd_tools.corrections.grazing import (
        GI_EXIT_ANGLE_CONVENTION,
        LEGACY_GI_EXIT_ANGLE_CONVENTION,
    )
    from xrd_tools.reduction import average as average_module

    reduction = ReductionPlan(
        integration_1d=Integration1DPlan(npt=17),
        gi=GIMode(incident_angle=0.3, mode_1d="q_oop"),
    )
    recipe = AverageScanRecipe(
        _source(tmp_path / "current-average-recipe"),
        tmp_path / "current-average.nxs",
        reduction,
    )
    assert len(recipe.integrator_gi) == 9
    assert recipe.integrator_gi[-1] == GI_EXIT_ANGLE_CONVENTION
    assert average_module._thaw_reduction(recipe).gi.gi_exit_angle_convention == (
        GI_EXIT_ANGLE_CONVENTION
    )

    legacy = replace(recipe, integrator_gi=recipe.integrator_gi[:-1])
    assert average_module._thaw_reduction(legacy).gi.gi_exit_angle_convention == (
        LEGACY_GI_EXIT_ANGLE_CONVENTION
    )
    with pytest.raises(ValueError, match="unsupported keyset"):
        average_module._thaw_reduction(
            replace(recipe, integrator_gi=recipe.integrator_gi + ("third",))
        )


def _join(slot: OperationSlot, identity):
    worker = slot._worker; assert worker is not None
    worker.join(20); assert not worker.is_alive()
    update = slot.poll(identity); assert update is not None and update.terminal is not None
    return update


def _runner_from_callable(call):
    class Runner:
        def __init__(self, recipe):
            self.recipe = recipe
            self.result = None

        def start(self, **kwargs):
            self.result = call(self.recipe, **kwargs)
            return self.result

        def command(self, *_args, **_kwargs):
            pytest.fail("terminal test runner received a cleanup command")

        def close(self):
            return self.result

    return Runner


def _run_terminal_average(recipe, **kwargs):
    runner = AverageScanRunner(recipe)
    result = runner.start(**kwargs)
    assert type(result) is AverageScanResult
    assert runner.close() is result
    return result


def _wait_update(slot: OperationSlot, identity, predicate, *, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        update = slot.poll(identity)
        if update is not None and predicate(update):
            return update
        time.sleep(0.005)
    pytest.fail("operation update did not arrive")


def test_average_cleanup_pending_requires_exact_explicit_commands(
    tmp_path, monkeypatch,
) -> None:
    commands = []
    result = _result("REFUSED", str((tmp_path / "average.nxs").resolve()))

    class Runner:
        def __init__(self, recipe):
            self.recipe = recipe

        def start(self, **_kwargs):
            return AverageScanPending(
                "headless-operation", 1,
                AveragePendingPhase.SOURCE_CLEANUP,
                "source close retained",
            )

        def command(self, command, pending):
            commands.append((command, pending))
            if len(commands) == 1:
                return AverageScanPending(
                    "headless-operation", 2,
                    AveragePendingPhase.OUTPUT_SETTLEMENT,
                    "output settlement retained",
                )
            return result

        def close(self):
            return result

    monkeypatch.setattr(adapter, "AverageScanRunner", Runner)
    slot = OperationSlot()
    identity = slot.begin_average(
        _configuration(_source(tmp_path / "source")), result.target,
        stamp=OperationContextStamp(0),
    )
    assert identity is not None
    first_update = _wait_update(
        slot, identity, lambda value: value.pending is not None,
    )
    first = first_update.pending
    assert type(first) is OperationPending and first.revision == 1
    assert first.phase == "source-cleanup"
    assert not slot.retry_average(identity, OperationPending(
        identity, first.revision, first.phase, first.diagnostic,
    ))
    assert slot.retry_average(identity, first)
    assert not slot.retry_average(identity, first)

    second_update = _wait_update(
        slot, identity, lambda value: value.pending is not None,
    )
    second = second_update.pending
    assert type(second) is OperationPending and second.revision == 2
    assert second.phase == "output-settlement"
    assert slot.cancel(identity)
    terminal = _wait_update(
        slot, identity, lambda value: value.terminal is not None,
    )
    assert terminal.terminal.status is OperationTerminalStatus.RETURNED
    assert [item[0] for item in commands] == [
        AverageCommand.RETRY, AverageCommand.CANCEL,
    ]
    assert [item[1].revision for item in commands] == [1, 2]


def test_average_cancel_does_not_auto_spin_across_pending_owners(
    tmp_path, monkeypatch,
) -> None:
    commands = []
    result = _result("CANCELLED", str((tmp_path / "average.nxs").resolve()))

    class Runner:
        def __init__(self, recipe):
            self.recipe = recipe

        def start(self, **_kwargs):
            return AverageScanPending(
                "headless-operation", 1,
                AveragePendingPhase.SOURCE_CLEANUP,
                "first source close retained",
            )

        def command(self, command, pending):
            commands.append((command, pending))
            if len(commands) == 1:
                return AverageScanPending(
                    "headless-operation", 2,
                    AveragePendingPhase.SOURCE_CLEANUP,
                    "second source close retained",
                )
            return result

        def close(self):
            return result

    monkeypatch.setattr(adapter, "AverageScanRunner", Runner)
    slot = OperationSlot()
    identity = slot.begin_average(
        _configuration(_source(tmp_path / "source")), result.target,
        stamp=OperationContextStamp(0),
    )
    first = _wait_update(
        slot, identity, lambda value: value.pending is not None,
    ).pending
    assert slot.cancel(identity)
    second = _wait_update(
        slot, identity,
        lambda value: value.pending is not None
        and value.pending.revision == 2,
    ).pending
    assert [item[0] for item in commands] == [AverageCommand.CANCEL]
    assert slot._worker is not None and slot._worker.is_alive()
    assert slot.cancel(identity)
    terminal = _wait_update(
        slot, identity, lambda value: value.terminal is not None,
    )
    assert terminal.terminal.status is OperationTerminalStatus.CANCELLED
    assert [item[0] for item in commands] == [
        AverageCommand.CANCEL, AverageCommand.CANCEL,
    ]


def test_average_close_workspace_advances_each_pending_revision_once(
    tmp_path, monkeypatch,
) -> None:
    commands = []
    result = _result("CANCELLED", str((tmp_path / "average.nxs").resolve()))

    class Runner:
        def __init__(self, recipe):
            self.recipe = recipe

        def start(self, **_kwargs):
            return AverageScanPending(
                "headless-operation", 1,
                AveragePendingPhase.SOURCE_CLEANUP,
                "first source close retained",
            )

        def command(self, command, pending):
            commands.append((command, pending))
            if len(commands) == 1:
                return AverageScanPending(
                    "headless-operation", 2,
                    AveragePendingPhase.SOURCE_CLEANUP,
                    "second source close retained",
                )
            return result

        def close(self):
            return result

    monkeypatch.setattr(adapter, "AverageScanRunner", Runner)
    slot = OperationSlot()
    identity = slot.begin_average(
        _configuration(_source(tmp_path / "source")), result.target,
        stamp=OperationContextStamp(0),
    )
    worker = slot._worker
    assert worker is not None and worker.daemon is False
    _wait_update(slot, identity, lambda value: value.pending is not None)
    receipts = []
    for _ in range(100):
        receipt = slot.close()
        receipts.append(receipt)
        if receipt.cleanup_status.value == "cleaned":
            break
        time.sleep(0.005)
    else:
        pytest.fail("close_workspace did not settle the Average worker")
    assert [item[0] for item in commands] == [
        AverageCommand.CLOSE, AverageCommand.CLOSE,
    ]
    assert [item[1].revision for item in commands] == [1, 2]
    assert receipts[-1].terminal is not None
    assert receipts[-1].terminal.status is OperationTerminalStatus.CANCELLED
    assert not worker.is_alive()
    assert slot._worker is None and not slot.owned


def test_average_page_projects_and_retries_the_exact_pending_token() -> None:
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace

    identity = OperationIdentity(7)
    pending = OperationPending(
        identity, 3, "source-cleanup", "source close retained",
    )
    calls = []

    class Slot:
        current_identity = identity

        @staticmethod
        def retry_average(owner, token):
            calls.append((owner, token))
            return True

    operations = WorkspaceOperationOwner()
    operations._slot = Slot()
    operations._average = AverageOperationState(
        identity, 0, "/detached/average.nxs", "entry"
    )
    page = SimpleNamespace(
        _workspace_operations=operations,
        _intents=SimpleNamespace(revision=0),
        _notice=lambda value: calls.append(("notice", value)),
        _refresh_shell=lambda: calls.append(("refresh",)),
        _ensure_timer=lambda: calls.append(("timer",)),
    )
    update = OperationUpdate(identity, pending=pending)
    assert (
        ScatteringWorkspace._consume_average_update(page, update)
        is WorkspaceRefreshEffect.CONTROLS
    )
    assert operations.average_pending is pending
    assert calls == [("notice", (
        "Average source cleanup pending; press Run to retry or Stop to cancel."
    ))]

    ScatteringWorkspace._run_action(page)
    assert operations.average_pending is None
    assert calls[1:] == [
        (identity, pending),
        ("notice", "Retrying Average cleanup…"),
        ("refresh",),
        ("timer",),
    ]


def test_average_pending_projection_enables_run_retry_and_stop(
    tmp_path, monkeypatch, qapp,
) -> None:
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
    from queue import Queue
    from xdart.gui.tabs.scattering.shell_values import (
        ShellCommand, ShellCommandKind,
    )

    page, _store = _page(tmp_path, monkeypatch)
    identity = OperationIdentity(77)
    pending = OperationPending(
        identity, 1, "source-cleanup", "source close retained",
    )
    slot = page._workspace_operations._slot
    with slot._lock:
        slot._identity = identity
        slot._pending = pending
        slot._command_queue = Queue()
    _set_average_state(
        page,
        identity,
        page._intents.revision,
        str(tmp_path / "average.nexus"),
        pending=pending,
    )
    try:
        page._refresh_shell()
        assert page._shell.run_controls.startButton.isEnabled()
        assert page._shell.run_controls.stopButton.isEnabled()
        assert "Average cleanup pending" in (
            page._shell.run_controls.readinessLabel.full_text()
        )
        page._handle_shell_command(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        assert page._workspace_operations.average_pending is None
        assert slot._command_queue.get_nowait() == "retry"
    finally:
        with slot._lock:
            slot._retire_locked()
        page._workspace_operations._average = None
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def _stub_integrators(monkeypatch):
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
    from xrd_tools.reduction import core
    calls = []
    def one(image, _ai, *, npt, normalization_factor=None, **_kwargs):
        value = float(np.nanmean(image)) / (normalization_factor or 1.0)
        calls.append(("1d", value))
        return IntegrationResult1D(np.arange(npt), np.full(npt, value), None, "q_A^-1")
    def two(image, _ai, *, npt_rad, npt_azim, normalization_factor=None, **_kwargs):
        value = float(np.nanmean(image)) / (normalization_factor or 1.0)
        calls.append(("2d", value))
        return IntegrationResult2D(
            np.arange(npt_rad), np.arange(npt_azim),
            np.full((npt_rad, npt_azim), value), None, "q_A^-1", "chi_deg",
        )
    monkeypatch.setattr(core, "integrate_1d", one)
    monkeypatch.setattr(core, "integrate_2d", two)
    return calls


def test_average_internal_source_failure_retains_exact_owner_until_close(
    tmp_path, monkeypatch,
) -> None:
    from xrd_tools.reduction import average as average_module
    from xrd_tools.sources import execution_graph

    source = _source(tmp_path / "source")
    target = tmp_path / "average.nxs"
    integrations = _stub_integrators(monkeypatch)
    real_close = execution_graph._AverageSourceReadWindow.close
    real_hold = average_module.AverageScanRunner._hold_source_cleanup
    real_runner = average_module.AverageScanRunner
    close_calls = []
    hold_calls = []
    runners = []

    def held_runtime_close(window):
        close_calls.append(window)
        # Preparation owns the first window. The runtime window then needs two
        # explicit cleanup commands before its real close can complete.
        if len(close_calls) in {2, 3}:
            raise OSError("injected source close hold")
        return real_close(window)

    def fail_first_hold(owner, *args, **kwargs):
        hold_calls.append(id(owner))
        if len(hold_calls) == 1:
            raise RuntimeError("injected source owner publication failure")
        return real_hold(owner, *args, **kwargs)

    def capture_runner(recipe):
        value = real_runner(recipe)
        runners.append(value)
        return value

    monkeypatch.setattr(
        execution_graph._AverageSourceReadWindow, "close", held_runtime_close,
    )
    monkeypatch.setattr(real_runner, "_hold_source_cleanup", fail_first_hold)
    monkeypatch.setattr(adapter, "AverageScanRunner", capture_runner)
    slot = OperationSlot()
    identity = slot.begin_average(
        _configuration(
            source,
            ReductionPlan(integration_1d=Integration1DPlan(npt=3)),
        ),
        target,
        stamp=OperationContextStamp(0),
    )
    assert identity is not None
    worker = slot._worker
    assert worker is not None and worker.daemon is False
    first = _wait_update(
        slot, identity,
        lambda value: value.pending is not None
        and value.pending.revision == 1,
    ).pending
    assert first.phase == "source-cleanup"
    frozen_integrations = tuple(integrations)

    assert slot.close().cleanup_status is CleanupStatus.CLEANUP_PENDING
    second = _wait_update(
        slot, identity,
        lambda value: value.pending is not None
        and value.pending.revision == 2,
    ).pending
    assert second.phase == "source-cleanup"
    assert tuple(integrations) == frozen_integrations

    receipts = []
    for _ in range(100):
        receipt = slot.close()
        receipts.append(receipt)
        if receipt.cleanup_status is CleanupStatus.CLEANED:
            break
        time.sleep(0.005)
    else:
        pytest.fail("close_workspace-shaped calls did not settle Average source")

    assert len(runners) == 1
    assert runners[0]._pending_revision == 2
    assert runners[0]._source_window is None
    assert len(close_calls) == 4
    assert close_calls[1] is close_calls[2] is close_calls[3]
    assert len(set(hold_calls)) == 1 and len(hold_calls) == 2
    assert tuple(integrations) == frozen_integrations
    assert receipts[-1].terminal is not None
    assert receipts[-1].terminal.status is OperationTerminalStatus.RETURNED
    assert receipts[-1].terminal.payload.disposition == "REFUSED"
    assert not worker.is_alive()


def test_average_internal_h23_failure_retains_revisioned_owner_until_close(
    tmp_path, monkeypatch,
) -> None:
    from xrd_tools.io.output_transaction import OutputTransaction
    from xrd_tools.reduction import average as average_module

    source = _source(tmp_path / "source")
    target = tmp_path / "average.nxs"
    integrations = _stub_integrators(monkeypatch)
    real_sink = average_module.NexusSink
    real_runner = average_module.AverageScanRunner
    real_retry = real_runner._retry_output_once
    real_commit = OutputTransaction.commit_stream
    sinks = []
    runners = []
    retries = []
    commits = []

    def capture_sink(*args, **kwargs):
        value = real_sink(*args, **kwargs)
        sinks.append(value)
        return value

    def capture_runner(recipe):
        value = real_runner(recipe)
        runners.append(value)
        return value

    def fail_internal_retry_once(owner):
        retries.append(id(owner))
        if len(retries) == 1:
            raise RuntimeError("injected retry owner failure")
        return real_retry(owner)

    def hold_commit_twice(owner, *args, **kwargs):
        commits.append(id(owner))
        if len(commits) <= 2:
            raise OSError("injected H23 commit hold")
        return real_commit(owner, *args, **kwargs)

    monkeypatch.setattr(average_module, "NexusSink", capture_sink)
    monkeypatch.setattr(real_runner, "_retry_output_once", fail_internal_retry_once)
    monkeypatch.setattr(OutputTransaction, "commit_stream", hold_commit_twice)
    monkeypatch.setattr(adapter, "AverageScanRunner", capture_runner)
    slot = OperationSlot()
    identity = slot.begin_average(
        _configuration(
            source,
            ReductionPlan(integration_1d=Integration1DPlan(npt=3)),
        ),
        target,
        stamp=OperationContextStamp(0),
    )
    assert identity is not None
    worker = slot._worker
    assert worker is not None and worker.daemon is False
    first = _wait_update(
        slot, identity,
        lambda value: value.pending is not None
        and value.pending.revision == 1,
    ).pending
    assert first.phase == "output-settlement"
    frozen_integrations = tuple(integrations)

    assert slot.close().cleanup_status is CleanupStatus.CLEANUP_PENDING
    second = _wait_update(
        slot, identity,
        lambda value: value.pending is not None
        and value.pending.revision == 2,
    ).pending
    assert second.phase == "output-settlement"
    assert tuple(integrations) == frozen_integrations

    assert slot.close().cleanup_status is CleanupStatus.CLEANUP_PENDING
    third = _wait_update(
        slot, identity,
        lambda value: value.pending is not None
        and value.pending.revision == 3,
    ).pending
    assert third.phase == "output-settlement"
    assert tuple(integrations) == frozen_integrations

    receipts = []
    for _ in range(100):
        receipt = slot.close()
        receipts.append(receipt)
        if receipt.cleanup_status is CleanupStatus.CLEANED:
            break
        time.sleep(0.005)
    else:
        pytest.fail("close_workspace-shaped calls did not settle Average H23")

    assert len(runners) == len(sinks) == 1
    assert runners[0]._pending_revision == 3
    assert runners[0]._source_window is None
    assert sinks[0]._transaction_owners is None
    assert len(set(commits)) == 1 and len(commits) == 3
    assert len(set(retries)) == 1 and len(retries) == 3
    assert tuple(integrations) == frozen_integrations
    assert receipts[-1].terminal is not None
    assert receipts[-1].terminal.status is OperationTerminalStatus.RETURNED
    assert receipts[-1].terminal.payload.disposition == "COMMITTED"
    assert not worker.is_alive()


def test_average_calibration_admission_cancel_is_a_cancelled_terminal(
    tmp_path, monkeypatch,
) -> None:
    entered = threading.Event()

    def cancelled_admission(_request, cancelled):
        entered.set()
        assert cancelled.wait(5)
        raise RuntimeError("admission cancelled")

    monkeypatch.setattr(adapter, "_average_calibration", cancelled_admission)
    slot = OperationSlot()
    identity = slot.begin_average(
        _configuration(_source(tmp_path)),
        tmp_path / "cancelled.nxs",
        stamp=OperationContextStamp(0),
    )
    assert identity is not None and entered.wait(5)
    assert slot.cancel(identity)

    update = _join(slot, identity)

    assert update.terminal.status is OperationTerminalStatus.CANCELLED
    assert update.terminal.diagnostic == ""


def test_average_page_accepts_only_exact_cancelled_payload_status_pairs(
    tmp_path, monkeypatch, qapp,
) -> None:
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _page

    page, store = _page(tmp_path, monkeypatch)
    target = str((tmp_path / "average.nxs").resolve())
    cancelled = _result("CANCELLED", target)
    refused = _result("REFUSED", target)
    committed = _result("COMMITTED", target)
    malformed = object.__new__(AverageScanResult)
    reloads = []
    clears = []
    catalogs = []
    notices = []
    monkeypatch.setattr(
        page._context_controller,
        "begin_browse",
        lambda *args, **kwargs: reloads.append((args, kwargs)),
    )
    # _clear_terminal_browse was RELOCATED, not removed: 1e3a4a1c "Extract
    # processed browser owner" moved it onto the browser owner as
    # ProcessedBrowser.retire_terminal (processed_browser.py:968), which
    # page.py:2512 calls after a successful Average reload.  Re-point the
    # sentinel there rather than dropping the coverage.
    monkeypatch.setattr(
        page._processed_browser,
        "retire_terminal",
        lambda *args, **kwargs: clears.append(None) or False,
    )
    monkeypatch.setattr(
        page, "_request_browser_catalog", lambda: catalogs.append(None),
    )
    monkeypatch.setattr(page, "_notice", notices.append)
    terminal_owner = page._processed_browser.terminal_request
    rows = (
        (OperationTerminalStatus.CANCELLED, None, "Average cancelled."),
        (
            OperationTerminalStatus.CANCELLED,
            cancelled,
            "Average cancelled.",
        ),
        (
            OperationTerminalStatus.RETURNED,
            cancelled,
            "Average failed: invalid terminal result",
        ),
        (
            OperationTerminalStatus.CANCELLED,
            refused,
            "Average failed: invalid terminal result",
        ),
        (
            OperationTerminalStatus.CANCELLED,
            committed,
            "Average failed: invalid terminal result",
        ),
        (
            OperationTerminalStatus.CANCELLED,
            malformed,
            "Average failed: invalid terminal result",
        ),
    )
    try:
        for token, (status, payload, expected_notice) in enumerate(rows, 1):
            identity = OperationIdentity(100 + token)
            _set_average_state(page, identity, store.revision, target)
            assert page._consume_average_update(OperationUpdate(
                identity,
                terminal=OperationTerminal(identity, status, payload=payload),
            )) is WorkspaceRefreshEffect.CONTROLS
            assert notices[-1] == expected_notice
            assert page._workspace_operations.average_state is None
        assert reloads == clears == catalogs == []
        # ...and the terminal the browser owns is untouched.  This holds on the
        # STATE rather than on one method name, so it survives the next
        # relocation the way the sentinel above did not.
        assert page._processed_browser.terminal_request is terminal_owner
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_average_cancellation_drain_refreshes_controls_without_science(
    tmp_path, monkeypatch, qapp,
) -> None:
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _page

    page, store = _page(tmp_path, monkeypatch)
    identity = OperationIdentity(170)
    update = OperationUpdate(
        identity,
        terminal=OperationTerminal(
            identity, OperationTerminalStatus.CANCELLED,
        ),
    )

    class TerminalSlot:
        current_identity = identity
        owned = False

        @staticmethod
        def observe_stamp(_stamp):
            return None

        @staticmethod
        def poll(candidate):
            return update if candidate is identity else None

    operations = _set_average_state(
        page,
        identity,
        store.revision,
        str((tmp_path / "cancelled.nxs").resolve()),
    )
    original_slot = operations._slot
    operations._slot = TerminalSlot()
    refreshes = []
    monkeypatch.setattr(
        page, "_refresh_shell", lambda **kwargs: refreshes.append(kwargs),
    )
    monkeypatch.setattr(
        page,
        "_refresh_event_shell",
        lambda **_kwargs: pytest.fail("Average cancellation repainted science"),
    )
    try:
        page._drain_executor()
        assert refreshes == [{
            "preserve_display": True,
            "preserve_scientific": True,
        }]
        assert operations.average_state is None
    finally:
        operations._slot = original_slot
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_average_accepts_fixed_eiger_config_without_redundant_max_shape(
    tmp_path,
) -> None:
    poni = tmp_path / "fixed-eiger.poni"
    poni.write_text(
        "poni_version: 2.1\n"
        "Detector: Eiger4M\n"
        'Detector_config: {"orientation": 3}\n'
        "Distance: 0.13846912503056505\n"
        "Poni1: 0.1785781587133659\n"
        "Poni2: 0.010010979828214216\n"
        "Rot1: -0.004247048916341726\n"
        "Rot2: 0.004266506815595174\n"
        "Rot3: 0.0\n"
        "Wavelength: 7.293188143129427e-11\n"
    )

    state = adapter._average_calibration(SimpleNamespace(
        poni_file=str(poni), mask_file="",
    ))
    assert state.detector_id == "Eiger4M"
    assert dict(state.detector_config) == {"orientation": 3}
    assert state.values is not None

    from xrd_tools.core import PONI
    from xrd_tools.core.geometry import DetectorCalibration

    values = state.values
    calibration = DetectorCalibration(
        PONI(
            values.dist,
            values.poni1,
            values.poni2,
            values.rot1,
            values.rot2,
            values.rot3,
            values.wavelength_m,
            state.detector_id,
        ),
        dict(state.detector_config),
    )
    detector = adapter.detector_calibration_to_integrator(
        calibration,
    ).detector
    assert tuple(detector.shape) == tuple(detector.max_shape) == (2167, 2070)
    assert detector.pixel1 == detector.pixel2 == 75e-6
    assert int(detector.orientation) == 3


def test_average_accepts_variable_binning_detector_shape(tmp_path) -> None:
    poni = tmp_path / "rayonix.poni"
    poni.write_text(
        "poni_version: 2.1\n"
        "Detector: RayonixMx225\n"
        'Detector_config: {"pixel1": 7.3242e-05, "pixel2": 7.3242e-05, "orientation": 3}\n'
        "Distance: 0.17939120815373186\n"
        "Poni1: 0.22358886498597383\n"
        "Poni2: 0.11322186872581771\n"
        "Rot1: 0.0005767109589810191\n"
        "Rot2: 0.000644934300335509\n"
        "Rot3: 0.0\n"
        "Wavelength: 9.762535309700809e-11\n"
    )

    state = adapter._average_calibration(SimpleNamespace(
        poni_file=str(poni), mask_file="",
    ))
    assert state.detector_id == "RayonixMx225"

    from xrd_tools.core import PONI
    from xrd_tools.core.geometry import DetectorCalibration

    values = state.values
    assert values is not None
    detector = adapter.detector_calibration_to_integrator(
        DetectorCalibration(
            PONI(
                values.dist,
                values.poni1,
                values.poni2,
                values.rot1,
                values.rot2,
                values.rot3,
                values.wavelength_m,
                state.detector_id,
            ),
            dict(state.detector_config),
        ),
    ).detector
    assert tuple(detector.shape) == (3072, 3072)
    assert tuple(detector.max_shape) == (6144, 6144)
    assert all(current <= maximum for current, maximum in
               zip(detector.shape, detector.max_shape, strict=True))


def test_average_active_background_is_a_stable_pre_source_refusal(
    tmp_path, monkeypatch,
) -> None:
    from xrd_tools.reduction import average as average_module
    from xrd_tools.reduction.background import FrameBackgroundPlan
    from xrd_tools.session.experiment_state import CalibrationState

    source = _source(tmp_path)
    target = tmp_path / "active-background.nxs"
    effects = []

    def forbidden(name):
        return lambda *_args, **_kwargs: (
            effects.append(name), pytest.fail(f"active background reached {name}")
        )[1]

    monkeypatch.setattr(
        adapter, "_average_calibration",
        lambda *_args, **_kwargs: CalibrationState(),
    )
    for name in (
        "_source_from_recipe", "qualify_source_execution_graph",
        "open_source_execution_graph", "_average_allocation",
        "capture_target_snapshot", "_background_fact",
        "resolve_frame_background",
    ):
        monkeypatch.setattr(average_module, name, forbidden(name))
    slot = OperationSlot()
    identity = slot.begin_average(
        _configuration(
            source,
            background=FrameBackgroundPlan(
                mode="Single BG File",
                locator=str(source.options["selected_file"]),
            ),
        ),
        target,
        stamp=OperationContextStamp(0),
    )
    assert identity is not None
    update = _join(slot, identity)
    result = update.terminal.payload
    assert update.terminal.status is OperationTerminalStatus.RETURNED
    assert (result.disposition, result.diagnostic_code) == (
        "REFUSED", "AVERAGE_BACKGROUND_AGGREGATE_PROVENANCE_UNSUPPORTED",
    )
    assert effects == [] and not target.exists()


@pytest.fixture
def qapp():
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_average_page_dispatches_one_revision_checked_canonical_freeze(
    tmp_path, monkeypatch, qapp,
) -> None:
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
    from xrd_tools.sources import selection

    source = _source(tmp_path / "source")
    intent = RunIntent(
        source_spec=source,
        processing_mode="Int 2D",
        output_mode="Overwrite",
        max_cores=3,
        project_root=str(tmp_path),
        save_path=str(tmp_path / "processed"),
        run_options={"series_average": True},
    )
    store = RunIntentStore(intent)
    page, _store = _page(tmp_path, monkeypatch, store=store)
    freezes = []
    dispatches = []
    real_freeze = store.freeze

    def freeze(**kwargs):
        freezes.append(kwargs)
        return real_freeze(**kwargs)

    def begin(configuration, target, **kwargs):
        dispatches.append((configuration, target, kwargs))
        return OperationIdentity(901)

    try:
        monkeypatch.setattr(store, "freeze", freeze)
        monkeypatch.setattr(
            selection,
            "image_series_spec",
            lambda *_args, **_kwargs: pytest.fail(
                "Average enumerated TIFF members on the GUI thread"
            ),
        )
        monkeypatch.setattr(page._workspace_operations._slot, "begin_average", begin)
        assert page._source_selection.observation is None
        snapshot = store.snapshot()

        page._average_action(snapshot)

        assert freezes == [{
            "expected_revision": snapshot.revision,
            "gi_motor_choices": None,
        }]
        assert len(dispatches) == 1
        configuration, target, kwargs = dispatches[0]
        assert configuration.generation == 1
        assert configuration.max_cores == 3
        assert configuration.thaw_source_spec() == source
        assert target.endswith("/processed/scan.nexus")
        assert kwargs == {
            "entry": "entry",
            "stamp": OperationContextStamp(snapshot.revision),
        }
        assert store.snapshot().thaw().generation == 1
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_average_private_request_builds_recipe_and_enumerates_only_on_worker(
    tmp_path, monkeypatch
) -> None:
    from xdart.gui.tabs.scattering.contracts import AcceptedScientificAssets
    from xdart.gui.tabs.scattering import output_preflight
    from xrd_tools.session.experiment_state import (
        CalibrationState, FactStatus, MaskState, PoniValues,
    )
    source = _source(tmp_path)
    mutable_options = copy.deepcopy(dict(source.options))
    mutable_options["files"] = list(mutable_options["files"])
    source = SourceSpec(source.uri, source.kind, options=mutable_options)
    reduction_extra = {"nested": [1, {"value": 2}], "enabled_modes_1d": ["q"]}
    reduction = ReductionPlan(
        integration_1d=Integration1DPlan(npt=7, extra=reduction_extra),
        integration_2d=Integration2DPlan(npt_rad=4, npt_azim=3),
    )
    poni_path = str(tmp_path / "accepted.poni"); mask_path = str(tmp_path / "accepted.npy")
    config = {"max_shape": [5, 7], "orientation": 3}
    assets = AcceptedScientificAssets(
        (0.2, 0.0002, 0.0003, 0.0, 0.0, 0.0, 1e-10, "Pilatus300kw"),
        "|b1", (2, 2), bytes((0, 1, 0, 0)), "p" * 64, "m" * 64,
        json.dumps(config, sort_keys=True, separators=(",", ":")),
    )
    calibration = CalibrationState(
        PoniValues(*assets.poni_values[:7]), assets.poni_values[7], config,
        "", assets.poni_sha256, poni_path,
        MaskState(mask_path, assets.mask_sha256, assets.mask_dtype,
                  assets.mask_shape, FactStatus.PRESENT), FactStatus.PRESENT,
    )
    from xrd_tools.core.geometry.diffractometer import DetectorCalibration
    from xrd_tools.integrate import calibration as calibration_module
    real_reconstruct = calibration_module.detector_calibration_to_integrator
    accepted_calibration = assets.detector_calibration
    accepted_detector = real_reconstruct(accepted_calibration).detector
    def detector_truth(value, detector):
        return (
            type(detector), tuple(detector.shape), tuple(detector.max_shape),
            detector.pixel1, detector.pixel2, int(detector.orientation),
            DetectorCalibration(value.poni, detector.get_config()).to_json(),
        )
    accepted_truth = detector_truth(accepted_calibration, accepted_detector)
    target = tmp_path / "average.nxs"; source_base = tmp_path / "source-base"
    source_copy = SourceSpec(source.uri, source.kind, source.metadata_uri, source.entry,
                             copy.deepcopy(dict(source.options)))
    reduction_copy = copy.deepcopy(reduction)
    configuration = _configuration(
        source_copy, reduction_copy, poni_file=poni_path, mask_file=mask_path,
        batch_mode=True, project_root=source_base, max_cores=1,
    )
    expected = AverageScanRecipe(
        configuration.thaw_source_spec(), target,
        reduction_copy, entry="entry",
        source_base=source_base, output_mode="Overwrite", live_mode=False,
        save_xye=False, batch_mode=True, calibration=calibration,
        numeric_metadata_keys=("I0",), invariant_metadata_keys=("temperature",),
        resource_requests={"workers": 1},
    )
    from xrd_tools.reduction import average as average_module
    from xrd_tools.sources import selection
    from xrd_tools.sources.image import TiffSeriesSource
    entered, release = threading.Event(), threading.Event(); calls = []; effects = []
    def observed(name, function):
        def call(*args, **kwargs):
            effects.append((name, threading.current_thread().name))
            return function(*args, **kwargs)
        return call
    for owner in (selection, average_module):
        for name in ("single_image_spec", "image_series_spec"):
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, observed(name, getattr(owner, name)))
    real_stat, real_open, real_iterdir = Path.stat, Path.open, Path.iterdir
    monkeypatch.setattr(Path, "stat", observed("stat", real_stat))
    monkeypatch.setattr(Path, "open", observed("open", real_open))
    monkeypatch.setattr(Path, "iterdir", observed("iterdir", real_iterdir))
    monkeypatch.setattr(average_module, "qualify_source_execution_graph",
                        lambda *_a, **_k: pytest.fail("qualification preceded public runner"))
    monkeypatch.setattr(average_module, "resolve_session_policy",
                        lambda *_a, **_k: pytest.fail("allocation preceded public runner"))
    monkeypatch.setattr(TiffSeriesSource, "metadata_for",
                        lambda *_a, **_k: pytest.fail("metadata preceded public runner"))
    real_recipe = AverageScanRecipe
    def load(intent, *, cancelled):
        assert callable(cancelled) and not cancelled()
        calls.append(("assets", threading.current_thread().name,
                      intent.poni_file, intent.mask_file))
        entered.set(); assert release.wait(5)
        return assets
    def reconstruct(value):
        result = real_reconstruct(value)
        calls.append(("reconstruct", threading.current_thread().name,
                      id(value), value.to_json(), detector_truth(value, result.detector)))
        return result
    def Recipe(*args, **kwargs):
        value = real_recipe(*args, **kwargs)
        calls.append(("recipe", threading.current_thread().name, value))
        return value
    def run(recipe, **kwargs):
        calls.append(("run", threading.current_thread().name, recipe, kwargs))
        return _result("REFUSED", recipe.target)
    monkeypatch.setattr(output_preflight, "_load_scientific_assets", load)
    monkeypatch.setattr(calibration_module, "detector_calibration_to_integrator", reconstruct)
    monkeypatch.setattr(adapter, "detector_calibration_to_integrator", reconstruct,
                        raising=False)
    monkeypatch.setattr(adapter, "AverageScanRecipe", Recipe)
    monkeypatch.setattr(adapter, "AverageScanRunner", _runner_from_callable(run))
    slot = OperationSlot()
    identity = slot.begin_average(
        configuration, target, entry="entry",
        numeric_metadata_keys=("I0",), invariant_metadata_keys=("temperature",),
        stamp=OperationContextStamp(0),
    )
    assert identity is not None and entered.wait(5)
    assert type(slot._frozen).__name__ == "_AverageRequest"
    assert tuple(item.name for item in fields(type(slot._frozen))) == (
        "configuration", "target", "entry", "numeric_metadata_keys",
        "invariant_metadata_keys",
    )
    assert slot._frozen.configuration is configuration
    assert calls == [("assets", calls[0][1], poni_path, mask_path)]
    assert calls[0][1].startswith("scattering-operation-")
    assert effects == []
    mutable_options["files"].append(str(tmp_path / "late.tif"))
    reduction_extra["nested"][1]["value"] = 99
    reduction_extra["enabled_modes_1d"].append("chi")
    reduction.integration_1d.npt = 99
    cancel_event = slot._cancel_event; release.set()
    update = _join(slot, identity)
    assert update.terminal.status is OperationTerminalStatus.RETURNED
    assert [row[0] for row in calls] == ["assets", "reconstruct", "reconstruct", "recipe", "run"]
    assert all(row[1].startswith("scattering-operation-") for row in calls)
    assert calls[1][2] != calls[2][2]
    assert calls[1][3:] == calls[2][3:] == (accepted_calibration.to_json(), accepted_truth)
    assert calls[3][2] == expected and calls[3][2].calibration == calibration
    assert calls[3][2].calibration.detector_config["max_shape"] == (5, 7)
    assert calls[3][2].calibration.detector_config["orientation"] == 3
    assert calls[3][2].calibration.value_fingerprint == ""
    assert not hasattr(calls[3][2].calibration.mask, "values")
    assert calls[4][2] is calls[3][2]
    assert calls[4][3]["cancel_token"] is cancel_event
    assert callable(calls[4][3]["publication_gate"])
    assert effects and all(thread.startswith("scattering-operation-")
                           for _name, thread in effects)

    bad_values = list(assets.poni_values); bad_values[6] = 0.0
    detector_config = copy.deepcopy(accepted_detector.get_config())
    mutants = []
    foreign = type("ForeignDetector", (), {})()
    for name in ("shape", "max_shape", "pixel1", "pixel2", "orientation"):
        setattr(foreign, name, getattr(accepted_detector, name))
    foreign.get_config = lambda: copy.deepcopy(detector_config); mutants.append(foreign)
    for name, value in (("shape", (4, 7)), ("max_shape", (6, 7)),
                        ("_pixel1", accepted_detector.pixel1 * 2),
                        ("_pixel2", accepted_detector.pixel2 * 2), ("_orientation", 2)):
        mutant = copy.copy(accepted_detector); setattr(mutant, name, value)
        mutant.get_config = lambda config=detector_config: copy.deepcopy(config)
        mutants.append(mutant)
    mutant = copy.copy(accepted_detector)
    mutant.get_config = lambda: {**copy.deepcopy(detector_config), "orientation": 2}; mutants.append(mutant)
    reconstructed = tuple(type("Reconstructed", (), {"detector": value})()
                          for value in mutants)
    def divergent(result):
        seen = []
        def rebuild(value):
            seen.append(1)
            if len(seen) == 1: return real_reconstruct(value)
            if len(seen) == 2: return result
            return pytest.fail("unexpected third reconstruction")
        return rebuild
    huge = replace(assets, poni_detector_config_json=json.dumps(
        {"orientation": 3, "payload": "x" * 65_536}, sort_keys=True, separators=(",", ":")))
    missing_poni = replace(assets, poni_values=None, poni_sha256=None,
                           poni_detector_config_json=None)
    missing_mask = replace(assets, mask_dtype=None, mask_shape=None,
                           mask_bytes=None, mask_sha256=None)
    def target_bytes():
        with real_open(target, "rb") as stream: return stream.read()
    target.write_bytes(b"prior-average-target"); target_before = target_bytes()
    rows = (
        (AcceptedScientificAssets(tuple(bad_values), assets.mask_dtype, assets.mask_shape,
         assets.mask_bytes, assets.poni_sha256, assets.mask_sha256,
         assets.poni_detector_config_json), real_reconstruct,
         "AVERAGE_CALIBRATION_UNREPRESENTABLE", poni_path, mask_path, 0),
        (replace(assets, mask_shape=(0, 2), mask_bytes=b""), real_reconstruct,
         "AVERAGE_MASK_UNREPRESENTABLE", poni_path, mask_path, 0),
        *((assets, divergent(result),
           "AVERAGE_DETECTOR_CONFIG_UNREPRESENTABLE", poni_path, mask_path, 2)
          for result in reconstructed),
        (huge, None, "AVERAGE_DETECTOR_CONFIG_UNREPRESENTABLE", poni_path, mask_path, 0),
        (missing_poni, None, "AVERAGE_CALIBRATION_UNAVAILABLE", poni_path, mask_path, 0),
        (missing_mask, None, "AVERAGE_MASK_UNAVAILABLE", poni_path, mask_path, 0),
        (OSError("AVERAGE_ASSET_UNSTABLE"), None, "AVERAGE_ASSET_UNSTABLE",
         poni_path, mask_path, 0),
    )
    for loaded, rebuild, diagnostic, requested_poni, requested_mask, expected_rebuilds in rows:
        with monkeypatch.context() as patch:
            effects = []
            def row_load(_intent, *, cancelled):
                assert callable(cancelled) and not cancelled()
                effects.append("assets")
                if isinstance(loaded, BaseException): raise loaded
                return loaded
            def row_reconstruct(value):
                effects.append("reconstruct")
                return pytest.fail("reconstruction reached after refusal") if rebuild is None else rebuild(value)
            patch.setattr(output_preflight, "_load_scientific_assets", row_load)
            patch.setattr(calibration_module, "detector_calibration_to_integrator", row_reconstruct)
            patch.setattr(adapter, "detector_calibration_to_integrator", row_reconstruct, raising=False)
            patch.setattr(adapter, "AverageScanRecipe", lambda *_a, **_k: pytest.fail("recipe constructed after refusal"))
            patch.setattr(adapter, "AverageScanRunner", lambda *_a, **_k: pytest.fail("runner reached after refusal"))
            refused = OperationSlot(); refused_identity = refused.begin_average(
                _configuration(
                    source_copy, reduction_copy, poni_file=requested_poni,
                    mask_file=requested_mask,
                ),
                target, stamp=OperationContextStamp(0),
            )
            assert refused_identity is not None
            refused_update = _join(refused, refused_identity)
        assert refused_update.terminal.status is OperationTerminalStatus.FAILED
        assert refused_update.terminal.diagnostic.endswith(f": {diagnostic}")
        assert effects == ["assets"] + ["reconstruct"] * expected_rebuilds
        assert target_bytes() == target_before
    from xrd_tools.sources import DirectorySourceSpec
    directory = OperationSlot(); before = (tuple(calls), tuple(effects), target_bytes())
    with monkeypatch.context() as patch:
        patch.setattr(
            output_preflight,
            "_load_scientific_assets",
            lambda *_args, **_kwargs: replace(
                assets,
                poni_values=None,
                mask_dtype=None,
                mask_shape=None,
                mask_bytes=None,
                poni_sha256=None,
                mask_sha256=None,
                poni_detector_config_json=None,
            ),
        )
        directory_identity = directory.begin_average(
            _configuration(DirectorySourceSpec(tmp_path), reduction_copy), target,
            stamp=OperationContextStamp(0),
        )
        assert directory_identity is not None
        directory_update = _join(directory, directory_identity)
    assert directory_update.terminal.status is OperationTerminalStatus.FAILED
    assert (tuple(calls), tuple(effects), target_bytes()) == before


def test_average_publication_gate_linearizes_cancel_wins_and_seal_wins(
    tmp_path, monkeypatch
) -> None:
    from xrd_tools.io import get_average_finite_counts
    from xrd_tools.io.output_transaction import OutputTransaction
    _stub_integrators(monkeypatch)
    real_run = _run_terminal_average

    def exercise(name, *, cancel_first=False, throw=False):
        root = tmp_path / name; source = _source(root)
        target = root / "average.nxs"
        with h5py.File(target, "w") as handle:
            handle.create_dataset("prior", data=np.arange(5, dtype="<i4"))
        before = target.read_bytes()
        before_gate = threading.Event(); after_gate = threading.Event()
        enter_gate = threading.Event(); leave_gate = threading.Event()
        progress_waiting = threading.Event(); progress_release = threading.Event()
        terminal_waiting = threading.Event(); terminal_release = threading.Event()
        gate_results = []; commits = []; progress_seen = []
        real_commit = OutputTransaction.commit_stream
        real_body = OperationSlot._run_average_request
        def commit(owner, *args, **kwargs):
            commits.append(id(owner)); return real_commit(owner, *args, **kwargs)
        def body(owner, *args, **kwargs):
            terminal = real_body(owner, *args, **kwargs)
            terminal_waiting.set(); assert terminal_release.wait(5)
            return terminal
        def run(recipe, **kwargs):
            slot_gate = kwargs["publication_gate"]
            progress_cb = kwargs["progress_cb"]
            def held_progress(value):
                if not progress_seen:
                    progress_seen.append(value); progress_waiting.set()
                    assert progress_release.wait(5)
                return progress_cb(value)
            def held_gate():
                before_gate.set(); assert enter_gate.wait(5)
                if throw:
                    gate_results.append("throw"); after_gate.set()
                    raise OSError("publication gate exploded")
                accepted = slot_gate(); gate_results.append(accepted)
                after_gate.set(); assert leave_gate.wait(5)
                return accepted
            return real_run(recipe, **{**kwargs, "progress_cb": held_progress,
                                       "publication_gate": held_gate})
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "AverageScanRunner", _runner_from_callable(run))
            patch.setattr(OutputTransaction, "commit_stream", commit)
            patch.setattr(OperationSlot, "_run_average_request", body)
            slot = OperationSlot()
            identity = slot.begin_average(
                _configuration(source, ReductionPlan(
                    integration_1d=Integration1DPlan(npt=3),
                )), target, stamp=OperationContextStamp(0),
            )
            assert identity is not None and progress_waiting.wait(5)
            assert not slot._cancel_sealed; progress_release.set()
            assert before_gate.wait(5)
            if cancel_first:
                assert slot.cancel(identity); enter_gate.set()
            else:
                enter_gate.set(); assert after_gate.wait(5)
                if not throw:
                    assert not slot.cancel(identity)
            leave_gate.set()
            assert terminal_waiting.wait(5)
            assert len(commits) == (1 if not cancel_first and not throw else 0)
            held = slot.poll(identity)
            assert held is None or held.terminal is None
            terminal_release.set()
            update = _join(slot, identity)
        assert len(gate_results) == 1
        if cancel_first:
            assert gate_results == [False]
            assert update.terminal.status is OperationTerminalStatus.CANCELLED
            assert target.read_bytes() == before
            with pytest.raises((KeyError, ValueError)):
                get_average_finite_counts(target)
        elif throw:
            assert gate_results == ["throw"]
            assert update.terminal.status is OperationTerminalStatus.FAILED
            assert "publication gate exploded" in update.terminal.diagnostic
            assert target.read_bytes() == before
        else:
            assert gate_results == [True]
            assert update.terminal.status is OperationTerminalStatus.RETURNED
            assert update.terminal.payload.disposition == "COMMITTED"
            # Counts live in the ARTIFACT, not the anchor.  `target` is only
            # the request, and the read_bytes() checks above already pin that
            # the anchor is never written at all.
            artifact = Path(update.terminal.payload.target)
            assert artifact != target and artifact.exists()
            assert get_average_finite_counts(artifact).evidence.contributor_extent == 2

    exercise("cancel-wins", cancel_first=True)
    exercise("seal-wins")
    exercise("gate-throws", throw=True)
    notebook = tmp_path / "notebook"
    result = real_run(AverageScanRecipe(
        _source(notebook), notebook / "average.nxs",
        ReductionPlan(integration_1d=Integration1DPlan(npt=3)),
    ), publication_gate=None)
    assert result.disposition == "COMMITTED"


def test_direct_and_gui_average_match_after_fresh_reopen(tmp_path, monkeypatch) -> None:
    from xrd_tools.core.provenance import read_provenance
    from xrd_tools.io import get_1d, get_2d, get_average_finite_counts, get_metadata
    from xrd_tools.reduction import average as average_module
    calls = _stub_integrators(monkeypatch)
    root = tmp_path / "parity"; source = _source(root)
    for index, path in enumerate(source.options["files"], 1):
        Path(path).with_suffix(".txt").write_text(
            f"# Counters\nI0 = {index}.0\n# Motors\n\n"
            "User: p36, time: Mon Jan 15 10:30:00 2024  # Temp\n"
        )
    source = SourceSpec(source.uri, source.kind, options={
        **dict(source.options), "metadata_format": "txt",
    })
    reduction = ReductionPlan(
        integration_1d=Integration1DPlan(npt=4, monitor_key="I0"),
        integration_2d=Integration2DPlan(
            npt_rad=3, npt_azim=2, monitor_key="I0",
        ),
    )
    direct_target = root / "direct.nxs"; gui_target = root / "gui.nxs"
    direct = _run_terminal_average(AverageScanRecipe(
        source, direct_target, reduction, numeric_metadata_keys=("I0",),
    ))
    assert direct.disposition == "COMMITTED"
    slot = OperationSlot()
    identity = slot.begin_average(
        _configuration(source, reduction), gui_target,
        numeric_metadata_keys=("I0",),
        stamp=OperationContextStamp(0),
    )
    assert identity is not None
    update = _join(slot, identity)
    gui = update.terminal.payload
    assert update.terminal.status is OperationTerminalStatus.RETURNED
    assert gui.disposition == "COMMITTED"
    # The successor route writes <anchor stem>.average-<version>.nexus, never
    # the anchor.  direct_target/gui_target are the REQUESTS; read the artifacts
    # the results name.
    direct_artifact = Path(direct.target); gui_artifact = Path(gui.target)
    assert direct_artifact != direct_target and gui_artifact != gui_target
    assert direct_artifact.exists() and gui_artifact.exists()
    assert direct.science_identity == gui.science_identity
    assert direct.operation_identity != gui.operation_identity
    assert direct.metadata_denominators == gui.metadata_denominators == (("I0", 2),)
    for getter, fields_to_compare in (
        (get_1d, ("q", "intensity", "sigma", "q_unit", "frames")),
        (get_2d, ("q", "chi", "intensity", "q_unit", "chi_unit", "frames")),
    ):
        direct_value = getter(direct_artifact, frame=1)
        gui_value = getter(gui_artifact, frame=1)
        for name in fields_to_compare:
            left, right = getattr(direct_value, name), getattr(gui_value, name)
            if isinstance(left, np.ndarray):
                np.testing.assert_allclose(left, right, equal_nan=True)
            else:
                assert left == right
    np.testing.assert_array_equal(get_average_finite_counts(direct_artifact).values,
                                  get_average_finite_counts(gui_artifact).values)
    assert tuple(average_module.iter_average_contributors(direct_artifact)) == tuple(average_module.iter_average_contributors(gui_artifact))
    direct_provenance = read_provenance(direct_artifact)["config"]["average_scan_v1"]
    gui_provenance = read_provenance(gui_artifact)["config"]["average_scan_v1"]
    operation_identities = tuple(value.pop("operation_identity") for value in (direct_provenance, gui_provenance))
    assert direct_provenance == gui_provenance
    assert operation_identities == (direct.operation_identity, gui.operation_identity)
    assert operation_identities[0] != operation_identities[1]
    direct_scan = get_metadata(direct_artifact)["scan_data"]
    gui_scan = get_metadata(gui_artifact)["scan_data"]
    assert set(direct_scan) == set(gui_scan) and "I0" in direct_scan
    for name in direct_scan:
        np.testing.assert_allclose(direct_scan[name], gui_scan[name], equal_nan=True)
    np.testing.assert_allclose(direct_scan["I0"], [1.5])
    import hashlib
    for result, target in ((direct, direct_artifact), (gui, gui_artifact)):
        commit = result.commit_identity
        assert (commit is not None and commit.target == str(target.resolve())
                and type(commit.ordinal) is int and commit.ordinal > 0)
        assert commit.size == target.stat().st_size and commit.digest == hashlib.sha256(target.read_bytes()).hexdigest()
    assert direct.commit_identity.digest != gui.commit_identity.digest
    assert len(calls) == 4


def test_average_stop_stale_close_and_single_slot_truth(tmp_path, monkeypatch) -> None:
    entered, release = threading.Event(), threading.Event()

    def run(_recipe, *, cancel_token, **_kwargs):
        entered.set(); release.wait(2)
        return _result("CANCELLED" if cancel_token.is_set() else "REFUSED")

    monkeypatch.setattr(adapter, "AverageScanRunner", _runner_from_callable(run))
    slot = OperationSlot()
    source = _source(tmp_path)
    configuration = _configuration(source)
    identity = slot.begin_average(
        configuration, tmp_path / "one.nxs",
        stamp=OperationContextStamp(1, "context", 2),
    )
    assert identity is not None and entered.wait(2)
    assert slot.begin_average(
        configuration, tmp_path / "two.nxs",
        stamp=OperationContextStamp(1, "context", 2),
    ) is None
    slot.observe_stamp(OperationContextStamp(2, "context", 2))
    assert slot.cancel(identity)
    pending = slot.close()
    assert pending.cleanup_status.value == "cleanup_pending"
    assert pending.identity is identity and not pending.cancel_accepted
    release.set(); worker = slot._worker; assert worker is not None
    worker.join(5); assert not worker.is_alive()
    cleaned = slot.close()
    assert cleaned.cleanup_status.value == "cleaned"
    assert cleaned.terminal.status is OperationTerminalStatus.CANCELLED
    assert cleaned.stale and not slot.owned
    assert slot.close() is cleaned


def test_average_terminal_projection_preserves_typed_truth_and_reload_boundary(
    tmp_path, monkeypatch, qapp
) -> None:
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
    from xdart.gui.tabs.scattering.operation_values import OperationUpdate
    source = _source(tmp_path / "source")
    target = tmp_path / "committed.nxs"; reduction = ReductionPlan(integration_1d=Integration1DPlan(npt=3))
    _stub_integrators(monkeypatch)
    committed = _run_terminal_average(AverageScanRecipe(source, target, reduction))
    assert committed.disposition == "COMMITTED"
    base = _result("REFUSED", str(target.resolve()))
    malformed_rows = (
        (committed, {"committed_labels": ()}), (committed, {"finite_counts": None}),
        (committed, {"h23_phase": None}), (committed, {"commit_identity": None}),
        (committed, {"contributor_extent": 1}), (base, {"finite_counts": committed.finite_counts}),
        (base, {"logical_labels": ()}), (base, {"logical_labels": (True,)}), (committed, {"committed_labels": (True,)}),
        (base, {"metadata_denominators": (("", 0),)}),
    )
    for owner, changes in malformed_rows:
        with pytest.raises(ValueError, match="result contract"): replace(owner, **changes)

    def scheduled(result, *, verification=None):
        with monkeypatch.context() as patch:
            patch.setattr(
                adapter, "AverageScanRunner",
                _runner_from_callable(lambda *_a, **_k: result),
            )
            slot = OperationSlot()
            identity = slot.begin_average(
                _configuration(source, reduction), target,
                stamp=OperationContextStamp(0),
            )
            assert identity is not None
            update = _join(slot, identity)
        expected_status = {"COMMITTED": OperationTerminalStatus.RETURNED,
            "REFUSED": OperationTerminalStatus.RETURNED, "CANCELLED": OperationTerminalStatus.CANCELLED,
            "ABORTED": OperationTerminalStatus.FAILED}[result.disposition]
        expected_diagnostic = f"{result.diagnostic_code}: {result.diagnostic}" if result.disposition == "ABORTED" else ""
        if verification is not None:
            expected_status = OperationTerminalStatus.FAILED
            expected_diagnostic = f"AVERAGE_COMMIT_VERIFICATION_FAILED: {verification}"
        assert update.terminal.status is expected_status
        assert update.terminal.payload is result
        assert update.terminal.diagnostic == expected_diagnostic
        return identity, update

    page, store = _page(tmp_path, monkeypatch)
    reloads = []; catalog = []; notices = []
    refreshes = []
    monkeypatch.setattr(
        page._context_controller,
        "begin_browse",
        lambda value, *, terminal_commit_identity=None, source_root=None:
            reloads.append((value, terminal_commit_identity, source_root)),
    )
    monkeypatch.setattr(page, "_request_browser_catalog", lambda: catalog.append(1))
    monkeypatch.setattr(page, "_notice", lambda value: notices.append(value))
    monkeypatch.setattr(
        page, "_refresh_shell",
        lambda *, preserve_display=False: refreshes.append(preserve_display),
    )

    def arm(identity, value=None):
        _set_average_state(
            page,
            identity,
            store.revision,
            value or str(target.resolve()),
            source_root=str(tmp_path),
        )

    identity, update = scheduled(committed)
    # The page's average state holds the ANCHOR, not the artifact: it verifies a
    # terminal by RECOMPUTING the successor from anchor + version_identity
    # (workspace_operations.py:719) and comparing against result.target.  Arming
    # with committed.target made it derive a DOUBLE-suffixed path, so it
    # correctly reported a mismatch and never reloaded.  Before the successor
    # route the two were the same string, which is why this read as equivalent.
    arm(identity)
    before_catalog = len(catalog); before_refreshes = len(refreshes)
    from xrd_tools.core import provenance as provenance_module
    def forbidden(*_args, **_kwargs):
        pytest.fail("the page performed committed-artifact I/O")
    with monkeypatch.context() as patch:
        patch.setattr(provenance_module, "read_provenance", forbidden)
        patch.setattr(Path, "stat", forbidden); patch.setattr(Path, "open", forbidden)
        assert (
            page._consume_average_update(update)
            is WorkspaceRefreshEffect.CONTROLS
        )
    assert reloads == [(
        committed.target, committed.commit_identity, str(tmp_path),
    )]
    assert len(catalog) == before_catalog + 1
    assert len(refreshes) == before_refreshes
    assert page._workspace_operations.average_state is None

    expected_notice = {
        "REFUSED": "AVERAGE_TEST_REFUSED: detached refusal",
        "CANCELLED": "Average cancelled.",
        "ABORTED": "AVERAGE_TEST_ABORT: detached diagnostic",
    }
    for disposition in ("REFUSED", "CANCELLED", "ABORTED"):
        result = _result(disposition, str(target.resolve()))
        current, terminal = scheduled(result)
        arm(current); before_notices = len(notices)
        assert (
            page._consume_average_update(terminal)
            is WorkspaceRefreshEffect.CONTROLS
        )
        assert len(reloads) == 1
        assert len(notices) == before_notices + 1
        assert notices[-1] == expected_notice[disposition]

    wrong_commit = replace(committed.commit_identity, digest="e" * 64)
    for token, malformed in (
        ("target", _result("COMMITTED", str(tmp_path / "wrong.nxs"))),
        ("entry", replace(committed, entry="wrong")),
        ("operation", replace(committed, operation_identity="e" * 64)),
        ("science", replace(committed, science_identity="f" * 64)),
        ("commit", replace(committed, commit_identity=wrong_commit)),
    ):
        bad, update = scheduled(malformed, verification=token)
        arm(bad); before_notices = len(notices)
        assert (
            page._consume_average_update(update)
            is WorkspaceRefreshEffect.CONTROLS
        )
        assert len(reloads) == 1 and len(notices) == before_notices + 1
        assert notices[-1] == (
            f"Average failed: AVERAGE_COMMIT_VERIFICATION_FAILED: {token}"
        )

    stale, stale_update = scheduled(committed)
    # Anchor again, not the artifact: arming with the artifact makes the
    # target-mismatch branch fire BEFORE the stale branch, and that branch does
    # not request a catalog -- so this asserted the wrong refusal for the wrong
    # reason.
    arm(stale); before_notices = len(notices); before_catalog = len(catalog)
    before_refreshes = len(refreshes)
    assert page._consume_average_update(OperationUpdate(
        stale, terminal=stale_update.terminal, stale=True,
    )) is WorkspaceRefreshEffect.CONTROLS
    assert len(reloads) == 1
    assert len(catalog) == before_catalog + 1
    assert len(refreshes) == before_refreshes
    assert len(notices) == before_notices + 1
    assert notices[-1] == (
        "Average committed but context changed; Browse was not reloaded."
    )

    from xrd_tools.sources import execution_graph
    entered, release = threading.Event(), threading.Event()
    real_close = execution_graph._AverageSourceReadWindow.close; closes = []
    def held_close(window):
        closes.append(id(window))
        if len(closes) == 1: return real_close(window)
        entered.set()
        if not release.is_set(): raise OSError("retained source cleanup")
        return real_close(window)
    monkeypatch.setattr(execution_graph._AverageSourceReadWindow, "close", held_close)
    pending_source = _source(tmp_path / "pending-source")
    slot = OperationSlot(); pending_identity = slot.begin_average(
        _configuration(pending_source, reduction), tmp_path / "pending.nxs",
        stamp=OperationContextStamp(0),
    )
    assert pending_identity is not None and entered.wait(5)
    pending_update = _wait_update(
        slot, pending_identity, lambda value: value.pending is not None,
    )
    pending = pending_update.pending
    assert slot._worker is not None and slot._worker.is_alive()
    assert type(pending) is OperationPending
    assert pending.phase == "source-cleanup"
    release.set()
    assert slot.retry_average(pending_identity, pending)
    terminal = _join(slot, pending_identity)
    assert terminal.terminal.payload.disposition == "COMMITTED"
    page.close_workspace(); page.deleteLater(); qapp.processEvents()


@pytest.mark.parametrize(
    ("mode", "visible", "accepted"),
    (
        ("Int 1D", True, True),
        ("Int 2D", True, True),
        ("Int 1D (XYE)", False, False),
        ("1D Viewer", False, False),
        ("2D Viewer", False, False),
    ),
)
def test_average_control_and_edit_are_integration_mode_only(
    tmp_path, mode, visible, accepted,
) -> None:
    intent = RunIntent(
        source_spec=_source(tmp_path / mode.replace(" ", "_")),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        processing_mode=mode,
    )
    snapshot = RunIntentStore(intent).snapshot()
    projected = project_controls(snapshot, None, RunPhase.IDLE)
    paths = {
        field.path
        for field in projected.fields
    }
    assert (AVERAGE_SCAN in paths) is visible
    edited = reduce_control_edit(snapshot, AVERAGE_SCAN, True)
    assert (not isinstance(edited, EditRefusal)) is accepted


@pytest.mark.parametrize(
    "blocked_by", ("viewer", "browse_cleanup", "cache_debt")
)
def test_average_commit_reload_refusal_retains_exact_retryable_directive(
    tmp_path, monkeypatch, qapp, blocked_by,
) -> None:
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _page

    page, store = _page(tmp_path, monkeypatch)
    target = str((tmp_path / "committed.nxs").resolve())
    result = _result("COMMITTED", target)
    identity = OperationIdentity(1)
    update = OperationUpdate(
        identity,
        terminal=OperationTerminal(
            identity,
            OperationTerminalStatus.RETURNED,
            payload=result,
        ),
    )
    catalogs: list[None] = []
    refreshes: list[bool] = []
    notices: list[str] = []
    timers: list[None] = []
    monkeypatch.setattr(
        page, "_request_browser_catalog", lambda: catalogs.append(None),
    )
    monkeypatch.setattr(
        page, "_refresh_shell",
        lambda *, preserve_display=False: refreshes.append(preserve_display),
    )
    monkeypatch.setattr(page, "_notice", notices.append)
    monkeypatch.setattr(page, "_ensure_timer", lambda: timers.append(None))
    _set_average_state(
        page,
        identity,
        store.revision,
        target,
        source_root=str(tmp_path),
    )
    controller = page._context_controller
    if blocked_by == "viewer":
        controller._viewer_2d_standalone = object()
    elif blocked_by == "browse_cleanup":
        controller._cleanup_receipt = BrowseCleanupReceipt(
            None, CleanupStatus.CLEANUP_PENDING,
        )
    else:
        page._browse_1d_release_debt = SimpleNamespace(
            released=False, release=lambda: None,
        )
    try:
        assert page._consume_average_update(
            update
        ) is WorkspaceRefreshEffect.CONTROLS
        assert catalogs == [None]
        assert refreshes == []
        assert notices == (
            [
                "Average committed; Browse reload queued.",
                "Average committed; Browse reload deferred: "
                "2D Viewer cleanup remains pending",
            ]
            if blocked_by == "viewer"
            else ["Average committed; Browse reload queued."]
        )
        assert timers == [None]
        directive = page._processed_browser.pending_average_reload
        assert directive is not None
        assert page._workspace_operations.average_state is None
        assert page._processed_browser.busy

        controller._viewer_2d_standalone = None
        controller._cleanup_receipt = None
        page._browse_1d_release_debt = None
        reloads = []
        monkeypatch.setattr(
            controller,
            "begin_browse",
            lambda *args, **kwargs: reloads.append((args, kwargs)),
        )
        assert page._retry_pending_average_reload()
        # The reload names the ARTIFACT the result committed, not the anchor
        # that was requested.
        assert reloads == [((result.target,), {
            "terminal_commit_identity": result.commit_identity,
            "source_root": str(tmp_path),
        })]
        assert result.target != target
        assert page._processed_browser.pending_average_reload is None
        assert page._workspace_operations.average_state is None
        assert not page._processed_browser.busy
    finally:
        controller._viewer_2d_standalone = None
        controller._cleanup_receipt = None
        page._browse_1d_release_debt = None
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_average_stale_committed_terminal_reports_without_auto_reload(
    tmp_path, monkeypatch, qapp,
) -> None:
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _page

    page, store = _page(tmp_path, monkeypatch)
    target = str((tmp_path / "stale-committed.nxs").resolve())
    result = _result("COMMITTED", target)
    identity = OperationIdentity(91)
    update = OperationUpdate(
        identity,
        terminal=OperationTerminal(
            identity, OperationTerminalStatus.RETURNED, payload=result,
        ),
    )
    reloads = []
    catalogs = []
    notices = []
    refreshes = []
    monkeypatch.setattr(
        page._context_controller, "begin_browse",
        lambda *_args, **_kwargs: reloads.append(1),
    )
    monkeypatch.setattr(
        page, "_request_browser_catalog", lambda: catalogs.append(1),
    )
    monkeypatch.setattr(page, "_notice", notices.append)
    monkeypatch.setattr(
        page, "_refresh_shell",
        lambda *, preserve_display=False: refreshes.append(preserve_display),
    )
    _set_average_state(page, identity, store.revision, target)
    before = store.snapshot()
    changed = before.thaw()
    changed.project_root = str(tmp_path / "changed-context")
    store.commit(changed, expected_revision=before.revision)
    try:
        assert (
            page._consume_average_update(update)
            is WorkspaceRefreshEffect.CONTROLS
        )
        assert reloads == []
        assert catalogs == [1]
        assert refreshes == []
        assert notices == [
            "Average committed but context changed; Browse was not reloaded."
        ]
        assert page._workspace_operations.average_state is None
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_average_persists_sensor_and_parallax_detector_fields(
    tmp_path, monkeypatch,
) -> None:
    from xrd_tools.session.experiment_state import (
        CalibrationState,
        FactStatus,
        PoniValues,
    )

    _stub_integrators(monkeypatch)
    source = _source(tmp_path / "v3-average-source")
    target = tmp_path / "v3-average.nexus"
    calibration = CalibrationState(
        PoniValues(0.2, 0.0001, 0.0001, 0.0, 0.0, 0.0, 1.0e-10),
        "Detector",
        {
            "pixel1": 1.0e-4,
            "pixel2": 1.0e-4,
            "max_shape": [2, 2],
            "orientation": 3,
            "sensor": {"material": "CdTe", "thickness": 0.001},
        },
        status=FactStatus.PRESENT,
        parallax=True,
    )
    recipe = AverageScanRecipe(
        source,
        target,
        ReductionPlan(integration_1d=Integration1DPlan(npt=4)),
        calibration=calibration,
    )

    result = _run_terminal_average(recipe)

    assert result.disposition == "COMMITTED"
    assert recipe.calibration.parallax is True
    # `target` is the anchor requested; the successor route wrote
    # result.target.  Read the artifact, not the request.
    artifact = Path(result.target)
    assert artifact != target and artifact.exists()
    with h5py.File(artifact, "r") as handle:
        detector = handle["entry/instrument/detector"]
        assert detector["sensor_material"].asstr()[()] == "CdTe"
        assert float(detector["sensor_thickness"][()]) == pytest.approx(
            0.001
        )
        assert detector["sensor_thickness"].attrs["units"] == "m"
        assert bool(detector["parallax"][()]) is True
