from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
from threading import Event
import time

import numpy as np
import pytest
import tifffile

from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tools.rsm_owner import (
    RSMOwnerAction,
    RSMOwnerFinalization,
    RSMOwnerOutcomeKind,
    RSMToolOwner,
)
from xdart.gui.tools.rsm_values import (
    RSMFrameSelector,
    RSMScanMemberForm,
    RSMToolFormV2,
    RSMToolPreflightRefused,
    prepare_rsm_tool_v2,
)
from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector,
    ModuleDisposition,
    ModuleProgress,
    ModuleSourceReceipt,
    ModuleTerminalResult,
)
from xrd_tools.analysis.rsm_geometry_asset import (
    CANONICAL_RSM_GEOMETRY_LOCATOR,
    install_canonical_rsm_geometry_asset,
    rsm_geometry_asset_input,
)
from xrd_tools.analysis.rsm_operation import (
    RSMImageConditioning,
    RSMNormalizationPolicy,
    RSMOperationCleanupPending,
    RSMOperationCleanupPendingV2,
    RSMOperationExecution,
    RSMOperationExecutionV2,
    RSMOperationResult,
    RSMOperationResultV2,
    RSMOperationVerificationError,
    RSMOperationVerificationErrorV2,
)
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame


# The RSM v2 owner's helpers and rows, moved from the classic RSM dialog's test file when the dialog was
# retired (AUD-9 stage 2b G10): they never touched the dialog. `_prepared_v2` installs the canonical RSM
# geometry asset, so a later retirement of that asset must replace it here too.
_ROLES = ("mu", "eta", "chi", "phi", "nu", "del")


def _write_member(root: Path, ordinal: int):
    source = root / f"source-{ordinal}"
    images = source / "images"
    images.mkdir(parents=True)
    spec = source / f"RSM_synth_{ordinal}"
    spec.write_text(
        f"""#F RSM_synth_{ordinal}
#E 1
#D Mon Jan 15 10:30:00 2024
#O0 energy  mu  chi  phi  nu  del

#S 1 ascan eta {ordinal} {ordinal + 1} 1 1
#D Mon Jan 15 10:30:00 2024
#P0 13000.007 {10 + ordinal} 20 30 40 50
#G3 1 0 0 0 1 0 0 0 1
#N 4
#L eta  Seconds  Seconds  foil status
{ordinal}.0 1 10 101
{ordinal + 1}.0 2 20 110
""",
        encoding="utf-8",
    )
    for label in range(2):
        tifffile.imwrite(
            images / f"member{ordinal}_{label:04d}.tif",
            np.arange(20, dtype=np.uint16).reshape(4, 5) + ordinal + label,
        )
    return RSMScanMemberForm(
        spec,
        "1.1",
        images,
        f"member{ordinal}_",
        RSMFrameSelector(0, 1, 1),
        (195, 487),
        "uint16",
        0,
        tuple((role, MetadataColumnSelector(role, 0)) for role in _ROLES),
    )


def _prepared_v2(root, coordinate_frame=RSMCoordinateFrame.HKL):
    install_canonical_rsm_geometry_asset(project_root=root)
    output = root / "output"
    output.mkdir(exist_ok=True)
    form = RSMToolFormV2(
        root,
        rsm_geometry_asset_input(CANONICAL_RSM_GEOMETRY_LOCATOR),
        (_write_member(root, 0), _write_member(root, 1)),
        RSMImageConditioning(0.0, None, None),
        RSMNormalizationPolicy.identity(),
        (4, 5, 6),
        1,
        16 * 1024 * 1024,
        16 * 1024 * 1024,
        output / "rsm-v2.nexus",
        AnalysisArtifactOverwrite.CREATE_NEW,
        coordinate_frame,
    )
    return prepare_rsm_tool_v2(form)


def _poll_terminal(owner: RSMToolOwner, *, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        update = owner.poll()
        if update is not None and update.terminal:
            return update
        time.sleep(0.001)
    raise AssertionError("RSM owner terminal did not arrive")


def _refused_result(execution: RSMOperationExecution) -> RSMOperationResult:
    return RSMOperationResult(
        execution.request,
        ModuleTerminalResult(
            execution.request.module,
            ModuleDisposition.REFUSED,
            "TEST_RESULT",
        ),
    )


def _prepare_owner(preflight):
    owner = RSMToolOwner(
        preflight_runner=lambda _form, *, cancel_token=None: preflight
    )
    assert owner.set_form(preflight.form) is False
    assert owner.begin_preflight() is not None
    update = _poll_terminal(owner)
    assert update.action is RSMOwnerAction.PREFLIGHT
    assert update.outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_READY
    assert owner.prepared is preflight
    return owner


def test_form_edit_and_bounce_cannot_resurrect_preflight(prepared_rsm_tool):
    preflight = prepared_rsm_tool
    owner = _prepare_owner(preflight)
    changed = replace(
        preflight.form,
        output_path=preflight.form.output_path.replace("rsm.nexus", "changed.nexus"),
    )
    assert owner.set_form(changed) is True
    assert owner.set_form(preflight.form) is False
    assert owner.prepared is None
    assert owner.begin_run() is None
    assert owner.close().status is PageCleanup.CLEAN


def test_inflight_preview_bounce_is_stale(prepared_rsm_tool):
    preflight = prepared_rsm_tool
    entered = Event()
    release = Event()

    def held(_form, *, cancel_token=None):
        entered.set()
        assert release.wait(2)
        return preflight

    owner = RSMToolOwner(preflight_runner=held)
    owner.set_form(preflight.form)
    assert owner.begin_preflight() is not None
    assert entered.wait(2)
    changed = replace(
        preflight.form,
        output_path=preflight.form.output_path.replace("rsm.nexus", "changed.nexus"),
    )
    owner.set_form(changed)
    owner.set_form(preflight.form)
    release.set()
    update = _poll_terminal(owner)
    assert update.stale is True
    assert update.outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_READY
    assert owner.prepared is None
    assert owner.close().status is PageCleanup.CLEAN


def test_failed_repreview_revokes_same_form_authority(prepared_rsm_tool):
    preflight = prepared_rsm_tool
    calls = 0

    def runner(_form, *, cancel_token=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return preflight
        raise RuntimeError("source revision changed")

    owner = RSMToolOwner(preflight_runner=runner)
    owner.set_form(preflight.form)
    assert owner.begin_preflight() is not None
    assert _poll_terminal(owner).outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_READY
    assert owner.begin_preflight() is not None
    assert owner.prepared is None
    failed = _poll_terminal(owner)
    assert failed.terminal_status is OperationTerminalStatus.FAILED
    assert failed.failure_type == "RuntimeError"
    assert owner.begin_run() is None
    assert owner.close().status is PageCleanup.CLEAN


def test_domain_failed_run_is_logged_once_through_the_shared_worker(monkeypatch, prepared_rsm_tool):
    from xdart.gui.widgets.log_panel import LogFeed

    feed = LogFeed(capacity=20)
    logger = logging.getLogger("xdart.gui.tools.rsm_owner")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(feed)

    def failed_run(self, **_kwargs):
        return RSMOperationResult(self.request, ModuleTerminalResult(
            self.request.module, ModuleDisposition.FAILED, "GRID_FAILED", diagnostic="verification failed"))

    monkeypatch.setattr(RSMOperationExecution, "run", failed_run)
    owner = _prepare_owner(prepared_rsm_tool)
    try:
        lines, _cursor = feed.held()
        assert not any("New Run" in line or "End of Run" in line for line in lines)
        identity = owner.begin_run()
        assert identity is not None
        update = _poll_terminal(owner)
        assert update.outcome.result.terminal.disposition is ModuleDisposition.FAILED
        lines, _cursor = feed.held()
        target = str(prepared_rsm_tool.request.module.output.target)
        assert len([line for line in lines if f"New Run · RSM · {target}" in line]) == 1
        assert len([line for line in lines if "End of Run · RSM · failed" in line]) == 1
        assert len([line for line in lines if f"RSM failed · {target}: GRID_FAILED: verification failed" in line]) == 1
        assert feed.warnings == 1
        assert owner.poll() is None
        assert feed.warnings == 1
    finally:
        assert owner.close().status is PageCleanup.CLEAN
        logger.removeHandler(feed)


def test_preview_and_science_are_cooperatively_cancellable(
    monkeypatch,
    prepared_rsm_tool,
):
    preflight = prepared_rsm_tool
    preview_entered = Event()

    def held_preview(_form, *, cancel_token=None):
        preview_entered.set()
        assert cancel_token.wait(2)
        raise RSMToolPreflightRefused("CANCELLED")

    preview_owner = RSMToolOwner(preflight_runner=held_preview)
    preview_owner.set_form(preflight.form)
    assert preview_owner.begin_preflight() is not None
    assert preview_entered.wait(2)
    assert preview_owner.cancel() is True
    assert preview_owner.cancel() is False
    assert (
        _poll_terminal(preview_owner).terminal_status
        is OperationTerminalStatus.CANCELLED
    )
    assert preview_owner.close().status is PageCleanup.CLEAN

    science_entered = Event()

    def held_run(self, *, cancel_token=None, progress_callback=None):
        science_entered.set()
        assert cancel_token.wait(2)
        return _refused_result(self)

    monkeypatch.setattr(RSMOperationExecution, "run", held_run)
    owner = _prepare_owner(preflight)
    assert owner.begin_run() is not None
    assert science_entered.wait(2)
    assert owner.cancel() is True
    update = _poll_terminal(owner)
    assert update.outcome.kind is RSMOwnerOutcomeKind.RESULT
    assert owner.close().status is PageCleanup.CLEAN


def test_typed_preview_refusal_is_returned_without_prepared_authority(
    prepared_rsm_tool,
):
    preflight = prepared_rsm_tool

    def refused(_form, *, cancel_token=None):
        raise RSMToolPreflightRefused(
            "SOURCE_REVISION_CHANGED",
            "source changed during Preview",
            diagnostics=("primary SPEC revision changed",),
        )

    owner = RSMToolOwner(preflight_runner=refused)
    owner.set_form(preflight.form)
    assert owner.begin_preflight() is not None
    update = _poll_terminal(owner)

    assert update.terminal_status is OperationTerminalStatus.RETURNED
    assert update.outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_REFUSED
    assert update.outcome.refusal_code == "SOURCE_REVISION_CHANGED"
    assert "source changed during Preview" in update.outcome.refusal_message
    assert update.outcome.diagnostics == ("primary SPEC revision changed",)
    assert owner.prepared is None
    assert owner.begin_run() is None
    assert owner.close().status is PageCleanup.CLEAN


def test_production_preview_maps_exact_source_binding_drift_to_refusal(
    monkeypatch,
    rsm_tool_form,
):
    def drifted_source(cls, *_args, **_kwargs):
        raise ValueError("metadata table is no longer an exact source fact")

    monkeypatch.setattr(
        ModuleSourceReceipt,
        "from_metadata_table",
        classmethod(drifted_source),
    )
    owner = RSMToolOwner()
    owner.set_form(rsm_tool_form)
    assert owner.begin_preflight() is not None
    update = _poll_terminal(owner)

    assert update.terminal_status is OperationTerminalStatus.RETURNED
    assert update.outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_REFUSED
    assert update.outcome.refusal_code == "SOURCE_REVISION_CHANGED"
    assert "changed while binding" in update.outcome.refusal_message
    assert update.outcome.diagnostics == (
        "metadata table is no longer an exact source fact",
    )
    assert owner.prepared is None
    assert owner.begin_run() is None
    assert owner.close().status is PageCleanup.CLEAN


def test_latest_progress_and_cleanup_retry_do_not_replay(
    monkeypatch,
    prepared_rsm_tool,
):
    preflight = prepared_rsm_tool
    entered = Event()
    release = Event()
    calls = []

    def held_run(self, *, cancel_token=None, progress_callback=None):
        calls.append("run")
        progress_callback(ModuleProgress(self.request.module, 1, "science", 1, 3))
        progress_callback(ModuleProgress(self.request.module, 2, "science", 2, 3))
        entered.set()
        assert release.wait(2)
        raise RSMOperationCleanupPending(self)

    def retry_cleanup(self):
        calls.append("retry-cleanup")
        return _refused_result(self)

    monkeypatch.setattr(RSMOperationExecution, "run", held_run)
    monkeypatch.setattr(RSMOperationExecution, "retry_cleanup", retry_cleanup)
    owner = _prepare_owner(preflight)
    assert owner.begin_run() is not None
    assert entered.wait(2)
    progress = owner.poll()
    assert (progress.progress.revision, progress.progress.completed) == (2, 2)
    assert owner.poll() is None
    release.set()
    pending = _poll_terminal(owner)
    assert pending.outcome.kind is RSMOwnerOutcomeKind.CLEANUP_PENDING
    assert owner.finalization is RSMOwnerFinalization.CLEANUP_PENDING
    assert owner.begin_retry_cleanup() is not None
    completed = _poll_terminal(owner)
    assert completed.outcome.kind is RSMOwnerOutcomeKind.RESULT
    assert calls == ["run", "retry-cleanup"]
    assert owner.close().status is PageCleanup.CLEAN


def test_verification_retry_uses_retained_execution(
    monkeypatch,
    prepared_rsm_tool,
):
    preflight = prepared_rsm_tool
    calls = []

    def run(self, *, cancel_token=None, progress_callback=None):
        calls.append("run")
        raise RSMOperationVerificationError(self, "held reload")

    def retry_verification(self):
        calls.append("retry-verification")
        return _refused_result(self)

    monkeypatch.setattr(RSMOperationExecution, "run", run)
    monkeypatch.setattr(
        RSMOperationExecution,
        "retry_verification",
        retry_verification,
    )
    owner = _prepare_owner(preflight)
    assert owner.begin_run() is not None
    pending = _poll_terminal(owner)
    assert pending.outcome.kind is RSMOwnerOutcomeKind.VERIFICATION_PENDING
    assert owner.begin_retry_verification() is not None
    assert _poll_terminal(owner).outcome.kind is RSMOwnerOutcomeKind.RESULT
    assert calls == ["run", "retry-verification"]
    assert owner.close().status is PageCleanup.CLEAN


def test_close_drains_and_retries_exact_cleanup(monkeypatch, prepared_rsm_tool):
    preflight = prepared_rsm_tool
    entered = Event()
    release = Event()
    calls = []

    def held_run(self, *, cancel_token=None, progress_callback=None):
        calls.append(("run", self))
        entered.set()
        assert release.wait(2)
        assert cancel_token.is_set()
        raise RSMOperationCleanupPending(self)

    def retry_cleanup(self):
        calls.append(("retry-cleanup", self))
        return _refused_result(self)

    monkeypatch.setattr(RSMOperationExecution, "run", held_run)
    monkeypatch.setattr(RSMOperationExecution, "retry_cleanup", retry_cleanup)
    owner = _prepare_owner(preflight)
    assert owner.begin_run() is not None
    assert entered.wait(2)
    assert owner.close().status is PageCleanup.PENDING
    release.set()
    deadline = time.monotonic() + 3
    while True:
        receipt = owner.close()
        if receipt.status is PageCleanup.CLEAN:
            break
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert [name for name, _execution in calls] == ["run", "retry-cleanup"]
    assert calls[0][1] is calls[1][1]


def test_close_drains_and_retries_exact_verification(
    monkeypatch,
    prepared_rsm_tool,
):
    preflight = prepared_rsm_tool
    calls = []

    def run(self, *, cancel_token=None, progress_callback=None):
        calls.append(("run", self))
        raise RSMOperationVerificationError(self, "held reload")

    def retry_verification(self):
        calls.append(("retry-verification", self))
        return _refused_result(self)

    monkeypatch.setattr(RSMOperationExecution, "run", run)
    monkeypatch.setattr(
        RSMOperationExecution,
        "retry_verification",
        retry_verification,
    )
    owner = _prepare_owner(preflight)
    assert owner.begin_run() is not None
    pending = _poll_terminal(owner)
    assert pending.outcome.kind is RSMOwnerOutcomeKind.VERIFICATION_PENDING
    assert owner.close().status is PageCleanup.PENDING

    deadline = time.monotonic() + 3
    while True:
        receipt = owner.close()
        if receipt.status is PageCleanup.CLEAN:
            break
        assert time.monotonic() < deadline
        time.sleep(0.001)

    assert [name for name, _execution in calls] == [
        "run",
        "retry-verification",
    ]
    assert calls[0][1] is calls[1][1]


def test_v2_owner_profile_adopts_only_exact_v2_preflight(tmp_path):
    prepared = _prepared_v2(tmp_path)
    owner = RSMToolOwner.v2(
        preflight_runner=lambda _form, *, cancel_token=None: prepared
    )
    try:
        assert owner.set_form(prepared.form) is False
        assert owner.begin_preflight() is not None
        update = _poll_terminal(owner)
        assert update.outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_READY
        assert owner.prepared is prepared
    finally:
        assert owner.close().status is PageCleanup.CLEAN


@pytest.mark.parametrize(
    ("pending_error", "pending_kind", "retry_name", "begin_name"),
    (
        (
            RSMOperationCleanupPendingV2,
            RSMOwnerOutcomeKind.CLEANUP_PENDING,
            "retry_cleanup",
            "begin_retry_cleanup",
        ),
        (
            RSMOperationVerificationErrorV2,
            RSMOwnerOutcomeKind.VERIFICATION_PENDING,
            "retry_verification",
            "begin_retry_verification",
        ),
    ),
)
def test_v2_owner_retains_exact_execution_for_finalization_retry(
    tmp_path,
    monkeypatch,
    pending_error,
    pending_kind,
    retry_name,
    begin_name,
):
    prepared = _prepared_v2(tmp_path)

    def run(self, *, cancel_token=None, progress_callback=None):
        if pending_error is RSMOperationVerificationErrorV2:
            raise pending_error(self, "held reload")
        raise pending_error(self)

    def retry(self):
        return RSMOperationResultV2(
            self.request,
            ModuleTerminalResult(
                self.request.module,
                ModuleDisposition.REFUSED,
                "TEST_RESULT",
            ),
        )

    monkeypatch.setattr(RSMOperationExecutionV2, "run", run)
    monkeypatch.setattr(RSMOperationExecutionV2, retry_name, retry)
    owner = RSMToolOwner.v2(
        preflight_runner=lambda _form, *, cancel_token=None: prepared
    )
    try:
        owner.set_form(prepared.form)
        assert owner.begin_preflight() is not None
        assert _poll_terminal(owner).outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_READY
        assert owner.begin_run() is not None
        assert _poll_terminal(owner).outcome.kind is pending_kind
        assert getattr(owner, begin_name)() is not None
        assert _poll_terminal(owner).outcome.kind is RSMOwnerOutcomeKind.RESULT
    finally:
        assert owner.close().status is PageCleanup.CLEAN


@pytest.mark.parametrize(
    "foreign_error",
    (RSMOperationCleanupPending, RSMOperationVerificationError),
)
def test_v2_owner_rejects_wrong_generation_finalization_exception(
    tmp_path,
    monkeypatch,
    foreign_error,
):
    prepared = _prepared_v2(tmp_path)

    def run(self, *, cancel_token=None, progress_callback=None):
        if foreign_error is RSMOperationVerificationError:
            raise foreign_error(self, "foreign R1 verification")
        raise foreign_error(self)

    monkeypatch.setattr(RSMOperationExecutionV2, "run", run)
    owner = RSMToolOwner.v2(
        preflight_runner=lambda _form, *, cancel_token=None: prepared
    )
    try:
        owner.set_form(prepared.form)
        assert owner.begin_preflight() is not None
        assert _poll_terminal(owner).outcome.kind is RSMOwnerOutcomeKind.PREFLIGHT_READY
        assert owner.begin_run() is not None
        failed = _poll_terminal(owner)
        assert failed.terminal_status is OperationTerminalStatus.FAILED
        assert failed.failure_type == "TypeError"
        assert "generation" in failed.failure_message
        assert owner.finalization is RSMOwnerFinalization.NONE
        assert owner.begin_retry_cleanup() is None
        assert owner.begin_retry_verification() is None
    finally:
        assert owner.close().status is PageCleanup.CLEAN
