from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event
import time

import numpy as np
import pytest

from xdart.gui.pages.operation_owner import OperationTerminalStatus
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tools.stitch_owner import (
    StitchOwnerAction,
    StitchOwnerFinalization,
    StitchOwnerOutcomeKind,
    StitchToolOwner,
)
from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xrd_tools.analysis.module_transaction import (
    ModuleDisposition,
    ModuleProgress,
    ModuleTerminalResult,
)
from xrd_tools.analysis.stitch_operation import (
    StitchOperationCleanupPending,
    StitchOperationExecution,
    StitchOperationResult,
    StitchOperationVerificationError,
)


def _poll_terminal(owner: StitchToolOwner, *, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        update = owner.poll()
        if update is not None and update.terminal:
            return update
        time.sleep(0.001)
    raise AssertionError("Stitch owner terminal did not arrive")


@pytest.fixture
def preflight(stitch_form):
    return prepare_stitch_tool(stitch_form)


def _refused_result(execution: StitchOperationExecution) -> StitchOperationResult:
    return StitchOperationResult(
        execution.request,
        ModuleTerminalResult(
            execution.request.module,
            ModuleDisposition.REFUSED,
            "TEST_RESULT",
        ),
    )


def _prepare_owner(preflight):
    owner = StitchToolOwner(preflight_runner=lambda _form: preflight)
    assert owner.set_form(preflight.form) is False
    identity = owner.begin_preflight()
    assert identity is not None
    update = _poll_terminal(owner)
    assert update.action is StitchOwnerAction.PREFLIGHT
    assert update.outcome.kind is StitchOwnerOutcomeKind.PREFLIGHT_READY
    assert owner.prepared is preflight
    return owner


def test_form_edit_stales_prepared_request_and_run_refuses(preflight):
    owner = _prepare_owner(preflight)
    changed = replace(preflight.form, output_path=replace(
        preflight.form, output_path=preflight.form.output_path
    ).output_path.replace("stitched.nexus", "changed.nexus"))
    assert owner.set_form(changed) is True
    assert owner.prepared is None
    assert owner.prepared_stale is False
    assert owner.begin_run() is None
    receipt = owner.close()
    assert receipt.status is PageCleanup.CLEAN
    assert owner.close() is receipt


def test_form_bounce_cannot_resurrect_revoked_preflight(preflight):
    owner = _prepare_owner(preflight)
    changed = replace(
        preflight.form,
        output_path=preflight.form.output_path.replace(
            "stitched.nexus", "changed.nexus"
        ),
    )
    assert owner.set_form(changed) is True
    assert owner.set_form(preflight.form) is False
    assert owner.prepared is None
    assert owner.begin_run() is None
    assert owner.close().status is PageCleanup.CLEAN


def test_inflight_preflight_bounce_cannot_adopt_stale_authority(preflight):
    entered = Event()
    release = Event()

    def held_preflight(_form):
        entered.set()
        assert release.wait(2)
        return preflight

    owner = StitchToolOwner(preflight_runner=held_preflight)
    assert owner.set_form(preflight.form) is False
    assert owner.begin_preflight() is not None
    assert entered.wait(2)
    changed = replace(
        preflight.form,
        output_path=preflight.form.output_path.replace(
            "stitched.nexus", "changed.nexus"
        ),
    )
    owner.set_form(changed)
    owner.set_form(preflight.form)
    release.set()
    update = _poll_terminal(owner)
    assert update.stale is True
    assert update.outcome.kind is StitchOwnerOutcomeKind.PREFLIGHT_READY
    assert owner.prepared is None
    assert owner.begin_run() is None
    assert owner.close().status is PageCleanup.CLEAN


def test_failed_repreflight_revokes_old_same_form_request(preflight):
    calls = 0

    def preflight_runner(_form):
        nonlocal calls
        calls += 1
        if calls == 1:
            return preflight
        raise RuntimeError("source revision changed")

    owner = StitchToolOwner(preflight_runner=preflight_runner)
    assert owner.set_form(preflight.form) is False
    assert owner.begin_preflight() is not None
    assert _poll_terminal(owner).outcome.kind is StitchOwnerOutcomeKind.PREFLIGHT_READY
    assert owner.prepared is preflight
    assert owner.begin_preflight() is not None
    assert owner.prepared is None
    failed = _poll_terminal(owner)
    assert failed.terminal_status is OperationTerminalStatus.FAILED
    assert failed.failure_type == "RuntimeError"
    assert owner.prepared is None
    assert owner.begin_run() is None
    assert owner.close().status is PageCleanup.CLEAN


def test_latest_progress_and_cleanup_retry_never_replay_science(
    monkeypatch, preflight
):
    entered = Event()
    release = Event()
    calls = []

    def held_run(self, *, cancel_token=None, progress_callback=None):
        calls.append("run")
        progress_callback(ModuleProgress(self.request.module, 1, "science", 1, 3))
        progress_callback(ModuleProgress(self.request.module, 2, "science", 2, 3))
        entered.set()
        assert release.wait(2)
        raise StitchOperationCleanupPending(self)

    def retry_cleanup(self):
        calls.append("retry-cleanup")
        return _refused_result(self)

    monkeypatch.setattr(StitchOperationExecution, "run", held_run)
    monkeypatch.setattr(StitchOperationExecution, "retry_cleanup", retry_cleanup)
    owner = _prepare_owner(preflight)
    identity = owner.begin_run()
    assert identity is not None
    assert entered.wait(2)
    progress = owner.poll()
    assert progress.progress is not None
    assert (progress.progress.revision, progress.progress.completed) == (2, 2)
    assert owner.poll() is None
    release.set()
    pending = _poll_terminal(owner)
    assert pending.outcome.kind is StitchOwnerOutcomeKind.CLEANUP_PENDING
    assert owner.finalization is StitchOwnerFinalization.CLEANUP_PENDING
    assert owner.begin_retry_verification() is None
    assert owner.begin_retry_cleanup() is not None
    completed = _poll_terminal(owner)
    assert completed.outcome.kind is StitchOwnerOutcomeKind.RESULT
    assert completed.outcome.result is owner.last_result
    assert owner.finalization is StitchOwnerFinalization.NONE
    assert calls == ["run", "retry-cleanup"]
    assert owner.close().status is PageCleanup.CLEAN


def test_published_inspection_failure_reaches_owner_with_publication_truth(monkeypatch, preflight):
    from xrd_tools.io import analysis_artifact

    # The form fixture uses tiny raw files for preflight-only tests. Execute
    # with the actual Pilatus100k shape required by its captured geometry.
    for path in Path(preflight.form.image_dir).glob("*.raw"):
        np.ones((195, 487), dtype=np.int32).tofile(path)
    preflight = prepare_stitch_tool(replace(preflight.form, detector_shape=(195, 487)))
    target = Path(preflight.request.module.output.target)
    inspect = analysis_artifact.inspect_analysis_artifact
    failures = []

    def inspect_once(path, **kwargs):
        if Path(path) == target and not failures:
            failures.append(path)
            raise OSError("transient published inspection")
        return inspect(path, **kwargs)

    monkeypatch.setattr(analysis_artifact, "inspect_analysis_artifact", inspect_once)
    owner = _prepare_owner(preflight)
    assert owner.begin_run() is not None
    pending = _poll_terminal(owner, timeout=10)
    assert pending.outcome.kind is StitchOwnerOutcomeKind.CLEANUP_PENDING
    assert pending.outcome.finalization_message == "analysis artifact published; verification remains pending"
    assert target.exists()
    assert owner.begin_retry_cleanup() is not None
    completed = _poll_terminal(owner, timeout=10)
    assert completed.outcome.result.terminal.disposition is ModuleDisposition.COMMITTED
    assert owner.close().status is PageCleanup.CLEAN


def test_verification_retry_uses_retained_execution_only(monkeypatch, preflight):
    calls = []

    def run(self, *, cancel_token=None, progress_callback=None):
        calls.append("run")
        raise StitchOperationVerificationError(self, "held reload")

    def retry_verification(self):
        calls.append("retry-verification")
        return _refused_result(self)

    monkeypatch.setattr(StitchOperationExecution, "run", run)
    monkeypatch.setattr(
        StitchOperationExecution, "retry_verification", retry_verification
    )
    owner = _prepare_owner(preflight)
    assert owner.begin_run() is not None
    pending = _poll_terminal(owner)
    assert pending.outcome.kind is StitchOwnerOutcomeKind.VERIFICATION_PENDING
    assert owner.finalization is StitchOwnerFinalization.VERIFICATION_PENDING
    assert owner.begin_retry_cleanup() is None
    assert owner.begin_retry_verification() is not None
    completed = _poll_terminal(owner)
    assert completed.outcome.kind is StitchOwnerOutcomeKind.RESULT
    assert calls == ["run", "retry-verification"]
    assert owner.close().status is PageCleanup.CLEAN


def test_close_drains_terminal_then_auto_retries_exact_cleanup(
    monkeypatch, preflight
):
    entered = Event()
    release = Event()
    calls = []

    def held_run(self, *, cancel_token=None, progress_callback=None):
        calls.append(("run", self))
        entered.set()
        assert release.wait(2)
        assert cancel_token.is_set()
        raise StitchOperationCleanupPending(self)

    def retry_cleanup(self):
        calls.append(("retry-cleanup", self))
        return _refused_result(self)

    monkeypatch.setattr(StitchOperationExecution, "run", held_run)
    monkeypatch.setattr(StitchOperationExecution, "retry_cleanup", retry_cleanup)
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
    assert owner.close() is receipt
    assert [name for name, _execution in calls] == ["run", "retry-cleanup"]
    assert calls[0][1] is calls[1][1]


def test_close_drains_terminal_then_auto_retries_exact_verification(
    monkeypatch, preflight
):
    calls = []

    def run(self, *, cancel_token=None, progress_callback=None):
        calls.append(("run", self))
        raise StitchOperationVerificationError(self, "held reload")

    def retry_verification(self):
        calls.append(("retry-verification", self))
        return _refused_result(self)

    monkeypatch.setattr(StitchOperationExecution, "run", run)
    monkeypatch.setattr(
        StitchOperationExecution, "retry_verification", retry_verification
    )
    owner = _prepare_owner(preflight)
    assert owner.begin_run() is not None
    pending = _poll_terminal(owner)
    assert pending.outcome.kind is StitchOwnerOutcomeKind.VERIFICATION_PENDING
    assert owner.close().status is PageCleanup.PENDING

    deadline = time.monotonic() + 3
    while True:
        receipt = owner.close()
        if receipt.status is PageCleanup.CLEAN:
            break
        assert time.monotonic() < deadline
        time.sleep(0.001)

    assert owner.close() is receipt
    assert [name for name, _execution in calls] == [
        "run",
        "retry-verification",
    ]
    assert calls[0][1] is calls[1][1]


def test_cancel_only_targets_active_science(monkeypatch, preflight):
    entered = Event()

    def run(self, *, cancel_token=None, progress_callback=None):
        entered.set()
        assert cancel_token.wait(2)
        return _refused_result(self)

    monkeypatch.setattr(StitchOperationExecution, "run", run)
    owner = _prepare_owner(preflight)
    assert owner.cancel() is False
    assert owner.begin_run() is not None
    assert entered.wait(2)
    assert owner.cancel() is True
    assert owner.cancel() is False
    update = _poll_terminal(owner)
    assert update.terminal_status is OperationTerminalStatus.RETURNED
    assert update.outcome.kind is StitchOwnerOutcomeKind.RESULT
    assert owner.close().status is PageCleanup.CLEAN
