"""Focused page composition oracles for authored detector assets."""
from __future__ import annotations

from dataclasses import fields, replace
import os
from pathlib import Path
from threading import current_thread

import numpy as np
import pytest
import tifffile
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering import experiment_authoring as authoring
from xdart.gui.tabs.scattering import page as page_module
from xdart.gui.tabs.scattering.adapters import external_operation
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.authored_assets import AuthoredAssetPhase
from xdart.gui.tabs.scattering.controls_inventory import PONI_FILE
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.experiment_authoring import (
    AssetValidationRequest,
    AssetValidationResult,
    AuthoredAssetCandidate,
    CalibrationCandidate,
    CalibrationFileProof,
    CalibrationResult,
    prepare_calibration_request,
    qualify_calibration_candidate,
)
from xdart.gui.tabs.scattering.operation_values import (
    OperationContextStamp,
    OperationIdentity,
    OperationTerminal,
    OperationTerminalStatus,
    OperationUpdate,
)
from xdart.gui.tabs.scattering.contracts import SourceFileState
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from tests.xdart.scattering.test_p3_calibrate_operation import _PONI


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _binary(tmp_path: Path, monkeypatch) -> Path:
    binary = tmp_path / "bin" / "xdart-calib2"
    binary.parent.mkdir(exist_ok=True)
    binary.write_text("fixture", encoding="utf-8")
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary.parent))
    return binary.resolve()


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "source.tiff"
    tifffile.imwrite(path, np.arange(8, dtype=np.uint16).reshape(2, 4))
    return path.resolve()


def _page(
    tmp_path: Path,
    monkeypatch,
    *,
    store: RunIntentStore | None = None,
    authoring_chooser=None,
    control_chooser=None,
):
    _binary(tmp_path, monkeypatch)
    owned = store or RunIntentStore(RunIntent(project_root=str(tmp_path)))
    page = ScatteringWorkspace(
        intents=owned,
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        authoring_source_chooser=authoring_chooser,
        control_path_chooser=control_chooser,
    )
    return page, owned


def _candidate(path: Path, *, mtime_ns: int | None = None):
    path.write_text(_PONI, encoding="utf-8")
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))
    return qualify_calibration_candidate(str(path))


def _queue_calibration(
    page: ScatteringWorkspace,
    store: RunIntentStore,
    source: Path,
    candidates,
    *,
    stale: bool = False,
    result_request=None,
):
    request = prepare_calibration_request(str(source))
    stamp = page._operation_context_stamp(store.revision)
    identity = OperationIdentity(101)
    page._apply_authored_asset_transition(
        page._authored_assets.adopt_operation(
            "poni", request, stamp, identity,
        )
    )
    result = CalibrationResult(
        request if result_request is None else result_request,
        tuple(candidates),
        0,
        (request.executable, request.source_path),
        request.monitored_directory,
    )
    update = OperationUpdate(
        identity,
        terminal=OperationTerminal(
            identity, OperationTerminalStatus.RETURNED, payload=result,
        ),
        stale=stale,
    )
    return page._consume_authored_asset_update(update), request


def _present_confirmation(page: ScatteringWorkspace):
    assert page._authored_assets.phase is AuthoredAssetPhase.TERMINAL_READY
    assert page._authored_asset_dialog is None
    page._advance_authored_asset_confirmation()
    assert page._authored_assets.phase is AuthoredAssetPhase.CONFIRM_ISSUED
    dialog = page._authored_asset_dialog
    assert dialog is not None
    page._advance_authored_asset_confirmation()
    assert page._authored_assets.phase is AuthoredAssetPhase.CONFIRM_PRESENTED
    return dialog


def _finish_validation(page: ScatteringWorkspace) -> OperationUpdate:
    identity = page._authored_assets.operation_identity
    assert type(identity) is OperationIdentity
    worker = page._workspace_operations._slot._worker
    assert worker is not None
    worker.join(3)
    assert not worker.is_alive()
    page._workspace_operations._slot.observe_stamp(
        page._operation_context_stamp()
    )
    update = page._workspace_operations._slot.poll(identity)
    assert type(update) is OperationUpdate
    assert page._consume_authored_asset_update(update)
    return update


def _close(page: ScatteringWorkspace, qapp) -> None:
    page.close_workspace()
    page.deleteLater()
    qapp.processEvents()


def _forge(value, **changes):
    forged = object.__new__(type(value))
    for field in fields(value):
        object.__setattr__(
            forged, field.name,
            changes.get(field.name, getattr(value, field.name)),
        )
    return forged


def test_calibrate_uses_dedicated_source_chooser_after_focus_settlement(
    tmp_path, monkeypatch, qapp,
) -> None:
    source = _source(tmp_path)
    order = []

    def authoring_chooser(asset, start):
        order.append(("authoring", asset, start))
        return str(source)

    page, _store = _page(
        tmp_path,
        monkeypatch,
        authoring_chooser=authoring_chooser,
        control_chooser=lambda *_args: pytest.fail(
            "existing-asset chooser used as authoring-source chooser"
        ),
    )
    identity = OperationIdentity(7)
    begun = []
    monkeypatch.setattr(
        page, "_commit_focused_control_edit_for_run",
        lambda: order.append(("settle",)) or True,
    )
    monkeypatch.setattr(
        page._workspace_operations, "begin_calibrate",
        lambda request, stamp: begun.append((request, stamp)) or identity,
    )
    try:
        page._handle_shell_command(
            ShellCommand(ShellCommandKind.CONTROL_ACTION, "calibrate")
        )
        assert order[0] == ("settle",)
        assert order[1][0:2] == ("authoring", "poni")
        assert Path(order[1][2]).is_dir()
        assert len(begun) == 1
        request, stamp = begun[0]
        assert request.source_path == str(source)
        assert request.exact_hdf_url is None
        assert page._authored_assets.asset == "poni"
        assert page._authored_assets.phase is AuthoredAssetPhase.RUNNING
        assert page._authored_assets.operation_identity is identity
    finally:
        _close(page, qapp)


def test_one_and_many_candidates_show_same_nonblocking_dialog_then_accept_worker(
    tmp_path, monkeypatch, qapp,
) -> None:
    source = _source(tmp_path)
    older = _candidate(tmp_path / "older.poni", mtime_ns=1_800_000_000_000_000_000)
    newest = _candidate(tmp_path / "newest.poni", mtime_ns=1_800_000_000_000_000_010)
    page, store = _page(tmp_path, monkeypatch)
    threads = []
    real_validate = external_operation.validate_authored_asset

    def counted(request):
        threads.append(current_thread().name)
        return real_validate(request)

    monkeypatch.setattr(external_operation, "validate_authored_asset", counted)
    try:
        page.show()
        qapp.processEvents()
        consumed, _request = _queue_calibration(
            page, store, source, (newest, older),
        )
        assert consumed
        owner = page._authored_assets
        assert owner.phase is AuthoredAssetPhase.TERMINAL_READY
        assert page._authored_asset_dialog is None
        dialog = _present_confirmation(page)
        assert dialog.selected_path == newest.path
        assert dialog.full_path.text() == newest.path
        assert dialog.full_path.isReadOnly()
        assert store.snapshot().thaw().poni_file == ""

        page._refresh_shell()
        qapp.processEvents()
        assert page._authored_assets is owner
        assert page._authored_asset_dialog is dialog
        assert dialog.isVisible() and dialog.parent() is page
        assert (dialog.windowModality()
                is QtCore.Qt.WindowModality.WindowModal)
        assert dialog.accept_button.hasFocus()
        assert dialog.paths.count() == 2
        dialog.paths.setCurrentIndex(1)
        assert dialog.selected_path == older.path
        assert dialog.full_path.text() == older.path
        dialog.paths.setCurrentIndex(0)
        assert dialog.full_path.text() == newest.path
        dialog.accept_button.click()
        _finish_validation(page)
        assert threads and threads[0].startswith("scattering-operation-")
        assert store.snapshot().thaw().poni_file == newest.path
        revision = store.revision
        foreign = OperationIdentity(999)
        assert page._consume_authored_asset_update(
            OperationUpdate(
                foreign,
                terminal=OperationTerminal(
                    foreign, OperationTerminalStatus.CANCELLED,
                ),
            )
        ) is False
        assert store.revision == revision
    finally:
        _close(page, qapp)


def test_zero_candidate_choose_another_uses_existing_chooser_at_exact_root(
    tmp_path, monkeypatch, qapp,
) -> None:
    source = _source(tmp_path)
    alternate = _candidate(tmp_path / "existing.poni")
    chooser_calls = []

    def chooser(control, current, start):
        chooser_calls.append((control, current, start, current_thread().name))
        return alternate.path

    page, store = _page(tmp_path, monkeypatch, control_chooser=chooser)
    try:
        consumed, _request = _queue_calibration(page, store, source, ())
        assert consumed
        owner = page._authored_assets
        dialog = _present_confirmation(page)
        qapp.processEvents()
        assert not dialog.accept_button.isEnabled()
        dialog.choose_button.click()
        _finish_validation(page)
        assert chooser_calls == [(
            PONI_FILE, "", str(tmp_path), current_thread().name,
        )]
        assert store.snapshot().thaw().poni_file == alternate.path
    finally:
        _close(page, qapp)


def test_cancel_duplicate_stale_and_forged_result_are_zero_write(
    tmp_path, monkeypatch, qapp,
) -> None:
    source = _source(tmp_path)
    candidate = _candidate(tmp_path / "generated.poni")
    monkeypatch.setattr(
        authoring, "calibration_candidate_current",
        lambda *_args: pytest.fail("GUI terminal consumption re-read PONI"),
    )
    page, store = _page(tmp_path, monkeypatch)
    try:
        assert _queue_calibration(page, store, source, (candidate,))[0]
        owner = page._authored_assets
        dialog = _present_confirmation(page)
        identity = page._authored_asset_dialog_identity
        assert identity is not None
        destroyed = []
        dialog.destroyed.connect(lambda *_args: destroyed.append(True))
        qapp.processEvents()
        assert dialog.isVisible() and dialog.paths.count() == 1
        assert dialog.selected_path == candidate.path
        assert dialog.full_path.text() == candidate.path
        dialog.cancel_button.click()
        qapp.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete,
        )
        qapp.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete,
        )
        assert destroyed == [True]
        assert owner.phase is AuthoredAssetPhase.IDLE
        assert page._authored_asset_dialog is None
        assert store.snapshot().thaw().poni_file == ""
        assert Path(candidate.path).read_text(encoding="utf-8") == _PONI
        page._cancel_authored_asset(identity, dialog)
        assert store.revision == 0

        assert _queue_calibration(
            page, store, source, (candidate,), stale=True,
        )[0]
        assert owner.phase is AuthoredAssetPhase.IDLE

        original = prepare_calibration_request(str(source))
        forged = replace(original)
        assert forged == original and forged is not original
        stamp = page._operation_context_stamp(store.revision)
        identity = OperationIdentity(202)
        page._authored_assets.adopt_operation(
            "poni", original, stamp, identity,
        )
        result = CalibrationResult(
            forged, (candidate,), 0,
            (forged.executable, forged.source_path), forged.monitored_directory,
        )
        assert page._consume_authored_asset_update(OperationUpdate(
            identity,
            terminal=OperationTerminal(
                identity, OperationTerminalStatus.RETURNED, payload=result,
            ),
        ))
        assert owner.phase is AuthoredAssetPhase.IDLE
        assert store.revision == 0
    finally:
        _close(page, qapp)


def test_calibration_result_candidate_count_and_aggregate_caps_precede_dialog(
    tmp_path, monkeypatch, qapp,
) -> None:
    source = _source(tmp_path)
    candidate = _candidate(tmp_path / "one.poni")
    page, store = _page(tmp_path, monkeypatch)
    request = prepare_calibration_request(str(source))
    baseline = CalibrationResult(
        request, (candidate,), 0,
        (request.executable, request.source_path),
        request.monitored_directory,
    )

    def consume(result, serial):
        identity = OperationIdentity(serial)
        stamp = page._operation_context_stamp(store.revision)
        page._authored_assets.adopt_operation(
            "poni", request, stamp, identity,
        )
        assert page._consume_authored_asset_update(OperationUpdate(
            identity,
            terminal=OperationTerminal(
                identity, OperationTerminalStatus.RETURNED,
                payload=result,
            ),
        ))
        assert page._authored_assets.phase is AuthoredAssetPhase.IDLE
        assert store.snapshot().thaw().poni_file == ""

    try:
        consume(_forge(baseline, candidates=(candidate,) * 257), 601)

        rows = []
        for index in range(17):
            path = str(tmp_path / f"aggregate-{index}.poni")
            state = SourceFileState(
                path, 1 << 20, index + 1, index + 1, 1, index + 1,
            )
            proof = CalibrationFileProof(
                state, candidate.proof.sha256,
                candidate.proof.detector_config_json,
                candidate.proof.geometry,
            )
            rows.append(CalibrationCandidate(path, proof))
        consume(_forge(baseline, candidates=tuple(rows)), 602)
        assert "exact publication proof" in page._notice_text.lower()
    finally:
        _close(page, qapp)


def test_context_and_candidate_drift_refuse_before_adoption(
    tmp_path, monkeypatch, qapp,
) -> None:
    source = _source(tmp_path)
    candidate = _candidate(tmp_path / "drift.poni")
    page, store = _page(tmp_path, monkeypatch)
    try:
        assert _queue_calibration(page, store, source, (candidate,))[0]
        owner = page._authored_assets
        changed = store.snapshot().thaw()
        changed.project_root = str(tmp_path / "other")
        store.commit(changed, expected_revision=store.revision)
        page._advance_authored_asset_confirmation()
        assert owner.phase is AuthoredAssetPhase.IDLE
        assert page._authored_asset_dialog is None
        assert store.snapshot().thaw().poni_file == ""

        browse_candidate = _candidate(tmp_path / "browse-drift.poni")
        assert _queue_calibration(
            page, store, source, (browse_candidate,),
        )[0]
        current_stamp = page._operation_context_stamp
        monkeypatch.setattr(
            page, "_operation_context_stamp",
            lambda revision=None: OperationContextStamp(
                store.revision if revision is None else revision,
                "different-browse", 1,
            ),
        )
        page._advance_authored_asset_confirmation()
        assert owner.phase is AuthoredAssetPhase.IDLE
        assert page._authored_asset_dialog is None
        assert store.snapshot().thaw().poni_file == ""
        monkeypatch.setattr(page, "_operation_context_stamp", current_stamp)

        candidate = _candidate(tmp_path / "replacement.poni")
        assert _queue_calibration(page, store, source, (candidate,))[0]
        dialog = _present_confirmation(page)
        Path(candidate.path).write_text(
            _PONI.replace("0.1234", "0.9234"), encoding="utf-8",
        )
        dialog.accept_button.click()
        update = _finish_validation(page)
        assert update.terminal.status is OperationTerminalStatus.FAILED
        assert owner.phase is AuthoredAssetPhase.CONFIRM_PRESENTED
        assert page._authored_asset_dialog is dialog
        assert store.snapshot().thaw().poni_file == ""
    finally:
        _close(page, qapp)


@pytest.mark.parametrize("moment", ("popup", "accept"))
@pytest.mark.parametrize("drift", ("source", "parent"))
def test_calibration_source_custody_survives_through_popup_and_accept(
    tmp_path, monkeypatch, qapp, moment, drift,
) -> None:
    source_dir = tmp_path / "source-dir"
    source_dir.mkdir()
    source = _source(source_dir)
    candidate = _candidate(source_dir / "newest.poni")
    page, store = _page(tmp_path, monkeypatch)
    try:
        assert _queue_calibration(page, store, source, (candidate,))[0]
        owner = page._authored_assets
        dialog = None
        if moment == "accept":
            dialog = _present_confirmation(page)

        if drift == "source":
            source.unlink()
            tifffile.imwrite(
                source, np.zeros((2, 4), dtype=np.uint16),
            )
        else:
            moved = tmp_path / "moved-source-dir"
            source_dir.rename(moved)
            source_dir.symlink_to(moved, target_is_directory=True)

        if moment == "popup":
            page._advance_authored_asset_confirmation()
        else:
            assert dialog is not None
            dialog.accept_button.click()
            QtCore.QCoreApplication.sendPostedEvents(
                None, QtCore.QEvent.Type.DeferredDelete,
            )
            qapp.processEvents()
        if moment == "popup":
            assert owner.phase is AuthoredAssetPhase.IDLE
            assert page._authored_asset_dialog is None
        else:
            assert owner.phase is AuthoredAssetPhase.IDLE
            assert owner.operation_identity is None
            assert page._authored_asset_dialog is None
        assert store.snapshot().thaw().poni_file == ""
    finally:
        _close(page, qapp)


@pytest.mark.parametrize("sabotage", ("invalid", "config", "geometry"))
def test_worker_scientifically_requalifies_admitted_poni_before_commit(
    tmp_path, monkeypatch, qapp, sabotage,
) -> None:
    source = _source(tmp_path)
    admitted = _candidate(tmp_path / f"{sabotage}.poni")
    if sabotage == "invalid":
        Path(admitted.path).write_text("not a PONI", encoding="utf-8")
        state = SourceFileState.capture(Path(admitted.path))
        proof = CalibrationFileProof(
            state, authoring._digest(Path(admitted.path)),
            admitted.proof.detector_config_json, admitted.proof.geometry,
        )
    elif sabotage == "config":
        proof = CalibrationFileProof(
            admitted.proof.state, admitted.proof.sha256,
            '{"orientation":2}', admitted.proof.geometry,
        )
    else:
        proof = CalibrationFileProof(
            admitted.proof.state, admitted.proof.sha256,
            admitted.proof.detector_config_json,
            (admitted.proof.geometry[0] + 0.5,)
            + admitted.proof.geometry[1:],
        )
    forged = CalibrationCandidate(admitted.path, proof)
    before = Path(forged.path).read_bytes()
    page, store = _page(tmp_path, monkeypatch)
    try:
        assert _queue_calibration(page, store, source, (forged,))[0]
        owner = page._authored_assets
        dialog = _present_confirmation(page)
        dialog.accept_button.click()
        update = _finish_validation(page)
        assert update.terminal.status is OperationTerminalStatus.FAILED
        assert store.revision == 0
        assert store.snapshot().thaw().poni_file == ""
        assert Path(forged.path).read_bytes() == before
        assert owner.phase is AuthoredAssetPhase.CONFIRM_PRESENTED
        assert page._authored_asset_dialog is dialog
        assert not dialog._busy
    finally:
        _close(page, qapp)


def test_calibration_result_resource_caps_precede_parser_deep_validation_and_qt(
    tmp_path, monkeypatch, qapp,
) -> None:
    source = _source(tmp_path)
    admitted = _candidate(tmp_path / "bounded.poni")
    page, store = _page(tmp_path, monkeypatch)
    request = prepare_calibration_request(str(source))
    baseline = CalibrationResult(
        request, (admitted,), 0,
        (request.executable, request.source_path),
        request.monitored_directory,
    )
    parses, deep = [], []

    def parse_bomb(*_args, **_kwargs):
        parses.append(1)
        pytest.fail("oversized configuration reached JSON parsing")

    def deep_bomb(*_args, **_kwargs):
        deep.append(1)
        pytest.fail("oversized result reached nested candidate validation")

    def consume(result, serial):
        identity = OperationIdentity(serial)
        stamp = page._operation_context_stamp(store.revision)
        page._authored_assets.adopt_operation(
            "poni", request, stamp, identity,
        )
        assert page._consume_authored_asset_update(OperationUpdate(
            identity,
            terminal=OperationTerminal(
                identity, OperationTerminalStatus.RETURNED,
                payload=result,
            ),
        ))
        assert page._authored_assets.phase is AuthoredAssetPhase.IDLE

    try:
        oversized = _forge(
            admitted.proof,
            detector_config_json="x" * ((1 << 20) + 1),
        )
        one = _forge(admitted, proof=oversized)
        monkeypatch.setattr(authoring.json, "loads", parse_bomb)
        monkeypatch.setattr(
            authoring.CalibrationCandidate, "__post_init__", deep_bomb,
        )
        consume(_forge(baseline, candidates=(one,)), 701)
        assert parses == deep == []

        rows = []
        config = "x" * (1 << 20)
        for index in range(17):
            path = str(tmp_path / f"config-{index}.poni")
            state = _forge(
                admitted.proof.state, path=path,
                inode=admitted.proof.state.inode + index + 1,
            )
            proof = _forge(
                admitted.proof, state=state, detector_config_json=config,
            )
            rows.append(_forge(admitted, path=path, proof=proof))
        consume(_forge(baseline, candidates=tuple(rows)), 702)
        assert parses == deep == []
        assert store.revision == 0
        assert "exact publication proof" in page._notice_text.lower()
    finally:
        _close(page, qapp)


def test_post_worker_replacement_and_pending_direct_entrypoints_are_blocked(
    tmp_path, monkeypatch, qapp,
) -> None:
    source = _source(tmp_path)
    admitted = _candidate(tmp_path / "post-worker.poni")
    page, store = _page(tmp_path, monkeypatch)
    try:
        assert _queue_calibration(page, store, source, (admitted,))[0]
        owner = page._authored_assets

        assert page._start_permitted() == (
            False, "Experiment operation is still active",
        )
        monkeypatch.setattr(
            page,
            "_authoring_source_chooser",
            lambda *_args: pytest.fail("calibration chooser was reached"),
        )
        page._calibrate_action()
        assert owner.operation_identity is None
        assert "unavailable while another operation" in page._notice_text
        assert page._begin_analysis(
            "metadata", object(), 0, request=("blocked",),
        ) is None
        monkeypatch.setattr(
            page, "_commit_focused_control_edit_for_run",
            lambda: pytest.fail("Reintegrate performed focused edit"),
        )
        page._reintegrate_action("1d")
        class BombSnapshot:
            def thaw(self):
                pytest.fail("Average evaluated an inadmissible pending context")
        page._average_action(BombSnapshot())
        monkeypatch.setattr(
            page, "_background_domain",
            lambda *_args: pytest.fail(
                "Background evaluated an inadmissible pending context"
            ),
        )
        page._background_action()

        candidate = AuthoredAssetCandidate(
            "poni", admitted.path, admitted.proof, admitted.proof.state,
        )
        dialog = _present_confirmation(page)
        dialog_identity = page._authored_asset_dialog_identity
        assert dialog_identity is not None
        request = owner.validation_request(
            dialog_identity,
            candidate.path,
            page._operation_context_stamp(),
        )
        assert type(request) is AssetValidationRequest
        assert request.candidate is not None
        result = AssetValidationResult(request, request.candidate)
        identity = OperationIdentity(303)
        page._apply_authored_asset_transition(
            owner.adopt_validation(dialog_identity, request, identity)
        )
        Path(candidate.path).write_text(
            _PONI.replace("0.1234", "0.8234"), encoding="utf-8",
        )
        assert page._consume_authored_asset_update(OperationUpdate(
            identity,
            terminal=OperationTerminal(
                identity, OperationTerminalStatus.RETURNED, payload=result,
            ),
        ))
        assert store.snapshot().thaw().poni_file == ""
        assert "inexact" in page._notice_text.lower()

        page.close_workspace()
        qapp.processEvents()
        assert page._authored_asset_dialog is None
        assert store.snapshot().thaw().poni_file == ""
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_average_refuses_active_background_before_dispatch(
    tmp_path, monkeypatch, qapp,
) -> None:
    from xrd_tools.reduction.background import FrameBackgroundPlan

    source = _source(tmp_path)
    intent = RunIntent(project_root=str(tmp_path))
    intent.background = FrameBackgroundPlan(
        mode="Single BG File",
        locator=str(source),
    )
    store = RunIntentStore(intent)
    page, _store = _page(tmp_path, monkeypatch, store=store)
    try:
        monkeypatch.setattr(
            page._workspace_operations,
            "begin_average",
            lambda *_args, **_kwargs: pytest.fail(
                "active Background reached Average dispatch"
            ),
        )
        page._average_action(store.snapshot())
        assert page._workspace_operations.average_identity is None
        assert page._notice_text == (
            "Average Scan does not support an active Background; "
            "choose Background: None before averaging."
        )
    finally:
        _close(page, qapp)
