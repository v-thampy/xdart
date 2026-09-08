from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pyqtgraph")
from pyqtgraph.Qt import QtWidgets

from xdart.gui.pages.operation_owner import (
    OperationIdentity,
    OperationProgress,
    OperationTerminalStatus,
)
from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import CloseReceipt, PageCleanup, STITCH_TOOL_KEY
from xdart.gui.tools.stitch_owner import (
    StitchOwnerAction,
    StitchOwnerFinalization,
    StitchOwnerOutcome,
    StitchOwnerOutcomeKind,
    StitchOwnerUpdate,
)
from xdart.gui.tools.stitch_tool import StitchToolDialog, build_stitch_tool
from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Status:
    def __init__(self):
        self.messages = []

    def show(self, text, timeout_ms=0):
        self.messages.append((str(text), int(timeout_ms)))


class _Owner:
    def __init__(self):
        self.busy = False
        self.form = None
        self.prepared = None
        self.finalization = StitchOwnerFinalization.NONE
        self.closed = 0

    def set_form(self, form):
        self.form = form
        return False

    def begin_preflight(self):
        return None

    def begin_run(self):
        return None

    def begin_retry_cleanup(self):
        return None

    def begin_retry_verification(self):
        return None

    def cancel(self):
        return False

    def poll(self):
        return None

    def close(self):
        self.closed += 1
        return CloseReceipt(PageCleanup.CLEAN)


def _dialog(qapp):
    status = _Status()
    owner = _Owner()
    dialog = StitchToolDialog(empty_host_services(status), owner=owner)
    return dialog, owner, status


def _fill_form(dialog, tmp_path):
    dialog.project_edit.setText(str(tmp_path))
    dialog.source_widget.set_uri(tmp_path / "extensionless_spec")
    dialog.scan_edit.setText("14")
    dialog.source_widget.image_dir_edit.setText(str(tmp_path / "images"))
    dialog.source_widget.image_stem_edit.setText("sample_scan14_")
    dialog.geometry_edit.setText(str(tmp_path / "geometry.json"))
    dialog.output_edit.setText(str(tmp_path / "stitched.nexus"))


def test_source_picker_and_form_projection_do_no_source_io(
    qapp, tmp_path, monkeypatch
):
    from xdart.gui.analysis import scan_source_widget

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Stitch form probing opened or decoded a source")

    monkeypatch.setattr(scan_source_widget.ScanSourceWidget, "_probe_source", forbidden)
    dialog, owner, _status = _dialog(qapp)
    try:
        _fill_form(dialog, tmp_path)
        qapp.processEvents()
        form = dialog._build_form()
        assert dialog.source_widget._external_execution is True
        assert dialog.source_widget._probe_executor is None
        assert form.project_root == str(tmp_path)
        assert form.spec_path == str(tmp_path / "extensionless_spec")
        assert form.scan == "14"
        assert form.image_dir == str(tmp_path / "images")
        assert form.image_stem == "sample_scan14_"
        assert form.detector_shape == (195, 1475)
        assert form.threshold == 800_000.0
        assert form.mode == "1d"
        assert form.overwrite is AnalysisArtifactOverwrite.REPLACE
        assert dialog.output_policy_label.text() == (
            "Create new (refuse if present)"
        )
        assert dialog.findChild(QtWidgets.QComboBox, "stitchOverwrite") is None
    finally:
        assert dialog.shutdown().status is PageCleanup.CLEAN
    assert owner.closed == 1


def test_any_form_edit_invalidates_run_enablement(qapp, tmp_path):
    dialog, _owner, _status = _dialog(qapp)
    try:
        _fill_form(dialog, tmp_path)
        dialog._prepared_form_fingerprint = "prepared"
        dialog.run_button.setEnabled(True)
        dialog.scan_edit.setText("15")
        assert dialog._prepared_form_fingerprint is None
        assert not dialog.run_button.isEnabled()
        assert "Preview again" in dialog.status_label.text()
    finally:
        dialog.shutdown()


def test_xu_backend_locks_asset_fields_and_installs_canonical_asset(qapp, tmp_path):
    dialog, _owner, _status = _dialog(qapp)
    try:
        _fill_form(dialog, tmp_path)
        dialog.backend_combo.setCurrentIndex(1)
        qapp.processEvents()
        assert dialog.backend_combo.currentData() == "xu_hist"
        assert dialog.geometry_label.text() == "XU calibration asset"
        assert not dialog.geometry_kind_combo.isEnabled()
        assert not dialog.motor_mapping_edit.isEnabled()
        assert not dialog.threshold_edit.isEnabled()
        assert not dialog.install_xu_asset_button.isHidden()
        dialog._install_xu_asset()
        asset = Path(dialog.geometry_edit.text())
        assert asset.is_file()
        assert asset.name == "psic_powder_1d_surface_v1.json"
        form = dialog._build_form()
        assert form.backend == "xu_hist"
        assert form.source_motors == (("del", "del"), ("nu", "nu"))
        assert form.detector_shape == (195, 1475)
        assert form.raw_dtype == np.dtype("int32").str
        assert form.q_range == (1.0, 5.2)
        assert form.use_detector_mask is True
    finally:
        dialog.shutdown()


def test_preflight_presentation_lists_exact_controls_and_every_member(
    qapp, stitch_form
):
    dialog, _owner, _status = _dialog(qapp)
    preflight = prepare_stitch_tool(stitch_form)
    try:
        dialog._render_preflight(preflight.summary)
        shown = dialog.preview_text.toPlainText()
        assert f"Project: {stitch_form.project_root}" in shown
        assert "Source: myscan · scan 5.1" in shown
        assert "Images: images · stem myscan_scan5_" in shown
        assert "Motor mapping: del_angle=del, nu_angle=nu" in shown
        assert "Raw decoder: 4×5" in shown
        assert "threshold 800000" in shown
        assert "Normalization: monitor I0[0] · detector mask on" in shown
        for member in preflight.summary.members:
            assert (
                f"{member.label}: {member.relative_path}"
                f"#{member.source_frame_index}"
            ) in shown
    finally:
        dialog.shutdown()


def test_progress_failure_and_cancellation_never_invoke_painter(
    qapp, monkeypatch
):
    dialog, _owner, _status = _dialog(qapp)
    painted = []
    monkeypatch.setattr(dialog, "_paint_result", painted.append)
    identity = OperationIdentity(1, object())
    try:
        dialog._accept_update(
            StitchOwnerUpdate(
                identity,
                StitchOwnerAction.RUN,
                progress=OperationProgress(identity, 1, "science", 2, 5),
            )
        )
        dialog._accept_update(
            StitchOwnerUpdate(
                identity,
                StitchOwnerAction.RUN,
                terminal_status=OperationTerminalStatus.FAILED,
                failure_module="tests",
                failure_type="ExpectedFailure",
                failure_message="failed",
            )
        )
        dialog._accept_update(
            StitchOwnerUpdate(
                identity,
                StitchOwnerAction.RUN,
                terminal_status=OperationTerminalStatus.CANCELLED,
            )
        )
        assert painted == []
    finally:
        dialog.shutdown()


def test_form_revision_change_during_run_refuses_stale_committed_paint(
    qapp, monkeypatch
):
    from xdart.gui.tools import stitch_owner, stitch_tool

    dialog, _owner, _status = _dialog(qapp)

    class _Result:
        pass

    result = _Result()
    result.terminal = SimpleNamespace(disposition=ModuleDisposition.COMMITTED)
    result.payload = object()
    monkeypatch.setattr(stitch_owner, "StitchOperationResult", _Result)
    monkeypatch.setattr(stitch_tool, "StitchOperationResult", _Result)
    painted = []
    monkeypatch.setattr(dialog, "_paint_result", painted.append)
    identity = OperationIdentity(1, object())
    dialog._active_form_revision = 4
    dialog._form_revision = 5
    try:
        dialog._accept_update(
            StitchOwnerUpdate(
                identity,
                StitchOwnerAction.RUN,
                terminal_status=OperationTerminalStatus.RETURNED,
                outcome=StitchOwnerOutcome(
                    StitchOwnerOutcomeKind.RESULT,
                    result=result,
                ),
            )
        )
        assert painted == []
        assert "was not painted" in dialog.status_label.text()
    finally:
        dialog.shutdown()


@pytest.mark.parametrize(
    ("retry_method", "action", "finalization"),
    (
        (
            "_begin_retry_cleanup",
            StitchOwnerAction.RETRY_CLEANUP,
            StitchOwnerFinalization.CLEANUP_PENDING,
        ),
        (
            "_begin_retry_verification",
            StitchOwnerAction.RETRY_VERIFICATION,
            StitchOwnerFinalization.VERIFICATION_PENDING,
        ),
    ),
)
def test_finalization_retry_after_form_edit_never_paints_retained_result(
    qapp,
    monkeypatch,
    retry_method,
    action,
    finalization,
):
    from xdart.gui.tools import stitch_owner, stitch_tool

    dialog, owner, _status = _dialog(qapp)

    class _Result:
        pass

    result = _Result()
    result.terminal = SimpleNamespace(disposition=ModuleDisposition.COMMITTED)
    result.payload = object()
    monkeypatch.setattr(stitch_owner, "StitchOperationResult", _Result)
    monkeypatch.setattr(stitch_tool, "StitchOperationResult", _Result)
    painted = []
    monkeypatch.setattr(dialog, "_paint_result", painted.append)
    identity = OperationIdentity(1, object())
    retry_name = (
        "begin_retry_cleanup"
        if action is StitchOwnerAction.RETRY_CLEANUP
        else "begin_retry_verification"
    )
    monkeypatch.setattr(owner, retry_name, lambda: identity)
    owner.finalization = finalization
    dialog._execution_form_revision = 4
    dialog._form_revision = 4
    dialog._form_changed()
    assert dialog._form_revision == 5
    try:
        getattr(dialog, retry_method)()
        dialog._poll_timer.stop()
        assert dialog._active_form_revision == 4
        dialog._accept_update(
            StitchOwnerUpdate(
                identity,
                action,
                terminal_status=OperationTerminalStatus.RETURNED,
                outcome=StitchOwnerOutcome(
                    StitchOwnerOutcomeKind.RESULT,
                    result=result,
                ),
            )
        )
        assert painted == []
        assert "was not painted" in dialog.status_label.text()
    finally:
        dialog.shutdown()


def test_committed_payload_replaces_all_three_curves_in_one_frozen_update(
    qapp, monkeypatch
):
    from xdart.gui.tools import stitch_tool

    dialog, _owner, _status = _dialog(qapp)

    class _Result:
        pass

    q = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    payload = SimpleNamespace(
        axis=lambda name: q if name == "q" else None,
        intensity=np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
        coverage=np.asarray([1.0, 2.0, 1.0], dtype=np.float32),
        normalization=np.asarray([2.0, 4.0, 2.0], dtype=np.float32),
        result_fingerprint="a" * 64,
    )
    result = _Result()
    result.terminal = SimpleNamespace(disposition=ModuleDisposition.COMMITTED)
    result.payload = payload
    monkeypatch.setattr(stitch_tool, "StitchOperationResult", _Result)
    updates = []
    original_updates = dialog.setUpdatesEnabled

    def record_updates(enabled):
        updates.append(bool(enabled))
        original_updates(enabled)

    monkeypatch.setattr(dialog, "setUpdatesEnabled", record_updates)
    calls = []
    for name, curve in (
        ("intensity", dialog.intensity_curve),
        ("coverage", dialog.coverage_curve),
        ("normalization", dialog.normalization_curve),
    ):
        monkeypatch.setattr(
            curve,
            "setData",
            lambda x, y, name=name: calls.append(
                (name, np.asarray(x).copy(), np.asarray(y).copy())
            ),
        )
    try:
        dialog._paint_result(result)
        assert updates == [False, True]
        assert [name for name, _x, _y in calls] == [
            "intensity",
            "coverage",
            "normalization",
        ]
        assert all(np.array_equal(x, q) for _name, x, _y in calls)
        assert dialog._painted_result_fingerprint == "a" * 64
    finally:
        dialog.shutdown()


def test_real_factory_returns_host_contract_without_starting_work(qapp):
    status = _Status()
    parent = QtWidgets.QMainWindow()
    handle = build_stitch_tool(empty_host_services(status), parent)
    try:
        assert handle.key == STITCH_TOOL_KEY
        assert isinstance(handle.widget, QtWidgets.QDialog)
        assert handle.widget.parent() is parent
        assert handle.activity.active() is False
        assert handle.close().status is PageCleanup.CLEAN
    finally:
        parent.close()
