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
from xdart.gui.pages.values import CloseReceipt, PageCleanup, RSM_TOOL_KEY
from xdart.gui.tools.rsm_owner import (
    RSMOwnerAction,
    RSMOwnerFinalization,
    RSMOwnerOutcome,
    RSMOwnerOutcomeKind,
    RSMOwnerUpdate,
)
from xdart.gui.tools.rsm_tool import RSMToolDialog, build_rsm_tool
from xrd_tools.analysis.module_transaction import ModuleDisposition


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
        self.finalization = RSMOwnerFinalization.NONE
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
    dialog = RSMToolDialog(empty_host_services(status), owner=owner)
    return dialog, owner, status


def _fill_form(dialog, tmp_path):
    dialog.project_edit.setText(str(tmp_path))
    dialog._apply_scan43_preset()
    dialog.output_edit.setText(str(tmp_path / "rsm.nexus"))


def test_form_projection_and_occurrence_editor_do_no_source_io(
    qapp,
    tmp_path,
    monkeypatch,
):
    from xrd_tools.sources.spec import SpecSource

    def forbidden(*_args, **_kwargs):
        raise AssertionError("RSM form editing opened or decoded a source")

    monkeypatch.setattr(SpecSource, "__init__", forbidden)
    monkeypatch.setattr(SpecSource, "load_frame", forbidden)
    dialog, owner, _status = _dialog(qapp)
    try:
        _fill_form(dialog, tmp_path)
        exposure_name, exposure_occurrence = dialog.selector_edits["exposure"]
        exposure_name.setText("Seconds")
        exposure_occurrence.setValue(1)
        form = dialog._build_form()
        assert form.project_root == str(tmp_path)
        assert form.spec_path == str(tmp_path / "STO_align")
        assert form.image_dir == str(tmp_path / "images")
        assert form.raw_header_skip == 0
        assert form.plan.normalization.exposure_selector.name == "Seconds"
        assert form.plan.normalization.exposure_selector.occurrence == 1
        assert form.plan.bins == (40, 40, 40)
    finally:
        assert dialog.shutdown().status is PageCleanup.CLEAN
    assert owner.closed == 1


def test_initial_preset_paths_follow_project_without_reapplying(qapp, tmp_path):
    dialog, _owner, _status = _dialog(qapp)
    try:
        assert dialog.spec_edit.text() == "STO_align"
        assert dialog.image_dir_edit.text() == "images"
        assert dialog.output_edit.text() == "rsm_scan43.nexus"
        dialog.project_edit.setText(str(tmp_path))
        form = dialog._build_form()
        assert form.spec_path == str(tmp_path / "STO_align")
        assert form.image_dir == str(tmp_path / "images")
        assert form.output_path == str(tmp_path / "rsm_scan43.nexus")
    finally:
        dialog.shutdown()


def test_any_form_edit_invalidates_run_enablement(qapp, tmp_path):
    dialog, _owner, _status = _dialog(qapp)
    try:
        _fill_form(dialog, tmp_path)
        dialog._prepared_form_fingerprint = "prepared"
        dialog._painted_result_fingerprint = "a" * 64
        dialog.result_facts.setText("old result")
        for image in dialog.slice_images:
            image.setImage(np.ones((2, 2), dtype=np.float32))
        dialog.run_button.setEnabled(True)
        dialog.scan_edit.setText("44.1")
        assert dialog._prepared_form_fingerprint is None
        assert dialog._painted_result_fingerprint is None
        assert "inputs changed" in dialog.result_facts.text().lower()
        assert all(image.image is None for image in dialog.slice_images)
        assert not dialog.run_button.isEnabled()
        assert "previous result cleared" in dialog.status_label.text()
    finally:
        dialog.shutdown()


def test_new_preview_clears_previously_painted_result(
    qapp,
    tmp_path,
    monkeypatch,
):
    dialog, owner, _status = _dialog(qapp)
    identity = OperationIdentity(1, object())
    try:
        _fill_form(dialog, tmp_path)
        dialog._painted_result_fingerprint = "a" * 64
        dialog.result_facts.setText("old result")
        for image in dialog.slice_images:
            image.setImage(np.ones((2, 2), dtype=np.float32))
        monkeypatch.setattr(owner, "begin_preflight", lambda: identity)

        dialog._begin_preflight()
        dialog._poll_timer.stop()

        assert dialog._painted_result_fingerprint is None
        assert "new Preview" in dialog.result_facts.text()
        assert all(image.image is None for image in dialog.slice_images)
    finally:
        dialog.shutdown()


def test_preflight_presentation_lists_occurrences_science_and_every_member(
    qapp,
    prepared_rsm_tool,
):
    dialog, _owner, _status = _dialog(qapp)
    try:
        dialog._render_preflight(prepared_rsm_tool.summary)
        shown = dialog.preview_text.toPlainText()
        assert f"Project: {prepared_rsm_tool.form.project_root}" in shown
        assert "Source: source/RSM_synth · scan 1.1" in shown
        assert "Seconds[1]" in shown
        assert "Energy: 13000.007 eV" in shown
        assert "Exact q bounds:" in shown
        assert "source threshold none · source rotation 0°" in shown
        assert "gi-and-refraction-corrections" in shown
        assert "hostile-shared-project-output-namespace" in shown
        for member in prepared_rsm_tool.summary.members:
            assert (
                f"{member.label}: {member.relative_path}"
                f"#{member.source_frame_index}"
            ) in shown
            assert f"{member.normalization_divisor:.12g}" in shown
    finally:
        dialog.shutdown()


def test_progress_failure_and_cancellation_never_paint(qapp, monkeypatch):
    dialog, _owner, _status = _dialog(qapp)
    painted = []
    monkeypatch.setattr(dialog, "_paint_result", painted.append)
    identity = OperationIdentity(1, object())
    try:
        dialog._accept_update(
            RSMOwnerUpdate(
                identity,
                RSMOwnerAction.RUN,
                progress=OperationProgress(identity, 1, "science", 2, 5),
            )
        )
        dialog._accept_update(
            RSMOwnerUpdate(
                identity,
                RSMOwnerAction.RUN,
                terminal_status=OperationTerminalStatus.FAILED,
                failure_module="tests",
                failure_type="ExpectedFailure",
                failure_message="failed",
            )
        )
        dialog._accept_update(
            RSMOwnerUpdate(
                identity,
                RSMOwnerAction.RUN,
                terminal_status=OperationTerminalStatus.CANCELLED,
            )
        )
        dialog._accept_update(
            RSMOwnerUpdate(
                identity,
                RSMOwnerAction.PREFLIGHT,
                terminal_status=OperationTerminalStatus.RETURNED,
                outcome=RSMOwnerOutcome(
                    RSMOwnerOutcomeKind.PREFLIGHT_REFUSED,
                    refusal_code="SOURCE_REVISION_CHANGED",
                    refusal_message="source changed",
                    diagnostics=("exact revision mismatch",),
                ),
            )
        )
        assert painted == []
        assert "Preview refused" in dialog.status_label.text()
    finally:
        dialog.shutdown()


def test_poll_contains_presentation_failure(qapp, monkeypatch):
    dialog, owner, _status = _dialog(qapp)
    monkeypatch.setattr(owner, "poll", lambda: object())
    monkeypatch.setattr(
        dialog,
        "_accept_update",
        lambda _update: (_ for _ in ()).throw(RuntimeError("paint fault")),
    )
    try:
        dialog._poll_timer.start()
        dialog._poll_owner()
        assert not dialog._poll_timer.isActive()
        assert "presentation failed" in dialog.status_label.text()
        assert "paint fault" in dialog.status_label.text()
    finally:
        dialog.shutdown()


def test_changed_form_never_paints_retained_commit(qapp, monkeypatch):
    from xdart.gui.tools import rsm_owner, rsm_tool

    dialog, _owner, _status = _dialog(qapp)

    class _Result:
        pass

    result = _Result()
    result.terminal = SimpleNamespace(disposition=ModuleDisposition.COMMITTED)
    result.payload = object()
    monkeypatch.setattr(rsm_owner, "RSMOperationResult", _Result)
    monkeypatch.setattr(rsm_tool, "RSMOperationResult", _Result)
    painted = []
    monkeypatch.setattr(dialog, "_paint_result", painted.append)
    identity = OperationIdentity(1, object())
    dialog._active_form_revision = 4
    dialog._form_revision = 5
    try:
        dialog._accept_update(
            RSMOwnerUpdate(
                identity,
                RSMOwnerAction.RUN,
                terminal_status=OperationTerminalStatus.RETURNED,
                outcome=RSMOwnerOutcome(
                    RSMOwnerOutcomeKind.RESULT,
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
            RSMOwnerAction.RETRY_CLEANUP,
            RSMOwnerFinalization.CLEANUP_PENDING,
        ),
        (
            "_begin_retry_verification",
            RSMOwnerAction.RETRY_VERIFICATION,
            RSMOwnerFinalization.VERIFICATION_PENDING,
        ),
    ),
)
def test_retry_after_edit_never_paints_old_result(
    qapp,
    monkeypatch,
    retry_method,
    action,
    finalization,
):
    from xdart.gui.tools import rsm_owner, rsm_tool

    dialog, owner, _status = _dialog(qapp)

    class _Result:
        pass

    result = _Result()
    result.terminal = SimpleNamespace(disposition=ModuleDisposition.COMMITTED)
    result.payload = object()
    monkeypatch.setattr(rsm_owner, "RSMOperationResult", _Result)
    monkeypatch.setattr(rsm_tool, "RSMOperationResult", _Result)
    painted = []
    monkeypatch.setattr(dialog, "_paint_result", painted.append)
    identity = OperationIdentity(1, object())
    retry_name = (
        "begin_retry_cleanup"
        if action is RSMOwnerAction.RETRY_CLEANUP
        else "begin_retry_verification"
    )
    monkeypatch.setattr(owner, retry_name, lambda: identity)
    owner.finalization = finalization
    dialog._execution_form_revision = 4
    dialog._form_revision = 4
    dialog._form_changed()
    try:
        getattr(dialog, retry_method)()
        dialog._poll_timer.stop()
        assert dialog._active_form_revision == 4
        dialog._accept_update(
            RSMOwnerUpdate(
                identity,
                action,
                terminal_status=OperationTerminalStatus.RETURNED,
                outcome=RSMOwnerOutcome(
                    RSMOwnerOutcomeKind.RESULT,
                    result=result,
                ),
            )
        )
        assert painted == []
        assert "was not painted" in dialog.status_label.text()
    finally:
        dialog.shutdown()


def test_committed_payload_replaces_three_central_slices_atomically(
    qapp,
    monkeypatch,
):
    from xdart.gui.tools import rsm_tool

    dialog, _owner, _status = _dialog(qapp)

    class _Result:
        pass

    intensity = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9)
    axes = {
        "h": np.linspace(1.0, 2.0, 5, dtype=np.float32),
        "k": np.linspace(3.0, 4.0, 7, dtype=np.float32),
        "l": np.linspace(5.0, 6.0, 9, dtype=np.float32),
    }
    payload = SimpleNamespace(
        intensity=intensity,
        axis=lambda name: axes[name],
        result_fingerprint="a" * 64,
    )
    result = _Result()
    result.terminal = SimpleNamespace(disposition=ModuleDisposition.COMMITTED)
    result.payload = payload
    monkeypatch.setattr(rsm_tool, "RSMOperationResult", _Result)
    updates = []
    original_updates = dialog.setUpdatesEnabled

    def record_updates(enabled):
        updates.append(bool(enabled))
        original_updates(enabled)

    monkeypatch.setattr(dialog, "setUpdatesEnabled", record_updates)
    original_images = tuple(dialog.slice_images)
    try:
        dialog._paint_result(result)
        assert updates == [False, True]
        assert tuple(dialog.slice_images) != original_images
        expected_slices = (
            intensity[2, :, :],
            intensity[:, 3, :],
            intensity[:, :, 4],
        )
        expected_rects = (
            (4.9375, 2.9166666666666665, 1.125, 1.166666666666667),
            (4.9375, 0.875, 1.125, 1.25),
            (2.9166666666666665, 0.875, 1.166666666666667, 1.25),
        )
        for plot, image, expected, expected_rect in zip(
            dialog.slice_plots,
            dialog.slice_images,
            expected_slices,
            expected_rects,
        ):
            assert image.axisOrder == "row-major"
            assert image in plot.getPlotItem().items
            np.testing.assert_array_equal(image.image, expected)
            rect = image.mapRectToParent(image.boundingRect())
            np.testing.assert_allclose(
                (rect.x(), rect.y(), rect.width(), rect.height()),
                expected_rect,
                rtol=0,
                atol=1.0e-6,
            )
        assert all(
            image not in plot.getPlotItem().items
            for plot, image in zip(dialog.slice_plots, original_images)
        )
        assert dialog._painted_result_fingerprint == "a" * 64
        assert "finite voxels 315" in dialog.result_facts.text()
    finally:
        dialog.shutdown()


def test_second_plot_add_failure_preserves_previous_presentation(
    qapp,
    monkeypatch,
):
    from xdart.gui.tools import rsm_tool

    dialog, _owner, _status = _dialog(qapp)

    class _Result:
        pass

    intensity = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9)
    axes = {
        "h": np.linspace(1.0, 2.0, 5, dtype=np.float32),
        "k": np.linspace(3.0, 4.0, 7, dtype=np.float32),
        "l": np.linspace(5.0, 6.0, 9, dtype=np.float32),
    }
    result = _Result()
    result.terminal = SimpleNamespace(disposition=ModuleDisposition.COMMITTED)
    result.payload = SimpleNamespace(
        intensity=intensity,
        axis=lambda name: axes[name],
        result_fingerprint="b" * 64,
    )
    monkeypatch.setattr(rsm_tool, "RSMOperationResult", _Result)
    original_images = tuple(dialog.slice_images)
    dialog.result_facts.setText("retained facts")
    dialog._painted_result_fingerprint = "a" * 64
    monkeypatch.setattr(
        dialog.slice_plots[1],
        "addItem",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("second plot failed")
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="second plot failed"):
            dialog._paint_result(result)

        assert tuple(dialog.slice_images) == original_images
        assert dialog.result_facts.text() == "retained facts"
        assert dialog._painted_result_fingerprint == "a" * 64
        assert all(
            image in plot.getPlotItem().items
            for plot, image in zip(dialog.slice_plots, original_images)
        )
    finally:
        dialog.shutdown()


def test_late_facts_failure_restores_previous_presentation(qapp, monkeypatch):
    from xdart.gui.tools import rsm_tool

    dialog, _owner, _status = _dialog(qapp)

    class _Result:
        pass

    intensity = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9)
    axes = {
        "h": np.linspace(1.0, 2.0, 5, dtype=np.float32),
        "k": np.linspace(3.0, 4.0, 7, dtype=np.float32),
        "l": np.linspace(5.0, 6.0, 9, dtype=np.float32),
    }
    result = _Result()
    result.terminal = SimpleNamespace(disposition=ModuleDisposition.COMMITTED)
    result.payload = SimpleNamespace(
        intensity=intensity,
        axis=lambda name: axes[name],
        result_fingerprint="b" * 64,
    )
    monkeypatch.setattr(rsm_tool, "RSMOperationResult", _Result)
    original_images = tuple(dialog.slice_images)
    dialog.result_facts.setText("retained facts")
    dialog._painted_result_fingerprint = "a" * 64
    original_set_text = dialog.result_facts.setText
    calls = []

    def fail_once(text):
        calls.append(str(text))
        if len(calls) == 1:
            raise RuntimeError("facts failed")
        return original_set_text(text)

    monkeypatch.setattr(dialog.result_facts, "setText", fail_once)
    try:
        with pytest.raises(RuntimeError, match="facts failed"):
            dialog._paint_result(result)

        assert tuple(dialog.slice_images) == original_images
        assert dialog.result_facts.text() == "retained facts"
        assert dialog._painted_result_fingerprint == "a" * 64
        assert all(
            image in plot.getPlotItem().items
            for plot, image in zip(dialog.slice_plots, original_images)
        )
    finally:
        dialog.shutdown()


def test_real_factory_returns_idle_cached_host_contract(qapp):
    status = _Status()
    parent = QtWidgets.QMainWindow()
    handle = build_rsm_tool(empty_host_services(status), parent)
    try:
        assert handle.key == RSM_TOOL_KEY
        assert isinstance(handle.widget, QtWidgets.QDialog)
        assert handle.widget.parent() is parent
        assert handle.activity.active() is False
        assert handle.close().status is PageCleanup.CLEAN
    finally:
        parent.close()
