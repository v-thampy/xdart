from __future__ import annotations

import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pyqtgraph")
import pyqtgraph as pg
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
    RSMToolOwner,
)
from xdart.gui.tools.rsm_tool import RSMToolDialog, build_rsm_tool
from xdart.gui.tools.rsm_values import (
    RSMFrameSelector,
    RSMScanMemberForm,
    RSMToolFormV2,
    prepare_rsm_tool_v2,
)
from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector,
    ModuleDisposition,
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
    RSMOperationExecutionV2,
    RSMOperationResultV2,
    RSMOperationVerificationError,
    RSMOperationVerificationErrorV2,
)
from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactKind,
    AnalysisArtifactOverwrite,
    project_analysis_artifact_result,
)
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from xrd_tools.session.rsm_viewer_model import make_rsm_viewer_values


_ROLES = ("mu", "eta", "chi", "phi", "nu", "del")


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


def _fill_form(dialog, root):
    dialog.project_edit.setText(str(root))
    dialog._apply_scan43_preset()
    dialog.output_edit.setText(str(root / "rsm-v2.nexus"))


def _values(shape=(5, 7, 9), *, offset=0.0):
    axes = tuple(
        (
            name,
            np.linspace(offset + index, offset + index + 1.0, size),
        )
        for index, (name, size) in enumerate(
            zip(("h", "k", "l"), shape, strict=True)
        )
    )
    intensity = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + offset
    projection = project_analysis_artifact_result(
        kind=AnalysisArtifactKind.RSM,
        axes=axes,
        axis_units=(("h", None), ("k", None), ("l", None)),
        intensity=intensity,
        sigma=None,
        coverage=None,
        normalization=None,
    )
    return make_rsm_viewer_values(projection)


def _q_values(shape=(5, 7, 9), *, offset=0.0):
    frame = RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU
    axes = tuple(
        (
            name,
            np.linspace(offset + index, offset + index + 1.0, size),
        )
        for index, (name, size) in enumerate(
            zip(frame.axis_names, shape, strict=True)
        )
    )
    intensity = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + offset
    projection = project_analysis_artifact_result(
        kind=AnalysisArtifactKind.RSM,
        axes=axes,
        axis_units=tuple(
            zip(frame.axis_names, frame.axis_units, strict=True)
        ),
        intensity=intensity,
        sigma=None,
        coverage=None,
        normalization=None,
    )
    return make_rsm_viewer_values(projection)


class _Result:
    pass


def _result(payload):
    result = _Result()
    result.terminal = SimpleNamespace(disposition=ModuleDisposition.COMMITTED)
    result.payload = payload
    return result


def _patch_results(monkeypatch, mapping):
    from xdart.gui.tools import rsm_owner, rsm_tool

    monkeypatch.setattr(rsm_owner, "RSMOperationResultV2", _Result)
    monkeypatch.setattr(rsm_tool, "RSMOperationResultV2", _Result)
    monkeypatch.setattr(rsm_tool, "make_rsm_viewer_values", mapping.__getitem__)


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


def _poll_terminal(owner):
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        update = owner.poll()
        if update is not None and update.terminal:
            return update
        time.sleep(0.001)
    raise AssertionError("RSM owner terminal did not arrive")


def test_v2_form_editor_and_six_panel_shell_do_no_source_io(
    qapp, tmp_path, monkeypatch
):
    from xrd_tools.sources.spec import SpecSource

    def forbidden(*_args, **_kwargs):
        raise AssertionError("form editing opened or decoded a source")

    monkeypatch.setattr(SpecSource, "__init__", forbidden)
    monkeypatch.setattr(SpecSource, "load_frame", forbidden)
    dialog, owner, _status = _dialog(qapp)
    try:
        _fill_form(dialog, tmp_path)
        exposure_name, exposure_occurrence = dialog.selector_edits["exposure"]
        exposure_name.setText("Seconds")
        exposure_occurrence.setValue(1)
        form = dialog._build_form()
        assert type(form) is RSMToolFormV2
        assert form.project_root == str(tmp_path)
        assert form.geometry_asset.locator == CANONICAL_RSM_GEOMETRY_LOCATOR
        assert len(form.members) == 1
        member = form.members[0]
        assert member.spec_path == str(tmp_path / "STO_align")
        assert member.image_dir == str(tmp_path / "images")
        assert form.normalization.exposure_selector.occurrence == 1
        assert form.bins == (40, 40, 40)
        assert len(dialog.surface_plots) == len(dialog.surface_items) == 6
        assert all(not control.isEnabled() for control in dialog.slice_index_controls)
    finally:
        assert dialog.shutdown().status is PageCleanup.CLEAN
    assert owner.closed == 1


def test_coordinate_frame_selector_binds_form_and_create_new_policy(qapp, tmp_path):
    dialog, _owner, _status = _dialog(qapp)
    try:
        _fill_form(dialog, tmp_path)
        initial = dialog._build_form()
        assert initial.coordinate_frame is RSMCoordinateFrame.HKL
        assert initial.overwrite is AnalysisArtifactOverwrite.CREATE_NEW
        assert dialog.output_policy_label.text() == "Create new (refuse if present)"
        dialog._prepared_form_fingerprint = initial.fingerprint
        prior_revision = dialog._form_revision

        dialog.coordinate_frame_combo.setCurrentIndex(1)

        selected = dialog._build_form()
        assert (
            selected.coordinate_frame
            is RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU
        )
        assert selected.fingerprint != initial.fingerprint
        assert dialog._form_revision == prior_revision + 1
        assert dialog._prepared_form_fingerprint is None
        assert not dialog.run_button.isEnabled()
    finally:
        dialog.shutdown()


def test_ordered_member_actions_change_exact_visible_tuple(qapp, tmp_path):
    dialog, _owner, _status = _dialog(qapp)
    try:
        _fill_form(dialog, tmp_path)
        one = dialog._build_form()
        dialog._add_scan42_member()
        forward = dialog._build_form()
        assert tuple(member.scan for member in forward.members) == ("43.1", "42.1")
        assert dialog.member_table.item(1, 3).text() == "161"
        dialog._move_member(-1)
        reverse = dialog._build_form()
        assert tuple(member.scan for member in reverse.members) == ("42.1", "43.1")
        assert forward.fingerprint != reverse.fingerprint != one.fingerprint
        dialog._remove_member()
        assert len(dialog._build_form().members) == 1
        assert not dialog.remove_member_button.isEnabled()
    finally:
        dialog.shutdown()


def test_install_canonical_presents_three_identities_and_revokes_preview(
    qapp, tmp_path
):
    dialog, _owner, _status = _dialog(qapp)
    try:
        _fill_form(dialog, tmp_path)
        dialog._prepared_form_fingerprint = "prepared"
        dialog.run_button.setEnabled(True)
        prior_revision = dialog._form_revision
        dialog._install_geometry_asset()
        target = tmp_path / CANONICAL_RSM_GEOMETRY_LOCATOR
        assert target.is_file()
        identity = dialog.geometry_identity.text()
        assert "raw " in identity and "semantic " in identity and "receipt " in identity
        assert dialog.geometry_asset_edit.text() == CANONICAL_RSM_GEOMETRY_LOCATOR
        assert dialog._prepared_form_fingerprint is None
        assert dialog._form_revision == prior_revision + 1
        assert not dialog.run_button.isEnabled()
        called = []
        dialog._owner.begin_run = lambda: called.append(True)
        dialog._begin_run()
        assert called == []
        assert "unchanged successful Preview" in dialog.status_label.text()
    finally:
        dialog.shutdown()


def test_any_form_edit_clears_all_six_and_disables_indices(qapp, tmp_path):
    dialog, _owner, _status = _dialog(qapp)
    try:
        _fill_form(dialog, tmp_path)
        dialog._painted_result_fingerprint = "a" * 64
        for item in dialog.surface_items[:3]:
            item.setImage(np.ones((2, 2), dtype=np.float32))
        for item in dialog.surface_items[3:]:
            item.setData([0, 1], [1, 2])
        for control in dialog.slice_index_controls:
            control.setEnabled(True)
        dialog.scan_edit.setText("44.1")
        assert dialog._painted_result_fingerprint is None
        assert all(item.image is None for item in dialog.surface_items[:3])
        assert all(item.getData()[0] is None for item in dialog.surface_items[3:])
        assert all(not control.isEnabled() for control in dialog.slice_index_controls)
        assert "previous result cleared" in dialog.status_label.text()
    finally:
        dialog.shutdown()


def test_common_normalization_occurrence_revokes_preview_and_result(
    qapp, tmp_path
):
    dialog, _owner, _status = _dialog(qapp)
    try:
        _fill_form(dialog, tmp_path)
        dialog._prepared_form_fingerprint = dialog._build_form().fingerprint
        dialog._painted_result_fingerprint = "a" * 64
        dialog.run_button.setEnabled(True)
        _name, occurrence = dialog.common_selector_edits["exposure"]
        occurrence.setValue(1)
        assert dialog._prepared_form_fingerprint is None
        assert dialog._painted_result_fingerprint is None
        assert not dialog.run_button.isEnabled()
        assert "previous result cleared" in dialog.status_label.text()
    finally:
        dialog.shutdown()


def test_new_preview_clears_previous_committed_surface(qapp, tmp_path, monkeypatch):
    dialog, owner, _status = _dialog(qapp)
    try:
        _fill_form(dialog, tmp_path)
        dialog._painted_result_fingerprint = "a" * 64
        for item in dialog.surface_items[:3]:
            item.setImage(np.ones((2, 2), dtype=np.float32))
        monkeypatch.setattr(
            owner,
            "begin_preflight",
            lambda: OperationIdentity(1, object()),
        )
        dialog._begin_preflight()
        dialog._poll_timer.stop()
        assert dialog._painted_result_fingerprint is None
        assert "new Preview" in dialog.result_facts.text()
    finally:
        dialog.shutdown()


def test_v2_preflight_presentation_lists_every_identity_member_and_dependency(
    qapp, tmp_path
):
    prepared = _prepared_v2(tmp_path)
    dialog, _owner, _status = _dialog(qapp)
    try:
        dialog._render_preflight(prepared.summary)
        shown = dialog.preview_text.toPlainText()
        for label in (
            "Asset raw:",
            "Asset semantic:",
            "Asset receipt:",
            "Effective geometry:",
            "Common grid:",
            "Union q bounds:",
            "Group source:",
            "Group preflight:",
            "Plan:",
            "Request:",
            "Output identity:",
            "Holds:",
        ):
            assert label in shown
        assert (
            "Source UB authority: authenticated source UB drives H/K/L conversion"
            in shown
        )
        for member in prepared.summary.members:
            assert f"Member {member.ordinal + 1}:" in shown
            assert member.source_fingerprint in shown
            assert member.table_fingerprint in shown
            assert member.member_preflight_fingerprint in shown
            assert f"energy: {member.energy_eV:.12g} eV" in shown
            for dependency in member.dependency_files:
                assert dependency in shown
        assert (
            prepared.summary.geometry_asset_raw_sha256
            in dialog.geometry_identity.text()
        )
    finally:
        dialog.shutdown()


def test_cartesian_q_preflight_and_surface_use_frame_accurate_labels(
    qapp,
    tmp_path,
    monkeypatch,
):
    prepared = _prepared_v2(
        tmp_path,
        RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU,
    )
    dialog, _owner, _status = _dialog(qapp)
    values = _q_values()
    _patch_results(monkeypatch, {"q": values})
    try:
        dialog._render_preflight(prepared.summary)
        shown = dialog.preview_text.toPlainText()
        assert "Coordinate frame: Q Cartesian" in shown
        assert "Stored axes: Qx [q_A^-1], Qy [q_A^-1], Qz [q_A^-1]" in shown
        assert "xrayutilities matrix policy: explicit-identity-ub-f8-v1" in shown
        assert (
            "Source UB authority: source UB is unused and non-driving; explicit "
            "identity drives Cartesian Q" in shown
        )
        assert "Union coordinate bounds: Qx " in shown
        assert "xrayutilities matrix: 1 0 0; 0 1 0; 0 0 1" in shown

        dialog._paint_result(_result("q"))
        assert dialog._surface_title_texts == (
            "QxQy slice · Qz index 4",
            "QxQz slice · Qy index 3",
            "QyQz slice · Qx index 2",
            "Qx mean projection",
            "Qy mean projection",
            "Qz mean projection",
        )
        assert tuple(label.text() for label in dialog.slice_index_labels) == (
            "Qx",
            "Qy",
            "Qz",
        )
        assert dialog._surface_axis_label_texts[:3] == (
            ("Qx (Å⁻¹)", "Qy (Å⁻¹)"),
            ("Qx (Å⁻¹)", "Qz (Å⁻¹)"),
            ("Qy (Å⁻¹)", "Qz (Å⁻¹)"),
        )
        assert "indices Qx/Qy/Qz 2/3/4" in dialog.result_facts.text()
    finally:
        dialog.shutdown()


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


def test_run_enablement_binds_exact_current_form_fingerprint(qapp, tmp_path):
    dialog, owner, _status = _dialog(qapp)
    try:
        _fill_form(dialog, tmp_path)
        form = dialog._build_form()

        class _Prepared:
            def __init__(self, exact):
                self.form = exact

            def is_current(self, candidate):
                return (
                    candidate is not None
                    and candidate.fingerprint == self.form.fingerprint
                )

        owner.form = form
        owner.prepared = _Prepared(form)
        dialog._prepared_form_fingerprint = form.fingerprint
        dialog._sync_actions()
        assert dialog.run_button.isEnabled()
        dialog.frame_stop.setText("59")
        assert not dialog.run_button.isEnabled()
        assert dialog._prepared_form_fingerprint is None
    finally:
        dialog.shutdown()


def test_progress_refusal_pending_and_stale_commit_never_paint(qapp, monkeypatch):
    from xdart.gui.tools import rsm_owner, rsm_tool

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
                ),
            )
        )
        for kind in (
            RSMOwnerOutcomeKind.CLEANUP_PENDING,
            RSMOwnerOutcomeKind.VERIFICATION_PENDING,
        ):
            dialog._accept_update(
                RSMOwnerUpdate(
                    identity,
                    RSMOwnerAction.RUN,
                    terminal_status=OperationTerminalStatus.RETURNED,
                    outcome=RSMOwnerOutcome(kind),
                )
            )
        monkeypatch.setattr(rsm_owner, "RSMOperationResultV2", _Result)
        monkeypatch.setattr(rsm_tool, "RSMOperationResultV2", _Result)
        result = _result(object())
        result.request = SimpleNamespace()
        dialog._active_form_revision = 4
        dialog._form_revision = 5
        dialog._accept_update(
            RSMOwnerUpdate(
                identity,
                RSMOwnerAction.RUN,
                terminal_status=OperationTerminalStatus.RETURNED,
                outcome=RSMOwnerOutcome(RSMOwnerOutcomeKind.RESULT, result=result),
            )
        )
        assert painted == []
        assert "was not painted" in dialog.status_label.text()
    finally:
        dialog.shutdown()


@pytest.mark.parametrize(
    ("method_name", "action", "finalization"),
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
def test_finalization_retry_after_edit_never_paints_retained_commit(
    qapp,
    monkeypatch,
    method_name,
    action,
    finalization,
):
    from xdart.gui.tools import rsm_owner, rsm_tool

    dialog, owner, _status = _dialog(qapp)
    monkeypatch.setattr(rsm_owner, "RSMOperationResultV2", _Result)
    monkeypatch.setattr(rsm_tool, "RSMOperationResultV2", _Result)
    result = _result(object())
    result.request = SimpleNamespace()
    painted = []
    monkeypatch.setattr(dialog, "_paint_result", painted.append)
    identity = OperationIdentity(1, object())
    begin_name = (
        "begin_retry_cleanup"
        if action is RSMOwnerAction.RETRY_CLEANUP
        else "begin_retry_verification"
    )
    monkeypatch.setattr(owner, begin_name, lambda: identity)
    owner.finalization = finalization
    dialog._execution_form_revision = 4
    dialog._form_revision = 4
    dialog._form_changed()
    try:
        getattr(dialog, method_name)()
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


def test_strict_commit_atomically_paints_six_and_central_controls(
    qapp, monkeypatch
):
    dialog, _owner, _status = _dialog(qapp)
    values = _values()
    _patch_results(monkeypatch, {"payload": values})
    old_items = dialog.surface_items
    updates = []
    original_updates = dialog.setUpdatesEnabled

    def record(enabled):
        updates.append(bool(enabled))
        original_updates(enabled)

    monkeypatch.setattr(dialog, "setUpdatesEnabled", record)
    try:
        dialog._paint_result(_result("payload"))
        assert updates == [False, True]
        assert len(dialog.surface_items) == 6
        assert dialog.surface_items != old_items
        assert all(
            item in plot.getPlotItem().items
            for plot, item in zip(
                dialog.surface_plots,
                dialog.surface_items,
                strict=True,
            )
        )
        assert all(
            item not in plot.getPlotItem().items
            for plot, item in zip(dialog.surface_plots, old_items, strict=True)
        )
        snapshot = dialog._painted_snapshot
        for item, product in zip(
            dialog.surface_items[:3],
            snapshot.products[:3],
            strict=True,
        ):
            assert isinstance(item, pg.ImageItem)
            np.testing.assert_array_equal(item.image, product.values)
        for item, product in zip(
            dialog.surface_items[3:],
            snapshot.products[3:],
            strict=True,
        ):
            assert isinstance(item, pg.PlotDataItem)
            x, y = item.getData()
            np.testing.assert_array_equal(x, product.x_axis)
            np.testing.assert_array_equal(y, product.values)
        assert tuple(
            control.value() for control in dialog.slice_index_controls
        ) == (2, 3, 4)
        assert tuple(
            control.maximum() for control in dialog.slice_index_controls
        ) == (4, 6, 8)
        assert all(control.isEnabled() for control in dialog.slice_index_controls)
        assert dialog._painted_result_fingerprint == values.result_fingerprint
    finally:
        dialog.shutdown()


@pytest.mark.parametrize("operation", ("add", "remove"))
@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("failure_timing", ("before", "after"))
def test_each_six_item_add_or_remove_failure_restores_previous_surface(
    qapp, monkeypatch, operation, index, failure_timing
):
    dialog, _owner, _status = _dialog(qapp)
    old_values = _values(offset=0.0)
    new_values = _values(offset=10.0)
    _patch_results(monkeypatch, {"old": old_values, "new": new_values})
    try:
        dialog._paint_result(_result("old"))
        old_items = dialog.surface_items
        old_facts = dialog.result_facts.text()
        old_snapshot = dialog._painted_snapshot
        method_name = "addItem" if operation == "add" else "removeItem"
        original = getattr(dialog.surface_plots[index], method_name)
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                if failure_timing == "after":
                    original(*args, **kwargs)
                raise RuntimeError(
                    f"{failure_timing}-{operation}-{index}"
                )
            return original(*args, **kwargs)

        monkeypatch.setattr(dialog.surface_plots[index], method_name, fail_once)
        with pytest.raises(
            RuntimeError,
            match=f"{failure_timing}-{operation}-{index}",
        ):
            dialog._paint_result(_result("new"))
        assert dialog.surface_items is old_items
        assert dialog._painted_snapshot is old_snapshot
        assert dialog.result_facts.text() == old_facts
        assert dialog._painted_result_fingerprint == old_values.result_fingerprint
        assert all(
            tuple(plot.getPlotItem().items) == (item,)
            for plot, item in zip(
                dialog.surface_plots,
                old_items,
                strict=True,
            )
        )
    finally:
        dialog.shutdown()


@pytest.mark.parametrize("failure", ("facts", "title"))
def test_late_facts_or_title_failure_restores_all_six(qapp, monkeypatch, failure):
    dialog, _owner, _status = _dialog(qapp)
    old_values = _values(offset=0.0)
    new_values = _values(offset=10.0)
    _patch_results(monkeypatch, {"old": old_values, "new": new_values})
    try:
        dialog._paint_result(_result("old"))
        old_items = dialog.surface_items
        old_facts = dialog.result_facts.text()
        old_titles = dialog._surface_title_texts
        calls = 0
        if failure == "facts":
            original = dialog.result_facts.setText

            def fail_once(value):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("facts failed")
                return original(value)

            monkeypatch.setattr(dialog.result_facts, "setText", fail_once)
        else:
            original = dialog.surface_plots[4].setTitle

            def fail_once(value):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("title failed")
                return original(value)

            monkeypatch.setattr(dialog.surface_plots[4], "setTitle", fail_once)
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            dialog._paint_result(_result("new"))
        assert dialog.surface_items is old_items
        assert dialog.result_facts.text() == old_facts
        assert dialog._surface_title_texts == old_titles
        assert all(
            tuple(plot.getPlotItem().items) == (item,)
            for plot, item in zip(
                dialog.surface_plots,
                old_items,
                strict=True,
            )
        )
    finally:
        dialog.shutdown()


def test_cross_frame_presentation_failure_restores_hkl_labels(qapp, monkeypatch):
    dialog, _owner, _status = _dialog(qapp)
    hkl_values = _values()
    q_values = _q_values()
    _patch_results(monkeypatch, {"hkl": hkl_values, "q": q_values})
    try:
        dialog._paint_result(_result("hkl"))
        old_items = dialog.surface_items
        old_snapshot = dialog._painted_snapshot
        old_titles = dialog._surface_title_texts
        old_axis_labels = dialog._surface_axis_label_texts
        old_slice_labels = tuple(
            label.text() for label in dialog.slice_index_labels
        )
        original = dialog.result_facts.setText
        calls = 0

        def fail_once(value):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("cross-frame facts failed")
            return original(value)

        monkeypatch.setattr(dialog.result_facts, "setText", fail_once)
        with pytest.raises(RuntimeError, match="cross-frame facts failed"):
            dialog._paint_result(_result("q"))
        assert dialog.surface_items is old_items
        assert dialog._painted_snapshot is old_snapshot
        assert dialog._surface_title_texts == old_titles
        assert dialog._surface_axis_label_texts == old_axis_labels
        assert tuple(
            label.text() for label in dialog.slice_index_labels
        ) == old_slice_labels
        assert dialog._viewer_values is hkl_values
    finally:
        dialog.shutdown()


def test_index_only_edit_uses_resident_cache_and_changes_no_form_authority(
    qapp, monkeypatch
):
    from xdart.gui.tools import rsm_tool

    dialog, owner, _status = _dialog(qapp)
    values = _values()
    _patch_results(monkeypatch, {"payload": values})
    try:
        dialog._paint_result(_result("payload"))
        dialog._prepared_form_fingerprint = "prepared"
        owner.form = object()
        owner.prepared = object()
        revision = dialog._form_revision
        old_items = dialog.surface_items
        monkeypatch.setattr(
            rsm_tool,
            "make_rsm_viewer_values",
            lambda *_args: pytest.fail("index edit reread payload"),
        )
        dialog.slice_index_controls[0].setValue(1)
        assert dialog._form_revision == revision
        assert dialog._prepared_form_fingerprint == "prepared"
        assert owner.form is not None and owner.prepared is not None
        assert dialog.surface_items != old_items
        assert dialog._painted_snapshot.state.h_index == 1
        assert dialog._painted_result_fingerprint == values.result_fingerprint
    finally:
        dialog.shutdown()


def test_index_presentation_failure_restores_visible_and_model_state(
    qapp, monkeypatch
):
    dialog, _owner, _status = _dialog(qapp)
    values = _values()
    _patch_results(monkeypatch, {"payload": values})
    try:
        dialog._paint_result(_result("payload"))
        old_items = dialog.surface_items
        old_snapshot = dialog._painted_snapshot
        original = dialog.surface_plots[2].addItem
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("index paint failed")
            return original(*args, **kwargs)

        monkeypatch.setattr(dialog.surface_plots[2], "addItem", fail_once)
        dialog.slice_index_controls[0].setValue(1)
        assert dialog.surface_items is old_items
        assert dialog._painted_snapshot is old_snapshot
        assert dialog._viewer_model.current_snapshot is old_snapshot
        assert tuple(
            control.value() for control in dialog.slice_index_controls
        ) == (2, 3, 4)
        assert "RSM_VIEW_PRESENTATION_FAILED" in dialog.status_label.text()
    finally:
        dialog.shutdown()


def test_poll_contains_presentation_failure_with_stable_code(qapp, monkeypatch):
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
        assert "RSM_VIEW_PRESENTATION_FAILED" in dialog.status_label.text()
        assert "presentation failed" in dialog.status_label.text()
        assert "paint fault" in dialog.status_label.text()
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
        assert len(handle.widget.surface_plots) == 6
        assert handle.activity.active() is False
        assert handle.close().status is PageCleanup.CLEAN
    finally:
        parent.close()
