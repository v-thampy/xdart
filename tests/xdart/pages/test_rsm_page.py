"""The native RSM page (AUD-9, stage 1). The maintainer, 2026-09-30: "Can you start aud 9 in a separate
branch while codex reviews and I test". The RSM dialog's workflow in the main workspace on the shared
Controls panel, run bar and scientific panes, over the one RSM v2 owner."""
from __future__ import annotations

from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from pyqtgraph.Qt import QtCore
from xdart.gui.pages.operation_owner import SingleWorkerOwner
from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.rsm import page as page_module
from xdart.gui.tabs.rsm.page import RSMPage, suggested_output
from xdart.gui.tabs.rsm.result_view import RSMResultView
from xdart.gui.tools import rsm_owner
from xdart.gui.tools.rsm_tool import RSMToolDialog
from xdart.gui.tools.rsm_values import RSMMemberDraft
from xrd_tools.analysis.module_transaction import ModuleDisposition, ModuleTerminalResult
from xrd_tools.analysis.rsm_geometry_asset import install_canonical_rsm_geometry_asset
from xrd_tools.analysis.rsm_operation import RSMOperationExecutionV2, RSMOperationResultV2
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from tests.core.test_rsm_operation_r2_real_data import _SCIENCE, rsm_r2_project  # noqa: F401
from tests.xdart.tools.test_rsm_tool_dialog import (  # noqa: F401
    _Owner, _Status, _q_values, _values, _write_member, qapp,
)


def _page(qapp, **owners):
    return RSMPage(empty_host_services(_Status()), **owners)


def _close(page, qapp):
    assert page.close_page().status is PageCleanup.CLEAN
    page.deleteLater()
    qapp.processEvents()


def _wait(page, qapp, done, seconds=20.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        page._poll()
        if done():
            return
        time.sleep(0.005)
    raise AssertionError(page.run_bar.readinessLabel.full_text())


def test_the_page_builds_the_dialogs_exact_form(qapp, tmp_path):
    """The page and the RSM dialog it replaces make the same exact RSM v2 form from the same inputs (its
    fingerprint names every member, selector, grid, frame and the output), for HKL and for Cartesian Q; the
    page's replaces its own result (stage 2), the dialog's creates a new one."""
    from dataclasses import replace as replaced
    page = _page(qapp, owner=_Owner())
    dialog = RSMToolDialog(empty_host_services(_Status()), owner=_Owner())
    try:
        output = tmp_path / "rsm.nexus"
        page.set_field("project_root", str(tmp_path))
        page.set_field("output_path", str(output))
        dialog.project_edit.setText(str(tmp_path))
        dialog._apply_scan43_preset()
        dialog.output_edit.setText(str(output))
        assert page.build_form().fingerprint == replaced(
            dialog._build_form(), overwrite=AnalysisArtifactOverwrite.REPLACE).fingerprint
        page.set_field("coordinates", RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.display_name)
        page.set_field("grid", "Full 200×200×200")
        page.set_field("normalization", "Identity")
        dialog.coordinate_frame_combo.setCurrentIndex(
            dialog.coordinate_frame_combo.findData(RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU))
        dialog.grid_combo.setCurrentIndex(1)
        dialog.normalization_combo.setCurrentIndex(1)
        assert page.build_form().fingerprint == replaced(
            dialog._build_form(), overwrite=AnalysisArtifactOverwrite.REPLACE).fingerprint
    finally:
        _close(page, qapp)
        dialog.shutdown()


def _synthetic_members(root):
    """Two ordered synthetic SPEC members on disk (the RSM dialog tests' fixture) and the canonical asset."""
    install_canonical_rsm_geometry_asset(project_root=root)
    drafts = []
    for ordinal in range(2):
        form = _write_member(root, ordinal)
        selectors = tuple(selector for _role, selector in form.motor_selectors)
        drafts.append(RSMMemberDraft(
            str(Path(form.spec_path).relative_to(root)), form.scan, str(Path(form.image_dir).relative_to(root)),
            form.image_stem, 0, "1", 1, form.detector_shape[0], form.detector_shape[1], "uint16", 0,
            tuple(selector.name for selector in selectors), tuple(selector.occurrence for selector in selectors)))
    return drafts


def _with_members(page, root, drafts):
    page.set_field("project_root", str(root))
    page._members = drafts
    page._load_member(0)
    page.set_field("normalization", "Identity")
    page.set_field("offset", "0")
    page.set_field("high_threshold", "")
    page.set_field("static_hot", "")


class _Committed:
    """A committed RSM v2 result as the owner and the page read one (the dialog tests' _Result): a real
    commit needs the publisher's receipt."""


def _commit_instead_of_gridding(monkeypatch, values):
    """The execution commits at once, its payload the given viewer values."""
    def run(self, *, cancel_token=None, progress_callback=None):
        result = _Committed()
        result.request = self.request
        result.terminal = SimpleNamespace(disposition=ModuleDisposition.COMMITTED)
        result.payload = "payload"
        return result
    monkeypatch.setattr(RSMOperationExecutionV2, "run", run)
    monkeypatch.setattr(rsm_owner, "RSMOperationResultV2", _Committed)
    monkeypatch.setattr(page_module, "RSMOperationResultV2", _Committed)
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", {"payload": values}.__getitem__)


def test_preview_lists_every_identity_and_run_paints_the_committed_six(qapp, tmp_path, monkeypatch):
    """Preview through the one RSM v2 owner (its real preflight on two ordered synthetic members) lists
    every member and identity and creates no folder (stage 2: the run makes the output's); Run previews,
    grids and paints the committed result's six products on the shared panes and lists it. An existing
    output leaves Run available: Append ⇄ Replace decides (stage 2)."""
    _commit_instead_of_gridding(monkeypatch, _values())
    page = _page(qapp)
    try:
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        assert page._readiness[1], page._readiness
        output = Path(page._value("output_path"))
        assert str(output) == suggested_output(tmp_path, page._members)
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready" in page.run_bar.readinessLabel.full_text())
        assert not output.parent.exists()  # Preview creates nothing (stage 2)
        shown = page.preview_text.toPlainText()
        assert "Ordered members: 2 · total frames 4" in shown and "Member 2:" in shown
        page.begin_run()
        _wait(page, qapp, lambda: page.view.shown is not None)
        snapshot = page.view.shown[1]
        for pane, product in zip(page.view.slices, snapshot.products[:3], strict=True):
            np.testing.assert_array_equal(pane.canvas.raw_image, np.asarray(product.values).T)
        for curve, product in zip(page.view.curves, snapshot.products[3:], strict=True):
            x, y = curve.getData()
            np.testing.assert_array_equal(x, product.x_axis)
            np.testing.assert_array_equal(y, product.values)
        assert [control.value() for control in page.view.index_controls] == [2, 3, 4]
        page.view.index_controls[0].setValue(0)  # H: the KL slice moves to the new index
        moved = page.view.shown[1]
        assert (moved.state.h_index, moved.state.k_index, moved.state.l_index) == (0, 3, 4)
        np.testing.assert_array_equal(page.view.slices[2].canvas.raw_image, np.asarray(moved.products[2].values).T)
        assert page.results.count() == 1 and page.results.item(0).text() == output.name
        assert page.run_bar.readinessLabel.full_text() == f"Saved {output.name}"
        output.parent.mkdir()
        output.write_bytes(b"")  # the run's file, as a real commit leaves it
        page.set_field("grid", "Full 200×200×200")  # a repeat over the existing output
        assert page._readiness[1] and page.run_bar.startButton.isEnabled()
    finally:
        _close(page, qapp)


def test_typing_in_a_field_then_run_runs_the_typed_inputs(qapp, tmp_path, monkeypatch):
    """Stitch's live lesson: on macOS a click on Run leaves the focus in the field being typed in, and
    disabling the controls for the check then committed the typing mid-check ("the inputs changed"). The
    typing commits before the check starts, and the run writes the typed Output."""
    from PySide6.QtTest import QTest
    _commit_instead_of_gridding(monkeypatch, _values())
    page = _page(qapp)
    page.show()
    try:
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        page.activateWindow()
        editor = page._rows["output_path"].editor
        editor.setFocus()
        qapp.processEvents()
        editor.selectAll()
        QTest.keyClicks(editor, str(tmp_path / "typed.nexus"))  # typed, not committed: no Return or Tab
        assert editor.hasFocus()
        page.run_bar.startButton.click()  # a click that leaves the focus in the field
        _wait(page, qapp, lambda: page.view.shown is not None)
        assert page.results.item(0).text() == "typed.nexus"
    finally:
        _close(page, qapp)


def test_stop_cancels_a_run(qapp, tmp_path, monkeypatch):
    """Stop cancels the owner's run; the operation's cancelled terminal (its own result type) shows."""
    started = []

    def run(self, *, cancel_token=None, progress_callback=None):
        started.append(True)
        deadline = time.monotonic() + 10
        while not cancel_token.is_set():
            if time.monotonic() > deadline:
                raise AssertionError("Stop never reached the operation")
            time.sleep(0.005)
        return RSMOperationResultV2(self.request, ModuleTerminalResult(
            self.request.module, ModuleDisposition.CANCELLED, "CANCELLED"))
    monkeypatch.setattr(RSMOperationExecutionV2, "run", run)
    page = _page(qapp)
    try:
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        page.begin_run()
        _wait(page, qapp, lambda: bool(started))
        assert page.active() and page.run_bar.stopButton.isEnabled()
        start = page.run_bar.startButton  # RSM has no pause: Run stays Run, held while it runs
        assert (start.text(), start.isEnabled()) == ("▶ Run", False)
        page.stop()
        _wait(page, qapp, lambda: not page.active())
        assert page.run_bar.readinessLabel.full_text() == "RSM stopped"
        assert page.view.shown is None and page.results.count() == 0 and page._readiness[1]
    finally:
        _close(page, qapp)


def test_open_reads_saved_results_off_the_gui_thread_and_keeps_the_newest_five(qapp, tmp_path, monkeypatch):
    """Open… reads a record on the shared single-worker owner and paints its six. The Results list shows a
    kept result at once and reads an older one again: five are kept (the maintainer, 2026-10-01: "about 4-5
    for quick access"; 32 MB each at Full 200³, probe 3)."""
    names = ("one", "two", "three", "four", "five", "six")
    shown = {name: _values(offset=float(index)) for index, name in enumerate(names)}
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", shown.__getitem__)
    reads = []

    def read(path, _cancel, _progress):
        reads.append(Path(path).name)
        return Path(path).name.split("_")[0]
    page = _page(qapp, owner=_Owner(), reader=SingleWorkerOwner(read))
    try:
        for name in shown:
            assert page.reopen(str(tmp_path / f"{name}_rsm.nexus"))
            _wait(page, qapp, lambda: not page.active())
            assert page.view.shown[0] is shown[name]
        assert [page.results.item(row).text() for row in range(6)] == [f"{name}_rsm.nexus" for name in shown]
        page.results.setCurrentRow(1)  # kept: shown at once
        assert page.view.shown[0] is shown["two"] and len(reads) == 6
        page.results.setCurrentRow(0)  # the oldest: read again
        _wait(page, qapp, lambda: not page.active())
        assert page.view.shown[0] is shown["one"] and reads == [f"{name}_rsm.nexus" for name in (*names, "one")]
    finally:
        _close(page, qapp)


def test_a_failed_paint_keeps_the_six_shown(qapp, monkeypatch):
    """All six or none: a pane failing mid-paint leaves the prior result's six on screen."""
    from xrd_tools.session.rsm_viewer_model import RSMViewerModel
    page = _page(qapp, owner=_Owner())
    try:
        values = _values()
        model = RSMViewerModel()
        page.view.show_snapshot(values, model.snapshot(values))
        prior = page.view.shown
        before = [np.array(pane.canvas.raw_image) for pane in page.view.slices]
        calls = []
        original = page.view.slices[1].render

        def failing(*args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                raise RuntimeError("pane failed")
            return original(*args, **kwargs)
        monkeypatch.setattr(page.view.slices[1], "render", failing)
        other = _values(offset=5.0)
        with pytest.raises(RuntimeError, match="pane failed"):
            page.view.show_snapshot(other, RSMViewerModel().snapshot(other))
        assert page.view.shown is prior
        for pane, image in zip(page.view.slices, before, strict=True):
            np.testing.assert_array_equal(pane.canvas.raw_image, image)
    finally:
        _close(page, qapp)


@pytest.mark.parametrize(("make", "names"), (
    (_values, ("H", "K", "L")), (_q_values, ("Qx (Å⁻¹)", "Qy (Å⁻¹)", "Qz (Å⁻¹)"))), ids=("hkl", "q"))
def test_each_pane_carries_the_frames_own_axis_labels(qapp, make, names):
    """Each axis is labelled as its frame and the RSM dialog label it (a Cartesian Q axis as Qx, Qy or Qz
    in Å⁻¹, not as Int's 1-D "Q"), and each slice title names the index it is taken at."""
    from xrd_tools.session.rsm_viewer_model import RSMViewerModel
    view = RSMResultView()
    try:
        values = make()
        view.show_snapshot(values, RSMViewerModel().snapshot(values))
        first, second, third = names
        assert [(pane.plot.getAxis("bottom").labelText, pane.plot.getAxis("left").labelText)
                for pane in view.slices] == [(first, second), (first, third), (second, third)]
        assert [plot.getAxis("bottom").labelText for plot in view.projections] == list(names)
        assert {pane.plot.getAxis(side).labelUnits for pane in view.slices for side in ("bottom", "left")} | {
            plot.getAxis("bottom").labelUnits for plot in view.projections} == {""}
        symbols = values.coordinate_frame.axis_symbols
        assert view.slices[0].plot.titleLabel.text == f"{symbols[0]}{symbols[1]} slice · {symbols[2]} index 4"
    finally:
        view.deleteLater()
        qapp.processEvents()


@pytest.mark.slow
def test_scan43_through_the_page_commits_and_reopens_the_reference_result(qapp, rsm_r2_project):  # noqa: F811
    """Real data: STO scan 43 at 40³ run from the page (its preset member, the canonical asset, the
    suggested output) commits the independent oracle's pinned result (test_rsm_operation_r2_real_data),
    paints it, and Open… reads the same result back from the file."""
    expected = _SCIENCE[((43,), 40)]["result"]
    page = _page(qapp)
    try:
        page.set_field("project_root", str(rsm_r2_project))
        assert page._readiness[1], page._readiness
        page.begin_run()
        _wait(page, qapp, lambda: not page.active(), seconds=600)
        output = page._value("output_path")
        assert page.run_bar.readinessLabel.full_text() == f"Saved {Path(output).name}"
        committed = page._results[output]
        assert committed.result_fingerprint == expected
        assert page.view.shown[0].result_fingerprint == expected
        assert page.reopen(output)
        _wait(page, qapp, lambda: not page.active())
        reopened = page._results[output]
        assert reopened is not committed and reopened.result_fingerprint == expected
        assert page.view.shown[0].result_fingerprint == expected
        np.testing.assert_array_equal(page.view.shown[0].intensity, committed.intensity)
    finally:
        _close(page, qapp)


# Codex's stage-1 review (2026-10-01, REWORK), its reproductions adapted


@pytest.mark.parametrize("key", ("member_first", "member_step"))
def test_invalid_frame_text_stays_as_typed_and_is_refused(qapp, tmp_path, key):
    """F1: a First or Step that is not a whole number stays as typed, and readiness and the form refuse it;
    the member's earlier numbers no longer run unseen."""
    page = _page(qapp)
    try:
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        assert page._readiness[1], page._readiness
        page.set_field(key, "invalid")
        assert page._value(key) == "invalid" and page.member_table.item(0, 3).text() == "invalid"
        assert not page._readiness[1] and "member 1's frames" in page._readiness[0]
        assert not page.run_bar.startButton.isEnabled()
        with pytest.raises(ValueError, match="must be a whole number"):
            page.build_form()
    finally:
        _close(page, qapp)


def test_a_zero_frame_step_is_refused_without_a_table_exception(qapp, tmp_path):
    """F1: Step 0 divided in the member table's frame count before anything refused it."""
    page = _page(qapp)
    try:
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        page.set_field("member_step", "0")
        assert page.member_table.item(0, 3).text() == "invalid"
        assert not page._readiness[1] and "Frame step must be 1 or more" in page._readiness[0]
        with pytest.raises(ValueError, match="Frame step must be 1 or more"):
            page.build_form()
    finally:
        _close(page, qapp)


def test_a_pending_verification_keeps_a_clickable_retry(qapp, tmp_path, monkeypatch):
    """F2: the real owner, preflight, gridding and publisher; one transient failure of the strict reload
    leaves the saved result VERIFICATION_PENDING. The Controls editors lock, its action row does not: Retry
    is clickable and verifies the saved result."""
    import tifffile
    import xrd_tools.analysis.rsm_operation as operation
    page = _page(qapp)
    real_read = operation.read_analysis_artifact
    reads = []

    def fail_once(*args, **kwargs):
        reads.append(str(args[0]))
        if len(reads) == 1:
            raise OSError("transient read failure")
        return real_read(*args, **kwargs)

    try:
        drafts = _synthetic_members(tmp_path)
        for ordinal in range(2):
            images = tmp_path / drafts[ordinal].image_dir
            for label in range(2):
                tifffile.imwrite(images / f"member{ordinal}_{label:04d}.tif",
                                 np.full((195, 487), 10 + ordinal + label, dtype=np.uint16))
        _with_members(page, tmp_path, drafts)
        monkeypatch.setattr(operation, "read_analysis_artifact", fail_once)
        page.begin_run()
        _wait(page, qapp, lambda: page._owner.finalization is rsm_owner.RSMOwnerFinalization.VERIFICATION_PENDING)
        assert Path(page._value("output_path")).is_file()
        assert not page._rows["output_path"].isEnabled() and page.retry_button.isEnabled()
        page.retry_button.click()
        qapp.processEvents()
        assert page._owner.busy, "the Retry click did not start the retained verification"
        _wait(page, qapp, lambda: not page.active())
        assert page.view.shown is not None and len(reads) == 2
    finally:
        monkeypatch.setattr(operation, "read_analysis_artifact", real_read)
        _wait(page, qapp, lambda: page.close_page().status is PageCleanup.CLEAN)
        page.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("frame", tuple(RSMCoordinateFrame), ids=lambda frame: frame.value)
def test_each_slice_pixel_sits_at_its_bin_centre(qapp, frame):
    """F3: the shared pane spans an image over its axes' first and last values, so the slices give it their
    bins' outer edges (the dialog's half-bin bounds, now one headless helper): each pixel centre is its bin
    centre, as the projections plot them. Codex's example: H 10..30, K 100..106."""
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, project_analysis_artifact_result
    from xrd_tools.session.rsm_viewer_model import RSMViewerModel, make_rsm_viewer_values
    axes = (np.array([10, 20, 30.]), np.array([100, 102, 104, 106.]), np.array([-2, -1.5, -1, -.5, 0.]))
    values = make_rsm_viewer_values(project_analysis_artifact_result(
        kind=AnalysisArtifactKind.RSM, axes=tuple(zip(frame.axis_names, axes, strict=True)),
        axis_units=tuple(zip(frame.axis_names, frame.axis_units, strict=True)),
        intensity=np.arange(60, dtype=np.float64).reshape(3, 4, 5) + 1, sigma=None, coverage=None,
        normalization=None))
    view = RSMResultView()
    try:
        snapshot = RSMViewerModel().snapshot(values, h_index=1, k_index=2, l_index=3)
        view.show_snapshot(values, snapshot)
        for pane, product in zip(view.slices, snapshot.products[:3], strict=True):
            x, y = np.asarray(product.x_axis), np.asarray(product.y_axis_or_none)
            first = pane.image.mapToParent(QtCore.QPointF(.5, .5))
            last = pane.image.mapToParent(QtCore.QPointF(x.size - .5, y.size - .5))
            assert (first.x(), first.y(), last.x(), last.y()) == pytest.approx(
                (x[0], y[0], x[-1], y[-1]), abs=1e-9)
        first = view.slices[0].image.mapToParent(QtCore.QPointF(.5, .5))
        last = view.slices[0].image.mapToParent(QtCore.QPointF(2.5, 3.5))
        assert (first.x(), first.y(), last.x(), last.y()) == pytest.approx((10, 100, 30, 106), abs=1e-9)
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_a_late_read_does_not_override_the_newer_selection(qapp, tmp_path):
    """F4: real saved records on the shared single-worker reader. An evicted result A is selected (its
    read held back, as a slow disk would), then a kept B: A's read, when it returns, is kept but neither
    selected nor painted over B."""
    from threading import Event
    from tests.core.test_analysis_artifact import (
        test_analysis_artifact_v2_rsm_round_trip_is_a_closed_exact_branch as write_record,
    )
    paths = []
    for index in range(page_module._RESIDENT_RESULTS + 1):  # the first is evicted
        root = tmp_path / f"r{index}"
        root.mkdir()
        write_record(root, False)
        paths.append(str(root / "rsm-artifact-v2.nexus"))
    started, released, hold = Event(), Event(), []

    def read(path, cancel, progress):
        if hold and path == paths[0]:
            started.set()
            assert released.wait(10), "the held read was not released"
        return page_module._read_rsm_record(path, cancel, progress)

    page = _page(qapp, owner=_Owner(), reader=SingleWorkerOwner(read))
    role = QtCore.Qt.ItemDataRole.UserRole
    try:
        for path in paths:
            assert page.reopen(path)
            _wait(page, qapp, lambda: not page.active())
        assert page._results[paths[0]] is None and page._results[paths[1]] is not None
        hold.append(True)
        page.results.setCurrentRow(0)
        assert started.wait(3)
        page.results.setCurrentRow(1)
        shown = page.view.shown
        assert page.results.currentItem().data(role) == paths[1]
        released.set()
        _wait(page, qapp, lambda: not page.active())
        assert page.results.currentItem().data(role) == paths[1] and page.view.shown is shown
        assert page._results[paths[0]] is not None  # kept for its next selection
    finally:
        released.set()
        _wait(page, qapp, lambda: page.close_page().status is PageCleanup.CLEAN)
        page.deleteLater()
        qapp.processEvents()



# AUD-9 stage 2: Append ⇄ Replace like Stitch's (the maintainer, 2026-10-01: "This should have the same
# replace append option like stitch")


def _terminal_instead_of_gridding(monkeypatch, disposition, code, *, until_stopped=False):
    """The execution ends at once (or once stopped) with the given terminal, no payload."""
    def run(self, *, cancel_token=None, progress_callback=None):
        deadline = time.monotonic() + 10
        while until_stopped and not cancel_token.is_set():
            if time.monotonic() > deadline:
                raise AssertionError("Stop never reached the operation")
            time.sleep(0.005)
        result = _Committed()
        result.request = self.request
        result.terminal = SimpleNamespace(disposition=disposition, code=code, diagnostic=None)
        result.payload = None
        return result
    monkeypatch.setattr(RSMOperationExecutionV2, "run", run)


def _leaving_the_file(monkeypatch):
    """The execution stand-in leaves its output file, as a real commit does."""
    stand_in = RSMOperationExecutionV2.run

    def run(self, **kwargs):
        target = Path(self.request.module.output.target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
        return stand_in(self, **kwargs)
    monkeypatch.setattr(RSMOperationExecutionV2, "run", run)


def test_append_skips_an_up_to_date_output_and_replace_grids_it_again(qapp, tmp_path, monkeypatch):
    """Stitch's semantics: the run bar's toggle (Append by default) is live. In Append a Run skips an output
    whose stored result is this request's (the preflight's rsm_output_current) and says so; a stale one is
    gridded again; in Replace every Run grids. The form is REPLACE throughout: Append is the page's skip."""
    from xdart.gui.tools import rsm_values
    _commit_instead_of_gridding(monkeypatch, _values())
    committed_run, runs = RSMOperationExecutionV2.run, []

    def counted(self, **kwargs):
        runs.append(self.request.module.output.overwrite)
        return committed_run(self, **kwargs)
    monkeypatch.setattr(RSMOperationExecutionV2, "run", counted)
    current = [True]
    monkeypatch.setattr(rsm_values, "rsm_output_current", lambda _request: current[0])
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        output = Path(page._value("output_path"))
        output.parent.mkdir()
        output.write_bytes(b"")  # a result already there
        toggle = page.run_bar.writeModeButton
        assert toggle.isEnabled() and page.run_bar.write_mode() == "Append"
        assert toggle.toolTip().startswith("Append: skip the output")
        page.begin_run()
        _wait(page, qapp, lambda: "Up to date" in label.full_text())
        assert runs == [] and label.full_text() == (
            f"Up to date · {output.name} already gridded with these settings (Replace ⇄ grids again)")
        assert "already gridded with these settings (Append skips it)" in page.preview_text.toPlainText()
        current[0] = False  # stale: Append grids it again
        page.begin_run()
        _wait(page, qapp, lambda: len(runs) == 1 and not page.active())
        current[0] = True
        toggle.click()  # Replace
        assert page.run_bar.write_mode() == "Overwrite" and toggle.toolTip().startswith("Replace: grid again")
        page.begin_run()
        _wait(page, qapp, lambda: len(runs) == 2 and not page.active())
        assert runs == [AnalysisArtifactOverwrite.REPLACE] * 2
    finally:
        _close(page, qapp)


def test_a_replace_repaints_the_new_result_under_the_same_name(qapp, tmp_path, monkeypatch):
    """A Run that replaces the output shows its new result in place of the old one, under the one listed
    name."""
    first, second = _values(), _values(offset=7.0)
    _commit_instead_of_gridding(monkeypatch, first)
    _leaving_the_file(monkeypatch)
    page = _page(qapp)
    try:
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        page.run_bar.writeModeButton.click()  # Replace
        page.begin_run()
        _wait(page, qapp, lambda: page.view.shown is not None)
        assert page.view.shown[0] is first
        assert Path(page._value("output_path")).is_file()
        monkeypatch.setattr(page_module, "make_rsm_viewer_values", {"payload": second}.__getitem__)
        page.begin_run()
        _wait(page, qapp, lambda: page.view.shown[0] is second and not page.active())
        assert page.results.count() == 1
        np.testing.assert_array_equal(page.view.slices[0].canvas.raw_image,
                                      np.asarray(page.view.shown[1].products[0].values).T)
    finally:
        _close(page, qapp)


def test_a_refused_or_stopped_replace_keeps_the_prior_result_shown(qapp, tmp_path, monkeypatch):
    """A replace the publisher refuses (a foreign or other-kind file at the target) or one stopped while
    gridding leaves the prior result listed and shown (its file is replaced only at commit)."""
    _commit_instead_of_gridding(monkeypatch, _values())
    _leaving_the_file(monkeypatch)
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        page.run_bar.writeModeButton.click()  # Replace
        page.begin_run()
        _wait(page, qapp, lambda: page.view.shown is not None and not page.active())
        prior = page.view.shown
        assert Path(page._value("output_path")).is_file()
        _terminal_instead_of_gridding(monkeypatch, ModuleDisposition.REFUSED, "OUTPUT_NOT_PREVIOUS_ANALYSIS")
        page.begin_run()
        _wait(page, qapp, lambda: "refused" in label.full_text() and not page.active())
        assert label.full_text() == "RSM refused: OUTPUT_NOT_PREVIOUS_ANALYSIS"
        assert page.view.shown is prior and page.results.count() == 1
        _terminal_instead_of_gridding(monkeypatch, ModuleDisposition.CANCELLED, "CANCELLED", until_stopped=True)
        page.begin_run()
        _wait(page, qapp, lambda: page._owner.busy and page.run_bar.stopButton.isEnabled())
        page.stop()
        _wait(page, qapp, lambda: "stopped" in label.full_text() and not page.active())
        assert page.view.shown is prior and page.results.count() == 1
    finally:
        _close(page, qapp)


def test_each_section_header_shows_its_readiness_as_ints_do(qapp, tmp_path):
    """Codex's reuse audit (REUSE.md): the page did not drive its sections' readiness headers. Each card now
    takes Int's ``SectionHeaderProjection`` through Int's ``apply_section_header``: its status, its ✓ when the
    section is ready, and what it still needs as the tip."""
    page = _page(qapp)
    try:
        cards = {"project": page.controls.project_card, "source": page.controls.source_card,
                 "experiment": page.controls.experiment_card, "processing": page.controls.processing_card}
        page.set_field("project_root", "")
        assert not cards["project"].valid_marker.isVisibleTo(page)
        assert cards["project"].valid_marker.toolTip() == "Needs a Project folder"
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        assert page.run_bar.readinessLabel.full_text().startswith("Ready · RSM · 2 members")
        assert all(card.valid_marker.isVisibleTo(page) for card in cards.values())
        assert cards["source"].status.text() == "2 members" and cards["processing"].status.text().startswith("grid ")
        page._members[1].spec_path = "missing.spec"
        page._edited()
        assert not cards["source"].valid_marker.isVisibleTo(page)
        assert cards["source"].valid_marker.toolTip() == "Needs member 2's SPEC file"
        assert cards["project"].valid_marker.isVisibleTo(page) and cards["experiment"].valid_marker.isVisibleTo(page)
    finally:
        _close(page, qapp)
