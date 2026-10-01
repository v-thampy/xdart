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
    fingerprint names every member, selector, grid, frame and the output), for HKL and for Cartesian Q."""
    page = _page(qapp, owner=_Owner())
    dialog = RSMToolDialog(empty_host_services(_Status()), owner=_Owner())
    try:
        output = tmp_path / "rsm.nexus"
        page.set_field("project_root", str(tmp_path))
        page.set_field("output_path", str(output))
        dialog.project_edit.setText(str(tmp_path))
        dialog._apply_scan43_preset()
        dialog.output_edit.setText(str(output))
        assert page.build_form().fingerprint == dialog._build_form().fingerprint
        page.set_field("coordinates", RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.display_name)
        page.set_field("grid", "Full 200×200×200")
        page.set_field("normalization", "Identity")
        dialog.coordinate_frame_combo.setCurrentIndex(
            dialog.coordinate_frame_combo.findData(RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU))
        dialog.grid_combo.setCurrentIndex(1)
        dialog.normalization_combo.setCurrentIndex(1)
        assert page.build_form().fingerprint == dialog._build_form().fingerprint
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
    every member and identity; Run previews, grids and paints the committed result's six products on the
    shared panes, lists it, and the readiness line then names the existing output."""
    _commit_instead_of_gridding(monkeypatch, _values())
    page = _page(qapp)
    try:
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        assert page._readiness[1], page._readiness
        output = Path(page._value("output_path"))
        assert str(output) == suggested_output(tmp_path, page._members)
        assert not output.parent.exists()  # Preview makes the Output folder the operation pins
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready" in page.run_bar.readinessLabel.full_text())
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
        output.write_bytes(b"")  # the run's file, as a real commit leaves it
        page.set_field("grid", "Full 200×200×200")  # a repeat: the next edit names the existing file
        assert page._readiness == (f"Output exists · choose another name ({output.name})", False)
        assert not page.run_bar.startButton.isEnabled()
        page.set_field("output_path", str(tmp_path / "again.nexus"))  # a repeat names another file
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


def test_open_reads_saved_results_off_the_gui_thread_and_keeps_the_newest_two(qapp, tmp_path, monkeypatch):
    """Open… reads a record on the shared single-worker owner and paints its six. The Results list shows a
    kept result at once and reads an older one again: two are kept (32 MB each at Full 200³, probe 1)."""
    shown = {name: _values(offset=float(index)) for index, name in enumerate(("one", "two", "three"))}
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
        assert [page.results.item(row).text() for row in range(3)] == [f"{name}_rsm.nexus" for name in shown]
        page.results.setCurrentRow(1)  # kept: shown at once
        assert page.view.shown[0] is shown["two"] and len(reads) == 3
        page.results.setCurrentRow(0)  # the oldest: read again
        _wait(page, qapp, lambda: not page.active())
        assert page.view.shown[0] is shown["one"] and reads == [
            "one_rsm.nexus", "two_rsm.nexus", "three_rsm.nexus", "one_rsm.nexus"]
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
