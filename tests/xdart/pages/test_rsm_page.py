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
from xdart.gui.tabs.rsm.display import rsm_keyed_projection
from xdart.gui.tabs.scattering.scientific_view import ScientificView
from xdart.gui.tools import rsm_owner
from xdart.gui.tools.rsm_tool import RSMToolDialog
from xdart.gui.tools.rsm_values import RSMMemberDraft
from xrd_tools.analysis.module_transaction import ModuleDisposition, ModuleTerminalResult
from xrd_tools.analysis.rsm_operation import RSMOperationExecutionV2, RSMOperationResultV2
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from tests.core.test_rsm_calibration_geometry import _canonical_partner
from tests.core.test_rsm_operation_r2_real_data import _SCIENCE, rsm_r2_project  # noqa: F401
from tests.xdart.tools.test_rsm_tool_dialog import (  # noqa: F401
    _Owner, _Status, _q_values, _values, _write_member, qapp,
)



def _keys(view):
    return [key for row in view.keyed_projection.layout for key in row]


def _slices(view):
    """The three slices' panes on Int's view, in the layout's order."""
    return [view.keyed_panel(key).pane for key in _keys(view)[:3]]


def _curves(view):
    return [view.keyed_panel(key).curve for key in _keys(view)[3:]]


def _plots(view):
    return [view.keyed_panel(key).plot for key in _keys(view)[3:]]


def _selectors(view):
    """Each slice's index selector: the HK slice's steps L, the HL slice's K, the KL slice's H."""
    return [view.keyed_panel(key).selector for key in _keys(view)[:3]]

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
    page's replaces its own result (stage 2), the dialog's creates a new one, the page's geometry is its
    Calibration row's where the dialog installs the canonical asset (stage 2b C7b), and the page's Max chunk
    MiB is Auto where the dialog's is the preset's 256 (stage 2b S3a)."""
    from dataclasses import replace as replaced
    page = _page(qapp, owner=_Owner())
    dialog = RSMToolDialog(empty_host_services(_Status()), owner=_Owner())
    try:
        output = tmp_path / "rsm.nexus"
        page.set_field("project_root", str(tmp_path))
        page.set_field("geometry_path", _canonical_partner(tmp_path))
        page.set_field("output_path", str(output))
        dialog.project_edit.setText(str(tmp_path))
        dialog._apply_scan43_preset()
        dialog.output_edit.setText(str(output))
        geometry = page.build_form().geometry_asset
        assert page.build_form().fingerprint == replaced(
            dialog._build_form(), overwrite=AnalysisArtifactOverwrite.REPLACE, geometry_asset=geometry,
            max_chunk_bytes=None).fingerprint
        page.set_field("coordinates", RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.display_name)
        page.set_field("grid", "Full 200×200×200")
        page.set_field("normalization", "Identity")
        dialog.coordinate_frame_combo.setCurrentIndex(
            dialog.coordinate_frame_combo.findData(RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU))
        dialog.grid_combo.setCurrentIndex(1)
        dialog.normalization_combo.setCurrentIndex(1)
        assert page.build_form().fingerprint == replaced(
            dialog._build_form(), overwrite=AnalysisArtifactOverwrite.REPLACE, geometry_asset=geometry,
            max_chunk_bytes=None).fingerprint
    finally:
        _close(page, qapp)
        dialog.shutdown()


def _synthetic_members(root):
    """Two ordered synthetic SPEC members on disk (the RSM dialog tests' fixture) and a calibration encoding the
    canonical geometry (row B's, AUD-9 stage 2b C7b: the page's geometry is its Calibration row's)."""
    _canonical_partner(root)
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
    page.set_field("geometry_path", "calibration/rsm_canonical.json")
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
        _wait(page, qapp, lambda: page.shown is not None)
        snapshot = page.shown[1]
        for pane, product in zip(_slices(page.view), snapshot.products[:3], strict=True):
            np.testing.assert_array_equal(pane.canvas.raw_image, np.asarray(product.values).T)
        for curve, product in zip(_curves(page.view), snapshot.products[3:], strict=True):
            x, y = curve.getData()
            np.testing.assert_array_equal(x, product.x_axis)
            np.testing.assert_array_equal(y, product.values)
        assert [selector.currentIndex() for selector in _selectors(page.view)] == [4, 3, 2]  # L, K, H
        _selectors(page.view)[2].setCurrentIndex(0)  # the KL slice's H index: it moves to H 0
        moved = page.shown[1]
        assert (moved.state.h_index, moved.state.k_index, moved.state.l_index) == (0, 3, 4)
        np.testing.assert_array_equal(_slices(page.view)[2].canvas.raw_image, np.asarray(moved.products[2].values).T)
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
        _wait(page, qapp, lambda: page.shown is not None)
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
        assert page.shown is None and page.results.count() == 0 and page._readiness[1]
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
            assert page.shown[0] is shown[name]
        assert [page.results.item(row).text() for row in range(6)] == [f"{name}_rsm.nexus" for name in shown]
        page.results.setCurrentRow(1)  # kept: shown at once
        assert page.shown[0] is shown["two"] and len(reads) == 6
        page.results.setCurrentRow(0)  # the oldest: read again
        _wait(page, qapp, lambda: not page.active())
        assert page.shown[0] is shown["one"] and reads == [f"{name}_rsm.nexus" for name in (*names, "one")]
    finally:
        _close(page, qapp)


def test_a_failed_paint_keeps_the_six_shown(qapp, monkeypatch):
    """All six or none: a pane failing mid-paint leaves the prior result's six on screen."""
    from xrd_tools.session.rsm_viewer_model import RSMViewerModel
    page = _page(qapp, owner=_Owner())
    try:
        values = _values()
        model = RSMViewerModel()
        page.view.reconcile_keyed(rsm_keyed_projection(values, model.snapshot(values)))
        prior = page.view.keyed_projection
        before = [np.array(pane.canvas.raw_image) for pane in _slices(page.view)]
        calls = []
        original = _slices(page.view)[1].render

        def failing(*args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                raise RuntimeError("pane failed")
            return original(*args, **kwargs)
        monkeypatch.setattr(_slices(page.view)[1], "render", failing)
        other = _values(offset=5.0)
        with pytest.raises(RuntimeError, match="pane failed"):
            page.view.reconcile_keyed(rsm_keyed_projection(other, RSMViewerModel().snapshot(other)))
        assert page.view.keyed_projection is prior
        for pane, image in zip(_slices(page.view), before, strict=True):
            np.testing.assert_array_equal(pane.canvas.raw_image, image)
    finally:
        _close(page, qapp)


@pytest.mark.parametrize(("make", "names"), (
    (_values, ("H", "K", "L")), (_q_values, ("Qx (Å⁻¹)", "Qy (Å⁻¹)", "Qz (Å⁻¹)"))), ids=("hkl", "q"))
def test_each_pane_carries_the_frames_own_axis_labels(qapp, make, names):
    """Each axis is labelled as its frame and the RSM dialog label it (a Cartesian Q axis as Qx, Qy or Qz
    in Å⁻¹, not as Int's 1-D "Q"), and each slice title names the index it is taken at."""
    from xrd_tools.session.rsm_viewer_model import RSMViewerModel
    view = ScientificView()
    try:
        values = make()
        view.reconcile_keyed(rsm_keyed_projection(values, RSMViewerModel().snapshot(values)))
        first, second, third = names
        assert [(pane.plot.getAxis("bottom").labelText, pane.plot.getAxis("left").labelText)
                for pane in _slices(view)] == [(first, second), (first, third), (second, third)]
        assert [plot.getAxis("bottom").labelText for plot in _plots(view)] == list(names)
        assert {pane.plot.getAxis(side).labelUnits for pane in _slices(view) for side in ("bottom", "left")} | {
            plot.getAxis("bottom").labelUnits for plot in _plots(view)} == {""}
        symbols = values.coordinate_frame.axis_symbols
        normal = float(np.asarray(values.axes[2][1])[4])
        assert _slices(view)[0].plot.titleLabel.text == (
            f"{symbols[0]}{symbols[1]} slice · {symbols[2]} = {normal:.6g} (index 4)")
    finally:
        view.deleteLater()
        qapp.processEvents()


@pytest.mark.slow
def test_scan43_through_the_page_commits_and_reopens_the_reference_result(qapp, rsm_r2_project):  # noqa: F811
    """Real data: STO scan 43 at 40³ run from the page (its preset member, a calibration encoding the canonical
    geometry on its Calibration row (row B's, bit for bit the canonical asset's), the suggested output) commits
    the same result as the headless run of the page's own form, paints it, and Open… reads the same result back
    from the file.

    The page's form carries decision 5's defaults (solid angle and polarization on), so its result is not the
    independent oracle's pinned one (test_rsm_operation_r2_real_data, every correction off): with them off, the
    form reproduces it. A default-on scan-43 reference is pinned after C7's integer-HKL row (decision 2)."""
    from dataclasses import replace
    from xrd_tools.analysis.rsm_operation import RSMFrameCorrections, run_rsm_operation_v2
    from xdart.gui.tools.rsm_values import prepare_rsm_tool_v2
    page = _page(qapp)
    try:
        page.set_field("project_root", str(rsm_r2_project))
        page.set_field("geometry_path", _canonical_partner(rsm_r2_project))
        assert page._readiness[1], page._readiness
        form = page.build_form()
        assert form.corrections == RSMFrameCorrections(solid_angle=True, polarization_factor=0.99)
        headless = Path(rsm_r2_project) / "rsm-page-headless.nexus"
        off = run_rsm_operation_v2(prepare_rsm_tool_v2(replace(
            form, output_path=headless, output_family=None, corrections=RSMFrameCorrections())).request)
        assert off.payload.result_fingerprint == _SCIENCE[((43,), 40)]["result"]
        on = run_rsm_operation_v2(prepare_rsm_tool_v2(replace(
            form, output_path=headless.with_name("rsm-page-headless-on.nexus"), output_family=None)).request)
        expected = on.payload.result_fingerprint
        assert expected != off.payload.result_fingerprint
        page.begin_run()
        _wait(page, qapp, lambda: not page.active(), seconds=600)
        output = page._value("output_path")
        assert page.run_bar.readinessLabel.full_text() == f"Saved {Path(output).name}"
        committed = page._results[output]
        assert committed.result_fingerprint == expected
        assert page.shown[0].result_fingerprint == expected
        assert page.reopen(output)
        _wait(page, qapp, lambda: not page.active())
        reopened = page._results[output]
        assert reopened is not committed and reopened.result_fingerprint == expected
        assert page.shown[0].result_fingerprint == expected
        np.testing.assert_array_equal(page.shown[0].intensity, committed.intensity)
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


def test_the_advanced_settings_are_ints_advanced_dialog_of_shared_rows(qapp, tmp_path):
    """Codex's reuse audits (2026-10-01): RSM's Advanced was its own dialog of raw spin boxes, a combo and line
    edits. It is Int's AdvancedDialog of shared rows: the selected member's raw decoder and psic motor columns,
    the column occurrences, and the chunk and memory. A refused whole number is named by readiness and refused
    by the form, so the member's earlier number never runs unseen (Codex F1); a valid one reaches the member and
    the form; each member shows its own."""
    from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
    from xdart.gui.widgets.controls_panel import FormRow, SubsectionCard
    page = _page(qapp)
    try:
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        assert isinstance(page.advanced, AdvancedDialog) and not page.advanced.isModal()
        cards = {card.title.text(): card for card in page.advanced.findChildren(SubsectionCard)}
        assert set(cards) == {"Raw decoder (selected member)", "psic motor columns (selected member)",
                              "Motor column occurrences (selected member)", "Normalization column occurrences",
                              "Chunk and memory"}
        page.set_field("adv_foil_occurrence", "-1")  # a normalization occurrence is Processing's
        assert "Advanced Foil column occurrence" in page._headers["processing"].detail
        page.set_field("adv_foil_occurrence", "0")
        assert all(type(page._rows[key]) is FormRow and cards["Raw decoder (selected member)"].isAncestorOf(
            page._rows[key]) for key in ("adv_rows", "adv_columns", "adv_dtype", "adv_header"))
        assert page._readiness[1], page._readiness
        page.set_field("adv_rows", "abc")
        assert page._value("adv_rows") == "abc" and not page._readiness[1]
        assert "Advanced Rows ('abc' is not a whole number)" in page._readiness[0]
        assert not page.run_bar.startButton.isEnabled()
        with pytest.raises(ValueError, match=r"Advanced: Rows \('abc' is not a whole number\)"):
            page.build_form()
        page.set_field("adv_rows", str(page._members[0].detector_rows))
        page.set_field("adv_chunk", "0")  # below the row's minimum, in Processing
        assert "Advanced Frames per chunk" in page._readiness[0]
        page.set_field("adv_chunk", "4")
        page.set_field("adv_header", "16")
        assert page._readiness[1], page._readiness
        form = page.build_form()
        assert form.chunk_size == 4 and form.members[0].raw_header_skip == 16
        second = page._members[1].raw_header_skip
        page._load_member(1)  # each member shows its own decoder
        assert page._value("adv_header") == str(second)
        page._load_member(0)
        assert page._value("adv_header") == "16"
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
        assert page.shown is not None and len(reads) == 2
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
    view = ScientificView()
    try:
        snapshot = RSMViewerModel().snapshot(values, h_index=1, k_index=2, l_index=3)
        view.reconcile_keyed(rsm_keyed_projection(values, snapshot))
        for pane, product in zip(_slices(view), snapshot.products[:3], strict=True):
            x, y = np.asarray(product.x_axis), np.asarray(product.y_axis_or_none)
            first = pane.image.mapToParent(QtCore.QPointF(.5, .5))
            last = pane.image.mapToParent(QtCore.QPointF(x.size - .5, y.size - .5))
            assert (first.x(), first.y(), last.x(), last.y()) == pytest.approx(
                (x[0], y[0], x[-1], y[-1]), abs=1e-9)
        first = _slices(view)[0].image.mapToParent(QtCore.QPointF(.5, .5))
        last = _slices(view)[0].image.mapToParent(QtCore.QPointF(2.5, 3.5))
        assert (first.x(), first.y(), last.x(), last.y()) == pytest.approx((10, 100, 30, 106), abs=1e-9)
    finally:
        view.deleteLater()
        qapp.processEvents()


def test_a_failed_older_read_still_reads_the_newer_selection(qapp, tmp_path):
    """Codex's stage-2 review S2-F1, its reproduction with real strict records and the real reader: seven
    records evict the first two from the five kept; an evicted A is selected (its read held back), then an
    evicted B; A's record is removed, so its read fails. The newer B must be read after the failure, as it is
    after a success, and A is not read again."""
    from threading import Event
    from tests.core.test_analysis_artifact import (
        test_analysis_artifact_v2_rsm_round_trip_is_a_closed_exact_branch as write_record,
    )
    paths = []
    for index in range(7):
        root = tmp_path / str(index)
        root.mkdir()
        write_record(root, False)
        paths.append(str(root / "rsm-artifact-v2.nexus"))
    started, release, hold, reads = Event(), Event(), [], []

    def read(path, cancel, progress):
        reads.append(path)
        if hold and path == paths[0]:
            started.set()
            assert release.wait(10)
        return page_module._read_rsm_record(path, cancel, progress)

    page = _page(qapp, owner=_Owner(), reader=SingleWorkerOwner(read))
    try:
        for path in paths:
            assert page.reopen(path)
            _wait(page, qapp, lambda: not page.active())
        assert page._results[paths[0]] is page._results[paths[1]] is None
        hold.append(True)
        page.results.setCurrentRow(0)
        assert started.wait(3)
        page.results.setCurrentRow(1)
        Path(paths[0]).unlink()  # a real missing record: the held read fails
        release.set()
        _wait(page, qapp, lambda: not page.active())
        assert reads.count(paths[1]) == 2, "the latest selection was never read"
        assert page._results[paths[1]] is not None
        assert reads.count(paths[0]) == 2  # its first open and the failed read: not retried
    finally:
        release.set()
        _wait(page, qapp, lambda: page.close_page().status is PageCleanup.CLEAN)
        page.deleteLater()
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
        shown = page.shown
        assert page.results.currentItem().data(role) == paths[1]
        released.set()
        _wait(page, qapp, lambda: not page.active())
        assert page.results.currentItem().data(role) == paths[1] and page.shown is shown
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
        _wait(page, qapp, lambda: page.shown is not None)
        assert page.shown[0] is first
        assert Path(page._value("output_path")).is_file()
        monkeypatch.setattr(page_module, "make_rsm_viewer_values", {"payload": second}.__getitem__)
        page.begin_run()
        _wait(page, qapp, lambda: page.shown[0] is second and not page.active())
        assert page.results.count() == 1
        np.testing.assert_array_equal(_slices(page.view)[0].canvas.raw_image,
                                      np.asarray(page.shown[1].products[0].values).T)
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
        _wait(page, qapp, lambda: page.shown is not None and not page.active())
        prior = page.shown
        assert Path(page._value("output_path")).is_file()
        _terminal_instead_of_gridding(monkeypatch, ModuleDisposition.REFUSED, "OUTPUT_NOT_PREVIOUS_ANALYSIS")
        page.begin_run()
        _wait(page, qapp, lambda: "refused" in label.full_text() and not page.active())
        assert label.full_text() == "RSM refused: OUTPUT_NOT_PREVIOUS_ANALYSIS"
        assert page.shown is prior and page.results.count() == 1
        _terminal_instead_of_gridding(monkeypatch, ModuleDisposition.CANCELLED, "CANCELLED", until_stopped=True)
        page.begin_run()
        _wait(page, qapp, lambda: page._owner.busy and page.run_bar.stopButton.isEnabled())
        page.stop()
        _wait(page, qapp, lambda: "stopped" in label.full_text() and not page.active())
        assert page.shown is prior and page.results.count() == 1
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
        assert cards["source"].status.text() == "2 members"
        assert cards["processing"].status.text().endswith(" · grid 40×40×40")  # Coordinates is Processing's (G8a)
        assert cards["experiment"].status.text() == "rsm_canonical.json"
        page._members[1].spec_path = "missing.spec"
        page._edited()
        assert not cards["source"].valid_marker.isVisibleTo(page)
        assert cards["source"].valid_marker.toolTip() == "Needs member 2's SPEC file"
        assert cards["project"].valid_marker.isVisibleTo(page) and cards["experiment"].valid_marker.isVisibleTo(page)
    finally:
        _close(page, qapp)


def test_a_resident_result_shown_again_recomputes_nothing_and_an_evicted_ones_viewer_goes(
        qapp, tmp_path, monkeypatch):
    """Codex's reuse audit rec. 3: each resident result keeps its viewer values and model, built once, so
    showing it again builds and slices nothing; its viewer leaves with its payload when it drops out of the
    five resident (no volume kept per evicted result)."""
    from xrd_tools.session import rsm_viewer_model
    names = ("one", "two", "three", "four", "five", "six")
    values = {name: _values(offset=float(index)) for index, name in enumerate(names)}
    built, sliced = [], []
    monkeypatch.setattr(page_module, "make_rsm_viewer_values",
                        lambda payload: built.append(payload) or values[payload])
    original = rsm_viewer_model.RSMViewerModel.snapshot

    def counting(self, *args, **kwargs):
        sliced.append(kwargs)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(rsm_viewer_model.RSMViewerModel, "snapshot", counting)

    def read(path, _cancel, _progress):
        return Path(path).name.split("_")[0]
    page = _page(qapp, owner=_Owner(), reader=SingleWorkerOwner(read))
    try:
        for name in names:
            assert page.reopen(str(tmp_path / f"{name}_rsm.nexus"))
            _wait(page, qapp, lambda: not page.active())
        assert built == list(names) and len(sliced) == 6
        built.clear(), sliced.clear()
        for row in (1, 2, 1):  # "two", "three", "two": resident
            page.results.setCurrentRow(row)
        assert built == [] and sliced == [] and page.shown[0] is values["two"]
        assert set(page._viewers) == {str(tmp_path / f"{name}_rsm.nexus") for name in names[1:]}
    finally:
        _close(page, qapp)


def test_ints_colour_map_and_log_redraw_the_six(qapp, monkeypatch):
    """Int's top bar drives the RSM's six (the maintainer: "Color map should be viridis by default. The other
    color maps should also be available like in Int"): "Default" is viridis, the others Int's list."""
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", {"payload": _values()}.__getitem__)
    page = _page(qapp, owner=_Owner())
    try:
        page._add_result("/tmp/x_rsm.nexus", "payload")
        assert page.view.keyed_projection.color_map == "Default" and not page.view.keyed_projection.log_scale
        page.view.color_map.setCurrentText("magma")
        page.view.log_scale.click()
        drawn = page.view.keyed_projection
        assert (drawn.color_map, drawn.log_scale) == ("magma", True)
        assert _slices(page.view)[0]._render_contract[6:8] == ("magma", True)
    finally:
        _close(page, qapp)


def test_the_3d_viewer_and_custom_slice_buttons_reach_the_page(qapp, monkeypatch):
    """The maintainer's two extra tools, "as extra tools that pop up separate windows to keep the scientific
    view clean for now": two buttons in Int's top bar whose presses reach the page (their windows come with
    G9). RSM has no χ (c/w) cut, so that tool is left out."""
    page = _page(qapp, owner=_Owner())
    pressed = []
    monkeypatch.setattr(page, "_page_tool", pressed.append)
    try:
        assert [page.view.page_tool(name).text() for name in ("3d", "custom")] == ["3D Viewer", "Custom Slice"]
        page.view.page_tool("3d").click()
        page.view.page_tool("custom").click()
        assert pressed == ["3d", "custom"]
        assert page.view.slice.isHidden() and page.view.pin.isHidden()
    finally:
        _close(page, qapp)


# AUD-9 stage 2b G8a: the RSM controls become Stitch's sections (the maintainer, 2026-10-01: "the control panel can
# be almost identical to the stitch control panel for the project, experiment (except the standard/grazing which
# are not required), and source. Only the processing panel would be different")


def test_the_experiment_section_is_stitchs_calibration_and_mask_rows(qapp, tmp_path):
    """Experiment is Stitch's, from Int's composer: Calibrate · Make Mask, the Detector subsection's Calibration and
    Mask File ("Mask File and Make Mask ... they should. And masks should be applied when specified"), and the
    calibration's refinement, greyed until a Calibration is named. The canonical asset's Install button and
    identity leave the page; the form's geometry is the Calibration row's (its refinement the toggle's), and the
    Mask File reaches the run's corrections."""
    from xdart.gui.widgets.controls_panel import SubsectionCard
    from xrd_tools.analysis.rsm_geometry_asset import RSMCalibrationGeometryInput
    page = _page(qapp)
    try:
        assert not hasattr(page, "install_button") and not hasattr(page, "identity")
        card = page.controls.experiment_card
        assert card.isAncestorOf(page.calibrate_button) and card.isAncestorOf(page.mask_button)
        detector = next(sub for sub in card.findChildren(SubsectionCard)
                        if sub.isAncestorOf(page._rows["geometry_path"]))
        assert detector.isAncestorOf(page._rows["mask_path"])
        assert not page._fields["refinement"].enabled  # greyed until a Calibration is named
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        assert page._fields["refinement"].enabled and page._readiness[1], page._readiness
        form = page.build_form()
        assert type(form.geometry_asset) is RSMCalibrationGeometryInput
        assert form.geometry_asset.locator == "calibration/rsm_canonical.json" and form.geometry_asset.refinement
        assert form.corrections.mask_path is None
        page.set_field("refinement", False)
        assert not page.build_form().geometry_asset.refinement
        assert page.build_form().fingerprint != form.fingerprint
        mask = tmp_path / "mask.npy"
        np.save(mask, np.zeros((195, 487), dtype=bool))
        page.set_field("mask_path", str(mask))
        assert page.build_form().corrections.mask_path == str(mask)
        page.set_field("geometry_path", "")
        assert not page._readiness[1] and "a Calibration (Calibrate, or choose its file)" in page._readiness[0]
        assert "Calibration" in page._headers["experiment"].detail
        page.set_field("geometry_path", str(tmp_path.parent / "elsewhere.json"))
        assert "a Calibration inside the Project" in page._readiness[0]
    finally:
        _close(page, qapp)


def test_calibrate_and_make_mask_open_the_shared_windows_with_rsms_context(qapp, tmp_path):
    """Calibrate opens the shared calibration window (tabs/calibration) with RSM's context, so RSM calibrates its
    own data ("reuse what was developed for stitch, not use the calibration that was done in stitch"): its title
    and save button name RSM, and its saved calibration (and the detector mask saved with it) fill the
    Calibration and the Mask File. Make Mask opens the shared mask window, whose saved mask fills the Mask File."""
    from xdart.gui.tabs.calibration.mask import MaskDialog
    from xrd_tools.session.readiness import ControlAction
    page = _page(qapp)
    try:
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        page._control_action(ControlAction.CALIBRATE)
        dialog = page._calibration_dialog
        assert dialog is not None and dialog.windowTitle() == "RSM calibration"
        assert dialog.save_button.text() == "Save geometry and use in RSM"
        dialog.saved_mask = str(tmp_path / "cal_mask.npy")
        dialog.calibrationSaved.emit(str(tmp_path / "calibration" / "new.json"), "multigeometry")
        assert page._value("geometry_path") == str(tmp_path / "calibration" / "new.json")
        assert page._value("mask_path") == str(tmp_path / "cal_mask.npy")
        dialog.hide()
        page._control_action(ControlAction.MAKE_MASK)
        mask_dialog = page._authoring_dialogs[-1]
        assert type(mask_dialog) is MaskDialog
        mask_dialog.maskSaved.emit(str(tmp_path / "drawn.npy"))
        assert page._value("mask_path") == str(tmp_path / "drawn.npy")
        mask_dialog.hide()
    finally:
        _close(page, qapp)


def test_a_calibration_outside_the_project_is_named_before_any_preview(qapp, tmp_path):
    """A relative Calibration that climbs out of the Project is refused by readiness, as an absolute one outside
    it is, rather than by the form at Preview."""
    page = _page(qapp)
    try:
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        page.set_field("geometry_path", "../calibration/rsm_canonical.json")
        assert not page._readiness[1] and "a Calibration inside the Project" in page._readiness[0]
        assert page._rows["refinement"].editor.isEnabled()  # a Calibration is named: the toggle stays usable
    finally:
        _close(page, qapp)


def test_coordinates_is_a_processing_choice_and_max_chunk_mib_defaults_to_auto(qapp, tmp_path):
    """Design §2: Coordinates moves from Experiment to Processing. Advanced Max chunk MiB is blank, Auto (stage 2b
    S3a: 256 MiB, or one frame's need when larger), and the form passes None; a cap below one frame's need is
    named by readiness with the MiB it needs, before any Preview; a sufficient one reaches the form."""
    page = _page(qapp)
    try:
        assert page.controls.processing_card.isAncestorOf(page._rows["coordinates"])
        assert not page.controls.experiment_card.isAncestorOf(page._rows["coordinates"])
        _with_members(page, tmp_path, _synthetic_members(tmp_path))
        assert page._value("adv_max_chunk_mib") == "" and page.build_form().max_chunk_bytes is None
        page.set_field("adv_max_chunk_mib", "1")
        assert not page._readiness[1] and "one frame needs" in page._readiness[0], page._readiness
        assert "Max chunk MiB" in page._headers["processing"].detail
        page.set_field("adv_max_chunk_mib", "256")
        assert page._readiness[1] and page.build_form().max_chunk_bytes == 256 * 1024 * 1024
    finally:
        _close(page, qapp)
