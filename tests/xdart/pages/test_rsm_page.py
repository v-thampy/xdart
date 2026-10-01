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
from xdart.gui.tabs.scattering.controls_inventory import THRESHOLD_ENABLED, THRESHOLD_MAX, THRESHOLD_MIN
from xdart.gui.tools import rsm_owner
from xdart.gui.tools.rsm_tool import RSMToolDialog
from xrd_tools.analysis.module_transaction import ModuleDisposition, ModuleTerminalResult
from xrd_tools.analysis.rsm_operation import RSMOperationExecutionV2, RSMOperationResultV2
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from tests.core.test_rsm_calibration_geometry import _canonical_partner
from tests.core.test_rsm_operation_r2_real_data import _SCIENCE, rsm_r2_project  # noqa: F401
from tests.xdart.tools.test_rsm_tool_dialog import (  # noqa: F401
    _Owner, _Status, _q_values, _values, qapp,
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
    MiB is Auto where the dialog's is the preset's 256 (stage 2b S3a). The page's threshold is Int's (Auto by
    default, as Stitch's), its "High → NaN" retired into Int's Manual High (G8b): the dialog's form takes the
    same in ``_as_page``."""
    from dataclasses import replace as replaced
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning

    def _as_page(form, **fields):
        return replaced(form, conditioning=replaced(form.conditioning, high_threshold=None),
                        corrections=replaced(form.corrections, threshold=StitchConditioning("auto")), **fields)
    page = _page(qapp, owner=_Owner())
    dialog = RSMToolDialog(empty_host_services(_Status()), owner=_Owner())
    try:
        output = tmp_path / "rsm.nexus"
        _sto_align(tmp_path)  # the preset's scan on disk: the page finds its member through Stitch's Source
        page.set_field("project_root", str(tmp_path))
        page.set_field("geometry_path", _canonical_partner(tmp_path))
        page.set_field("output_path", str(output))
        page.set_sources(["STO_align"])  # Project relative, as the preset names it
        for key, value in (("scan", "43"), ("image_dir", "images"), ("image_stem", "b_thampy_STO_align_scan43_"),
                           ("frame_last", "60")):
            page.set_field(key, value)
        page.resolve_source()
        dialog.project_edit.setText(str(tmp_path))
        dialog._apply_scan43_preset()
        dialog.output_edit.setText(str(output))
        geometry = page.build_form().geometry_asset
        assert page.build_form().fingerprint == _as_page(
            dialog._build_form(), overwrite=AnalysisArtifactOverwrite.REPLACE, geometry_asset=geometry,
            max_chunk_bytes=None).fingerprint
        page.set_field("coordinates", RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.display_name)
        page.set_field("grid", "Full 200×200×200")
        page.set_field("normalization", "None")
        dialog.coordinate_frame_combo.setCurrentIndex(
            dialog.coordinate_frame_combo.findData(RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU))
        dialog.grid_combo.setCurrentIndex(1)
        dialog.normalization_combo.setCurrentIndex(1)
        assert page.build_form().fingerprint == _as_page(
            dialog._build_form(), overwrite=AnalysisArtifactOverwrite.REPLACE, geometry_asset=geometry,
            max_chunk_bytes=None).fingerprint
    finally:
        _close(page, qapp)
        dialog.shutdown()


def _spec_scan(number, mu):
    """One scan of the RSM dialog tests' synthetic member (two points; eta and the normalization columns, a
    repeated Seconds among them)."""
    return (f"#S {number} ascan eta {number - 1} {number} 1 1\n#D Mon Jan 15 10:30:00 2024\n"
            f"#P0 13000.007 {mu} 20 30 40 50\n#G3 1 0 0 0 1 0 0 0 1\n#N 4\n#L eta  Seconds  Seconds  foil status\n"
            f"{number - 1}.0 1 10 101\n{number}.0 2 20 110\n")


def _spec_file(path, scans):
    path.write_text(f"#F {path.name}\n#E 1\n#D Mon Jan 15 10:30:00 2024\n#O0 energy  mu  chi  phi  nu  del\n\n"
                    + "\n".join(scans), encoding="utf-8")


def _synthetic_source(root, *, full=False):
    """One synthetic SPEC file with two scans and their images (the RSM dialog tests' member, read through Stitch's
    Source: groups never span files), named for the blank Image stem (<SPEC name>_scan<N>_), and a calibration
    encoding the canonical geometry (row B's, AUD-9 stage 2b C7b: the page's geometry is its Calibration row's).
    ``full``: frames of the calibrated detector's shape, for a real gridding."""
    import tifffile
    _canonical_partner(root)
    images = root / "source" / "images"
    images.mkdir(parents=True)
    _spec_file(root / "source" / "RSM_synth", [_spec_scan(scan, 9 + scan) for scan in (1, 2)])
    for scan in (1, 2):
        for label in range(2):
            frame = (np.full((195, 487), 9 + scan + label, dtype=np.uint16) if full
                     else np.arange(20, dtype=np.uint16).reshape(4, 5) + scan - 1 + label)
            tifffile.imwrite(images / f"RSM_synth_scan{scan}_{label:04d}.tif", frame)
    return root / "source" / "RSM_synth"


def _sto_align(root):
    """The scan-43 preset's SPEC file, scan and image names, two frames: the dialog's preset member on disk."""
    import tifffile
    (root / "images").mkdir()
    _spec_file(root / "STO_align", [_spec_scan(43, 10)])
    for label in range(2):
        tifffile.imwrite(root / "images" / f"b_thampy_STO_align_scan43_{label:04d}.tif",
                         np.arange(20, dtype=np.uint16).reshape(4, 5) + label)


def _with_source(page, root, scans="[1-2]"):
    """The page on the synthetic source: its two scans one RSM (Stitch's Selection), the Images folder Project
    relative, and the scans found as Preview and Run find them (``resolve_source``)."""
    page.set_field("project_root", str(root))
    page.set_field("geometry_path", "calibration/rsm_canonical.json")
    page.set_sources([str(root / "source" / "RSM_synth")])
    page.set_field("scan", scans)
    page.set_field("image_dir", "source/images")
    page.set_field("adv_dtype", "uint16")
    page.set_field("normalization", "None")  # the Normalization group's None (decision 4)
    page.set_field("offset", "0")
    page._threshold_edited(THRESHOLD_ENABLED, False)  # Int's threshold off ("High → NaN" blank before)
    page.set_field("static_hot", "")
    page.resolve_source()


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
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        assert page._readiness[1], page._readiness
        output = Path(page._value("output_path"))
        assert str(output) == suggested_output(tmp_path, page.build_form().members)
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
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
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
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
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
        page.set_sources(["STO_align"])  # the preset's scan as Stitch's Source rows (design §2)
        page.set_field("scan", "43")
        page.set_field("image_dir", "images")  # the stem blank (STO_align_scan43_) and Last blank (to 60)
        assert page._readiness[1], page._readiness
        page.resolve_source()
        form = page.build_form()
        from xrd_tools.integrate.stitch_conditioning import StitchConditioning
        assert form.corrections == RSMFrameCorrections(threshold=StitchConditioning("auto"), solid_angle=True,
                                                       polarization_factor=0.99)
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


@pytest.mark.parametrize(("key", "label"), (("frame_first", "First frame"), ("frame_step", "Frame step")))
def test_invalid_frame_text_stays_as_typed_and_is_refused(qapp, tmp_path, key, label):
    """F1: a First or Step that is not a whole number stays as typed, and readiness and the form refuse it;
    the earlier numbers never run unseen. Stitch's one frame range, in Advanced (design §2)."""
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        assert page._readiness[1], page._readiness
        page.set_field(key, "invalid")
        assert page._value(key) == "invalid"
        assert not page._readiness[1] and f"Advanced {label} ('invalid' is not a whole number)" in page._readiness[0]
        assert not page.run_bar.startButton.isEnabled()
        with pytest.raises(ValueError, match="is not a whole number"):
            page.build_form()
    finally:
        _close(page, qapp)


def test_the_advanced_settings_are_ints_advanced_dialog_of_shared_rows(qapp, tmp_path):
    """Codex's reuse audits (2026-10-01): RSM's Advanced was its own dialog of raw spin boxes, a combo and line
    edits. It is Int's AdvancedDialog of shared rows: Stitch's Source and geometry (one raw decoder, Angle
    mapping and frame range for every scan, design §2), the column occurrences, and the chunk and memory. A
    refused whole number is named by readiness and refused by the form, so an earlier number never runs unseen
    (Codex F1); a valid one reaches every member once the scans are found again."""
    from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
    from xdart.gui.widgets.controls_panel import FormRow, SubsectionCard
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        assert isinstance(page.advanced, AdvancedDialog) and not page.advanced.isModal()
        cards = {card.title.text(): card for card in page.advanced.findChildren(SubsectionCard)}
        assert set(cards) == {"Source and geometry", "Normalization column occurrences", "Corrections",
                              "Chunk and memory"}
        page.set_field("adv_foil_occurrence", "-1")  # a normalization occurrence is Processing's
        assert "Advanced Foil column occurrence" in page._headers["processing"].detail
        page.set_field("adv_foil_occurrence", "0")
        assert all(type(page._rows[key]) is FormRow and cards["Source and geometry"].isAncestorOf(page._rows[key])
                   for key in ("adv_rows", "adv_columns", "adv_dtype", "adv_header", "adv_angle_mapping",
                               "frame_first", "frame_last", "frame_step"))
        assert page._readiness[1], page._readiness
        page.set_field("adv_rows", "abc")
        assert page._value("adv_rows") == "abc" and not page._readiness[1]
        assert "Advanced Rows ('abc' is not a whole number)" in page._readiness[0]
        assert not page.run_bar.startButton.isEnabled()
        with pytest.raises(ValueError, match=r"Advanced: Rows \('abc' is not a whole number\)"):
            page.build_form()
        page.set_field("adv_rows", "195")
        page.set_field("adv_chunk", "0")  # below the row's minimum, in Processing
        assert "Advanced Frames per chunk" in page._readiness[0]
        page.set_field("adv_chunk", "4")
        page.set_field("adv_header", "16")
        assert page._readiness[1], page._readiness
        with pytest.raises(ValueError, match="not found yet"):  # a decoder edit: the scans are found again
            page.build_form()
        page.resolve_source()
        form = page.build_form()
        assert form.chunk_size == 4 and [member.raw_header_skip for member in form.members] == [16, 16]
    finally:
        _close(page, qapp)


def test_a_zero_frame_step_or_a_last_before_the_first_is_refused(qapp, tmp_path):
    """F1: Step 0 divided in the member table's frame count before anything refused it. The one range refuses
    it as a row, and a Last before the First by ``rsm_frame_selector``, in readiness and in the form."""
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        page.set_field("frame_step", "0")
        assert not page._readiness[1] and "Advanced Frame step (must be at least 1)" in page._readiness[0]
        with pytest.raises(ValueError, match=r"Frame step \(must be at least 1\)"):
            page.build_form()
        page.set_field("frame_step", "1")
        page.set_field("frame_first", "1")
        page.set_field("frame_last", "0")
        assert not page._readiness[1] and "Last frame must not come before First frame" in page._readiness[0]
        with pytest.raises(ValueError, match="Last frame must not come before First frame"):
            page.resolve_source()
    finally:
        _close(page, qapp)


def test_a_pending_verification_keeps_a_clickable_retry(qapp, tmp_path, monkeypatch):
    """F2: the real owner, preflight, gridding and publisher; one transient failure of the strict reload
    leaves the saved result VERIFICATION_PENDING. The Controls editors lock, its action row does not: Retry
    is clickable and verifies the saved result."""
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
        _synthetic_source(tmp_path, full=True)
        _with_source(page, tmp_path)
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


def _rsm_result_at(target):
    """An RSM result's record at ``target`` (its envelope what Replace's header check admits, G8c)."""
    from tests.core.test_analysis_artifact import _request, _writer
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, admit_analysis_artifact
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    request = _request(target, AnalysisArtifactKind.RSM)
    admit_analysis_artifact(request).publish(_writer(request))


def _leaving_the_file(monkeypatch):
    """The execution stand-in leaves its output's record, as a real commit does."""
    stand_in = RSMOperationExecutionV2.run

    def run(self, **kwargs):
        _rsm_result_at(Path(self.request.module.output.target))
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
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        output = Path(page._value("output_path"))
        _rsm_result_at(output)  # a result already there
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
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
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
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
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
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        assert page.run_bar.readinessLabel.full_text().startswith("Ready · RSM · 1 SPEC file · 1 RSM")
        assert all(card.valid_marker.isVisibleTo(page) for card in cards.values())
        assert cards["source"].status.text() == "1 SPEC file · 1 RSM"  # Stitch's Source status
        assert cards["processing"].status.text().endswith(" · grid 40×40×40")  # Coordinates is Processing's (G8a)
        assert cards["experiment"].status.text() == "rsm_canonical.json"
        page.set_sources([str(tmp_path / "missing.spec")])
        assert not cards["source"].valid_marker.isVisibleTo(page)
        assert cards["source"].valid_marker.toolTip() == "Needs the source (missing.spec not found)"
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
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
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
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
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
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
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
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        assert page._value("adv_max_chunk_mib") == "" and page.build_form().max_chunk_bytes is None
        page.set_field("adv_max_chunk_mib", "1")
        assert not page._readiness[1] and "one frame needs" in page._readiness[0], page._readiness
        assert "Max chunk MiB" in page._headers["processing"].detail
        page.set_field("adv_max_chunk_mib", "256")
        assert page._readiness[1] and page.build_form().max_chunk_bytes == 256 * 1024 * 1024
    finally:
        _close(page, qapp)


# AUD-9 stage 2b G8a, commit 3: Source is Stitch's rows over its owner (design §2)


def test_the_source_is_stitchs_rows_and_the_member_table_is_gone(qapp, tmp_path):
    """Design §2 (the maintainer: "nexus files as source as well. And directory mode just like stitch"): the
    Source card is Stitch's rows from Int's composer (the kind with Directory's Subdirs on its row, the files or
    folder, Filter, Images, Image stem, Scans and Selection), each kind showing its own and keeping its own files.
    The member table, its Add/Remove/Up/Down and the Scan 43 preset are gone; one raw decoder, Angle mapping and
    frame range are Advanced's, the range from the first frame to each scan's last."""
    from xdart.gui.tabs.scattering.controls_inventory import SOURCE_RECURSIVE, SOURCE_TYPE
    page = _page(qapp)
    try:
        assert not hasattr(page, "member_table") and not any(
            hasattr(page, f"member_{name}_button") for name in ("add", "remove", "up", "down", "preset"))
        keys = ("source_kind", "spec_path", "subdirs", "name_filter", "image_dir", "image_stem", "scan_mode", "scan")
        assert all(page.controls.source_card.isAncestorOf(page._rows[key]) for key in keys)
        assert (page._fields["source_kind"].path, page._fields["subdirs"].path) == (SOURCE_TYPE, SOURCE_RECURSIVE)
        assert page._rows["source_kind"].parentWidget() is page._rows["subdirs"].parentWidget()  # one row
        assert page._fields["source_kind"].choices == ("Spec", "NeXus", "Directory")
        assert page._source_paths == () and page._value("scan") == "1"  # no preset: Stitch's empty Source

        def shown():
            return {key for key in keys if not page._rows[key].isHidden()}
        assert shown() == {"source_kind", "spec_path", "image_dir", "image_stem", "scan_mode", "scan"}
        page.set_field("scan_mode", "All Scans")
        assert "scan" not in shown()
        page.set_field("source_kind", "Directory")
        assert shown() == {"source_kind", "spec_path", "subdirs", "name_filter", "image_dir", "image_stem"}
        assert page._rows["spec_path"].label.text() == "Folder"
        page.set_field("source_kind", "NeXus")
        assert shown() == {"source_kind", "spec_path"} and page._rows["spec_path"].label.text() == "NeXus files"
        page.set_sources([str(tmp_path / "a.nxs"), str(tmp_path / "b.nxs")])
        assert page._value("spec_path") == "2 files selected"
        page.set_field("source_kind", "Spec")  # each kind keeps its own files
        assert page._source_paths == () and page._value("spec_path") == ""
        page.set_field("source_kind", "NeXus")
        assert page._source_paths == (str(tmp_path / "a.nxs"), str(tmp_path / "b.nxs"))
        assert page._value("adv_angle_mapping") == "mu=mu, eta=eta, chi=chi, phi=phi, nu=nu, del=del"
        assert [str(page._value(key)) for key in ("frame_first", "frame_last", "frame_step")] == ["0", "", "1"]
    finally:
        _close(page, qapp)


def test_a_bracket_selection_is_one_rsm_and_a_comma_selection_is_queued(qapp, tmp_path):
    """Stitch's Selection: brackets combine scans into one RSM ([1-2]); 1,2 makes two RSMs, which Preview and Run
    queue one at a time (G8c; G8a refused them), each in its own family's stable slot in the Save Path's folder, so
    each records its family. All Scans makes each scan its own RSM, known once the scans are found."""
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path, scans="1,2")
        assert page._readiness[1], page._readiness
        assert page.controls.source_card.status.text() == "1 SPEC file · 2 RSMs"
        with pytest.raises(ValueError, match="2 RSMs in the Source: Preview and Run queue them"):
            page.build_form()
        forms = page.build_forms()
        assert [[member.scan for member in form.members] for form in forms] == [["1.1"], ["2.1"]]
        assert all(form.output_family and Path(form.output_path).name == f"{form.output_family}_rsm.nexus"
                   for form in forms)
        assert len({form.output_path for form in forms}) == 2
        assert {Path(form.output_path).parent for form in forms} == {Path(page._value("output_path")).parent}
        page.set_field("scan", "[1-2]")
        assert page._readiness[1], page._readiness
        page.resolve_source()
        assert [member.scan for member in page.build_form().members] == ["1.1", "2.1"]
        assert [form.fingerprint for form in page.build_forms()] == [page.build_form().fingerprint]
        page.set_field("scan", "2")
        page.resolve_source()
        assert [member.scan for member in page.build_form().members] == ["2.1"]
        page.set_field("scan_mode", "All Scans")
        assert page._readiness[1] and page.controls.source_card.status.text() == "1 SPEC file · all scans"
        page.resolve_source()
        with pytest.raises(ValueError, match="2 RSMs in the Source"):
            page.build_form()
        assert len(page.build_forms()) == 2
    finally:
        _close(page, qapp)


def _counting_grids(monkeypatch):
    """The committing stand-in, recording each gridded output's name in order."""
    _commit_instead_of_gridding(monkeypatch, _values())
    committed, grids = RSMOperationExecutionV2.run, []

    def counted(self, **kwargs):
        grids.append(Path(self.request.module.output.target).name)
        return committed(self, **kwargs)
    monkeypatch.setattr(RSMOperationExecutionV2, "run", counted)
    return grids


def test_a_comma_selection_previews_every_rsm_and_grids_each_in_turn(qapp, tmp_path, monkeypatch):
    """G8c: Preview prepares each RSM through the one owner (its real preflight) and lists them all under their
    own headings; Run prepares and grids each in turn into its own output, and lists both results."""
    grids = _counting_grids(monkeypatch)
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path, scans="1,2")
        names = [Path(form.output_path).name for form in page.build_forms()]
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready · 2 RSMs" in label.full_text())
        shown = page.preview_text.toPlainText()
        assert shown.startswith("2 RSMs") and f"RSM 1 of 2 · {names[0]}" in shown
        assert f"RSM 2 of 2 · {names[1]}" in shown and grids == []
        page.begin_run()
        _wait(page, qapp, lambda: "Saved 2 of 2 RSMs" in label.full_text())
        assert grids == names and page.results.count() == 2 and not page.active()
    finally:
        _close(page, qapp)


def test_preview_refuses_a_foreign_file_at_an_output_before_any_gridding(qapp, tmp_path, monkeypatch):
    """G8c (Codex's RSM stage-2 review, question 1): the file at an RSM's output is checked at Preview as its
    run's Replace checks it, from the header: a foreign file is refused there, left untouched, and nothing is
    gridded, by Preview or by Run; in a queue the refusal names the RSM whose output it is."""
    grids = _counting_grids(monkeypatch)
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path, scans="1")
        target = Path(page.build_form().output_path)
        target.parent.mkdir()
        target.write_bytes(b"an operator's own file")
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview refused" in label.full_text())
        assert label.full_text().startswith("Preview refused: OUTPUT_NOT_PREVIOUS_ANALYSIS: ")
        assert f"{target.name} is not an RSM result that Replace can overwrite (" in label.full_text()
        page.begin_run()
        _wait(page, qapp, lambda: not page.active())
        assert "Preview refused: " in label.full_text() and grids == []
        _with_source(page, tmp_path, scans="1,2")
        second = Path(page.build_forms()[1].output_path)
        target.unlink()
        second.write_bytes(b"an operator's own file")
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview refused" in label.full_text())
        assert label.full_text().startswith("Preview refused (RSM 2 of 2): ")
        assert f"{second.name} is not an RSM result" in label.full_text()
        assert grids == [] and second.read_bytes() == b"an operator's own file"
    finally:
        _close(page, qapp)


def test_stop_mid_queue_cancels_the_rsm_gridding_and_starts_no_other(qapp, tmp_path, monkeypatch):
    """Stop stops the queue: the RSM being gridded is cancelled and the next is never prepared."""
    started = []

    def run(self, *, cancel_token=None, progress_callback=None):
        started.append(Path(self.request.module.output.target).name)
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
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path, scans="1,2")
        page.begin_run()
        _wait(page, qapp, lambda: bool(started))
        page.stop()
        _wait(page, qapp, lambda: not page.active() and page.run_bar.readinessLabel.full_text() == "RSM stopped")
        for _ in range(20):  # nothing further starts
            qapp.processEvents()
            page._poll()
        assert len(started) == 1 and page._jobs == [] and page.results.count() == 0
    finally:
        _close(page, qapp)


def test_append_skips_the_up_to_date_rsm_of_a_queue_and_grids_the_other(qapp, tmp_path, monkeypatch):
    """Each queued RSM has its own Append check: in Append the one whose output already holds it is skipped and
    named, the other is gridded."""
    from xdart.gui.tools import rsm_values
    grids = _counting_grids(monkeypatch)
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path, scans="1,2")
        first, second = (Path(form.output_path) for form in page.build_forms())
        _rsm_result_at(first)  # RSM 1's output already there, holding this request
        monkeypatch.setattr(rsm_values, "rsm_output_current",
                            lambda request: Path(request.module.output.target) == first)
        assert page.run_bar.write_mode() == "Append"
        page.begin_run()
        _wait(page, qapp, lambda: "Saved 1 of 2 RSMs" in label.full_text())
        assert label.full_text() == "Saved 1 of 2 RSMs · 1 already gridded with these settings, skipped"
        assert grids == [second.name]
    finally:
        _close(page, qapp)


def test_two_queued_rsms_naming_one_slot_are_refused(qapp, tmp_path, monkeypatch):
    """Two RSMs whose families name one stable slot would write one file: the queue refuses them with the
    remedy, before any Preview."""
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path, scans="1,2")
        monkeypatch.setattr(page_module, "suggested_family", lambda _project, _members: "same")
        with pytest.raises(ValueError, match="RSMs 1 and 2 would both write same_rsm.nexus"):
            page.build_forms()
    finally:
        _close(page, qapp)


def test_relative_source_files_and_images_are_the_projects(qapp, tmp_path):
    """A relative SPEC file and Images folder are the Project's (Stitch's selection would take them from the
    working directory), in readiness and in the members found."""
    page = _page(qapp)
    try:
        spec = _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        page.set_sources(["source/RSM_synth"])
        assert page._readiness[1], page._readiness
        page.resolve_source()
        assert {(member.spec_path, member.image_dir) for member in page.build_form().members} == {
            (str(spec), str(tmp_path / "source" / "images"))}
    finally:
        _close(page, qapp)


def test_a_directory_source_finds_its_spec_scans_as_stitch_does(qapp, tmp_path, monkeypatch):
    """Directory mode, as Stitch's: the folder's SPEC files (its subfolders' with Subdirs), each scan its own
    RSM; Filter matches file names; the Images folder blank is each SPEC file's own. A scan without images is
    skipped, and the Preview lists it; the one scan found is one RSM."""
    import tifffile
    _commit_instead_of_gridding(monkeypatch, _values())
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        page.set_field("source_kind", "Directory")
        page.set_sources([str(tmp_path)])
        assert page._readiness[1], page._readiness  # how many RSMs a folder makes is known once found
        assert page.resolve_source()[0] == ()  # the SPEC file is in a subfolder
        page.set_field("subdirs", True)
        groups, _skipped = page.resolve_source()
        assert [[member.scan for member in group] for group in groups] == [["1.1"], ["2.1"]]
        with pytest.raises(ValueError, match="2 RSMs in the Source"):
            page.build_form()
        page.set_field("name_filter", "nothing")
        assert page.resolve_source()[0] == ()
        with pytest.raises(ValueError, match="no scan to grid"):
            page.build_form()
        single = tmp_path / "single"
        single.mkdir()
        _spec_file(single / "half_imaged", [_spec_scan(1, 10), _spec_scan(2, 11)])  # scan 2 has no images
        for label in range(2):
            tifffile.imwrite(single / f"half_imaged_scan1_{label:04d}.tif",
                             np.arange(20, dtype=np.uint16).reshape(4, 5))
        page.set_field("name_filter", "")
        page.set_field("subdirs", False)
        page.set_field("image_dir", "")
        page.set_sources([str(single)])
        page.resolve_source()
        member, = page.build_form().members
        assert (member.spec_path, member.scan, member.image_dir) == (str(single / "half_imaged"), "1.1",
                                                                     str(single))
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready" in page.run_bar.readinessLabel.full_text())
        assert "Skipped half_imaged 2.1: No detector images named *half_imaged_scan2_*" in (
            page.preview_text.toPlainText())
    finally:
        _close(page, qapp)


def test_a_raw_nexus_source_is_one_rsm_named_by_its_entry(qapp, tmp_path):
    """NeXus files as the Source (the maintainer: "nexus files as source as well"): a raw NeXus stack is a member
    by its entry (C5c), the suggested output is named from the file, and the Preview, through the one owner,
    names the entry. Its run commits, and Append finds the output up to date."""
    from tests.core.test_rsm_nexus_members import _write_rsm_nexus
    from xdart.gui.tools.rsm_values import RSMNexusMemberForm
    page = _page(qapp)
    try:
        _canonical_partner(tmp_path)
        nexus = _write_rsm_nexus(tmp_path / "member_ub.nxs", ub=True)
        page.set_field("project_root", str(tmp_path))
        page.set_field("geometry_path", "calibration/rsm_canonical.json")
        page.set_field("source_kind", "NeXus")
        page.set_sources([str(nexus)])
        page.set_field("normalization", "None")
        assert page._readiness[1], page._readiness
        assert Path(page._value("output_path")).name == "member_ub_rsm.nexus"
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready" in page.run_bar.readinessLabel.full_text())
        member, = page.build_form().members
        assert type(member) is RSMNexusMemberForm and member.entry == "entry"
        assert "member_ub.nxs · NeXus entry entry · 5 frames" in page.preview_text.toPlainText()
        page.begin_run()  # gridded for real: five frames
        _wait(page, qapp, lambda: page.shown is not None and not page.active(), seconds=300)
        output = Path(page._value("output_path"))
        assert page.run_bar.readinessLabel.full_text() == f"Saved {output.name}" and output.is_file()
        page.begin_run()  # Append: the output already holds this RSM
        _wait(page, qapp, lambda: "Up to date" in page.run_bar.readinessLabel.full_text() and not page.active())
    finally:
        _close(page, qapp)


def test_preview_and_run_find_the_scans_off_the_gui_thread(qapp, tmp_path, monkeypatch):
    """S2: finding the scans reads the SPEC files and lists the images, so Preview and Run find them on the shared
    single worker, the controls held meanwhile; Stop drops what it finds."""
    import threading
    from xdart.gui.tools import rsm_values
    _commit_instead_of_gridding(monkeypatch, _values())
    real, threads, gate = rsm_values.rsm_member_groups, [], threading.Event()

    def recorded(*args, **kwargs):
        threads.append(threading.current_thread())
        assert gate.wait(10)
        return real(*args, **kwargs)
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        monkeypatch.setattr(rsm_values, "rsm_member_groups", recorded)
        page.begin_preview()
        qapp.processEvents()
        assert page.active() and not page.preview_button.isEnabled()
        assert not page._rows["scan"].editor.isEnabled() and page.run_bar.stopButton.isEnabled()
        page.stop()  # finding the scans has no stop point: what it finds is dropped
        gate.set()
        _wait(page, qapp, lambda: not page.active())
        assert page.run_bar.readinessLabel.full_text() == "RSM stopped" and page._owner.prepared is None
        assert page._rows["scan"].editor.isEnabled()
        page.begin_run()
        _wait(page, qapp, lambda: page.shown is not None and not page.active())
        assert len(threads) == 2 and all(thread is not threading.main_thread() for thread in threads)
    finally:
        _close(page, qapp)


def test_processing_takes_ints_threshold_and_the_shared_normalization_group(qapp, tmp_path):
    """G8b (design §2; the maintainer, 2026-10-01: "Other processing options like thresholding, monitor etc should
    also apply"): the Conditioning rows are laid out by Int's processing composer. Int's Threshold is its RangeRow
    driven by Int's transitions, Auto by default as Stitch's, and it is the run's ``RSMFrameCorrections.threshold``;
    "High → NaN" is its Manual High now, so the conditioning's own high threshold is None. The one Normalization
    group Stitch has (decision 4's four choices) maps onto RSM's policy: its columns are the first scan's counters,
    the occurrences Advanced's, the rows a choice does not read greyed, and a missing column is named."""
    from xdart.gui.tools import normalization_values as normalization
    from xdart.gui.widgets.controls_panel import RangeRow
    from xrd_tools.analysis.module_transaction import MetadataColumnSelector
    from xrd_tools.analysis.rsm_operation import RSMNormalizationMode, RSMNormalizationPolicy
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning
    page = _page(qapp)
    try:
        assert isinstance(page.threshold_range, RangeRow)  # Int's composer's row
        assert page._threshold_policy() == StitchConditioning("auto")  # Auto by default, as Stitch's
        default = page._normalization_policy()  # the preset's foil transmission × exposure
        assert (default.mode, default.foil_selector, default.exposure_selector, default.absorption_lengths) == (
            RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE, MetadataColumnSelector("foil status", 0),
            MetadataColumnSelector("Seconds", 0), (1.06, 3.04, 4.65, 9.5))
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)  # Normalization None, the threshold off
        form = page.build_form()
        assert form.normalization == RSMNormalizationPolicy.identity()
        assert form.corrections.threshold is None and form.conditioning.high_threshold is None
        page._threshold_edited(THRESHOLD_MAX, 50000.0)  # a High makes it manual, as Int's
        threshold = page.build_form().corrections.threshold
        assert (threshold.mode, threshold.threshold_max) == ("manual", 50000.0)
        assert page.build_form().conditioning.high_threshold is None
        assert "Seconds" in page._fields["monitor"].choices  # the first scan's counters
        page.set_sources([])  # a Browse pick refreshes them (Stitch's _source_committed)
        assert "Seconds" not in page._fields["monitor"].choices
        page.set_sources(["source/RSM_synth"])  # a relative SPEC file is the Project's, as the run reads it
        assert "Seconds" in page._fields["monitor"].choices
        page.resolve_source()
        page.set_field("normalization", normalization.MONITOR)
        assert "Normalization: a monitor column" in page._headers["processing"].detail
        assert not page._fields["foil_column"].enabled  # the rows the choice does not read, greyed
        page.controls.set_editors_enabled(False)  # a run locks the rows and unlocks them (Codex's F2) ...
        page.controls.set_editors_enabled(True)
        foil, monitor = page._rows["foil_column"].editor, page._rows["monitor"].editor
        assert not foil.isEnabled() and monitor.isEnabled()  # ... the greyed row stays greyed
        page.set_field("monitor", "Seconds")
        page.set_field("adv_monitor_occurrence", "1")
        assert page._readiness[1], page._readiness
        assert page.build_form().normalization == RSMNormalizationPolicy(
            RSMNormalizationMode.MONITOR, None, None, (0.0, 0.0, 0.0, 0.0),
            monitor_selector=MetadataColumnSelector("Seconds", 1))
        page.set_field("normalization", normalization.MONITOR_FOIL)  # a monitor before the foils
        policy = page.build_form().normalization
        assert (policy.mode, policy.foil_selector, policy.exposure_selector, policy.monitor_selector) == (
            RSMNormalizationMode.MONITOR_FOIL_TRANSMISSION, MetadataColumnSelector("foil status", 0), None,
            MetadataColumnSelector("Seconds", 1))
    finally:
        page.close_page()


def test_advanced_corrections_reach_the_runs_frame_corrections(qapp, tmp_path):
    """G8b: Advanced's Corrections group of shared rows: solid angle and polarization (0.99) on by default
    (decision 5), the dark, the flat and its cutoff, each reaching the run's ``RSMFrameCorrections``; a refused
    value, or a named file that is not there, is named in the Processing header."""
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        corrections = page.build_form().corrections
        assert (corrections.solid_angle, corrections.polarization_factor, corrections.flat_min_fraction,
                corrections.dark_path, corrections.flat_path) == (True, 0.99, 0.1, None, None)
        np.save(tmp_path / "dark.npy", np.zeros((4, 5)))
        np.save(tmp_path / "flat.npy", np.ones((4, 5)))
        for key, value in (("dark_path", "dark.npy"), ("flat_path", "flat.npy"), ("adv_flat_cutoff", "20"),
                           ("adv_polarization", ""), ("adv_solid_angle", "Off")):
            page.set_field(key, value)
        assert page._readiness[1], page._readiness
        corrections = page.build_form().corrections
        assert (corrections.dark_path, corrections.flat_path) == (str(tmp_path / "dark.npy"),
                                                                  str(tmp_path / "flat.npy"))
        assert (corrections.flat_min_fraction, corrections.polarization_factor, corrections.solid_angle) == (
            0.2, None, False)
        page.set_field("adv_polarization", "2")
        assert "Advanced Polarization" in page._headers["processing"].detail and not page._readiness[1]
        with pytest.raises(ValueError, match="Polarization"):
            page.build_form()
        page.set_field("adv_polarization", "0.5")
        page.set_field("dark_path", "missing.npy")
        assert "the Dark (missing.npy not found)" in page._headers["processing"].detail
    finally:
        page.close_page()


def test_preview_lists_the_shared_dark_and_flat_checks_warnings(qapp, tmp_path, monkeypatch):
    """G8c: RSM's Preview lists the dark and flat checks' warnings as Stitch's does (the shared check, its
    ``detector_image_warnings``), and its status counts them."""
    _commit_instead_of_gridding(monkeypatch, _values())
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        flat = np.full((195, 487), 1000.)  # the members' detector
        flat[:97] = 0  # dead: masked by the flat's cutoff
        np.save(tmp_path / "flat.npy", flat)
        page.set_field("flat_path", "flat.npy")
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready" in label.full_text())
        assert label.full_text().endswith(" frames · 2 warnings (Preview lists them)")
        assert page.preview_text.toPlainText().startswith(
            "Warnings\nFlat field median 1000 is far from 1, so intensities scale by its inverse\n"
            f"The flat field masks {100 * 97 / 195:.1f} % of the detector's usable pixels (dead or below 10 % of its "
            "median)\n\nProject: ")
    finally:
        _close(page, qapp)


def test_preview_shows_stitchs_auto_ceiling_hint_on_the_threshold_row(qapp, tmp_path, monkeypatch):
    """G8c (G8b choice): once a Preview resolved each member's automatic ceiling (decision 39's calibrated-detector
    counter limit, on the worker), the Threshold's High box shows Stitch's hint (the shared ``auto_ceiling_hint``),
    never its value, and the transitions start a typed High from it; a new selection returns to the default."""
    from PySide6 import QtWidgets
    from xrd_tools.analysis import rsm_operation
    monkeypatch.setattr(rsm_operation, "_rsm_member_saturation",
                        lambda threshold, members, detector: (65535,) * len(members))
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        page._threshold_edited(THRESHOLD_ENABLED, True)

        def hints():
            return {edit.placeholderText() for edit in page.threshold_range.findChildren(QtWidgets.QLineEdit)}
        assert "Auto · the detector's saturation" in hints()
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready" in label.full_text())
        assert "Auto · 65535 (saturation)" in hints() and page._auto_threshold_max == 65534.0
        page.set_sources(page._source_paths)
        assert "Auto · the detector's saturation" in hints() and page._auto_threshold_max is None
    finally:
        _close(page, qapp)


_NO_UB = "This source records no UB matrix. Enter one, or take it from a SPEC scan, or choose Q coordinates."


def test_a_ub_less_nexus_member_is_warned_and_waits_in_hkl_and_is_ready_in_q(qapp, tmp_path):
    """G8b (the maintainer, 2026-10-01: "For files with no ub matrix, a warning with a chance to enter the ub matrix
    manually or through a spec file can be offered"): once the Source is found, an HKL map of a NeXus member that
    records no UB is warned in the Source header, the UB rows show, and Preview and Run wait; Q coordinates need
    none and are ready; a typed UB is then the form's."""
    from tests.core.test_rsm_nexus_members import _write_rsm_nexus
    from xrd_tools.analysis.rsm_operation import RSMUBOverride
    page = _page(qapp)
    try:
        _canonical_partner(tmp_path)
        nexus = _write_rsm_nexus(tmp_path / "member_noub.nxs", ub=False)
        page.set_field("project_root", str(tmp_path))
        page.set_field("geometry_path", "calibration/rsm_canonical.json")
        page.set_field("source_kind", "NeXus")
        page.set_sources([str(nexus)])
        page.set_field("normalization", "None")
        assert page._readiness[1] and page.ub_card._collapsed  # not known until the Source is found
        page.begin_preview()  # finds the member, then waits for a UB: nothing is prepared
        _wait(page, qapp, lambda: not page.active())
        assert page.run_bar.readinessLabel.full_text() == _NO_UB and page._owner.prepared is None
        header = page._headers["source"]
        assert header.detail == _NO_UB and not header.ready and "⚠ no UB matrix" in header.text
        assert not page._readiness[1] and "a UB matrix" in page._readiness[0]
        assert not page.ub_card._collapsed  # the UB rows show
        page.begin_run()  # Run waits too
        assert not page.active() and page._owner.prepared is None
        page.set_field("coordinates", RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.display_name)
        assert page._readiness[1] and page._headers["source"].ready and _NO_UB not in page._headers["source"].detail
        assert page.build_form().ub is None
        page.set_field("coordinates", RSMCoordinateFrame.HKL.display_name)
        assert not page._readiness[1]
        page.set_field("ub_from", "Typed")
        for number, row in enumerate(("1 0 0", "0 1 0", "0 0 1"), 1):
            page.set_field(f"ub_row{number}", row)
        assert page._readiness[1], page._readiness
        assert page.build_form().ub == RSMUBOverride(matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    finally:
        _close(page, qapp)


def test_a_typed_or_spec_scan_ub_reaches_the_form_and_a_refused_one_is_named(qapp, tmp_path):
    """G8b: a source with a UB keeps it (the form's ``ub`` None); the UB matrix group can override it, typed as three
    rows of three numbers in SPEC's #G3 convention or taken from a SPEC scan's #G3 (a relative file is the Project's),
    for an HKL map only. A row that is not three numbers, a singular matrix or a SPEC file not there is named in
    the Source header."""
    from xrd_tools.analysis.rsm_operation import RSMUBOverride
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        assert page._readiness[1] and page.build_form().ub is None  # the source's own (#G3)
        assert _NO_UB not in page._headers["source"].detail
        assert not page._rows["ub_row1"].editor.isEnabled()  # rows the choice does not read are greyed
        page.set_field("ub_from", "Typed")
        for number, row in enumerate(("2, 0, 0", "0 2 0", "0 0 2"), 1):
            page.set_field(f"ub_row{number}", row)
        assert page._readiness[1], page._readiness
        assert page.build_form().ub == RSMUBOverride(matrix=(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0))
        page.set_field("ub_row2", "4 0 0")
        assert "UB matrix: the matrix is singular" in page._headers["source"].detail and not page._readiness[1]
        with pytest.raises(ValueError, match="UB matrix: the matrix is singular"):
            page.build_form()
        page.set_field("ub_row2", "0 2")
        assert "UB matrix: row 2 needs three numbers" in page._headers["source"].detail
        page.set_field("ub_from", "A SPEC scan's")
        page.set_field("ub_spec_path", "source/RSM_synth")
        page.set_field("ub_scan", "2")
        assert page._readiness[1], page._readiness
        assert page.build_form().ub == RSMUBOverride(spec_path=str(tmp_path / "source" / "RSM_synth"), scan="2")
        page.set_field("ub_spec_path", "source/absent")
        assert "the UB's SPEC file (absent not found)" in page._headers["source"].detail
        page.set_field("coordinates", RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU.display_name)  # HKL only
        assert page.build_form().ub is None and page._readiness[1]
        assert not page._rows["ub_spec_path"].editor.isEnabled()
        assert page._fields["ub_spec_path"].reason == "Q coordinates use no UB"
    finally:
        _close(page, qapp)


def test_the_preview_names_the_solid_angle_the_detector_mask_and_a_given_ub_beside_the_sources(qapp, tmp_path):
    """G8b: the Preview names where the solid angle comes from (the canonical asset: the untilted formula about the
    beam pixel; off when it is off) and the detector mask, keeps each member's #G3 line (S3c's cross-check), and
    shows a UB given for an HKL map beside each member's own (the design: "An override is shown in the Preview
    beside the source's own UB"), the given one the matrix used."""
    page = _page(qapp)
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready" in page.run_bar.readinessLabel.full_text())
        shown = page.preview_text.toPlainText()
        assert "Solid angle: the untilted formula about the beam pixel" in shown
        assert "Detector mask: none" in shown and "UB given" not in shown and "UB the source records" not in shown
        assert shown.count("  UB: #G3 (the header holds no orientation reflections to recompute it from)") == 2
        page.set_field("ub_from", "Typed")
        for number, row in enumerate(("2 0 0", "0 2 0", "0 0 2"), 1):
            page.set_field(f"ub_row{number}", row)
        page.set_field("adv_solid_angle", "Off")
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready" in page.run_bar.readinessLabel.full_text())
        shown = page.preview_text.toPlainText()
        assert ("UB given: typed, in SPEC's #G3 convention: 2 0 0; 0 2 0; 0 0 2, in place of the sources' own "
                "(each member's below)") in shown
        assert shown.count("  UB the source records: 1 0 0; 0 1 0; 0 0 1") == 2  # each member's own #G3
        assert shown.count("  xrayutilities matrix: 2 0 0; 0 2 0; 0 0 2") == 2  # the given one is used
        assert "Solid angle: off" in shown
    finally:
        _close(page, qapp)


# AUD-9 stage 2b G8d: Set BG on the keyed 2×3 layout (the maintainer: "Other quick toggles like switching to log
# scale set bg, etc should also be there")


def _grid_values(scale, *, shape=(5, 7, 9), axis_offset=0.0):
    """Viewer values on one grid (``axis_offset`` moves it), their intensity scaled by ``scale``, one voxel NaN."""
    from xrd_tools.io.analysis_artifact import AnalysisArtifactKind, project_analysis_artifact_result
    from xrd_tools.session.rsm_viewer_model import make_rsm_viewer_values
    axes = tuple((name, np.linspace(axis_offset + index, axis_offset + index + 1.0, size))
                 for index, (name, size) in enumerate(zip(("h", "k", "l"), shape, strict=True)))
    intensity = (np.arange(np.prod(shape), dtype=np.float64).reshape(shape) ** 1.5) * scale + scale
    intensity[1, 2, 3] = np.nan
    return make_rsm_viewer_values(project_analysis_artifact_result(
        kind=AnalysisArtifactKind.RSM, axes=axes, axis_units=(("h", None), ("k", None), ("l", None)),
        intensity=intensity, sigma=None, coverage=None, normalization=None))


def _panels_at(values, indices):
    """``values``' six panels at ``indices``, as an independent RSMViewerModel draws them."""
    from xrd_tools.session.rsm_viewer_model import RSMViewerModel
    model = RSMViewerModel()
    model.snapshot(values)
    return rsm_keyed_projection(values, model.snapshot(values, h_index=indices[0], k_index=indices[1],
                                                       l_index=indices[2])).panels


def _drawn_arrays(view):
    return [panel.image if hasattr(panel, "image") else panel.intensity for panel in view.keyed_projection.panels]


def test_set_bg_draws_every_panel_minus_the_background_at_the_same_indices(qapp, monkeypatch):
    """Set BG takes the result shown as the background; every result then drawn is minus it panel by panel,
    each slice minus the background's slice at the same index (also after a step), each projection minus its
    projection; a result against itself draws zeros, and says so. Display only."""
    values = {"A": _grid_values(1.0), "B": _grid_values(3.0)}
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", values.__getitem__)
    page = _page(qapp, owner=_Owner())
    try:
        assert not page.view.background.isHidden() and page.view.background.text() == "Set BG"
        page._add_result("/tmp/a_rsm.nexus", "A")
        page.view.background.click()
        drawn = page.view.keyed_projection
        assert drawn.background_set and page.view.background.text() == "Clear BG"
        assert "every panel is zero" in drawn.status
        for array in _drawn_arrays(page.view):
            finite = np.isfinite(array)
            assert finite.any() and np.all(array[finite] == 0)
        page._add_result("/tmp/b_rsm.nexus", "B")
        indices = page.shown[1].state.axis_indices
        expected = [np.subtract(*(getattr(p, "image", None) if hasattr(p, "image") else p.intensity
                                  for p in pair))
                    for pair in zip(_panels_at(values["B"], indices), _panels_at(values["A"], indices))]
        for got, want in zip(_drawn_arrays(page.view), expected, strict=True):
            np.testing.assert_array_equal(got, want)
        assert "minus the background a_rsm.nexus" in page.view.keyed_projection.status
        instance = _keys(page.view)[0].instance  # the HK slice, stepped along L
        page._step_slice(instance, 2)
        stepped = page.shown[1].state.axis_indices
        assert stepped[2] == 2
        want = np.subtract(_panels_at(values["B"], stepped)[0].image, _panels_at(values["A"], stepped)[0].image)
        np.testing.assert_array_equal(page.view.keyed_projection.panels[0].image, want)
    finally:
        _close(page, qapp)


def test_a_background_on_another_grid_is_drawn_unsubtracted_and_says_why(qapp, monkeypatch):
    """A result on another grid (axes or bins) is not subtracted: its panels are drawn as they are, the status
    and the button's tip say why, and Clear BG stays usable."""
    values = {"A": _grid_values(1.0), "C": _grid_values(2.0, axis_offset=0.5)}
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", values.__getitem__)
    page = _page(qapp, owner=_Owner())
    try:
        page._add_result("/tmp/a_rsm.nexus", "A")
        page.view.background.click()
        page._add_result("/tmp/c_rsm.nexus", "C")
        drawn = page.view.keyed_projection
        assert drawn.background_set and "another grid" in drawn.status
        assert "another grid" in page.view.background.toolTip()
        assert page.view.background.isEnabled() and page.view.background.text() == "Clear BG"
        own = _panels_at(values["C"], page.shown[1].state.axis_indices)
        for got, panel in zip(_drawn_arrays(page.view), own, strict=True):
            np.testing.assert_array_equal(got, panel.image if hasattr(panel, "image") else panel.intensity)
    finally:
        _close(page, qapp)


def test_the_background_is_released_by_clear_bg_by_its_result_admitted_again_and_by_close(qapp, monkeypatch):
    """Each release path lets go of the background through Int's handshake (the view's receipt, then the
    owner): Clear BG; the background's result admitted again under its name (Stitch's 30f74f29: its identity is
    the data it was set from); and a CLEAN close, before the panels go."""
    values = {"A": _grid_values(1.0), "A2": _grid_values(5.0), "B": _grid_values(3.0)}
    monkeypatch.setattr(page_module, "make_rsm_viewer_values", values.__getitem__)
    page = _page(qapp, owner=_Owner())
    try:
        page._add_result("/tmp/a_rsm.nexus", "A")
        page.view.background.click()
        assert page._background.phase == "ACTIVE" and page.view._rendered_background_key is not None
        page.view.background.click()  # Clear BG
        assert page._background.phase == "RELEASED" and page.view._rendered_background_key is None
        assert not page.view.keyed_projection.background_set and page.view.background.text() == "Set BG"
        page.view.background.click()
        page._add_result("/tmp/b_rsm.nexus", "B")
        assert page.view.keyed_projection.background_set
        page._add_result("/tmp/a_rsm.nexus", "A2")  # the background's result admitted again
        assert page._background.phase == "RELEASED" and not page.view.keyed_projection.background_set
        page.view.background.click()
        assert page._background.phase == "ACTIVE"
    finally:
        _close(page, qapp)
    assert page._background.phase == "RELEASED" and page._background.keyed_source() is None


def test_a_queued_preview_counts_the_dark_and_flat_warnings_as_the_single_one_does(qapp, tmp_path, monkeypatch):
    """G8c's residual: a queued Preview's status counts the dark and flat checks' warnings, each distinct one
    once, as the single Preview's does (each RSM of the queue shares the page's flat)."""
    _counting_grids(monkeypatch)
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path, scans="1,2")
        flat = np.full((195, 487), 1000.)
        flat[:97] = 0
        np.save(tmp_path / "flat.npy", flat)
        page.set_field("flat_path", "flat.npy")
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready · 2 RSMs" in label.full_text())
        assert label.full_text().endswith("Preview ready · 2 RSMs · 2 warnings (Preview lists them)")
    finally:
        _close(page, qapp)


def test_another_calibration_returns_the_auto_hint_to_its_default(qapp, tmp_path, monkeypatch):
    """G8c's residual: the Auto hint is the ceilings of the calibration a Preview resolved; another Calibration
    row returns it to the default until the next Preview, as a new selection does."""
    from PySide6 import QtWidgets
    from xrd_tools.analysis import rsm_operation
    monkeypatch.setattr(rsm_operation, "_rsm_member_saturation",
                        lambda threshold, members, detector: (65535,) * len(members))
    page = _page(qapp)
    label = page.run_bar.readinessLabel
    try:
        _synthetic_source(tmp_path)
        _with_source(page, tmp_path)
        page._threshold_edited(THRESHOLD_ENABLED, True)

        def hints():
            return {edit.placeholderText() for edit in page.threshold_range.findChildren(QtWidgets.QLineEdit)}
        page.begin_preview()
        _wait(page, qapp, lambda: "Preview ready" in label.full_text())
        assert "Auto · 65535 (saturation)" in hints()
        page.set_field("geometry_path", "calibration/another.json")
        assert "Auto · the detector's saturation" in hints() and page._auto_threshold_max is None
    finally:
        _close(page, qapp)
